// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/config.h>

#include <string>
#include <vector>

#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/tensor.hpp>
#include "../cpu_conv.hpp"
#include "../get_handle.hpp"
#include "../workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#endif

namespace {

// Parameter names remain available in builds that disable the plugin feature.
enum class TileDirection
{
    Forward,
    BackwardData,
    BackwardWeight
};

enum class TileShape
{
    Padded,
    DilatedStrideTwo,
    AsymmetricSame,
    PaddedC32K64,
    PaddedC64K32
};

bool IsAligned(TileShape shape)
{
    return shape == TileShape::PaddedC32K64 || shape == TileShape::PaddedC64K32;
}

struct TileCase
{
    int spatial_dims;
    TileDirection direction;
    bool channels_last;
    TileShape shape;
    int split_k = 1;
};

std::vector<TileCase> NumericalCases()
{
    std::vector<TileCase> cases;
    for(const int rank : {2, 3})
        for(const auto direction :
            {TileDirection::Forward, TileDirection::BackwardData, TileDirection::BackwardWeight})
            for(const bool channels_last : {true, false})
                for(const auto shape :
                    {TileShape::Padded, TileShape::DilatedStrideTwo, TileShape::AsymmetricSame})
                    cases.push_back({rank, direction, channels_last, shape});
    for(const int rank : {2, 3})
        cases.push_back({rank, TileDirection::BackwardWeight, true, TileShape::Padded, 2});
    for(const int rank : {2, 3})
        for(const auto direction :
            {TileDirection::Forward, TileDirection::BackwardData, TileDirection::BackwardWeight})
            for(const bool channels_last : {true, false})
                for(const auto shape : {TileShape::PaddedC32K64, TileShape::PaddedC64K32})
                    cases.push_back({rank, direction, channels_last, shape});
    for(const int rank : {2, 3})
        for(const bool channels_last : {true, false})
            cases.push_back(
                {rank, TileDirection::BackwardWeight, channels_last, TileShape::PaddedC32K64, 2});
    // Only the 2D BWD consumer exposes split-K IDs. Even output vectors allow
    // native FP16/BF16 atomic accumulation; odd-channel scalar cases cannot.
    for(const bool channels_last : {true, false})
        cases.push_back(
            {2, TileDirection::BackwardData, channels_last, TileShape::PaddedC32K64, 2});
    return cases;
}

std::string CaseName(const TileCase& p)
{
    const auto direction = p.direction == TileDirection::Forward        ? "Fwd"
                           : p.direction == TileDirection::BackwardData ? "BwdData"
                                                                        : "Wrw";
    const auto layout    = p.spatial_dims == 2 ? (p.channels_last ? "NHWC" : "NCHW")
                                               : (p.channels_last ? "NDHWC" : "NCDHW");
    const auto shape     = p.shape == TileShape::Padded             ? "Padded"
                           : p.shape == TileShape::DilatedStrideTwo ? "DilatedStrideTwo"
                           : p.shape == TileShape::AsymmetricSame   ? "AsymmetricSame"
                           : p.shape == TileShape::PaddedC32K64     ? "PaddedC32K64"
                                                                    : "PaddedC64K32";
    return "D" + std::to_string(p.spatial_dims) + direction + layout + shape +
           (p.split_k == 1 ? "" : "Split" + std::to_string(p.split_k));
}

#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE

using CKSolverType = miopen::solver::CKSolverType;
using Direction    = miopen::conv::Direction;
using Loader       = miopen::solver::CkImplLibLoader;

Direction ConvDirection(TileDirection direction)
{
    if(direction == TileDirection::Forward)
        return Direction::Forward;
    if(direction == TileDirection::BackwardData)
        return Direction::BackwardData;
    return Direction::BackwardWeights;
}

CKSolverType SolverType(const TileCase& p)
{
    if(p.spatial_dims == 2)
    {
        if(p.direction == TileDirection::Forward)
            return CKSolverType::GrpConvFwd;
        return p.direction == TileDirection::BackwardData ? CKSolverType::GrpConvBwd
                                                          : CKSolverType::GrpConvWrw;
    }
    if(p.direction == TileDirection::Forward)
        return CKSolverType::GrpConv3dFwd;
    return p.direction == TileDirection::BackwardData ? CKSolverType::GrpConv3dBwd
                                                      : CKSolverType::GrpConv3dWrw;
}

bool HasSplitSuffix(const TileCase& p)
{
    return p.direction == TileDirection::BackwardWeight ||
           (p.spatial_dims == 2 && p.direction == TileDirection::BackwardData);
}

std::string SelectedId(const TileCase& p, const std::string& catalog_id)
{
    return catalog_id + (HasSplitSuffix(p) ? "+" + std::to_string(p.split_k) : "");
}

struct TileProblem
{
    miopen::TensorDescriptor x;
    miopen::TensorDescriptor w;
    miopen::TensorDescriptor y;
    miopen::ConvolutionDescriptor conv;

    miopen::conv::ProblemDescription
    Describe(TileDirection direction, float alpha = 1, float beta = 0) const
    {
        return {direction == TileDirection::Forward ? x : y,
                w,
                direction == TileDirection::Forward ? y : x,
                conv,
                ConvDirection(direction),
                0,
                miopen::Scalar(alpha),
                miopen::Scalar(beta)};
    }
};

TileProblem MakeProblem(const TileCase& p, miopenDataType_t datatype)
{
    const auto layout = p.spatial_dims == 2
                            ? (p.channels_last ? miopenTensorNHWC : miopenTensorNCHW)
                            : (p.channels_last ? miopenTensorNDHWC : miopenTensorNCDHW);
    // Unequal C/G and K/G reveal cross-group/channel indexing mistakes. Keep
    // odd tails, and add aligned channels so vector and larger recipes compete.
    const std::size_t c = p.shape == TileShape::PaddedC32K64   ? 32
                          : p.shape == TileShape::PaddedC64K32 ? 64
                                                               : 3;
    const std::size_t k = p.shape == TileShape::PaddedC32K64   ? 64
                          : p.shape == TileShape::PaddedC64K32 ? 32
                                                               : 5;
    std::vector<std::size_t> x_lengths{2, 2 * c};
    std::vector<std::size_t> w_lengths{2 * k, c};
    std::vector<int> pads(p.spatial_dims, 1);
    std::vector<int> strides(p.spatial_dims, 1);
    std::vector<int> dilations(p.spatial_dims, 1);
    for(int d = 0; d < p.spatial_dims; ++d)
    {
        // Small aligned inputs retain spatial/GEMM tails. Split-two reductions
        // exceed the largest K/block=128 so both atomic contributors have work.
        const int spatial = IsAligned(p.shape) ? (p.split_k > 1 ? (p.spatial_dims == 2 ? 8 : 4) : 3)
                                               : (p.split_k > 1 ? 7 : 5);
        x_lengths.push_back(static_cast<std::size_t>(spatial + d));
        w_lengths.push_back(3);
    }
    if(p.shape == TileShape::DilatedStrideTwo)
    {
        // Even input sizes leave adjusted right padding = 1, versus left padding = 2.
        // Since dilation and stride are two, every odd dX coordinate is an unwritten
        // backward hole unless the selected invocation clears its entire output.
        std::fill(pads.begin(), pads.end(), 2);
        std::fill(strides.begin(), strides.end(), 2);
        std::fill(dilations.begin(), dilations.end(), 2);
        for(int d = 0; d < p.spatial_dims; ++d)
            x_lengths[d + 2] = static_cast<std::size_t>(6 + 2 * d);
    }
    else if(p.shape == TileShape::AsymmetricSame)
    {
        // MIOpen exposes SAME output sizing rather than separate left/right pad
        // vectors. Even filters with explicit left pad zero produce right pad one.
        std::fill(pads.begin(), pads.end(), 0);
        std::fill(w_lengths.begin() + 2, w_lengths.end(), 2);
    }
    miopen::TensorDescriptor x(datatype, layout, x_lengths);
    miopen::TensorDescriptor w(datatype, layout, w_lengths);
    miopen::ConvolutionDescriptor conv(p.spatial_dims,
                                       miopenConvolution,
                                       p.shape == TileShape::AsymmetricSame ? miopenPaddingSame
                                                                            : miopenPaddingDefault,
                                       pads,
                                       strides,
                                       dilations,
                                       std::vector<int>(p.spatial_dims, 0),
                                       2);
    auto y = conv.GetForwardOutputTensor(x, w, datatype);
    return {std::move(x), std::move(w), std::move(y), std::move(conv)};
}

template <class T>
void FillSignedPattern(std::vector<T>& values, int seed, int sign)
{
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        const int value = static_cast<int>((i * 7 + i / 11 + seed) % 9) - 4;
        values[i]       = T{static_cast<float>(sign * value) * 0.25f};
    }
}

template <class T>
void CheckNumerics(const std::vector<T>& actual,
                   const tensor<T>& expected,
                   miopenDataType_t datatype)
{
    ASSERT_EQ(actual.size(), expected.data.size());
    // The inputs are exactly representable dyadic values. Allow the output rounding
    // difference between host BF16 configuration and native round-to-nearest WMMA.
    const float tolerance = datatype == miopenBFloat16 ? 0.016f
                            : datatype == miopenHalf   ? 0.0015f
                                                       : 0.00001f;
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        const float want = static_cast<float>(expected.data[i]);
        const float got  = static_cast<float>(actual[i]);
        if(!std::isfinite(got) || std::abs(got - want) > tolerance * (1 + std::abs(want)))
        {
            ADD_FAILURE() << "element=" << i << " actual=" << got << " expected=" << want;
            return;
        }
    }
}

template <class T>
void CheckBackwardHoles(const std::vector<T>& actual, const miopen::TensorDescriptor& dx)
{
    const auto& lengths = dx.GetLengths();
    const auto& strides = dx.GetStrides();
    // Recover logical spatial coordinates from a physical offset for either packed layout.
    for(std::size_t i = 0; i < actual.size(); ++i)
        for(std::size_t d = 2; d < lengths.size(); ++d)
            if(((i / strides[d]) % lengths[d]) % 2 != 0)
            {
                ASSERT_EQ(static_cast<float>(actual[i]), 0.0f)
                    << "backward hole at physical offset=" << i << " dimension=" << d;
                break;
            }
}

template <class T>
struct NumericalReference
{
    tensor<T> x;
    tensor<T> w;
    tensor<T> y;

    explicit NumericalReference(const TileProblem& problem)
        : x(problem.x), w(problem.w), y(problem.y)
    {
    }
};

template <class T>
void RunNumericalCase(const TileCase& p, miopenDataType_t datatype)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "CK Tile grouped convolution requires gfx1250";
    const auto& loader = Loader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded()) << "Enabled Tile validation requires the gfx1250 plugin";

    const auto tensors = MakeProblem(p, datatype);
    auto problem       = tensors.Describe(p.direction);
    // Exercise the consumer on its own nonblocking stream rather than relying on
    // legacy default-stream ordering for output clears or workspace conversions.
    const auto stream = get_handle().CreateExclusiveStream();
    miopen::Handle handle(stream.get());
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);
    const auto solver  = SolverType(p);
    const auto catalog = loader.FillValidKernels(solver, problem, datatype, false);
    std::vector<std::string> tile_ids;
    for(const auto& id : catalog)
    {
        if(id.find("CKTileGroupedConv") != 0)
            continue;
        SCOPED_TRACE(id);
        const auto selected  = SelectedId(p, id);
        const bool supported = loader.IsArgsSupported(solver, problem, selected, datatype, false);
        // A split-one catalog also contains scalar BWD recipes, whose low-precision
        // output cannot use split-two atomics. Exercise every eligible split-two ID.
        if(p.split_k == 1)
            ASSERT_TRUE(supported);
        if(supported)
            tile_ids.push_back(id);
    }
    ASSERT_FALSE(tile_ids.empty()) << "No eligible Tile candidate for " << CaseName(p);

    // Compute each signed-content reference once, then use it for every candidate.
    // All three passes remain: original pointers, changed contents at the same
    // pointers, and fresh pointers while the old allocations are still alive.
    std::vector<NumericalReference<T>> references;
    references.reserve(3);
    for(const int pass : {0, 1, 2})
    {
        references.emplace_back(tensors);
        auto& ref = references.back();
        FillSignedPattern(ref.x.data, 1 + pass, pass == 1 ? -1 : 1);
        FillSignedPattern(ref.w.data, 3 + 2 * pass, 1);
        FillSignedPattern(ref.y.data, 5 + pass, pass == 1 ? -1 : 1);
        const auto& pads      = tensors.conv.GetConvPads();
        const auto& strides   = tensors.conv.GetConvStrides();
        const auto& dilations = tensors.conv.GetConvDilations();
        if(p.direction == TileDirection::Forward)
            cpu_convolution_forward(
                p.spatial_dims, ref.x, ref.w, ref.y, pads, strides, dilations, 2);
        else if(p.direction == TileDirection::BackwardData)
            cpu_convolution_backward_data(
                p.spatial_dims, ref.x, ref.w, ref.y, pads, strides, dilations, 2);
        else
            cpu_convolution_backward_weight(
                p.spatial_dims, ref.x, ref.w, ref.y, pads, strides, dilations, 2);
    }
    auto x_dev          = handle.Write(references.front().x.data);
    auto w_dev          = handle.Write(references.front().w.data);
    auto y_dev          = handle.Write(references.front().y.data);
    auto other_x_dev    = handle.Write(references.front().x.data);
    auto other_w_dev    = handle.Write(references.front().w.data);
    auto other_y_dev    = handle.Write(references.front().y.data);
    const auto metadata = datatype == miopenFloat ? loader.GetAllKernelTypeStrings(solver)
                                                  : std::vector<std::string>{};

    for(const auto& tile_id : tile_ids)
    {
        const auto selected = SelectedId(p, tile_id);
        SCOPED_TRACE(selected);
        if(datatype == miopenFloat)
        {
            // Metadata publishes the FP32 family; low-precision IDs come from
            // the actual per-problem catalog.
            ASSERT_NE(std::find(metadata.begin(), metadata.end(), tile_id), metadata.end());
        }
        std::size_t workspace_bytes = 0;
        // Keep only the invoker, not the solution: plugin-owned solution
        // destruction must not leave dangling references in the consumer callable.
        const auto invoker = [&] {
            const auto solution = loader.GetSolution(solver, ctx, problem, selected, false);
            EXPECT_EQ(solution.status, miopenStatusSuccess);
            workspace_bytes = solution.workspace_sz;
            if(!solution.invoker_factory)
            {
                ADD_FAILURE() << "Tile solution has no invoker factory";
                return miopen::Invoker{};
            }
            return handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        }();
        ASSERT_TRUE(invoker);
        const auto engine_bytes = datatype == miopenFloat ? 0
                                  : p.direction == TileDirection::BackwardWeight
                                      ? tensors.w.GetElementSize() * sizeof(float)
                                  : p.direction == TileDirection::BackwardData && p.split_k > 1
                                      ? tensors.x.GetElementSize() * sizeof(float)
                                      : 0;
        ASSERT_EQ(workspace_bytes,
                  miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem, engine_bytes));
        // Re-query this exact configuration, not the solver-wide maximum over
        // legacy and Tile instances, before allocating the advertised minimum.
        {
            const auto queried = loader.GetSolution(solver, ctx, problem, selected, false);
            ASSERT_EQ(queried.status, miopenStatusSuccess);
            ASSERT_EQ(queried.workspace_sz, workspace_bytes);
        }
        Workspace scratch(workspace_bytes);
        Workspace other_scratch(workspace_bytes);
        if(workspace_bytes != 0)
        {
            ASSERT_NE(scratch.ptr(), nullptr);
            ASSERT_NE(other_scratch.ptr(), nullptr);
        }

        for(const int pass : {0, 1, 2})
        {
            SCOPED_TRACE(pass);
            const auto& ref = references[pass];
            auto& xd        = pass == 2 ? other_x_dev : x_dev;
            auto& wd        = pass == 2 ? other_w_dev : w_dev;
            auto& yd        = pass == 2 ? other_y_dev : y_dev;
            auto& ws        = pass == 2 ? other_scratch : scratch;
            handle.WriteTo(ref.x.data.data(), xd, tensors.x.GetNumBytes());
            handle.WriteTo(ref.w.data.data(), wd, tensors.w.GetNumBytes());
            handle.WriteTo(ref.y.data.data(), yd, tensors.y.GetNumBytes());

            auto* output            = p.direction == TileDirection::Forward        ? yd.get()
                                      : p.direction == TileDirection::BackwardData ? xd.get()
                                                                                   : wd.get();
            const auto& output_desc = p.direction == TileDirection::Forward        ? tensors.y
                                      : p.direction == TileDirection::BackwardData ? tensors.x
                                                                                   : tensors.w;
            ASSERT_EQ(hipMemsetAsync(output, 0x7f, output_desc.GetNumBytes(), handle.GetStream()),
                      hipSuccess);
            if(workspace_bytes != 0)
                ASSERT_EQ(hipMemsetAsync(
                              ws.ptr(), pass == 1 ? 0x5a : 0xa5, ws.size(), handle.GetStream()),
                          hipSuccess);

            const auto invoke = [&](void* workspace, std::size_t bytes) {
                if(p.direction == TileDirection::Forward)
                {
                    const miopen::ConvFwdTensors args{
                        tensors.x, xd.get(), tensors.w, wd.get(), tensors.y, yd.get()};
                    invoker(handle, miopen::conv::DataInvokeParams{args, workspace, bytes, false});
                }
                else if(p.direction == TileDirection::BackwardData)
                {
                    const miopen::ConvBwdTensors args{
                        tensors.y, yd.get(), tensors.w, wd.get(), tensors.x, xd.get()};
                    invoker(handle, miopen::conv::DataInvokeParams{args, workspace, bytes, false});
                }
                else
                {
                    const miopen::ConvWrwTensors args{
                        tensors.y, yd.get(), tensors.x, xd.get(), tensors.w, wd.get()};
                    invoker(handle, miopen::conv::WrWInvokeParams{args, workspace, bytes, false});
                }
            };
            if(pass == 0 && workspace_bytes != 0)
            {
                EXPECT_THROW(invoke(nullptr, workspace_bytes), miopen::Exception);
                EXPECT_THROW(invoke(ws.ptr(), workspace_bytes - 1), miopen::Exception);
                EXPECT_THROW(invoke(ws.ptr(), 0), miopen::Exception);
            }
            invoke(ws.ptr(), ws.size());
            handle.Finish();

            if(p.direction == TileDirection::Forward)
            {
                CheckNumerics(handle.Read<T>(yd, ref.y.data.size()), ref.y, datatype);
                EXPECT_EQ(handle.Read<T>(xd, ref.x.data.size()), ref.x.data);
                EXPECT_EQ(handle.Read<T>(wd, ref.w.data.size()), ref.w.data);
            }
            else if(p.direction == TileDirection::BackwardData)
            {
                const auto actual = handle.Read<T>(xd, ref.x.data.size());
                CheckNumerics(actual, ref.x, datatype);
                if(p.shape == TileShape::DilatedStrideTwo)
                    CheckBackwardHoles(actual, tensors.x);
                EXPECT_EQ(handle.Read<T>(yd, ref.y.data.size()), ref.y.data);
                EXPECT_EQ(handle.Read<T>(wd, ref.w.data.size()), ref.w.data);
            }
            else
            {
                CheckNumerics(handle.Read<T>(wd, ref.w.data.size()), ref.w, datatype);
                EXPECT_EQ(handle.Read<T>(xd, ref.x.data.size()), ref.x.data);
                EXPECT_EQ(handle.Read<T>(yd, ref.y.data.size()), ref.y.data);
            }
        }
    }
}

struct SupportCase
{
    int spatial_dims;
    TileDirection direction;
    miopenDataType_t datatype;
};

std::vector<SupportCase> SupportCases()
{
    std::vector<SupportCase> cases;
    for(const auto datatype : {miopenHalf, miopenBFloat16, miopenFloat})
        for(const int rank : {2, 3})
            for(const auto direction : {TileDirection::Forward,
                                        TileDirection::BackwardData,
                                        TileDirection::BackwardWeight})
                cases.push_back({rank, direction, datatype});
    return cases;
}

class GPU_CkTileGroupedConvSupport_NONE : public testing::TestWithParam<SupportCase>
{
};

void RunSupportCase(const Loader& loader, const SupportCase& parameter, const std::string& tile_id)
{
    const TileCase p{parameter.spatial_dims, parameter.direction, true, TileShape::PaddedC32K64};
    const auto solver   = SolverType(p);
    const auto valid    = MakeProblem(p, parameter.datatype);
    const auto selected = SelectedId(p, tile_id);
    SCOPED_TRACE(selected);
    ASSERT_TRUE(loader.IsArgsSupported(
        solver, valid.Describe(p.direction), selected, parameter.datatype, false));

    const auto reject = [&](const auto& problem) {
        EXPECT_FALSE(loader.IsArgsSupported(solver, problem, selected, parameter.datatype, false));
    };
    {
        auto mixed_layout = valid;
        mixed_layout.w =
            miopen::TensorDescriptor(parameter.datatype,
                                     p.spatial_dims == 2 ? miopenTensorNCHW : miopenTensorNCDHW,
                                     valid.w.GetLengths());
        reject(mixed_layout.Describe(p.direction));
    }
    for(const int tensor_index : {0, 1, 2})
    {
        SCOPED_TRACE(tensor_index);
        auto nonpacked = valid;
        auto& desc     = tensor_index == 0   ? nonpacked.x
                         : tensor_index == 1 ? nonpacked.w
                                             : nonpacked.y;
        auto strides   = desc.GetStrides();
        strides[0] += 1;
        desc = miopen::TensorDescriptor(parameter.datatype, desc.GetLengths(), strides);
        reject(nonpacked.Describe(p.direction));
    }
    reject(valid.Describe(p.direction, 2, 0));
    reject(valid.Describe(p.direction, 1, 1));

    // The tensor descriptor accepts 64-bit lengths and strides, while the kernel
    // descriptors are int32. Probe support without allocating enormous buffers.
    auto oversized = valid;
    auto lengths   = oversized.x.GetLengths();
    lengths[0]     = static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) + 1;
    oversized.x    = miopen::TensorDescriptor(
        parameter.datatype, p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC, lengths);
    auto output_lengths = oversized.y.GetLengths();
    output_lengths[0]   = lengths[0];
    oversized.y =
        miopen::TensorDescriptor(parameter.datatype,
                                 p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC,
                                 output_lengths);
    reject(oversized.Describe(p.direction));
    if(p.direction == TileDirection::BackwardWeight)
    {
        // Lengths, strides and element counts fit int32. A 3 GiB FP32
        // weight/scratch extent already violates signed byte addressing; keep
        // the 4 GiB probe as well.
        const auto layout = p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC;
        for(const std::size_t weight_k : {24576, 32768})
        {
            std::vector<std::size_t> x_lengths{1, 32768};
            std::vector<std::size_t> w_lengths{weight_k, 32768};
            x_lengths.insert(x_lengths.end(), p.spatial_dims, 1);
            w_lengths.insert(w_lengths.end(), p.spatial_dims, 1);
            const miopen::TensorDescriptor x(parameter.datatype, layout, x_lengths);
            const miopen::TensorDescriptor w(parameter.datatype, layout, w_lengths);
            const miopen::ConvolutionDescriptor conv(std::vector<int>(p.spatial_dims, 0),
                                                     std::vector<int>(p.spatial_dims, 1),
                                                     std::vector<int>(p.spatial_dims, 1),
                                                     std::vector<int>(p.spatial_dims, 0),
                                                     1);
            const auto y = conv.GetForwardOutputTensor(x, w, parameter.datatype);
            const TileProblem byte_overflow{x, w, y, conv};
            reject(byte_overflow.Describe(p.direction));
        }
    }
    {
        // Exactly 2 GiB of input/output storage: still legal element counts,
        // but one byte past the signed native byte-offset range. Aligned
        // channels keep vector divisibility from masking the byte-bound check.
        const auto layout = p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC;
        std::vector<std::size_t> x_lengths{1, 64};
        std::vector<std::size_t> w_lengths{64, 64};
        x_lengths.insert(x_lengths.end(), p.spatial_dims, 1);
        w_lengths.insert(w_lengths.end(), p.spatial_dims, 1);
        x_lengths.back() =
            parameter.datatype == miopenFloat ? (std::size_t{1} << 23) : (std::size_t{1} << 24);
        const miopen::TensorDescriptor x(parameter.datatype, layout, x_lengths);
        const miopen::TensorDescriptor w(parameter.datatype, layout, w_lengths);
        const miopen::ConvolutionDescriptor conv(std::vector<int>(p.spatial_dims, 0),
                                                 std::vector<int>(p.spatial_dims, 1),
                                                 std::vector<int>(p.spatial_dims, 1),
                                                 std::vector<int>(p.spatial_dims, 0),
                                                 1);
        const auto y = conv.GetForwardOutputTensor(x, w, parameter.datatype);
        const TileProblem byte_overflow{x, w, y, conv};
        reject(byte_overflow.Describe(p.direction));
    }

    auto bad_stride = valid;
    lengths         = bad_stride.x.GetLengths();
    lengths[0]      = 1;
    auto strides    = bad_stride.x.GetStrides();
    strides[0]      = static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) + 1;
    bad_stride.x =
        miopen::TensorDescriptor(parameter.datatype,
                                 p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC,
                                 lengths,
                                 strides);
    output_lengths    = bad_stride.y.GetLengths();
    output_lengths[0] = 1;
    bad_stride.y =
        miopen::TensorDescriptor(parameter.datatype,
                                 p.spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC,
                                 output_lengths);
    reject(bad_stride.Describe(p.direction));

    auto bad_rank            = valid;
    bad_rank.conv.spatialDim = p.spatial_dims == 2 ? 3 : 2;
    reject(bad_rank.Describe(p.direction));

    if(HasSplitSuffix(p))
    {
        EXPECT_FALSE(loader.IsArgsSupported(
            solver, valid.Describe(p.direction), tile_id + "+0", parameter.datatype, false));
        EXPECT_FALSE(loader.IsArgsSupported(
            solver, valid.Describe(p.direction), tile_id + "+-1", parameter.datatype, false));
        if(p.direction == TileDirection::BackwardData && parameter.datatype != miopenFloat)
        {
            const TileCase odd{p.spatial_dims, p.direction, true, TileShape::Padded};
            const auto odd_problem = MakeProblem(odd, parameter.datatype).Describe(p.direction);
            EXPECT_FALSE(loader.IsArgsSupported(
                solver, odd_problem, tile_id + "+2", parameter.datatype, false));
        }
    }
}

TEST_P(GPU_CkTileGroupedConvSupport_NONE, RejectsUnsupportedContracts)
{
    const auto device_name = miopen::solver::GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "CK Tile grouped convolution requires gfx1250";
    const auto& loader = Loader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded());
    const auto& parameter = GetParam();
    const TileCase p{parameter.spatial_dims, parameter.direction, true, TileShape::PaddedC32K64};
    const auto catalog =
        loader.FillValidKernels(SolverType(p),
                                MakeProblem(p, parameter.datatype).Describe(p.direction),
                                parameter.datatype,
                                false);
    const auto tile = std::find_if(catalog.begin(), catalog.end(), [](const auto& id) {
        return id.find("CKTileGroupedConv") == 0;
    });
    ASSERT_NE(tile, catalog.end()) << "Enabled gfx1250 validation requires a Tile candidate";
    for(const auto& id : catalog)
        if(id.find("CKTileGroupedConv") == 0)
            RunSupportCase(loader, parameter, id);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CkTileGroupedConvSupport_NONE,
                         testing::ValuesIn(SupportCases()),
                         [](const testing::TestParamInfo<SupportCase>& info) {
                             const auto datatype = info.param.datatype == miopenHalf       ? "FP16"
                                                   : info.param.datatype == miopenBFloat16 ? "BF16"
                                                                                           : "FP32";
                             return datatype + CaseName({info.param.spatial_dims,
                                                         info.param.direction,
                                                         true,
                                                         TileShape::Padded});
                         });

#endif

class GPU_CkTileGroupedConv_FP16 : public testing::TestWithParam<TileCase>
{
};
class GPU_CkTileGroupedConv_BF16 : public testing::TestWithParam<TileCase>
{
};
class GPU_CkTileGroupedConv_FP32 : public testing::TestWithParam<TileCase>
{
};

TEST_P(GPU_CkTileGroupedConv_FP16, NumericalAndInvocationLifetime)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE
    RunNumericalCase<half_float::half>(GetParam(), miopenHalf);
#else
    GTEST_SKIP() << "MIOPEN_USE_CK_TILE and the HIP backend are required";
#endif
}

TEST_P(GPU_CkTileGroupedConv_BF16, NumericalAndInvocationLifetime)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE
    RunNumericalCase<bfloat16>(GetParam(), miopenBFloat16);
#else
    GTEST_SKIP() << "MIOPEN_USE_CK_TILE and the HIP backend are required";
#endif
}

TEST_P(GPU_CkTileGroupedConv_FP32, NumericalAndInvocationLifetime)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE
    RunNumericalCase<float>(GetParam(), miopenFloat);
#else
    GTEST_SKIP() << "MIOPEN_USE_CK_TILE and the HIP backend are required";
#endif
}

const auto numerical_name = [](const testing::TestParamInfo<TileCase>& info) {
    return CaseName(info.param);
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CkTileGroupedConv_FP16,
                         testing::ValuesIn(NumericalCases()),
                         numerical_name);
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CkTileGroupedConv_BF16,
                         testing::ValuesIn(NumericalCases()),
                         numerical_name);
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CkTileGroupedConv_FP32,
                         testing::ValuesIn(NumericalCases()),
                         numerical_name);

} // namespace
