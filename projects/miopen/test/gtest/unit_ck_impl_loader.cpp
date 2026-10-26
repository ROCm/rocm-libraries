// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/convolution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/tensor.hpp>
#include "get_handle.hpp"
#include "../workspace.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <cmath>
#include <iterator>

#include <thread>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime.h>
#endif

using miopen::solver::CKSolverType;
using miopen::solver::GetCurrentDeviceName;
using miopen::solver::IsDeterministicSplitKValid;

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_CK_LIB_PATH)

namespace {

class ScopedCKLibraryPath
{
public:
    explicit ScopedCKLibraryPath(std::string value)
        : had_previous_value(static_cast<bool>(MIOPEN_CK_LIB_PATH))
    {
        if(had_previous_value)
            previous_value = miopen::env::value(MIOPEN_CK_LIB_PATH);
        miopen::env::update(MIOPEN_CK_LIB_PATH, std::move(value));
    }

    ~ScopedCKLibraryPath()
    {
        if(had_previous_value)
            miopen::env::update(MIOPEN_CK_LIB_PATH, previous_value);
        else
            miopen::env::clear(MIOPEN_CK_LIB_PATH);
    }

    ScopedCKLibraryPath(const ScopedCKLibraryPath&)            = delete;
    ScopedCKLibraryPath& operator=(const ScopedCKLibraryPath&) = delete;

private:
    bool had_previous_value = false;
    std::string previous_value;
};

miopen::fs::path MakeMissingDirectory(const std::string& tag)
{
    return miopen::fs::temp_directory_path() / "miopen_ck_impl_loader_missing" / tag /
           "not_created";
}

/// Build a minimal grouped-conv forward ProblemDescription suitable for
/// querying the CK loader.  Uses group=4, NHWC, FP16, small spatial dims.
miopen::conv::ProblemDescription MakeGroupedConvProblem()
{
    // Input:   N=1, C=16, H=8, W=8  (NHWC layout)
    // Weights: K=16, C/G=4, R=3, S=3 (NHWC layout, group=4 → C_per_group=4)
    // Conv:    pad=1, stride=1, dilation=1, group=4
    const miopen::TensorDescriptor in_desc(miopenHalf, miopenTensorNHWC, {1, 16, 8, 8});
    const miopen::TensorDescriptor wei_desc(miopenHalf, miopenTensorNHWC, {16, 4, 3, 3});

    const miopen::ConvolutionDescriptor conv_desc(
        /*pads=*/{1, 1},
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*trans_output_pads=*/{0, 0},
        /*group_count=*/4);

    const auto out_desc = conv_desc.GetForwardOutputTensor(in_desc, wei_desc, miopenHalf);

    return miopen::conv::ProblemDescription(
        in_desc, wei_desc, out_desc, conv_desc, miopen::conv::Direction::Forward);
}

} // namespace

// -- GPU tests (require a HIP device) -----------------------------------------

#if MIOPEN_BACKEND_HIP

TEST(GPU_CkImplLoader_FP16, LoaderLoadsForCurrentDevice)
{
    const auto device_name = GetCurrentDeviceName();
    ASSERT_FALSE(device_name.empty()) << "Failed to query HIP device";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);

    // The library may or may not be installed; if it loads, symbols must resolve.
    // We don't hard-fail on IsLoaded()==false because the .so might not be
    // present in all CI environments.
    if(loader.IsLoaded())
    {
        SUCCEED() << "Loader successfully loaded library for " << device_name;
    }
    else
    {
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;
    }
}

TEST(GPU_CkImplLoader_FP16, LoaderFillsValidKernels)
{
    const auto device_name = GetCurrentDeviceName();
    ASSERT_FALSE(device_name.empty());

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem = MakeGroupedConvProblem();
    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvFwd, problem, miopenHalf, false);

    EXPECT_FALSE(kernels.empty()) << "Expected at least one valid CK grouped conv kernel for "
                                  << device_name;
}

static bool IsNativeOutputKernel(const std::string& id) { return id.find("NativeOutput:") == 0; }

static miopen::conv::ProblemDescription
MakeNativeOutputForwardProblem(miopenDataType_t dtype,
                               unsigned spatial_dims,
                               int stride,
                               std::size_t outputs_per_group,
                               bool channels_last   = false,
                               miopen::Scalar alpha = miopen::Scalar(1.0),
                               miopen::Scalar beta  = miopen::Scalar(0.0))
{
    const auto layout = spatial_dims == 2 ? (channels_last ? miopenTensorNHWC : miopenTensorNCHW)
                                          : (channels_last ? miopenTensorNDHWC : miopenTensorNCDHW);
    // C/G=8, two groups/batches; K/G=20 exercises the output tile tail.
    std::vector<std::size_t> xl{2, 16};
    if(spatial_dims == 3)
        xl.push_back(stride == 1 ? 5 : 9);
    xl.push_back(stride == 1 ? 7 : 9);
    xl.push_back(stride == 1 ? 9 : 15);
    std::vector<std::size_t> wl{2 * outputs_per_group, 8};
    wl.resize(spatial_dims + 2, 3);
    const miopen::TensorDescriptor x(dtype, layout, xl);
    const miopen::TensorDescriptor w(dtype, layout, wl);
    const miopen::ConvolutionDescriptor conv(std::vector<int>(spatial_dims, stride),
                                             std::vector<int>(spatial_dims, stride),
                                             std::vector<int>(spatial_dims, stride),
                                             std::vector<int>(spatial_dims, 0),
                                             2);
    const auto y = conv.GetForwardOutputTensor(x, w, dtype);
    return {x, w, y, conv, miopen::conv::Direction::Forward, 0, alpha, beta};
}

template <typename DataType>
static std::vector<DataType>
NativeOutputForwardReference(const miopen::conv::ProblemDescription& problem,
                             const std::vector<DataType>& input,
                             const std::vector<DataType>& weights)
{
    const auto& xl   = problem.GetIn().GetLengths();
    const auto& xs   = problem.GetIn().GetStrides();
    const auto& wl   = problem.GetWeights().GetLengths();
    const auto& ws   = problem.GetWeights().GetStrides();
    const auto& yl   = problem.GetOut().GetLengths();
    const auto dims  = problem.GetSpatialDims();
    const auto& conv = problem.GetConv();
    std::size_t taps = 1;
    for(unsigned d = 0; d < dims; ++d)
        taps *= wl[d + 2];
    std::vector<DataType> expected(problem.GetOut().GetElementSize());
    for(std::size_t i = 0; i < expected.size(); ++i)
    {
        std::array<std::size_t, 5> output_coordinate{};
        auto remainder = i;
        for(std::size_t d = yl.size(); d-- > 0;)
        {
            output_coordinate[d] = remainder % yl[d];
            remainder /= yl[d];
        }
        const auto n     = output_coordinate[0];
        const auto k     = output_coordinate[1];
        const auto group = k / (wl[0] / problem.GetGroupCount());
        double sum       = 0;
        for(std::size_t c = 0; c < wl[1]; ++c)
            for(std::size_t tap = 0; tap < taps; ++tap)
            {
                auto xi               = n * xs[0] + (group * wl[1] + c) * xs[1];
                auto wi               = k * ws[0] + c * ws[1];
                auto filter_remainder = tap;
                bool inside           = true;
                for(unsigned d = dims; d-- > 0;)
                {
                    const auto filter_coordinate = filter_remainder % wl[d + 2];
                    filter_remainder /= wl[d + 2];
                    const auto position =
                        static_cast<int>(output_coordinate[d + 2]) * conv.GetConvStrides()[d] +
                        static_cast<int>(filter_coordinate) * conv.GetConvDilations()[d] -
                        conv.GetConvPads()[d];
                    if(position < 0 || static_cast<std::size_t>(position) >= xl[d + 2])
                    {
                        inside = false;
                        break;
                    }
                    xi += static_cast<std::size_t>(position) * xs[d + 2];
                    wi += filter_coordinate * ws[d + 2];
                }
                if(inside)
                    sum += static_cast<double>(static_cast<float>(input[xi])) *
                           static_cast<float>(weights[wi]);
            }
        expected[i] = DataType{static_cast<float>(sum)};
    }
    return expected;
}

template <typename DataType>
static void CheckNativeOutputForward(miopenDataType_t dtype, unsigned spatial_dims)
{
    const auto device_name = GetCurrentDeviceName();
    const bool gfx1250     = device_name.substr(0, device_name.find(':')) == "gfx1250";
    const auto& loader     = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;
    const auto solver = spatial_dims == 2 ? CKSolverType::GrpConvFwd : CKSolverType::GrpConv3dFwd;
    const auto expect_no_native = [&](const auto& problem, bool tf32 = false) {
        const auto ids = loader.FillValidKernels(solver, problem, dtype, tf32);
        EXPECT_TRUE(std::none_of(ids.begin(), ids.end(), IsNativeOutputKernel));
    };
    const auto base = MakeNativeOutputForwardProblem(dtype, spatial_dims, 1, 20);
    expect_no_native(MakeNativeOutputForwardProblem(dtype, spatial_dims, 1, 20, true));
    expect_no_native(MakeNativeOutputForwardProblem(dtype, spatial_dims, 1, 19));
    {
        auto pointwise_lengths = base.GetWeights().GetLengths();
        std::fill(pointwise_lengths.begin() + 2, pointwise_lengths.end(), 1);
        const auto layout = spatial_dims == 2 ? miopenTensorNCHW : miopenTensorNCDHW;
        const miopen::TensorDescriptor weights(dtype, layout, pointwise_lengths);
        const miopen::ConvolutionDescriptor conv(std::vector<int>(spatial_dims, 0),
                                                 std::vector<int>(spatial_dims, 1),
                                                 std::vector<int>(spatial_dims, 1),
                                                 std::vector<int>(spatial_dims, 0),
                                                 2);
        expect_no_native(miopen::conv::ProblemDescription{
            base.GetIn(),
            weights,
            conv.GetForwardOutputTensor(base.GetIn(), weights, dtype),
            conv,
            miopen::conv::Direction::Forward});
    }
    for(const auto scalars : {std::pair<double, double>{2.0, 0.0},
                              {1.0, 1.0},
                              {1.0 + std::ldexp(1.0, -30), 0.0},
                              {1.0, std::ldexp(1.0, -150)}})
        expect_no_native(MakeNativeOutputForwardProblem(dtype,
                                                        spatial_dims,
                                                        1,
                                                        20,
                                                        false,
                                                        miopen::Scalar(scalars.first),
                                                        miopen::Scalar(scalars.second)));
    if(dtype == miopenFloat)
        expect_no_native(base, true);
    if(!gfx1250)
    {
        expect_no_native(base);
        return;
    }

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    constexpr std::size_t outputs_per_group = 20;
    for(const int stride : {1, 2})
    {
        auto problem =
            MakeNativeOutputForwardProblem(dtype, spatial_dims, stride, outputs_per_group);
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
        const auto kernels = loader.FillValidKernels(solver, problem, dtype, false);
        std::vector<std::string> native;
        std::copy_if(
            kernels.begin(), kernels.end(), std::back_inserter(native), IsNativeOutputKernel);
        ASSERT_FALSE(native.empty()) << "dims=" << spatial_dims << " stride=" << stride;
        const auto& x_desc = problem.GetIn();
        const auto& w_desc = problem.GetWeights();
        const auto& y_desc = problem.GetOut();
        const miopen::MultiBufferWorkspaceTraits regions(
            {miopen::solver::GetPackedSize(x_desc), miopen::solver::GetPackedSize(w_desc)});
        // A 16-byte-aligned, non-256-byte-aligned base is supported; region offsets still
        // follow MultiBufferWorkspaceTraits' 256-byte padding.
        constexpr std::size_t guard = 16;
        // Oversize each operand allocation so an erroneously accepted alias stays in-bounds.
        const auto operand_bytes = std::max(
            {x_desc.GetNumBytes(), w_desc.GetNumBytes(), y_desc.GetNumBytes(), regions.GetSize()});
        Workspace x_buffer(operand_bytes + 2 * guard);
        Workspace w_buffer(operand_bytes + 2 * guard);
        Workspace y_buffer(operand_bytes + 2 * guard);
        auto* x_ptr = static_cast<unsigned char*>(x_buffer.ptr()) + guard;
        auto* w_ptr = static_cast<unsigned char*>(w_buffer.ptr()) + guard;
        auto* y_ptr = static_cast<unsigned char*>(y_buffer.ptr()) + guard;
        std::vector<DataType> input(x_desc.GetElementSize());
        std::vector<DataType> weights(w_desc.GetElementSize());
        for(const auto& selected : native)
        {
            SCOPED_TRACE(selected);
            const auto solution = loader.GetSolution(solver, ctx, problem, selected, false);
            ASSERT_EQ(solution.status, miopenStatusSuccess);
            ASSERT_TRUE(solution.invoker_factory);
            ASSERT_EQ(solution.workspace_sz, regions.GetSize());
            Workspace scratch(solution.workspace_sz + y_desc.GetNumBytes() + 2 * guard);
            auto* scratch_ptr = static_cast<unsigned char*>(scratch.ptr()) + guard;
            const auto invoker =
                handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
            for(const int pass : {0, 1})
            {
                SCOPED_TRACE(pass);
                // Change BOTH operands without changing their device pointers.
                // FP32 fractions distinguish exact input products from TF32 truncation.
                for(std::size_t i = 0; i < input.size(); ++i)
                {
                    float value =
                        static_cast<float>(static_cast<int>((i * 13 + i / 7 + pass) % 5) - 2);
                    if constexpr(std::is_same_v<DataType, float>)
                        value = value / 16 + (pass == 0 ? 1.0f : -1.0f) / 8192;
                    input[i] = DataType{value};
                }
                for(std::size_t i = 0; i < weights.size(); ++i)
                {
                    float value =
                        static_cast<float>(static_cast<int>((i * 17 + i / 11 + pass) % 3) - 1);
                    if constexpr(std::is_same_v<DataType, float>)
                        value = value / 16 + (pass == 0 ? 1.0f : -1.0f) / 4096;
                    weights[i] = DataType{value};
                }
                ASSERT_EQ(hipMemset(x_buffer.ptr(), 0xa5, x_buffer.size()), hipSuccess);
                ASSERT_EQ(hipMemset(w_buffer.ptr(), 0xa5, w_buffer.size()), hipSuccess);
                if(pass == 0)
                    ASSERT_EQ(hipMemset(y_buffer.ptr(), 0xa5, y_buffer.size()), hipSuccess);
                ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
                ASSERT_EQ(
                    hipMemcpy(x_ptr, input.data(), x_desc.GetNumBytes(), hipMemcpyHostToDevice),
                    hipSuccess);
                ASSERT_EQ(
                    hipMemcpy(w_ptr, weights.data(), w_desc.GetNumBytes(), hipMemcpyHostToDevice),
                    hipSuccess);
                // First launch sees poison; second launch sees the previous result.
                if(pass == 0)
                    ASSERT_EQ(hipMemset(y_ptr, 0x7f, y_desc.GetNumBytes()), hipSuccess);
                const auto x_before = x_buffer.Read<std::vector<unsigned char>>();
                const auto w_before = w_buffer.Read<std::vector<unsigned char>>();
                const miopen::ConvFwdTensors tensors{x_desc, x_ptr, w_desc, w_ptr, y_desc, y_ptr};
                miopen::conv::DataInvokeParams params{
                    tensors, scratch_ptr, solution.workspace_sz, false};
                if(pass == 0 && stride == 1 && selected == native.front())
                {
                    const auto y_before       = y_buffer.Read<std::vector<unsigned char>>();
                    const auto scratch_before = scratch.Read<std::vector<unsigned char>>();
                    const auto reject         = [&](const auto& invalid) {
                        EXPECT_THROW(invoker(handle, invalid), miopen::Exception);
                        handle.Finish();
                        EXPECT_EQ(y_buffer.Read<std::vector<unsigned char>>(), y_before);
                        EXPECT_EQ(x_buffer.Read<std::vector<unsigned char>>(), x_before);
                        EXPECT_EQ(w_buffer.Read<std::vector<unsigned char>>(), w_before);
                        EXPECT_EQ(scratch.Read<std::vector<unsigned char>>(), scratch_before);
                    };
                    auto invalid      = params;
                    invalid.workSpace = nullptr;
                    reject(invalid);
                    invalid = params;
                    invalid.workSpaceSize--;
                    reject(invalid);
                    invalid           = params;
                    invalid.workSpace = scratch_ptr + 1;
                    reject(invalid);
                    for(auto* operand : {x_ptr, w_ptr, y_ptr})
                    {
                        invalid           = params;
                        invalid.workSpace = operand;
                        reject(invalid);
                    }
                    for(auto* alias : {x_ptr, x_ptr + 16, w_ptr, w_ptr + 16, scratch_ptr})
                    {
                        invalid             = params;
                        invalid.tensors.out = alias;
                        reject(invalid);
                    }
                    invalid           = params;
                    invalid.tensors.w = x_ptr;
                    reject(invalid);
                    for(unsigned slot = 0; slot < 3; ++slot)
                    {
                        invalid = params;
                        if(slot == 0)
                            invalid.tensors.in = nullptr;
                        else if(slot == 1)
                            invalid.tensors.w = nullptr;
                        else
                            invalid.tensors.out = nullptr;
                        reject(invalid);
                        invalid = params;
                        if(slot == 0)
                            invalid.tensors.in = x_ptr + 1;
                        else if(slot == 1)
                            invalid.tensors.w = w_ptr + 1;
                        else
                            invalid.tensors.out = y_ptr + 1;
                        reject(invalid);
                        for(unsigned mismatch = 0; mismatch < 4; ++mismatch)
                        {
                            invalid      = params;
                            auto& desc   = slot == 0   ? invalid.tensors.inDesc
                                           : slot == 1 ? invalid.tensors.wDesc
                                                       : invalid.tensors.outDesc;
                            auto lengths = desc.GetLengths();
                            auto strides = desc.GetStrides();
                            const auto layout =
                                spatial_dims == 2 ? miopenTensorNCHW : miopenTensorNCDHW;
                            if(mismatch == 0)
                            {
                                strides[1]++;
                                desc = miopen::TensorDescriptor(dtype, layout, lengths, strides);
                            }
                            else if(mismatch == 1)
                            {
                                lengths.back()--;
                                desc = miopen::TensorDescriptor(dtype, layout, lengths);
                            }
                            else if(mismatch == 2)
                                desc = miopen::TensorDescriptor(
                                    dtype,
                                    spatial_dims == 2 ? miopenTensorNHWC : miopenTensorNDHWC,
                                    lengths);
                            else
                                desc = miopen::TensorDescriptor(dtype == miopenFloat ? miopenHalf
                                                                                     : miopenFloat,
                                                                layout,
                                                                lengths);
                            reject(invalid);
                        }
                    }
                    for(const auto scalars : {std::pair<double, double>{2.0, 0.0},
                                              {1.0, 1.0},
                                              {1.0 + std::ldexp(1.0, -30), 0.0},
                                              {1.0, std::ldexp(1.0, -150)}})
                    {
                        invalid       = params;
                        invalid.alpha = miopen::Scalar(scalars.first);
                        invalid.beta  = miopen::Scalar(scalars.second);
                        reject(invalid);
                    }
                }
                if(pass == 1)
                {
                    const float alpha = 1, beta = 0;
                    params.alpha = miopen::Scalar(&alpha, miopenFloat);
                    params.beta  = miopen::Scalar(&beta, miopenFloat);
                }
                invoker(handle, params);
                handle.Finish();
                const auto expected = NativeOutputForwardReference(problem, input, weights);
                std::vector<DataType> actual(expected.size());
                ASSERT_EQ(
                    hipMemcpy(actual.data(), y_ptr, y_desc.GetNumBytes(), hipMemcpyDeviceToHost),
                    hipSuccess);
                for(std::size_t i = 0; i < actual.size(); ++i)
                {
                    if constexpr(std::is_same_v<DataType, float>)
                        EXPECT_NEAR(actual[i], expected[i], 1e-6f) << "index=" << i;
                    else
                        EXPECT_EQ(static_cast<float>(actual[i]), static_cast<float>(expected[i]))
                            << "index=" << i;
                }
                EXPECT_EQ(x_buffer.Read<std::vector<unsigned char>>(), x_before);
                EXPECT_EQ(w_buffer.Read<std::vector<unsigned char>>(), w_before);
                const auto check_guards = [&](const Workspace& buffer, std::size_t used) {
                    const auto bytes = buffer.Read<std::vector<unsigned char>>();
                    EXPECT_TRUE(std::all_of(bytes.begin(),
                                            bytes.begin() + guard,
                                            [](unsigned char b) { return b == 0xa5; }));
                    EXPECT_TRUE(std::all_of(bytes.begin() + guard + used,
                                            bytes.end(),
                                            [](unsigned char b) { return b == 0xa5; }));
                };
                check_guards(y_buffer, y_desc.GetNumBytes());
                check_guards(scratch, solution.workspace_sz);
            }
        }
    }
}

TEST(GPU_CkImplLoader_FP16, NchwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<half_float::half>(miopenHalf, 2);
}

TEST(GPU_CkImplLoader_BFP16, NchwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<bfloat16>(miopenBFloat16, 2);
}

TEST(GPU_CkImplLoader_FP32, NchwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<float>(miopenFloat, 2);
}

TEST(GPU_CkImplLoader_FP16, NcdhwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<half_float::half>(miopenHalf, 3);
}

TEST(GPU_CkImplLoader_BFP16, NcdhwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<bfloat16>(miopenBFloat16, 3);
}

TEST(GPU_CkImplLoader_FP32, NcdhwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<float>(miopenFloat, 3);
}

TEST(GPU_CkImplLoader_FP16, LoaderFillsValidKernelsWithTf32Fallback)
{
    const auto device_name = GetCurrentDeviceName();
    ASSERT_FALSE(device_name.empty());

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const auto problem = MakeGroupedConvProblem();
    bool use_tf32      = false;
    const auto kernels = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConvFwd, problem, miopenHalf, use_tf32);

    EXPECT_FALSE(kernels.empty()) << "Expected at least one valid kernel for " << device_name;
    EXPECT_FALSE(use_tf32) << "use_tf32 should remain false when called with false";
}

TEST(GPU_CkImplLoader_FP16, LoaderCachesPerDevice)
{
    const auto device_name = GetCurrentDeviceName();
    ASSERT_FALSE(device_name.empty());

    const auto& loader1 = miopen::solver::CkImplLibLoader::Get(device_name);
    const auto& loader2 = miopen::solver::CkImplLibLoader::Get(device_name);

    EXPECT_EQ(&loader1, &loader2) << "Get() should return the same cached instance";
}

TEST(GPU_CkImplLoader_FP16, LoaderStripsDeviceSuffix)
{
    const auto device_name = GetCurrentDeviceName();
    ASSERT_FALSE(device_name.empty());

    // Strip any existing suffix to get the base arch
    auto base_arch = device_name;
    auto colon_pos = base_arch.find(':');
    if(colon_pos != std::string::npos)
        base_arch = base_arch.substr(0, colon_pos);

    // Query with a synthesized suffix and with the bare base arch
    const auto suffixed = base_arch + ":sramecc+:xnack-";
    const auto& loader1 = miopen::solver::CkImplLibLoader::Get(suffixed);
    const auto& loader2 = miopen::solver::CkImplLibLoader::Get(base_arch);

    EXPECT_EQ(&loader1, &loader2)
        << "Suffixed and bare device names should resolve to the same cached loader";
}

#endif // MIOPEN_BACKEND_HIP

// -- CPU tests (no GPU required) ----------------------------------------------

TEST(CPU_CkImplLoader_NONE, LoaderFailsGracefullyForUnknownDevice)
{
    const auto& loader = miopen::solver::CkImplLibLoader::Get("gfx_nonexistent");
    EXPECT_FALSE(loader.IsLoaded());
}

TEST(CPU_CkImplLoader_NONE, LoaderStripsDeviceSuffixWithoutGpu)
{
    const auto base_arch = std::string{"gfx_ck_loader_suffix_unit_test"};
    const auto suffixed  = base_arch + ":sramecc+:xnack-";

    const auto& loader1 = miopen::solver::CkImplLibLoader::Get(suffixed);
    const auto& loader2 = miopen::solver::CkImplLibLoader::Get(base_arch);

    EXPECT_EQ(&loader1, &loader2)
        << "Suffixed and bare device names should resolve to the same cached loader";
    EXPECT_FALSE(loader1.IsLoaded());
}

TEST(CPU_CkImplLoader_NONE, LoaderFailsGracefullyWithInvalidEnvPath)
{
    const auto missing_dir = MakeMissingDirectory("invalid_env_path");
    ASSERT_FALSE(miopen::fs::exists(missing_dir));

    const ScopedCKLibraryPath scoped_ck_lib_path{missing_dir.string()};
    const auto& loader = miopen::solver::CkImplLibLoader::Get("gfx_ck_loader_invalid_env");

    EXPECT_FALSE(loader.IsLoaded());
    EXPECT_TRUE(
        loader
            .FillValidKernels(CKSolverType::GrpConvFwd, MakeGroupedConvProblem(), miopenHalf, false)
            .empty());
}

TEST(CPU_CkImplLoader_NONE, LoaderReturnsEmptyOnFailure)
{
    const auto& loader = miopen::solver::CkImplLibLoader::Get("gfx_bogus");
    ASSERT_FALSE(loader.IsLoaded());

    const auto problem = MakeGroupedConvProblem();
    miopen::ExecutionContext ctx;

    // All wrappers should return safe defaults when the library is not loaded
    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConvFwd, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvFwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvFwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConvFwd, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConvFwd, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConvFwd, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConvFwd, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConvBwd, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvBwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvBwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConvBwd, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConvBwd, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConvBwd, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConvBwd, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConvWrw, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvWrw, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConvWrw, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConvWrw, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConvWrw, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConvWrw, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConvWrw, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConv3dFwd, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dFwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dFwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConv3dFwd, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConv3dFwd, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConv3dFwd, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConv3dFwd, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConv3dBwd, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dBwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dBwd, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConv3dBwd, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConv3dBwd, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConv3dBwd, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConv3dBwd, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::GrpConv3dWrw, problem, miopenHalf, false).empty());
    {
        bool use_tf32 = true;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dWrw, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should become false when no kernels found";
    }
    {
        bool use_tf32 = false;
        EXPECT_TRUE(loader
                        .FillValidKernelsWithTf32Fallback(
                            CKSolverType::GrpConv3dWrw, problem, miopenHalf, use_tf32)
                        .empty());
        EXPECT_FALSE(use_tf32) << "use_tf32 should remain false";
    }
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::GrpConv3dWrw, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConv3dWrw, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConv3dWrw, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::GrpConv3dWrw, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::FusedBiasActiv, problem, miopenHalf, false).empty());
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::FusedBiasActiv, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::FusedBiasActiv, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::FusedBiasActiv, problem, miopenHalf, false),
              0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::FusedBiasActiv, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::FusedBiasResAddActiv, problem, miopenHalf, false)
            .empty());
    EXPECT_FALSE(
        loader.IsApplicable(CKSolverType::FusedBiasResAddActiv, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::FusedBiasResAddActiv, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(
        loader.GetWorkspaceSize(CKSolverType::FusedBiasResAddActiv, problem, miopenHalf, false),
        0u);
    EXPECT_EQ(
        loader.GetSolution(CKSolverType::FusedBiasResAddActiv, ctx, problem, "dummy", false).status,
        miopenStatusInternalError);

    EXPECT_TRUE(
        loader.FillValidKernels(CKSolverType::FusedGrpActiv, problem, miopenHalf, false).empty());
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::FusedGrpActiv, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::FusedGrpActiv, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::FusedGrpActiv, problem, miopenHalf, false), 0u);
    EXPECT_EQ(loader.GetSolution(CKSolverType::FusedGrpActiv, ctx, problem, "dummy", false).status,
              miopenStatusInternalError);

    EXPECT_TRUE(loader.FillValidKernels(CKSolverType::FusedGrpBiasActiv, problem, miopenHalf, false)
                    .empty());
    EXPECT_FALSE(loader.IsApplicable(CKSolverType::FusedGrpBiasActiv, problem, miopenHalf, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::FusedGrpBiasActiv, problem, "dummy_kernel", miopenHalf, false));
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::FusedGrpBiasActiv, problem, miopenHalf, false),
              0u);
    EXPECT_EQ(
        loader.GetSolution(CKSolverType::FusedGrpBiasActiv, ctx, problem, "dummy", false).status,
        miopenStatusInternalError);
}

TEST(CPU_CkImplLoader_NONE, IsDeterministicSplitKValid)
{
    // Non-deterministic mode: always valid regardless of split_k
    EXPECT_TRUE(IsDeterministicSplitKValid("Kernel+4", false));
    EXPECT_TRUE(IsDeterministicSplitKValid("Kernel+1", false));
    EXPECT_TRUE(IsDeterministicSplitKValid("Kernel", false));

    // Deterministic mode with split_k == 1 or no split_k: valid
    EXPECT_TRUE(IsDeterministicSplitKValid("Kernel+1", true));
    EXPECT_TRUE(IsDeterministicSplitKValid("Kernel", true));

    // Deterministic mode with split_k > 1: invalid
    EXPECT_FALSE(IsDeterministicSplitKValid("Kernel+4", true));
    EXPECT_FALSE(IsDeterministicSplitKValid("Kernel+2", true));

    // Deterministic mode with parse failures: invalid
    EXPECT_FALSE(IsDeterministicSplitKValid("Kernel+abc", true));
    EXPECT_FALSE(IsDeterministicSplitKValid("Kernel+", true));
}

// -- Error infrastructure tests (CPU, no GPU required) ------------------------

TEST(CPU_CkImplError_NONE, LastErrorSetAndGet)
{
    // Setting a non-success status stores the message
    CkImplLastError::setLastError(CK_IMPL_STATUS_BAD_PARAM, "test error message");
    EXPECT_STREQ(CkImplLastError::getLastError(), "test error message");

    // Setting SUCCESS does NOT overwrite the buffer
    CkImplLastError::setLastError(CK_IMPL_STATUS_SUCCESS, "should not appear");
    EXPECT_STREQ(CkImplLastError::getLastError(), "test error message");

    // Setting another error overwrites the buffer
    CkImplLastError::setLastError(CK_IMPL_STATUS_INTERNAL_ERROR, "second error");
    EXPECT_STREQ(CkImplLastError::getLastError(), "second error");

    // Null message clears the buffer
    CkImplLastError::setLastError(CK_IMPL_STATUS_BAD_PARAM, nullptr);
    EXPECT_STREQ(CkImplLastError::getLastError(), "");
}

TEST(CPU_CkImplError_NONE, LastErrorThreadIsolation)
{
    // Set an error on the main thread
    CkImplLastError::setLastError(CK_IMPL_STATUS_BAD_PARAM, "main thread error");

    std::string child_error;
    std::thread t([&child_error]() {
        // Child thread should see empty last error (thread-local)
        child_error = CkImplLastError::getLastError();
    });
    t.join();

    EXPECT_EQ(child_error, "") << "Child thread should have empty last error";
    EXPECT_STREQ(CkImplLastError::getLastError(), "main thread error")
        << "Main thread error should be unaffected";
}

TEST(CPU_CkImplError_NONE, TryCatchSuccess)
{
    auto status = ck_impl_try_catch([]() {
        // no-op, no throw
    });
    EXPECT_EQ(status, CK_IMPL_STATUS_SUCCESS);
}

TEST(CPU_CkImplError_NONE, TryCatchCkImplException)
{
    auto status = ck_impl_try_catch(
        []() { throw CkImplException(CK_IMPL_STATUS_BAD_PARAM, "bad parameter test"); });
    EXPECT_EQ(status, CK_IMPL_STATUS_BAD_PARAM);
    EXPECT_STREQ(CkImplLastError::getLastError(), "bad parameter test");
}

TEST(CPU_CkImplError_NONE, TryCatchStdException)
{
    auto status = ck_impl_try_catch([]() { throw std::runtime_error("std runtime error test"); });
    EXPECT_EQ(status, CK_IMPL_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(CkImplLastError::getLastError(), "std runtime error test");
}

TEST(CPU_CkImplError_NONE, TryCatchUnknownException)
{
    auto status = ck_impl_try_catch([]() {
        throw 42; // NOLINT(hicpp-exception-baseclass)
    });
    EXPECT_EQ(status, CK_IMPL_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(CkImplLastError::getLastError(), "Unknown exception occurred");
}

TEST(CPU_CkImplError_NONE, StatusToString)
{
    EXPECT_STREQ(toString(CK_IMPL_STATUS_SUCCESS), "CK_IMPL_STATUS_SUCCESS");
    EXPECT_STREQ(toString(CK_IMPL_STATUS_BAD_PARAM), "CK_IMPL_STATUS_BAD_PARAM");
    EXPECT_STREQ(toString(CK_IMPL_STATUS_INVALID_VALUE), "CK_IMPL_STATUS_INVALID_VALUE");
    EXPECT_STREQ(toString(CK_IMPL_STATUS_INTERNAL_ERROR), "CK_IMPL_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(toString(CK_IMPL_STATUS_ALLOC_FAILED), "CK_IMPL_STATUS_ALLOC_FAILED");
    EXPECT_STREQ(toString(static_cast<ck_impl_status_t>(99)), "CK_IMPL_STATUS_UNKNOWN");
}

// -- CkImplException direct tests -----------------------------------------

TEST(CPU_CkImplError_NONE, ExceptionStoresStatusAndMessage)
{
    CkImplException ex(CK_IMPL_STATUS_INVALID_VALUE, "invalid value message");
    EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_INVALID_VALUE);
    EXPECT_STREQ(ex.what(), "invalid value message");
}

TEST(CPU_CkImplError_NONE, ExceptionIsStdException)
{
    CkImplException ex(CK_IMPL_STATUS_ALLOC_FAILED, "alloc failed");
    const std::exception& base = ex;
    EXPECT_STREQ(base.what(), "alloc failed");
}

TEST(CPU_CkImplError_NONE, ExceptionCopyPreservesFields)
{
    CkImplException original(CK_IMPL_STATUS_BAD_PARAM, "original message");
    CkImplException copy(original); // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT_EQ(copy.getStatus(), CK_IMPL_STATUS_BAD_PARAM);
    EXPECT_STREQ(copy.what(), "original message");
}

// -- THROW_IF macro tests ----------------------------------------------------

TEST(CPU_CkImplError_NONE, ThrowIfNullThrowsOnNull)
{
    int* ptr = nullptr;
    EXPECT_THROW(
        { CK_IMPL_THROW_IF_NULL(ptr, CK_IMPL_STATUS_BAD_PARAM, "null pointer"); }, CkImplException);

    try
    {
        CK_IMPL_THROW_IF_NULL(ptr, CK_IMPL_STATUS_BAD_PARAM, "null pointer detail");
        FAIL() << "Expected CkImplException";
    }
    catch(const CkImplException& ex)
    {
        EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_BAD_PARAM);
        EXPECT_STREQ(ex.what(), "null pointer detail");
    }
}

TEST(CPU_CkImplError_NONE, ThrowIfNullPassesOnNonNull)
{
    int value = 42;
    int* ptr  = &value;
    EXPECT_NO_THROW({ CK_IMPL_THROW_IF_NULL(ptr, CK_IMPL_STATUS_BAD_PARAM, "should not throw"); });
}

TEST(CPU_CkImplError_NONE, ThrowIfFalseThrowsOnFalse)
{
    try
    {
        CK_IMPL_THROW_IF_FALSE(false, CK_IMPL_STATUS_INVALID_VALUE, "condition was false");
        FAIL() << "Expected CkImplException";
    }
    catch(const CkImplException& ex)
    {
        EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_INVALID_VALUE);
        EXPECT_STREQ(ex.what(), "condition was false");
    }
}

TEST(CPU_CkImplError_NONE, ThrowIfFalsePassesOnTrue)
{
    EXPECT_NO_THROW(
        { CK_IMPL_THROW_IF_FALSE(true, CK_IMPL_STATUS_INVALID_VALUE, "should not throw"); });
}

TEST(CPU_CkImplError_NONE, ThrowIfTrueThrowsOnTrue)
{
    try
    {
        CK_IMPL_THROW_IF_TRUE(true, CK_IMPL_STATUS_INTERNAL_ERROR, "condition was true");
        FAIL() << "Expected CkImplException";
    }
    catch(const CkImplException& ex)
    {
        EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_INTERNAL_ERROR);
        EXPECT_STREQ(ex.what(), "condition was true");
    }
}

TEST(CPU_CkImplError_NONE, ThrowIfTruePassesOnFalse)
{
    EXPECT_NO_THROW(
        { CK_IMPL_THROW_IF_TRUE(false, CK_IMPL_STATUS_INTERNAL_ERROR, "should not throw"); });
}

TEST(CPU_CkImplError_NONE, ThrowIfNeThrowsOnNotEqual)
{
    try
    {
        CK_IMPL_THROW_IF_NE(3, 5, CK_IMPL_STATUS_INVALID_VALUE, "values not equal");
        FAIL() << "Expected CkImplException";
    }
    catch(const CkImplException& ex)
    {
        EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_INVALID_VALUE);
        EXPECT_STREQ(ex.what(), "values not equal");
    }
}

TEST(CPU_CkImplError_NONE, ThrowIfNePassesOnEqual)
{
    EXPECT_NO_THROW(
        // cppcheck-suppress duplicateExpression
        { CK_IMPL_THROW_IF_NE(7, 7, CK_IMPL_STATUS_INVALID_VALUE, "should not throw"); });
}

TEST(CPU_CkImplError_NONE, ThrowIfEqThrowsOnEqual)
{
    try
    {
        // cppcheck-suppress duplicateExpression
        CK_IMPL_THROW_IF_EQ(10, 10, CK_IMPL_STATUS_BAD_PARAM, "values equal");
        FAIL() << "Expected CkImplException";
    }
    catch(const CkImplException& ex)
    {
        EXPECT_EQ(ex.getStatus(), CK_IMPL_STATUS_BAD_PARAM);
        EXPECT_STREQ(ex.what(), "values equal");
    }
}

TEST(CPU_CkImplError_NONE, ThrowIfEqPassesOnNotEqual)
{
    EXPECT_NO_THROW({ CK_IMPL_THROW_IF_EQ(1, 2, CK_IMPL_STATUS_BAD_PARAM, "should not throw"); });
}

// -- LastError additional coverage -------------------------------------------

TEST(CPU_CkImplError_NONE, LastErrorSetReturnsStatus)
{
    auto result = CkImplLastError::setLastError(CK_IMPL_STATUS_BAD_PARAM, "msg");
    EXPECT_EQ(result, CK_IMPL_STATUS_BAD_PARAM);

    result = CkImplLastError::setLastError(CK_IMPL_STATUS_ALLOC_FAILED, "alloc");
    EXPECT_EQ(result, CK_IMPL_STATUS_ALLOC_FAILED);

    result = CkImplLastError::setLastError(CK_IMPL_STATUS_SUCCESS, "ignored");
    EXPECT_EQ(result, CK_IMPL_STATUS_SUCCESS);
}

TEST(CPU_CkImplError_NONE, LastErrorStdStringOverload)
{
    std::string msg = "std::string overload test";
    CkImplLastError::setLastError(CK_IMPL_STATUS_INTERNAL_ERROR, msg);
    EXPECT_STREQ(CkImplLastError::getLastError(), "std::string overload test");
}

TEST(CPU_CkImplError_NONE, LastErrorTruncatesLongMessage)
{
    // Create a message longer than CK_IMPL_ERROR_STRING_MAX_LENGTH
    std::string long_msg(CK_IMPL_ERROR_STRING_MAX_LENGTH + 100, 'X');
    CkImplLastError::setLastError(CK_IMPL_STATUS_INTERNAL_ERROR, long_msg);

    const char* stored = CkImplLastError::getLastError();
    auto stored_len    = std::strlen(stored);
    EXPECT_EQ(stored_len, CK_IMPL_ERROR_STRING_MAX_LENGTH - 1)
        << "Stored message should be truncated to buffer size - 1";
    EXPECT_EQ(stored[stored_len], '\0') << "Stored message must be null-terminated";
}

// -- tryCatch status preservation for each status code -----------------------

TEST(CPU_CkImplError_NONE, TryCatchPreservesAllStatusCodes)
{
    const ck_impl_status_t codes[] = {
        CK_IMPL_STATUS_BAD_PARAM,
        CK_IMPL_STATUS_INVALID_VALUE,
        CK_IMPL_STATUS_INTERNAL_ERROR,
        CK_IMPL_STATUS_ALLOC_FAILED,
    };

    for(auto code : codes)
    {
        auto status = ck_impl_try_catch([code]() { throw CkImplException(code, toString(code)); });
        EXPECT_EQ(status, code) << "tryCatch should preserve status " << toString(code);
        EXPECT_STREQ(CkImplLastError::getLastError(), toString(code));
    }
}

// -- Throw macros through tryCatch (end-to-end error propagation) ------------

TEST(CPU_CkImplError_NONE, ThrowIfNullPropagatesThroughTryCatch)
{
    int* ptr    = nullptr;
    auto status = ck_impl_try_catch(
        [&]() { CK_IMPL_THROW_IF_NULL(ptr, CK_IMPL_STATUS_BAD_PARAM, "null ptr in tryCatch"); });
    EXPECT_EQ(status, CK_IMPL_STATUS_BAD_PARAM);
    EXPECT_STREQ(CkImplLastError::getLastError(), "null ptr in tryCatch");
}

TEST(CPU_CkImplError_NONE, ThrowIfFalsePropagatesThroughTryCatch)
{
    auto status = ck_impl_try_catch([]() {
        size_t idx  = 10;
        size_t size = 5;
        CK_IMPL_THROW_IF_FALSE(idx < size, CK_IMPL_STATUS_INVALID_VALUE, "Index out of range");
    });
    EXPECT_EQ(status, CK_IMPL_STATUS_INVALID_VALUE);
    EXPECT_STREQ(CkImplLastError::getLastError(), "Index out of range");
}

TEST(CPU_CkImplError_NONE, NoThrowOnValidInputsThroughTryCatch)
{
    int value   = 42;
    int* ptr    = &value;
    auto status = ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(ptr, CK_IMPL_STATUS_BAD_PARAM, "should not throw");
        CK_IMPL_THROW_IF_FALSE(true, CK_IMPL_STATUS_INVALID_VALUE, "should not throw");
        CK_IMPL_THROW_IF_TRUE(false, CK_IMPL_STATUS_INTERNAL_ERROR, "should not throw");
        // cppcheck-suppress duplicateExpression
        CK_IMPL_THROW_IF_NE(1, 1, CK_IMPL_STATUS_BAD_PARAM, "should not throw");
        CK_IMPL_THROW_IF_EQ(1, 2, CK_IMPL_STATUS_BAD_PARAM, "should not throw");
    });
    EXPECT_EQ(status, CK_IMPL_STATUS_SUCCESS);
}
