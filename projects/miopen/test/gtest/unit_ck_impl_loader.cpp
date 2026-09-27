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
#include <miopen/conv_solution.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <utility>

#include <vector>
#include <thread>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime.h>
#include "get_handle.hpp"
#include "../workspace.hpp"
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

miopen::conv::ProblemDescription MakePackedForwardProblem(miopenTensorLayout_t layout)
{
    const miopen::TensorDescriptor in_desc(miopenHalf, layout, {64, 128, 28, 28});
    const miopen::TensorDescriptor wei_desc(miopenHalf, layout, {128, 4, 3, 3});
    const miopen::ConvolutionDescriptor conv_desc({1, 1}, {1, 1}, {1, 1}, {0, 0}, 32);
    const auto out_desc = conv_desc.GetForwardOutputTensor(in_desc, wei_desc, miopenHalf);
    return miopen::conv::ProblemDescription(
        in_desc, wei_desc, out_desc, conv_desc, miopen::conv::Direction::Forward);
}

miopen::conv::ProblemDescription MakeBf16NchwPointwiseProblem(miopen::conv::Direction direction,
                                                              std::size_t spatial  = 1,
                                                              std::size_t channels = 16,
                                                              std::size_t outputs  = 256)
{
    const miopen::TensorDescriptor x_desc(
        miopenBFloat16, miopenTensorNCHW, {42, channels, spatial, spatial});
    const miopen::TensorDescriptor w_desc(
        miopenBFloat16, miopenTensorNCHW, {outputs, channels, 1, 1});
    const miopen::ConvolutionDescriptor conv({0, 0}, {1, 1}, {1, 1}, {0, 0}, 1);
    const auto y_desc = conv.GetForwardOutputTensor(x_desc, w_desc, miopenBFloat16);
    return direction == miopen::conv::Direction::Forward
               ? miopen::conv::ProblemDescription{x_desc, w_desc, y_desc, conv, direction}
               : miopen::conv::ProblemDescription{y_desc, w_desc, x_desc, conv, direction};
}

} // namespace

TEST(CPU_CkImplLoader_NONE, ForwardCKScratchLayoutPartition)
{
    constexpr size_t packed_bytes = 32 * 4 * 9 * 4 * sizeof(uint16_t) * 4;
    const auto nhwc               = MakePackedForwardProblem(miopenTensorNHWC);
    const auto nchw               = MakePackedForwardProblem(miopenTensorNCHW);
    EXPECT_EQ(miopen::solver::GetWorkspaceSizeLayoutTransformConv(nhwc), 0u);
    EXPECT_EQ(miopen::solver::GetWorkspaceSizeLayoutTransformConv(nhwc, packed_bytes),
              packed_bytes);

    const miopen::MultiBufferWorkspaceTraits regions(
        {miopen::solver::GetPackedSize(nchw.GetIn()),
         miopen::solver::GetPackedSize(nchw.GetWeights()),
         miopen::solver::GetPackedSize(nchw.GetOut()),
         packed_bytes});
    EXPECT_EQ(regions.GetOffset(3) % 256, 0u);
    EXPECT_GE(regions.GetOffset(3),
              regions.GetOffset(2) + miopen::solver::GetPackedSize(nchw.GetOut()));
    EXPECT_GE(regions.GetSize(), regions.GetOffset(3) + packed_bytes);
    EXPECT_EQ(miopen::solver::GetWorkspaceSizeLayoutTransformConv(nchw, packed_bytes),
              regions.GetSize());
}

// -- GPU tests (require a HIP device) -----------------------------------------

#if MIOPEN_BACKEND_HIP

static std::array<bool, 3> ReadStagingWrites(Data_t workspace,
                                             const std::array<std::size_t, 3>& region_sizes)
{
    const miopen::MultiBufferWorkspaceTraits regions(
        {region_sizes[0], region_sizes[1], region_sizes[2]});
    std::array<bool, 3> changed{};
    for(std::size_t i = 0; i < changed.size(); ++i)
    {
        std::vector<unsigned char> bytes(std::min<std::size_t>(64, region_sizes[i]));
        const auto* src = reinterpret_cast<const unsigned char*>(workspace) + regions.GetOffset(i);
        const auto status = hipMemcpy(bytes.data(), src, bytes.size(), hipMemcpyDeviceToHost);
        EXPECT_EQ(status, hipSuccess);
        changed[i] =
            status == hipSuccess && std::any_of(bytes.begin(), bytes.end(), [](unsigned char byte) {
                return byte != 0xa5;
            });
    }
    return changed;
}

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

TEST(GPU_CkImplLoader_FP16, PackedForwardSelectedWorkspace)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "Packed forward candidate is registered only on gfx1250";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "Matching CK grouped conv library not installed";

    const auto nhwc    = MakePackedForwardProblem(miopenTensorNHWC);
    const auto nchw    = MakePackedForwardProblem(miopenTensorNCHW);
    const auto kernels = loader.FillValidKernels(CKSolverType::GrpConvFwd, nhwc, miopenHalf, false);
    const auto packed  = std::find_if(kernels.begin(), kernels.end(), [](const auto& id) {
        return id.find("DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3") != std::string::npos &&
               id.size() >= 4 && id.compare(id.size() - 4, 4, ", 4>") == 0;
    });
    ASSERT_NE(packed, kernels.end());
    constexpr size_t packed_bytes = 36864;
    EXPECT_GE(loader.GetWorkspaceSize(CKSolverType::GrpConvFwd, nhwc, miopenHalf, false),
              packed_bytes);
    EXPECT_GE(loader.GetWorkspaceSize(CKSolverType::GrpConvFwd, nchw, miopenHalf, false),
              packed_bytes);
    const miopen::TensorDescriptor plain_in(miopenHalf, miopenTensorNHWC, {1, 12, 8, 8});
    const miopen::TensorDescriptor plain_weight(miopenHalf, miopenTensorNHWC, {12, 4, 3, 3});
    const miopen::ConvolutionDescriptor plain_conv({1, 1}, {1, 1}, {1, 1}, {0, 0}, 3);
    const auto plain_out = plain_conv.GetForwardOutputTensor(plain_in, plain_weight, miopenHalf);
    const miopen::conv::ProblemDescription plain(
        plain_in, plain_weight, plain_out, plain_conv, miopen::conv::Direction::Forward);
    EXPECT_EQ(loader.GetWorkspaceSize(CKSolverType::GrpConvFwd, plain, miopenHalf, false), 0u);
    const auto plain_kernels =
        loader.FillValidKernels(CKSolverType::GrpConvFwd, plain, miopenHalf, false);
    ASSERT_FALSE(plain_kernels.empty());

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    nhwc.SetupFloats(ctx);
    nhwc.SetupComputeType(ctx);
    const auto plain_solution =
        loader.GetSolution(CKSolverType::GrpConvFwd, ctx, plain, plain_kernels.front(), false);
    ASSERT_EQ(plain_solution.status, miopenStatusSuccess);
    EXPECT_EQ(plain_solution.workspace_sz, 0u);
    const auto native_solution =
        loader.GetSolution(CKSolverType::GrpConvFwd, ctx, nhwc, *packed, false);
    ASSERT_EQ(native_solution.status, miopenStatusSuccess);
    EXPECT_EQ(native_solution.workspace_sz, packed_bytes);
    const auto transformed_solution =
        loader.GetSolution(CKSolverType::GrpConvFwd, ctx, nchw, *packed, false);
    ASSERT_EQ(transformed_solution.status, miopenStatusSuccess);
    EXPECT_EQ(transformed_solution.workspace_sz,
              miopen::solver::GetWorkspaceSizeLayoutTransformConv(nchw, packed_bytes));

    const auto run_twice = [&](const auto& problem, const auto& solution) {
        ASSERT_TRUE(solution.invoker_factory);
        const auto& in_desc  = problem.GetIn();
        const auto& wei_desc = problem.GetWeights();
        const auto& out_desc = problem.GetOut();
        using Half           = half_float::half;
        const std::vector<Half> input(in_desc.GetElementSize(), Half{1.0f});
        std::vector<Half> weights(wei_desc.GetElementSize(), Half{0.0f});
        const std::vector<Half> output(out_desc.GetElementSize(), Half{0.0f});
        auto in_dev  = handle.Write(input);
        auto wei_dev = handle.Write(weights);
        auto out_dev = handle.Write(output);
        Workspace scratch(solution.workspace_sz);
        ASSERT_NE(scratch.ptr(), nullptr);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        const auto wei_strides = wei_desc.GetStrides();
        const auto out_strides = out_desc.GetStrides();
        const miopen::ConvFwdTensors tensors{
            in_desc, in_dev.get(), wei_desc, wei_dev.get(), out_desc, out_dev.get()};
        EXPECT_THROW(invoker(handle, miopen::conv::DataInvokeParams{tensors, nullptr, 0, false}),
                     miopen::Exception);
        EXPECT_THROW(invoker(handle,
                             miopen::conv::DataInvokeParams{
                                 tensors, scratch.ptr(), scratch.size() - 1, false}),
                     miopen::Exception);

        for(int weight_scale : {1, 2})
        {
            // Same GPU weight pointer on both runs: no cached packed copy may survive.
            for(std::size_t k = 0; k < 128; ++k)
                for(std::size_t c = 0; c < 4; ++c)
                    for(std::size_t y = 0; y < 3; ++y)
                        for(std::size_t x = 0; x < 3; ++x)
                            weights[k * wei_strides[0] + c * wei_strides[1] + y * wei_strides[2] +
                                    x * wei_strides[3]] =
                                Half{static_cast<float>((k / 4 + 1) * weight_scale)};
            handle.WriteTo(weights.data(), wei_dev, miopen::solver::GetPackedSize(wei_desc));
            ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
            ASSERT_EQ(hipMemset(out_dev.get(), 0x7f, out_desc.GetNumBytes()), hipSuccess);

            const miopen::conv::DataInvokeParams params{
                tensors, scratch.ptr(), scratch.size(), false};
            invoker(handle, params);
            handle.Finish();

            const auto actual = handle.Read<Half>(out_dev, out_desc.GetElementSize());
            for(std::size_t n = 0; n < 64; ++n)
                for(std::size_t k = 0; k < 128; ++k)
                    for(std::size_t h = 0; h < 28; ++h)
                        for(std::size_t w = 0; w < 28; ++w)
                        {
                            const auto index = n * out_strides[0] + k * out_strides[1] +
                                               h * out_strides[2] + w * out_strides[3];
                            const int taps =
                                (h == 0 || h == 27 ? 2 : 3) * (w == 0 || w == 27 ? 2 : 3);
                            const int expected =
                                taps * 4 * static_cast<int>(k / 4 + 1) * weight_scale;
                            if(static_cast<float>(actual[index]) != expected)
                            {
                                ADD_FAILURE() << "layout=" << in_desc.GetLayout_str()
                                              << " pass=" << weight_scale << " n=" << n
                                              << " k=" << k << " h=" << h << " w=" << w
                                              << " actual=" << static_cast<float>(actual[index])
                                              << " expected=" << expected;
                                return;
                            }
                        }
        }
    };

    run_twice(nhwc, native_solution);
    run_twice(nchw, transformed_solution);
}

TEST(GPU_CkImplLoader_BF16, NchwPointwiseBorrowAndStagedForward)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "Unit-spatial BF16 experiment targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
    {
        if(MIOPEN_CK_LIB_PATH)
            FAIL() << "Explicit CK plugin could not load for " << device_name;
        GTEST_SKIP() << "CK grouped conv library not installed";
    }

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    const auto direction = miopen::conv::Direction::Forward;
    auto unit            = MakeBf16NchwPointwiseProblem(direction);
    unit.SetupFloats(ctx);
    unit.SetupComputeType(ctx);
    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvFwd, unit, miopenBFloat16, false);
    ASSERT_FALSE(kernels.empty());
    const std::string selected =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<64, 64, 32, 32, "
        "Default, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>";
    ASSERT_NE(std::find(kernels.begin(), kernels.end(), selected), kernels.end()) << selected;

    const auto run = [&](const auto& problem, bool overlap, bool expect_staging) {
        const auto solution =
            loader.GetSolution(CKSolverType::GrpConvFwd, ctx, problem, selected, false);
        ASSERT_EQ(solution.status, miopenStatusSuccess);
        ASSERT_TRUE(solution.invoker_factory);
        const auto layout_bytes = miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem);
        EXPECT_EQ(solution.workspace_sz, layout_bytes);
        Workspace scratch(solution.workspace_sz);
        ASSERT_NE(scratch.ptr(), nullptr);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        const auto& x_desc   = problem.GetIn();
        const auto& w_desc   = problem.GetWeights();
        const auto& y_desc   = problem.GetOut();
        const auto y_strides = y_desc.GetStrides();

        for(int scale : {1, 2})
        {
            // New allocations and changed values on each call; never reuse cached pointers.
            const std::vector<bfloat16> x(x_desc.GetElementSize(),
                                          bfloat16{static_cast<float>(scale)});
            std::vector<bfloat16> w(w_desc.GetElementSize(), bfloat16{1.0f});
            for(std::size_t k = 1; k < w_desc.GetLengths()[0]; k += 2)
                for(std::size_t c = 0; c < w_desc.GetLengths()[1]; ++c)
                    w[k * w_desc.GetLengths()[1] + c] = bfloat16{-1.0f};
            auto x_dev = handle.Write(x);
            auto w_dev = handle.Write(w);
            auto y_dev =
                handle.Write(std::vector<bfloat16>(y_desc.GetElementSize(), bfloat16{0.0f}));
            // The source and destination share one allocation only in the fallback case.
            auto alias_dev   = overlap
                                   ? handle.Write(std::vector<bfloat16>(
                                       std::max(x_desc.GetElementSize(), y_desc.GetElementSize()),
                                       bfloat16{0.0f}))
                                   : decltype(y_dev){};
            const auto x_ptr = overlap ? alias_dev.get() : x_dev.get();
            const auto y_ptr = overlap ? alias_dev.get() : y_dev.get();
            if(overlap)
                handle.WriteTo(x.data(), alias_dev, x_desc.GetNumBytes());
            else
                ASSERT_EQ(hipMemset(y_ptr, 0x7f, y_desc.GetNumBytes()), hipSuccess);

            ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
            const miopen::ConvFwdTensors tensors{x_desc, x_ptr, w_desc, w_dev.get(), y_desc, y_ptr};
            if(scale == 1 && !overlap)
                EXPECT_THROW(
                    invoker(handle, miopen::conv::DataInvokeParams{tensors, nullptr, 0, false}),
                    miopen::Exception);
            invoker(handle,
                    miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
            handle.Finish();

            const auto staged = ReadStagingWrites(scratch.ptr(),
                                                  {miopen::solver::GetPackedSize(x_desc),
                                                   miopen::solver::GetPackedSize(w_desc),
                                                   miopen::solver::GetPackedSize(y_desc)});
            EXPECT_EQ(staged, (std::array<bool, 3>{expect_staging, expect_staging, expect_staging}))
                << "selected=" << selected << " overlap=" << overlap;

            const auto actual = overlap ? handle.Read<bfloat16>(alias_dev, y_desc.GetElementSize())
                                        : handle.Read<bfloat16>(y_dev, y_desc.GetElementSize());
            for(std::size_t n = 0; n < 42; ++n)
                for(std::size_t k = 0; k < y_desc.GetLengths()[1]; ++k)
                    for(std::size_t h = 0; h < y_desc.GetLengths()[2]; ++h)
                        for(std::size_t col = 0; col < y_desc.GetLengths()[3]; ++col)
                        {
                            const auto index = n * y_strides[0] + k * y_strides[1] +
                                               h * y_strides[2] + col * y_strides[3];
                            EXPECT_EQ(static_cast<float>(actual[index]),
                                      static_cast<float>((k % 2 ? -1 : 1) *
                                                         static_cast<int>(x_desc.GetLengths()[1]) *
                                                         scale));
                        }
        }
    };

    run(unit, false, false);
    run(unit, true, true);
    for(const auto [channels, outputs] :
        std::array<std::pair<std::size_t, std::size_t>, 3>{{{256, 16}, {32, 512}, {512, 32}}})
    {
        auto control = MakeBf16NchwPointwiseProblem(direction, 1, channels, outputs);
        control.SetupFloats(ctx);
        control.SetupComputeType(ctx);
        ASSERT_TRUE(loader.IsArgsSupported(
            CKSolverType::GrpConvFwd, control, selected, miopenBFloat16, false));
        run(control, false, false);
        run(control, true, true);
    }
    auto spatial = MakeBf16NchwPointwiseProblem(direction, 2);
    spatial.SetupFloats(ctx);
    spatial.SetupComputeType(ctx);
    ASSERT_TRUE(
        loader.IsArgsSupported(CKSolverType::GrpConvFwd, spatial, selected, miopenBFloat16, false));
    run(spatial, false, true);
}

TEST(GPU_CkImplLoader_BF16, NchwPointwiseBackwardBorrowAndFallback)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "Unit-spatial BF16 experiment targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
    {
        if(MIOPEN_CK_LIB_PATH)
            FAIL() << "Explicit CK plugin could not load for " << device_name;
        GTEST_SKIP() << "CK grouped conv library not installed";
    }

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    for(const bool wrw : {false, true})
    {
        const auto direction =
            wrw ? miopen::conv::Direction::BackwardWeights : miopen::conv::Direction::BackwardData;
        auto problem = MakeBf16NchwPointwiseProblem(direction);
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
        const auto solver_type = wrw ? CKSolverType::GrpConvWrw : CKSolverType::GrpConvBwd;
        const auto kernels = loader.FillValidKernels(solver_type, problem, miopenBFloat16, false);
        ASSERT_FALSE(kernels.empty());
        const auto split1 = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
            return loader.IsArgsSupported(solver_type, problem, id + "+1", miopenBFloat16, false);
        });
        ASSERT_NE(split1, kernels.end());
        const auto selected = *split1 + "+1";
        const auto solution = loader.GetSolution(solver_type, ctx, problem, selected, false);
        ASSERT_EQ(solution.status, miopenStatusSuccess);
        ASSERT_TRUE(solution.invoker_factory);
        EXPECT_GE(solution.workspace_sz,
                  miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem));
        Workspace scratch(solution.workspace_sz);
        ASSERT_NE(scratch.ptr(), nullptr);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        const auto& x_desc = problem.GetOut();
        const auto& y_desc = problem.GetIn();
        const auto& w_desc = problem.GetWeights();

        for(bool overlap : {false, true})
            for(int scale : {1, 2})
            {
                const std::vector<bfloat16> x(x_desc.GetElementSize(),
                                              bfloat16{static_cast<float>(scale)});
                const std::vector<bfloat16> dy(y_desc.GetElementSize(), bfloat16{1.0f});
                const std::vector<bfloat16> w(w_desc.GetElementSize(),
                                              bfloat16{static_cast<float>(scale)});
                auto x_dev            = handle.Write(x);
                auto y_dev            = handle.Write(dy);
                auto w_dev            = handle.Write(w);
                const auto alias_size = std::max(
                    {x_desc.GetElementSize(), y_desc.GetElementSize(), w_desc.GetElementSize()});
                auto alias_dev =
                    overlap ? handle.Write(std::vector<bfloat16>(alias_size, bfloat16{0.0f}))
                            : decltype(x_dev){};
                if(overlap)
                {
                    if(wrw)
                        handle.WriteTo(x.data(), alias_dev, x_desc.GetNumBytes());
                    else
                        handle.WriteTo(dy.data(), alias_dev, y_desc.GetNumBytes());
                }
                else if(wrw)
                    ASSERT_EQ(hipMemset(w_dev.get(), 0x7f, w_desc.GetNumBytes()), hipSuccess);
                else
                    ASSERT_EQ(hipMemset(x_dev.get(), 0x7f, x_desc.GetNumBytes()), hipSuccess);

                ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
                if(wrw)
                {
                    const miopen::ConvWrwTensors tensors{y_desc,
                                                         y_dev.get(),
                                                         x_desc,
                                                         overlap ? alias_dev.get() : x_dev.get(),
                                                         w_desc,
                                                         overlap ? alias_dev.get() : w_dev.get()};
                    invoker(handle,
                            miopen::conv::WrWInvokeParams{
                                tensors, scratch.ptr(), scratch.size(), false});
                }
                else
                {
                    const miopen::ConvBwdTensors tensors{y_desc,
                                                         overlap ? alias_dev.get() : y_dev.get(),
                                                         w_desc,
                                                         w_dev.get(),
                                                         x_desc,
                                                         overlap ? alias_dev.get() : x_dev.get()};
                    invoker(handle,
                            miopen::conv::DataInvokeParams{
                                tensors, scratch.ptr(), scratch.size(), false});
                }
                handle.Finish();

                const std::array<std::size_t, 3> region_sizes =
                    wrw ? std::array<std::size_t, 3>{miopen::solver::GetPackedSize(x_desc),
                                                     miopen::solver::GetPackedSize(y_desc),
                                                     miopen::solver::GetPackedSize(w_desc)}
                        : std::array<std::size_t, 3>{miopen::solver::GetPackedSize(y_desc),
                                                     miopen::solver::GetPackedSize(w_desc),
                                                     miopen::solver::GetPackedSize(x_desc)};
                EXPECT_EQ(ReadStagingWrites(scratch.ptr(), region_sizes),
                          (std::array<bool, 3>{overlap, overlap, overlap}))
                    << "selected=" << selected << " wrw=" << wrw;

                const auto actual =
                    wrw ? (overlap ? handle.Read<bfloat16>(alias_dev, w_desc.GetElementSize())
                                   : handle.Read<bfloat16>(w_dev, w_desc.GetElementSize()))
                        : (overlap ? handle.Read<bfloat16>(alias_dev, x_desc.GetElementSize())
                                   : handle.Read<bfloat16>(x_dev, x_desc.GetElementSize()));
                const auto expected = wrw ? 42 * scale : 256 * scale;
                for(const auto value : actual)
                    EXPECT_EQ(static_cast<float>(value), static_cast<float>(expected))
                        << "selected=" << selected << " overlap=" << overlap;
            }
    }
}

TEST(GPU_CkImplLoader_BF16, NchwK10Vector2BackwardThroughPlugin)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "BF16 paired A-load backward-data row targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
    {
        if(MIOPEN_CK_LIB_PATH)
            FAIL() << "Explicit CK plugin could not load for " << device_name;
        GTEST_SKIP() << "CK grouped conv library not installed";
    }

    const miopen::TensorDescriptor x_desc(miopenBFloat16, miopenTensorNCHW, {2, 128, 3, 5});
    const miopen::TensorDescriptor w_desc(miopenBFloat16, miopenTensorNCHW, {10, 128, 1, 1});
    const miopen::ConvolutionDescriptor conv({0, 0}, {1, 1}, {1, 1}, {0, 0}, 1);
    const auto dy_desc = conv.GetForwardOutputTensor(x_desc, w_desc, miopenBFloat16);
    auto problem       = miopen::conv::ProblemDescription{
        dy_desc, w_desc, x_desc, conv, miopen::conv::Direction::BackwardData};
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const std::string kernel_id =
        "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128, 128, 128, 32, 8, 8, "
        "Filter1x1Stride1Pad0, 16, 16, 8, 2, 2, 4, 1, 1>";
    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvBwd, problem, miopenBFloat16, false);
    ASSERT_NE(std::find(kernels.begin(), kernels.end(), kernel_id), kernels.end());
    const auto selected = kernel_id + "+1";
    ASSERT_TRUE(
        loader.IsArgsSupported(CKSolverType::GrpConvBwd, problem, selected, miopenBFloat16, false));
    const auto solution =
        loader.GetSolution(CKSolverType::GrpConvBwd, ctx, problem, selected, false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_TRUE(solution.invoker_factory);
    EXPECT_GE(solution.workspace_sz, miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem));
    Workspace scratch(solution.workspace_sz);
    ASSERT_NE(scratch.ptr(), nullptr);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    const auto dy_strides = dy_desc.GetStrides();
    const auto w_strides  = w_desc.GetStrides();

    for(int sign : {1, -1})
    {
        std::vector<bfloat16> dy(dy_desc.GetElementSize(), bfloat16{0.0f});
        std::vector<bfloat16> w(w_desc.GetElementSize(), bfloat16{0.0f});
        for(std::size_t n = 0; n < 2; ++n)
            for(std::size_t h = 0; h < 3; ++h)
                for(std::size_t col = 0; col < 5; ++col)
                {
                    const auto spatial =
                        n * dy_strides[0] + h * dy_strides[2] + col * dy_strides[3];
                    dy[spatial + 8 * dy_strides[1]] = bfloat16{3.0f * sign};
                    dy[spatial + 9 * dy_strides[1]] = bfloat16{-2.0f * sign};
                }
        for(std::size_t c = 0; c < 128; ++c)
        {
            w[8 * w_strides[0] + c * w_strides[1]] = bfloat16{1.0f};
            w[9 * w_strides[0] + c * w_strides[1]] = bfloat16{-1.0f};
        }
        auto dy_dev = handle.Write(dy);
        auto w_dev  = handle.Write(w);
        auto dx_dev = handle.Write(std::vector<bfloat16>(x_desc.GetElementSize(), bfloat16{37.0f}));
        ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
        const miopen::ConvBwdTensors tensors{
            dy_desc, dy_dev.get(), w_desc, w_dev.get(), x_desc, dx_dev.get()};
        invoker(handle,
                miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
        handle.Finish();
        EXPECT_EQ(ReadStagingWrites(scratch.ptr(),
                                    {miopen::solver::GetPackedSize(dy_desc),
                                     miopen::solver::GetPackedSize(w_desc),
                                     miopen::solver::GetPackedSize(x_desc)}),
                  (std::array<bool, 3>{true, true, true}));
        for(const auto value : handle.Read<bfloat16>(dx_dev, x_desc.GetElementSize()))
            EXPECT_EQ(static_cast<float>(value), 5.0f * sign);
    }
}

TEST(GPU_CkImplLoader_BF16, DepthwiseWrwTailThroughPlugin)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "BF16 group-local WRW candidate is registered only on gfx1250";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
    {
        if(MIOPEN_CK_LIB_PATH)
            FAIL() << "Explicit CK plugin could not load for " << device_name;
        GTEST_SKIP() << "CK grouped conv library not installed";
    }

    constexpr std::size_t G = 450;
    const miopen::TensorDescriptor x_desc(miopenBFloat16, miopenTensorNHWC, {4, G, 8, 8});
    const miopen::TensorDescriptor wei_desc(miopenBFloat16, miopenTensorNHWC, {G, 1, 3, 3});
    const miopen::ConvolutionDescriptor conv_desc({1, 1}, {1, 1}, {1, 1}, {0, 0}, G);
    const auto dy_desc = conv_desc.GetForwardOutputTensor(x_desc, wei_desc, miopenBFloat16);
    const miopen::conv::ProblemDescription problem(
        dy_desc, wei_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardWeights);

    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvWrw, problem, miopenBFloat16, false);
    const auto selected = std::find_if(kernels.begin(), kernels.end(), [](const auto& id) {
        return id.find("DeviceGroupedConvBwdWeightDepthwiseBf16<") == 0;
    });
    ASSERT_NE(selected, kernels.end()) << "CK BF16 depthwise row missing from MIOpen WRW";
    const auto selected_id = *selected + "+1";
    ASSERT_TRUE(loader.IsArgsSupported(
        CKSolverType::GrpConvWrw, problem, selected_id, miopenBFloat16, false));

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);
    const auto solution =
        loader.GetSolution(CKSolverType::GrpConvWrw, ctx, problem, selected_id, false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_EQ(solution.workspace_sz,
              miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem, 0));
    ASSERT_TRUE(solution.invoker_factory);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    Workspace scratch(solution.workspace_sz);
    ASSERT_NE(scratch.ptr(), nullptr);

    std::vector<bfloat16> x(x_desc.GetElementSize(), bfloat16{0.0f});
    const std::vector<bfloat16> dy(dy_desc.GetElementSize(), bfloat16{1.0f});
    const auto x_strides   = x_desc.GetStrides();
    const auto wei_strides = wei_desc.GetStrides();
    std::vector<bfloat16> wei(wei_desc.GetElementSize(), bfloat16{0.0f});
    auto x_dev   = handle.Write(x);
    auto dy_dev  = handle.Write(dy);
    auto wei_dev = handle.Write(wei);
    const miopen::ConvWrwTensors tensors{
        dy_desc, dy_dev.get(), x_desc, x_dev.get(), wei_desc, wei_dev.get()};

    for(int scale : {1, 2})
    {
        for(std::size_t n = 0; n < 4; ++n)
            for(std::size_t h = 0; h < 8; ++h)
                for(std::size_t w = 0; w < 8; ++w)
                    for(std::size_t g = 0; g < G; ++g)
                        x[n * x_strides[0] + g * x_strides[1] + h * x_strides[2] +
                          w * x_strides[3]] =
                            bfloat16{static_cast<float>((g % 2 == 0 ? 1 : -1) * scale)};
        // Keep the same device pointers; the second invocation must use new input.
        handle.WriteTo(x.data(), x_dev, x_desc.GetNumBytes());
        ASSERT_EQ(hipMemset(wei_dev.get(), 0xff, wei_desc.GetNumBytes()), hipSuccess);
        ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
        invoker(handle,
                miopen::conv::WrWInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
        handle.Finish();

        const auto actual = handle.Read<bfloat16>(wei_dev, wei_desc.GetElementSize());
        ASSERT_EQ(actual.size(), wei_desc.GetElementSize());
        for(std::size_t g = 0; g < G; ++g)
            for(std::size_t y = 0; y < 3; ++y)
                for(std::size_t filter_x = 0; filter_x < 3; ++filter_x)
                {
                    const auto offset =
                        g * wei_strides[0] + y * wei_strides[2] + filter_x * wei_strides[3];
                    const int overlap_h = y == 1 ? 8 : 7;
                    const int overlap_w = filter_x == 1 ? 8 : 7;
                    // Both passes are BF16-exact: 4*{7,8}*{7,8} is integral,
                    // and doubling gives even values where BF16 spacing is two.
                    const int want = (g % 2 == 0 ? 1 : -1) * scale * 4 * overlap_h * overlap_w;
                    EXPECT_EQ(static_cast<float>(actual[offset]), static_cast<float>(want))
                        << "group=" << g << " y=" << y << " x=" << filter_x << " pass=" << scale;
                }
    }
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

TEST(GPU_CkImplLoader_BF16, NchwNativePointwiseWrwAndAliasFallback)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "Native NCHW BF16 pointwise WRW targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
    {
        if(MIOPEN_CK_LIB_PATH)
            FAIL() << "Explicit CK plugin could not load for " << device_name;
        GTEST_SKIP() << "CK grouped conv library not installed";
    }

    constexpr std::size_t N = 2, C = 24, K = 128, H = 32, W = 32;
    constexpr std::size_t Spatial = H * W;
    const miopen::TensorDescriptor x_desc(miopenBFloat16, miopenTensorNCHW, {N, C, H, W});
    const miopen::TensorDescriptor dw_desc(miopenBFloat16, miopenTensorNCHW, {K, C, 1, 1});
    const miopen::ConvolutionDescriptor conv({0, 0}, {1, 1}, {1, 1}, {0, 0}, 1);
    const auto dy_desc = conv.GetForwardOutputTensor(x_desc, dw_desc, miopenBFloat16);
    auto problem       = miopen::conv::ProblemDescription{
        dy_desc, dw_desc, x_desc, conv, miopen::conv::Direction::BackwardWeights};
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const std::string native_id = "DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>";
    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvWrw, problem, miopenBFloat16, false);
    if(std::find(kernels.begin(), kernels.end(), native_id) == kernels.end() && !MIOPEN_CK_LIB_PATH)
        GTEST_SKIP() << "Installed CK plugin does not contain " << native_id;
    ASSERT_NE(std::find(kernels.begin(), kernels.end(), native_id), kernels.end());
    const auto selected = native_id + "+1";
    EXPECT_TRUE(
        loader.IsArgsSupported(CKSolverType::GrpConvWrw, problem, selected, miopenBFloat16, false));
    EXPECT_TRUE(loader.IsArgsSupported(
        CKSolverType::GrpConvWrw, problem, native_id + "+-1", miopenBFloat16, false));
    EXPECT_FALSE(loader.IsArgsSupported(
        CKSolverType::GrpConvWrw, problem, native_id + "+2", miopenBFloat16, false));

    const auto solution =
        loader.GetSolution(CKSolverType::GrpConvWrw, ctx, problem, selected, false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, miopen::solver::GetWorkspaceSizeLayoutTransformConv(problem));
    Workspace scratch(solution.workspace_sz);
    ASSERT_NE(scratch.ptr(), nullptr);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    const std::array<std::size_t, 3> staging_sizes = {miopen::solver::GetPackedSize(x_desc),
                                                      miopen::solver::GetPackedSize(dy_desc),
                                                      miopen::solver::GetPackedSize(dw_desc)};

    std::vector<bfloat16> x(x_desc.GetElementSize());
    std::vector<bfloat16> dy(dy_desc.GetElementSize());
    for(std::size_t n = 0; n < N; ++n)
        for(std::size_t k = 0; k < K; ++k)
            for(std::size_t s = 0; s < Spatial; ++s)
                dy[(n * K + k) * Spatial + s] = bfloat16{static_cast<float>(
                    static_cast<int>((n * 3 + k * 7 + (s / W) * 5 + (s % W) * 2) % 11) - 5)};
    auto x_dev  = handle.Write(x);
    auto dy_dev = handle.Write(dy);
    auto dw_dev = handle.Write(std::vector<bfloat16>(dw_desc.GetElementSize()));
    const miopen::ConvWrwTensors tensors{
        dy_desc, dy_dev.get(), x_desc, x_dev.get(), dw_desc, dw_dev.get()};

    const auto check_weights =
        [&](const auto& device_weights, const std::vector<bfloat16>& expected, const char* path) {
            const auto actual = handle.Read<bfloat16>(device_weights, dw_desc.GetElementSize());
            for(std::size_t k = 0; k < K; ++k)
                for(std::size_t c = 0; c < C; ++c)
                {
                    const auto index = k * C + c;
                    const float want = static_cast<float>(expected[index]);
                    EXPECT_NEAR(static_cast<float>(actual[index]),
                                want,
                                std::max(1.0f, std::abs(want) * 0.01f))
                        << path << " k=" << k << " c=" << c;
                }
        };

    std::vector<bfloat16> expected(dw_desc.GetElementSize());
    for(int pass : {0, 1})
    {
        // Reuse all three device pointers; the second pass changes x, not merely output/scratch.
        for(std::size_t n = 0; n < N; ++n)
            for(std::size_t c = 0; c < C; ++c)
                for(std::size_t s = 0; s < Spatial; ++s)
                    x[(n * C + c) * Spatial + s] = bfloat16{static_cast<float>(
                        static_cast<int>((n * 7 + c * 5 + (s / W) * 3 + (s % W) * 11 + pass * 3) %
                                         7) -
                        3)};
        for(std::size_t k = 0; k < K; ++k)
            for(std::size_t c = 0; c < C; ++c)
            {
                float sum = 0;
                for(std::size_t n = 0; n < N; ++n)
                    for(std::size_t s = 0; s < Spatial; ++s)
                        sum += static_cast<float>(x[(n * C + c) * Spatial + s]) *
                               static_cast<float>(dy[(n * K + k) * Spatial + s]);
                expected[k * C + c] = bfloat16{sum};
            }

        handle.WriteTo(x.data(), x_dev, x_desc.GetNumBytes());
        ASSERT_EQ(hipMemset(dw_dev.get(), 0x7f, dw_desc.GetNumBytes()), hipSuccess);
        ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
        invoker(handle,
                miopen::conv::WrWInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
        handle.Finish();
        // The native op writes FP32 P partials into scratch, but never stages dW.
        const auto native_writes = ReadStagingWrites(scratch.ptr(), staging_sizes);
        EXPECT_TRUE(native_writes[0]) << "Missing native FP32 partials on pass " << pass;
        EXPECT_FALSE(native_writes[2]) << "Native path staged dW on pass " << pass;
        check_weights(dw_dev, expected, "native");
        EXPECT_EQ(handle.Read<bfloat16>(x_dev, x_desc.GetElementSize()), x);
        EXPECT_EQ(handle.Read<bfloat16>(dy_dev, dy_desc.GetElementSize()), dy);
    }

    // Both input/output aliases require the staged NHWGC route for the same selected native ID.
    auto x_dw_alias = handle.Write(
        std::vector<bfloat16>(std::max(x_desc.GetElementSize(), dw_desc.GetElementSize())));
    handle.WriteTo(x.data(), x_dw_alias, x_desc.GetNumBytes());
    ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
    const miopen::ConvWrwTensors x_dw_tensors{
        dy_desc, dy_dev.get(), x_desc, x_dw_alias.get(), dw_desc, x_dw_alias.get()};
    invoker(handle,
            miopen::conv::WrWInvokeParams{x_dw_tensors, scratch.ptr(), scratch.size(), false});
    handle.Finish();
    EXPECT_EQ(ReadStagingWrites(scratch.ptr(), staging_sizes),
              (std::array<bool, 3>{true, true, true}));
    check_weights(x_dw_alias, expected, "x==dw fallback");
    EXPECT_EQ(handle.Read<bfloat16>(dy_dev, dy_desc.GetElementSize()), dy);

    auto dy_dw_alias = handle.Write(
        std::vector<bfloat16>(std::max(dy_desc.GetElementSize(), dw_desc.GetElementSize())));
    handle.WriteTo(dy.data(), dy_dw_alias, dy_desc.GetNumBytes());
    ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
    const miopen::ConvWrwTensors dy_dw_tensors{
        dy_desc, dy_dw_alias.get(), x_desc, x_dev.get(), dw_desc, dy_dw_alias.get()};
    invoker(handle,
            miopen::conv::WrWInvokeParams{dy_dw_tensors, scratch.ptr(), scratch.size(), false});
    handle.Finish();
    EXPECT_EQ(ReadStagingWrites(scratch.ptr(), staging_sizes),
              (std::array<bool, 3>{true, true, true}));
    check_weights(dy_dw_alias, expected, "dy==dw fallback");
    EXPECT_EQ(handle.Read<bfloat16>(x_dev, x_desc.GetElementSize()), x);

    EXPECT_THROW(invoker(handle, miopen::conv::WrWInvokeParams{tensors, nullptr, 0, false}),
                 miopen::Exception);
    ASSERT_GT(scratch.size(), 0u);
    EXPECT_THROW(
        invoker(handle,
                miopen::conv::WrWInvokeParams{tensors, scratch.ptr(), scratch.size() - 1, false}),
        miopen::Exception);
    const miopen::ConvWrwTensors scratch_overlaps_dw{
        dy_desc, dy_dev.get(), x_desc, x_dev.get(), dw_desc, scratch.ptr()};
    EXPECT_THROW(invoker(handle,
                         miopen::conv::WrWInvokeParams{
                             scratch_overlaps_dw, scratch.ptr(), scratch.size(), false}),
                 miopen::Exception);
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
