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

TEST(CPU_CkImplLoader_NONE, SelectedNchwOutputDefinitionIsFamilySpecific)
{
    using miopen::solver::SelectedNCHWCKOutputIsFullyDefined;
    const auto fwd = MakeBf16NchwPointwiseProblem(miopen::conv::Direction::Forward, 2);
    const auto bwd = MakeBf16NchwPointwiseProblem(miopen::conv::Direction::BackwardData, 2);
    const std::string wmma_fwd =
        "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 64, 64, Default, "
        "16, 16, 2, 2, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1, 1>";
    const std::string xdl_fwd =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<64, 64, 32, 32, "
        "Default, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>";
    const auto merged_fwd = wmma_fwd.substr(0, wmma_fwd.size() - 2) + "16>";
    const auto packed_fwd = wmma_fwd.substr(0, wmma_fwd.size() - 1) + ", GroupsPerWmma: 4>";
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(fwd, wmma_fwd, std::nullopt));
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(fwd, xdl_fwd, std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, merged_fwd, std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, packed_fwd, std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, wmma_fwd, 2));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, "Unknown" + wmma_fwd, std::nullopt));

    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(
        bwd, "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<128>", 4));
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(
        bwd, "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128>", 1));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        bwd, "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128>", 1));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        bwd, "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<128>", 0));
}

TEST(CPU_CkImplLoader_NONE, SelectedNcdhwBackwardFullClearRequiresDefaultXdlV1)
{
    using miopen::solver::SelectedNCHWCKOutputIsFullyDefined;
    using Direction = miopen::conv::Direction;
    const std::string xdl_v1 =
        "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 64, 8, 8, Default, "
        "32, 32, 2, 2, 1, 8, 1, 1>";
    const auto make_problem = [](miopenDataType_t type,
                                 Direction direction,
                                 miopenTensorLayout_t layout,
                                 std::vector<int> output_pads,
                                 float alpha,
                                 float beta) {
        const miopen::TensorDescriptor dx(type, layout, {2, 16, 5, 5, 5});
        const miopen::TensorDescriptor w(type, layout, {16, 16, 1, 1, 1});
        const miopen::TensorDescriptor dy(type, layout, {2, 16, 3, 3, 3});
        const miopen::ConvolutionDescriptor conv(
            {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, std::move(output_pads), 1);
        return direction == Direction::Forward
                   ? miopen::conv::ProblemDescription{dx,
                                                      w,
                                                      dy,
                                                      conv,
                                                      direction,
                                                      0,
                                                      miopen::Scalar(alpha),
                                                      miopen::Scalar(beta)}
                   : miopen::conv::ProblemDescription{dy,
                                                      w,
                                                      dx,
                                                      conv,
                                                      direction,
                                                      0,
                                                      miopen::Scalar(alpha),
                                                      miopen::Scalar(beta)};
    };
    const auto problem = make_problem(
        miopenBFloat16, Direction::BackwardData, miopenTensorNCDHW, {0, 0, 0}, 1.0f, 0.0f);
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(problem, xdl_v1, std::nullopt));
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenHalf, Direction::BackwardData, miopenTensorNCDHW, {0, 0, 0}, 1, 0),
        xdl_v1,
        std::nullopt));
    for(const auto split : {1, 2, 0, -1})
        EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(problem, xdl_v1, split));
    for(const auto direction : {Direction::Forward, Direction::BackwardWeights})
        EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
            make_problem(miopenBFloat16, direction, miopenTensorNCDHW, {0, 0, 0}, 1, 0),
            xdl_v1,
            std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenBFloat16, Direction::Forward, miopenTensorNCDHW, {0, 0, 0}, 1, 0),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 1>",
        std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenFloat, Direction::BackwardData, miopenTensorNCDHW, {0, 0, 0}, 1, 0),
        xdl_v1,
        std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenBFloat16, Direction::BackwardData, miopenTensorNDHWC, {0, 0, 0}, 1, 0),
        xdl_v1,
        std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenBFloat16, Direction::BackwardData, miopenTensorNCDHW, {0, 1, 0}, 1, 0),
        xdl_v1,
        std::nullopt));
    for(const auto scalars : {std::array<float, 2>{2, 0}, {1, 1}, {0, 1}})
        EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(make_problem(miopenBFloat16,
                                                                     Direction::BackwardData,
                                                                     miopenTensorNCDHW,
                                                                     {0, 0, 0},
                                                                     scalars[0],
                                                                     scalars[1]),
                                                        xdl_v1,
                                                        std::nullopt));
    for(const auto* unsupported :
        {"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3<64, Default>",
         "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, Default>",
         "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, Scale>",
         "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, Bilinear>",
         "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1_Unknown<64, Default>",
         "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, Default>_Unknown"})
        EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(problem, unsupported, std::nullopt))
            << unsupported;
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
        return id.find("DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<") == 0 &&
               id.find(", GroupsPerWmma: 4>") != std::string::npos;
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
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    const auto direction = miopen::conv::Direction::Forward;
    auto unit            = MakeBf16NchwPointwiseProblem(direction);
    unit.SetupFloats(ctx);
    unit.SetupComputeType(ctx);

    const auto run = [&](const auto& problem,
                         bool overlap,
                         const std::array<bool, 3>& expect_staging) {
        const auto kernels =
            loader.FillValidKernels(CKSolverType::GrpConvFwd, problem, miopenBFloat16, false);
        const auto candidate = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
            return miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, std::nullopt) &&
                   loader.IsArgsSupported(
                       CKSolverType::GrpConvFwd, problem, id, miopenBFloat16, false);
        });
        ASSERT_NE(candidate, kernels.end()) << "No supported forward candidate for " << device_name;
        const auto& selected = *candidate;
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
            EXPECT_EQ(staged, expect_staging) << "selected=" << selected << " overlap=" << overlap;

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

    run(unit, false, {false, false, false});
    run(unit, true, {true, true, true});
    for(const auto [channels, outputs] :
        std::array<std::pair<std::size_t, std::size_t>, 3>{{{256, 16}, {32, 512}, {512, 32}}})
    {
        auto control = MakeBf16NchwPointwiseProblem(direction, 1, channels, outputs);
        control.SetupFloats(ctx);
        control.SetupComputeType(ctx);
        run(control, false, {false, false, false});
        run(control, true, {true, true, true});
    }
    auto spatial = MakeBf16NchwPointwiseProblem(direction, 2);
    spatial.SetupFloats(ctx);
    spatial.SetupComputeType(ctx);
    run(spatial, false, {true, false, true});
    run(spatial, true, {true, true, true});
}

TEST(GPU_CkImplLoader_BF16, NchwForwardFullOverwriteAndAliasFallback)
{
    const auto device_name = GetCurrentDeviceName();
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    constexpr std::size_t SpatialExtent = 65;
    const miopen::TensorDescriptor x_desc(
        miopenBFloat16, miopenTensorNCHW, {2, 24, SpatialExtent, SpatialExtent});
    const miopen::TensorDescriptor w_desc(miopenBFloat16, miopenTensorNCHW, {24, 24, 3, 3});
    const miopen::ConvolutionDescriptor conv({1, 1}, {1, 1}, {1, 1}, {0, 0}, 1);
    const auto y_desc = conv.GetForwardOutputTensor(x_desc, w_desc, miopenBFloat16);
    auto problem      = miopen::conv::ProblemDescription{
        x_desc, w_desc, y_desc, conv, miopen::conv::Direction::Forward};
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvFwd, problem, miopenBFloat16, false);
    std::vector<std::string> selected_kernels;
    for(const auto* family : {"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<",
                              "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<",
                              "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<"})
    {
        const auto selected = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
            return id.find(family) == 0 &&
                   miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, std::nullopt) &&
                   loader.IsArgsSupported(
                       CKSolverType::GrpConvFwd, problem, id, miopenBFloat16, false);
        });
        if(selected != kernels.end())
            selected_kernels.push_back(*selected);
    }
    ASSERT_FALSE(selected_kernels.empty())
        << "No supported full-overwrite forward candidate for " << device_name;

    const auto run_selected = [&](const auto& selected) {
        const auto solution =
            loader.GetSolution(CKSolverType::GrpConvFwd, ctx, problem, selected, false);
        EXPECT_TRUE(
            miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, selected, std::nullopt))
            << selected;
        ASSERT_EQ(solution.status, miopenStatusSuccess);
        ASSERT_TRUE(solution.invoker_factory);

        const miopen::TensorDescriptor nhwc_x(
            miopenBFloat16, miopenTensorNHWC, {2, 24, SpatialExtent, SpatialExtent});
        const miopen::TensorDescriptor nhwc_w(miopenBFloat16, miopenTensorNHWC, {24, 24, 3, 3});
        const auto nhwc_y = conv.GetForwardOutputTensor(nhwc_x, nhwc_w, miopenBFloat16);
        auto native       = miopen::conv::ProblemDescription{
            nhwc_x, nhwc_w, nhwc_y, conv, miopen::conv::Direction::Forward};
        native.SetupFloats(ctx);
        native.SetupComputeType(ctx);
        const auto native_solution =
            loader.GetSolution(CKSolverType::GrpConvFwd, ctx, native, selected, false);
        ASSERT_EQ(native_solution.status, miopenStatusSuccess);
        EXPECT_EQ(solution.workspace_sz,
                  miopen::solver::GetWorkspaceSizeLayoutTransformConv(
                      problem, native_solution.workspace_sz));
        Workspace scratch(solution.workspace_sz);
        ASSERT_NE(scratch.ptr(), nullptr);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);

        const auto xs = x_desc.GetStrides();
        const auto ws = w_desc.GetStrides();
        const auto ys = y_desc.GetStrides();
        std::vector<bfloat16> x(x_desc.GetElementSize());
        for(std::size_t n = 0; n < 2; ++n)
            for(std::size_t c = 0; c < 24; ++c)
                for(std::size_t h = 0; h < SpatialExtent; ++h)
                    for(std::size_t col = 0; col < SpatialExtent; ++col)
                        x[n * xs[0] + c * xs[1] + h * xs[2] + col * xs[3]] =
                            bfloat16{static_cast<float>(
                                static_cast<int>((n * 3 + c * 2 + h * 3 + col) % 5) - 2)};
        std::vector<bfloat16> weights(w_desc.GetElementSize());
        auto x_dev = handle.Write(x);
        auto w_dev = handle.Write(weights);
        auto y_dev = handle.Write(std::vector<bfloat16>(y_desc.GetElementSize(), bfloat16{0.0f}));

        const auto check_output = [&](const auto& output, const auto& expected, int pass) {
            const auto actual = handle.Read<bfloat16>(output, y_desc.GetElementSize());
            for(std::size_t n = 0; n < 2; ++n)
                for(std::size_t k = 0; k < 24; ++k)
                    for(std::size_t h = 0; h < SpatialExtent; ++h)
                        for(std::size_t col = 0; col < SpatialExtent; ++col)
                        {
                            const auto index = n * ys[0] + k * ys[1] + h * ys[2] + col * ys[3];
                            if(static_cast<float>(actual[index]) !=
                               static_cast<float>(expected[index]))
                            {
                                ADD_FAILURE()
                                    << "selected=" << selected << " pass=" << pass << " n=" << n
                                    << " k=" << k << " h=" << h << " w=" << col
                                    << " actual=" << static_cast<float>(actual[index])
                                    << " expected=" << static_cast<float>(expected[index]);
                                return;
                            }
                        }
        };

        std::vector<bfloat16> expected(y_desc.GetElementSize());
        for(int pass : {0, 1})
        {
            // Use the same device pointers but different signed weights, dirty output and scratch.
            for(std::size_t k = 0; k < 24; ++k)
                for(std::size_t c = 0; c < 24; ++c)
                    for(std::size_t r = 0; r < 3; ++r)
                        for(std::size_t s = 0; s < 3; ++s)
                            weights[k * ws[0] + c * ws[1] + r * ws[2] + s * ws[3]] =
                                bfloat16{static_cast<float>(
                                    (pass == 0 ? 1 : -1) *
                                    (static_cast<int>((k * 3 + c + r * 2 + s * 3 + pass * 2) % 7) -
                                     3))};
            for(std::size_t n = 0; n < 2; ++n)
                for(std::size_t k = 0; k < 24; ++k)
                    for(std::size_t h = 0; h < SpatialExtent; ++h)
                        for(std::size_t col = 0; col < SpatialExtent; ++col)
                        {
                            int sum = 0;
                            for(std::size_t c = 0; c < 24; ++c)
                                for(int r = 0; r < 3; ++r)
                                    for(int s = 0; s < 3; ++s)
                                    {
                                        const int ih = static_cast<int>(h) + r - 1;
                                        const int iw = static_cast<int>(col) + s - 1;
                                        if(ih >= 0 && ih < static_cast<int>(SpatialExtent) &&
                                           iw >= 0 && iw < static_cast<int>(SpatialExtent))
                                        {
                                            const auto xi = n * xs[0] + c * xs[1] +
                                                            static_cast<std::size_t>(ih) * xs[2] +
                                                            static_cast<std::size_t>(iw) * xs[3];
                                            const auto wi = k * ws[0] + c * ws[1] +
                                                            static_cast<std::size_t>(r) * ws[2] +
                                                            static_cast<std::size_t>(s) * ws[3];
                                            sum +=
                                                static_cast<int>(static_cast<float>(x[xi])) *
                                                static_cast<int>(static_cast<float>(weights[wi]));
                                        }
                                    }
                            expected[n * ys[0] + k * ys[1] + h * ys[2] + col * ys[3]] =
                                bfloat16{static_cast<float>(sum)};
                        }

            handle.WriteTo(weights.data(), w_dev, w_desc.GetNumBytes());
            const std::vector<bfloat16> poisoned(y_desc.GetElementSize(),
                                                 bfloat16{pass == 0 ? 37.0f : -47.0f});
            handle.WriteTo(poisoned.data(), y_dev, y_desc.GetNumBytes());
            ASSERT_EQ(hipMemset(scratch.ptr(), pass == 0 ? 0xa5 : 0x5a, scratch.size()),
                      hipSuccess);
            const miopen::ConvFwdTensors tensors{
                x_desc, x_dev.get(), w_desc, w_dev.get(), y_desc, y_dev.get()};
            invoker(handle,
                    miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
            handle.Finish();
            check_output(y_dev, expected, pass);
            EXPECT_EQ(handle.Read<bfloat16>(x_dev, x_desc.GetElementSize()), x);
            EXPECT_EQ(handle.Read<bfloat16>(w_dev, w_desc.GetElementSize()), weights);
        }

        // Identical x/y pointers require the conservative staged fallback, unlike disjoint buffers.
        auto alias_dev = handle.Write(x);
        ASSERT_EQ(hipMemset(scratch.ptr(), 0x3c, scratch.size()), hipSuccess);
        const miopen::ConvFwdTensors alias_tensors{
            x_desc, alias_dev.get(), w_desc, w_dev.get(), y_desc, alias_dev.get()};
        invoker(
            handle,
            miopen::conv::DataInvokeParams{alias_tensors, scratch.ptr(), scratch.size(), false});
        handle.Finish();
        check_output(alias_dev, expected, 2);
        EXPECT_EQ(handle.Read<bfloat16>(w_dev, w_desc.GetElementSize()), weights);
    };

    for(const auto& selected : selected_kernels)
        run_selected(selected);
}

TEST(GPU_CkImplLoader_BF16, NchwPointwiseBackwardBorrowAndFallback)
{
    const auto device_name = GetCurrentDeviceName();
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

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

TEST(GPU_CkImplLoader_BF16, NchwBackwardHolesUseSelectedFullClear)
{
    const auto device_name = GetCurrentDeviceName();
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;

    const miopen::TensorDescriptor x_desc(miopenBFloat16, miopenTensorNCHW, {2, 16, 5, 5});
    const miopen::TensorDescriptor w_desc(miopenBFloat16, miopenTensorNCHW, {16, 16, 1, 1});
    const miopen::ConvolutionDescriptor conv({0, 0}, {2, 2}, {1, 1}, {0, 0}, 1);
    const auto dy_desc = conv.GetForwardOutputTensor(x_desc, w_desc, miopenBFloat16);
    auto problem       = miopen::conv::ProblemDescription{
        dy_desc, w_desc, x_desc, conv, miopen::conv::Direction::BackwardData};
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const auto kernels =
        loader.FillValidKernels(CKSolverType::GrpConvBwd, problem, miopenBFloat16, false);
    const auto selected = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
        return miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, 1) &&
               loader.IsArgsSupported(
                   CKSolverType::GrpConvBwd, problem, id + "+1", miopenBFloat16, false);
    });
    if(selected == kernels.end())
        GTEST_SKIP() << "No selected full-clear backward kernel supports stride-two holes";
    const auto solution =
        loader.GetSolution(CKSolverType::GrpConvBwd, ctx, problem, *selected + "+1", false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_TRUE(solution.invoker_factory);
    Workspace scratch(solution.workspace_sz);
    ASSERT_NE(scratch.ptr(), nullptr);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    const auto xs = x_desc.GetStrides();
    for(const int sign : {1, -1})
    {
        auto dy_dev = handle.Write(
            std::vector<bfloat16>(dy_desc.GetElementSize(), bfloat16{static_cast<float>(sign)}));
        auto w_dev  = handle.Write(std::vector<bfloat16>(w_desc.GetElementSize(), bfloat16{1.0f}));
        auto dx_dev = handle.Write(std::vector<bfloat16>(x_desc.GetElementSize(), bfloat16{37.0f}));
        ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
        const miopen::ConvBwdTensors tensors{
            dy_desc, dy_dev.get(), w_desc, w_dev.get(), x_desc, dx_dev.get()};
        invoker(handle,
                miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
        handle.Finish();
        const auto dx = handle.Read<bfloat16>(dx_dev, x_desc.GetElementSize());
        for(std::size_t n = 0; n < 2; ++n)
            for(std::size_t c = 0; c < 16; ++c)
                for(std::size_t h = 0; h < 5; ++h)
                    for(std::size_t w = 0; w < 5; ++w)
                        EXPECT_EQ(
                            static_cast<float>(dx[n * xs[0] + c * xs[1] + h * xs[2] + w * xs[3]]),
                            static_cast<float>((h % 2 == 0 && w % 2 == 0) ? 16 * sign : 0))
                            << "selected=" << *selected << " h=" << h << " w=" << w;
    }
}

template <typename DataType>
void CheckNcdhwBackwardSelectedFullClearAndAliasFallback(miopenDataType_t dtype)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "3D FP16/BF16 XDL-v1 backward candidates target gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded())
        << "Required CK grouped conv library not installed for " << device_name;

    const miopen::TensorDescriptor dx_desc(dtype, miopenTensorNCDHW, {2, 16, 5, 5, 5});
    const miopen::TensorDescriptor w_desc(dtype, miopenTensorNCDHW, {16, 16, 1, 1, 1});
    const miopen::ConvolutionDescriptor conv({0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1);
    const auto dy_desc = conv.GetForwardOutputTensor(dx_desc, w_desc, dtype);
    auto problem       = miopen::conv::ProblemDescription{
        dy_desc, w_desc, dx_desc, conv, miopen::conv::Direction::BackwardData};
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    const auto kernels = loader.FillValidKernels(CKSolverType::GrpConv3dBwd, problem, dtype, false);
    const auto selected = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
        return miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, std::nullopt) &&
               loader.IsArgsSupported(CKSolverType::GrpConv3dBwd, problem, id, dtype, false);
    });
    ASSERT_NE(selected, kernels.end())
        << "No supported 3D FP16/BF16 no-D XDL-v1 backward candidate";
    const auto solution =
        loader.GetSolution(CKSolverType::GrpConv3dBwd, ctx, problem, *selected, false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_TRUE(solution.invoker_factory);
    Workspace scratch(solution.workspace_sz);
    ASSERT_NE(scratch.ptr(), nullptr);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);

    const auto dx_lengths                         = dx_desc.GetLengths();
    const auto dy_lengths                         = dy_desc.GetLengths();
    const auto w_lengths                          = w_desc.GetLengths();
    const auto dxs                                = dx_desc.GetStrides();
    const auto dys                                = dy_desc.GetStrides();
    const auto ws                                 = w_desc.GetStrides();
    const std::array<std::size_t, 3> region_sizes = {miopen::solver::GetPackedSize(dy_desc),
                                                     miopen::solver::GetPackedSize(w_desc),
                                                     miopen::solver::GetPackedSize(dx_desc)};
    for(const int sign : {1, -1})
    {
        std::vector<DataType> dy(dy_desc.GetElementSize());
        std::vector<DataType> weights(w_desc.GetElementSize());
        std::vector<float> expected(dx_desc.GetElementSize(), 0.0f);
        for(std::size_t k = 0; k < w_lengths[0]; ++k)
            for(std::size_t c = 0; c < w_lengths[1]; ++c)
                weights[k * ws[0] + c * ws[1]] =
                    DataType{static_cast<float>(static_cast<int>((k + 2 * c) % 3) - 1)};
        for(std::size_t n = 0; n < dy_lengths[0]; ++n)
            for(std::size_t k = 0; k < dy_lengths[1]; ++k)
                for(std::size_t od = 0; od < dy_lengths[2]; ++od)
                    for(std::size_t oh = 0; oh < dy_lengths[3]; ++oh)
                        for(std::size_t ow = 0; ow < dy_lengths[4]; ++ow)
                        {
                            const auto dy_index =
                                n * dys[0] + k * dys[1] + od * dys[2] + oh * dys[3] + ow * dys[4];
                            dy[dy_index] = DataType{static_cast<float>(
                                sign * (1 + static_cast<int>((n + 2 * k + 3 * od + oh + ow) % 5)))};
                            for(std::size_t c = 0; c < dx_lengths[1]; ++c)
                                for(std::size_t rz = 0; rz < w_lengths[2]; ++rz)
                                    for(std::size_t ry = 0; ry < w_lengths[3]; ++ry)
                                        for(std::size_t rx = 0; rx < w_lengths[4]; ++rx)
                                        {
                                            const auto z = 2 * od + rz;
                                            const auto y = 2 * oh + ry;
                                            const auto x = 2 * ow + rx;
                                            if(z < dx_lengths[2] && y < dx_lengths[3] &&
                                               x < dx_lengths[4])
                                                expected[n * dxs[0] + c * dxs[1] + z * dxs[2] +
                                                         y * dxs[3] + x * dxs[4]] +=
                                                    static_cast<float>(dy[dy_index]) *
                                                    static_cast<float>(
                                                        weights[k * ws[0] + c * ws[1] + rz * ws[2] +
                                                                ry * ws[3] + rx * ws[4]]);
                                        }
                        }
        auto dy_dev = handle.Write(dy);
        auto w_dev  = handle.Write(weights);
        for(const bool alias : {false, true})
        {
            auto dx_dev =
                handle.Write(std::vector<DataType>(dx_desc.GetElementSize(), DataType{37.0f}));
            if(alias)
                handle.WriteTo(dy.data(), dx_dev, dy_desc.GetNumBytes());
            ASSERT_EQ(hipMemset(scratch.ptr(), 0xa5, scratch.size()), hipSuccess);
            const miopen::ConvBwdTensors tensors{dy_desc,
                                                 alias ? dx_dev.get() : dy_dev.get(),
                                                 w_desc,
                                                 w_dev.get(),
                                                 dx_desc,
                                                 dx_dev.get()};
            invoker(handle,
                    miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
            handle.Finish();
            EXPECT_EQ(ReadStagingWrites(scratch.ptr(), region_sizes),
                      (std::array<bool, 3>{true, alias, true}))
                << "selected=" << *selected << " alias=" << alias;
            const auto actual = handle.Read<DataType>(dx_dev, dx_desc.GetElementSize());
            for(std::size_t i = 0; i < actual.size(); ++i)
                EXPECT_EQ(static_cast<float>(actual[i]), static_cast<float>(DataType{expected[i]}))
                    << "selected=" << *selected << " sign=" << sign << " alias=" << alias
                    << " index=" << i;
            EXPECT_EQ(handle.Read<DataType>(w_dev, w_desc.GetElementSize()), weights);
        }
    }
}

TEST(GPU_CkImplLoader_BF16, NcdhwBackwardSelectedFullClearAndAliasFallback)
{
    CheckNcdhwBackwardSelectedFullClearAndAliasFallback<bfloat16>(miopenBFloat16);
}

TEST(GPU_CkImplLoader_FP16, NcdhwBackwardSelectedFullClearAndAliasFallback)
{
    CheckNcdhwBackwardSelectedFullClearAndAliasFallback<half_float::half>(miopenHalf);
}

TEST(GPU_CkImplLoader_BF16, DepthwiseWrwTailThroughPlugin)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name.find("gfx1250") != 0)
        GTEST_SKIP() << "BF16 group-local WRW candidate is registered only on gfx1250";

    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded())
        << "Required CK grouped conv library not installed for " << device_name;

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
    // Per-group C=1 makes weights identity even with a 3x3 filter. Total
    // activation channels are G>1, so both high-spatial sources still stage.
    const miopen::TensorDescriptor nchw_x(miopenBFloat16, miopenTensorNCHW, {4, G, 8, 8});
    const miopen::TensorDescriptor nchw_w(miopenBFloat16, miopenTensorNCHW, {G, 1, 3, 3});
    const auto nchw_dy = conv_desc.GetForwardOutputTensor(nchw_x, nchw_w, miopenBFloat16);
    auto nchw_problem  = miopen::conv::ProblemDescription{
        nchw_dy, nchw_w, nchw_x, conv_desc, miopen::conv::Direction::BackwardWeights};
    nchw_problem.SetupFloats(ctx);
    nchw_problem.SetupComputeType(ctx);
    ASSERT_TRUE(loader.IsArgsSupported(
        CKSolverType::GrpConvWrw, nchw_problem, selected_id, miopenBFloat16, false));
    const auto nchw_solution =
        loader.GetSolution(CKSolverType::GrpConvWrw, ctx, nchw_problem, selected_id, false);
    ASSERT_EQ(nchw_solution.status, miopenStatusSuccess);
    Workspace nchw_scratch(nchw_solution.workspace_sz);
    ASSERT_NE(nchw_scratch.ptr(), nullptr);
    const auto nchw_invoker =
        handle.PrepareInvoker(*nchw_solution.invoker_factory, nchw_solution.construction_params);
    const auto nx = handle.Write(std::vector<bfloat16>(nchw_x.GetElementSize(), bfloat16{2.0f}));
    const auto ny = handle.Write(std::vector<bfloat16>(nchw_dy.GetElementSize(), bfloat16{1.0f}));
    const auto nw = handle.Write(std::vector<bfloat16>(nchw_w.GetElementSize(), bfloat16{-47.0f}));
    ASSERT_EQ(hipMemset(nchw_scratch.ptr(), 0xa5, nchw_scratch.size()), hipSuccess);
    const miopen::ConvWrwTensors nchw_tensors{
        nchw_dy, ny.get(), nchw_x, nx.get(), nchw_w, nw.get()};
    nchw_invoker(handle,
                 miopen::conv::WrWInvokeParams{
                     nchw_tensors, nchw_scratch.ptr(), nchw_scratch.size(), false});
    handle.Finish();
    EXPECT_EQ(ReadStagingWrites(nchw_scratch.ptr(),
                                {miopen::solver::GetPackedSize(nchw_x),
                                 miopen::solver::GetPackedSize(nchw_dy),
                                 miopen::solver::GetPackedSize(nchw_w)}),
              (std::array<bool, 3>{true, true, false}));
    const auto nchw_weights = handle.Read<bfloat16>(nw, nchw_w.GetElementSize());
    const auto nchw_strides = nchw_w.GetStrides();
    for(std::size_t g = 0; g < G; ++g)
        for(std::size_t y = 0; y < 3; ++y)
            for(std::size_t fx = 0; fx < 3; ++fx)
            {
                const auto offset =
                    g * nchw_strides[0] + y * nchw_strides[2] + fx * nchw_strides[3];
                const auto expected = 8 * (y == 1 ? 8 : 7) * (fx == 1 ? 8 : 7);
                EXPECT_EQ(static_cast<float>(nchw_weights[offset]), static_cast<float>(expected));
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
