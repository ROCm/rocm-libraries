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
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include "get_handle.hpp"
#include "../workspace.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <cmath>
#include <type_traits>
#include <iterator>

#include <vector>
#include <thread>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <miopen/conv/data_invoke_params.hpp>
#include "get_handle.hpp"
#include <algorithm>
#include <cmath>
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
    const auto fwd =
        make_problem(miopenFloat, Direction::Forward, miopenTensorNCDHW, {0, 0, 0}, 1, 0);
    const std::string fwd_v3 =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_WmmaPorted<128, 64, 16, 64, Default, "
        "16, 16, 2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>";
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(fwd, fwd_v3, std::nullopt));
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, fwd_v3, 1));
    for(const auto* suffix : {"_MergedGroups", "_DirectLoad", "_Large_Tensor"})
    {
        auto excluded = fwd_v3;
        excluded.insert(excluded.find('<'), suffix);
        EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(fwd, excluded, std::nullopt));
    }
    EXPECT_FALSE(SelectedNCHWCKOutputIsFullyDefined(
        make_problem(miopenFloat, Direction::Forward, miopenTensorNCDHW, {0, 0, 0}, 1, 1),
        fwd_v3,
        std::nullopt));
    EXPECT_TRUE(SelectedNCHWCKOutputIsFullyDefined(
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
        GTEST_SKIP() << "3D FP16/BF16/FP32 XDL-v1 backward candidates target gfx1250";
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
        << "No supported 3D FP16/BF16/FP32 no-D XDL-v1 backward candidate";
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

TEST(GPU_CkImplLoader_FP32, NcdhwBackwardSelectedFullClearAndAliasFallback)
{
    CheckNcdhwBackwardSelectedFullClearAndAliasFallback<float>(miopenFloat);
}

template <typename DataType>
void CheckNcdhwForwardOverwriteAndAliasFallback(miopenDataType_t dtype)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name != "gfx1250")
        GTEST_SKIP() << "3D forward copy elision targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded());
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);

    // Grouped, non-tile-aligned volumes exercise padding and both stride regimes.
    for(const int stride : {1, 2})
    {
        const miopen::TensorDescriptor x_desc(dtype, miopenTensorNCDHW, {2, 16, 5, 7, 9});
        const miopen::TensorDescriptor w_desc(dtype, miopenTensorNCDHW, {16, 8, 3, 3, 3});
        const miopen::ConvolutionDescriptor conv(
            {1, 1, 1}, {stride, stride, stride}, {1, 1, 1}, {0, 0, 0}, 2);
        const auto y_desc = conv.GetForwardOutputTensor(x_desc, w_desc, dtype);
        auto problem      = miopen::conv::ProblemDescription{
            x_desc, w_desc, y_desc, conv, miopen::conv::Direction::Forward};
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
        const auto kernels =
            loader.FillValidKernels(CKSolverType::GrpConv3dFwd, problem, dtype, false);
        const auto selected = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
            return miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, std::nullopt) &&
                   loader.IsArgsSupported(CKSolverType::GrpConv3dFwd, problem, id, dtype, false);
        });
        ASSERT_NE(selected, kernels.end());
        const auto solution =
            loader.GetSolution(CKSolverType::GrpConv3dFwd, ctx, problem, *selected, false);
        ASSERT_EQ(solution.status, miopenStatusSuccess);
        ASSERT_TRUE(solution.invoker_factory);
        Workspace scratch(solution.workspace_sz);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        const auto xs = x_desc.GetStrides();
        const auto ws = w_desc.GetStrides();
        const auto ys = y_desc.GetStrides();
        const auto yl = y_desc.GetLengths();
        std::vector<DataType> x(x_desc.GetElementSize());
        std::vector<DataType> weights(w_desc.GetElementSize());
        for(std::size_t i = 0; i < x.size(); ++i)
            x[i] = DataType{static_cast<float>(static_cast<int>(i % 5) - 2)};
        auto x_dev = handle.Write(x);
        auto w_dev = handle.Write(weights);
        auto y_dev = handle.Write(std::vector<DataType>(y_desc.GetElementSize()));
        for(const int sign : {1, -1})
        {
            for(std::size_t i = 0; i < weights.size(); ++i)
                weights[i] = DataType{static_cast<float>(sign * (static_cast<int>(i % 3) - 1))};
            std::vector<DataType> expected(y_desc.GetElementSize());
            for(std::size_t n = 0; n < yl[0]; ++n)
                for(std::size_t k = 0; k < yl[1]; ++k)
                    for(std::size_t od = 0; od < yl[2]; ++od)
                        for(std::size_t oh = 0; oh < yl[3]; ++oh)
                            for(std::size_t ow = 0; ow < yl[4]; ++ow)
                            {
                                int sum = 0;
                                for(std::size_t c = 0; c < 8; ++c)
                                    for(int rz = 0; rz < 3; ++rz)
                                        for(int ry = 0; ry < 3; ++ry)
                                            for(int rx = 0; rx < 3; ++rx)
                                            {
                                                const int z =
                                                    stride * static_cast<int>(od) + rz - 1;
                                                const int h =
                                                    stride * static_cast<int>(oh) + ry - 1;
                                                const int w =
                                                    stride * static_cast<int>(ow) + rx - 1;
                                                if(z < 0 || z >= 5 || h < 0 || h >= 7 || w < 0 ||
                                                   w >= 9)
                                                    continue;
                                                const auto xi =
                                                    n * xs[0] + (k / 8 * 8 + c) * xs[1] +
                                                    static_cast<std::size_t>(z) * xs[2] +
                                                    static_cast<std::size_t>(h) * xs[3] +
                                                    static_cast<std::size_t>(w) * xs[4];
                                                const auto wi = k * ws[0] + c * ws[1] + rz * ws[2] +
                                                                ry * ws[3] + rx * ws[4];
                                                sum += static_cast<int>(static_cast<float>(x[xi])) *
                                                       static_cast<int>(
                                                           static_cast<float>(weights[wi]));
                                            }
                                expected[n * ys[0] + k * ys[1] + od * ys[2] + oh * ys[3] +
                                         ow * ys[4]] = DataType{static_cast<float>(sum)};
                            }
            handle.WriteTo(weights.data(), w_dev, w_desc.GetNumBytes());
            for(const bool alias : {false, true})
            {
                handle.WriteTo(x.data(), x_dev, x_desc.GetNumBytes());
                const std::vector<DataType> poison(y_desc.GetElementSize(), DataType{37.0f});
                handle.WriteTo(poison.data(), y_dev, y_desc.GetNumBytes());
                ASSERT_EQ(hipMemset(scratch.ptr(), sign == 1 ? 0xa5 : 0x5a, scratch.size()),
                          hipSuccess);
                const miopen::ConvFwdTensors tensors{x_desc,
                                                     x_dev.get(),
                                                     w_desc,
                                                     w_dev.get(),
                                                     y_desc,
                                                     alias ? x_dev.get() : y_dev.get()};
                invoker(
                    handle,
                    miopen::conv::DataInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
                handle.Finish();
                const auto actual = handle.Read<DataType>(alias ? x_dev : y_dev, expected.size());
                EXPECT_EQ(actual, expected) << "selected=" << *selected << " stride=" << stride
                                            << " sign=" << sign << " alias=" << alias;
                EXPECT_EQ(handle.Read<DataType>(w_dev, weights.size()), weights);
                if(!alias)
                    EXPECT_EQ(handle.Read<DataType>(x_dev, x.size()), x);
            }
        }
    }
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
    if(!gfx1250 && !loader.IsLoaded())
        GTEST_SKIP() << "CK grouped conv library not installed for " << device_name;
    ASSERT_TRUE(loader.IsLoaded()) << "Required CK library not installed for " << device_name;
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

TEST(GPU_CkImplLoader_FP32, NcdhwForwardOverwriteAndAliasFallback)
{
    CheckNcdhwForwardOverwriteAndAliasFallback<float>(miopenFloat);
}

TEST(GPU_CkImplLoader_FP16, NcdhwForwardOverwriteAndAliasFallback)
{
    CheckNcdhwForwardOverwriteAndAliasFallback<half_float::half>(miopenHalf);
}

TEST(GPU_CkImplLoader_BFP16, NcdhwForwardOverwriteAndAliasFallback)
{
    CheckNcdhwForwardOverwriteAndAliasFallback<bfloat16>(miopenBFloat16);
}

template <typename DataType>
void CheckNchwWrwOverwriteAndAliasFallback(miopenDataType_t dtype)
{
    const auto device_name = GetCurrentDeviceName();
    if(device_name != "gfx1250")
        GTEST_SKIP() << "WRW copy elision targets gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device_name);
    ASSERT_TRUE(loader.IsLoaded());
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx(&handle);
    const miopen::TensorDescriptor x_desc(dtype, miopenTensorNCHW, {2, 16, 5, 7});
    const miopen::TensorDescriptor w_desc(dtype, miopenTensorNCHW, {16, 8, 3, 3});
    const miopen::ConvolutionDescriptor conv({1, 1}, {1, 1}, {1, 1}, {0, 0}, 2);
    const auto dy_desc = conv.GetForwardOutputTensor(x_desc, w_desc, dtype);
    auto problem       = miopen::conv::ProblemDescription{
        dy_desc, w_desc, x_desc, conv, miopen::conv::Direction::BackwardWeights};
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);
    const auto kernels  = loader.FillValidKernels(CKSolverType::GrpConvWrw, problem, dtype, false);
    const auto selected = std::find_if(kernels.begin(), kernels.end(), [&](const auto& id) {
        return miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, id, 1) &&
               loader.IsArgsSupported(CKSolverType::GrpConvWrw, problem, id + "+1", dtype, false);
    });
    ASSERT_NE(selected, kernels.end());
    EXPECT_FALSE(miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, *selected, -1));
    EXPECT_FALSE(miopen::solver::SelectedNCHWCKOutputIsFullyDefined(problem, *selected, 2));
    const auto solution =
        loader.GetSolution(CKSolverType::GrpConvWrw, ctx, problem, *selected + "+1", false);
    ASSERT_EQ(solution.status, miopenStatusSuccess);
    ASSERT_TRUE(solution.invoker_factory);
    Workspace scratch(solution.workspace_sz);
    const auto invoker =
        handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    const auto xs = x_desc.GetStrides();
    const auto ys = dy_desc.GetStrides();
    const auto ws = w_desc.GetStrides();
    std::vector<DataType> x(x_desc.GetElementSize());
    std::vector<DataType> dy(dy_desc.GetElementSize());
    for(std::size_t i = 0; i < x.size(); ++i)
        x[i] = DataType{static_cast<float>(static_cast<int>(i % 5) - 2)};
    auto x_dev  = handle.Write(x);
    auto dy_dev = handle.Write(dy);
    auto w_dev  = handle.Write(std::vector<DataType>(w_desc.GetElementSize()));
    auto alias_dev =
        handle.Write(std::vector<DataType>(std::max(x.size(), w_desc.GetElementSize())));
    for(const int sign : {1, -1})
    {
        for(std::size_t i = 0; i < dy.size(); ++i)
            dy[i] = DataType{static_cast<float>(sign * (static_cast<int>(i % 3) - 1))};
        std::vector<DataType> expected(w_desc.GetElementSize());
        for(std::size_t k = 0; k < 16; ++k)
            for(std::size_t c = 0; c < 8; ++c)
                for(int fy = 0; fy < 3; ++fy)
                    for(int fx = 0; fx < 3; ++fx)
                    {
                        int sum = 0;
                        for(std::size_t n = 0; n < 2; ++n)
                            for(int ho = 0; ho < 5; ++ho)
                                for(int wo = 0; wo < 7; ++wo)
                                {
                                    const int hi = ho + fy - 1;
                                    const int wi = wo + fx - 1;
                                    if(hi < 0 || hi >= 5 || wi < 0 || wi >= 7)
                                        continue;
                                    const auto xi = n * xs[0] + (k / 8 * 8 + c) * xs[1] +
                                                    hi * xs[2] + wi * xs[3];
                                    const auto yi = n * ys[0] + k * ys[1] + ho * ys[2] + wo * ys[3];
                                    sum += static_cast<int>(static_cast<float>(x[xi])) *
                                           static_cast<int>(static_cast<float>(dy[yi]));
                                }
                        expected[k * ws[0] + c * ws[1] + fy * ws[2] + fx * ws[3]] =
                            DataType{static_cast<float>(sum)};
                    }
        handle.WriteTo(dy.data(), dy_dev, dy_desc.GetNumBytes());
        for(const bool alias : {false, true})
        {
            handle.WriteTo(x.data(), x_dev, x_desc.GetNumBytes());
            const std::vector<DataType> poison(expected.size(), DataType{37.0f});
            handle.WriteTo(poison.data(), w_dev, w_desc.GetNumBytes());
            if(alias)
                handle.WriteTo(x.data(), alias_dev, x_desc.GetNumBytes());
            ASSERT_EQ(hipMemset(scratch.ptr(), sign == 1 ? 0xa5 : 0x5a, scratch.size()),
                      hipSuccess);
            const miopen::ConvWrwTensors tensors{dy_desc,
                                                 dy_dev.get(),
                                                 x_desc,
                                                 alias ? alias_dev.get() : x_dev.get(),
                                                 w_desc,
                                                 alias ? alias_dev.get() : w_dev.get()};
            invoker(handle,
                    miopen::conv::WrWInvokeParams{tensors, scratch.ptr(), scratch.size(), false});
            handle.Finish();
            EXPECT_EQ(handle.Read<DataType>(alias ? alias_dev : w_dev, expected.size()), expected)
                << "selected=" << *selected << " sign=" << sign << " alias=" << alias;
            EXPECT_EQ(handle.Read<DataType>(dy_dev, dy.size()), dy);
            if(!alias)
                EXPECT_EQ(handle.Read<DataType>(x_dev, x.size()), x);
        }
    }
}

TEST(GPU_CkImplLoader_FP16, NchwWrwOverwriteAndAliasFallback)
{
    CheckNchwWrwOverwriteAndAliasFallback<half_float::half>(miopenHalf);
}

TEST(GPU_CkImplLoader_BFP16, NchwWrwOverwriteAndAliasFallback)
{
    CheckNchwWrwOverwriteAndAliasFallback<bfloat16>(miopenBFloat16);
}

TEST(GPU_CkImplLoader_FP16, NchwNativeOutputOverwriteAndValidation)
{
    CheckNativeOutputForward<half_float::half>(miopenHalf, 2);
}

TEST(GPU_CkImplLoader_BF16, NchwNativeOutputOverwriteAndValidation)
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

TEST(GPU_CkImplLoader_BF16, NcdhwNativeOutputOverwriteAndValidation)
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

namespace {

// Exercise the installed plugin, not a mock factory. Different C/K sizes make
// an accidental dY/dX descriptor reversal observable in the backward path.
template <typename T, bool Backward>
void RunNativePointwiseConsumer(const std::string& kernel)
{
    const auto device = GetCurrentDeviceName();
    if(device.find("gfx1250") != 0)
        GTEST_SKIP() << "Native pointwise requires gfx1250";
    const auto& loader = miopen::solver::CkImplLibLoader::Get(device);
    if(!loader.IsLoaded())
        GTEST_SKIP() << "CK plugin not installed for " << device;
    auto& handle = get_handle();
    const int c = Backward ? 64 : 32;
    const int k = Backward ? 8 : 128;
    const int h = Backward ? 32 : 64;
    const int spatial = h * h;
    const auto dtype = Backward ? miopenFloat : miopenHalf;
    const miopen::TensorDescriptor xdesc(dtype, miopenTensorNCHW, {1, c, h, h});
    const miopen::TensorDescriptor wdesc(dtype, miopenTensorNCHW, {k, c, 1, 1});
    const miopen::TensorDescriptor ydesc(dtype, miopenTensorNCHW, {1, k, h, h});
    const miopen::ConvolutionDescriptor conv({0, 0}, {1, 1}, {1, 1}, {0, 0}, 1);
    const auto direction = Backward ? miopen::conv::Direction::BackwardData
                                    : miopen::conv::Direction::Forward;
    miopen::conv::ProblemDescription problem(Backward ? ydesc : xdesc,
                                              wdesc,
                                              Backward ? xdesc : ydesc,
                                              conv,
                                              direction);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);
    const auto solver = Backward ? CKSolverType::GrpConvBwd : CKSolverType::GrpConvFwd;
    const auto kernels = loader.FillValidKernels(solver, problem, dtype, false);
    ASSERT_NE(std::find(kernels.begin(), kernels.end(), kernel), kernels.end());
    const auto selected = kernel + (Backward ? "+1" : "");
    ASSERT_TRUE(loader.IsArgsSupported(solver, problem, selected, dtype, false));
    EXPECT_FALSE(loader.IsArgsSupported(solver, problem, selected, dtype, true));
    if constexpr(Backward)
    {
        for(const auto* suffix : {"+2", "+0", "+1+1"})
        {
            EXPECT_FALSE(loader.IsArgsSupported(solver, problem, kernel + suffix, dtype, false));
            EXPECT_FALSE(loader.GetSolution(solver, ctx, problem, kernel + suffix, false).Succeeded());
        }
    }
    auto solution = loader.GetSolution(solver, ctx, problem, selected, false);
    ASSERT_TRUE(solution.Succeeded());
    EXPECT_EQ(solution.workspace_sz, 0u);
    ASSERT_TRUE(solution.invoker_factory);
    const auto invoke = handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    std::vector<T> input((Backward ? k : c) * spatial);
    std::vector<T> weights(k * c);
    std::vector<T> output((Backward ? c : k) * spatial);
    auto in_dev = handle.Write(input);
    auto w_dev = handle.Write(weights);
    auto out_dev = handle.Write(output);
    const miopen::ConvDataTensors tensors{Backward ? ydesc : xdesc, in_dev.get(),
                                          wdesc, w_dev.get(),
                                          Backward ? xdesc : ydesc, out_dev.get()};
    const miopen::conv::DataInvokeParams params{tensors, nullptr, 0, false};
    for(int pass = 0; pass < 2; ++pass)
    {
        for(size_t i = 0; i < input.size(); ++i)
            input[i] = T(float(int((i * 7 + pass * 3) % 17) - 8) / 16);
        for(size_t i = 0; i < weights.size(); ++i)
            weights[i] = T(float(int((i * 3 + pass * 5) % 11) - 5) / 16);
        std::fill(output.begin(), output.end(), T(NAN));
        // Rewrite contents without changing pointers captured by the invoker.
        ASSERT_EQ(hipMemcpy(in_dev.get(), input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(w_dev.get(), weights.data(), weights.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(out_dev.get(), output.data(), output.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        invoke(handle, params);
        handle.Finish();
        handle.ReadToVec(out_dev, output);
        for(int channel = 0; channel < (Backward ? c : k); ++channel)
            for(int s = 0; s < spatial; ++s)
            {
                float reference = 0;
                for(int reduction = 0; reduction < (Backward ? k : c); ++reduction)
                {
                    const int wi = Backward ? reduction * c + channel : channel * c + reduction;
                    reference = std::fma(float(input[reduction * spatial + s]),
                                         float(weights[wi]), reference);
                }
                // Binary fractions keep this bounded fixture exactly representable.
                ASSERT_EQ(float(output[channel * spatial + s]), float(T(reference)))
                    << "pass=" << pass << " channel=" << channel << " spatial=" << s;
            }
    }
}

} // namespace

TEST(GPU_CkImplLoader_FP16, NativePointwiseForwardReusesBuffers)
{
    RunNativePointwiseConsumer<__half, false>("DeviceGroupedConvFwdNCHWPointwise<Wmma128x128x32>");
}

TEST(GPU_CkImplLoader_FP32, NativePointwiseBackwardReusesBuffers)
{
    RunNativePointwiseConsumer<float, true>("DeviceGroupedConvBwdDataNCHWPointwise<256,4>");
    RunNativePointwiseConsumer<float, true>("DeviceGroupedConvBwdDataNCHWPointwise<256,8>");
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
