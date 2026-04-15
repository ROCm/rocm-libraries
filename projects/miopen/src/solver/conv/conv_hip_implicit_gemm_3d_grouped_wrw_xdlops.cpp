/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <cstdint>
#include <type_traits>

#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

bool IsArgsSupportedForMode(const CkImplLibLoader& loader,
                            const ProblemDescription& problem,
                            const std::string& kernel_id,
                            miopenDataType_t data_type,
                            bool use_tf32)
{
    return loader.IsArgsSupported(
        CKSolverType::GrpConv3dWrw, problem, kernel_id, data_type, use_tf32);
}

} // namespace

void PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::InitValidKernels(
    const ::miopen::conv::ProblemDescription& problem)
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();
    valid_kernels  = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConv3dWrw, problem, data_type, use_tf32);

    if(!valid_kernels.empty())
    {
        index     = 0;
        split_k   = 1;
        kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
    }
}

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::tuple<std::string, int>> ranked_gemm_3d_grp_wrw = {
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 256, BlkTile: 128x128x64, WaveTile: 32x32, WaveMap: 2x2, VmemReadVec: 8x8, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, BlkGemmPipelinePrefetchStages: 2>", 128),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 256, BlkTile: 128x128x64, WaveTile: 32x32, WaveMap: 2x2, VmemReadVec: 4x4, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4, BlkGemmPipelinePrefetchStages: 3>", 64),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 16, 256, 32, Default, 8, 1, 16, 8, 8, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 8>", -1),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 16, 128, 32, Default, 8, 1, 8, 4, 8, 4, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 4>", 64),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 32, 64, 32, Default, 8, 1, 2, 2, 2, 2, 2, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2, 2>", -1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 128, BlkTile: 16x64x64, WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 2x8, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 128, BlkTile: 64x16x64, WaveTile: 16x16, WaveMap: 2x1, VmemReadVec: 8x2, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 16x64x64, WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 2x8, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 128, BlkTile: 64x16x64, WaveTile: 16x16, WaveMap: 2x1, VmemReadVec: 8x2, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 16),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 16x64x64, WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 2x8, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 16),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 4x2, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default, CRR> BlkSize: 64, BlkTile: 16x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 4x4, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 16x64x64, WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 2x8, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 2>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 256, BlkTile: 128x128x64, WaveTile: 32x32, WaveMap: 2x2, VmemReadVec: 8x8, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 4x2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 3>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 4x2, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 3>", 16),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 4x2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 16x32x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 2x4, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 3>", 128),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 2x2, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 1x2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 128),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 1x2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 8),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 16x32x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 2x1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2, BlkGemmPipelinePrefetchStages: 3>", -1),
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<MNKPadding, CRR> BlkSize: 128, BlkTile: 32x16x64, WaveTile: 16x16, WaveMap: 1x1, VmemReadVec: 2x1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 64),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<256, 64, 64, 64, Default, 8, 1, 1, 2, 8, 2, 8, 1, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", -1),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<256, 64, 64, 64, Default, 8, 1, 1, 1, 8, 4, 8, 1, 1, 4, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", -1),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<256, 64, 64, 64, Default, 8, 1, 1, 4, 8, 1, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", -1),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<256, 64, 64, 64, Default, 8, 1, 1, 1, 8, 1, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", -1),
std::make_tuple("DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 1, 4, 1, 4, 1, 1, 1>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Default, 8, 1, 1, 4, 4, 4, 4, 1, 1, 4>", 8),
std::make_tuple("DeviceGroupedConvBwdWeight_Xdl_CShuffle<64, 32, 64, 4, Default, 4, 1, 2, 1, 2, 4, 4, 1, 1, 4>", 128),
std::make_tuple("DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Default, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>", 128),
std::make_tuple("DeviceGroupedConvBwdWeight_Xdl_CShuffle<64, 64, 64, 4, Default, 4, 2, 2, 1, 4, 1, 4, 1, 1, 1>", 32)
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::tuple<std::string, int>> ranked_gemm_3d_grp_wrw_navi = {
std::make_tuple("DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmMultipleD_Wmma_CShuffleV3<MNKPadding, CRR> BlkSize: 256, BlkTile: 128x64x32, WaveTile: 16x16, WaveMap: 1x4, VmemReadVec: 1x1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages: 1>", 64),
std::make_tuple("DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<128, 96, 128, 64, Default, 8, 6, 2, 6, 8, 8, 8, 1, 1, 8, 1, 1>", 8),
std::make_tuple("DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<128, 48, 64, 128, Default, 8, 3, 1, 6, 8, 8, 8, 1, 1, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<128, 64, 64, 128, Default, 8, 4, 1, 8, 8, 8, 8, 1, 1, 8, 1, 1>", 16),
std::make_tuple("DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<128, 64, 64, 128, Default, 8, 4, 1, 8, 8, 8, 8, 1, 1, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 16, 32, Default, 8, 1, 1, 1, 4, 1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", 64),
std::make_tuple("DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 16, 32, Default, 8, 1, 1, 1, 4, 1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", 1)
};
// clang-format on

void PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::DefaultKernelFromList(
    const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_3d_grp_wrw;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_3d_grp_wrw_navi;

    const auto ranked_1st_applicable = *ranked_p;

    for(const auto& kernel : ranked_1st_applicable)
    {
        const auto& kernel_str = std::get<0>(kernel);
        const auto& it         = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_str);
        if(it != valid_kernels.end())
        {
            index     = it - valid_kernels.begin();
            split_k   = 1;
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            return;
        }
    }
}

void PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::HeuristicInit(
    const miopen::ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "None";
    split_k   = 1;

    const bool is_deterministic = problem.GetConv().attribute.deterministic;

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR))
    {
        MIOPEN_LOG_I2(
            "Step 1: Attempting AI heuristics for data type: " << problem.GetInDataType());

        std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";

        bool ai_success = false;
        miopen::ai::tuning::candidate_selection::CandidateSelectionResult result;
        const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
        if(!loader.IsLoaded())
            return;
        auto run_ai_heuristics = [&](auto CKDataType, auto CKComputeType) {
            using T        = decltype(CKDataType);
            using TCompute = decltype(CKComputeType);
            constexpr bool mode_use_tf32 =
                std::is_same_v<T, float> && std::is_same_v<TCompute, TF32Tag>;
            auto fill_valid_kernels =
                [&loader](const ::miopen::conv::ProblemDescription& p) -> std::vector<std::string> {
                return loader.FillValidKernels(
                    CKSolverType::GrpConv3dWrw, p, p.GetInDataType(), mode_use_tf32);
            };

            // Validation lambda for AI-predicted kernel + split_k combinations
            auto is_kernel_split_k_valid = [&](int kernel_index, int split_k_value) -> bool {
                if(kernel_index < 0 || kernel_index >= static_cast<int>(valid_kernels.size()))
                    return false;

                std::string test_kernel_id =
                    valid_kernels[kernel_index] + "+" + std::to_string(split_k_value);
                return IsArgsSupportedForMode(
                    loader, problem, test_kernel_id, problem.GetInDataType(), mode_use_tf32);
            };
            auto ai_result =
                miopen::solver::conv::RunParameterPredictionModel<T>(ctx,
                                                                     problem,
                                                                     valid_kernels,
                                                                     index,
                                                                     split_k,
                                                                     kernel_id,
                                                                     fill_valid_kernels,
                                                                     solver_name,
                                                                     is_kernel_split_k_valid);
            if(ai_result.first && !ai_result.second.IsEmpty())
                use_tf32 = mode_use_tf32;
            return std::move(ai_result);
        };
        switch(problem.GetInDataType())
        {
        case miopenHalf:
            std::tie(ai_success, result) = run_ai_heuristics(HalfTag{}, HalfTag{});
            break;
        case miopenFloat:
            if(problem.UseTF32())
            {
                std::tie(ai_success, result) = run_ai_heuristics(float{}, TF32Tag{});
                if(!ai_success || result.IsEmpty())
                {
                    MIOPEN_LOG_I2("Step 3: AI heuristics with TF32 failed, retrying with FP32");
                    std::tie(ai_success, result) = run_ai_heuristics(float{}, float{});
                }
            }
            else
            {
                std::tie(ai_success, result) = run_ai_heuristics(float{}, float{});
            }
            break;
        case miopenBFloat16:
            std::tie(ai_success, result) = run_ai_heuristics(BFloat16Tag{}, BFloat16Tag{});
            break;
        default: break;
        }

        if(ai_success && !result.IsEmpty())
        {
            MIOPEN_LOG_I("Step 1: AI heuristics selected kernel: " << kernel_id);
            if(is_deterministic && split_k != 1)
            {
                MIOPEN_LOG_I("Deterministic mode: Overriding AI-predicted split_k="
                             << split_k << " to split_k=1");
                split_k = 1;
                if(!valid_kernels.empty())
                    kernel_id = valid_kernels[index] + "+1";
            }
            return;
        }
        else
        {
            MIOPEN_LOG_I2("Step 1: AI heuristics failed, proceeding to default initialization");
            // print results to log to help debugging
            if(!ai_success)
                MIOPEN_LOG_I2("Step 1: AI heuristics internal failure");
            else if(result.IsEmpty())
                MIOPEN_LOG_I2("Step 1: AI heuristics returned empty result");
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 1: AI heuristics skipped (disabled or dummy context)");
    }
#else
    MIOPEN_LOG_I2("Step 1: AI heuristics not available (MIOPEN_ENABLE_AI_KERNEL_TUNING disabled)");
#endif

    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        index     = 0;
        split_k   = 1;
        kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
        if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
            DefaultKernelFromList(ctx);

        MIOPEN_LOG_I("Step 2: Default initialization selected kernel: " << kernel_id
                                                                        << " at index: " << index);
    }
    else
    {
        MIOPEN_LOG_W("Step 2: Default initialization failed - no valid kernels found");
    }

    // Invariant: split_k must always be 1 in deterministic mode
    assert(!is_deterministic || split_k == 1);
}

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::SetNextValue(
    const ::miopen::conv::ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        InitValidKernels(problem);
        if(valid_kernels.empty())
        {
            return false;
        }
    }

    const bool is_deterministic = problem.GetConv().attribute.deterministic;

    // Deterministic mode: only iterate over kernels (index), split_k is always 1
    if(is_deterministic)
    {
        if(!NextLinear(0, valid_kernels.size() - 1, index))
        {
            return false; // All kernels exhausted
        }
        split_k   = 1;
        kernel_id = valid_kernels[index] + "+1";
        return true;
    }

    // General (non-deterministic) mode: iterate over both split_k and kernels
    do
    {
        bool flag = NextCKSplitkValue<1, 128>(split_k);

        if(!flag)
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }

        if(!NextLinear(0, valid_kernels.size() - 1, index))
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }
        // All split_k and index values were iterated
        return false;
    } while(false);
    return true;
}

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::IsValid(
    const ::miopen::conv::ProblemDescription& problem) const
{
    if(!IsDeterministicSplitKValid(kernel_id, problem.GetConv().attribute.deterministic))
        return false;

    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    switch(problem.GetInDataType())
    {
    case miopenHalf:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dWrw, problem, kernel_id, miopenHalf, false);
    case miopenFloat:
        if(problem.UseTF32() &&
           loader.IsArgsSupported(
               CKSolverType::GrpConv3dWrw, problem, kernel_id, miopenFloat, true))
        {
            use_tf32 = true;
            return true;
        }
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dWrw, problem, kernel_id, miopenFloat, false);
    case miopenInt8:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dWrw, problem, kernel_id, miopenInt8, false);
    case miopenBFloat16:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dWrw, problem, kernel_id, miopenBFloat16, false);
    default: return false;
    }
}

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
ConvHipImplicitGemm3DGroupWrwXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupWrwXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ::miopen::conv::ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemm3DGroupWrwXdlops::GetCKMaxWorkspaceSize(
    const ::miopen::conv::ProblemDescription& problem) const
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return 0;

    auto data_type = problem.GetInDataType();
    bool try_tf32  = (data_type == miopenFloat) && problem.UseTF32();
    auto ws        = loader.GetWorkspaceSize(CKSolverType::GrpConv3dWrw, problem, data_type, false);
    if(try_tf32)
        ws = std::max(
            ws, loader.GetWorkspaceSize(CKSolverType::GrpConv3dWrw, problem, data_type, true));
    return ws;
}

size_t ConvHipImplicitGemm3DGroupWrwXdlops::GetWorkspaceSize(
    const ExecutionContext&, const ::miopen::conv::ProblemDescription& problem) const
{
    auto ck_ws_size = GetCKMaxWorkspaceSize(problem);
    return GetWorkspaceSizeLayoutTransformConv(problem, ck_ws_size);
}

PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
ConvHipImplicitGemm3DGroupWrwXdlops::Search(const ExecutionContext& ctx,
                                            const ::miopen::conv::ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupWrwXdlops::IsApplicable(
    const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS))
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.Is3d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    bool try_tf32  = (data_type == miopenFloat) && problem.UseTF32();

    if(try_tf32 && loader.IsApplicable(CKSolverType::GrpConv3dWrw, problem, data_type, true))
        return true;

    return loader.IsApplicable(CKSolverType::GrpConv3dWrw, problem, data_type, false);
}

ConvSolution ConvHipImplicitGemm3DGroupWrwXdlops::GetSolution(
    const ExecutionContext& ctx,
    const ::miopen::conv::ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& config) const
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::GrpConv3dWrw, ctx, problem, config.kernel_id, config.UseTF32());
}

} // namespace conv
} // namespace solver
} // namespace miopen
