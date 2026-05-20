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
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util_common.hpp>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

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

namespace {

// ============================================================================
// Solver Configuration for WRW 3D Grouped Convolution
// ============================================================================

// clang-format off
constexpr SolverHeuristicConfig k3DWrwSolverConfig = {
    /* solver_name                 */ "ConvHipImplicitGemm3DGroupWrwXdlops",
    /* solver_name_ktn             */ "ConvHipImplicitGemm3DGroupWrwXdlops", // No KTN for 3D
    /* spatial_dims                */ 3,
    /* uses_split_k                */ true,
    /* split_k_min                 */ 1,
    /* split_k_max                 */ 128,
    /* supports_split_k_autodeduce */ false,
    /* supports_ktn                */ false,
};
// clang-format on

} // namespace

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
    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.Reset(k3DWrwSolverConfig.uses_split_k);

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

    const bool is_deterministic = problem.GetConv().attribute.deterministic;

    // AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR))
    {
        bool mode_use_tf32 = (problem.GetInDataType() == miopenFloat) && problem.UseTF32();

        auto fill_valid_kernels = [&loader](const ProblemDescription& p, bool try_tf32) {
            return loader.FillValidKernels(
                CKSolverType::GrpConv3dWrw, p, p.GetInDataType(), try_tf32);
        };

        auto ck_val_creator = MakeCKValidatorCreator(
            loader, CKSolverType::GrpConv3dWrw, problem.GetInDataType(), mode_use_tf32);

        // Note: No KTN runner needed for 3D (supports_ktn = false)
        if(RunAIHeuristics(k3DWrwSolverConfig,
                           state,
                           ctx,
                           problem,
                           is_deterministic,
                           fill_valid_kernels,
                           nullptr,
                           ck_val_creator,
                           mode_use_tf32))
        {
            return;
        }
    }
#endif

    // Fallback to default initialization
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
            DefaultKernelFromList(ctx);
        state.SetResult(index, split_k, k3DWrwSolverConfig.uses_split_k);
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
