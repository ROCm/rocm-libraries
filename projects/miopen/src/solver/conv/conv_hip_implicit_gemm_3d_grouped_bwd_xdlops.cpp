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

#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::InitValidKernels(
    const ProblemDescription& problem)
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConv3dBwd, problem, data_type, use_tf32);

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
    }
}

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_3d_grp_bwd = {
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 64, 16, 16, Filter1x1Stride1Pad0, 32, 32, 1, 1, 16, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 1, 1, 8, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 1, 1, 8, 2, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 16, 4, 4, Filter1x1Stride1Pad0, 32, 32, 1, 1, 4, 2, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 4, 1, 4, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 1, 8, 1, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 4, 1, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 2, 2, 1, 1, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 64, 16, 16, Default, 32, 32, 1, 1, 16, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Default, 32, 32, 1, 1, 8, 2, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Default, 16, 16, 1, 1, 8, 1, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 128, 32, 8, 8, Default, 32, 32, 1, 2, 1, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Default, 32, 32, 2, 2, 1, 1, 1, 1>"
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_3d_grp_bwd_navi = {
"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128, 128, 128, 32, 8, 8, Default, 16, 16, 8, 2, 8, 4, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 64, 64, 32, Filter1x1Stride1Pad0, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 128, 64, 64, Default, 8, 8, 2>",
"DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Default, 16, 16, 1, 4, 8, 1, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 64, 64, 32, Default, 8, 1, 1>",
"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 64, 32, 8, 8, Default, 16, 16, 4, 2, 1, 1, 1, 1>"
};
// clang-format on

void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::DefaultKernelFromList(
    const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_3d_grp_bwd;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_3d_grp_bwd_navi;

    const auto ranked_1st_applicable = *ranked_p;

    for(const auto& kernel_str : ranked_1st_applicable)
    {
        auto it = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_str);
        if(it != valid_kernels.end())
        {
            index     = it - valid_kernels.begin();
            kernel_id = valid_kernels[index];
            return;
        }
    }
}

void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::HeuristicInit(
    const miopen::ExecutionContext& ctx, const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "None";
    split_k   = 0; // split_k is not used in this solver, but it is required by the interface

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

        // 1. AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR))
    {
        MIOPEN_LOG_I2(
            "Step 1: Attempting AI heuristics for data type: " << problem.GetInDataType());

        std::string solver_name = "ConvHipImplicitGemm3DGroupBwdXdlops";

        bool ai_success = false;
        miopen::ai::tuning::candidate_selection::CandidateSelectionResult result;

        auto run_ai_heuristics = [&](auto CKDataType, auto CKComputeType) {
            using T        = decltype(CKDataType);
            using TCompute = decltype(CKComputeType);
            constexpr bool mode_use_tf32 =
                std::is_same_v<T, float> && std::is_same_v<TCompute, TF32Tag>;

            auto fill_valid_kernels =
                [&loader](const ProblemDescription& p) -> std::vector<std::string> {
                return loader.FillValidKernels(
                    CKSolverType::GrpConv3dBwd, p, p.GetInDataType(), mode_use_tf32);
            };
            // Validation lambda for AI-predicted kernel + split_k combinations
            auto is_kernel_split_k_valid = [&](int kernel_index, int split_k_value) -> bool {
                if(kernel_index < 0 || kernel_index >= static_cast<int>(valid_kernels.size()))
                    return false;

                // TODO: Add split_k validation if split_k support is implemented
                // for now, only allow split_k_value == 0
                if(split_k_value != 0)
                    return false;

                return true;
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
                    MIOPEN_LOG_I2("Step 1: AI heuristics with TF32 failed, retrying with FP32");
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
            return;
        }
        else
        {
            MIOPEN_LOG_I2("Step 1: AI heuristics failed, proceeding to default initialization");
            // Continue to default initialization
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 1: AI heuristics skipped (disabled or dummy context)");
    }
#else
    MIOPEN_LOG_I2("Step 1: AI heuristics not available (MIOPEN_ENABLE_AI_KERNEL_TUNING disabled)");
#endif

    // 2. Default: index remains 0, first valid_kernel will be used
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
        if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
            DefaultKernelFromList(ctx);

        MIOPEN_LOG_I("Step 2: Default initialization selected kernel: " << kernel_id
                                                                        << " at index: " << index);
    }
    else
    {
        MIOPEN_LOG_W("Step 2: Default initialization failed - no valid kernels found");
    }
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::SetNextValue(
    const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        // For generic search, we want all available kernels, not heuristic selection
        InitValidKernels(problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::IsValid(
    const ProblemDescription& problem) const
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    switch(problem.GetInDataType())
    {
    case miopenHalf:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dBwd, problem, kernel_id, miopenHalf, false);
    case miopenFloat:
        if(problem.UseTF32() &&
           loader.IsArgsSupported(
               CKSolverType::GrpConv3dBwd, problem, kernel_id, miopenFloat, true))
        {
            use_tf32 = true;
            return true;
        }
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dBwd, problem, kernel_id, miopenFloat, false);
    case miopenInt8:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dBwd, problem, kernel_id, miopenInt8, false);
    case miopenBFloat16:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dBwd, problem, kernel_id, miopenBFloat16, false);
    default: return false;
    }
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
ConvHipImplicitGemm3DGroupBwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupBwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemm3DGroupBwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                      const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
ConvHipImplicitGemm3DGroupBwdXdlops::Search(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupBwdXdlops::IsApplicable(const ExecutionContext& ctx,
                                                       const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS))
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsDirectionBackwardData())
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

    if(try_tf32 && loader.IsApplicable(CKSolverType::GrpConv3dBwd, problem, data_type, true))
        return true;

    return loader.IsApplicable(CKSolverType::GrpConv3dBwd, problem, data_type, false);
}

ConvSolution ConvHipImplicitGemm3DGroupBwdXdlops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& config) const
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::GrpConv3dBwd, ctx, problem, config.kernel_id, config.UseTF32());
}

} // namespace conv
} // namespace solver
} // namespace miopen
