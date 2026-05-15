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
#include <optional>
#include <type_traits>

#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE);
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

// ============================================================================
// Solver Configuration for FWD 3D Grouped Convolution
// ============================================================================

// clang-format off
constexpr SolverHeuristicConfig k3DFwdSolverConfig = {
    /* solver_name                 */ "ConvHipImplicitGemm3DGroupFwdXdlops",
    /* solver_name_ktn             */ "ConvHipImplicitGemm3DGroupFwdXdlops", // No KTN for 3D
    /* spatial_dims                */ 3,
    /* uses_split_k                */ false,
    /* split_k_min                 */ 0,
    /* split_k_max                 */ 0,
    /* supports_split_k_autodeduce */ false,
    /* supports_ktn                */ false,
};
// clang-format on

} // namespace

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::InitValidKernels(
    const ProblemDescription& problem)
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConv3dFwd, problem, data_type, use_tf32);

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
    }
}

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_3d_grp_fwd = {
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, 1, 1, 1, 1, 8>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 64, 64, Filter1x1Pad0, 32, 32, 2, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Stride1Pad0, 32, 32, 2, 1, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, 16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, 32, 2, 1, 4, 4, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, 2, 8, 8, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, 2, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, 2, 2, 1, 2, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 1, 1, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, 32, 2, 1, 8, 8, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 128, 64, Default, 32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, 8, 8, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, 1, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 16, Default, 32, 32, 4, 2, 4, 4, 4, 1, 1>",
"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1>"
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_3d_grp_fwd_navi = {
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 8, 8, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 128, 32, Filter1x1Pad0, 16, 16, 4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, 16, 16, 2, 1, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<256, 128, 96, 64, Default, 16, 16, 2, 3, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>"
};
// clang-format on

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::DefaultKernelFromList(
    const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_3d_grp_fwd;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_3d_grp_fwd_navi;

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

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::HeuristicInit(
    const miopen::ExecutionContext& ctx, const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "None";
    split_k   = 0; // split_k is not used in this solver, but it is required by the interface

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

    // 1. IDX_OVERRIDE is preferred
    auto idx_override = env::value(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE);
    if(idx_override != 0)
    {
        MIOPEN_LOG_I2("Step 1: Attempting index override with value: " << idx_override);

        use_tf32 = false;
        if(problem.GetInDataType() == miopenHalf || problem.GetInDataType() == miopenBFloat16)
        {
            valid_kernels = loader.FillValidKernels(
                CKSolverType::GrpConv3dFwd, problem, problem.GetInDataType(), false);
        }

        if(idx_override < valid_kernels.size())
        {
            index     = idx_override;
            kernel_id = valid_kernels[index];
            MIOPEN_LOG_I("Step 1: Index override selected kernel: " << kernel_id
                                                                    << " at index: " << index);
            return;
        }
        else
        {
            MIOPEN_LOG_W("Step 1: Index override failed, index "
                         << idx_override << " out of range, proceeding to next step");
            // Continue to hard-coded heuristics
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 1: Index override not set, proceeding to next step");
    }

    // 2. Hard-coded heuristics for BF16/FP16 on gfx942 and gfx950 only
    if((problem.GetInDataType() == miopenBFloat16 || problem.GetInDataType() == miopenHalf) &&
       (ctx.GetStream().GetDeviceName() == "gfx942" || ctx.GetStream().GetDeviceName() == "gfx950"))
    {
        MIOPEN_LOG_I2("Step 2: Attempting hard-coded heuristics for "
                      << (problem.GetInDataType() == miopenBFloat16 ? "BF16" : "FP16"));

        use_tf32      = false;
        valid_kernels = loader.FillValidKernels(
            CKSolverType::GrpConv3dFwd, problem, problem.GetInDataType(), false);

        auto find_kernel = [&valid_kernels_ = std::as_const(valid_kernels)](
                               const std::size_t& expected_index,
                               const std::string& kernel_id_) -> std::optional<std::size_t> {
            if(expected_index < valid_kernels_.size() &&
               valid_kernels_[expected_index] == kernel_id_)
                return expected_index;
            auto it = std::find(valid_kernels_.begin(), valid_kernels_.end(), kernel_id_);
            if(it != valid_kernels_.end())
                return static_cast<std::size_t>(it - valid_kernels_.begin());
            MIOPEN_LOG_I2("Hard-coded heuristics did not find kernel: " << kernel_id_);
            return std::nullopt;
        };

        std::optional<std::size_t> found_index;
        if(ctx.GetStream().GetDeviceName() == "gfx942")
        {
            if(index == 0 && problem.GetGroupCount() == 1 && problem.GetAlphaBetaCase() == DEFAULT)
            {
                int K = problem.GetOutChannels();

                MIOPEN_LOG_I("3D Conv Implicit GEMM Fwd Xdlops: selecting kernel for K="
                             << K << " C=" << problem.GetInChannels() << " G="
                             << problem.GetGroupCount() << " Type=" << problem.GetInDataType());

                if((problem.GetInDataType() == miopenHalf) ||
                   (problem.GetInDataType() == miopenBFloat16))
                {
                    if((problem.GetInChannels()) <= 32)
                    {
                        if(K < 128)
                        {
                            found_index = find_kernel(
                                1,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, "
                                "Filter1x1Pad0, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>");
                        }
                        else
                        {
                            found_index = find_kernel(
                                2,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, "
                                "32, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, 1>");
                        }
                    }
                    else
                    {
                        if(K < 16)
                        {
                            found_index = find_kernel(
                                1,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, "
                                "Default, 32, 32, 2, 1, 8, 8, 1, 1, 1, 1>");
                        }
                        else if(K <= 32)
                        {
                            found_index = find_kernel(
                                2,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, "
                                "Default, "
                                "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
                                "BlkGemmPipelineVersion: v1>");
                        }
                        else if(K < 64)
                        {
                            found_index = find_kernel(
                                57,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                                "<64, 16, 16, 128, Default, 16, 16, 1, 1, 8, 8, 4, 1, 1, "
                                "BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>");
                        }
                        else if(K < 256)
                        {
                            found_index = find_kernel(
                                10,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, "
                                "64, 32, Default, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>");
                        }
                        else
                        {
                            found_index = find_kernel(
                                31,
                                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                                "<256, 128, 128, 64, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, "
                                "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>");
                        }
                    }
                }
                else if(problem.GetInDataType() == miopenFloat)
                {
                    if((problem.GetInChannels()) >= 256)
                    {
                        found_index = find_kernel(
                            2,
                            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, "
                            "Default, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: "
                            "Interwave, BlkGemmPipelineVersion: v2>");
                    }
                }
            }
        }
        else if(ctx.GetStream().GetDeviceName() == "gfx950")
        {
            if(index == 0 && ((problem.GetInDataType() == miopenHalf) ||
                              (problem.GetInDataType() == miopenBFloat16)))
            {
                if(problem.GetInDepth() >= 3 && problem.GetInWidth() >= 256)
                {
                    found_index = find_kernel(
                        11,
                        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, "
                        "32, Default, 32, 32, 4, 4, 8, 8, 8, 1, 1, "
                        "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>");
                }
            }
        }

        if(found_index.has_value())
        {
            index     = found_index.value();
            kernel_id = valid_kernels[index];
            MIOPEN_LOG_I("Step 2: Hard-coded heuristics selected kernel: "
                         << kernel_id << " at index: " << index);
            return;
        }

        MIOPEN_LOG_I2(
            "Step 2: Hard-coded heuristics did not select a kernel, proceeding to next step");
        // Continue to AI heuristics
    }
    else
    {
        MIOPEN_LOG_I2("Step 2: Hard-coded heuristics skipped (data type: "
                      << problem.GetInDataType() << ", device: " << ctx.GetStream().GetDeviceName()
                      << ")");
    }

    // 3. AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR))
    {
        HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

        bool mode_use_tf32 = (problem.GetInDataType() == miopenFloat) && problem.UseTF32();

        auto fill_valid_kernels = [&loader](const ProblemDescription& p, bool try_tf32) {
            return loader.FillValidKernels(
                CKSolverType::GrpConv3dFwd, p, p.GetInDataType(), try_tf32);
        };

        auto ck_val_creator = MakeCKValidatorCreator(
            loader, CKSolverType::GrpConv3dFwd, problem.GetInDataType(), mode_use_tf32);

        // Note: No KTN runner needed for 3D (supports_ktn = false)
        if(RunAIHeuristics(k3DFwdSolverConfig,
                           state,
                           ctx,
                           problem,
                           false,
                           fill_valid_kernels,
                           nullptr,
                           ck_val_creator,
                           mode_use_tf32))
        {
            return;
        }
    }
#endif

    // 4. Default: index remains 0, first valid_kernel will be used
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
        if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
            DefaultKernelFromList(ctx);

        MIOPEN_LOG_I("Step 4: Default initialization selected kernel: " << kernel_id
                                                                        << " at index: " << index);
    }
    else
    {
        MIOPEN_LOG_W("Step 4: Default initialization failed - no valid kernels found");
    }
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::SetNextValue(
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

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValid(
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
            CKSolverType::GrpConv3dFwd, problem, kernel_id, miopenHalf, false);
    case miopenFloat:
        if(problem.UseTF32() &&
           loader.IsArgsSupported(
               CKSolverType::GrpConv3dFwd, problem, kernel_id, miopenFloat, true))
        {
            use_tf32 = true;
            return true;
        }
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dFwd, problem, kernel_id, miopenFloat, false);
    case miopenInt8:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dFwd, problem, kernel_id, miopenInt8, false);
    case miopenBFloat16:
        use_tf32 = false;
        return loader.IsArgsSupported(
            CKSolverType::GrpConv3dFwd, problem, kernel_id, miopenBFloat16, false);
    default: return false;
    }
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemm3DGroupFwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                      const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsApplicable(const ExecutionContext& ctx,
                                                       const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
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

    if(try_tf32 && loader.IsApplicable(CKSolverType::GrpConv3dFwd, problem, data_type, true))
        return true;

    return loader.IsApplicable(CKSolverType::GrpConv3dFwd, problem, data_type, false);
}

float ConvHipImplicitGemm3DGroupFwdXdlops::GetWti(const ExecutionContext&,
                                                  const ProblemDescription& problem) const
{
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();

    if(xDesc.GetType() == miopenHalf || xDesc.GetType() == miopenBFloat16)
    {
        std::size_t in_n, in_c, w_x, w_y, w_d;
        std::tie(in_n, in_c)    = tie_pick<0, 1>()(xDesc.GetLengths());
        std::tie(w_x, w_y, w_d) = tie_pick<2, 3, 4>()(wDesc.GetLengths());
        // For cases where the filter shape is not 1x1x1 and the input channel (in_c) is greater
        // than 8, CK's implementation offers better performance.
        if((w_x == 1 && w_y == 1 && w_d == 1) == false)
        {
            if(in_c < 8 && in_n < 4)
            {
                return 0.00002; // force disable
            }
            else
            {
                return 1.0; // force enable
            }
        }
    }
    return 0.02f;
}

ConvSolution ConvHipImplicitGemm3DGroupFwdXdlops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::GrpConv3dFwd, ctx, problem, config.kernel_id, config.UseTF32());
}

} // namespace conv
} // namespace solver
} // namespace miopen
