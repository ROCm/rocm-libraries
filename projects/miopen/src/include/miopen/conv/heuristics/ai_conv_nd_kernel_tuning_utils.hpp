/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <vector>
#include <string>
#include <miopen/config.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>

// ============================================================================
// Centralized Heuristic Configuration for CK Implicit GEMM Solvers
// ============================================================================
/**
 * @file ai_conv_nd_kernel_tuning_utils.hpp
 * @brief Centralized AI heuristics infrastructure for CK implicit GEMM solvers
 *
 * This file provides a unified framework for AI-based kernel selection across
 * all hip_implicit_gemm solvers (2D/3D, Forward/Backward/WrW). The centralized
 * approach eliminates code duplication and ensures consistent behavior.
 *
 * Key components:
 * - SolverHeuristicConfig: Declarative solver configuration (split_k rules, naming, etc.)
 * - HeuristicInitState: State management for heuristic results
 * - RunAIHeuristics: Main template function that orchestrates AI model execution
 *
 * Architecture support:
 * - gfx942/gfx950: Candidate Selection models (newer architecture)
 * - gfx90a: KTN models (legacy compatibility)
 *
 * @see RunAIHeuristics for usage examples
 */

namespace miopen {
namespace solver {
namespace conv {

/// @brief Check if a value is a positive power of two
/// @param x The value to check
/// @return true if x is a positive power of two (1, 2, 4, 8, ...), false otherwise
inline constexpr bool IsPowerOfTwo(int x) { return x > 0 && (x & (x - 1)) == 0; }

/// @brief The special value used by CK to indicate automatic split_k deduction
static constexpr int CkSplitkAutoDeduce = -1;

/**
 * @brief Configuration struct for solver-specific heuristic behavior
 *
 * This struct provides runtime configuration for the centralized heuristic
 * initialization machinery. Each solver creates a static constexpr instance
 * with its specific settings.
 *
 * ## Usage Example
 * ```cpp
 * static constexpr SolverHeuristicConfig kFwdSolverConfig = {
 *     .solver_name = "ConvHipImplicitGemmGroupFwdXdlops",
 *     .solver_name_ktn = "ConvHipIgemmGroupFwdXdlops",  // Abbreviated for gfx90a
 *     .spatial_dims = 2,
 *     .uses_split_k = false,
 *     .split_k_min = 0,
 *     .split_k_max = 0,
 *     .supports_ktn = true,
 * };
 * ```
 */
struct SolverHeuristicConfig
{
    // Solver name for Candidate Selection models (gfx942/gfx950)
    const char* solver_name = "Unknown";

    // Solver name for KTN models (gfx90a) - often abbreviated
    const char* solver_name_ktn = "Unknown";

    // Spatial dimensions: 2 for 2D convolutions, 3 for 3D
    int spatial_dims = 2;

    // Whether this solver uses split_k parameter
    // Forward: false, Backward/WrW: true
    bool uses_split_k = false;

    // Valid split_k range (only used if uses_split_k is true)
    // For most solvers: min=1, max=128 (powers of 2)
    int split_k_min = 1;
    int split_k_max = 128;

    // Whether this solver supports split_k autodeduce (CkSplitkAutoDeduce = -1)
    // Some solvers only support explicit split_k values (1, 2, 4, ..., 128),
    // while others allow CK to automatically determine the optimal split_k value
    bool supports_split_k_autodeduce = false;

    // Whether this solver supports KTN models (gfx90a)
    bool supports_ktn = false;

    // Helper: Get solver name for a specific architecture.
    // Note: gfx90a can uses the legacy KTN heuristics which use a different solver naming
    // convention
    const char* GetSolverNameForArch(const std::string& arch) const
    {
        return (arch == "gfx90a") ? solver_name_ktn : solver_name;
    }

    /// @brief Validate a split_k value against this solver's configuration
    /// @param split_k_value The split_k value to validate
    /// @return true if the split_k value is valid for this solver
    bool IsValidSplitK(int split_k_value) const
    {
        // If solver doesn't use split_k, only 0 is valid
        if(!uses_split_k)
            return split_k_value == 0;

        // Check for autodeduce special value
        if(split_k_value == CkSplitkAutoDeduce)
            return supports_split_k_autodeduce;

        // Check range constraint
        if(split_k_value < split_k_min || split_k_value > split_k_max)
            return false;

        // Power-of-2 constraint: This is a design choice (not a CK requirement).
        // CK itself does not require split_k to be a power of 2.
        // We enforce this to:
        //   1. Limit the search space during tuning
        //   2. Match historical behavior of these solvers
        // Future versions may relax this constraint.
        return IsPowerOfTwo(split_k_value);
    }
};

/**
 * @brief Configuration state for heuristic initialization results
 *
 * This struct holds the output state from AI heuristics. It's designed to be
 * passed by reference to the centralized heuristics function, which will
 * populate it with results. It is also used by the non-AI fallback path, so
 * it must be available regardless of MIOPEN_ENABLE_AI_KERNEL_TUNING.
 */
struct HeuristicInitState
{
    std::vector<std::string>& valid_kernels;
    int& index;
    int& split_k;
    std::string& kernel_id;

    HeuristicInitState(std::vector<std::string>& vk, int& idx, int& sk, std::string& kid)
        : valid_kernels(vk), index(idx), split_k(sk), kernel_id(kid)
    {
    }

    void Reset(bool uses_split_k)
    {
        index     = 0;
        kernel_id = "";
        split_k   = uses_split_k ? 1 : 0;
    }

    void SetResult(int idx, int sk, bool uses_split_k)
    {
        index   = idx;
        split_k = sk;
        if(uses_split_k && idx >= 0 && idx < static_cast<int>(valid_kernels.size()))
        {
            kernel_id = valid_kernels[idx] + "+" + std::to_string(sk);
        }
        else if(idx >= 0 && idx < static_cast<int>(valid_kernels.size()))
        {
            kernel_id = valid_kernels[idx];
        }
    }
};

} // namespace conv
} // namespace solver
} // namespace miopen

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace solver {
namespace conv {
const miopen::ExecutionContext& GetDummyCtx();

MIOPEN_INTERNALS_EXPORT std::map<std::string, float>
GetFeaturesND(const miopen::conv::ProblemDescription&, int max_cu, const std::string& arch);

MIOPEN_INTERNALS_EXPORT std::vector<std::string> GetKernelAsTokens(const std::string& kernel);
MIOPEN_INTERNALS_EXPORT std::vector<std::string>
ProcessExplicitXdlParams(const std::vector<std::string>& params);
MIOPEN_INTERNALS_EXPORT void FillHeuristicKernels(const std::vector<std::string>& valid_kernels,
                                                  std::vector<int>& indexes,
                                                  std::vector<std::vector<std::string>>& kernels);

MIOPEN_INTERNALS_EXPORT
std::vector<int> GenerateSplitK(int max_split_k);

// Main template implementation with validation function
template <typename DataType, typename ValidationFunc>
std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel(
    const miopen::ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem,
    std::vector<std::string>& valid_kernels,
    int& index,
    int& split_k,
    std::string& kernel_id,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
        fill_valid_kernels,
    std::string solver_name,
    ValidationFunc&& is_valid)
{
    valid_kernels = fill_valid_kernels(problem);

    // Filter kernels by type
    std::vector<int> heuristic_indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    FillHeuristicKernels(valid_kernels, heuristic_indexes, heuristic_kernels);
    // Prepare features and split_k values
    const std::string& arch = ctx.GetStream().GetDeviceName();

    // Use AI model to select best candidate
    try
    {
        std::map<std::string, float> features =
            GetFeaturesND(problem, ctx.GetStream().GetMaxComputeUnits(), arch);

        bool use_split_k = split_k != 0;
        if(split_k > 1)
        {
            MIOPEN_THROW("Invalid initial split_k value for performing AI Heuristics: " +
                         std::to_string(split_k) + ". Expected 0 (no split) or 1 (default split).");
        }

        auto result = ai::tuning::candidate_selection::ModelSelectBestCandidate(
            arch,
            solver_name,
            features,
            heuristic_kernels,
            use_split_k,
            std::forward<ValidationFunc>(is_valid));

        // Check if we have any candidates
        if(!result.IsEmpty())
        {
            // Get the best candidate (first in the sorted list)
            int best_index   = result.GetBestKernelIndex();
            int best_split_k = result.GetBestSplitK();

            if(best_index >= 0 && best_index < static_cast<int>(valid_kernels.size()))
            {
                index   = best_index;
                split_k = best_split_k;
                if(use_split_k)
                {
                    kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
                }
                else
                {
                    kernel_id = valid_kernels[index];
                }
                return {true, result};
            }
        }
        MIOPEN_LOG_I("AI prediction returned invalid kernel index, falling back");
        return {false, result};
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] AI model failed: " << ex.what());
        return {false, miopen::ai::tuning::candidate_selection::CandidateSelectionResult{}};
    }
}
// ============================================================================
// Centralized AI Heuristics Runner
// ============================================================================

// ============================================================================
// Generic Helper Template for KTN Lambda Functions
// ============================================================================

/**
 * @brief Generic implementation for running KTN heuristics across all data types
 *
 * This template function dispatches KTN model execution based on data type,
 * providing a reusable implementation for the ktn_runner lambda that appears
 * in all hip_implicit_gemm solvers supporting gfx90a.
 *
 * @tparam ConfigType Performance configuration type for the solver
 *
 * @param config Reference to the solver's performance configuration object
 * @param ctx Execution context
 * @param problem Convolution problem description
 * @return true if KTN heuristics successfully selected a kernel, false otherwise
 *
 * @note This function calls the RunParameterPredictionModelKTN method on the
 *       config object with the appropriate data type template parameter.
 *
 * ## Usage Example (in solver HeuristicInit)
 * ```cpp
 * auto ktn_runner = [&](const ExecutionContext& c, const ProblemDescription& p) {
 *     return RunKTNGeneric(*this, c, p);
 * };
 * ```
 */
template <typename ConfigType>
bool RunKTNGeneric(ConfigType& config,
                   const miopen::ExecutionContext& ctx,
                   const miopen::conv::ProblemDescription& problem)
{
    switch(problem.GetInDataType())
    {
    case miopenFloat: return config.template RunParameterPredictionModelKTN<float>(ctx, problem);
    case miopenBFloat16:
        return config.template RunParameterPredictionModelKTN<BFloat16Tag>(ctx, problem);
    case miopenHalf: return config.template RunParameterPredictionModelKTN<HalfTag>(ctx, problem);
    default: return false;
    }
}

/**
 * @brief Run AI-based heuristics to select optimal kernel configuration
 *
 * This centralized function handles both Candidate Selection (gfx942/gfx950)
 * and KTN (gfx90a) heuristics, reducing code duplication across solvers.
 *
 * @tparam FillKernelsFunc Callable that returns valid kernel IDs for a given problem
 * @tparam KTNRunnerFunc Callable that runs KTN heuristics for gfx90a (optional)
 *
 * @param solver_cfg The solver's static configuration (includes split_k validation rules)
 * @param state The heuristic state to be populated with results
 * @param ctx The execution context
 * @param problem The convolution problem description
 * @param is_deterministic Whether deterministic mode is enabled
 * @param fill_valid_kernels Callable to fill valid kernel IDs
 * @param ktn_runner Optional callable to run KTN heuristics for gfx90a
 * @param ck_validator_creator Optional callable to create a CK validator for split_k validation
 *
 * @return true if AI heuristics successfully selected a kernel, false otherwise
 *
 * ## Usage Example
 * ```cpp
 * HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
 *
 * auto fill_kernels = [&loader](const ProblemDescription& p, bool try_tf32) {
 *     return loader.FillValidKernels(CKSolverType::GrpConvFwd, p, p.GetInDataType(), try_tf32);
 * };
 *
 * // Optional: Create CK validator for split_k support check
 * auto ck_val_creator = MakeCKValidatorCreator(
 *     loader, CKSolverType::GrpConvFwd, problem.GetInDataType(), mode_use_tf32);
 *
 * if(RunAIHeuristics(kFwdSolverConfig, state, ctx, problem,
 *                    is_deterministic, fill_kernels, ktn_runner, ck_val_creator))
 * {
 *     return; // AI heuristics succeeded
 * }
 * // Fallback to default initialization
 * ```
 */
template <typename FillKernelsFunc,
          typename KTNRunnerFunc          = std::nullptr_t,
          typename CKValidatorCreatorFunc = std::nullptr_t>
bool RunAIHeuristics(const SolverHeuristicConfig& solver_cfg,
                     HeuristicInitState& state,
                     const miopen::ExecutionContext& ctx,
                     const miopen::conv::ProblemDescription& problem,
                     bool is_deterministic,
                     FillKernelsFunc&& fill_valid_kernels,
                     KTNRunnerFunc&& ktn_runner              = nullptr,
                     CKValidatorCreatorFunc&& ck_val_creator = nullptr,
                     bool use_tf32                           = false)
{
    const std::string& arch = ctx.GetStream().GetDeviceName();

    MIOPEN_LOG_I2(solver_cfg.solver_name << ": RunAIHeuristics on " << arch
                                         << " (use_tf32=" << std::boolalpha << use_tf32 << ")");

    // Only applicable for supported architectures
    if(arch != "gfx90a" && arch != "gfx942" && arch != "gfx950")
    {
        MIOPEN_LOG_I2(solver_cfg.solver_name
                      << ": Unsupported architecture, skipping AI heuristics");
        return false;
    }

    // Candidate Selection heuristics for gfx942/gfx950
    if(arch == "gfx942" || arch == "gfx950")
    {
        MIOPEN_LOG_I2(solver_cfg.solver_name << ": Candidate Selection heuristics for " << arch);
        std::string solver_name = solver_cfg.GetSolverNameForArch(arch);

        // Fill valid kernels with TF32 fallback support
        // If use_tf32 is true, first try with TF32; if no kernels, retry without TF32
        state.valid_kernels = fill_valid_kernels(problem, use_tf32);

        if(state.valid_kernels.empty() && use_tf32)
        {
            // TF32 fallback: retry without TF32
            MIOPEN_LOG_I2(solver_cfg.solver_name
                          << ": TF32 yielded no kernels, retrying without TF32");
            state.valid_kernels = fill_valid_kernels(problem, false);
        }

        if(state.valid_kernels.empty())
        {
            MIOPEN_LOG_I(solver_cfg.solver_name << ": No valid kernels found");
            return false;
        }

        // Filter kernels by type
        std::vector<int> heuristic_indexes;
        std::vector<std::vector<std::string>> heuristic_kernels;
        FillHeuristicKernels(state.valid_kernels, heuristic_indexes, heuristic_kernels);

        // Prepare features
        std::map<std::string, float> features =
            GetFeaturesND(problem, ctx.GetStream().GetMaxComputeUnits(), arch);

        bool use_split_k = solver_cfg.uses_split_k;

        try
        {
            // Validation function: Filters candidate (kernel_index, split_k) combinations
            // - Checks kernel index bounds
            // - Enforces deterministic mode restrictions (split_k must be 1)
            // - Validates split_k using solver configuration rules
            // - Uses CK's IsSupportedBySplitK when ck_validator is provided
            auto is_valid = [&](int ki, int sk) {
                if(ki < 0 || ki >= static_cast<int>(state.valid_kernels.size()))
                    return false;
                if(is_deterministic && solver_cfg.uses_split_k && sk != 1)
                    return false;
                if(!solver_cfg.IsValidSplitK(sk))
                    return false;

                // If CK validator creator is provided, use CK's IsSupportedBySplitK
                if constexpr(!std::is_same_v<CKValidatorCreatorFunc, std::nullptr_t>)
                {
                    auto ck_validator = ck_val_creator(problem);
                    if(!ck_validator(state.valid_kernels[ki], sk))
                    {
                        MIOPEN_LOG_T("CK rejected kernel+split_k: " << state.valid_kernels[ki]
                                                                    << "+" << sk);
                        return false;
                    }
                }
                return true;
            };

            auto result = ai::tuning::candidate_selection::ModelSelectBestCandidate(
                arch, solver_name, features, heuristic_kernels, use_split_k, is_valid);

            if(!result.IsEmpty())
            {
                int best_index   = result.GetBestKernelIndex();
                int best_split_k = result.GetBestSplitK();

                if(best_index >= 0 && best_index < static_cast<int>(state.valid_kernels.size()))
                {
                    state.SetResult(best_index, best_split_k, use_split_k);
                    MIOPEN_LOG_I(solver_cfg.solver_name << ": Candidate Selection selected: "
                                                        << state.kernel_id);
                    return true;
                }
            }

            MIOPEN_LOG_I(solver_cfg.solver_name
                         << ": AI prediction returned no valid candidates, falling back");
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_I2(solver_cfg.solver_name << ": [Warning] AI model failed: " << ex.what());
        }

        return false;
    }

    // KTN heuristics for gfx90a
    if(arch == "gfx90a" && solver_cfg.supports_ktn)
    {
        MIOPEN_LOG_I2(solver_cfg.solver_name << ": KTN heuristics for gfx90a");

        // Use the provided KTN runner if available
        if constexpr(!std::is_same_v<KTNRunnerFunc, std::nullptr_t>)
        {
            bool ktn_succeeded = ktn_runner(ctx, problem);

            if(ktn_succeeded)
            {
                // Enforce split_k == 1 for deterministic mode
                if(is_deterministic && solver_cfg.uses_split_k && state.split_k != 1)
                {
                    MIOPEN_LOG_I(solver_cfg.solver_name
                                 << ": Deterministic mode: Overriding KTN-predicted split_k="
                                 << state.split_k << " to split_k=1");
                    state.split_k = 1;
                    if(!state.valid_kernels.empty())
                    {
                        state.kernel_id = state.valid_kernels[state.index] + "+1";
                    }
                }
                MIOPEN_LOG_I(solver_cfg.solver_name << ": KTN selected: " << state.kernel_id);
                return true;
            }
        }
    }

    return false;
}

} // namespace conv
} // namespace solver
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
