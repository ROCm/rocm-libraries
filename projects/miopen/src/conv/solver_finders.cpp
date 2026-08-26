// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solver_finders.hpp>

#include <algorithm>
#include <numeric>

#include <miopen/conv_algo_name.hpp>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solution.hpp>
#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/utility/modified_z.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_COMPILE_ONLY)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_NAIVE_DISABLE_IF_ALT, true)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_SEARCH_CUTOFF, false)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_FIND_SKIP_PCT, 130)

namespace miopen {

namespace conv {
namespace {

class DirectSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoDirect,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& problem,
                   const ConvFindParameters& parameters) const override
    {
        return (!parameters.use_winograd_only &&
                !IsAlgorithmDisabled(miopenConvolutionAlgoDirect, problem));
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllDirectSolutions(ctx, problem, invoke_ctx)
                   : FindAllBwdWrW2DSolutions(ctx, problem, invoke_ctx);
    }
};

class ImplicitGemmSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoImplicitGEMM,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only && !env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllImplicitGemmSolutions(ctx, problem, invoke_ctx)
                   : FindImplicitGemmWrWAllSolutions(ctx, problem, invoke_ctx);
    }
};

class FftSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoFFT, problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& problem,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only &&
               problem.GetDirection() != conv::Direction::BackwardWeights &&
               !env::disabled(MIOPEN_DEBUG_CONV_FFT);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return FindAllFFTSolutions(ctx, problem, invoke_ctx);
    }
};

class GemmSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{
            ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoGEMM, problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& parameters) const override
    {
        return !parameters.use_winograd_only && !env::disabled(MIOPEN_DEBUG_CONV_GEMM);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters&,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        return FindAllGemmSolutions(ctx, problem, invoke_ctx);
    }
};

class WinogradSolverFinder : public SolversFinderMixin<ProblemDescription, ConvFindParameters>
{
protected:
    AlgorithmName GetAlgorithmName(const ProblemDescription& problem) const override
    {
        return AlgorithmName{ConvolutionAlgoToDirectionalString(miopenConvolutionAlgoWinograd,
                                                                problem.GetDirection())};
    }

    bool IsEnabled(const ExecutionContext& /*ctx*/,
                   const ProblemDescription& /*problem*/,
                   const ConvFindParameters& /*parameters*/) const override
    {
        return !env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD);
    }

    std::vector<solver::ConvSolution> FindImpl(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const AnyInvokeParams& invoke_ctx,
                                               const ConvFindParameters& parameters,
                                               const std::optional<FindOptions>&) const override
    {
        /// \todo: actually use FindOptions
        auto ctx_copy = ctx;
        if(parameters.use_winograd_only)
            ctx_copy.use_dynamic_solutions_only = true;
        return problem.GetDirection() != conv::Direction::BackwardWeights
                   ? FindAllWinogradSolutions(ctx_copy, problem, invoke_ctx)
                   : FindWinogradWrWAllSolutions(ctx_copy, problem, invoke_ctx);
    }
};

} // namespace

const std::vector<std::unique_ptr<ISolversFinder>>& GetConvSolverFinders()
{
    static const auto finders = []() {
        auto tmp = std::vector<std::unique_ptr<ISolversFinder>>{};
        tmp.emplace_back(std::make_unique<ImplicitGemmSolverFinder>());
        tmp.emplace_back(std::make_unique<GemmSolverFinder>());
        tmp.emplace_back(std::make_unique<WinogradSolverFinder>());
        tmp.emplace_back(std::make_unique<FftSolverFinder>());
        tmp.emplace_back(std::make_unique<DirectSolverFinder>());
        return tmp;
    }();

    return finders;
}

} // namespace conv

/// Decide whether the Naive convolution solver should be *skipped* -- i.e. not
/// executed in Find's mini-benchmark -- for the current solution.
///
/// Why this exists: Naive is MIOpen's universal fallback and stays *applicable* at
/// any problem size, but its kernel is un-tiled, so on a large problem a single
/// launch runs for multiple seconds. Find times every applicable solver by actually
/// *executing* it, so merely benchmarking Naive on such a shape trips the OS GPU
/// watchdog (a TDR / driver reset) even when a fast solver (CK/GEMM) also applies
/// and would ultimately win. Naive does not have to be *selected* to cause the
/// hang -- being *benchmarked* is enough. We therefore skip *running* Naive during
/// the benchmark exactly when BOTH of these hold:
///
///   * non_naive_exists   -- some non-Naive solver was found applicable for this
///                           problem. The caller computes this from the full
///                           solution list *before* workspace filtering, so a fast
///                           solver that is later workspace-filtered still counts;
///                           that is what closes the TDR leak where the alternative
///                           never gets to time itself. AND
///   * naive_exceeds_work -- the problem's total MAC work is over the Naive work
///                           limit (~16 GMAC, see ConvDirectNaiveConvExceedsWorkLimit),
///                           i.e. large enough that a single launch could TDR.
///
/// Consequences of this exact condition (the whole selection policy in one place):
///   * Naive is the *sole* applicable solver -> not skipped -> it runs, preserving
///     the universal-fallback guarantee. A huge sole-Naive shape may then still TDR
///     -- an honest "extend coverage here" signal we deliberately do not mask.
///   * Any shape under the work limit -> not skipped -> Naive competes and wins on
///     merit wherever it is fastest. This notably covers *all* real depthwise convs
///     (group == C == K => C_per_group == 1 => ~C x less work), which never approach
///     the limit and for which Naive is often the fastest option (e.g. NCHW): they
///     keep competing with no special case.
///   * Only large-and-avoidable shapes (over the limit with an alternative present)
///     are skipped -- the actual TDR case.
///
/// MIOPEN_NAIVE_DISABLE_IF_ALT (naive_disable, default on) gates the whole policy;
/// unset it to force Naive to always compete. The AlwaysEnable debug path
/// (reference/verification) is never skipped so golden references still run.
/// A solver is Naive iff its id contains "Naive" (ConvDirectNaiveConv{Fwd,Bwd,Wrw}).
/// Single source of truth for the string test used by both the per-solver skip check
/// and the FindCore "does any non-Naive solver apply" scan.
static bool IsNaiveSolverId(const std::string& solver_id)
{
    return solver_id.find("Naive") != std::string::npos;
}

static bool ShouldSkipNaiveBenchmark(bool is_naive,
                                     bool naive_disable,
                                     bool non_naive_exists,
                                     bool naive_exceeds_work)
{
    if(!is_naive || !naive_disable)
        return false;
    if(miopen::debug::AlwaysEnableConvDirectNaive)
        return false;
    return non_naive_exists && naive_exceeds_work;
}

/// Register invoker only for the best solution within algorithm.
std::vector<Solution> EvaluateInvokers(const Handle& handle,
                                       const std::vector<solver::ConvSolution>& solutions,
                                       const AlgorithmName& algorithm_name,
                                       const NetworkConfig& network_config,
                                       const AnyInvokeParams& invoke_ctx,
                                       FindCoreResult& core_result,
                                       bool force_attach_binary,
                                       bool non_naive_exists,
                                       bool naive_exceeds_work)
{
    std::vector<Solution> ret;

    const auto arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return ret;

    bool naive_disable       = env::value(MIOPEN_NAIVE_DISABLE_IF_ALT);
    bool using_search_cutoff = env::value(MIOPEN_SEARCH_CUTOFF);
    auto selected            = miopen::solver::ConvSolution{miopenStatusUnknownError};
    auto best                = std::numeric_limits<float>::max();
    auto best_invoker        = Invoker{};
    std::vector<float> samples;

    for(const auto& sol : solutions)
    {
        const bool is_naive = IsNaiveSolverId(sol.solver_id);
        if(ShouldSkipNaiveBenchmark(is_naive, naive_disable, non_naive_exists, naive_exceeds_work))
        {
            MIOPEN_LOG_I("Skipping Naive Solver: " << algorithm_name.ToString() << ":"
                                                   << sol.solver_id);
            continue;
        }
        if(naive_disable && is_naive)
        {
            // Naive is being kept in the benchmark set even though the skip policy is on.
            // Name the reason so this reads as an expected retention, not an anomaly.
            const auto* reason = miopen::debug::AlwaysEnableConvDirectNaive ? "reference-forced"
                                 : !non_naive_exists   ? "sole applicable solver"
                                 : !naive_exceeds_work ? "below work limit"
                                                       : "skip criteria not met";
            MIOPEN_LOG_I("Retaining Naive Solver (" << reason << "): " << algorithm_name.ToString()
                                                    << ":" << sol.solver_id);
        }

        if(!conv::IsEnoughWorkspace(
               "EvaluateInvokers", solver::Id{sol.solver_id}, sol.workspace_sz, &invoke_ctx))
        {
            // Providing smaller workspace may result in the selection of a slow convolution
            // algorithm, and therefore affect library performance. Moreover, sub-optimal data may
            // be cached in the user's find-db. This means that the performance drop will become
            // persistent, i.e. even providing sufficient workspace won't restore the performance.
            // To get rid of this problem, the user will need to either remove the user's find-db,
            // or repeat miopenFindConvolution*() with affected convolution configs in Normal Find
            // Mode (the latter will overwrite sub-optimal user's find-db records).
            //
            // That is why we do not write sub-optimal results into persistent find-db (on disk)
            // unless this is explicitly enabled via environment setting.
            if(!env::enabled(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE))
                core_result.is_optimal = false;
            continue;
        }

        if(!sol.invoker_factory)
            MIOPEN_THROW("Invoker is not provided by solver " + sol.solver_id);

        float skip_time = core_result.find_search_best_time;
        if(skip_time < std::numeric_limits<float>::max())
        {
            skip_time *= env::value(MIOPEN_FIND_SKIP_PCT) / 100.0f;
        }
        MIOPEN_LOG_I("Evaluating Solver: " << algorithm_name.ToString() << ":" << sol.solver_id);

        std::vector<Program> programs;
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory,
                                                   sol.construction_params,
                                                   force_attach_binary ? &programs : nullptr);

        try
        {
            // Log solution name for grouped kernel logging
            const auto solver_id_obj = solver::Id{sol.solver_id};

            if(IsLoggingKernel())
            {
                LogSolutionName(sol.solver_id, solver_id_obj.Value(), sol.workspace_sz);

                // Extract kernel name from first kernel in solution (if available)
                std::string kernel_name;
                if(!sol.construction_params.empty() &&
                   !sol.construction_params[0].kernel_name.empty())
                {
                    kernel_name = sol.construction_params[0].kernel_name;
                }
                else
                {
                    kernel_name = sol.solver_id; // Fallback to solver name
                }

                // Log performance config before timing runs. We don't have config descriptor so
                // leave it blank.
                AddPerformanceConfig(kernel_name, "");
            }
            // Run invoker max 8 times, with ~5 sec time limit.
            using elapsed_t                 = decltype(handle.GetKernelTime());
            constexpr elapsed_t TIME_MS_MAX = 5000.0;
            constexpr int N_RUNS_MAX        = 8;
            auto elapsed                    = static_cast<elapsed_t>(0);
            auto first_elapsed              = static_cast<elapsed_t>(0);
            int i                           = 0;
            samples.clear();

            while(i < N_RUNS_MAX && elapsed < TIME_MS_MAX)
            {
                invoker(handle, invoke_ctx);

                // don't include warm-up run in our samples.
                if(i > 0)
                {
                    samples.push_back(handle.GetKernelTime());
                    if(i == 1 && using_search_cutoff && samples.front() > 1.0f &&
                       samples.front() > skip_time)
                    {
                        MIOPEN_LOG_I("Skipping (Slow) Solver: "
                                     << algorithm_name.ToString() << ":" << sol.solver_id << " "
                                     << samples.front() << " > " << skip_time);
                        break;
                    }
                }
                else
                {
                    // Keep first run just in case we go over the limit, and have no samples.
                    first_elapsed = handle.GetKernelTime();
                }
                ++i;
            }

            if(samples.size() > 0)
            {
                if(IsLoggingKernel())
                {
                    // Update the performance config with the collected samples
                    AddInvokerTimes(samples);
                }
                // Remove outliers that are more than 2 positive modified z-score's away, and get
                // the mean.
                elapsed = miopen::removeHighOutliersAndGetMean(samples, 2.0f);
            }
            else
            {
                elapsed = first_elapsed;
            }

            MIOPEN_THROW_IF(elapsed <= 0, "Invalid elapsed time detected in EvaluateInvokers");

            MIOPEN_LOG_I("solution(current vs best):" << sol << ": " << elapsed
                                                      << (elapsed < best ? " < " : " >= ") << best);
            if(elapsed < best)
            {
                best         = elapsed;
                selected     = sol;
                best_invoker = invoker;
                if(best < core_result.find_search_best_time)
                    core_result.find_search_best_time = best;
            }

            auto solution = Solution{solver::Id{sol.solver_id}, elapsed, sol.workspace_sz};
            if(force_attach_binary)
                solution.SetInvoker(invoker, programs, selected.construction_params);
            else
                solution.SetInvoker(invoker, {}, {});
            ret.emplace_back(std::move(solution));
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_E(ex.what());
        }
    }

    if(!selected.Succeeded())
    {
        ret.clear();
        return ret;
    }

    handle.RegisterInvoker(best_invoker, network_config, selected.solver_id, algorithm_name);
    MIOPEN_LOG_I("Selected: " << selected << ": " << best
                              << ", workspace_sz = " << selected.workspace_sz);

    return ret;
}

FindCoreResult FindCore(const AnyInvokeParams& invoke_ctx,
                        const ExecutionContext& ctx,
                        const ProblemDescriptionBase& problem,
                        const PrimitiveFindParameters& parameters,
                        const std::vector<std::unique_ptr<ISolversFinder>>& finders,
                        const std::optional<FindOptions>& options,
                        bool force_attach_binary)
{
    auto& handle = ctx.GetStream();

    // Find
    auto solutions = std::vector<std::pair<AlgorithmName, std::vector<solver::ConvSolution>>>{};
    std::transform(
        finders.begin(), finders.end(), std::inserter(solutions, solutions.end()), [&](auto&& f) {
            return std::make_pair(f->GetAlgorithmName(problem),
                                  f->Find(ctx, problem, invoke_ctx, parameters, options));
        });

    std::size_t total = 0;

    for(auto it = solutions.begin(); it != solutions.end();)
    {
        if(it->second.empty())
        {
            it = solutions.erase(it);
            continue;
        }

        total += it->second.size();
        ++it;
    }

    // Precompile
    {
        auto all = std::vector<const miopen::solver::ConvSolution*>{};
        all.reserve(total);
        for(const auto& ss : solutions)
            std::transform(ss.second.begin(),
                           ss.second.end(),
                           std::back_inserter(all),
                           [](auto&& s) { return &s; });
        PrecompileSolutions(handle, all, force_attach_binary);
    }

    if(env::enabled((MIOPEN_DEBUG_COMPILE_ONLY)))
        MIOPEN_THROW(
            miopenStatusGpuOperationsSkipped,
            "MIOPEN_DEBUG_COMPILE_ONLY is enabled, escaping forward convolution. Search skipped.");

    // Evaluate Invokers
    AutoEnableProfiling enableProfiling{handle};
    const auto network_config = problem.MakeNetworkConfig();
    auto ret                  = FindCoreResult();
    ret.is_optimal            = true;

    ret.solutions.reserve(total);

    // Does any non-Naive solver apply, across all algorithms? Computed from the full solution
    // list *before* per-solver workspace filtering, so a fast-but-workspace-limited alternative
    // still counts and Naive is deferred rather than benchmarked (the TDR-leak fix). Also decide
    // once whether this conv's total MAC work is large enough that the un-tiled Naive kernel would
    // trip the OS GPU watchdog if benchmarked. Non-conv (fusion) problems yield nullptr and are
    // inert here.
    const bool non_naive_exists =
        std::any_of(solutions.begin(), solutions.end(), [](const auto& g) {
            return std::any_of(g.second.begin(), g.second.end(), [](const auto& s) {
                return !IsNaiveSolverId(s.solver_id);
            });
        });
    const auto* conv_problem = dynamic_cast<const conv::ProblemDescription*>(&problem);
    const bool naive_exceeds_work =
        (conv_problem != nullptr) &&
        solver::conv::ConvDirectNaiveConvExceedsWorkLimit(*conv_problem);

    for(const auto& ss : solutions)
    {
        auto evaluated = EvaluateInvokers(handle,
                                          ss.second,
                                          ss.first,
                                          network_config,
                                          invoke_ctx,
                                          ret,
                                          force_attach_binary,
                                          non_naive_exists,
                                          naive_exceeds_work);

        ret.solutions.insert(ret.solutions.end(),
                             std::make_move_iterator(evaluated.begin()),
                             std::make_move_iterator(evaluated.end()));
    }

    // Universal-fallback guarantee. The work gate skips *benchmarking* Naive when a
    // non-Naive solver is applicable, but applicability is not success: every non-Naive
    // candidate can still be rejected at evaluation (e.g. insufficient workspace). If
    // that leaves no solution at all, we would otherwise fail the convolution outright
    // even though the un-tiled Naive kernel could have served it. Re-evaluate with the
    // skip disabled so Naive remains the true universal fallback. This only triggers in
    // the rare empty-result case, and the already-rejected non-Naive candidates are
    // re-rejected cheaply (workspace-filtered before any kernel launch).
    if(ret.solutions.empty() && non_naive_exists && naive_exceeds_work)
    {
        MIOPEN_LOG_I("No solver survived evaluation; re-running with the Naive benchmark "
                     "re-enabled as a last-resort fallback.");
        for(const auto& ss : solutions)
        {
            auto evaluated = EvaluateInvokers(handle,
                                              ss.second,
                                              ss.first,
                                              network_config,
                                              invoke_ctx,
                                              ret,
                                              force_attach_binary,
                                              /*non_naive_exists=*/false,
                                              /*naive_exceeds_work=*/false);

            ret.solutions.insert(ret.solutions.end(),
                                 std::make_move_iterator(evaluated.begin()),
                                 std::make_move_iterator(evaluated.end()));
        }
    }

    return ret;
}

namespace conv {

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo, const ProblemDescription& /*problem*/)
{
    switch(algo)
    { // clang-format off
    case miopenConvolutionAlgoGEMM:
#if MIOPEN_USE_GEMM
        return env::disabled(MIOPEN_DEBUG_CONV_GEMM);
#else
        return true;
#endif
    case miopenConvolutionAlgoDirect:
        return env::disabled(MIOPEN_DEBUG_CONV_DIRECT);
    case miopenConvolutionAlgoFFT:
        return env::disabled(MIOPEN_DEBUG_CONV_FFT);
    case miopenConvolutionAlgoWinograd:
        return env::disabled(MIOPEN_DEBUG_CONV_WINOGRAD);
    case miopenConvolutionAlgoImplicitGEMM:
        return env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM);
    } // clang-format on

    // Disable future algos by default to enforce explicit handling
    return true;
}

bool IsEnoughWorkspace(std::string_view where,
                       const miopen::solver::Id& solver_id,
                       const std::size_t required_size,
                       const miopen::AnyInvokeParams* const invokeParams,
                       bool log_as_warning)
{
    if(invokeParams != nullptr && required_size > 0)
    {
        const auto provided_size = invokeParams->GetWorkspaceSize();
        const auto provided_ptr  = invokeParams->GetWorkspace();
        if(provided_ptr == nullptr || provided_size < required_size)
        {
            std::stringstream log;
            log << "[" << where << "] Solver <" << solver_id.ToString() << ">"
                << ", workspace required: " << required_size << ", provided ptr: " << provided_ptr
                << " size: " << provided_size;
            if(log_as_warning)
                MIOPEN_LOG_W(log.str());
            else
                MIOPEN_LOG_I2(log.str());
            return false;
        }
    }
    return true;
}

} // namespace conv
} // namespace miopen
