// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solver_finders.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>

#include <hip/hip_runtime.h>
#include <miopen/any_solver.hpp>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/generic_search_controls.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solution.hpp>
#include <miopen/utility/modified_z.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE)

// MIOPEN_DEBUG_COMPILE_ONLY and MIOPEN_SEARCH_CUTOFF come from generic_search_controls.hpp.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_FIND_SKIP_PCT, 130)

/// Master kill switch for the whole speed-class mechanism. Default on; set to 0 to restore
/// the pre-gate behaviour of benchmarking and offering every applicable solver regardless
/// of how slow it declares itself to be. Present so that a mis-classification -- a solver
/// deferred on a shape where it was in fact the right choice -- is a one-variable
/// workaround rather than a rebuild, at the cost of re-exposing the watchdog reset (TDR)
/// this mechanism exists to prevent.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_DEFER_SLOW_SOLVERS)

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

bool IsDeferSlowSolversEnabled() { return !env::disabled(MIOPEN_DEBUG_DEFER_SLOW_SOLVERS); }

/// The speed class a solver reported for this problem, defaulting to Normal for anything
/// the caller did not record (which is everything, when the policy is disabled).
static solver::SolverSpeedClass
SpeedClassOf(const std::map<std::string, solver::SolverSpeedClass>& speed_classes,
             const std::string& solver_id)
{
    const auto it = speed_classes.find(solver_id);
    return it == speed_classes.end() ? solver::SolverSpeedClass::Normal : it->second;
}

/// Decide whether a solver should be skipped -- i.e. not executed at all in Find's
/// mini-benchmark -- because something better-classed is available.
///
/// Why this exists: some solvers are deliberately *applicable* far outside the region
/// where they are a sensible choice. Naive convolution is MIOpen's universal fallback and
/// must stay applicable at any problem size, but its kernel is un-tiled, so on a large
/// problem a single launch runs for multiple seconds. Explicit GEMM stays applicable into
/// shapes where im2col expansion and per-batch BLAS calls make it wildly uncompetitive.
/// Find times every applicable solver by actually *executing* it, so merely benchmarking
/// such a solver costs real time -- and in the naive case trips the OS GPU watchdog
/// (a TDR / driver reset) even when a fast solver also applies and would ultimately win.
/// The solver does not have to be *selected* to cause the hang; being *benchmarked* is
/// enough.
///
/// A solver classifies itself via SolverInterface::GetSpeedClass(ctx, problem), which sees
/// the concrete problem and the device. That keeps the *criterion* with the solver that
/// understands it (MAC work vs. a device-derived limit for naive, arch-aware shape
/// heuristics for GEMM) while this function owns the one thing a per-solver hook cannot
/// see: what else is available.
///
/// The rule is a single comparison: benchmark everything in the best (lowest) class that
/// has an applicable member, skip everything worse. benchmark_class is that class,
/// computed by the caller over the full solution list across *all* algorithms and *before*
/// workspace filtering -- so a fast-but-workspace-limited alternative still counts, which
/// is what closes the leak where the alternative never gets to time itself.
///
/// Consequences of that one rule (the whole policy in one place):
///   * Any Normal candidate exists -> every deferred solver is skipped and behaviour is
///     exactly as if this mechanism did not exist.
///   * No Normal, several Slow -> all the Slow ones are benchmarked and the least slow
///     wins on measurement. Deferral ranks candidates; it does not discard a whole tier
///     in favour of an arbitrary survivor.
///   * Only ExceedsLaunchBudget left -> it is benchmarked, preserving the universal-
///     fallback guarantee. A huge sole-naive shape may then still TDR -- an honest
///     "extend coverage here" signal we deliberately do not mask.
///   * Crucially, Slow and ExceedsLaunchBudget are *separate* classes rather than one
///     "slow" flag, so a merely wasteful solver outranks a watchdog risk: naive is never
///     launched just because GEMM also declared itself off the pace. With a single flag
///     that case -- all candidates slow, which is most likely exactly when the problem is
///     huge -- would have fallen through the gate and launched naive anyway.
///   * A solver outside its deferred regime (Normal) competes and wins on merit wherever
///     it is fastest. For naive this notably covers *all* real depthwise convs
///     (group == C == K => c_per_group == 1 => ~C x less work), which never approach the
///     work limit and for which naive is often the fastest option: they keep competing
///     with no special case.
///
/// This is deliberately a *pre-launch* gate. Bounding how long Find *waits* on a slow
/// solver cannot avert a TDR, because the dispatch has already happened and the kernel
/// keeps occupying the GPU after the wait is abandoned -- which is exactly what the OS
/// watchdog measures.
///
/// Golden references are unaffected: GpuConvReference compiles and launches the naive
/// kernel directly, bypassing the solver framework, so verification never reaches here.
static bool ShouldSkipSlowBenchmark(solver::SolverSpeedClass speed_class,
                                    solver::SolverSpeedClass benchmark_class)
{
    return speed_class > benchmark_class;
}

/// Register invoker only for the best solution within algorithm.
std::vector<Solution>
EvaluateInvokers(const Handle& handle,
                 const std::vector<solver::ConvSolution>& solutions,
                 const AlgorithmName& algorithm_name,
                 const NetworkConfig& network_config,
                 const AnyInvokeParams& invoke_ctx,
                 FindCoreResult& core_result,
                 bool force_attach_binary,
                 const std::map<std::string, solver::SolverSpeedClass>& speed_classes,
                 solver::SolverSpeedClass benchmark_class)
{
    std::vector<Solution> ret;

    const auto arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return ret;

    const auto speed_class_of = [&](const solver::ConvSolution& s) {
        return SpeedClassOf(speed_classes, s.solver_id);
    };

    bool using_search_cutoff = env::value(MIOPEN_SEARCH_CUTOFF);

    auto selected     = miopen::solver::ConvSolution{miopenStatusUnknownError};
    auto best         = std::numeric_limits<float>::max();
    auto best_invoker = Invoker{};
    std::vector<float> samples;

    // Benchmark in speed-class order, best first, so that the cheap candidates establish
    // find_search_best_time (and hence the measured-time cutoff below) before an expensive
    // one is timed against it. Stable, so solvers within a class keep finder order.
    std::vector<std::size_t> order(solutions.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return speed_class_of(solutions[a]) < speed_class_of(solutions[b]);
    });

    for(std::size_t idx : order)
    {
        const auto& sol = solutions[idx];

        const auto speed_class = speed_class_of(sol);
        if(ShouldSkipSlowBenchmark(speed_class, benchmark_class))
        {
            MIOPEN_LOG_I("Skipping last-resort (slow) Solver: " << algorithm_name.ToString() << ":"
                                                                << sol.solver_id);
            continue;
        }
        if(speed_class != solver::SolverSpeedClass::Normal)
        {
            // A last-resort solver is being benchmarked anyway because nothing better-classed
            // applies. Name the reason so this reads as an expected retention, not an anomaly.
            MIOPEN_LOG_I("Retaining last-resort (slow) Solver (no better alternative applies): "
                         << algorithm_name.ToString() << ":" << sol.solver_id);
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
                    // Emit this solver's record now. Relying on the *next* LogSolutionName() to
                    // flush loses the record of the last solver evaluated in a Find -- which is
                    // always a Direct-algorithm solver, i.e. ConvDirectNaiveConv*, because the
                    // Direct finder runs last. Flushing here makes every evaluated solver
                    // observable in the performance logs.
                    FlushJsonAccumulator();
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
    // Solver selection benchmarking. Without an explicit phase the thread-local default
    // (KernelPhase::Unknown) is used, so every solver timed here is logged as
    // "phase":"unknown". EvaluateConvSolutions() already scopes its identical call to
    // EvaluateInvokers() with SolverTuning; match it so Find timings are attributable.
    ScopedKernelPhase phase_scope(KernelPhase::SolverTuning);
    AutoEnableProfiling enableProfiling{handle};
    const auto network_config = problem.MakeNetworkConfig();
    auto ret                  = FindCoreResult();
    ret.is_optimal            = true;

    ret.solutions.reserve(total);

    // Ask every candidate how far off the pace it considers itself for *this* problem on
    // *this* device (SolverInterface::GetSpeedClass), and record the ones that are not
    // Normal. Computed here rather than inside EvaluateInvokers for two reasons:
    // EvaluateInvokers sees one algorithm group at a time, whereas "what is the best class
    // available" is a question about the whole candidate set; and this evaluates
    // GetSpeedClass once per solver instead of once per benchmark iteration.
    //
    // The scan runs over the full solution list *before* per-solver workspace filtering,
    // so a fast-but-workspace-limited alternative still counts and the deferred solver is
    // skipped rather than benchmarked -- that is the leak this closes, since otherwise the
    // alternative never gets to time itself and the deferred solver runs anyway.
    //
    // Non-conv (fusion) problems yield nullptr and leave the map empty, making the whole
    // mechanism inert for them -- as does MIOPEN_DEBUG_DEFER_SLOW_SOLVERS=0. An empty map
    // means every candidate reads back as Normal, so classes_present is {Normal}, the loop
    // below runs exactly once and nothing is skipped: the pre-gate behaviour, restored by
    // not gathering the data rather than by a second code path.
    std::map<std::string, solver::SolverSpeedClass> speed_classes;
    const auto* conv_problem = dynamic_cast<const conv::ProblemDescription*>(&problem);
    if(conv_problem != nullptr && IsDeferSlowSolversEnabled())
    {
        for(const auto& group : solutions)
        {
            for(const auto& s : group.second)
            {
                const auto any = solver::Id{s.solver_id}.GetSolver();
                if(any.IsEmpty())
                    continue;
                const auto speed_class = any.GetSpeedClass(ctx, *conv_problem);
                if(speed_class != solver::SolverSpeedClass::Normal)
                    speed_classes.emplace(s.solver_id, speed_class);
            }
        }
    }

    // The distinct classes present among the candidates, ascending (std::set is ordered),
    // i.e. the sequence of benchmark_class values to try. Normally this is just {Normal}
    // and the loop below runs once.
    std::set<solver::SolverSpeedClass> classes_present;
    for(const auto& group : solutions)
        for(const auto& s : group.second)
            classes_present.insert(SpeedClassOf(speed_classes, s.solver_id));
    if(classes_present.empty())
        classes_present.insert(solver::SolverSpeedClass::Normal);

    // Universal-fallback guarantee. The gate benchmarks only the best class present, but
    // applicability is not success: every candidate in that class can still be rejected at
    // evaluation (e.g. insufficient workspace). If that leaves no solution at all, we would
    // otherwise fail the convolution outright even though a deferred solver could have
    // served it. So descend one class at a time until something survives -- one class, not
    // straight to "gate off", because dropping from Slow to ExceedsLaunchBudget wholesale
    // is exactly the watchdog risk the classes exist to order. Past the first iteration
    // this only triggers in the rare empty-result case, and the already-rejected candidates
    // are re-rejected cheaply (workspace-filtered before any kernel launch).
    for(const auto benchmark_class : classes_present)
    {
        for(const auto& ss : solutions)
        {
            auto evaluated = EvaluateInvokers(handle,
                                              ss.second,
                                              ss.first,
                                              network_config,
                                              invoke_ctx,
                                              ret,
                                              force_attach_binary,
                                              speed_classes,
                                              benchmark_class);

            ret.solutions.insert(ret.solutions.end(),
                                 std::make_move_iterator(evaluated.begin()),
                                 std::make_move_iterator(evaluated.end()));
        }

        if(!ret.solutions.empty())
            break;

        if(benchmark_class != *classes_present.rbegin())
            MIOPEN_LOG_I("No solver survived evaluation; descending to the next (slower) "
                         "speed class as a fallback.");
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
