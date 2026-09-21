// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solver_finders.hpp>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <thread>

#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>

#include <hip/hip_runtime.h>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solution.hpp>
#include <miopen/utility/modified_z.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_COMPILE_ONLY)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_FIND_CONV_INSUFFICIENT_WORKSPACE_ALLOW_FINDDB_UPDATE)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_NAIVE_TIMEOUT, true)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_NAIVE_TIMEOUT_FACTOR, 300)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_SEARCH_CUTOFF, false)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_FIND_SKIP_PCT, 130)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_CONV_DIRECT_MAX_SIZE, 0)

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

namespace {

struct NaiveWarmup
{
    enum class Status
    {
        Completed,
        TimedOut,
        ScratchUnavailable,
        UnsupportedInvokeParams,
    };

    Status status;
    float elapsed;
};

/// Redirects the handle onto a tracker-owned stream with profiling off, restoring
/// both on scope exit so no early return can strand the handle off the root stream.
struct AutoExclusiveStream
{
    AutoExclusiveStream(const Handle& h, hipStream_t stream)
        : handle(h), prev_profiling(h.IsProfilingEnabled())
    {
        handle.SetExclusiveStream(stream);
        handle.EnableProfiling(false);
    }

    ~AutoExclusiveStream()
    {
        handle.SetExclusiveStream(nullptr);
        handle.EnableProfiling(prev_profiling);
    }

    AutoExclusiveStream(const AutoExclusiveStream&)            = delete;
    AutoExclusiveStream& operator=(const AutoExclusiveStream&) = delete;

private:
    const Handle& handle;
    bool prev_profiling;
};

std::string NaiveSkipReason(NaiveWarmup::Status status, float best_time)
{
    switch(status)
    {
    case NaiveWarmup::Status::TimedOut:
        return "exceeded " + std::to_string(env::value(MIOPEN_NAIVE_TIMEOUT_FACTOR)) +
               "% of best non-naive time (" + std::to_string(best_time) + " ms)";
    case NaiveWarmup::Status::ScratchUnavailable:
        return "output tensor exceeds the scratch buffer cap";
    case NaiveWarmup::Status::UnsupportedInvokeParams:
        return "invoke params type does not support scratch redirection";
    case NaiveWarmup::Status::Completed: break;
    }
    return "completed";
}

} // namespace

static NaiveWarmup TryNaiveWithTimeout(const Handle& handle,
                                       const Invoker& invoker,
                                       const AnyInvokeParams& invoke_ctx,
                                       float best_time)
{
    std::shared_ptr<ScratchAllocation> scratch;
    AnyInvokeParams scratch_ctx;

    if(invoke_ctx.IsOfType<conv::DataInvokeParams>())
    {
        auto params = invoke_ctx.CastTo<conv::DataInvokeParams>();
        scratch     = handle.GetScratchBuffer(params.tensors.outDesc.GetNumBytes());
        if(!scratch)
            return {NaiveWarmup::Status::ScratchUnavailable, 0.0f};
        params.tensors.out = scratch->buffer.get();
        scratch_ctx        = AnyInvokeParams{params};
    }
    else if(invoke_ctx.IsOfType<conv::WrWInvokeParams>())
    {
        auto params = invoke_ctx.CastTo<conv::WrWInvokeParams>();
        scratch     = handle.GetScratchBuffer(params.tensors.dwDesc.GetNumBytes());
        if(!scratch)
            return {NaiveWarmup::Status::ScratchUnavailable, 0.0f};
        params.tensors.dw = scratch->buffer.get();
        scratch_ctx       = AnyInvokeParams{params};
    }
    else
        return {NaiveWarmup::Status::UnsupportedInvokeParams, 0.0f};

    auto& tracker = handle.GetStreamTracker();
    auto slot     = tracker.acquire(handle);
    slot.scratch  = scratch;

    AutoExclusiveStream stream_guard{handle, slot.stream};

    HipEventPtr ev_start = make_hip_event();
    HipEventPtr ev_stop  = make_hip_event();

    try
    {
        auto ev_status = hipEventRecord(ev_start.get(), slot.stream);
        if(ev_status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(ev_status, "Failed to record naive start event");
        invoker(handle, scratch_ctx);
        ev_status = hipEventRecord(ev_stop.get(), slot.stream);
        if(ev_status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(ev_status, "Failed to record naive stop event");
    }
    catch(...)
    {
        tracker.abandon(slot);
        throw;
    }

    const float timeout_factor =
        static_cast<float>(env::value(MIOPEN_NAIVE_TIMEOUT_FACTOR)) / 100.0f;
    const float naive_budget = best_time * timeout_factor;
    const auto deadline      = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(static_cast<long long>(naive_budget * 1000));
    bool finished = false;
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(hipEventQuery(ev_stop.get()) == hipSuccess)
        {
            finished = true;
            break;
        }
        std::this_thread::yield();
    }

    if(finished)
    {
        tracker.release(slot);
        float warmup_elapsed = 0.0f;
        (void)hipEventElapsedTime(&warmup_elapsed, ev_start.get(), ev_stop.get());
        return {NaiveWarmup::Status::Completed, warmup_elapsed};
    }

    tracker.abandon(slot);
    return {NaiveWarmup::Status::TimedOut, 0.0f};
}

/// Register invoker only for the best solution within algorithm.
std::vector<Solution> EvaluateInvokers(const Handle& handle,
                                       const std::vector<solver::ConvSolution>& solutions,
                                       const AlgorithmName& algorithm_name,
                                       const NetworkConfig& network_config,
                                       const AnyInvokeParams& invoke_ctx,
                                       FindCoreResult& core_result,
                                       bool force_attach_binary,
                                       bool& non_naive_succeeded)
{
    std::vector<Solution> ret;

    const auto arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return ret;

    const auto is_naive_solver = [](const solver::ConvSolution& s) {
        return s.solver_id.find("Naive") != std::string::npos;
    };

    bool naive_timeout       = env::value(MIOPEN_NAIVE_TIMEOUT);
    bool using_search_cutoff = env::value(MIOPEN_SEARCH_CUTOFF);
    // Defer Naive only when a non-Naive alternative exists across all algorithms or this one.
    const bool defer_naive =
        naive_timeout &&
        (non_naive_succeeded || std::any_of(solutions.begin(), solutions.end(), [&](const auto& s) {
             return !is_naive_solver(s);
         }));
    auto selected     = miopen::solver::ConvSolution{miopenStatusUnknownError};
    auto best         = std::numeric_limits<float>::max();
    auto best_invoker = Invoker{};
    std::vector<float> samples;

    // Iterate non-Naive solutions first, Naive last
    std::vector<std::size_t> order(solutions.size());
    std::iota(order.begin(), order.end(), 0);
    if(defer_naive)
    {
        std::stable_partition(order.begin(), order.end(), [&](std::size_t i) {
            return !is_naive_solver(solutions[i]);
        });
    }

    for(std::size_t idx : order)
    {
        const auto& sol = solutions[idx];

        const bool is_naive     = is_naive_solver(sol);
        const bool cutoff_naive = defer_naive && is_naive && non_naive_succeeded;

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

            if(cutoff_naive)
            {
                const auto warmup = TryNaiveWithTimeout(
                    handle, invoker, invoke_ctx, core_result.find_search_best_time);
                if(warmup.status != NaiveWarmup::Status::Completed)
                {
                    MIOPEN_LOG_I(
                        "Skipped naive solver "
                        << algorithm_name.ToString() << ":" << sol.solver_id << ": "
                        << NaiveSkipReason(warmup.status, core_result.find_search_best_time));
                    continue;
                }
                first_elapsed = warmup.elapsed;
                i             = 1;
            }

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
            if(!is_naive)
                non_naive_succeeded = true;
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

    bool non_naive_succeeded = false;
    for(const auto& ss : solutions)
    {
        auto evaluated = EvaluateInvokers(handle,
                                          ss.second,
                                          ss.first,
                                          network_config,
                                          invoke_ctx,
                                          ret,
                                          force_attach_binary,
                                          non_naive_succeeded);

        ret.solutions.insert(ret.solutions.end(),
                             std::make_move_iterator(evaluated.begin()),
                             std::make_move_iterator(evaluated.end()));
    }

    return ret;
}

namespace conv {

namespace detail {
/// Determine if problem size exceeds threshold for Direct solver.
///
/// The result tensor is used to estimate problem size.
/// The maximum size is determined by MIOPEN_CONV_DIRECT_MAX_SIZE environment variable.
///
/// @param problem The convolution problem description.
bool IsDirectProblemTooLarge(const ProblemDescription& problem)
{
    const unsigned long long max_size = env::value(MIOPEN_CONV_DIRECT_MAX_SIZE);
    // 0 means no limit
    if(max_size == 0)
        return false;

    // For FWD/BWD: 'out' is the result (swapped in BWD)
    // For WRW: 'weights' is the result (out is dy, not dw)
    const size_t problem_size = problem.IsDirectionBackwardWrW()
                                    ? problem.GetWeights().GetElementSize()
                                    : problem.GetOut().GetElementSize();

    // Problem size is within limit
    if(problem_size <= max_size)
        return false;

    MIOPEN_LOG_I2("DirectSolverFinder disabled for problem size "
                  << problem_size << " > " << max_size << " (MIOPEN_CONV_DIRECT_MAX_SIZE)");
    return true;
}
} // namespace detail

bool IsAlgorithmDisabled(miopenConvAlgorithm_t algo, const ProblemDescription& problem)
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
        return env::disabled(MIOPEN_DEBUG_CONV_DIRECT) || detail::IsDirectProblemTooLarge(problem);
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
