// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file XengineEval.cpp
 * @brief Measures cross-engine heuristic selection: its latency, and the regret of what it
 *        picks against what is fastest.
 *
 * One process sweeps a graph list through one handle, so plugin load and model load are paid
 * once. Per graph it records, as one flushed JSON line (schema "hipdnn.xengine_eval/1"):
 *
 *  - the engines applicable to the graph,
 *  - per policy (ModeA, StaticOrdering, built-in default; forced through
 *    HIPDNN_HEUR_POLICY_ORDER around each query): the cold and warm latency of
 *    get_ranked_engine_ids({HeuristicMode::A}), the ranked list, and the timed execution of the
 *    plan create_execution_plans({HeuristicMode::A}) actually activates,
 *  - per engine: the engine-level (L1) prediction in the ranking metric, and the immediate
 *    time (global.benchmarking=0, timed exactly as hipdnn_bench --collect-immediate does),
 *  - the baseline: an EXHAUSTIVE autotune over every applicable engine.
 *
 * It never calls autotuneExhaustiveSweep(), which writes the exact-match autotune cache, and
 * it never passes an autotune storage file. Every phase builds its own graph: the plan-spec
 * path (add_all_engines) and the compiled-plan path (create_execution_plan*) are mutually
 * exclusive on one Graph. Device memory is allocated per phase and freed at its end.
 *
 * The selection phase runs before the predictions and timings so that the first heuristic
 * query of the process (first_query_in_process) is the first thing to touch the heuristic
 * models, and its cold latency therefore holds the model load it is meant to expose.
 */

#include <hipdnn_bench/DeviceIdentity.hpp>
#include <hipdnn_bench/GraphFormat.hpp>
#include <hipdnn_bench/GraphList.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>
#include <hipdnn_tools/CliArguments.hpp>
#include <hipdnn_tools/GraphContentKey.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/TimedRunLoop.hpp>
#include <hipdnn_frontend/detail/EngineQueries.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace
{

using hipdnn_frontend::AutotuneConfig;
using hipdnn_frontend::AutotuneResult;
using hipdnn_frontend::AutotuneStrategy;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::HeuristicMode;
using hipdnn_frontend::KnobSetting;
using hipdnn_frontend::TuneMode;
using nlohmann::json;

constexpr const char* SCHEMA = "hipdnn.xengine_eval/1";
constexpr const char* POLICY_ORDER_ENV = "HIPDNN_HEUR_POLICY_ORDER";

using Clock = std::chrono::steady_clock;

/// Exposes the protected finalized backend descriptor for engine-inspection queries.
class BenchGraph : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::get_raw_graph_descriptor;
};

// ---------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------

/// One heuristic policy the selection is measured under. A nullopt order leaves
/// HIPDNN_HEUR_POLICY_ORDER unset, which is the built-in [Config, ModeA, StaticOrdering].
struct Policy
{
    std::string name;
    std::optional<std::string> order;
};

Policy namedPolicy(const std::string& name)
{
    if(name == "modeA")
    {
        return {name,
                std::string(hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME)
                    + ",SelectionHeuristic::StaticOrdering"};
    }
    if(name == "static")
    {
        return {name, std::string("SelectionHeuristic::StaticOrdering")};
    }
    return {name, std::nullopt};
}

struct Options
{
    std::vector<std::string> pluginDirs;
    std::string graphListPath;
    std::vector<Policy> policies;
    int latencyRepeats = 50;
    bool autotune = true;
    bool immediate = true;
    /// Enough warmup to keep first-run kernel compilation out of the timed loop.
    int warmup = 10;
    int maxIterations = 100;
    float stability = 0.05F;
    /// Metric predictions are made in. Passed to the graphs (set_ranking_metric) only when the
    /// caller named one, so the environment and the backend default decide otherwise.
    std::string rankingMetric{hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC};
    bool haveRankingMetric = false;
    /// Soft per-graph budget in seconds; 0 disables it.
    double graphTimeoutS = 0.0;
};

void printHelp(const char* program)
{
    std::cout
        << "Usage: " << program << " --graph-list <file> [options]\n\n"
        << "  --graph-list <file>      One graph path per line (JSON or FlatBuffer); one JSON\n"
        << "                           line per graph, in list order\n"
        << "  --plugin-dir <path>      Engine plugin directory, or one plugin library\n"
        << "                           (repeatable)\n"
        << "  --policies <a,b,...>     Policies to time, from modeA,static,default\n"
        << "                           (default: all three)\n"
        << "  --latency-repeats <n>    Warm get_ranked_engine_ids calls per policy (default 50)\n"
        << "  --no-autotune            Skip the EXHAUSTIVE autotune baseline\n"
        << "  --no-immediate           Skip per-engine immediate timing\n"
        << "  --warmup <n>             Untimed iterations before timing (default 10)\n"
        << "  --max-iterations <n>     Ceiling for the stability loop (default 100)\n"
        << "  --stability <f>          Coefficient-of-variation threshold (default 0.05)\n"
        << "  --ranking-metric <m>     tflops or time (default: environment, then tflops)\n"
        << "  --graph-timeout-s <s>    Skip the remaining phases of a graph once it has run this\n"
        << "                           long (0 = no limit); a call already running is not\n"
        << "                           interrupted\n";
}

enum class ParseOutcome
{
    RUN,
    HELP,
    INVALID
};

ParseOutcome parseArguments(const std::vector<std::string>& args, Options& options)
{
    std::string policyList = "modeA,static,default";
    for(size_t i = 1; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        hipdnn_tools::OptionReader reader(args, i);
        const auto next = [&reader]() { return reader.next(); };

        if(arg == "--help" || arg == "-h")
        {
            printHelp(args[0].c_str());
            return ParseOutcome::HELP;
        }
        if(arg == "--graph-list")
        {
            options.graphListPath = next();
        }
        else if(arg == "--plugin-dir")
        {
            options.pluginDirs.push_back(next());
        }
        else if(arg == "--policies")
        {
            policyList = next();
        }
        else if(arg == "--latency-repeats")
        {
            reader.integer(options.latencyRepeats, 0);
        }
        else if(arg == "--no-autotune")
        {
            options.autotune = false;
        }
        else if(arg == "--no-immediate")
        {
            options.immediate = false;
        }
        else if(arg == "--warmup")
        {
            reader.integer(options.warmup, 0);
        }
        else if(arg == "--max-iterations")
        {
            reader.integer(options.maxIterations, 1);
        }
        else if(arg == "--stability")
        {
            reader.real(options.stability, 0.0);
        }
        else if(arg == "--ranking-metric")
        {
            options.rankingMetric = next();
            options.haveRankingMetric = true;
            if(hipdnn_data_sdk::utilities::findRankingMetric(options.rankingMetric) == nullptr)
            {
                throw std::invalid_argument("Unregistered ranking metric '" + options.rankingMetric
                                            + "'");
            }
        }
        else if(arg == "--graph-timeout-s")
        {
            reader.real(options.graphTimeoutS, 0.0);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printHelp(args[0].c_str());
            return ParseOutcome::INVALID;
        }
        if(reader.refused(arg, std::cerr))
        {
            return ParseOutcome::INVALID;
        }
    }

    if(options.graphListPath.empty())
    {
        std::cerr << "--graph-list is required\n";
        return ParseOutcome::INVALID;
    }
    if(options.warmup < 1 || options.maxIterations < AutotuneConfig{}.windowSize
       || !std::isfinite(options.stability) || options.stability <= 0.0F
       || options.stability >= 1.0F)
    {
        throw std::invalid_argument("Timing requires positive warmup, 0 < stability < 1, and "
                                    "max-iterations >= the stability window");
    }

    std::stringstream names(policyList);
    std::string name;
    while(std::getline(names, name, ','))
    {
        if(name != "modeA" && name != "static" && name != "default")
        {
            throw std::invalid_argument("Unknown policy '" + name
                                        + "'; expected modeA, static or default");
        }
        if(std::any_of(options.policies.begin(),
                       options.policies.end(),
                       [&name](const Policy& policy) { return policy.name == name; }))
        {
            throw std::invalid_argument("Duplicate policy '" + name + "'");
        }
        options.policies.push_back(namedPolicy(name));
    }
    if(options.policies.empty())
    {
        throw std::invalid_argument("--policies names no policy");
    }
    return ParseOutcome::RUN;
}

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

/// Forces HIPDNN_HEUR_POLICY_ORDER for the scope and restores what it was. The backend
/// re-reads the variable on every heuristic finalize, so the setting holds only for the
/// queries made inside the scope.
class ScopedPolicyOrder
{
public:
    explicit ScopedPolicyOrder(const std::optional<std::string>& order)
        : _previous(hipdnn_data_sdk::utilities::getEnv(POLICY_ORDER_ENV))
    {
        if(order)
        {
            hipdnn_data_sdk::utilities::setEnv(POLICY_ORDER_ENV, order->c_str());
        }
        else
        {
            hipdnn_data_sdk::utilities::unsetEnv(POLICY_ORDER_ENV);
        }
    }

    ~ScopedPolicyOrder()
    {
        if(_previous.empty())
        {
            hipdnn_data_sdk::utilities::unsetEnv(POLICY_ORDER_ENV);
        }
        else
        {
            hipdnn_data_sdk::utilities::setEnv(POLICY_ORDER_ENV, _previous.c_str());
        }
    }

    ScopedPolicyOrder(const ScopedPolicyOrder&) = delete;
    ScopedPolicyOrder& operator=(const ScopedPolicyOrder&) = delete;
    ScopedPolicyOrder(ScopedPolicyOrder&&) = delete;
    ScopedPolicyOrder& operator=(ScopedPolicyOrder&&) = delete;

private:
    std::string _previous;
};

/// A soft per-graph budget, checked between phases and repeats.
class Deadline
{
public:
    explicit Deadline(double seconds)
        : _enabled(seconds > 0.0)
        , _end(Clock::now()
               + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                   seconds > 0.0 ? seconds : 0.0)))
    {
    }

    bool expired() const
    {
        return _enabled && Clock::now() >= _end;
    }

private:
    bool _enabled;
    Clock::time_point _end;
};

std::string timeoutMessage(const Options& options)
{
    return "graph timeout of " + std::to_string(options.graphTimeoutS) + " s exceeded";
}

/// Runs @p phase, returning why it failed (an Error or an exception) or nothing. One failing
/// phase must not drop the rest of a graph.
template <typename Phase>
std::optional<std::string> guarded(Phase&& phase)
{
    try
    {
        const Error error = phase();
        if(error.is_bad())
        {
            const auto& message = error.get_message();
            return message.empty() ? std::string("unspecified error") : message;
        }
        return std::nullopt;
    }
    catch(const std::exception& exception)
    {
        return std::string("exception: ") + exception.what();
    }
    catch(...)
    {
        return std::string("non-standard exception");
    }
}

std::string joinErrors(const std::vector<std::string>& errors)
{
    std::string joined;
    for(const auto& error : errors)
    {
        joined += joined.empty() ? error : "; " + error;
    }
    return joined;
}

Error hipError(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                std::string(operation) + ": " + hipGetErrorString(status)};
    }
    return {};
}

double microseconds(Clock::duration duration)
{
    return std::chrono::duration<double, std::micro>(duration).count();
}

/// The @p fraction quantile of an ascending @p sorted, interpolating between ranks.
double quantile(const std::vector<double>& sorted, double fraction)
{
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const auto upper = static_cast<size_t>(std::ceil(position));
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - std::floor(position));
}

const char* predictionStatus(hipdnn_frontend::PredictionStatus status)
{
    switch(status)
    {
    case hipdnn_frontend::PredictionStatus::AVAILABLE:
        return "available";
    case hipdnn_frontend::PredictionStatus::UNAVAILABLE:
        return "unavailable";
    case hipdnn_frontend::PredictionStatus::INVALID:
    default:
        return "invalid";
    }
}

/// A timing result the way every phase reports it; fields stay null until measured.
json timingRecord()
{
    return {{"is_valid", false},
            {"robustMeanMs", nullptr},
            {"min_time_ms", nullptr},
            {"avg_time_ms", nullptr},
            {"stddev_ms", nullptr},
            {"converged", false},
            {"iterations", 0},
            {"timing_quality", nullptr},
            {"workspace_bytes", nullptr},
            {"error", nullptr}};
}

// ---------------------------------------------------------------------------------------
// Variant-pack memory
// ---------------------------------------------------------------------------------------

/// Variant-pack memory for one phase. Not a general allocator -- it exists so the buffers
/// outlive execute() and are released even when a measurement fails.
class PackBuffers
{
public:
    ~PackBuffers()
    {
        for(void* pointer : _devicePointers)
        {
            (void)hipFree(pointer);
        }
    }
    PackBuffers() = default;
    PackBuffers(const PackBuffers&) = delete;
    PackBuffers& operator=(const PackBuffers&) = delete;
    PackBuffers(PackBuffers&&) = delete;
    PackBuffers& operator=(PackBuffers&&) = delete;

    /// Allocates zero-filled device memory; uninitialised denormals/NaNs can slow kernels.
    void* addDevice(int64_t bytes)
    {
        void* pointer = nullptr;
        if(hipMalloc(&pointer, static_cast<size_t>(bytes)) != hipSuccess || pointer == nullptr)
        {
            return nullptr;
        }
        if(hipMemset(pointer, 0, static_cast<size_t>(bytes)) != hipSuccess)
        {
            (void)hipFree(pointer);
            return nullptr;
        }
        _devicePointers.push_back(pointer);
        return pointer;
    }

    /// Zero-filled host memory for the runtime pass-by-value scalars a provider reads on the
    /// CPU. Stable for the object's lifetime: moving a block's vector keeps its buffer.
    void* addHost(int64_t bytes)
    {
        _hostBlocks.emplace_back(static_cast<size_t>(bytes));
        return _hostBlocks.back().data();
    }

    /// Memory for @p tensor where the plan says it lives.
    void* add(const hipdnn_bench::TensorRequirement& tensor)
    {
        return tensor.storage == hipdnn_bench::TensorStorage::HOST ? addHost(tensor.bytes)
                                                                   : addDevice(tensor.bytes);
    }

private:
    std::vector<void*> _devicePointers;
    std::vector<std::vector<std::byte>> _hostBlocks;
};

Error allocateVariantPack(const hipdnn_bench::VariantPackPlan& plan,
                          PackBuffers& buffers,
                          std::unordered_map<int64_t, void*>& variantPack)
{
    if(!plan.error.empty())
    {
        return {ErrorCode::INVALID_VALUE, plan.error};
    }
    for(const auto& tensor : plan.tensors)
    {
        void* pointer = buffers.add(tensor);
        if(pointer == nullptr)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Out of device memory for '" + tensor.name + "' ("
                        + std::to_string(tensor.bytes) + " bytes)"};
        }
        variantPack[tensor.uid] = pointer;
    }
    return {};
}

/// The variant pack and workspace of one phase.
struct PhaseMemory
{
    PackBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    void* workspace = nullptr;
};

Error allocateWorkspace(PhaseMemory& memory, int64_t workspaceSize)
{
    if(workspaceSize <= 0)
    {
        return {};
    }
    memory.workspace = memory.buffers.addDevice(workspaceSize);
    if(memory.workspace == nullptr)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Out of device memory for a " + std::to_string(workspaceSize) + " byte workspace"};
    }
    return {};
}

// ---------------------------------------------------------------------------------------
// Session: everything the phases share
// ---------------------------------------------------------------------------------------

/// What one process holds across graphs.
struct Session
{
    hipdnnHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
    Options options;
    std::string arch;
    std::string deviceName;
    std::string deviceId;
    /// True until the first heuristic query of the process has been issued.
    bool firstQueryPending = true;
    std::unordered_map<int64_t, std::string> engineNames;

    /// The name the backend gives @p engineId; a placeholder when it has none.
    const std::string& engineName(int64_t engineId)
    {
        const auto known = engineNames.find(engineId);
        if(known != engineNames.end())
        {
            return known->second;
        }
        return engineNames.emplace(engineId, queryEngineName(engineId)).first->second;
    }

private:
    std::string queryEngineName(int64_t engineId) const
    {
        size_t nameSize = 0;
        if(hipdnnGetEngineNameById_ext(handle, engineId, nullptr, &nameSize)
               != HIPDNN_STATUS_SUCCESS
           || nameSize == 0)
        {
            return "unknown:" + std::to_string(engineId);
        }
        std::vector<char> name(nameSize);
        if(hipdnnGetEngineNameById_ext(handle, engineId, name.data(), &nameSize)
           != HIPDNN_STATUS_SUCCESS)
        {
            return "unknown:" + std::to_string(engineId);
        }
        return std::string(name.data());
    }
};

/// A graph file read once: its problem bytes (any embedded plan discarded) and content key.
struct GraphSource
{
    std::vector<uint8_t> problem;
    std::string key;
};

/// Reads @p path and keys it. Returns why it could not, or nothing.
std::optional<std::string> loadSource(const Session& session,
                                      const std::string& path,
                                      GraphSource& source)
{
    std::ifstream graphFile(path, std::ios::binary);
    if(!graphFile)
    {
        return "Cannot read " + path;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(graphFile)),
                                     std::istreambuf_iterator<char>());
    if(bytes.empty())
    {
        return path + " is empty";
    }

    BenchGraph probe;
    const bool isJson = hipdnn_bench::isJsonGraph(bytes);
    const auto failure = guarded([&]() -> Error {
        if(isJson)
        {
            HIPDNN_CHECK_ERROR(
                probe.deserialize(session.handle, std::string(bytes.begin(), bytes.end())));
        }
        else
        {
            // Discard any embedded plan; re-serialize rather than re-lower to keep the graph ID.
            HIPDNN_CHECK_ERROR(probe.deserialize(bytes));
            std::vector<uint8_t> stripped;
            HIPDNN_CHECK_ERROR(probe.serialize(stripped));
            HIPDNN_CHECK_ERROR(probe.deserialize(session.handle, stripped));
        }
        HIPDNN_CHECK_ERROR(probe.serialize(source.problem));
        source.key = hipdnn_tools::graphContentKey(source.problem.data(), source.problem.size());
        return {};
    });
    if(failure)
    {
        return "Could not read " + path + ": " + *failure;
    }
    return std::nullopt;
}

/// Restores @p source into a fresh @p graph on the session's handle.
Error restore(const Session& session, const GraphSource& source, BenchGraph& graph)
{
    HIPDNN_CHECK_ERROR(graph.deserialize(session.handle, source.problem));
    if(session.options.haveRankingMetric)
    {
        HIPDNN_CHECK_ERROR(graph.set_ranking_metric(session.options.rankingMetric));
    }
    return {};
}

// ---------------------------------------------------------------------------------------
// Timed loop (the collectImmediate loop of hipdnn_bench)
// ---------------------------------------------------------------------------------------

/// Times the active plan of @p graph into @p record: warmup, then the stability loop, taking
/// device-only time and falling back to an unstalled pass when the stall cannot hold, exactly
/// as hipdnn_bench --collect-immediate does. Allocates its buffers and frees them on return.
Error timeActivePlan(const Session& session, BenchGraph& graph, json& record)
{
    const auto& options = session.options;
    PhaseMemory memory;
    HIPDNN_CHECK_ERROR(
        allocateVariantPack(hipdnn_bench::planVariantPack(graph), memory.buffers, memory.variantPack));
    int64_t workspaceSize = 0;
    HIPDNN_CHECK_ERROR(graph.get_workspace_size(workspaceSize));
    record["workspace_bytes"] = workspaceSize;
    HIPDNN_CHECK_ERROR(allocateWorkspace(memory, workspaceSize));
    // execute() takes the pack by non-const reference.
    auto& variantPack = memory.variantPack;
    void* workspace = memory.workspace;
    const hipStream_t stream = session.stream;
    const hipdnnHandle_t handle = session.handle;

    // Reuse the HIP events across samples; compilation, allocation and warmup are untimed.
    hipEvent_t startEvent = nullptr;
    HIPDNN_CHECK_ERROR(hipError(hipEventCreate(&startEvent), "Could not create start event"));
    const hipdnn_data_sdk::utilities::ScopedResource start(startEvent, hipEventDestroy);
    hipEvent_t stopEvent = nullptr;
    HIPDNN_CHECK_ERROR(hipError(hipEventCreate(&stopEvent), "Could not create stop event"));
    const hipdnn_data_sdk::utilities::ScopedResource stop(stopEvent, hipEventDestroy);

    for(int w = 0; w < options.warmup; ++w)
    {
        HIPDNN_CHECK_ERROR(graph.execute(handle, variantPack, workspace));
    }
    HIPDNN_CHECK_ERROR(hipError(hipStreamSynchronize(stream), "Warmup synchronization failed"));

    // The runtime ranks kernels with the stream held until host submission has finished, so
    // the span holds device work and no host dispatch time. execute_timed_ext() takes the same
    // measurement.
    const auto timeDeviceOnly = [&](hipdnn_frontend::ExecutionTiming& timing) -> Error {
        return graph.execute_timed_ext(handle, variantPack, workspace, timing);
    };
    // Events around execute(), for a device that cannot hold the stream. The span then includes
    // host dispatch time, as the runtime's own unstalled re-measurement does.
    const auto timeUnstalled = [&](hipdnn_frontend::ExecutionTiming& timing) -> Error {
        HIPDNN_CHECK_ERROR(hipError(hipEventRecord(start.get(), stream), "Could not start timing"));
        HIPDNN_CHECK_ERROR(graph.execute(handle, variantPack, workspace));
        HIPDNN_CHECK_ERROR(hipError(hipEventRecord(stop.get(), stream), "Could not stop timing"));
        HIPDNN_CHECK_ERROR(
            hipError(hipEventSynchronize(stop.get()), "Timing synchronization failed"));
        float elapsed = 0.0F;
        HIPDNN_CHECK_ERROR(hipError(hipEventElapsedTime(&elapsed, start.get(), stop.get()),
                                    "Could not read timing"));
        if(!std::isfinite(elapsed) || elapsed == 0.0F)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "HIP event timing must be finite and positive"};
        }
        // A negative reading leaves `timing` INVALID so the loop retries the slot instead of
        // failing the collection.
        if(elapsed > 0.0F)
        {
            timing.elapsedMs = elapsed;
            timing.quality = hipdnn_frontend::TimingQuality::UNSTALLED;
        }
        return {};
    };
    const auto onIteration = [](int, float, float, bool) {};
    const int window = AutotuneConfig{}.windowSize;
    auto quality = hipdnn_frontend::TimingQuality::DEVICE_ONLY;
    auto outcome = hipdnn_frontend::autotune::detail::runUntilStable(
        options.maxIterations, window, options.stability, /*stalled=*/true, timeDeviceOnly, onIteration);
    if(outcome.restartUnstalled)
    {
        // The stall could not hold (a watchdog timeout, or a device without stalling): discard
        // the pass and measure again unstalled, as the runtime does.
        quality = hipdnn_frontend::TimingQuality::UNSTALLED;
        outcome = hipdnn_frontend::autotune::detail::runUntilStable(options.maxIterations,
                                                                    window,
                                                                    options.stability,
                                                                    /*stalled=*/false,
                                                                    timeUnstalled,
                                                                    onIteration);
    }
    record["timing_quality"]
        = quality == hipdnn_frontend::TimingQuality::DEVICE_ONLY ? "device_only" : "unstalled";
    record["iterations"] = outcome.timings.size();
    record["converged"] = outcome.converged;
    if(outcome.benchmarkFailed)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, outcome.errorMessage};
    }
    if(outcome.timings.empty())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "The timed loop recorded no sample"};
    }
    record["robustMeanMs"]
        = static_cast<double>(hipdnn_data_sdk::utilities::detail::robustMean(outcome.timings));
    record["min_time_ms"]
        = static_cast<double>(*std::min_element(outcome.timings.begin(), outcome.timings.end()));
    record["avg_time_ms"]
        = static_cast<double>(hipdnn_data_sdk::utilities::detail::mean(outcome.timings));
    record["stddev_ms"]
        = static_cast<double>(hipdnn_data_sdk::utilities::detail::stddev(outcome.timings));
    // `is_valid` means a measurement was obtained, never numerical correctness.
    record["is_valid"] = true;
    return {};
}

/// timeActivePlan() with its failure recorded in @p record's `error`.
void measureActivePlan(const Session& session, BenchGraph& graph, json& record)
{
    if(const auto failure = guarded([&]() -> Error { return timeActivePlan(session, graph, record); }))
    {
        record["is_valid"] = false;
        record["error"] = *failure;
    }
}

// ---------------------------------------------------------------------------------------
// Selection (per policy)
// ---------------------------------------------------------------------------------------

/// Cold and warm latency of get_ranked_engine_ids({HeuristicMode::A}) on a built graph, and the
/// ranking it returns. Must run with the policy order already forced.
Error measureLatency(Session& session,
                     const GraphSource& source,
                     const Deadline& deadline,
                     json& selection)
{
    BenchGraph graph;
    HIPDNN_CHECK_ERROR(restore(session, source, graph));

    const std::vector<HeuristicMode> modes{HeuristicMode::A};
    const bool first = session.firstQueryPending;
    session.firstQueryPending = false;
    selection["first_query_in_process"] = first;

    std::vector<int64_t> ranked;
    const auto coldStart = Clock::now();
    const Error coldStatus = graph.get_ranked_engine_ids(ranked, modes);
    const auto coldStop = Clock::now();
    HIPDNN_CHECK_ERROR(coldStatus);
    selection["cold_us"] = microseconds(coldStop - coldStart);

    json rankedJson = json::array();
    for(const int64_t engineId : ranked)
    {
        rankedJson.push_back(
            json{{"engine_id", engineId}, {"engine_name", session.engineName(engineId)}});
    }
    selection["ranked"] = std::move(rankedJson);
    if(ranked.empty())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "The heuristic ranked no engine"};
    }
    selection["picked_engine_id"] = ranked.front();
    selection["picked_engine_name"] = session.engineName(ranked.front());

    const int repeats = session.options.latencyRepeats;
    if(repeats == 0)
    {
        return {};
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    std::vector<int64_t> scratch;
    scratch.reserve(ranked.size());
    Error pending;
    for(int i = 0; i < repeats; ++i)
    {
        if(deadline.expired())
        {
            pending = {ErrorCode::HIPDNN_BACKEND_ERROR,
                       timeoutMessage(session.options) + " after " + std::to_string(i) + " of "
                           + std::to_string(repeats) + " warm latency repeats"};
            break;
        }
        scratch.clear();
        const auto begin = Clock::now();
        const Error status = graph.get_ranked_engine_ids(scratch, modes);
        const auto end = Clock::now();
        if(status.is_bad())
        {
            pending = {ErrorCode::HIPDNN_BACKEND_ERROR,
                       "warm latency repeat " + std::to_string(i) + " failed: "
                           + status.get_message()};
            break;
        }
        samples.push_back(microseconds(end - begin));
    }
    if(!samples.empty())
    {
        std::sort(samples.begin(), samples.end());
        selection["warm_us"] = {{"median", quantile(samples, 0.5)},
                                {"p10", quantile(samples, 0.1)},
                                {"p90", quantile(samples, 0.9)},
                                {"min", samples.front()},
                                {"repeats", samples.size()}};
    }
    return pending;
}

/// Plans a fresh graph with create_execution_plans({HeuristicMode::A}), builds its active plan
/// and times it. Must run with the policy order already forced.
Error measureExecution(Session& session, const GraphSource& source, json& selection)
{
    BenchGraph graph;
    HIPDNN_CHECK_ERROR(restore(session, source, graph));
    HIPDNN_CHECK_ERROR(graph.create_execution_plans({HeuristicMode::A}));
    HIPDNN_CHECK_ERROR(graph.build_plans());
    int64_t engineId = 0;
    HIPDNN_CHECK_ERROR(graph.get_execution_plan_engine_id(engineId));

    json record = timingRecord();
    record["engine_id"] = engineId;
    record["engine_name"] = session.engineName(engineId);
    measureActivePlan(session, graph, record);
    selection["execution"] = std::move(record);
    return {};
}

json measurePolicy(Session& session,
                   const GraphSource& source,
                   const Policy& policy,
                   const Deadline& deadline)
{
    json selection = {{"policy_order", policy.order ? json(*policy.order) : json(nullptr)},
                      {"ranked", json::array()},
                      {"picked_engine_id", nullptr},
                      {"picked_engine_name", nullptr},
                      {"cold_us", nullptr},
                      {"first_query_in_process", false},
                      {"warm_us", nullptr},
                      {"execution", nullptr},
                      {"error", nullptr}};
    std::vector<std::string> errors;
    const ScopedPolicyOrder forced(policy.order);

    if(deadline.expired())
    {
        errors.push_back("latency: " + timeoutMessage(session.options));
    }
    else if(const auto latencyFailure = guarded(
                [&]() -> Error { return measureLatency(session, source, deadline, selection); }))
    {
        errors.push_back("latency: " + *latencyFailure);
    }

    if(deadline.expired())
    {
        errors.push_back("execution: " + timeoutMessage(session.options));
    }
    else if(const auto executionFailure = guarded(
                [&]() -> Error { return measureExecution(session, source, selection); }))
    {
        errors.push_back("execution: " + *executionFailure);
    }

    if(!errors.empty())
    {
        selection["error"] = joinErrors(errors);
    }
    return selection;
}

// ---------------------------------------------------------------------------------------
// Per-engine predictions and immediate times
// ---------------------------------------------------------------------------------------

/// The engine-level prediction of @p engineId in the ranking metric. Also lifts the graph's
/// logical work count (`graph.flops` of the unevaluated description) into @p flops, once.
json measurePrediction(const Session& session,
                       BenchGraph& graph,
                       int64_t engineId,
                       std::optional<double>& flops)
{
    const auto& metric = session.options.rankingMetric;
    json record = {{"status", "invalid"}, {"value", nullptr}, {"model", ""}, {"reason", nullptr}};
    const auto failure = guarded([&]() -> Error {
        hipdnn_frontend::EnginePrediction prediction;
        HIPDNN_CHECK_ERROR(
            hipdnn_frontend::detail::getEnginePrediction(graph.get_raw_graph_descriptor(),
                                                         engineId,
                                                         prediction,
                                                         hipdnn_frontend::PredictionKind::ENGINE,
                                                         /*evaluate=*/true,
                                                         {},
                                                         metric));
        record["status"] = predictionStatus(prediction.status);
        record["value"] = prediction.value ? json(*prediction.value) : json(nullptr);
        record["model"] = prediction.model;
        record["reason"] = prediction.reason.empty() ? json(nullptr) : json(prediction.reason);
        return {};
    });
    if(failure)
    {
        record["reason"] = "prediction query failed: " + *failure;
    }

    if(!flops)
    {
        // Best effort: a graph without a counting convention publishes no `graph.flops`.
        (void)guarded([&]() -> Error {
            hipdnn_frontend::EnginePrediction description;
            HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::getEnginePrediction(
                graph.get_raw_graph_descriptor(),
                engineId,
                description,
                hipdnn_frontend::PredictionKind::ENGINE,
                /*evaluate=*/false,
                {},
                metric));
            const auto found = description.features.find("graph.flops");
            if(found != description.features.end() && found->is_number())
            {
                const double value = found->get<double>();
                if(std::isfinite(value) && value > 0.0)
                {
                    flops = value;
                }
            }
            return {};
        });
    }
    return record;
}

/// The immediate time of @p engineId: its normal plan with global.benchmarking=0, never
/// tuned, on a fresh graph.
json measureImmediate(Session& session,
                      const GraphSource& source,
                      int64_t engineId,
                      const Deadline& deadline)
{
    json record = timingRecord();
    if(deadline.expired())
    {
        record["error"] = timeoutMessage(session.options);
        return record;
    }
    const auto failure = guarded([&]() -> Error {
        BenchGraph graph;
        HIPDNN_CHECK_ERROR(restore(session, source, graph));
        const std::vector<KnobSetting> settings{KnobSetting(
            std::string(hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME), int64_t{0})};
        HIPDNN_CHECK_ERROR(graph.create_execution_plan_ext(engineId, settings));
        HIPDNN_CHECK_ERROR(graph.build_plans());
        measureActivePlan(session, graph, record);
        return {};
    });
    if(failure)
    {
        record["is_valid"] = false;
        record["error"] = *failure;
    }
    return record;
}

// ---------------------------------------------------------------------------------------
// Autotune baseline
// ---------------------------------------------------------------------------------------

json knobsJson(const std::vector<KnobSetting>& settings)
{
    json knobs = json::object();
    for(const auto& setting : settings)
    {
        std::visit([&](const auto& value) { knobs[setting.knobId()] = value; }, setting.value());
    }
    return knobs;
}

/// The statistic every comparison uses: robust mean, falling back to the average when a
/// result carries none.
double comparisonMs(const AutotuneResult& result)
{
    return result.robustTimeMs > 0.0F ? static_cast<double>(result.robustTimeMs)
                                      : static_cast<double>(result.avgTimeMs);
}

/// EXHAUSTIVE autotune over every applicable engine of a fresh graph, with no storage file.
Error runAutotune(Session& session, const GraphSource& source, json& autotune)
{
    const auto& options = session.options;
    BenchGraph graph;
    HIPDNN_CHECK_ERROR(restore(session, source, graph));
    HIPDNN_CHECK_ERROR(graph.add_all_engines());
    int64_t workspaceSize = 0;
    HIPDNN_CHECK_ERROR(graph.get_estimated_max_workspace_size(workspaceSize));

    PhaseMemory memory;
    HIPDNN_CHECK_ERROR(
        allocateVariantPack(hipdnn_bench::planVariantPack(graph), memory.buffers, memory.variantPack));
    HIPDNN_CHECK_ERROR(allocateWorkspace(memory, workspaceSize));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = options.warmup;
    config.maxIterations = options.maxIterations;
    config.stabilityThreshold = options.stability;
    // One engine whose priming fails is benchmarked unprimed (its result says so in
    // ran_exhaustive) rather than aborting the baseline for every other engine.
    config.primingFailurePolicy = hipdnn_frontend::PrimingFailurePolicy::BENCHMARK_UNPRIMED;

    std::vector<AutotuneResult> results;
    const auto begin = Clock::now();
    // No storage config: it would persist only the winner.
    const Error tuned = graph.autotune(session.handle,
                                       memory.variantPack,
                                       memory.workspace,
                                       workspaceSize,
                                       config,
                                       {},
                                       &results);
    autotune["wall_s"] = std::chrono::duration<double>(Clock::now() - begin).count();

    const AutotuneResult* best = nullptr;
    json resultsJson = json::array();
    for(const auto& result : results)
    {
        const std::string name
            = result.engineName.empty() ? session.engineName(result.engineId) : result.engineName;
        resultsJson.push_back(
            json{{"engine_id", result.engineId},
                 {"engine_name", name},
                 {"succeeded", result.succeeded},
                 {"avg_time_ms", static_cast<double>(result.avgTimeMs)},
                 {"robust_time_ms", static_cast<double>(result.robustTimeMs)},
                 {"min_time_ms", static_cast<double>(result.minTimeMs)},
                 {"stddev_ms", static_cast<double>(result.stddevMs)},
                 {"iterations", result.iterationsRun},
                 {"converged", result.converged},
                 {"rank", result.rank},
                 {"ran_exhaustive", result.ranExhaustive},
                 {"knobs", knobsJson(result.knobSettings)},
                 {"error", result.errorMessage.empty() ? json(nullptr) : json(result.errorMessage)}});
        if(result.succeeded && comparisonMs(result) > 0.0
           && (best == nullptr || comparisonMs(result) < comparisonMs(*best)))
        {
            best = &result;
        }
    }
    autotune["results"] = std::move(resultsJson);
    if(best != nullptr)
    {
        autotune["best"]
            = {{"engine_id", best->engineId},
               {"engine_name",
                best->engineName.empty() ? session.engineName(best->engineId) : best->engineName},
               {"robust_time_ms", comparisonMs(*best)}};
    }
    if(tuned.is_bad() && results.empty())
    {
        return tuned;
    }
    if(tuned.is_bad())
    {
        // Partial results are kept; the failure is still reported on the line.
        autotune["error"] = tuned.get_message();
    }
    return {};
}

json measureAutotune(Session& session, const GraphSource& source, const Deadline& deadline)
{
    json autotune = {{"mode", "EXHAUSTIVE"},
                     {"wall_s", nullptr},
                     {"results", json::array()},
                     {"best", nullptr},
                     {"error", nullptr}};
    if(deadline.expired())
    {
        autotune["error"] = timeoutMessage(session.options);
        return autotune;
    }
    if(const auto failure
       = guarded([&]() -> Error { return runAutotune(session, source, autotune); }))
    {
        autotune["error"] = *failure;
    }
    return autotune;
}

// ---------------------------------------------------------------------------------------
// One graph
// ---------------------------------------------------------------------------------------

/// The id @p graph carries, as its serialized form names it.
Error graphIdOf(const BenchGraph& graph, std::string& graphId)
{
    json serialized;
    HIPDNN_CHECK_ERROR(graph.serialize(serialized));
    graphId = serialized.at("id").get<std::string>();
    return {};
}

/// Measures one graph into its JSON line. A graph that fails is reported on its line
/// (`is_valid:false`, `skip_reason`); phases that fail keep their own `error` field.
json evaluateGraph(Session& session, const std::string& path)
{
    const auto& options = session.options;
    json line = {{"schema", SCHEMA},
                 {"graph_path", path},
                 {"graph_key", nullptr},
                 {"graph_id", nullptr},
                 {"arch", session.arch},
                 {"device_name", session.deviceName},
                 {"device_id", session.deviceId},
                 {"ranking_metric", options.rankingMetric},
                 {"is_valid", false},
                 {"skip_reason", nullptr},
                 {"timed_out", false},
                 {"engines", json::array()},
                 {"selection", json::object()},
                 {"autotune", nullptr}};
    const Deadline deadline(options.graphTimeoutS);

    try
    {
        GraphSource source;
        if(const auto unread = loadSource(session, path, source))
        {
            line["skip_reason"] = *unread;
            return line;
        }
        line["graph_key"] = source.key;

        // Applicable engines, and the graph the predictions are read from.
        BenchGraph inspected;
        std::vector<hipdnn_frontend::EngineConfigInfo> engines;
        if(const auto failure = guarded([&]() -> Error {
               HIPDNN_CHECK_ERROR(restore(session, source, inspected));
               std::string graphId;
               HIPDNN_CHECK_ERROR(graphIdOf(inspected, graphId));
               line["graph_id"] = graphId;
               return inspected.get_engine_configs(session.handle, engines);
           }))
        {
            line["skip_reason"] = "No engine configurations available for this graph: " + *failure;
            return line;
        }
        if(engines.empty())
        {
            line["skip_reason"] = "No engine is applicable to this graph";
            return line;
        }
        line["is_valid"] = true;
        for(const auto& engine : engines)
        {
            session.engineNames[engine.engineId] = engine.engineName;
            line["engines"].push_back(json{{"engine_id", engine.engineId},
                                           {"engine_name", engine.engineName},
                                           {"prediction", nullptr},
                                           {"immediate", nullptr}});
        }

        // Selection first: the first heuristic query of the process must be the one that pays
        // the model load, and nothing before it may have touched the heuristic models.
        for(const auto& policy : options.policies)
        {
            line["selection"][policy.name] = measurePolicy(session, source, policy, deadline);
        }

        std::optional<double> flops;
        for(auto& entry : line["engines"])
        {
            entry["prediction"]
                = measurePrediction(session, inspected, entry["engine_id"].get<int64_t>(), flops);
        }
        if(flops)
        {
            line["flops"] = *flops;
        }

        if(options.immediate)
        {
            for(auto& entry : line["engines"])
            {
                entry["immediate"]
                    = measureImmediate(session, source, entry["engine_id"].get<int64_t>(), deadline);
            }
        }

        if(options.autotune)
        {
            line["autotune"] = measureAutotune(session, source, deadline);
        }
        line["timed_out"] = deadline.expired();
    }
    catch(const std::exception& exception)
    {
        line["is_valid"] = false;
        line["skip_reason"] = std::string("exception: ") + exception.what();
    }
    catch(...)
    {
        line["is_valid"] = false;
        line["skip_reason"] = "non-standard exception";
    }
    return line;
}

/// Writes one flushed line: a later graph that crashes the process must not take the lines
/// already measured with it.
void emit(const json& line)
{
    std::cout << line.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
}

Error identifyDevice(Session& session)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnGetStream(session.handle, &session.stream),
                                     "Could not get the handle's stream");
    int device = 0;
    HIPDNN_CHECK_ERROR(hipError(session.stream == nullptr
                                    ? hipGetDevice(&device)
                                    : hipStreamGetDevice(session.stream, &device),
                                "Could not resolve the stream's device"));
    hipDeviceProp_t properties{};
    HIPDNN_CHECK_ERROR(
        hipError(hipGetDeviceProperties(&properties, device), "Could not read device properties"));
    const std::string deviceArch = properties.gcnArchName;
    session.arch = deviceArch.substr(0, deviceArch.find(':'));
    session.deviceName = properties.name;
    session.deviceId = hipdnn_bench::deviceKindId(properties);
    return {};
}

int runEval(const std::vector<std::string>& args)
{
    Session session;
    const auto parsed = parseArguments(args, session.options);
    if(parsed != ParseOutcome::RUN)
    {
        return parsed == ParseOutcome::HELP ? 0 : 1;
    }

    std::ifstream listFile(session.options.graphListPath);
    if(!listFile)
    {
        std::cerr << "Cannot read " << session.options.graphListPath << "\n";
        return 1;
    }
    const auto paths = hipdnn_bench::readGraphList(listFile);

    if(!session.options.pluginDirs.empty())
    {
        std::vector<const char*> pluginPaths;
        pluginPaths.reserve(session.options.pluginDirs.size());
        for(const auto& dir : session.options.pluginDirs)
        {
            pluginPaths.push_back(dir.c_str());
        }
        if(hipdnnSetEnginePluginPaths_ext(
               pluginPaths.size(), pluginPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE)
           != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Failed to set engine plugin paths\n";
            return 1;
        }
    }

    if(hipdnnCreate(&session.handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create a hipDNN handle\n";
        return 1;
    }
    const hipdnn_data_sdk::utilities::ScopedResource ownedHandle(session.handle, hipdnnDestroy);

    if(const auto failure = guarded([&]() -> Error { return identifyDevice(session); }))
    {
        std::cerr << "Could not identify the device: " << *failure << "\n";
        return 1;
    }

    for(const auto& path : paths)
    {
        const auto line = evaluateGraph(session, path);
        if(!line.at("is_valid").get<bool>())
        {
            std::cerr << path << ": " << line.at("skip_reason").dump() << "\n";
        }
        emit(line);
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    // Report escaping exceptions instead of terminating without a diagnostic.
    try
    {
        return runEval(std::vector<std::string>(argv, argv + argc));
    }
    catch(const std::exception& error)
    {
        std::cerr << "hipdnn_xengine_eval failed: " << error.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "hipdnn_xengine_eval failed with a non-standard exception\n";
        return 1;
    }
}
