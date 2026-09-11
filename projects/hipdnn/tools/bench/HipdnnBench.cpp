// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBench.cpp
 * @brief Runs one problem against one engine and reports its kernel times (RFC 0019.13 §5.3).
 *
 * The harvest end of corpus generation. A UHD is trained on rows of (problem, configuration)
 * -> time; the generator produces the problems, and this produces the rows.
 *
 * One invocation, one problem, many rows -- one per configuration. That split is deliberate:
 * a corpus is 10^4-10^6 rows, and a process per row would pay plugin load, graph build and
 * kernel compilation for every one of them. Sweeping inside one process amortises all three.
 *
 * Candidate timing and immediate timing share hipDNN's stable-iteration statistics. Immediate
 * collection builds the requested engine's normal plan with global.benchmarking disabled and
 * never invokes autotune or candidate discovery. Prediction and description do not execute it.
 *
 * Two things this deliberately does not do:
 *
 *  - It does not write autotune's result file. That file keeps the rank-0 winner and replaces
 *    matching entries, which is right for a heuristic cache and backwards for training: a
 *    ranking model learns from the candidates that lost.
 *  - It does not tune in EXHAUSTIVE mode. That primes engines via `global.benchmarking`, and
 *    an engine given that knob selects a kernel itself -- so the row would not describe the
 *    configuration this tool pinned.
 */

#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/TimedRunLoop.hpp>
#include <hipdnn_frontend/detail/EngineQueries.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using hipdnn_frontend::AutotuneConfig;
using hipdnn_frontend::AutotuneResult;
using hipdnn_frontend::AutotuneStrategy;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::KnobSetting;
using hipdnn_frontend::TuneMode;

/// Generation-tool access to the finalized backend graph descriptor the engine-inspection
/// attributes are read through. `Graph` keeps that accessor protected for internal and
/// tooling use; re-exporting it in a derived type is the established convention
/// (hipdnn_test_sdk::TestableGraph does the same).
class BenchGraph : public hipdnn_frontend::graph::Graph
{
public:
    using Graph::get_raw_graph_descriptor;
};

std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>
    toVariantKnobs(const std::vector<KnobSetting>& configuration)
{
    std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant> knobs;
    for(const auto& setting : configuration)
    {
        knobs.emplace(setting.knobId(), setting.value());
    }
    return knobs;
}

enum class EngineMode
{
    NONE,
    PREDICT,
    DESCRIBE,
    COLLECT_IMMEDIATE
};

struct Options
{
    std::vector<std::string> pluginDirs;
    std::string graphPath;
    std::string engineName;
    int64_t engineId = 0;
    bool haveEngineId = false;
    bool sweep = false;
    bool header = true;
    bool enumerate = false;
    bool json = false;
    EngineMode engineMode = EngineMode::NONE;
    bool havePageOptions = false;
    int64_t offset = 0;
    int64_t limit = 10000;
    std::vector<std::pair<std::string, int64_t>> knobs;
    int maxIterations = 100;
    /// Ten, not one. Measured: with a single warmup iteration the first timed run of a fresh
    /// process came in at 0.038 ms against a steady state of 0.014 -- kernel compilation
    /// landing inside the timed loop. That contaminates the first problem of every fleet
    /// invocation, and it is invisible in the row: a compile is just a slower number.
    int warmup = 10;
    float stability = 0.05F;
    std::string problemId;

    /// The problem's declared parameters, as `name=value` pairs. Carried on the command line
    /// rather than looked up, so a row is complete on its own: a timing whose q.* values live
    /// in another file is a timing that can be joined to the wrong problem.
    std::vector<std::pair<std::string, std::string>> query;
};

void printHelp(const char* program)
{
    std::cout
        << "Usage: " << program << " [enumerate] --graph <file> --engine-name <name> [options]\n\n"
        << "  --graph <file>         Serialized problem graph (JSON or FlatBuffer)\n"
        << "  --engine-name <name>   Engine under test, e.g. hipkernel:ConvFwd\n"
        << "  --engine-id <id>       Same, by id; decimal or 0x-prefixed hex\n"
        << "  --plugin-dir <dir>     Engine plugin directory (repeatable)\n"
        << "  enumerate              Emit a matched-catalog JSON page; do not benchmark\n"
        << "  --predict-engine       Evaluate the engine-level TFLOPS prediction as JSON\n"
        << "  --describe-engine-prediction  Describe binding/features without model evaluation\n"
        << "  --collect-immediate    Time the requested engine without tuning; emit JSON\n"
        << "  --workspace-limit <n>  Set global.workspace_size_limit in bytes\n"
        << "  --offset <n>           Enumeration page offset (default 0)\n"
        << "  --limit <n>            Enumeration page size (1..10000, default 10000)\n"
        << "  --json                 Emit measured candidate identity/features as JSON\n"
        << "  --sweep                Time every matched catalog candidate, not Cartesian guesses\n"
        << "  --knob <name=value>    Pin one knob, or restrict --sweep (repeatable)\n"
        << "  --max-iterations <n>   Ceiling for the stability loop (default 100)\n"
        << "  --warmup <n>           Untimed iterations before timing (default 10)\n"
        << "  --stability <f>        Coefficient-of-variation threshold (default 0.05)\n"
        << "  --problem-id <s>       Value for the problem column; defaults to the path\n"
        << "  --query <k=v,...>      Declared problem parameters, emitted as q.* columns\n"
        << "  --no-header            Omit the CSV header, for concatenating runs\n";
}

bool parseArguments(const std::vector<std::string>& args, Options& options)
{
    for(size_t i = 1; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        const auto next = [&]() { return (i + 1 < args.size()) ? args[++i] : std::string(); };

        if(arg == "--help" || arg == "-h")
        {
            printHelp(args[0].c_str());
            return false;
        }
        if(i == 1 && arg == "enumerate")
        {
            options.enumerate = true;
            options.json = true;
            continue;
        }
        if(arg == "--graph")
        {
            options.graphPath = next();
        }
        else if(arg == "--engine-name")
        {
            options.engineName = next();
            options.engineId = hipdnn_data_sdk::utilities::engineNameToId(options.engineName);
            options.haveEngineId = true;
        }
        else if(arg == "--engine-id")
        {
            options.engineId = static_cast<int64_t>(std::strtoull(next().c_str(), nullptr, 0));
            options.haveEngineId = true;
        }
        else if(arg == "--plugin-dir")
        {
            options.pluginDirs.push_back(next());
        }
        else if(arg == "--predict-engine" || arg == "--describe-engine-prediction"
                || arg == "--collect-immediate")
        {
            if(options.engineMode != EngineMode::NONE)
            {
                throw std::invalid_argument("Engine prediction/description/collection modes "
                                            "are mutually exclusive");
            }
            options.engineMode = arg == "--predict-engine" ? EngineMode::PREDICT
                                 : arg == "--describe-engine-prediction"
                                     ? EngineMode::DESCRIBE
                                     : EngineMode::COLLECT_IMMEDIATE;
            options.json = true;
        }
        else if(arg == "--json")
        {
            options.json = true;
        }
        else if(arg == "--offset" || arg == "--limit")
        {
            const auto value = next();
            options.havePageOptions = true;
            size_t consumed = 0;
            const auto parsed = std::stoll(value, &consumed);
            if(consumed != value.size() || parsed < 0)
            {
                throw std::invalid_argument("Invalid candidate page offset/limit");
            }
            (arg == "--offset" ? options.offset : options.limit) = parsed;
        }
        else if(arg == "--sweep")
        {
            options.sweep = true;
        }
        else if(arg == "--knob" || arg == "--workspace-limit")
        {
            const auto setting
                = arg == "--workspace-limit" ? "global.workspace_size_limit=" + next() : next();
            const auto split = setting.find('=');
            if(split == std::string::npos)
            {
                std::cerr << "--knob expects name=value, got '" << setting << "'\n";
                return false;
            }
            const auto name = setting.substr(0, split);
            const auto text = setting.substr(split + 1);
            size_t consumed = 0;
            const auto value = std::stoll(text, &consumed);
            if(name.empty() || consumed != text.size()
               || std::any_of(options.knobs.begin(),
                              options.knobs.end(),
                              [&name](const auto& item) { return item.first == name; }))
            {
                throw std::invalid_argument("Invalid or duplicate knob setting '" + setting + "'");
            }
            options.knobs.emplace_back(name, value);
            if(name == "global.workspace_size_limit" && value < 0)
            {
                throw std::invalid_argument("Workspace limit must be nonnegative");
            }
        }
        else if(arg == "--max-iterations")
        {
            options.maxIterations = static_cast<int>(std::strtol(next().c_str(), nullptr, 10));
        }
        else if(arg == "--warmup")
        {
            options.warmup = static_cast<int>(std::strtol(next().c_str(), nullptr, 10));
        }
        else if(arg == "--stability")
        {
            options.stability = std::strtof(next().c_str(), nullptr);
        }
        else if(arg == "--query")
        {
            std::stringstream fields(next());
            std::string field;
            while(std::getline(fields, field, ','))
            {
                const auto split = field.find('=');
                if(split == std::string::npos)
                {
                    std::cerr << "--query expects name=value pairs, got '" << field << "'\n";
                    return false;
                }
                options.query.emplace_back(field.substr(0, split), field.substr(split + 1));
            }
        }
        else if(arg == "--problem-id")
        {
            options.problemId = next();
        }
        else if(arg == "--no-header")
        {
            options.header = false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printHelp(args[0].c_str());
            return false;
        }
    }
    if(options.limit < 1 || options.limit > 10000)
    {
        throw std::invalid_argument("--limit must be in [1, 10000]");
    }
    if(options.engineMode != EngineMode::NONE)
    {
        if(options.sweep || options.enumerate || options.havePageOptions)
        {
            throw std::invalid_argument("Engine prediction/description/collection modes cannot "
                                        "be combined with sweep or candidate enumeration");
        }
        const auto benchmarking
            = std::find_if(options.knobs.begin(), options.knobs.end(), [](const auto& setting) {
                  return setting.first == hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME;
              });
        if(benchmarking == options.knobs.end())
        {
            options.knobs.emplace_back(hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME,
                                       int64_t{0});
        }
        else if(benchmarking->second != 0)
        {
            throw std::invalid_argument("Engine prediction/description/collection requires "
                                        "global.benchmarking=0");
        }
        if(options.engineMode == EngineMode::COLLECT_IMMEDIATE
           && (options.warmup < 1 || options.maxIterations < AutotuneConfig{}.windowSize
               || !std::isfinite(options.stability) || options.stability <= 0.0F
               || options.stability >= 1.0F))
        {
            throw std::invalid_argument(
                "Immediate collection requires positive warmup, "
                "0 < stability < 1, and max-iterations >= the stability window");
        }
    }
    return true;
}

/// Device memory for the duration of the run. Not a general allocator -- it exists so the
/// buffers outlive execute() and are released even when a measurement fails.
class DeviceBuffers
{
public:
    ~DeviceBuffers()
    {
        for(void* pointer : _pointers)
        {
            (void)hipFree(pointer);
        }
    }
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    DeviceBuffers(DeviceBuffers&&) = delete;
    DeviceBuffers& operator=(DeviceBuffers&&) = delete;

    /// Allocates and zero-fills. Filled rather than left as whatever the device held: on some
    /// hardware denormals and NaNs read from uninitialised memory are slower than normal
    /// values, and a corpus row that recorded that would be measuring the allocator.
    void* add(int64_t bytes)
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
        _pointers.push_back(pointer);
        return pointer;
    }

private:
    std::vector<void*> _pointers;
};

hipdnn_frontend::Error allocateVariantPack(const hipdnn_frontend::graph::Graph& graph,
                                           DeviceBuffers& buffers,
                                           std::unordered_map<int64_t, void*>& variantPack)
{
    const auto plan = hipdnn_bench::planVariantPack(graph);
    if(!plan.error.empty())
    {
        return {hipdnn_frontend::ErrorCode::INVALID_VALUE, plan.error};
    }
    for(const auto& tensor : plan.tensors)
    {
        void* pointer = buffers.add(tensor.bytes);
        if(pointer == nullptr)
        {
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Out of device memory for '" + tensor.name + "' ("
                        + std::to_string(tensor.bytes) + " bytes)"};
        }
        variantPack[tensor.uid] = pointer;
    }
    return {};
}

/// One knob's value as text.
std::string knobValue(const KnobSetting& setting)
{
    std::ostringstream stream;
    std::visit([&stream](const auto& value) { stream << value; }, setting.value());
    return stream.str();
}

/// Every knob name any variant sets, sorted.
///
/// Collected across all results rather than from the first, because a variant may omit a knob
/// it left at its default; a header taken from one row would then shift the columns of another.
std::vector<std::string> kernelColumns(const std::vector<AutotuneResult>& results)
{
    std::set<std::string> names;
    for(const auto& result : results)
    {
        for(const auto& setting : result.knobSettings)
        {
            names.insert(setting.knobId());
        }
    }
    return {names.begin(), names.end()};
}

/// The value @p result gives @p knob, or empty when it did not set it.
std::string knobFor(const AutotuneResult& result, const std::string& knob)
{
    for(const auto& setting : result.knobSettings)
    {
        if(setting.knobId() == knob)
        {
            return knobValue(setting);
        }
    }
    return {};
}

nlohmann::json
    knobJson(const std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>& knobs)
{
    nlohmann::json result = nlohmann::json::object();
    for(const auto& entry : knobs)
    {
        std::visit([&result, &entry](const auto& value) { result[entry.first] = value; },
                   entry.second);
    }
    return result;
}

nlohmann::json pageJson(const hipdnn_frontend::EngineCandidatePage& page)
{
    nlohmann::json candidates = nlohmann::json::array();
    for(const auto& candidate : page.candidates)
    {
        candidates.push_back({{"id", candidate.id},
                              {"knob_settings", knobJson(candidate.variant.knobSettings)},
                              {"kernel_features", candidate.kernelFeatures}});
    }
    return {{"engine_id", page.engineId},
            {"graph_id", page.graphId},
            {"engine_name", page.engineName},
            {"engine_descriptor_id", page.engineDescriptorId},
            {"device_id", page.deviceId},
            {"device_arch", page.deviceArch},
            {"problem_features", page.problemFeatures},
            {"device_features", page.deviceFeatures},
            {"candidates", std::move(candidates)},
            {"total_count", page.totalCount},
            {"offset", page.offset},
            {"next_offset",
             page.nextOffset ? nlohmann::json(*page.nextOffset) : nlohmann::json(nullptr)}};
}

hipdnn_frontend::Error hipError(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                std::string(operation) + ": " + hipGetErrorString(status)};
    }
    return {};
}

hipdnn_frontend::Error engineIdentity(hipdnnHandle_t handle,
                                      const hipdnn_frontend::graph::Graph& graph,
                                      int64_t engineId,
                                      hipStream_t& stream,
                                      nlohmann::json& output)
{
    nlohmann::json serializedGraph;
    HIPDNN_CHECK_ERROR(graph.serialize(serializedGraph));
    output["graph_id"] = serializedGraph.at("id").get<std::string>();

    size_t nameSize = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnGetEngineNameById_ext(handle, engineId, nullptr, &nameSize),
        "Could not query the requested engine's name");
    if(nameSize == 0)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR, "Engine name is empty"};
    }
    std::vector<char> name(nameSize);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnGetEngineNameById_ext(handle, engineId, name.data(), &nameSize),
        "Could not read the requested engine's name");
    output["engine_name"] = std::string(name.data());

    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnGetStream(handle, &stream),
                                     "Could not get the handle's stream");
    int device = 0;
    HIPDNN_CHECK_ERROR(
        hipError(stream == nullptr ? hipGetDevice(&device) : hipStreamGetDevice(stream, &device),
                 "Could not resolve the stream's device"));
    hipDeviceProp_t properties{};
    HIPDNN_CHECK_ERROR(
        hipError(hipGetDeviceProperties(&properties, device), "Could not read device properties"));
    const std::string deviceArch = properties.gcnArchName;
    output["arch"] = deviceArch.substr(0, deviceArch.find(':'));
    output["device_arch"] = deviceArch;
    output["device_name"] = properties.name;
    output["device_ordinal"] = device;

    hipUUID uuid{};
    HIPDNN_CHECK_ERROR(hipError(hipDeviceGetUuid(&uuid, device), "Could not read device UUID"));
    static constexpr char HEX[] = "0123456789abcdef";
    std::string deviceId;
    deviceId.reserve(sizeof(uuid.bytes) * 2);
    for(const auto byte : uuid.bytes)
    {
        const auto value = static_cast<unsigned char>(byte);
        deviceId.push_back(HEX[value >> 4]);
        deviceId.push_back(HEX[value & 0x0f]);
    }
    output["device_id"] = std::move(deviceId);
    return {};
}

const char* predictionStatus(hipdnn_frontend::PredictionStatus status)
{
    switch(status)
    {
    case hipdnn_frontend::PredictionStatus::AVAILABLE:
        return "available";
    case hipdnn_frontend::PredictionStatus::INVALID:
        return "invalid";
    case hipdnn_frontend::PredictionStatus::UNAVAILABLE:
        return "unavailable";
    default:
        return "invalid";
    }
}

hipdnn_frontend::Error collectImmediate(hipdnnHandle_t handle,
                                        hipdnn_frontend::graph::Graph& graph,
                                        const Options& options,
                                        const std::vector<KnobSetting>& settings,
                                        hipStream_t stream,
                                        nlohmann::json& output)
{
    HIPDNN_CHECK_ERROR(graph.create_execution_plan_ext(options.engineId, settings));
    HIPDNN_CHECK_ERROR(graph.build_plans());

    DeviceBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    HIPDNN_CHECK_ERROR(allocateVariantPack(graph, buffers, variantPack));
    int64_t workspaceSize = 0;
    HIPDNN_CHECK_ERROR(graph.get_workspace_size(workspaceSize));
    output["workspace_bytes"] = workspaceSize;
    void* workspace = workspaceSize > 0 ? buffers.add(workspaceSize) : nullptr;
    if(workspaceSize > 0 && workspace == nullptr)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                "Out of device memory for a " + std::to_string(workspaceSize) + " byte workspace"};
    }

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
        output["warmup_iterations"] = w + 1;
    }
    HIPDNN_CHECK_ERROR(hipError(hipStreamSynchronize(stream), "Warmup synchronization failed"));

    const auto timeOnce = [&](float& elapsed) -> hipdnn_frontend::Error {
        HIPDNN_CHECK_ERROR(hipError(hipEventRecord(start.get(), stream), "Could not start timing"));
        HIPDNN_CHECK_ERROR(graph.execute(handle, variantPack, workspace));
        HIPDNN_CHECK_ERROR(hipError(hipEventRecord(stop.get(), stream), "Could not stop timing"));
        HIPDNN_CHECK_ERROR(
            hipError(hipEventSynchronize(stop.get()), "Timing synchronization failed"));
        HIPDNN_CHECK_ERROR(hipError(hipEventElapsedTime(&elapsed, start.get(), stop.get()),
                                    "Could not read timing"));
        if(!std::isfinite(elapsed) || elapsed <= 0.0F)
        {
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                    "HIP event timing must be finite and positive"};
        }
        return {};
    };
    const auto outcome
        = hipdnn_frontend::autotune::detail::runUntilStable(options.maxIterations,
                                                            AutotuneConfig{}.windowSize,
                                                            options.stability,
                                                            timeOnce,
                                                            [](int, float, float, bool) {});
    output["iterations"] = outcome.timings.size();
    output["converged"] = outcome.converged;
    if(outcome.benchmarkFailed)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR, outcome.errorMessage};
    }
    output["robustMeanMs"] = hipdnn_data_sdk::utilities::detail::robustMean(outcome.timings);
    output["min_time_ms"] = *std::min_element(outcome.timings.begin(), outcome.timings.end());
    output["avg_time_ms"] = hipdnn_data_sdk::utilities::detail::mean(outcome.timings);
    output["stddev_ms"] = hipdnn_data_sdk::utilities::detail::stddev(outcome.timings);
    output["is_valid"] = true; // Measured successfully, not a numerical correctness assertion.
    return {};
}

int runEngineMode(hipdnnHandle_t handle, BenchGraph& graph, const Options& options)
{
    std::vector<KnobSetting> settings;
    settings.reserve(options.knobs.size());
    for(const auto& [name, value] : options.knobs)
    {
        settings.emplace_back(name, value);
    }
    const bool collect = options.engineMode == EngineMode::COLLECT_IMMEDIATE;
    const bool evaluate = options.engineMode == EngineMode::PREDICT;
    // User knob constraints describe an exact configuration, so they select the kind:
    // an engine-level estimate is by definition unconstrained. The benchmarking pin this
    // tool adds for its own execution is not such a constraint and is never queried with:
    // the backend requires an exact-configuration prediction to preserve every requested
    // knob, and an engine whose catalog has no global.benchmarking knob cannot.
    std::vector<KnobSetting> queryConstraints;
    for(const auto& [name, value] : options.knobs)
    {
        if(name != hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME)
        {
            queryConstraints.emplace_back(name, value);
        }
    }
    const auto kind = queryConstraints.empty() ? hipdnn_frontend::PredictionKind::ENGINE
                                               : hipdnn_frontend::PredictionKind::CONFIGURATION;
    nlohmann::json output
        = {{"engine_id", options.engineId},
           {"is_valid", false},
           {"prediction_kind",
            kind == hipdnn_frontend::PredictionKind::ENGINE ? "ENGINE" : "CONFIGURATION"},
           {"evaluate", evaluate},
           {"constraints", knobJson(toVariantKnobs(settings))}};
    if(collect)
    {
        output.update({{"selection_mode", "immediate"},
                       {"timing_statistic", "robustMeanMs"},
                       {"robustMeanMs", nullptr},
                       {"warmup_iterations", 0},
                       {"iterations", 0},
                       {"converged", false},
                       {"max_iterations", options.maxIterations},
                       {"stability_window", AutotuneConfig{}.windowSize},
                       {"stability_threshold", options.stability}});
    }
    // Generation-tool surface: the descriptor attributes are driven directly, because
    // engine inspection is not part of the consumer Graph API.
    hipdnn_frontend::EnginePrediction description;
    auto error = hipdnn_frontend::detail::getEnginePrediction(graph.get_raw_graph_descriptor(),
                                                              options.engineId,
                                                              description,
                                                              kind,
                                                              /*evaluate=*/false,
                                                              queryConstraints);
    hipdnn_frontend::EnginePrediction prediction;
    if(error.is_good())
    {
        // Evaluated predictions omit metadata on the policy hot path. The CLI publishes the
        // authoritative description with the same constraints alongside the evaluated score.
        output["binding"] = std::move(description.binding);
        output["features"] = std::move(description.features);
        if(evaluate)
        {
            error = hipdnn_frontend::detail::getEnginePrediction(graph.get_raw_graph_descriptor(),
                                                                 options.engineId,
                                                                 prediction,
                                                                 kind,
                                                                 /*evaluate=*/true,
                                                                 queryConstraints);
        }
        else
        {
            prediction = std::move(description);
        }
    }
    hipStream_t stream = nullptr;
    if(error.is_good())
    {
        output["status"] = predictionStatus(prediction.status);
        output["model"] = prediction.model;
        output["reason"] = prediction.reason;
        if(!collect)
        {
            output["tflops"]
                = prediction.tflops ? nlohmann::json(*prediction.tflops) : nlohmann::json(nullptr);
        }
        error = engineIdentity(handle, graph, options.engineId, stream, output);
    }
    if(error.is_good())
    {
        if(collect)
        {
            error = collectImmediate(handle, graph, options, settings, stream, output);
        }
        else
        {
            output["is_valid"]
                = evaluate ? prediction.status == hipdnn_frontend::PredictionStatus::AVAILABLE
                           : prediction.status != hipdnn_frontend::PredictionStatus::INVALID;
        }
    }
    if(error.is_bad())
    {
        output["skip_reason"] = error.get_message();
        std::cerr << "Engine query/collection failed: " << error.get_message() << "\n";
    }
    std::cout << output.dump() << "\n";
    return error.is_good() ? 0 : 1;
}

int runBench(const std::vector<std::string>& args)
{

    Options options;
    if(!parseArguments(args, options))
    {
        return std::find(args.begin(), args.end(), "--help") != args.end()
                       || std::find(args.begin(), args.end(), "-h") != args.end()
                   ? 0
                   : 1;
    }
    if(options.graphPath.empty() || !options.haveEngineId)
    {
        std::cerr << "--graph and --engine-name (or --engine-id) are required\n";
        return 1;
    }

    std::ifstream graphFile(options.graphPath, std::ios::binary);
    if(!graphFile)
    {
        std::cerr << "Cannot read " << options.graphPath << "\n";
        return 1;
    }
    const std::vector<uint8_t> graphBytes((std::istreambuf_iterator<char>(graphFile)),
                                          std::istreambuf_iterator<char>());
    if(graphBytes.empty())
    {
        std::cerr << options.graphPath << " is empty\n";
        return 1;
    }

    if(!options.pluginDirs.empty())
    {
        std::vector<const char*> paths;
        paths.reserve(options.pluginDirs.size());
        for(const auto& dir : options.pluginDirs)
        {
            paths.push_back(dir.c_str());
        }
        if(hipdnnSetEnginePluginPaths_ext(
               paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE)
           != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Failed to set engine plugin paths\n";
            return 1;
        }
    }

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create a hipDNN handle\n";
        return 1;
    }
    const hipdnn_data_sdk::utilities::ScopedResource ownedHandle(handle, hipdnnDestroy);

    // A problem file is a serialized graph in either form: the generator writes the binary
    // FlatBuffers a builder produces, while a hand-made or exported problem is often JSON.
    // Distinguished by content rather than by extension, so a renamed file still loads.
    BenchGraph graph;
    const bool looksLikeJson = graphBytes.front() == static_cast<uint8_t>('{');
    const auto restored = [&]() -> Error {
        if(looksLikeJson)
        {
            return graph.deserialize(handle, std::string(graphBytes.begin(), graphBytes.end()));
        }
        if(options.engineMode == EngineMode::NONE)
        {
            return graph.deserialize(handle, graphBytes);
        }

        // L1 consumes only the graph: discard any embedded plan before attaching a handle.
        // Re-serialize the restored graph rather than re-lowering its nodes, preserving its ID.
        HIPDNN_CHECK_ERROR(graph.deserialize(graphBytes));
        std::vector<uint8_t> problem;
        HIPDNN_CHECK_ERROR(graph.serialize(problem));
        return graph.deserialize(handle, problem);
    }();
    if(!restored.is_good())
    {
        std::cerr << "Could not read " << options.graphPath << ": " << restored.get_message()
                  << "\n";
        return 1;
    }
    if(options.engineMode != EngineMode::NONE)
    {
        return runEngineMode(handle, graph, options);
    }

    std::vector<KnobSetting> pinned;
    pinned.reserve(options.knobs.size());
    for(const auto& [name, value] : options.knobs)
    {
        pinned.emplace_back(name, value);
    }
    hipdnn_frontend::EngineCandidatePage catalog;
    const bool useCatalog = options.enumerate || options.json || options.sweep;
    if(useCatalog)
    {
        const auto discovery
            = hipdnn_frontend::detail::getEngineCandidates(graph.get_raw_graph_descriptor(),
                                                           options.engineId,
                                                           catalog,
                                                           options.enumerate ? options.offset : 0,
                                                           options.limit,
                                                           pinned);
        if(discovery.is_bad())
        {
            std::cerr << "Candidate enumeration failed: " << discovery.get_message() << "\n";
            return 1;
        }
        if(options.enumerate)
        {
            std::cout << pageJson(catalog).dump() << "\n";
            return 0;
        }
        while(catalog.nextOffset)
        {
            hipdnn_frontend::EngineCandidatePage next;
            const auto discoveryNext = hipdnn_frontend::detail::getEngineCandidates(
                graph.get_raw_graph_descriptor(),
                options.engineId,
                next,
                static_cast<int64_t>(*catalog.nextOffset),
                options.limit,
                pinned);
            if(discoveryNext.is_bad() || next.graphId != catalog.graphId
               || next.deviceId != catalog.deviceId || next.deviceArch != catalog.deviceArch
               || next.totalCount != catalog.totalCount
               || next.problemFeatures != catalog.problemFeatures
               || next.deviceFeatures != catalog.deviceFeatures
               || next.engineName != catalog.engineName
               || next.engineDescriptorId != catalog.engineDescriptorId)
            {
                std::cerr << "Candidate enumeration snapshot changed or failed: "
                          << discoveryNext.get_message() << "\n";
                return 1;
            }
            catalog.nextOffset = next.nextOffset;
            for(auto& candidate : next.candidates)
            {
                catalog.candidates.push_back(std::move(candidate));
            }
        }
        std::set<std::string> ids;
        std::set<std::string> tuples;
        for(const auto& candidate : catalog.candidates)
        {
            if(!ids.insert(candidate.id).second
               || !tuples.insert(knobJson(candidate.variant.knobSettings).dump()).second)
            {
                std::cerr << "Ambiguous candidate identity across enumeration pages\n";
                return 1;
            }
        }
        if(!options.sweep
           && (catalog.candidates.size() != 1
               || catalog.candidates.front().variant.knobSettings != toVariantKnobs(pinned)))
        {
            std::cerr << "--json timing requires a complete explicit candidate knob tuple "
                         "(or --sweep for all matched candidates)\n";
            return 1;
        }
    }

    DeviceBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    const auto allocated = allocateVariantPack(graph, buffers, variantPack);
    if(allocated.is_bad())
    {
        std::cerr << allocated.get_message() << "\n";
        return 1;
    }

    std::vector<hipdnn_frontend::EngineConfigInfo> engines;
    if(!graph.get_engine_configs(handle, engines).is_good())
    {
        std::cerr << "No engine configurations available for this graph\n";
        return 1;
    }

    const auto engine = std::find_if(
        engines.begin(), engines.end(), [&](const hipdnn_frontend::EngineConfigInfo& candidate) {
            return candidate.engineId == options.engineId;
        });
    if(engine == engines.end())
    {
        std::cerr << "Engine is not applicable to this problem\n";
        return 1;
    }

    std::vector<hipdnn_frontend::EngineVariant> variants;
    if(useCatalog)
    {
        for(const auto& candidate : catalog.candidates)
        {
            variants.push_back(candidate.variant);
        }
    }
    else
    {
        variants.push_back({options.engineId, toVariantKnobs(pinned)});
    }

    if(!graph.add_engine_variants(variants).is_good())
    {
        std::cerr << "Could not build plan specs for the requested configurations\n";
        return 1;
    }

    int64_t workspaceSize = 0;
    (void)graph.get_estimated_max_workspace_size(workspaceSize);
    void* workspace = nullptr;
    if(workspaceSize > 0)
    {
        workspace = buffers.add(workspaceSize);
        if(workspace == nullptr)
        {
            std::cerr << "Out of device memory for a " << workspaceSize << " byte workspace\n";
            return 1;
        }
    }

    AutotuneConfig config;
    // STANDARD, not EXHAUSTIVE: see the file comment. An engine primed with the benchmarking
    // knob picks its own kernel, and the row would then not describe the pinned configuration.
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = options.warmup;
    config.maxIterations = options.maxIterations;
    config.stabilityThreshold = options.stability;
    config.engineIdFilter = {options.engineId};

    std::vector<AutotuneResult> results;
    // Storage config deliberately left default (no file): it persists only the winner.
    const auto tuned
        = graph.autotune(handle, variantPack, workspace, workspaceSize, config, {}, &results);
    if(!tuned.is_good() && results.empty())
    {
        std::cerr << "Benchmarking failed: " << tuned.get_message() << "\n";
        return 1;
    }

    const std::string problemId = options.problemId.empty() ? options.graphPath : options.problemId;

    if(options.json)
    {
        auto output = pageJson(catalog);
        output.erase("candidates");
        output["problem"] = problemId;
        output["results"] = nlohmann::json::array();
        for(const auto& result : results)
        {
            const auto tuple = toVariantKnobs(result.knobSettings);
            const auto candidate = std::find_if(
                catalog.candidates.begin(), catalog.candidates.end(), [&tuple](const auto& item) {
                    return item.variant.knobSettings == tuple;
                });
            if(candidate == catalog.candidates.end())
            {
                std::cerr << "Measured configuration was not an enrolled catalog candidate\n";
                return 1;
            }
            // Preserve existing is_valid semantics: measured, not numerical correctness.
            const bool timed = result.succeeded && result.iterationsRun > 0;
            const std::string reason
                = !result.succeeded
                      ? "config_not_applicable: engine declined or failed to run this configuration"
                  : result.iterationsRun == 0
                      ? "not_timed: autotune reported success without running an iteration"
                      : "";
            output["results"].push_back({{"candidate_id", candidate->id},
                                         {"knob_settings", knobJson(tuple)},
                                         {"kernel_features", candidate->kernelFeatures},
                                         {"rank", result.rank},
                                         {"succeeded", result.succeeded},
                                         {"is_valid", timed},
                                         {"skip_reason", reason},
                                         {"min_time_ms", result.minTimeMs},
                                         {"avg_time_ms", result.avgTimeMs},
                                         {"robust_time_ms", result.robustTimeMs},
                                         {"stddev_ms", result.stddevMs},
                                         {"iterations", result.iterationsRun},
                                         {"converged", result.converged},
                                         {"workspace_bytes", result.workspaceSize}});
        }
        std::cout << output.dump() << "\n";
        return results.empty() ? 2 : 0;
    }

    // One column per feature, not a blob. RFC 0019.13 §7 keys a row on q.* and kernel.*, and
    // uhd_gen hashes the header as the features signature -- so the header a harvest emits is
    // the contract the model is trained against, and a `kernel.config` field packing several
    // knobs into one string cannot be read as features at all.
    const auto kernelNames = kernelColumns(results);

    if(options.header)
    {
        std::cout << "problem";
        for(const auto& entry : options.query)
        {
            std::cout << ",q." << entry.first;
        }
        for(const auto& knob : kernelNames)
        {
            std::cout << ",kernel." << knob;
        }
        // RFC 0019.13 §8.3 names the timing columns of the result envelope --
        // `minTimeMs`, `avgTimeMs`, `stddevMs`, `iters` -- and reads them by name, so
        // this header spells them that way and not in snake_case. `robustMeanMs` is not
        // one of §8.3's columns but is the name `export-benchmarks` and the uhd_gen
        // corpus already use for the same statistic; a second spelling for it would
        // make a harvested CSV unreadable by `uhd_gen evaluate --target robustMeanMs`.
        std::cout << ",engine,rank,succeeded,is_valid,skip_reason,minTimeMs,avgTimeMs,"
                     "robustMeanMs,stddevMs,iters,converged,workspace_bytes\n";
    }

    // Every variant is emitted, including the ones that lost and the ones that failed. A
    // ranking model is trained on the comparison, so a corpus of winners teaches it nothing;
    // and a configuration that cannot run is a fact about the engine worth keeping.
    //
    // Read `rank` as advisory and train on the times. Configurations are routinely separated
    // by less than run-to-run variation -- on gfx1100 the two conv block sizes came within
    // 0.3% of each other and their order flipped between identical runs at every stability
    // threshold tried, including an exact tie. That is not noise to be tightened away; there
    // is no difference there to resolve. A model fitted to the winner would be fitting the
    // coin flip, which is why RFC 0019.13 §5.6 ranks on per-problem normalised time and why
    // `stddevMs` is emitted beside every measurement rather than folded into it.
    for(const auto& result : results)
    {
        std::cout << problemId;
        for(const auto& entry : options.query)
        {
            std::cout << "," << entry.second;
        }
        for(const auto& knob : kernelNames)
        {
            std::cout << "," << knobFor(result, knob);
        }
        // RFC 0019.13 §7.4 / §8: a pair that was not timed is written with is_valid=False and
        // a populated skip_reason rather than dropped. Pre-filtering saves benchmark time and
        // destroys the record of what was filtered, which is the record coverage auditing
        // needs -- "which variants were never eligible, and why" is unanswerable from a file
        // containing only the ones that ran. Training excludes them by filtering on is_valid.
        const bool timed = result.succeeded && result.iterationsRun > 0;
        std::string skipReason;
        if(!result.succeeded)
        {
            skipReason = "config_not_applicable: engine declined or failed to run this "
                         "configuration";
        }
        else if(result.iterationsRun == 0)
        {
            // Reported as a success with nothing measured. Emitting it as valid would put a
            // zero time in the training set, which reads as an infinitely fast kernel.
            skipReason = "not_timed: autotune reported success without running an iteration";
        }

        std::cout << "," << (options.engineName.empty() ? result.engineName : options.engineName)
                  << "," << result.rank << "," << (result.succeeded ? 1 : 0) << ","
                  << (timed ? "True" : "False") << "," << hipdnn_bench::csvField(skipReason) << ","
                  << result.minTimeMs << "," << result.avgTimeMs << "," << result.robustTimeMs
                  << "," << result.stddevMs << "," << result.iterationsRun << ","
                  << (result.converged ? 1 : 0) << "," << result.workspaceSize << "\n";
    }

    return results.empty() ? 2 : 0;
}

} // namespace

int main(int argc, char* argv[])
{
    // The graph, the plugins and the device are all external input; a throw escaping main is
    // a terminate with no diagnostic, which on a fleet is a row that silently never appears.
    try
    {
        return runBench(std::vector<std::string>(argv, argv + argc));
    }
    catch(const std::exception& error)
    {
        std::cerr << "hipdnn_bench failed: " << error.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "hipdnn_bench failed with a non-standard exception\n";
        return 1;
    }
}
