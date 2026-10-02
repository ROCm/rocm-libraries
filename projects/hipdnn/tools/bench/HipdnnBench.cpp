// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBench.cpp
 * @brief Runs one problem against one engine and reports its kernel times (RFC 0019.13 §5.3).
 *
 * One process sweeps many configurations so plugin load, graph build and kernel compilation
 * are paid once. Immediate collection runs the engine's normal plan with global.benchmarking
 * disabled and never autotunes.
 *
 * Does not write autotune's result file (it keeps only the winner; ranking training needs the
 * losers) and does not tune in EXHAUSTIVE mode (engines would then pick their own kernel).
 */

#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_bench/NumericalValidation.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/TimedRunLoop.hpp>
#include <hipdnn_frontend/detail/EngineQueries.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
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

/// Exposes the protected finalized backend descriptor for engine-inspection queries.
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
    /// Metric the prediction is made in and immediate collection selects its kernel by.
    /// A label is only comparable with predictions in its metric.
    std::string rankingMetric{hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC};
    bool haveRankingMetric = false;
    bool havePageOptions = false;
    int64_t offset = 0;
    int64_t limit = 10000;
    std::vector<std::pair<std::string, int64_t>> knobs;
    int maxIterations = 100;
    /// Enough warmup to keep first-run kernel compilation out of the timed loop.
    int warmup = 10;
    float stability = 0.05F;
    std::string problemId;

    /// Declared problem parameters as `name=value` pairs, so each row is self-contained.
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
        << "  --predict-engine       Evaluate the engine-level prediction as JSON\n"
        << "  --describe-engine-prediction  Describe binding/features without model evaluation\n"
        << "  --collect-immediate    Time the requested engine without tuning; emit JSON\n"
        << "  --ranking-metric <m>   Metric to predict in and select the immediate kernel by:\n"
        << "                         tflops (default) or time\n"
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
            if(arg == "--predict-engine")
            {
                options.engineMode = EngineMode::PREDICT;
            }
            else if(arg == "--describe-engine-prediction")
            {
                options.engineMode = EngineMode::DESCRIBE;
            }
            else
            {
                options.engineMode = EngineMode::COLLECT_IMMEDIATE;
            }
            options.json = true;
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
    if(options.haveRankingMetric && options.engineMode == EngineMode::NONE)
    {
        // Swept and enumerated candidates pin every knob, so no ranker chooses among them.
        throw std::invalid_argument("--ranking-metric requires --predict-engine, "
                                    "--describe-engine-prediction or --collect-immediate");
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

/// Variant-pack memory for the duration of the run. Not a general allocator -- it exists so
/// the buffers outlive execute() and are released even when a measurement fails.
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

hipdnn_frontend::Error allocateVariantPack(const hipdnn_bench::VariantPackPlan& plan,
                                           PackBuffers& buffers,
                                           std::unordered_map<int64_t, void*>& variantPack)
{
    if(!plan.error.empty())
    {
        return {hipdnn_frontend::ErrorCode::INVALID_VALUE, plan.error};
    }
    for(const auto& tensor : plan.tensors)
    {
        void* pointer = buffers.add(tensor);
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

std::string knobValue(const KnobSetting& setting)
{
    std::ostringstream stream;
    std::visit([&stream](const auto& value) { stream << value; }, setting.value());
    return stream.str();
}

/// Every knob name any variant sets, sorted. Collected across all results because a variant
/// may omit a knob left at its default.
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
    static constexpr std::string_view HEX = "0123456789abcdef";
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

/// The row's three-valued `numerically_valid`: true, false, or null when not cross-checkable.
nlohmann::json numericallyValid(hipdnn_bench::NumericalVerdict verdict)
{
    switch(verdict)
    {
    case hipdnn_bench::NumericalVerdict::AGREED:
        return true;
    case hipdnn_bench::NumericalVerdict::DISAGREED:
        return false;
    default:
        return nullptr;
    }
}

/// @brief Writes the validation fill into every input buffer of @p variantPack.
///
/// Results keep their zero fill so a kernel that writes nothing is detectable. Types this
/// build cannot encode exactly also keep zeros rather than risk filling NaNs.
hipdnn_frontend::Error fillGraphInputs(const hipdnn_bench::VariantPackPlan& plan,
                                       const std::unordered_map<int64_t, void*>& variantPack,
                                       uint64_t seed)
{
    for(const auto& tensor : plan.tensors)
    {
        const auto buffer = variantPack.find(tensor.uid);
        if(tensor.produced || buffer == variantPack.end())
        {
            continue; // A result, or something not in the pack at all.
        }
        const auto image = hipdnn_bench::detail::inputFillImage(
            tensor.dataType, static_cast<size_t>(tensor.bytes), seed, tensor.uid);
        if(image.empty())
        {
            continue;
        }
        if(tensor.storage == hipdnn_bench::TensorStorage::HOST)
        {
            std::memcpy(buffer->second, image.data(), image.size());
            continue;
        }
        HIPDNN_CHECK_ERROR(
            hipError(hipMemcpy(buffer->second, image.data(), image.size(), hipMemcpyHostToDevice),
                     "Could not fill a candidate's input"));
    }
    return {};
}

/// @brief Runs one candidate once, untimed, and copies back every non-virtual tensor it
///        wrote (hipdnn_bench::crossCheckedOutputs), for the RFC 0019 §13.2 cross-check.
///
/// Uses a fresh graph because create_execution_plan_ext() is refused after
/// add_engine_variants(). Inputs get a per-graph seeded fill, so every candidate of one
/// problem reads identical non-zero bytes.
hipdnn_frontend::Error
    captureCandidateOutput(hipdnnHandle_t handle,
                           const std::vector<uint8_t>& graphBytes,
                           bool looksLikeJson,
                           int64_t engineId,
                           const std::vector<KnobSetting>& settings,
                           std::map<int64_t, hipdnn_bench::TensorDescription>& tensors,
                           std::map<int64_t, std::vector<uint8_t>>& images)
{
    BenchGraph graph;
    HIPDNN_CHECK_ERROR(
        looksLikeJson ? graph.deserialize(handle, std::string(graphBytes.begin(), graphBytes.end()))
                      : graph.deserialize(handle, graphBytes));
    HIPDNN_CHECK_ERROR(graph.create_execution_plan_ext(engineId, settings));
    HIPDNN_CHECK_ERROR(graph.build_plans());

    const auto plan = hipdnn_bench::planVariantPack(graph);
    PackBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    HIPDNN_CHECK_ERROR(allocateVariantPack(plan, buffers, variantPack));
    HIPDNN_CHECK_ERROR(
        fillGraphInputs(plan, variantPack, hipdnn_bench::detail::graphFillSeed(graphBytes)));
    int64_t workspaceSize = 0;
    HIPDNN_CHECK_ERROR(graph.get_workspace_size(workspaceSize));
    void* workspace = workspaceSize > 0 ? buffers.addDevice(workspaceSize) : nullptr;
    if(workspaceSize > 0 && workspace == nullptr)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                "Out of device memory for a " + std::to_string(workspaceSize)
                    + " byte validation workspace"};
    }
    HIPDNN_CHECK_ERROR(graph.execute(handle, variantPack, workspace));
    HIPDNN_CHECK_ERROR(hipError(hipDeviceSynchronize(), "Validation execution did not complete"));

    // Produced tensors are always device-resident: only operand scalars live on the host.
    for(const auto& tensor : hipdnn_bench::crossCheckedOutputs(plan))
    {
        tensors[tensor.uid] = {tensor.name, tensor.dataType};
        std::vector<uint8_t> image(static_cast<size_t>(tensor.bytes));
        HIPDNN_CHECK_ERROR(hipError(
            hipMemcpy(
                image.data(), variantPack.at(tensor.uid), image.size(), hipMemcpyDeviceToHost),
            "Could not read a candidate's output back to the host"));
        images[tensor.uid] = std::move(image);
    }
    return {};
}

hipdnn_frontend::Error collectImmediate(hipdnnHandle_t handle,
                                        hipdnn_frontend::graph::Graph& graph,
                                        const Options& options,
                                        const std::vector<KnobSetting>& settings,
                                        hipStream_t stream,
                                        nlohmann::json& output)
{
    // The engine's ordinary selection for this metric is what an L1 label in it measures.
    HIPDNN_CHECK_ERROR(graph.set_ranking_metric(options.rankingMetric));
    HIPDNN_CHECK_ERROR(graph.create_execution_plan_ext(options.engineId, settings));
    HIPDNN_CHECK_ERROR(graph.build_plans());

    PackBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    HIPDNN_CHECK_ERROR(
        allocateVariantPack(hipdnn_bench::planVariantPack(graph), buffers, variantPack));
    int64_t workspaceSize = 0;
    HIPDNN_CHECK_ERROR(graph.get_workspace_size(workspaceSize));
    output["workspace_bytes"] = workspaceSize;
    void* workspace = workspaceSize > 0 ? buffers.addDevice(workspaceSize) : nullptr;
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

    // Unstalled HIP events: run with stalled=false, so the loop never asks for a restart.
    const auto timeOnce = [&](hipdnn_frontend::ExecutionTiming& timing) -> hipdnn_frontend::Error {
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
            return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
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
    const auto outcome
        = hipdnn_frontend::autotune::detail::runUntilStable(options.maxIterations,
                                                            AutotuneConfig{}.windowSize,
                                                            options.stability,
                                                            /*stalled=*/false,
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
    // `is_valid` means a measurement was obtained, never numerical correctness (RFC 0019 §8.1).
    output["is_valid"] = true;
    // RFC 0019 §13.2: one selection has nothing to cross-check against, so record null rather
    // than omit the field (an absent field may be defaulted to "valid").
    output["numerically_valid"] = nullptr;
    output["validation"] = "no_reference: engine-immediate collection times one selection, "
                           "so there is no second candidate to cross-check it against";
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
    // User knob constraints select CONFIGURATION prediction. The benchmarking pin this tool
    // adds is excluded: engines without that knob could not honour it.
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
           {"metric", options.rankingMetric},
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
    // Engine inspection is not part of the consumer Graph API, so use the descriptor directly.
    hipdnn_frontend::EnginePrediction description;
    auto error = hipdnn_frontend::detail::getEnginePrediction(graph.get_raw_graph_descriptor(),
                                                              options.engineId,
                                                              description,
                                                              kind,
                                                              /*evaluate=*/false,
                                                              queryConstraints,
                                                              options.rankingMetric);
    hipdnn_frontend::EnginePrediction prediction;
    if(error.is_good())
    {
        // Evaluated predictions omit metadata, so publish the unevaluated description too.
        output["binding"] = std::move(description.binding);
        output["features"] = std::move(description.features);
        if(evaluate)
        {
            error = hipdnn_frontend::detail::getEnginePrediction(graph.get_raw_graph_descriptor(),
                                                                 options.engineId,
                                                                 prediction,
                                                                 kind,
                                                                 /*evaluate=*/true,
                                                                 queryConstraints,
                                                                 options.rankingMetric);
        }
        else
        {
            prediction = std::move(description);
        }
    }
    if(error.is_good() && options.engineMode == EngineMode::DESCRIBE)
    {
        // Which other (kind, metric) pairs this engine could answer, so a caller asking in
        // one metric learns about the rest without guessing.
        std::vector<hipdnn_frontend::PredictionCapability> capabilities;
        error = hipdnn_frontend::detail::getPredictionCapabilities(
            graph.get_raw_graph_descriptor(), options.engineId, capabilities);
        nlohmann::json published = nlohmann::json::array();
        for(const auto& capability : capabilities)
        {
            published.push_back(nlohmann::json{
                {"prediction_kind",
                 capability.kind == hipdnn_frontend::PredictionKind::ENGINE ? "ENGINE"
                                                                            : "CONFIGURATION"},
                {"metric", capability.metric},
                {"model", capability.model}});
        }
        output["capabilities"] = std::move(published);
    }
    hipStream_t stream = nullptr;
    if(error.is_good())
    {
        output["status"] = predictionStatus(prediction.status);
        output["model"] = prediction.model;
        output["reason"] = prediction.reason;
        if(!collect)
        {
            output["value"]
                = prediction.value ? nlohmann::json(*prediction.value) : nlohmann::json(nullptr);
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

    // Detect JSON vs FlatBuffer by content, not extension.
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

        // Discard any embedded plan; re-serialize rather than re-lower to preserve the graph ID.
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

    PackBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    const auto allocated
        = allocateVariantPack(hipdnn_bench::planVariantPack(graph), buffers, variantPack);
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
        workspace = buffers.addDevice(workspaceSize);
        if(workspace == nullptr)
        {
            std::cerr << "Out of device memory for a " << workspaceSize << " byte workspace\n";
            return 1;
        }
    }

    AutotuneConfig config;
    // STANDARD, not EXHAUSTIVE: see the file comment.
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = options.warmup;
    config.maxIterations = options.maxIterations;
    config.stabilityThreshold = options.stability;
    config.engineIdFilter = {options.engineId};

    std::vector<AutotuneResult> results;
    // No storage file: it would persist only the winner.
    const auto tuned
        = graph.autotune(handle, variantPack, workspace, workspaceSize, config, {}, &results);
    if(!tuned.is_good() && results.empty())
    {
        std::cerr << "Benchmarking failed: " << tuned.get_message() << "\n";
        return 1;
    }

    const std::string problemId = options.problemId.empty() ? options.graphPath : options.problemId;

    // RFC 0019 §13.2: re-run each measured candidate once, untimed, and cross-check it against
    // the rest of the catalog. `verdicts` is index-aligned with `results`. Captures go straight
    // to the cross-check, which keeps one image per distinct answer, not per candidate.
    std::map<int64_t, hipdnn_bench::TensorDescription> tensors;
    hipdnn_bench::CatalogCrossCheck crossCheck(tensors);
    for(const auto& result : results)
    {
        hipdnn_bench::CandidateOutput captured;
        if(!result.succeeded || result.iterationsRun == 0)
        {
            captured.failure = "the candidate carries no measurement to validate";
        }
        else
        {
            const auto ran = captureCandidateOutput(handle,
                                                    graphBytes,
                                                    looksLikeJson,
                                                    options.engineId,
                                                    result.knobSettings,
                                                    tensors,
                                                    captured.images);
            captured.executed = ran.is_good();
            if(!ran.is_good())
            {
                // Drop partial images: a missing tensor would silently narrow the check.
                captured.failure = ran.get_message();
                captured.images.clear();
            }
        }
        crossCheck.add(std::move(captured));
    }
    const auto verdicts = crossCheck.verdicts();

    if(options.json)
    {
        auto output = pageJson(catalog);
        // --sweep keeps the catalog so callers need not enumerate separately; a single
        // configuration run drops it since the caller already named the tuple.
        if(!options.sweep)
        {
            output.erase("candidates");
        }
        output["problem"] = problemId;
        output["results"] = nlohmann::json::array();
        for(size_t index = 0; index < results.size(); ++index)
        {
            const auto& result = results[index];
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
            // is_valid means measured, not numerically correct.
            const bool timed = result.succeeded && result.iterationsRun > 0;
            std::string reason;
            if(!result.succeeded)
            {
                reason = "config_not_applicable: engine declined or failed to run this "
                         "configuration";
            }
            else if(result.iterationsRun == 0)
            {
                reason = "not_timed: autotune reported success without running an iteration";
            }
            // Separate from `is_valid` so an unmeasured row and an incorrect row stay distinct.
            const auto& verdict = verdicts[index];
            output["results"].push_back({{"candidate_id", candidate->id},
                                         {"knob_settings", knobJson(tuple)},
                                         {"kernel_features", candidate->kernelFeatures},
                                         {"rank", result.rank},
                                         {"succeeded", result.succeeded},
                                         {"is_valid", timed},
                                         {"numerically_valid", numericallyValid(verdict.verdict)},
                                         {"validation", verdict.reason},
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

    // One column per knob: uhd_gen reads kernel.* columns as features and hashes the header.
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
        // Timing column names follow RFC 0019.13 §8.3 (and `robustMeanMs` matches uhd_gen), so
        // they are camelCase. `numerically_valid` is three-valued text, kept separate from
        // `is_valid` (RFC 0019 §13.2).
        std::cout << ",engine,rank,succeeded,is_valid,numerically_valid,validation,skip_reason,"
                     "minTimeMs,avgTimeMs,robustMeanMs,stddevMs,iters,converged,workspace_bytes\n";
    }

    // Emit every variant, including losers and failures: ranking needs the comparison.
    // Treat `rank` as advisory; near-ties flip between runs, so train on times (§5.6).
    for(size_t index = 0; index < results.size(); ++index)
    {
        const auto& result = results[index];
        std::cout << problemId;
        for(const auto& entry : options.query)
        {
            std::cout << "," << entry.second;
        }
        for(const auto& knob : kernelNames)
        {
            std::cout << "," << knobFor(result, knob);
        }
        // RFC 0019.13 §7.4/§8: untimed pairs are kept with is_valid=False and a skip_reason
        // for coverage auditing; training filters on is_valid.
        const bool timed = result.succeeded && result.iterationsRun > 0;
        std::string skipReason;
        if(!result.succeeded)
        {
            skipReason = "config_not_applicable: engine declined or failed to run this "
                         "configuration";
        }
        else if(result.iterationsRun == 0)
        {
            // Emitting this as valid would put a zero time in the training set.
            skipReason = "not_timed: autotune reported success without running an iteration";
        }

        std::cout << "," << (options.engineName.empty() ? result.engineName : options.engineName)
                  << "," << result.rank << "," << (result.succeeded ? 1 : 0) << ","
                  << (timed ? "True" : "False") << ","
                  << hipdnn_bench::verdictText(verdicts[index].verdict) << ","
                  << hipdnn_bench::csvField(verdicts[index].reason) << ","
                  << hipdnn_bench::csvField(skipReason) << "," << result.minTimeMs << ","
                  << result.avgTimeMs << "," << result.robustTimeMs << "," << result.stddevMs << ","
                  << result.iterationsRun << "," << (result.converged ? 1 : 0) << ","
                  << result.workspaceSize << "\n";
    }

    return results.empty() ? 2 : 0;
}

} // namespace

int main(int argc, char* argv[])
{
    // Report escaping exceptions instead of terminating without a diagnostic.
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
