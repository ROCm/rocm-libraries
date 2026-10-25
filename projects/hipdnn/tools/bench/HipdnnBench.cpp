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

#include <hipdnn_bench/CandidatePageJson.hpp>
#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_bench/DeviceIdentity.hpp>
#include <hipdnn_bench/GraphFormat.hpp>
#include <hipdnn_bench/InputDomain.hpp>
#include <hipdnn_bench/NumericalValidation.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>
#include <hipdnn_tools/CliArguments.hpp>
#include <hipdnn_tools/CliNumbers.hpp>

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
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
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
        << "  --workspace-limit <n>  Set global.workspace_size_limit in bytes (not valid with\n"
        << "                         the engine-level modes above)\n"
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
        << "  --no-header            Omit the CSV header, for concatenating runs that report\n"
        << "                         the same knobs (each knob is a kernel.* column)\n";
}

/// What the command line asks for: a run, the help text, or nothing it can use.
enum class ParseOutcome
{
    RUN,
    HELP,
    INVALID
};

ParseOutcome parseArguments(const std::vector<std::string>& args, Options& options)
{
    for(size_t i = 1; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        // A following flag is not a value: `--problem-id --no-header` must not swallow the flag.
        // A number outside what the option can hold is refused: `--warmup -1`, or a count that
        // does not fit an int.
        hipdnn_tools::OptionReader reader(args, i);
        const auto next = [&reader]() { return reader.next(); };

        if(arg == "--help" || arg == "-h")
        {
            printHelp(args[0].c_str());
            return ParseOutcome::HELP;
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
            reader.number(
                options.engineId, hipdnn_tools::parseEngineId, std::numeric_limits<int64_t>::min());
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
            options.havePageOptions = true;
            reader.integer(arg == "--offset" ? options.offset : options.limit, 0);
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
                return ParseOutcome::INVALID;
            }
            const auto name = setting.substr(0, split);
            const auto parsed = hipdnn_tools::parseInteger(setting.substr(split + 1));
            const auto value = parsed.value_or(0);
            if(name.empty() || !parsed
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
            reader.integer(options.maxIterations, 1);
        }
        else if(arg == "--warmup")
        {
            reader.integer(options.warmup, 0);
        }
        else if(arg == "--stability")
        {
            reader.real(options.stability, 0.0);
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
                    return ParseOutcome::INVALID;
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
            return ParseOutcome::INVALID;
        }
        if(reader.refused(arg, std::cerr))
        {
            return ParseOutcome::INVALID;
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
        // A workspace limit is a query constraint: it turns the engine-level prediction into a
        // configuration-level one, which the frontend rejects on engine queries and which binds
        // a different role than an engine-level (L1) model.
        const auto workspaceLimit
            = std::find_if(options.knobs.begin(), options.knobs.end(), [](const auto& setting) {
                  return setting.first == "global.workspace_size_limit";
              });
        if(workspaceLimit != options.knobs.end())
        {
            throw std::invalid_argument("--workspace-limit (global.workspace_size_limit) cannot "
                                        "be combined with --predict-engine, "
                                        "--describe-engine-prediction or --collect-immediate: "
                                        "engine-level measurements take no workspace constraint");
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
    return ParseOutcome::RUN;
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

hipdnn_frontend::Error hipError(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        return {hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                std::string(operation) + ": " + hipGetErrorString(status)};
    }
    return {};
}

/// The id @p graph carries, as its serialized form names it.
hipdnn_frontend::Error graphIdOf(const hipdnn_frontend::graph::Graph& graph, std::string& graphId)
{
    nlohmann::json serializedGraph;
    HIPDNN_CHECK_ERROR(graph.serialize(serializedGraph));
    graphId = serializedGraph.at("id").get<std::string>();
    return {};
}

hipdnn_frontend::Error engineIdentity(hipdnnHandle_t handle,
                                      const hipdnn_frontend::graph::Graph& graph,
                                      int64_t engineId,
                                      hipStream_t& stream,
                                      nlohmann::json& output)
{
    std::string graphId;
    HIPDNN_CHECK_ERROR(graphIdOf(graph, graphId));
    output["graph_id"] = graphId;

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
    // The kind of device, as the catalog page names it: a row is of a device kind, so a
    // shape measured on two GPUs of one SKU is one problem.
    output["device_id"] = hipdnn_bench::deviceKindId(properties);

    // Which physical GPU ran it, for diagnosing one bad board; never a training identity.
    hipUUID uuid{};
    HIPDNN_CHECK_ERROR(hipError(hipDeviceGetUuid(&uuid, device), "Could not read device UUID"));
    static constexpr std::string_view HEX = "0123456789abcdef";
    std::string deviceUuid;
    deviceUuid.reserve(sizeof(uuid.bytes) * 2);
    for(const auto byte : uuid.bytes)
    {
        const auto value = static_cast<unsigned char>(byte);
        deviceUuid.push_back(HEX[value >> 4]);
        deviceUuid.push_back(HEX[value & 0x0f]);
    }
    output["device_uuid"] = std::move(deviceUuid);
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

/// What one autotune result reports, with its cross-check verdict; @p engineName overrides
/// the engine the result names.
hipdnn_bench::CandidateOutcome outcomeOf(const AutotuneResult& result,
                                         const hipdnn_bench::ValidationOutcome& verdict,
                                         const std::string& engineName)
{
    hipdnn_bench::CandidateOutcome outcome;
    outcome.engine = engineName.empty() ? result.engineName : engineName;
    outcome.rank = result.rank;
    outcome.succeeded = result.succeeded;
    outcome.iterationsRun = result.iterationsRun;
    outcome.verdict = hipdnn_bench::verdictText(verdict.verdict);
    outcome.numericallyValid = numericallyValid(verdict.verdict);
    outcome.validation = verdict.reason;
    outcome.minTimeMs = result.minTimeMs;
    outcome.avgTimeMs = result.avgTimeMs;
    outcome.robustTimeMs = result.robustTimeMs;
    outcome.stddevMs = result.stddevMs;
    outcome.converged = result.converged;
    outcome.workspaceBytes = static_cast<int64_t>(result.workspaceSize);
    return outcome;
}

/// @brief Writes the validation fill into every input buffer of @p variantPack.
///
/// Results keep their zero fill so a kernel that writes nothing is detectable. Types this
/// build cannot encode exactly also keep zeros rather than risk filling NaNs. The inputs in
/// @p positive -- read by an operation undefined on negative values -- are filled positive.
///
/// @p fills keeps each input's image (a function of seed, uid and size) across one problem's
/// candidates, uploaded afresh into each candidate's buffers, so a kernel that scribbles on an
/// input cannot pass its damage on.
hipdnn_frontend::Error fillGraphInputs(const hipdnn_bench::VariantPackPlan& plan,
                                       const std::unordered_map<int64_t, void*>& variantPack,
                                       uint64_t seed,
                                       const std::set<int64_t>& positive,
                                       std::map<int64_t, std::vector<uint8_t>>& fills)
{
    for(const auto& tensor : plan.tensors)
    {
        const auto buffer = variantPack.find(tensor.uid);
        if(tensor.produced || buffer == variantPack.end())
        {
            continue; // A result, or something not in the pack at all.
        }
        const auto bytes = static_cast<size_t>(tensor.bytes);
        auto cached = fills.find(tensor.uid);
        // Empty is the declined-type answer and is reused as such; a size that moved between
        // candidates is not the same image, so it is generated again rather than reused.
        if(cached == fills.end() || (!cached->second.empty() && cached->second.size() != bytes))
        {
            cached = fills
                         .insert_or_assign(
                             tensor.uid,
                             hipdnn_bench::detail::inputFillImage(tensor.dataType,
                                                                  bytes,
                                                                  seed,
                                                                  tensor.uid,
                                                                  positive.count(tensor.uid) != 0))
                         .first;
        }
        const auto& image = cached->second;
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
                           std::map<int64_t, std::vector<uint8_t>>& images,
                           const std::set<int64_t>& positive,
                           std::map<int64_t, std::vector<uint8_t>>& fills)
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
    HIPDNN_CHECK_ERROR(fillGraphInputs(
        plan, variantPack, hipdnn_bench::detail::graphFillSeed(graphBytes), positive, fills));
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

    // The runtime ranks kernels with the stream held until host submission has finished, so
    // the span holds device work and no host dispatch time. execute_timed_ext() takes the same
    // measurement; a label here is then the quantity the runtime's own selection ranks.
    const auto timeDeviceOnly
        = [&](hipdnn_frontend::ExecutionTiming& timing) -> hipdnn_frontend::Error {
        return graph.execute_timed_ext(handle, variantPack, workspace, timing);
    };
    // Events around execute(), for a device that cannot hold the stream. The span then includes
    // host dispatch time, as the runtime's own unstalled re-measurement does.
    const auto timeUnstalled
        = [&](hipdnn_frontend::ExecutionTiming& timing) -> hipdnn_frontend::Error {
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
    const auto onIteration = [](int, float, float, bool) {};
    auto quality = hipdnn_frontend::TimingQuality::DEVICE_ONLY;
    auto outcome = hipdnn_frontend::autotune::detail::runUntilStable(options.maxIterations,
                                                                     AutotuneConfig{}.windowSize,
                                                                     options.stability,
                                                                     /*stalled=*/true,
                                                                     timeDeviceOnly,
                                                                     onIteration);
    if(outcome.restartUnstalled)
    {
        // The stall could not hold (a watchdog timeout, or a device without stalling): discard
        // the pass and measure again unstalled, as the runtime does.
        quality = hipdnn_frontend::TimingQuality::UNSTALLED;
        outcome = hipdnn_frontend::autotune::detail::runUntilStable(options.maxIterations,
                                                                    AutotuneConfig{}.windowSize,
                                                                    options.stability,
                                                                    /*stalled=*/false,
                                                                    timeUnstalled,
                                                                    onIteration);
    }
    output["timing_quality"]
        = quality == hipdnn_frontend::TimingQuality::DEVICE_ONLY ? "device_only" : "unstalled";
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

int runEngineMode(hipdnnHandle_t handle,
                  BenchGraph& graph,
                  const std::string& graphKey,
                  const Options& options)
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
           {"constraints", hipdnn_bench::knobJson(toVariantKnobs(settings))}};
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
        output["graph_key"] = graphKey;
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
    const auto parsed = parseArguments(args, options);
    if(parsed != ParseOutcome::RUN)
    {
        return parsed == ParseOutcome::HELP ? 0 : 1;
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
    const bool looksLikeJson = hipdnn_bench::isJsonGraph(graphBytes);
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
    // The key of the bytes read, so a corpus graph's key equals its manifest `benchmark`; a
    // JSON graph is keyed as it serializes.
    std::string graphKey;
    // Inputs the validation fill keeps positive (an operation undefined on negatives reads them).
    std::set<int64_t> positiveReads;
    try
    {
        if(looksLikeJson)
        {
            std::vector<uint8_t> serialized;
            const auto written = graph.serialize(serialized);
            if(!written.is_good())
            {
                std::cerr << "Could not serialize " << options.graphPath << ": "
                          << written.get_message() << "\n";
                return 1;
            }
            graphKey = hipdnn_bench::graphContentKey(serialized.data(), serialized.size());
            positiveReads = hipdnn_bench::positiveDomainReads(serialized.data(), serialized.size());
        }
        else
        {
            graphKey = hipdnn_bench::graphContentKey(graphBytes.data(), graphBytes.size());
            positiveReads = hipdnn_bench::positiveDomainReads(graphBytes.data(), graphBytes.size());
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << "Could not key " << options.graphPath << ": " << error.what() << "\n";
        return 1;
    }
    if(options.engineMode != EngineMode::NONE)
    {
        return runEngineMode(handle, graph, graphKey, options);
    }

    std::vector<KnobSetting> pinned;
    pinned.reserve(options.knobs.size());
    for(const auto& [name, value] : options.knobs)
    {
        pinned.emplace_back(name, value);
    }
    hipdnn_frontend::EngineCandidatePage catalog;
    const bool useCatalog = options.enumerate || options.json || options.sweep;
    // What the graph is called in output: its own id, not the page's structural hash.
    std::string graphId;
    if(useCatalog)
    {
        const auto named = graphIdOf(graph, graphId);
        if(named.is_bad())
        {
            std::cerr << "Could not read the graph's id: " << named.get_message() << "\n";
            return 1;
        }
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
            std::cout << hipdnn_bench::pageJson(catalog, graphKey, graphId).dump() << "\n";
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
               || !tuples.insert(hipdnn_bench::knobJson(candidate.variant.knobSettings).dump())
                       .second)
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
    std::map<int64_t, std::vector<uint8_t>> fills; // One input image per uid, every candidate.
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
                                                    captured.images,
                                                    positiveReads,
                                                    fills);
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
        auto output = hipdnn_bench::pageJson(catalog, graphKey, graphId);
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
            output["results"].push_back(
                hipdnn_bench::sweepResultJson(candidate->id,
                                              hipdnn_bench::knobJson(tuple),
                                              candidate->kernelFeatures,
                                              outcomeOf(result, verdicts[index], "")));
        }
        std::cout << output.dump() << "\n";
        return results.empty() ? 2 : 0;
    }

    // One column per knob: uhd_gen reads kernel.* columns as features and hashes the header.
    const auto kernelNames = kernelColumns(results);

    // A row is of a graph (its content key) on a kind of device, so the dataset keys it as
    // the catalog page and engine mode do (RFC 0019.13 §4.1, §8.3).
    int device = 0;
    hipDeviceProp_t deviceProperties{};
    if(hipGetDevice(&device) != hipSuccess
       || hipGetDeviceProperties(&deviceProperties, device) != hipSuccess)
    {
        std::cerr << "Could not read the device's properties\n";
        return 1;
    }
    // The device, then the problem as the runtime sees it: a sweep has the catalog page's
    // extracted features, and a run without one is identified by its --query alone.
    auto featureColumns = hipdnn_bench::deviceColumns(deviceProperties);
    for(auto& column : hipdnn_bench::problemColumns(catalog.problemFeatures))
    {
        featureColumns.push_back(std::move(column));
    }

    if(options.header)
    {
        std::vector<std::string> featureNames;
        featureNames.reserve(featureColumns.size());
        for(const auto& column : featureColumns)
        {
            featureNames.push_back(column.first);
        }
        std::vector<std::string> queryNames;
        queryNames.reserve(options.query.size());
        for(const auto& entry : options.query)
        {
            queryNames.push_back(entry.first);
        }
        std::cout << hipdnn_bench::csvHeader(featureNames, queryNames, kernelNames) << "\n";
    }

    // Emit every variant, including losers and failures: ranking needs the comparison.
    // Treat `rank` as advisory; near-ties flip between runs, so train on times (§5.6).
    std::vector<std::string> features;
    features.reserve(featureColumns.size());
    for(const auto& column : featureColumns)
    {
        features.push_back(column.second);
    }
    std::vector<std::string> queries;
    queries.reserve(options.query.size());
    for(const auto& entry : options.query)
    {
        queries.push_back(entry.second);
    }
    for(size_t index = 0; index < results.size(); ++index)
    {
        const auto& result = results[index];
        std::vector<std::string> knobs;
        knobs.reserve(kernelNames.size());
        for(const auto& knob : kernelNames)
        {
            knobs.push_back(knobFor(result, knob));
        }
        const auto outcome = outcomeOf(result, verdicts[index], options.engineName);
        std::cout << hipdnn_bench::csvRow(problemId, graphKey, features, queries, knobs, outcome)
                  << "\n";
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
