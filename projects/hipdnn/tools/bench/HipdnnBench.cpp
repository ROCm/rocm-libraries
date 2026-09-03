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
 * Timing is hipDNN's own autotune loop rather than anything written here. It already stops on
 * a coefficient-of-variation threshold instead of a fixed iteration count, which is the
 * difference between a corpus of measurements and a corpus of noise.
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

#include <hipdnn_bench/KnobEnumeration.hpp>
#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
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
using hipdnn_frontend::KnobSetting;
using hipdnn_frontend::TuneMode;

/// EngineVariant carries its knobs as a map, while AutotuneResult reports them as a vector.
/// Converting here keeps the enumeration header in the form the results come back in, so one
/// spelling covers both what is asked for and what is recorded.
std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>
    toVariantKnobs(const hipdnn_bench::Configuration& configuration)
{
    std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant> knobs;
    for(const auto& setting : configuration)
    {
        knobs.emplace(setting.knobId(), setting.value());
    }
    return knobs;
}

struct Options
{
    std::vector<std::string> pluginDirs;
    std::string graphPath;
    std::string engineName;
    int64_t engineId = 0;
    bool haveEngineId = false;
    bool sweep = false;
    bool header = true;
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
    std::cout << "Usage: " << program << " --graph <file> --engine-name <name> [options]\n\n"
              << "  --graph <file>         Serialized problem graph (JSON)\n"
              << "  --engine-name <name>   Engine under test, e.g. hipkernel:ConvFwd\n"
              << "  --engine-id <id>       Same, by id; decimal or 0x-prefixed hex\n"
              << "  --plugin-dir <dir>     Engine plugin directory (repeatable)\n"
              << "  --sweep                Time every value of every knob, not just defaults\n"
              << "  --knob <name=value>    Pin one knob (repeatable); implies no sweep\n"
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
        else if(arg == "--sweep")
        {
            options.sweep = true;
        }
        else if(arg == "--knob")
        {
            const auto setting = next();
            const auto split = setting.find('=');
            if(split == std::string::npos)
            {
                std::cerr << "--knob expects name=value, got '" << setting << "'\n";
                return false;
            }
            options.knobs.emplace_back(
                setting.substr(0, split),
                std::strtoll(setting.c_str() + split + 1, nullptr, 10));
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

int runBench(const std::vector<std::string>& args)
{

    Options options;
    if(!parseArguments(args, options))
    {
        return 0;
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

    // A problem file is a serialized graph in either form: the generator writes the binary
    // FlatBuffers a builder produces, while a hand-made or exported problem is often JSON.
    // Distinguished by content rather than by extension, so a renamed file still loads.
    hipdnn_frontend::graph::Graph graph;
    const bool looksLikeJson = graphBytes.front() == static_cast<uint8_t>('{');
    const auto restored
        = looksLikeJson
              ? graph.deserialize(handle,
                                  std::string(graphBytes.begin(), graphBytes.end()))
              : graph.deserialize(handle, graphBytes);
    if(!restored.is_good())
    {
        std::cerr << "Could not read " << options.graphPath << ": " << restored.get_message()
                  << "\n";
        hipdnnDestroy(handle);
        return 1;
    }
    if(!graph.build_operation_graph(handle).is_good())
    {
        std::cerr << "Could not finalize the problem graph\n";
        hipdnnDestroy(handle);
        return 1;
    }

    const auto plan = hipdnn_bench::planVariantPack(graph);
    if(!plan.error.empty())
    {
        std::cerr << plan.error << "\n";
        hipdnnDestroy(handle);
        return 1;
    }

    DeviceBuffers buffers;
    std::unordered_map<int64_t, void*> variantPack;
    for(const auto& tensor : plan.tensors)
    {
        void* pointer = buffers.add(tensor.bytes);
        if(pointer == nullptr)
        {
            std::cerr << "Out of device memory for '" << tensor.name << "' (" << tensor.bytes
                      << " bytes)\n";
            hipdnnDestroy(handle);
            return 1;
        }
        variantPack[tensor.uid] = pointer;
    }

    std::vector<hipdnn_frontend::EngineConfigInfo> engines;
    if(!graph.get_engine_configs(handle, engines).is_good())
    {
        std::cerr << "No engine configurations available for this graph\n";
        hipdnnDestroy(handle);
        return 1;
    }

    const auto engine = std::find_if(
        engines.begin(), engines.end(), [&](const hipdnn_frontend::EngineConfigInfo& candidate) {
            return candidate.engineId == options.engineId;
        });
    if(engine == engines.end())
    {
        std::cerr << "Engine is not applicable to this problem\n";
        hipdnnDestroy(handle);
        return 1;
    }

    std::vector<hipdnn_frontend::EngineVariant> variants;
    if(!options.knobs.empty())
    {
        std::vector<KnobSetting> pinned;
        pinned.reserve(options.knobs.size());
        for(const auto& [name, value] : options.knobs)
        {
            pinned.emplace_back(name, value);
        }
        variants.push_back({options.engineId, toVariantKnobs(pinned)});
    }
    else
    {
        // Read the configuration space off the engine's knobs. Without --sweep that is a
        // single entry meaning "engine defaults", which is one row per problem: enough to
        // rank engines against each other, not enough to train a kernel-choosing heuristic.
        auto space = options.sweep
                         ? hipdnn_bench::enumerateConfigurations(engine->knobs)
                         : hipdnn_bench::ConfigurationSet{{hipdnn_bench::Configuration{}}, {}};
        for(const auto& gap : space.notFullyCovered)
        {
            std::cerr << "note: configuration space not fully covered: " << gap << "\n";
        }
        for(const auto& configuration : space.configurations)
        {
            variants.push_back({options.engineId, toVariantKnobs(configuration)});
        }
    }

    if(!graph.add_engine_variants(variants).is_good())
    {
        std::cerr << "Could not build plan specs for the requested configurations\n";
        hipdnnDestroy(handle);
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
            hipdnnDestroy(handle);
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
        hipdnnDestroy(handle);
        return 1;
    }

    const std::string problemId
        = options.problemId.empty() ? options.graphPath : options.problemId;

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
        std::cout << ",engine,rank,succeeded,is_valid,skip_reason,min_time_ms,avg_time_ms,"
                     "robust_time_ms,stddev_ms,iterations,converged,workspace_bytes\n";
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
    // `stddev_ms` is emitted beside every measurement rather than folded into it.
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

    hipdnnDestroy(handle);
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
