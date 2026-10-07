// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file CorpusGen.cpp
 * @brief Generates an engine's problem corpus from declarations (RFC 0019.13 §4, §5).
 *
 * Operations are `*.opmeta.json` declarations explored uniformly; the engine is asked, not
 * modelled. Output: the problems the engine accepts, as graphs plus one bench command each.
 */

#include <hipdnn_corpus_gen/CorpusManifest.hpp>
#include <hipdnn_corpus_gen/CorpusOutput.hpp>
#include <hipdnn_corpus_gen/EngineCoverage.hpp>
#include <hipdnn_corpus_gen/GraphIdentity.hpp>
#include <hipdnn_corpus_gen/KernelCatalogSource.hpp>
#include <hipdnn_corpus_gen/MetadataCorpus.hpp>
#include <hipdnn_corpus_gen/ModelShapeSource.hpp>
#include <hipdnn_corpus_gen/PointFilter.hpp>
#include <hipdnn_corpus_gen/PoolAssembly.hpp>
#include <hipdnn_corpus_gen/RegimeFocus.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>
#include <hipdnn_corpus_gen/ServedExtent.hpp>

#include <hipdnn_frontend.hpp>
#include <sstream>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using hipdnn_corpus_gen::asQueryArgument;
using hipdnn_corpus_gen::asQueryColumns;
using hipdnn_corpus_gen::ExplorationRequest;
using hipdnn_corpus_gen::ProblemPoint;

/// @brief The name the graph document carries: the operation, its regime, and its parameters.
///
/// Also the file stem, `--problem-id` and `<op>.problems.csv` key. Scoring and L2 collection
/// join on this name alone, so it must be unique: every parameter is spelled out.
std::string
    graphNameFor(const std::string& operation, const std::string& regime, const ProblemPoint& point)
{
    std::string name = operation;
    if(!regime.empty())
    {
        name += "_" + regime;
    }
    for(const auto& parameter : point)
    {
        name += "_" + parameter.first + hipdnn_corpus_gen::asText(parameter.second);
    }
    return name;
}

/// @brief Every engine the loaded plugins registered, as (id, name).
///
/// Used to reject an unregistered engine: a misspelled name still hashes to a valid id and
/// would silently yield an empty corpus.
std::vector<std::pair<int64_t, std::string>> loadedEngines(hipdnnHandle_t handle)
{
    std::vector<std::pair<int64_t, std::string>> engines;
    size_t count = 0;
    if(hipdnnGetEngineCount_ext(handle, &count) != HIPDNN_STATUS_SUCCESS)
    {
        return engines;
    }
    for(size_t index = 0; index < count; ++index)
    {
        int64_t id = 0;
        size_t nameLength = 0;
        size_t pluginLength = 0;
        size_t versionLength = 0;
        size_t typeLength = 0;
        if(hipdnnGetEngineInfo_ext(handle,
                                   index,
                                   &id,
                                   nullptr,
                                   &nameLength,
                                   nullptr,
                                   &pluginLength,
                                   nullptr,
                                   &versionLength,
                                   nullptr,
                                   &typeLength)
           != HIPDNN_STATUS_SUCCESS)
        {
            continue;
        }
        std::string name(nameLength, '\0');
        std::string plugin(pluginLength, '\0');
        std::string version(versionLength, '\0');
        std::string type(typeLength, '\0');
        if(hipdnnGetEngineInfo_ext(handle,
                                   index,
                                   nullptr,
                                   name.data(),
                                   &nameLength,
                                   plugin.data(),
                                   &pluginLength,
                                   version.data(),
                                   &versionLength,
                                   type.data(),
                                   &typeLength)
           != HIPDNN_STATUS_SUCCESS)
        {
            continue;
        }
        name.resize(std::strlen(name.c_str()));
        engines.emplace_back(id, std::move(name));
    }
    return engines;
}

struct Options
{
    std::vector<std::string> pluginDirs;
    std::string operationsDir;
    std::string engineName;
    std::string outputDir;
    std::string benchPath = "hipdnn_bench";
    std::string onlyOperation;
    std::string probe;
    int64_t engineId = 0;
    bool haveEngineId = false;
    /// Engines that must also serve every problem (`--also-engine-name`); resolved to ids.
    std::vector<std::string> alsoEngineNames;
    std::vector<int64_t> alsoEngineIds;

    /// Explicit opt-out of engine verification. Without an engine the corpus is only what the
    /// declarations express, and engine refusals correlate with performance axes.
    bool withoutEngine = false;

    ExplorationRequest exploration;

    /// Where the kernel pool comes from: `*.kdp.json` packs, or directories holding them.
    std::vector<std::filesystem::path> packRoots;

    /// Where the model pool comes from: CSVs of `q.<parameter>` columns.
    std::vector<std::filesystem::path> modelShapes;

    /// Corpus size per operation; 0 means everything the pools hold. Never padded.
    int64_t count = 0;

    std::map<std::string, double> shares = hipdnn_corpus_gen::defaultShares();

    std::vector<std::string> keep;

    /// Manifests whose graphs must not appear here (the held-out comparison set).
    std::vector<std::filesystem::path> excludeCorpora;

    /// Per-operation regime quotas; key "" (from `--regime-quota`) applies to every operation.
    std::map<std::string, std::map<std::string, int64_t>> regimeQuotas;
    std::vector<std::filesystem::path> regimeQuotaFiles;
    std::string quotaError;

    /// `--regime-floor`: at least this many problems in every regime the declaration can spell.
    /// Unlike a quota, a regime the engine does not serve is reported short and the run succeeds.
    int64_t regimeFloor = 0;

    /// `--include-extremes`: take each numeric parameter's served min and max points first.
    bool includeExtremes = false;

    /// Ceiling in bytes across a searched problem's tensors.
    int64_t maxBytes = 256LL * 1024 * 1024;
};

void printHelp(const char* program)
{
    std::cout << "Usage: " << program << " --operations <dir> --output <dir> [options]\n\n"
              << "  --operations <dir>     Directory of *.opmeta.json declarations\n"
              << "  --engine-name <name>   REQUIRED. Engine to generate for, e.g.\n"
              << "                         hipkernel:ConvFwd. Every problem is offered to it,\n"
              << "                         so the corpus is what that engine actually serves.\n"
              << "                         Needs a GPU and --plugin-dir.\n"
              << "  --engine-id <id>       Same, by id; decimal or 0x-prefixed hex\n"
              << "  --also-engine-name <name>  Another engine that must ALSO serve every\n"
              << "                         problem (repeatable): the corpus is then the\n"
              << "                         shapes all of them serve, for comparing engines.\n"
              << "  --without-engine       Generate with no engine, on no GPU. The corpus is\n"
              << "                         then every problem the DECLARATIONS express, which\n"
              << "                         is a superset of what any engine serves, and any\n"
              << "                         narrowing you add with --keep or --kdp-root is a\n"
              << "                         guess that nothing here verifies. Use only when a\n"
              << "                         provider cannot be staged.\n"
              << "  --plugin-dir <dir>     Engine plugin directory (repeatable)\n"
              << "  --output <dir>         Corpus root: graphs/, manifest.json, manifest.csv\n"
              << "  --bench-path <path>    hipdnn_bench to name in commands.txt\n"
              << "  --operation <name>     Restrict to one declared operation\n"
              << "  --count <n>            Corpus size per operation. Combinations the engine\n"
              << "                         serves are searched further until it is met; fewer\n"
              << "                         is returned only when every one saturates (exit 0,\n"
              << "                         with a warning) and otherwise fails (exit 3).\n"
              << "                         0 (default) takes what the first pass finds. The\n"
              << "                         per-combination count is --per-combination.\n"
              << "  --per-combination <n>  First-pass problems per categorical combination\n"
              << "                         (default 50)\n"
              << "  --kdp-root <path>      A *.kdp.json pack, or a directory of them (repeatable)\n"
              << "  --model-shapes <csv>   Recorded shapes as q.<parameter> columns (repeatable)\n"
              << "  --model-share <f>      Share of the corpus per source; a share of 0\n"
              << "  --kernel-share <f>     excludes that source rather than deferring it\n"
              << "  --sweep-share <f>\n"
              << "  --keep q.<p>=<v>       Keep only problems with this facet. Repeatable:\n"
              << "                         different parameters conjoin, the same parameter\n"
              << "                         repeated widens it (q.head_dim=64 q.head_dim=128)\n"
              << "  --exclude-corpus <m>   manifest.json whose graphs must not recur (repeatable)\n"
              << "  --regime-quota <r>=<n> At least n problems in regime r, as manifest.csv\n"
              << "                         labels it (decode_short_mha); every operation.\n"
              << "                         Repeatable. A regime the first pass under-fills is\n"
              << "                         searched again with its declared equalities pinned.\n"
              << "                         --count below the quotas' sum does not trim them;\n"
              << "                         above it, the rest is cut as usual. 0 (default)\n"
              << "                         with quotas means the quotas alone. Exit 3 when a\n"
              << "                         quota is short and not shown saturated.\n"
              << "  --regime-quotas <f>    The same as JSON: {\"<operation>\": {\"<regime>\": n}}\n"
              << "  --regime-floor <n>     At least n problems in every regime the declaration\n"
              << "                         can spell (a named quota overrides it). Searched as\n"
              << "                         quotas are; a regime the engine does not serve is\n"
              << "                         reported short in manifest.json, never an error.\n"
              << "  --include-extremes     Take, before anything else, the served point with the\n"
              << "                         smallest and the largest value of each numeric\n"
              << "                         parameter: the edges of what the engine serves, which\n"
              << "                         a spread cut can leave out. Counts against --count.\n"
              << "                         manifest.json's served_extent is reported either way.\n"
              << "  --budget <n>           First-pass oracle calls per combination (default\n"
              << "                         20000); growth toward --count may reach 64x this\n"
              << "  --ceiling <n>          Largest extent to propose (default 4096). A served\n"
              << "                         maximum within a tenth of it is marked\n"
              << "                         at_search_ceiling in served_extent: the search's\n"
              << "                         edge, not the engine's\n"
              << "  --probe <k=v,...>      Report what happens to one point, and stop\n"
              << "  --max-bytes <n>        Ceiling on a searched problem's tensors (default\n"
              << "                         256 MiB). Pack and model shapes are exempt: they\n"
              << "                         are real workloads, not proposals.\n"
              << "  --max-skeleton <n>     Declared regime combinations to try (default 512)\n"
              << "  --seed <n>             Reproducibility seed\n\n"
              << "Exit status: 0 success; 1 bad arguments or an unregistered engine; 2 empty\n"
              << "corpus; 3 fewer than --count without the engine being shown to serve no more.\n";
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
        if(arg == "--operations")
        {
            options.operationsDir = next();
        }
        else if(arg == "--engine-name")
        {
            options.engineName = next();
            options.engineId = hipdnn_data_sdk::utilities::engineNameToId(options.engineName);
            options.haveEngineId = true;
        }
        else if(arg == "--also-engine-name")
        {
            options.alsoEngineNames.push_back(next());
        }
        else if(arg == "--engine-id")
        {
            options.engineId = static_cast<int64_t>(std::strtoull(next().c_str(), nullptr, 0));
            options.haveEngineId = true;
        }
        else if(arg == "--without-engine")
        {
            options.withoutEngine = true;
        }
        else if(arg == "--plugin-dir")
        {
            options.pluginDirs.push_back(next());
        }
        else if(arg == "--output")
        {
            options.outputDir = next();
        }
        else if(arg == "--bench-path")
        {
            options.benchPath = next();
        }
        else if(arg == "--operation")
        {
            options.onlyOperation = next();
        }
        else if(arg == "--count")
        {
            options.count = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--per-combination")
        {
            options.exploration.pointsPerCombination = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--kdp-root" || arg == "--kdp")
        {
            options.packRoots.emplace_back(next());
        }
        else if(arg == "--model-shapes")
        {
            options.modelShapes.emplace_back(next());
        }
        else if(arg == "--model-share")
        {
            options.shares["model"] = std::strtod(next().c_str(), nullptr);
        }
        else if(arg == "--kernel-share")
        {
            options.shares["kernel"] = std::strtod(next().c_str(), nullptr);
        }
        else if(arg == "--sweep-share")
        {
            options.shares["sweep"] = std::strtod(next().c_str(), nullptr);
        }
        else if(arg == "--keep")
        {
            options.keep.push_back(next());
        }
        else if(arg == "--exclude-corpus")
        {
            options.excludeCorpora.emplace_back(next());
        }
        else if(arg == "--regime-quota")
        {
            const auto clause = next();
            const auto split = clause.rfind('=');
            if(split == std::string::npos || split == 0)
            {
                options.quotaError = "--regime-quota takes <regime>=<n>, not '" + clause + "'";
                continue;
            }
            options.regimeQuotas[""][clause.substr(0, split)]
                = std::strtoll(clause.substr(split + 1).c_str(), nullptr, 10);
        }
        else if(arg == "--regime-quotas")
        {
            options.regimeQuotaFiles.emplace_back(next());
        }
        else if(arg == "--regime-floor")
        {
            options.regimeFloor = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--include-extremes")
        {
            options.includeExtremes = true;
        }
        else if(arg == "--budget")
        {
            options.exploration.budgetPerCombination = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--ceiling")
        {
            options.exploration.numericCeiling = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--probe")
        {
            options.probe = next();
        }
        else if(arg == "--restarts")
        {
            options.exploration.restarts = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--steps")
        {
            options.exploration.stepsPerStart = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--max-bytes")
        {
            options.maxBytes = std::strtoll(next().c_str(), nullptr, 10);
        }
        else if(arg == "--max-skeleton")
        {
            options.exploration.maxSkeleton
                = static_cast<size_t>(std::strtoull(next().c_str(), nullptr, 10));
        }
        else if(arg == "--seed")
        {
            options.exploration.seed = std::strtoull(next().c_str(), nullptr, 10);
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

/// The manifest's `served_extent` for one operation. With @p takeEdges, the edge points are
/// added to the sweep pool (where no pool holds them) and named in @p reserved so the cut keeps
/// them. @p admit stamps a new pool entry and returns its graph id, or empty when held out.
///
/// A served maximum within a tenth of @p searchCeiling is marked `at_search_ceiling` (the walk
/// proposes nothing above it), unless the declaration bounds the parameter there.
nlohmann::json
    reportServedExtent(const hipdnn_corpus_gen::OperationMetadata& metadata,
                       const hipdnn_corpus_gen::ProblemCorpus& corpus,
                       hipdnn_corpus_gen::SourcePools& pools,
                       bool takeEdges,
                       bool searched,
                       int64_t searchCeiling,
                       std::set<std::string>& reserved,
                       const std::function<std::string(hipdnn_corpus_gen::PoolEntry&)>& admit)
{
    size_t pooledCount = 0;
    for(const auto& pool : pools)
    {
        pooledCount += pool.second.size();
    }
    std::vector<hipdnn_corpus_gen::ProblemPoint> pooled;
    pooled.reserve(pooledCount);
    std::set<std::string> pooledKeys;
    for(const auto& pool : pools)
    {
        for(const auto& entry : pool.second)
        {
            pooled.push_back(entry.point);
            pooledKeys.insert(hipdnn_corpus_gen::detail::describe(entry.point));
        }
    }
    const auto extent = hipdnn_corpus_gen::servedExtent(
        metadata, hipdnn_corpus_gen::servedPoints(corpus, pooled));

    nlohmann::json parameters = nlohmann::json::object();
    for(const auto& [name, range] : extent.ranges)
    {
        const auto* declared = metadata.find(name);
        const bool boundedBelowCeiling = declared != nullptr && declared->range.has_value()
                                         && declared->range->second <= searchCeiling;
        parameters[name]
            = {{"min", range.first},
               {"max", range.second},
               {"at_search_ceiling",
                searched && range.second * 10 >= searchCeiling * 9 && !boundedBelowCeiling}};
    }
    nlohmann::json taken = nlohmann::json::array();
    for(const auto& edge : takeEdges ? extent.edges : std::vector<hipdnn_corpus_gen::ServedEdge>{})
    {
        const auto key = hipdnn_corpus_gen::detail::describe(edge.point);
        hipdnn_corpus_gen::PoolEntry entry;
        entry.point = edge.point;
        entry.source = "sweep";
        entry.origin = metadata.operation + " extreme " + edge.end + " " + edge.parameter;
        entry.regime = hipdnn_corpus_gen::regimeLabel(metadata, entry.point);
        // Pooled or not, the id comes from the same stamp emission uses.
        const auto id = admit(entry);
        if(id.empty())
        {
            continue; // held out; the next most extreme is not searched for
        }
        if(pooledKeys.insert(key).second)
        {
            pools["sweep"].push_back(std::move(entry));
        }
        reserved.insert(key);
        taken.push_back({{"parameter", edge.parameter},
                         {"end", edge.end},
                         {"value", edge.value},
                         {"benchmark", id}});
    }
    bool capped = false;
    for(const auto& combination : corpus.combinations)
    {
        capped = capped || combination.searchCapped;
    }
    // How far the walks reached: a lower bound on the served region, not its edge.
    // `search_capped`: some search stopped while still finding points.
    return {{"parameters", parameters},
            {"search_ceiling", searchCeiling},
            {"served_points", extent.servedPoints},
            {"searched", searched},
            {"search_capped", capped},
            {"extremes", taken}};
}

int runGenerator(const std::vector<std::string>& args)
{
    Options options;
    if(!parseArguments(args, options))
    {
        return 0;
    }
    if(options.operationsDir.empty())
    {
        std::cerr << "--operations is required\n";
        return 1;
    }
    if(!options.quotaError.empty())
    {
        std::cerr << options.quotaError << "\n";
        return 1;
    }
    // An engine is required unless explicitly waived: without one the corpus is a superset of
    // what any engine serves, and the survivors of benchmarking are a skewed sample.
    if(options.haveEngineId && options.withoutEngine)
    {
        std::cerr << "--without-engine contradicts --engine-name/--engine-id; pick one\n";
        return 1;
    }
    if(!options.haveEngineId && !options.withoutEngine)
    {
        std::cerr << "--engine-name (or --engine-id) is required.\n"
                  << "\n"
                  << "  A corpus is generated FOR an engine: every problem is offered to it, so\n"
                  << "  what comes out is what that engine serves. This needs a GPU and\n"
                  << "  --plugin-dir pointing at the provider.\n"
                  << "\n"
                  << "  To generate with no engine and no GPU, pass --without-engine. Understand\n"
                  << "  what that gives you: every problem the declarations express, which is a\n"
                  << "  superset of what any engine serves. --keep and --kdp-root narrow it by\n"
                  << "  inference, not by asking, and an engine trained on a corpus it mostly\n"
                  << "  declines is biased rather than merely small.\n";
        return 1;
    }
    if(options.withoutEngine)
    {
        std::cerr << "WARNING: generating without an engine. This corpus is UNVERIFIED -- no\n"
                  << "WARNING: engine was asked whether it serves any of these problems, and\n"
                  << "WARNING: any --keep/--kdp-root narrowing here is a guess. Do not train\n"
                  << "WARNING: an engine model on it without checking what survives\n"
                  << "WARNING: benchmarking, and do not treat the survivors as a random\n"
                  << "WARNING: sample of it.\n";
    }
    if(!options.probe.empty() && !options.haveEngineId)
    {
        // A probe reports which engines accept a point, so it needs an engine.
        std::cerr << "--probe needs --engine-name (or --engine-id)\n";
        return 1;
    }

    const auto declarations = hipdnn_corpus_gen::loadOperationDirectory(options.operationsDir);
    for(const auto& error : declarations.errors)
    {
        // A declaration that fails to load would otherwise look like an engine refusal.
        std::cerr << "metadata error: " << error << "\n";
    }
    if(declarations.operations.empty())
    {
        std::cerr << "no usable declarations in " << options.operationsDir << "\n";
        return 1;
    }

    auto selected = declarations;
    if(!options.onlyOperation.empty())
    {
        selected.operations.clear();
        for(const auto& entry : declarations.operations)
        {
            if(entry.second.operation == options.onlyOperation)
            {
                selected.operations.push_back(entry);
            }
        }
        if(selected.operations.empty())
        {
            std::cerr << "no declaration for operation '" << options.onlyOperation << "'\n";
            return 1;
        }
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

    // Clauses are validated against the loaded declarations' parameters.
    std::vector<std::string> declaredParameters;
    for(const auto& entry : selected.operations)
    {
        for(const auto& parameter : entry.second.parameters)
        {
            declaredParameters.push_back(parameter.name);
        }
    }
    std::vector<hipdnn_corpus_gen::KeepClause> keep;
    std::string keepError;
    if(!hipdnn_corpus_gen::parseKeepClauses(options.keep, declaredParameters, keep, keepError))
    {
        std::cerr << keepError << "\n";
        return 1;
    }

    // Compile quotas up front: an unspellable regime would otherwise search, then report as
    // saturated.
    for(const auto& path : options.regimeQuotaFiles)
    {
        std::ifstream file(path);
        nlohmann::json quotas;
        if(!file || !(file >> quotas) || !quotas.is_object())
        {
            std::cerr << "cannot read --regime-quotas " << path
                      << " as {\"<operation>\": {\"<regime>\": n}}\n";
            return 1;
        }
        for(const auto& operation : quotas.items())
        {
            for(const auto& quota : operation.value().items())
            {
                if(!quota.value().is_number_integer())
                {
                    std::cerr << path << ": quota for " << operation.key() << "/" << quota.key()
                              << " is not an integer\n";
                    return 1;
                }
                options.regimeQuotas[operation.key()][quota.key()] = quota.value().get<int64_t>();
            }
        }
    }
    std::map<std::string, std::map<std::string, int64_t>> quotasFor;
    std::map<std::string, std::map<std::string, hipdnn_corpus_gen::RegimeFocus>> focusFor;
    for(const auto& requested : options.regimeQuotas)
    {
        const auto& operation = requested.first;
        const auto& quotas = requested.second;
        const bool named = !operation.empty();
        const bool selectedHere
            = std::any_of(selected.operations.begin(),
                          selected.operations.end(),
                          [&](const auto& entry) { return entry.second.operation == operation; });
        if(named && !selectedHere)
        {
            std::cerr << "--regime-quotas names operation '" << operation
                      << "', which is not among those being generated\n";
            return 1;
        }
        for(const auto& entry : selected.operations)
        {
            const auto& metadata = entry.second;
            if(named && metadata.operation != operation)
            {
                continue;
            }
            for(const auto& [regime, count] : quotas)
            {
                std::string error;
                const auto focus = hipdnn_corpus_gen::compileRegimeFocus(metadata, regime, error);
                if(!focus.has_value())
                {
                    std::cerr << error << "\n";
                    return 1;
                }
                if(count < 0)
                {
                    std::cerr << "quota for " << regime << " is negative\n";
                    return 1;
                }
                quotasFor[metadata.operation][regime] = count;
                focusFor[metadata.operation].emplace(regime, *focus);
            }
        }
    }
    // Floors: every regime the declaration spells, where no quota named it. Kept apart so a
    // short floor reports instead of failing the run.
    std::map<std::string, std::set<std::string>> floorsFor;
    if(options.regimeFloor < 0)
    {
        std::cerr << "--regime-floor is negative\n";
        return 1;
    }
    for(const auto& entry : selected.operations)
    {
        const auto& operation = entry.second.operation;
        for(auto& [regime, focus] :
            options.regimeFloor > 0
                ? hipdnn_corpus_gen::regimeFloors(entry.second, quotasFor[operation])
                : std::map<std::string, hipdnn_corpus_gen::RegimeFocus>{})
        {
            quotasFor[operation][regime] = options.regimeFloor;
            focusFor[operation].emplace(regime, std::move(focus));
            floorsFor[operation].insert(regime);
        }
    }

    // `benchmark` ids are content-derived, so exclusion is an exact set difference.
    std::set<std::string> excluded;
    for(const auto& path : options.excludeCorpora)
    {
        std::ifstream file(path);
        if(!file)
        {
            std::cerr << "cannot read --exclude-corpus " << path << "\n";
            return 1;
        }
        nlohmann::json manifest;
        file >> manifest;
        for(const auto& graph : manifest.value("graphs", nlohmann::json::array()))
        {
            if(graph.contains("benchmark"))
            {
                excluded.insert(graph.at("benchmark").get<std::string>());
            }
        }
    }

    // Only with an engine, so engine-free runs need no GPU. MIOpen's AI solver predictor
    // dominates applicability queries without changing the solution count this tool reads, so
    // disable it unless the caller set it.
    if(options.haveEngineId)
    {
        // An empty value counts as unset.
        if(hipdnn_data_sdk::utilities::getEnv("MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK").empty())
        {
            hipdnn_data_sdk::utilities::setEnv("MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK", "0");
        }
    }

    hipdnnHandle_t handle = nullptr;
    if(options.haveEngineId && hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create a hipDNN handle\n";
        return 1;
    }
    const auto release = [&handle]() {
        if(handle != nullptr)
        {
            hipdnnDestroy(handle);
        }
    };

    std::string resolvedEngine = options.engineName;
    if(handle != nullptr)
    {
        const auto engines = loadedEngines(handle);
        const auto requested = std::find_if(engines.begin(), engines.end(), [&](const auto& e) {
            return options.engineName.empty() ? e.first == options.engineId
                                              : e.second == options.engineName;
        });
        if(requested == engines.end())
        {
            std::cerr << "Engine "
                      << (options.engineName.empty() ? std::to_string(options.engineId)
                                                     : "'" + options.engineName + "'")
                      << " is not registered by any loaded plugin";
            if(options.pluginDirs.empty())
            {
                std::cerr << " (no --plugin-dir was given, so only the default search path "
                             "was loaded)";
            }
            std::cerr << ".\n";
            if(engines.empty())
            {
                std::cerr << "No engines are loaded at all: check --plugin-dir.\n";
            }
            else
            {
                std::cerr << "Loaded engines:\n";
                for(const auto& engine : engines)
                {
                    std::fprintf(stderr,
                                 "  %s (0x%016llX)\n",
                                 engine.second.c_str(),
                                 static_cast<unsigned long long>(engine.first));
                }
            }
            release();
            return 1;
        }
        // Use the registered id rather than the name's hash.
        options.engineId = requested->first;
        resolvedEngine = requested->second;
        for(const auto& name : options.alsoEngineNames)
        {
            const auto also = std::find_if(
                engines.begin(), engines.end(), [&](const auto& e) { return e.second == name; });
            if(also == engines.end())
            {
                std::cerr << "Engine '" << name << "' (--also-engine-name) is not registered by "
                          << "any loaded plugin.\n";
                release();
                return 1;
            }
            options.alsoEngineIds.push_back(also->first);
        }
    }
    else if(!options.alsoEngineNames.empty())
    {
        std::cerr << "--also-engine-name needs an engine to ask: it cannot be combined with "
                     "--without-engine\n";
        release();
        return 1;
    }

    // Recorded engine coverage from `engines.json`; absent means search every engine.
    hipdnn_corpus_gen::EngineCoverageEntry coverage;
    std::string coverageEngine = resolvedEngine;
    {
        const auto tablePath = std::filesystem::path(options.operationsDir) / "engines.json";
        if(std::filesystem::exists(tablePath))
        {
            hipdnn_corpus_gen::EngineCoverageTable table;
            std::string tableError;
            std::ifstream tableFile(tablePath);
            nlohmann::json tableDocument;
            try
            {
                tableFile >> tableDocument;
            }
            catch(const std::exception& error)
            {
                tableError = error.what();
            }
            if(!tableError.empty()
               || !hipdnn_corpus_gen::parseEngineCoverage(tableDocument, table, tableError))
            {
                std::cerr << tablePath.string() << ": " << tableError << "\n";
                release();
                return 1;
            }
            // The corpus must be served by every named engine, so the narrowest coverage wins.
            std::vector<std::string> named{resolvedEngine};
            named.insert(
                named.end(), options.alsoEngineNames.begin(), options.alsoEngineNames.end());
            for(const auto& name : named)
            {
                const auto known = table.find(name);
                if(known != table.end()
                   && (coverage.coverage != hipdnn_corpus_gen::EngineCoverage::PACK))
                {
                    coverage = known->second;
                    coverageEngine = name;
                }
            }
        }
    }
    const bool coverageIsPack = coverage.coverage == hipdnn_corpus_gen::EngineCoverage::PACK;
    if(coverageIsPack)
    {
        if(options.packRoots.empty())
        {
            std::cerr << coverageEngine
                      << " serves exactly its pack's shapes (engines.json: " << coverage.reason
                      << ")\n"
                      << "so its corpus comes from the pack: pass --kdp-root.\n";
            release();
            return 1;
        }
        std::cerr << coverageEngine << ": coverage is its pack (engines.json); no search is run.\n";
    }

    if(!options.probe.empty())
    {
        // Explain what happens to one point at each stage. Probe fields name no operation, so
        // exactly one must be selected.
        if(selected.operations.size() != 1)
        {
            std::cerr << "--probe needs --operation: " << selected.operations.size()
                      << " declarations are loaded and a probe names no operation\n";
            release();
            return 1;
        }

        ProblemPoint point;
        std::stringstream fields(options.probe);
        std::string field;
        while(std::getline(fields, field, ','))
        {
            const auto split = field.find('=');
            const auto name = field.substr(0, split);
            const auto text = field.substr(split + 1);
            const auto* parameter = selected.operations.front().second.find(name);
            if(parameter != nullptr && parameter->type == hipdnn_corpus_gen::ParameterType::ENUM)
            {
                point[name] = text;
            }
            else if(parameter != nullptr
                    && parameter->type == hipdnn_corpus_gen::ParameterType::BOOL)
            {
                // Parse bools as bools; an integer `1` would build a graph with the flag unset.
                point[name] = text == "true" || text == "1";
            }
            else
            {
                point[name] = static_cast<int64_t>(std::strtoll(text.c_str(), nullptr, 10));
            }
        }

        const auto& metadata = selected.operations.front().second;
        std::cout << "constraints: "
                  << (hipdnn_corpus_gen::detail::satisfiesConstraints(metadata, point) ? "satisfied"
                                                                                       : "REFUSED")
                  << "\n";

        const auto built = hipdnn_corpus_gen::buildGraphFor(metadata, point);
        std::cout << "build: " << (built.ok() ? "ok" : built.error) << "\n";
        if(built.ok())
        {
            hipdnn_frontend::graph::Graph graph;
            const auto restored = graph.deserialize(handle, built.bytes);
            std::cout << "deserialize: " << (restored.is_good() ? "ok" : restored.get_message())
                      << "\n";
            if(restored.is_good())
            {
                const auto finalized = graph.build_operation_graph(handle);
                std::cout << "finalize: " << (finalized.is_good() ? "ok" : finalized.get_message())
                          << "\n";
                if(finalized.is_good())
                {
                    std::string asJson;
                    if(graph.serialize(asJson).is_good())
                    {
                        std::cout << "graph: " << asJson << "\n";
                    }
                    std::vector<int64_t> engines;
                    const auto ranked = graph.get_ranked_engine_ids(engines);
                    std::cout << "engines: "
                              << (ranked.is_good() ? std::to_string(engines.size())
                                                   : ranked.get_message())
                              << "\n";
                    for(const auto id : engines)
                    {
                        std::printf("  0x%016llX%s\n",
                                    static_cast<unsigned long long>(id),
                                    id == options.engineId ? "  <- requested" : "");
                    }
                }
            }
        }
        release();
        return 0;
    }

    // Filters run inside the search so its target is spent on surviving points. The exclusion
    // id must be computed exactly as emission computes it.
    int64_t heldOutDuringSearch = 0;
    const hipdnn_corpus_gen::CorpusFilter searchFilter = [&](const std::string& operation,
                                                             const ProblemPoint& point) {
        if(!hipdnn_corpus_gen::keeps(keep, point))
        {
            return false;
        }
        if(excluded.empty())
        {
            return true;
        }
        const auto declaration
            = std::find_if(selected.operations.begin(),
                           selected.operations.end(),
                           [&](const auto& entry) { return entry.second.operation == operation; });
        if(declaration == selected.operations.end())
        {
            return true;
        }
        const auto graph = hipdnn_corpus_gen::buildGraphFor(declaration->second, point);
        if(!graph.ok())
        {
            return true; // not this filter's refusal; the oracle reports build failures
        }
        const auto id
            = hipdnn_corpus_gen::stampGraphIdentity(
                  graph.bytes,
                  graphNameFor(
                      operation, hipdnn_corpus_gen::regimeLabel(declaration->second, point), point))
                  .id;
        if(excluded.count(id) > 0)
        {
            ++heldOutDuringSearch;
            return false;
        }
        return true;
    };

    const auto start = std::chrono::steady_clock::now();

    // Every pack is offered to every operation; a declaration accepts a pack by being able to
    // read its fields.
    const auto packPaths = hipdnn_corpus_gen::discoverPacks(options.packRoots);

    int64_t total = 0;
    int64_t requested = 0;
    std::vector<hipdnn_corpus_gen::ManifestEntry> manifestRows;
    /// Keyed `<operation>|<point>`: two operations may describe a point identically.
    std::map<std::string, hipdnn_corpus_gen::IdentifiedGraph> stamped;
    std::map<std::string, int64_t> allocationTotals;
    std::map<std::string, int64_t> droppedTotals;
    nlohmann::json sourceReports = nlohmann::json::object();
    int64_t excludedRows = 0;
    std::vector<std::string> shortfall;
    bool searchCapped = false;
    /// A shortfall not shown to be saturation; only saturation may return fewer than `--count`.
    bool shortfallUnproven = false;
    /// Per operation and regime: what a quota asked, what the pools held before a focused
    /// search, what the search added, and what the cut took.
    nlohmann::json quotaReports = nlohmann::json::object();
    /// A quota left short without its focused search being shown saturated.
    std::vector<std::string> quotaShort;
    /// Per operation: each numeric parameter's served extent, and the extremes taken.
    nlohmann::json servedExtent = nlohmann::json::object();
    bool searchFoundMore = false;
    std::ofstream commands;
    std::filesystem::path root;
    if(!options.outputDir.empty())
    {
        root = options.outputDir;
        std::filesystem::create_directories(root / "graphs");
        commands.open(root / "commands.txt");
        if(!commands)
        {
            std::cerr << "Cannot write " << (root / "commands.txt") << "\n";
            release();
            return 1;
        }
        commands << "# hipdnn_bench invocations, one per problem, for "
                 << (options.engineName.empty() ? "NO ENGINE -- applicability is declared, "
                                                  "not tested (--without-engine)"
                                                : options.engineName)
                 << "\n"
                 << "# Generated from declarations in " << options.operationsDir << "\n"
                 << "# Train on the times, not the rank column: configurations are often\n"
                 << "# separated by less than run-to-run variation.\n";
    }

    for(const auto& operationEntry : selected.operations)
    {
        const auto& metadata = operationEntry.second;
        hipdnn_corpus_gen::OracleTiming timing;
        const auto searchStart = std::chrono::steady_clock::now();
        hipdnn_corpus_gen::MetadataOperationCorpus result;
        result.metadataPath = operationEntry.first;
        result.operation = metadata.operation;

        // Every source passes the same engine test; a problem is served only if every named
        // engine serves it. The byte ceiling applies only to the search: pack and model shapes
        // are real workloads.
        std::vector<int64_t> askedEngines{options.engineId};
        askedEngines.insert(
            askedEngines.end(), options.alsoEngineIds.begin(), options.alsoEngineIds.end());
        const auto everyEngine = [&](int64_t maxBytes) {
            std::vector<hipdnn_corpus_gen::ProblemOracle> each;
            each.reserve(askedEngines.size());
            for(const auto id : askedEngines)
            {
                each.push_back(hipdnn_corpus_gen::makeCorpusOracle(handle,
                                                                   id,
                                                                   metadata,
                                                                   &result.buildFailures,
                                                                   &result.firstBuildError,
                                                                   maxBytes,
                                                                   &timing));
            }
            return hipdnn_corpus_gen::allOf(std::move(each));
        };
        const auto oracle = everyEngine(/*maxBytes=*/0);
        const auto searchOracle = everyEngine(options.maxBytes);

        // Graphs are built and stamped at admission, before selection, so `--exclude-corpus`
        // filters by final id and `--count` is not left short afterwards.
        const auto admit = [&](hipdnn_corpus_gen::PoolEntry& entry, bool alreadyAdmitted) {
            if(!hipdnn_corpus_gen::keeps(keep, entry.point))
            {
                return false;
            }
            const auto key
                = result.operation + "|" + hipdnn_corpus_gen::detail::describe(entry.point);
            auto known = stamped.find(key);
            if(known == stamped.end())
            {
                if(!alreadyAdmitted && !oracle(entry.point))
                {
                    return false;
                }
                const auto graph = hipdnn_corpus_gen::buildGraphFor(metadata, entry.point);
                if(!graph.ok())
                {
                    return false;
                }
                // Stamp the identity before writing: an id-less graph gets a random id from
                // `GraphDescriptor::finalize`, changing every run. See GraphIdentity.hpp.
                known = stamped
                            .emplace(key,
                                     hipdnn_corpus_gen::stampGraphIdentity(
                                         graph.bytes,
                                         graphNameFor(result.operation, entry.regime, entry.point)))
                            .first;
            }
            if(excluded.count(known->second.id) > 0)
            {
                ++excludedRows;
                return false;
            }
            return true;
        };

        // Pack and model shapes are admitted first (finite and cheap); the search then fills
        // only the remainder, excluding but still walking through already-pooled points. A
        // source with share 0 is not collected at all, or it would displace enabled sources.
        const auto enabled = [&](const char* source) {
            return hipdnn_corpus_gen::sourceEnabled(options.shares, source);
        };
        hipdnn_corpus_gen::SourcePools pools;
        std::set<std::string> pooled;
        if(enabled("kernel"))
        {
            auto harvested = hipdnn_corpus_gen::collectPacks(metadata, packPaths, /*maxBytes=*/0);
            for(auto& entry : harvested.first)
            {
                if(admit(entry, false))
                {
                    pooled.insert(hipdnn_corpus_gen::detail::describe(entry.point));
                    pools["kernel"].push_back(std::move(entry));
                }
            }
            for(const auto& report : harvested.second)
            {
                sourceReports["packs"].push_back(report.asJson());
                if(!report.shutOut.empty())
                {
                    std::cerr << "  " << report.shutOut << "\n";
                }
            }
        }

        for(const auto& path : options.modelShapes)
        {
            if(!enabled("model"))
            {
                continue;
            }
            hipdnn_corpus_gen::ModelShapeReport report;
            auto shapes = hipdnn_corpus_gen::readModelShapes(metadata, path, report);
            for(auto& entry : shapes)
            {
                if(admit(entry, false))
                {
                    pooled.insert(hipdnn_corpus_gen::detail::describe(entry.point));
                    pools["model"].push_back(std::move(entry));
                }
            }
            sourceReports["model_shapes"].push_back(
                nlohmann::json{{"path", report.path.string()},
                               {"op", result.operation},
                               {"rows", report.rows},
                               {"other_operation", report.otherOperation},
                               {"unusable", report.unusable},
                               {"first_problem", report.firstProblem}});
        }

        auto request = options.exploration;
        request.corpusTarget
            = options.count > 0
                  ? std::max<int64_t>(0, options.count - static_cast<int64_t>(pooled.size()))
                  : 0;
        const hipdnn_corpus_gen::ProblemOracle admits = [&](const ProblemPoint& point) {
            return searchFilter(result.operation, point) && searchOracle(point);
        };
        const hipdnn_corpus_gen::ProblemOracle alreadyPooled = [&](const ProblemPoint& point) {
            return pooled.count(hipdnn_corpus_gen::detail::describe(point)) > 0;
        };
        // A pack-coverage engine serves exactly its pack, so it is not searched; neither is a
        // sweep with no share.
        const bool searched = !coverageIsPack && enabled("sweep");
        if(searched)
        {
            result.corpus
                = hipdnn_corpus_gen::exploreProblemSpace(metadata, request, admits, alreadyPooled);
        }
        else
        {
            result.corpus.operation = metadata.operation;
        }

        const auto problems = result.corpus.problems();
        if(timing.queries > 0)
        {
            const auto wall
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart)
                      .count();
            const auto perQuery = [&timing](double seconds) {
                return seconds * 1e6 / static_cast<double>(timing.queries);
            };
            std::fprintf(stderr,
                         "%s: %lld engine queries in %.1f s of %.1f s -- per query: build %.0f us, "
                         "load %.0f us, ask %.0f us\n",
                         result.operation.c_str(),
                         static_cast<long long>(timing.queries),
                         timing.buildSeconds + timing.loadSeconds + timing.askSeconds,
                         wall,
                         perQuery(timing.buildSeconds),
                         perQuery(timing.loadSeconds),
                         perQuery(timing.askSeconds));
        }
        std::cerr << result.operation << ": " << problems.size() << " problems";
        if(result.buildFailures > 0)
        {
            // A metadata bug, not an engine refusal.
            std::cerr << " (" << result.buildFailures
                      << " failed to build: " << result.firstBuildError << ")";
        }
        // Distinct feasible points reached, and cells the corpus spreads them over.
        for(const auto& combination : result.corpus.combinations)
        {
            if(combination.stats.distinct > 0)
            {
                std::cerr << "\n    "
                          << hipdnn_corpus_gen::detail::describe(combination.categorical) << ": "
                          << combination.stats.distinct << " distinct feasible, "
                          << combination.stats.cellsOccupied << "/" << combination.stats.cells
                          << " cells";
            }
        }
        if(searched
           && (result.corpus.constraintRejections > 0 || result.corpus.constraintAdmissions == 0))
        {
            std::cerr << " [constraints admitted " << result.corpus.constraintAdmissions
                      << ", refused " << result.corpus.constraintRejections << "]";
        }
        for(const auto& skipped : result.corpus.skippedCombinations)
        {
            std::cerr << "\n  " << skipped;
        }
        // Sweep shortfalls are reported only if the corpus ends up short after selection.
        for(const auto& reason : result.corpus.shortfall)
        {
            shortfall.push_back(result.operation + ": " + reason);
        }
        for(const auto& combination : result.corpus.combinations)
        {
            searchCapped = searchCapped || combination.searchCapped;
            searchFoundMore = searchFoundMore || !combination.problems.empty();
            if(!result.corpus.shortfall.empty() && !combination.problems.empty()
               && !combination.saturated)
            {
                shortfallUnproven = true;
            }
        }
        std::cerr << "\n";

        for(size_t i = 0; i < problems.size(); ++i)
        {
            hipdnn_corpus_gen::PoolEntry entry;
            entry.point = problems[i];
            entry.source = "sweep";
            entry.origin = result.operation + " draw " + std::to_string(i);
            entry.regime = hipdnn_corpus_gen::regimeLabel(metadata, entry.point);
            if(admit(entry, true))
            {
                pools["sweep"].push_back(std::move(entry));
            }
        }

        // Under-filled regimes are searched again with their declared equalities pinned, for
        // the shortfall only, through the same engine and filters.
        const auto quotas = quotasFor.find(result.operation);
        std::map<std::string, hipdnn_corpus_gen::RegimeSearchResult> focused;
        std::map<std::string, int64_t> pooledBefore;
        if(quotas != quotasFor.end())
        {
            std::vector<ProblemPoint> anchors;
            std::map<std::string, std::set<std::string>> byRegime;
            std::set<std::string> everything;
            for(const auto& pool : pools)
            {
                for(const auto& entry : pool.second)
                {
                    const auto key = hipdnn_corpus_gen::detail::describe(entry.point);
                    byRegime[entry.regime].insert(key);
                    everything.insert(key);
                    anchors.push_back(entry.point);
                }
            }
            const hipdnn_corpus_gen::ProblemOracle alreadyHave = [&](const ProblemPoint& point) {
                return everything.count(hipdnn_corpus_gen::detail::describe(point)) > 0;
            };
            for(const auto& [regime, asked] : quotas->second)
            {
                const auto have = static_cast<int64_t>(byRegime[regime].size());
                pooledBefore[regime] = have;
                if(have >= asked || coverageIsPack)
                {
                    // A pack-coverage engine serves only its pack; searching cannot help.
                    continue;
                }
                auto found
                    = hipdnn_corpus_gen::exploreRegime(metadata,
                                                       focusFor.at(result.operation).at(regime),
                                                       options.exploration,
                                                       asked - have,
                                                       admits,
                                                       alreadyHave,
                                                       anchors);
                size_t draw = 0;
                for(const auto& point : found.problems)
                {
                    hipdnn_corpus_gen::PoolEntry entry;
                    entry.point = point;
                    entry.source = "sweep";
                    entry.origin
                        = result.operation + " focus " + regime + " draw " + std::to_string(draw++);
                    entry.regime = hipdnn_corpus_gen::regimeLabel(metadata, entry.point);
                    if(admit(entry, true))
                    {
                        everything.insert(hipdnn_corpus_gen::detail::describe(entry.point));
                        pools["sweep"].push_back(std::move(entry));
                    }
                }
                const char* stop = "";
                if(found.saturated)
                {
                    stop = ", saturated";
                }
                else if(found.searchCapped)
                {
                    stop = ", budget limit";
                }
                std::cerr << "  focus " << regime << ": " << have << " pooled, " << asked - have
                          << " wanted, " << found.problems.size() << " found (" << found.inRegime
                          << " of " << found.proposed << " proposals in the regime)" << stop
                          << "\n";
                focused.emplace(regime, std::move(found));
            }
        }

        // The served edges, reported always and taken first with --include-extremes.
        std::set<std::string> reserved;
        servedExtent[result.operation] = reportServedExtent(
            metadata,
            result.corpus,
            pools,
            options.includeExtremes,
            searched,
            options.exploration.numericCeiling,
            reserved,
            [&](hipdnn_corpus_gen::PoolEntry& entry) {
                if(!admit(entry, true))
                {
                    return std::string(); // held out
                }
                return stamped
                    .at(result.operation + "|" + hipdnn_corpus_gen::detail::describe(entry.point))
                    .id;
            });

        // Spread a cut over categorical combinations as well as regimes.
        for(auto& pool : pools)
        {
            for(auto& entry : pool.second)
            {
                std::string stratum;
                for(const auto& parameter : metadata.parameters)
                {
                    if(parameter.type != hipdnn_corpus_gen::ParameterType::ENUM
                       && parameter.type != hipdnn_corpus_gen::ParameterType::BOOL)
                    {
                        continue;
                    }
                    const auto held = entry.point.find(parameter.name);
                    if(held != entry.point.end())
                    {
                        stratum
                            += parameter.name + "=" + hipdnn_corpus_gen::asText(held->second) + ",";
                    }
                }
                entry.stratum = stratum + "|" + entry.regime;
            }
        }

        std::map<std::string, int64_t> dropped;
        const auto deduplicated = hipdnn_corpus_gen::deduplicate(pools, options.shares, dropped);

        // 0 means everything the pools hold. Larger requests are never filled: `allocate` is
        // capped by pool capacity and the shortfall is recorded against `requested`.
        int64_t count = options.count;
        if(count == 0)
        {
            for(const auto& pool : deduplicated)
            {
                count += static_cast<int64_t>(pool.second.size());
            }
        }

        std::map<std::string, int64_t> allocation;
        std::map<std::string, hipdnn_corpus_gen::RegimeQuotaOutcome> quotaOutcome;
        const auto& owed
            = quotas != quotasFor.end() ? quotas->second : std::map<std::string, int64_t>{};
        // With quotas and count 0, the corpus is the quotas alone; a short quota stays short
        // rather than being padded from another regime.
        int64_t cut = count;
        if(!owed.empty() && options.count == 0)
        {
            cut = 0;
            count = 0;
            for(const auto& quota : owed)
            {
                count += quota.second;
            }
        }
        const auto chosen = hipdnn_corpus_gen::select(
            deduplicated, cut, options.shares, allocation, owed, quotaOutcome, reserved);
        for(const auto& [regime, outcome] : quotaOutcome)
        {
            const auto search = focused.find(regime);
            const bool focusSearched = search != focused.end();
            const bool saturated = coverageIsPack || (focusSearched && search->second.saturated);
            const bool floor = floorsFor[result.operation].count(regime) > 0;
            quotaReports[result.operation][regime] = nlohmann::json{
                {"floor", floor},
                {"asked", outcome.asked},
                {"delivered", outcome.taken},
                {"pooled_before_focus", pooledBefore[regime]},
                {"found_by_focus", focusSearched ? search->second.problems.size() : 0},
                {"focus_proposals", focusSearched ? search->second.proposed : 0},
                {"focus_proposals_in_regime", focusSearched ? search->second.inRegime : 0},
                {"saturated", saturated},
                {"search_capped", focusSearched && search->second.searchCapped}};
            if(outcome.taken < outcome.asked && !saturated && !floor)
            {
                quotaShort.push_back(result.operation + " " + regime + ": "
                                     + std::to_string(outcome.taken) + " of "
                                     + std::to_string(outcome.asked)
                                     + (searched && search->second.searchCapped
                                            ? " (focused search stopped at its budget limit while "
                                              "still finding points; raise --budget)"
                                            : " (not shown saturated)"));
            }
        }

        requested += count;
        for(const auto& share : allocation)
        {
            allocationTotals[share.first] += share.second;
        }
        for(const auto& loss : dropped)
        {
            droppedTotals[loss.first] += loss.second;
        }

        if(root.empty())
        {
            for(const auto& entry : chosen)
            {
                std::cout << asQueryColumns(entry.point, false) << "\n";
            }
            total += static_cast<int64_t>(chosen.size());
            continue;
        }

        // Maps problem id to q.* values without re-running the generator.
        std::ofstream index(root / (result.operation + ".problems.csv"));
        bool wroteHeader = false;

        for(const auto& entry : chosen)
        {
            const auto& graph = stamped.at(result.operation + "|"
                                           + hipdnn_corpus_gen::detail::describe(entry.point));

            // Stem must equal manifest `name`: scoring joins bench output by stem.
            const auto name = graph.name + ".fb";
            const auto graphPath = root / "graphs" / name;
            std::ofstream problem(graphPath, std::ios::binary);
            problem.write(reinterpret_cast<const char*>(graph.bytes.data()),
                          static_cast<std::streamsize>(graph.bytes.size()));
            problem.close();
            // The manifest lists this file, so a corpus missing its bytes must not be written.
            if(!problem)
            {
                throw std::runtime_error("Failed to write graph file: " + graphPath.string());
            }

            hipdnn_corpus_gen::ManifestEntry row;
            row.entry = entry;
            row.benchmark = graph.id;
            row.name = graph.name;
            row.file = "graphs/" + name;
            row.operation = result.operation;
            row.regimeAxes = metadata.regimeLabel;
            // Tensor footprint, not file size.
            row.bytes = hipdnn_corpus_gen::graphBytes(graph.bytes);
            manifestRows.push_back(std::move(row));

            if(!wroteHeader)
            {
                index << "problem," << asQueryColumns(entry.point, true) << "\n";
                wroteHeader = true;
            }
            index << graph.name << "," << asQueryColumns(entry.point, false) << "\n";

            commands << options.benchPath;
            for(const auto& dir : options.pluginDirs)
            {
                commands << " --plugin-dir " << dir;
            }
            commands << " --graph " << graphPath.string();
            if(!options.engineName.empty())
            {
                commands << " --engine-name " << options.engineName;
            }
            commands << " --sweep --no-header" << " --problem-id " << graph.name << " --query "
                     << asQueryArgument(entry.point) << "\n";
            ++total;
        }
    }

    excludedRows += heldOutDuringSearch;
    if(excludedRows > 0)
    {
        std::cerr << "Held out " << excludedRows << " problem(s) already in the excluded corpora"
                  << "\n";
    }

    if(!root.empty())
    {
        // Written even when empty, so the run's inputs are still recorded.
        hipdnn_corpus_gen::ManifestContext manifest;
        manifest.seed = options.exploration.seed;
        // What was asked for; never filled, so the gap to the row count is the shortfall.
        manifest.requested = requested;
        manifest.allocation = allocationTotals;
        manifest.duplicatesDropped = droppedTotals;
        for(const auto& entry : selected.operations)
        {
            manifest.operations.push_back(entry.second.operation);
            manifest.inputs.emplace_back(entry.first);
        }
        for(const auto& path : packPaths)
        {
            manifest.inputs.push_back(path);
        }
        for(const auto& path : options.modelShapes)
        {
            manifest.inputs.push_back(path);
        }
        manifest.reports = sourceReports;
        manifest.reports["engine"]
            = options.engineName.empty() ? nlohmann::json() : nlohmann::json(options.engineName);
        if(!options.alsoEngineNames.empty())
        {
            manifest.reports["also_engines"] = options.alsoEngineNames;
        }
        manifest.reports["excluded"] = excludedRows;
        if(!quotaReports.empty())
        {
            manifest.reports["regime_quota"] = quotaReports;
        }
        if(!servedExtent.empty())
        {
            manifest.reports["served_extent"] = servedExtent;
        }
        if(options.count > 0 && total < requested && !shortfall.empty())
        {
            manifest.reports["shortfall"] = shortfall;
        }

        // Rows alone cannot tell an engine-verified corpus from a declaration-wide one.
        manifest.reports["engine_verified"] = options.haveEngineId;
        if(coverageIsPack)
        {
            manifest.reports["coverage"]
                = nlohmann::json{{"kind", "pack"}, {"reason", coverage.reason}};
        }
        if(options.withoutEngine)
        {
            manifest.reports["warning"]
                = "Generated with --without-engine: no engine was asked whether it serves "
                  "these problems. Applicability here is declared, not tested, and any "
                  "--keep/--kdp-root narrowing is unverified.";
        }

        hipdnn_corpus_gen::writeCorpusManifest(root, manifestRows, manifest);
    }

    const auto elapsed
        = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "Generated " << total << " problems in " << elapsed << " s\n";
    release();
    if(total == 0)
    {
        return 2;
    }
    if(!quotaShort.empty())
    {
        std::cerr << "SHORT of regime quotas:\n";
        for(const auto& line : quotaShort)
        {
            std::cerr << "  " << line << "\n";
        }
        return 3;
    }
    if(coverageIsPack)
    {
        if(options.count > 0 && total < requested)
        {
            std::cerr << "Corpus is the engine's whole coverage: " << total << " of " << requested
                      << " requested. " << coverageEngine
                      << " serves only its pack's shapes (engines.json).\n";
        }
        return 0;
    }
    if(options.count > 0 && total < requested)
    {
        std::cerr << "SHORT: " << total << " of " << requested << " requested problems.\n";
        for(const auto& reason : shortfall)
        {
            std::cerr << "  " << reason << "\n";
        }
        if(searchCapped || shortfallUnproven || shortfall.empty())
        {
            std::cerr << "  Not shown to be all the engine serves"
                      << (searchCapped ? ": a search reached its budget limit while still "
                                         "finding problems. Raise --budget."
                                       : "; see the SHORT lines above.")
                      << "\n";
            return 3;
        }
        if(!searchFoundMore)
        {
            std::cerr << "  The search found no served problem outside the pack and model shapes "
                         "already taken:\n"
                      << "  every problem this engine was found to serve is in the corpus.\n";
            return 0;
        }
        std::cerr << "  Every served combination saturated: doubling the search found no new "
                     "point.\n"
                  << "  That is the search's limit, not proof of the engine's: an engine whose "
                     "kernels are\n"
                  << "  compiled per exact shape serves isolated points a walk cannot step "
                     "between.\n"
                  << "  --kdp-root proposes those points directly.\n";
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        return runGenerator(std::vector<std::string>(argv, argv + argc));
    }
    catch(const std::exception& error)
    {
        std::cerr << "hipdnn_corpus_gen failed: " << error.what() << "\n";
        return 1;
    }
}
