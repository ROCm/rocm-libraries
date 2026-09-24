// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file CorpusGen.cpp
 * @brief Generates an engine's problem corpus from declarations (RFC 0019.13 §4, §5).
 *
 * Inputs: a directory of operation declarations, and an engine. Output: the problems that
 * engine accepts, as serialized graphs, plus one benchmark invocation per problem.
 *
 * Nothing here knows what a convolution is. The operations are `*.opmeta.json` files, the
 * exploration is the same for all of them, and the engine is consulted rather than modelled --
 * so an operation is added by writing a file and an engine is characterised by being asked.
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
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <hipdnn_frontend.hpp>
#include <sstream>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{

using hipdnn_corpus_gen::ExplorationRequest;
using hipdnn_corpus_gen::ProblemPoint;
using hipdnn_corpus_gen::asQueryArgument;
using hipdnn_corpus_gen::asQueryColumns;

/// @brief The name the graph document carries: the operation, its regime, and its parameters.
///
/// Also the file's stem, the `--problem-id` in `commands.txt` and the key in `<op>.problems.csv`:
/// one name everywhere, because scoring joins bench output by file stem to measurements by
/// manifest `name`. It is the key an L2 collection joins on -- L2 mints its own graph ids, so the id this tool stamps is not
/// available there and the name is all that is left. That makes uniqueness the requirement:
/// every parameter is spelled out rather than abbreviated, because two problems that differ
/// only in a field the name elided would join to one row and be reported as one.
std::string graphNameFor(const std::string& operation,
                         const std::string& regime,
                         const ProblemPoint& point)
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
/// Used to refuse an engine nobody registered before any problem is offered to it. Without
/// that check a misspelled `--engine-name` still hashes to a well-formed id, every problem is
/// declined by every engine, and the run ends in an empty corpus that reads exactly like an
/// engine that serves nothing.
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
        if(hipdnnGetEngineInfo_ext(handle, index, &id, nullptr, &nameLength, nullptr,
                                   &pluginLength, nullptr, &versionLength, nullptr, &typeLength)
           != HIPDNN_STATUS_SUCCESS)
        {
            continue;
        }
        std::string name(nameLength, '\0');
        std::string plugin(pluginLength, '\0');
        std::string version(versionLength, '\0');
        std::string type(typeLength, '\0');
        if(hipdnnGetEngineInfo_ext(handle, index, nullptr, name.data(), &nameLength,
                                   plugin.data(), &pluginLength, version.data(), &versionLength,
                                   type.data(), &typeLength)
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

    /// Deliberate consent to generate without asking an engine anything. Naming an engine is
    /// otherwise required, because the alternative is a corpus whose applicability is inferred
    /// rather than tested, and that inference fails silently: which shapes an engine declines
    /// correlates with head_dim, dtype, causal and sequence length -- the same axes performance
    /// varies along -- so what survives benchmarking is a biased subsample, not a thinned one.
    bool withoutEngine = false;

    ExplorationRequest exploration;

    /// Where the kernel pool comes from: `*.kdp.json` packs, or directories holding them.
    std::vector<std::filesystem::path> packRoots;

    /// Where the model pool comes from: CSVs of `q.<parameter>` columns.
    std::vector<std::filesystem::path> modelShapes;

    /// Corpus size per operation, 0 meaning "everything the pools hold". A floor in intent and
    /// a ceiling in fact: an engine that serves 664 problems yields 664 whatever is asked for.
    int64_t count = 0;

    std::map<std::string, double> shares = hipdnn_corpus_gen::defaultShares();

    /// `q.<parameter>=<value>` clauses, applied to every source alike.
    std::vector<std::string> keep;

    /// Manifests whose graphs must not appear here -- the comparison set, held out by
    /// construction rather than by trusting a random split.
    std::vector<std::filesystem::path> excludeCorpora;

    /// Benchmarking ceiling in bytes across a problem's tensors. 256 MiB by default: large
    /// enough for real layers, small enough that no single problem dominates a corpus run.
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
              << "                         0 (default) takes what the first pass finds. NOTE:\n"
              << "                         this used to mean problems per combination -- that\n"
              << "                         is --per-combination.\n"
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
              << "  --budget <n>           First-pass oracle calls per combination (default\n"
              << "                         20000); growth toward --count may reach 64x this\n"
              << "  --ceiling <n>          Largest extent to propose (default 4096)\n"
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
            options.engineId
                = hipdnn_data_sdk::utilities::engineNameToId(options.engineName);
            options.haveEngineId = true;
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
            options.exploration.pointsPerCombination
                = std::strtoll(next().c_str(), nullptr, 10);
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
        else if(arg == "--budget")
        {
            options.exploration.budgetPerCombination
                = std::strtoll(next().c_str(), nullptr, 10);
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

/// Renders a problem point as `name=value` pairs for the benchmark's --query.
///
/// Passed on the command line rather than left in an index for the harvest to join against.
/// A timing whose parameters live in another file is a timing that can be joined to the wrong
/// problem, and nothing in the row would show it.

/// Renders a problem point as `q.*` columns, which is the half of a training row the corpus
/// owns and the form RFC 0019.13 §7 requires.

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
    // An engine is required, and the way out of it has to be said out loud. Generating without
    // one is not a cheaper way to do the same thing: the declared oracle answers "can this
    // operation express the problem", never "does anything run it", so the corpus is a superset
    // of what any engine serves. Narrowing it offline with --keep or --kdp-root only moves the
    // inference, it does not test it -- a pack records what was built, not what the matcher
    // accepts. The cost is not wasted GPU time (a declined problem fails at plan time, before a
    // single dispatch); it is that the survivors are skewed toward one end of the shape
    // distribution and nothing downstream reports the skew. AITER's first gfx950 L1 was trained
    // on such a corpus and under-predicted by 285 TFLOPS.
    if(options.haveEngineId && options.withoutEngine)
    {
        std::cerr << "--without-engine contradicts --engine-name/--engine-id; pick one\n";
        return 1;
    }
    if(!options.haveEngineId && !options.withoutEngine)
    {
        std::cerr
            << "--engine-name (or --engine-id) is required.\n"
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
        // A probe reports which engines rank a point, so without one there is nothing to
        // probe *for*; the declared half of the answer is what a plain run already prints.
        std::cerr << "--probe needs --engine-name (or --engine-id)\n";
        return 1;
    }

    const auto declarations
        = hipdnn_corpus_gen::loadOperationDirectory(options.operationsDir);
    for(const auto& error : declarations.errors)
    {
        // Reported, never skipped silently: a declaration that does not load is an operation
        // missing from the corpus, which looks identical to an engine that does not serve it.
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

    // Parsed against the declarations, not against a fixed list: the facets are whatever the
    // loaded operations declare, and a clause outside them is refused rather than ignored.
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

    // Held out by construction. `benchmark` is content-derived, so the check is a set
    // difference over ids -- a random split would leave the comparison graphs in the training
    // corpus often enough to flatter every model trained on it.
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

    // Created only when an engine was named. Without one the whole run is device-free, and
    // opening a handle would make a corpus that needs no GPU fail on a machine without one.
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
        // The registry's id, not the name's hash: they agree today, and taking the registered
        // value means they never have to.
        options.engineId = requested->first;
        resolvedEngine = requested->second;
    }

    // What shape generation records about an engine's coverage, beside the declarations.
    // Absent means nothing is recorded and every engine is searched.
    hipdnn_corpus_gen::EngineCoverageEntry coverage;
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
            const auto known = table.find(resolvedEngine);
            if(known != table.end())
            {
                coverage = known->second;
            }
        }
    }
    const bool coverageIsPack = coverage.coverage == hipdnn_corpus_gen::EngineCoverage::PACK;
    if(coverageIsPack)
    {
        if(options.packRoots.empty())
        {
            std::cerr << resolvedEngine << " serves exactly its pack's shapes (engines.json: "
                      << coverage.reason << ")\n"
                      << "so its corpus comes from the pack: pass --kdp-root.\n";
            release();
            return 1;
        }
        std::cerr << resolvedEngine << ": coverage is its pack (engines.json); no search is run.\n";
    }

    if(!options.probe.empty())
    {
        // One point, every stage named. The generator reports aggregates, and an aggregate
        // cannot say why a particular problem was refused -- which is the question that
        // actually arises when a corpus comes back empty.
        //
        // A probe names bare field values, so which operation they belong to has to come from
        // somewhere else. Taking the first loaded declaration would answer a question about an
        // operation the caller never named, and report it as though it were about theirs.
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
                // Parsed as the bool the declaration says it is. Read as an integer, `1` built a
                // graph with the flag unset, so a probe of a causal problem tested a plain one.
                point[name] = text == "true" || text == "1";
            }
            else
            {
                point[name] = static_cast<int64_t>(std::strtoll(text.c_str(), nullptr, 10));
            }
        }

        const auto& metadata = selected.operations.front().second;
        std::cout << "constraints: "
                  << (hipdnn_corpus_gen::detail::satisfiesConstraints(metadata, point)
                          ? "satisfied"
                          : "REFUSED")
                  << "\n";

        const auto built = hipdnn_corpus_gen::buildGraphFor(metadata, point);
        std::cout << "build: " << (built.ok() ? "ok" : built.error) << "\n";
        if(built.ok())
        {
            hipdnn_frontend::graph::Graph graph;
            const auto restored = graph.deserialize(handle, built.bytes);
            std::cout << "deserialize: "
                      << (restored.is_good() ? "ok" : restored.get_message()) << "\n";
            if(restored.is_good())
            {
                const auto finalized = graph.build_operation_graph(handle);
                std::cout << "finalize: "
                          << (finalized.is_good() ? "ok" : finalized.get_message()) << "\n";
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

    // Filters run inside the search, ahead of the engine, so the search's target is spent on
    // points that survive them. Exclusion needs the id the graph will be written under, which
    // is content-derived, so it is computed exactly as emission computes it.
    int64_t heldOutDuringSearch = 0;
    const hipdnn_corpus_gen::CorpusFilter searchFilter
        = [&](const std::string& operation, const ProblemPoint& point) {
              if(!hipdnn_corpus_gen::keeps(keep, point))
              {
                  return false;
              }
              if(excluded.empty())
              {
                  return true;
              }
              const auto declaration = std::find_if(
                  selected.operations.begin(),
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
                        graphNameFor(operation,
                                     hipdnn_corpus_gen::regimeLabel(declaration->second, point),
                                     point))
                        .id;
              if(excluded.count(id) > 0)
              {
                  ++heldOutDuringSearch;
                  return false;
              }
              return true;
          };

    const auto start = std::chrono::steady_clock::now();

    // Discovered once, offered to every operation: a pack says which operation it describes by
    // whether the declaration can read its fields at all, so there is no pack-to-operation
    // mapping to maintain and no list of packs per operation to keep in step.
    const auto packPaths = hipdnn_corpus_gen::discoverPacks(options.packRoots);

    int64_t total = 0;
    int64_t requested = 0;
    std::vector<hipdnn_corpus_gen::ManifestEntry> manifestRows;
    /// Keyed `<operation>|<point>`, because two operations may describe a point identically and
    /// their graphs are not interchangeable.
    std::map<std::string, hipdnn_corpus_gen::IdentifiedGraph> stamped;
    std::map<std::string, int64_t> allocationTotals;
    std::map<std::string, int64_t> droppedTotals;
    nlohmann::json sourceReports = nlohmann::json::object();
    int64_t excludedRows = 0;
    std::vector<std::string> shortfall;
    bool searchCapped = false;
    /// A shortfall some combination did not demonstrate was forced on it -- saturation is the
    /// only accepted reason to return fewer problems than `--count` asked for.
    bool shortfallUnproven = false;
    /// Whether the search found any served problem beyond the pack and model shapes.
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

        // One engine test for all three sources: a pack geometry or a recorded model shape the
        // engine declines is not a corpus row any more than a swept one is, and finding that out
        // at collection time costs a sweep.
        //
        // The byte ceiling is the search's alone. The search proposes extents up to the numeric
        // ceiling on every axis, so without one it would propose tensors no device can hold. The
        // pack and model lists are real workloads -- a pack geometry is a kernel the engine
        // ships -- and a default sized for the search dropped 133 of rocKE's 664 served shapes.
        const auto oracle = hipdnn_corpus_gen::makeCorpusOracle(handle,
                                                                options.engineId,
                                                                metadata,
                                                                &result.buildFailures,
                                                                &result.firstBuildError,
                                                                /*maxBytes=*/0,
                                                                &timing);
        const auto searchOracle = hipdnn_corpus_gen::makeCorpusOracle(handle,
                                                                      options.engineId,
                                                                      metadata,
                                                                      &result.buildFailures,
                                                                      &result.firstBuildError,
                                                                      options.maxBytes,
                                                                      &timing);

        // Every admitted point's graph, built once here and looked up again at emission.
        // Stamping before selection is what makes `--exclude-corpus` exact: the id is the key
        // the held-out set names, so dropping an excluded point afterwards would leave
        // `--count` short and say nothing about why.
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
                // Named and identified before it is written, never after. A graph that reaches
                // the bench without an id is not rejected -- `GraphDescriptor::finalize` mints
                // a random v4 for it -- so the corpus would be measured under a different
                // identity on every run and nothing would report an error. See
                // GraphIdentity.hpp.
                known = stamped
                            .emplace(key,
                                     hipdnn_corpus_gen::stampGraphIdentity(
                                         graph.bytes,
                                         graphNameFor(result.operation, entry.regime,
                                                      entry.point)))
                            .first;
            }
            if(excluded.count(known->second.id) > 0)
            {
                ++excludedRows;
                return false;
            }
            return true;
        };

        // Three pools, one admission. The sweep says what the declaration can express, the
        // pack says what the engine was compiled for, and the model shapes say what anyone
        // runs; none of the three is a superset of the others.
        //
        // Pack and model shapes first, because they are finite lists and cheap to check: every
        // one the engine accepts is a problem the search does not have to find. The search is
        // then grown only toward what they left short of `--count`, and what it returns excludes
        // points they already hold. It still walks through those points: they are served, and
        // treating them as refused would cut the walk off from exactly the region it is in.
        hipdnn_corpus_gen::SourcePools pools;
        std::set<std::string> pooled;
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

        for(const auto& path : options.modelShapes)
        {
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
        if(coverageIsPack)
        {
            // The pack is the whole of what this engine serves; a search could only
            // rediscover it. Nothing to explore, and nothing short about that.
            result.corpus.operation = metadata.operation;
        }
        else
        {
            result.corpus = hipdnn_corpus_gen::exploreProblemSpace(metadata, request, admits,
                                                                   alreadyPooled);
        }

        const auto problems = result.corpus.problems();
        if(timing.queries > 0)
        {
            const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now()
                                                            - searchStart)
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
            // A metadata bug, not an engine refusal, and the difference matters: the first
            // makes an operation look unsupported when it is undeclared.
            std::cerr << " (" << result.buildFailures << " failed to build: "
                      << result.firstBuildError << ")";
        }
        // Coverage as measured, not asserted: how many distinct feasible points the search
        // reached, and how many cells the corpus spreads them over.
        for(const auto& combination : result.corpus.combinations)
        {
            if(combination.stats.distinct > 0)
            {
                std::cerr << "\n    " << hipdnn_corpus_gen::detail::describe(combination.categorical)
                          << ": " << combination.stats.distinct << " distinct feasible, "
                          << combination.stats.cellsOccupied << "/" << combination.stats.cells
                          << " cells";
            }
        }
        if(result.corpus.constraintRejections > 0 || result.corpus.constraintAdmissions == 0)
        {
            std::cerr << " [constraints admitted " << result.corpus.constraintAdmissions
                      << ", refused " << result.corpus.constraintRejections << "]";
        }
        for(const auto& skipped : result.corpus.skippedCombinations)
        {
            std::cerr << "\n  " << skipped;
        }
        // Collected, not printed: the sweep is short against its own remainder, and whether the
        // corpus is short is only known after selection. Reported only if it is.
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
            entry.point  = problems[i];
            entry.source = "sweep";
            entry.origin = result.operation + " draw " + std::to_string(i);
            entry.regime = hipdnn_corpus_gen::regimeLabel(metadata, entry.point);
            if(admit(entry, true))
            {
                pools["sweep"].push_back(std::move(entry));
            }
        }


        std::map<std::string, int64_t> dropped;
        const auto deduplicated = hipdnn_corpus_gen::deduplicate(pools, dropped);

        // 0 means everything the pools hold, which is the honest default: a corpus is bounded
        // by what the engine serves, not by a number anyone picked. Asking for more than that
        // does not invent shapes -- `allocate` is capped by each pool's capacity -- so the
        // shortfall is recorded against `requested` and nothing is filled.
        int64_t count = options.count;
        if(count == 0)
        {
            for(const auto& pool : deduplicated)
            {
                count += static_cast<int64_t>(pool.second.size());
            }
        }

        std::map<std::string, int64_t> allocation;
        const auto chosen
            = hipdnn_corpus_gen::select(deduplicated, count, options.shares, allocation);

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

        // An index beside the graphs, so a row's q.* values can be recovered from its problem
        // id without re-running the generator.
        std::ofstream index(root / (result.operation + ".problems.csv"));
        bool wroteHeader = false;

        for(size_t i = 0; i < chosen.size(); ++i)
        {
            const auto& entry = chosen[i];
            const auto& graph
                = stamped.at(result.operation + "|"
                             + hipdnn_corpus_gen::detail::describe(entry.point));

            // Stem == manifest `name`, as the corpus contract has always had it: scoring joins
            // bench output (keyed by file stem) to measurements (keyed by `name`), and a file
            // named anything else joins to nothing without an error.
            const auto name = graph.name + ".fb";
            std::ofstream problem(root / "graphs" / name, std::ios::binary);
            problem.write(reinterpret_cast<const char*>(graph.bytes.data()),
                          static_cast<std::streamsize>(graph.bytes.size()));
            problem.close();

            hipdnn_corpus_gen::ManifestEntry row;
            row.entry      = entry;
            row.benchmark  = graph.id;
            row.name       = graph.name;
            row.file       = "graphs/" + name;
            row.operation  = result.operation;
            row.regimeAxes = metadata.regimeLabel;
            // The tensor footprint the byte budget admitted it on, not the file size: the
            // latter is a few kilobytes for every graph and says nothing about the problem.
            row.bytes = hipdnn_corpus_gen::graphBytes(graph.bytes);
            manifestRows.push_back(std::move(row));

            if(!wroteHeader)
            {
                index << "problem," << asQueryColumns(entry.point, true) << "\n";
                wroteHeader = true;
            }
            index << graph.name << "," << asQueryColumns(entry.point, false)
                  << "\n";

            commands << options.benchPath;
            for(const auto& dir : options.pluginDirs)
            {
                commands << " --plugin-dir " << dir;
            }
            commands << " --graph " << (root / "graphs" / name).string();
            if(!options.engineName.empty())
            {
                commands << " --engine-name " << options.engineName;
            }
            commands << " --sweep --no-header"
                     << " --problem-id " << graph.name
                     << " --query " << asQueryArgument(entry.point) << "\n";
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
        // Written even when empty, because the absence of a manifest and a manifest recording
        // nothing are different findings and only the second one says which inputs were read.
        hipdnn_corpus_gen::ManifestContext manifest;
        manifest.seed = options.exploration.seed;
        // What was asked for, per operation and summed. Reported, never filled: a corpus of
        // 664 problems from an engine that serves 664 is complete, and the gap between this
        // and the row count is the only place that shows.
        manifest.requested        = requested;
        manifest.allocation       = allocationTotals;
        manifest.duplicatesDropped = droppedTotals;
        for(const auto& entry : selected.operations)
        {
            manifest.operations.push_back(entry.second.operation);
            manifest.inputs.push_back(entry.first);
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
        manifest.reports["excluded"] = excludedRows;
        if(options.count > 0 && total < requested && !shortfall.empty())
        {
            manifest.reports["shortfall"] = shortfall;
        }

        // Stamped so the corpus carries its own provenance. A reader months later cannot tell
        // an engine-verified corpus from a declaration-wide one by looking at the rows -- both
        // are just problems -- and the difference decides whether the survivors of a
        // benchmarking run are a sample or a selection.
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
    if(coverageIsPack)
    {
        if(options.count > 0 && total < requested)
        {
            std::cerr << "Corpus is the engine's whole coverage: " << total << " of " << requested
                      << " requested. " << resolvedEngine
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
