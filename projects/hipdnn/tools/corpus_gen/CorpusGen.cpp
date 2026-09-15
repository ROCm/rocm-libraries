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

#include <hipdnn_corpus_gen/CorpusOutput.hpp>
#include <hipdnn_corpus_gen/MetadataCorpus.hpp>

#include <hipdnn_frontend.hpp>
#include <sstream>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using hipdnn_corpus_gen::ExplorationRequest;
using hipdnn_corpus_gen::ProblemPoint;
using hipdnn_corpus_gen::asQueryArgument;
using hipdnn_corpus_gen::asQueryColumns;

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
    ExplorationRequest exploration;

    /// Benchmarking ceiling in bytes across a problem's tensors. 256 MiB by default: large
    /// enough for real layers, small enough that no single problem dominates a corpus run.
    int64_t maxBytes = 256LL * 1024 * 1024;
};

void printHelp(const char* program)
{
    std::cout << "Usage: " << program
              << " --operations <dir> --engine-name <name> --output <dir> [options]\n\n"
              << "  --operations <dir>     Directory of *.opmeta.json declarations\n"
              << "  --engine-name <name>   Engine to generate for, e.g. hipkernel:ConvFwd\n"
              << "  --engine-id <id>       Same, by id; decimal or 0x-prefixed hex\n"
              << "  --plugin-dir <dir>     Engine plugin directory (repeatable)\n"
              << "  --output <dir>         Where problems/ and commands.txt are written\n"
              << "  --bench-path <path>    hipdnn_bench to name in commands.txt\n"
              << "  --operation <name>     Restrict to one declared operation\n"
              << "  --count <n>            Problems per categorical combination (default 50)\n"
              << "  --budget <n>           Oracle calls per combination (default 20000)\n"
              << "  --ceiling <n>          Largest extent to propose (default 4096)\n"
              << "  --probe <k=v,...>      Report what happens to one point, and stop\n"
              << "  --max-bytes <n>        Benchmarking ceiling across a problem's tensors\n"
              << "  --max-skeleton <n>     Declared regime combinations to try (default 512)\n"
              << "  --seed <n>             Reproducibility seed\n";
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
            options.exploration.pointsPerCombination
                = std::strtoll(next().c_str(), nullptr, 10);
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
    if(options.operationsDir.empty() || !options.haveEngineId)
    {
        std::cerr << "--operations and --engine-name (or --engine-id) are required\n";
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

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create a hipDNN handle\n";
        return 1;
    }

    if(!options.probe.empty())
    {
        // One point, every stage named. The generator reports aggregates, and an aggregate
        // cannot say why a particular problem was refused -- which is the question that
        // actually arises when a corpus comes back empty.
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
        hipdnnDestroy(handle);
        return 0;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto corpora = hipdnn_corpus_gen::generateCorpus(
        handle, options.engineId, selected, options.exploration, options.maxBytes);
    const auto elapsed
        = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    int64_t total = 0;
    std::ofstream commands;
    std::filesystem::path root;
    if(!options.outputDir.empty())
    {
        root = options.outputDir;
        std::filesystem::create_directories(root / "problems");
        commands.open(root / "commands.txt");
        if(!commands)
        {
            std::cerr << "Cannot write " << (root / "commands.txt") << "\n";
            hipdnnDestroy(handle);
            return 1;
        }
        commands << "# hipdnn_bench invocations, one per problem, for "
                 << options.engineName << "\n"
                 << "# Generated from declarations in " << options.operationsDir << "\n"
                 << "# Train on the times, not the rank column: configurations are often\n"
                 << "# separated by less than run-to-run variation.\n";
    }

    for(const auto& result : corpora)
    {
        const auto problems = result.corpus.problems();
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
        std::cerr << "\n";

        if(root.empty())
        {
            for(const auto& point : problems)
            {
                std::cout << asQueryColumns(point, false) << "\n";
            }
            total += static_cast<int64_t>(problems.size());
            continue;
        }

        // An index beside the graphs, so a row's q.* values can be recovered from its problem
        // id without re-running the generator.
        std::ofstream index(root / (result.operation + ".problems.csv"));
        bool wroteHeader = false;

        for(size_t i = 0; i < problems.size(); ++i)
        {
            const auto built = hipdnn_corpus_gen::buildGraphFor(
                selected.operations.front().second, problems[i]);
            const auto declaration = std::find_if(
                selected.operations.begin(),
                selected.operations.end(),
                [&](const auto& entry) { return entry.second.operation == result.operation; });
            const auto graph = declaration == selected.operations.end()
                                   ? built
                                   : hipdnn_corpus_gen::buildGraphFor(declaration->second,
                                                                      problems[i]);
            if(!graph.ok())
            {
                continue;
            }

            const auto name = result.operation + "_" + std::to_string(i) + ".fb";
            std::ofstream problem(root / "problems" / name, std::ios::binary);
            problem.write(reinterpret_cast<const char*>(graph.bytes.data()),
                          static_cast<std::streamsize>(graph.bytes.size()));
            problem.close();

            if(!wroteHeader)
            {
                index << "problem," << asQueryColumns(problems[i], true) << "\n";
                wroteHeader = true;
            }
            index << result.operation << "_" << i << "," << asQueryColumns(problems[i], false)
                  << "\n";

            commands << options.benchPath;
            for(const auto& dir : options.pluginDirs)
            {
                commands << " --plugin-dir " << dir;
            }
            commands << " --graph " << (root / "problems" / name).string()
                     << " --engine-name " << options.engineName << " --sweep --no-header"
                     << " --problem-id " << result.operation << "_" << i
                     << " --query " << asQueryArgument(problems[i]) << "\n";
            ++total;
        }
    }

    std::cerr << "Generated " << total << " problems in " << elapsed << " s\n";
    hipdnnDestroy(handle);
    return total == 0 ? 2 : 0;
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
