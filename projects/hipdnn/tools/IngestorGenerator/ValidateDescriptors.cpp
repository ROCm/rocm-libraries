// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

/**
 * @file ValidateDescriptors.cpp
 * @brief Standalone STRUCTURAL validator for generic-kernel-ingestor descriptor bundles.
 *
 * Wraps `loadValidatedDescriptorSets`, the loader's own provider-facing entry point and
 * "the only place validation happens" (`DescriptorLoader.hpp`). This tool exists because
 * that entry point requires two things a standalone binary does not have for free: a
 * registered log sink (the loader never throws -- every rejection is
 * `HIPDNN_PLUGIN_LOG_ERROR(...); continue`, and the default log level is off), and
 * native symbols registered by a linked provider. Neither gap can be closed by calling
 * the loader differently; both are worked around below.
 *
 * The symbols registered here are STUBS, harvested from the descriptors themselves. That
 * makes this a check of descriptor structure, cross-references and completeness. It says
 * nothing about whether any provider implements those symbols: that question is answered
 * only by the provider host checks, which run the real typed registration
 * (`discoverDescriptorSets()` -> `registerNativeIngestorSymbols()` ->
 * `loadValidatedDescriptorSets<Handle>()`) and census the loaded bundle. A clean run here
 * is not native-implementation proof.
 */

namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

/// The validator's own THandle. `NativeRegistry<T>` is one instance per `T` per image,
/// so this cannot collide with any provider's registrations. `getStream()` is not
/// required by anything `makeStateManager` instantiates (that static_assert lives in
/// GenericPlanBuilder/BenchmarkPlan, neither reached here), but is provided anyway so
/// the handle stays usable if the loader ever needs more of it.
struct ValidatorHandle
{
    static hipStream_t getStream()
    {
        return nullptr;
    }
};

/// A stub dispatch handler. Its methods are never invoked -- DispatchRegistry only ever
/// stores a pointer to it, resolved during the native-symbol pre-flight -- so the bodies
/// are trivial. Static storage duration: the registry holds a non-owning pointer.
class StubDispatchHandler : public IKernelDispatchHandler<ValidatorHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*tokens*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*tokens*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return nullptr;
    }

    void launch(const ValidatorHandle& /*handle*/,
                const PreparedDispatch& /*dispatch*/,
                const hipdnnPluginDeviceBuffer_t* /*buffers*/,
                uint32_t /*bufferCount*/,
                void* /*workspace*/) const override
    {
        // Never called: the validator never builds a real plan.
    }
};

/// The stub `GraphMatchFn`. Must return an *engaged* optional -- `nullopt` is the
/// engine-level verdict that empties the whole catalog and skips every remaining pack
/// of that engine (`KernelIngestorStateManager.hpp`), which would make every engine
/// declaring a graph_match symbol validate as empty rather than as its real shape.
std::optional<BoundTokens> stubGraphMatch(const MatchContext& /*context*/)
{
    return BoundTokens{};
}

/// The stub `GraphCriterionFn`/`KernelMatcherFn`/`ScoreFn`. Never invoked by
/// `makeStateManager`'s construction-only probe; only their registration is checked.
bool stubGraphCriterion(const MatchContext& /*context*/, const BoundTokens& /*tokens*/)
{
    return true;
}

bool stubKernelMatcher(const MatchContext& /*context*/,
                       const BoundTokens& /*tokens*/,
                       const KernelDefinition& /*kernel*/)
{
    return true;
}

double stubScore(const MatchContext& /*context*/,
                 const BoundTokens& /*tokens*/,
                 const KernelDefinition& /*kernel*/)
{
    return 0.0;
}

/// One captured diagnostic from the loader's log sink.
struct Diagnostic
{
    hipdnnSeverity_t severity;
    std::string message;
};

/// Accumulates every message the loader logs. The callback registered with
/// `registerLoggingCallback` is a bare function pointer with no user-data slot, so the
/// sink must be a namespace-scope (file-static) collection rather than a captured
/// lambda.
class DiagnosticSink
{
public:
    static DiagnosticSink& instance()
    {
        static DiagnosticSink s_instance;
        return s_instance;
    }

    void record(hipdnnSeverity_t severity, const char* message)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _diagnostics.push_back(Diagnostic{severity, message == nullptr ? std::string() : message});
    }

    std::vector<Diagnostic> take() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _diagnostics;
    }

private:
    mutable std::mutex _mutex;
    std::vector<Diagnostic> _diagnostics;
};

void diagnosticCallback(hipdnnSeverity_t severity, const char* message)
{
    DiagnosticSink::instance().record(severity, message);
}

/// RAII guard around the log sink: installs the callback and level on construction,
/// unregisters on every exit path (including an exception) on destruction.
class LogSinkGuard
{
public:
    LogSinkGuard()
    {
        hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
        hipdnn_data_sdk::logging::registerLoggingCallback(&diagnosticCallback);
    }

    LogSinkGuard(const LogSinkGuard&) = delete;
    LogSinkGuard& operator=(const LogSinkGuard&) = delete;

    ~LogSinkGuard()
    {
        hipdnn_data_sdk::logging::unregisterLoggingCallback();
    }
};

const char* severityName(hipdnnSeverity_t severity)
{
    switch(severity)
    {
    case HIPDNN_SEV_INFO:
        return "INFO";
    case HIPDNN_SEV_WARN:
        return "WARN";
    case HIPDNN_SEV_ERROR:
        return "ERROR";
    case HIPDNN_SEV_FATAL:
        return "FATAL";
    case HIPDNN_SEV_OFF:
        return "OFF";
    }
    return "UNKNOWN";
}

/// Every native symbol name one DescriptorSet references, across all five hook kinds:
/// `engine.graphMatchNativeSymbol`, every `matchers[].matchSymbol` (dispatched by
/// `matcher.scope` onto the graph- or kernel-scoped registry), every
/// `dispatches[].dispatchSymbol`, and `heuristic->payload` when the heuristic is
/// native. Harvested from pass 1's (unresolved-symbol) sets, before any stub is
/// registered.
struct HarvestedSymbols
{
    std::set<std::string> graphMatch;
    std::set<std::string> graphCriterion;
    std::set<std::string> kernelMatcher;
    std::set<std::string> dispatch;
    std::set<std::string> score;
};

HarvestedSymbols harvestSymbols(const std::vector<DescriptorSet>& sets)
{
    HarvestedSymbols harvested;
    for(const auto& set : sets)
    {
        if(!set.engine.graphMatchNativeSymbol.empty())
        {
            harvested.graphMatch.insert(set.engine.graphMatchNativeSymbol);
        }
        for(const auto& matcher : set.matchers)
        {
            if(matcher.scope == MatchScope::GRAPH)
            {
                harvested.graphCriterion.insert(matcher.matchSymbol);
            }
            else
            {
                harvested.kernelMatcher.insert(matcher.matchSymbol);
            }
        }
        for(const auto& dispatch : set.dispatches)
        {
            harvested.dispatch.insert(dispatch.dispatchSymbol);
        }
        if(set.heuristic.has_value() && set.heuristic->kind == HeuristicKind::NATIVE)
        {
            harvested.score.insert(set.heuristic->payload);
        }
    }
    return harvested;
}

/// Registers a no-op stub per unique harvested name into each registry. Names are
/// pre-deduped into `std::set`s by `harvestSymbols`, which is required: two descriptor
/// sets may legally share a symbol name (e.g. two engines' matchers), and
/// `NativeRegistry::registerSymbol` throws `std::runtime_error` on a duplicate. That
/// throw must never reach here -- it is Phase 1's halt condition, not a validator
/// failure mode -- so registering from a `std::set` rather than a raw harvested list
/// keeps the registration itself well-formed regardless of what the descriptors name.
StubDispatchHandler stubDispatchHandler;

void registerStubs(const HarvestedSymbols& harvested)
{
    for(const auto& symbol : harvested.graphMatch)
    {
        GraphMatchRegistry::registerSymbol(symbol, &stubGraphMatch);
    }
    for(const auto& symbol : harvested.graphCriterion)
    {
        GraphCriterionRegistry::registerSymbol(symbol, &stubGraphCriterion);
    }
    for(const auto& symbol : harvested.kernelMatcher)
    {
        KernelMatcherRegistry::registerSymbol(symbol, &stubKernelMatcher);
    }
    for(const auto& symbol : harvested.score)
    {
        ScoreRegistry::registerSymbol(symbol, &stubScore);
    }
    for(const auto& symbol : harvested.dispatch)
    {
        DispatchRegistry<ValidatorHandle>::registerSymbol(symbol, &stubDispatchHandler);
    }
}

struct Options
{
    std::vector<std::string> roots;
    std::vector<std::string> expectEngines;
    bool json = false;
    bool showHelp = false;
};

void printHelp(const char* programName)
{
    std::cout << "Usage: " << programName << " <root>... [--expect-engine <name>]... [--json]\n"
              << "Loads and structurally validates generic-kernel-ingestor descriptor\n"
              << "bundles under one or more root directories: cross-references, metadata\n"
              << "completion and catalog identity, with a no-op stub standing in for every\n"
              << "native symbol the descriptors name. It does NOT check that a provider\n"
              << "implements those symbols -- the provider host checks do that, by running\n"
              << "the real typed registration and censusing what loads.\n"
              << "Options:\n"
              << "  <root>                    Descriptor root directory (repeatable)\n"
              << "  --expect-engine <name>    Require this engine name in the validated set "
                 "(repeatable)\n"
              << "  --json                    Emit machine-readable JSON instead of text\n"
              << "  --help, -h                Show this help message\n";
}

std::optional<Options> parseArgs(int argc, const char* const* argv)
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--help" || arg == "-h")
        {
            options.showHelp = true;
            return options;
        }
        if(arg == "--expect-engine")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: --expect-engine requires a name argument\n";
                return std::nullopt;
            }
            options.expectEngines.emplace_back(argv[++i]);
        }
        else if(arg == "--json")
        {
            options.json = true;
        }
        else if(!arg.empty() && arg[0] == '-')
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            return std::nullopt;
        }
        else
        {
            options.roots.emplace_back(arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char* argv[])
try
{
    const auto options = parseArgs(argc, argv);
    if(!options.has_value())
    {
        printHelp(argv[0]);
        return 1;
    }
    if(options->showHelp)
    {
        printHelp(argv[0]);
        return 0;
    }
    if(options->roots.empty())
    {
        std::cerr << "Error: at least one <root> directory is required\n";
        printHelp(argv[0]);
        return 1;
    }

    const std::vector<std::filesystem::path> roots(options->roots.begin(), options->roots.end());

    // Installed for the whole run, before the first load: the loader never throws, so
    // without this sink every rejection is invisible and the tool would report nothing
    // more useful than a bare engine count.
    const LogSinkGuard logSinkGuard;

    // Pass 1: harvest every symbol name the descriptors reference. Neither
    // loadDescriptorCatalog nor resolveDescriptorSets checks symbol registration, so
    // this pass runs before anything is registered.
    const auto unresolvedSets = resolveDescriptorSets(loadDescriptorCatalog(roots));
    const auto harvested = harvestSymbols(unresolvedSets);

    // Register a no-op stub per unique name. Duplicate names across sets are legal and
    // already deduped by harvestSymbols' std::set members; registerStubs must never
    // observe NativeRegistry::registerSymbol's duplicate-throw on well-formed input.
    registerStubs(harvested);

    // Pass 2: the real verdict. Every rejection this call makes reaches DiagnosticSink
    // as an ERROR, which is what actually drives this tool's exit code.
    const auto validatedSets = loadValidatedDescriptorSets<ValidatorHandle>(roots);

    std::vector<std::string> engineNames;
    engineNames.reserve(validatedSets.size());
    for(const auto& set : validatedSets)
    {
        engineNames.push_back(set.engine.name);
    }

    const auto diagnostics = DiagnosticSink::instance().take();
    std::vector<std::string> errorMessages;
    for(const auto& diagnostic : diagnostics)
    {
        if(diagnostic.severity == HIPDNN_SEV_ERROR || diagnostic.severity == HIPDNN_SEV_FATAL)
        {
            errorMessages.push_back(diagnostic.message);
        }
    }

    std::vector<std::string> missingEngines;
    for(const auto& expected : options->expectEngines)
    {
        if(std::find(engineNames.begin(), engineNames.end(), expected) == engineNames.end())
        {
            missingEngines.push_back(expected);
        }
    }

    const bool success = errorMessages.empty() && missingEngines.empty();

    if(options->json)
    {
        nlohmann::json report;
        report["success"] = success;
        report["roots"] = options->roots;
        report["engines"] = engineNames;
        report["expected_engines_missing"] = missingEngines;

        auto& diagnosticsJson = report["diagnostics"];
        diagnosticsJson = nlohmann::json::array();
        for(const auto& diagnostic : diagnostics)
        {
            diagnosticsJson.push_back(
                {{"severity", severityName(diagnostic.severity)}, {"message", diagnostic.message}});
        }

        std::cout << report.dump(2) << "\n";
    }
    else
    {
        std::cout << "Loaded engines:\n";
        if(engineNames.empty())
        {
            std::cout << "  (none)\n";
        }
        for(const auto& set : validatedSets)
        {
            std::cout << "  " << set.engine.name << "\n";
        }

        std::cout << "Diagnostics:\n";
        if(diagnostics.empty())
        {
            std::cout << "  (none)\n";
        }
        for(const auto& diagnostic : diagnostics)
        {
            std::cout << "  [" << severityName(diagnostic.severity) << "] " << diagnostic.message
                      << "\n";
        }

        for(const auto& missing : missingEngines)
        {
            std::cerr << "VIOLATION: expected engine not found: '" << missing << "'\n";
        }

        for(const auto& message : errorMessages)
        {
            std::cerr << "VIOLATION: " << message << "\n";
        }
    }

    return success ? 0 : 1;
}
catch(const std::exception& error)
{
    // The tool walks the filesystem and parses JSON, both of which throw. Letting one
    // escape `main` gives the caller a terminate() and no diagnostic, which in a
    // validator is indistinguishable from a crash in the thing being validated.
    std::cerr << "FATAL: " << error.what() << "\n";
    return 2;
}

#else // HIPDNN_ENABLE_KERNEL_INGESTOR

#include <iostream>

int main()
{
    std::cerr << "hipdnn_validate_descriptors was built without "
                 "HIPDNN_ENABLE_KERNEL_INGESTOR; the generic kernel ingestor is not "
                 "compiled into this build, so there is nothing to validate. Rebuild "
                 "with -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON.\n";
    return 1;
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
