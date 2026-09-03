// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
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
 * @brief Standalone validator for generic-kernel-ingestor descriptor bundles.
 *
 * Wraps `loadValidatedDescriptorSets`, the loader's own provider-facing entry point and
 * "the only place validation happens" (`DescriptorLoader.hpp`). This tool exists because
 * that entry point requires two things a standalone binary does not have for free: a
 * registered log sink (the loader never throws -- every rejection is
 * `HIPDNN_PLUGIN_LOG_ERROR(...); continue`, and the default log level is off), and
 * real native symbols registered by a linked provider. Neither gap can be closed by
 * calling the loader differently; both are worked around below.
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

    /// The union across every registry kind -- what `--native-source` diffs against.
    std::set<std::string> all() const
    {
        std::set<std::string> combined;
        combined.insert(graphMatch.begin(), graphMatch.end());
        combined.insert(graphCriterion.begin(), graphCriterion.end());
        combined.insert(kernelMatcher.begin(), kernelMatcher.end());
        combined.insert(dispatch.begin(), dispatch.end());
        combined.insert(score.begin(), score.end());
        return combined;
    }
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
        if(set.heuristic.has_value() && set.heuristic->adapter == UhdAdapter::NATIVE)
        {
            harvested.score.insert(set.heuristic->nativeSymbol);
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

/// One `--native-source` cross-check result.
///
/// The two diff directions are deliberately asymmetric, because one native `.cpp`
/// declares one engine's symbols while the descriptor roots hold every engine's:
/// - `inSourceNotInDescriptors` is **per file**: a symbol this source registers that no
///   descriptor names is a defect in this source no matter what else was passed.
/// - the reverse direction is **aggregated across every `--native-source`** and lives on
///   the run, not here. Diffing one file against the union of all engines' symbols would
///   report every *other* engine's symbols as missing -- pointing `--native-source` at
///   `ConvNative.cpp` over the shipped tree would flag all seven pointwise symbols and
///   exit non-zero on a healthy tree.
struct NativeSourceCheck
{
    std::string sourceFile;
    std::set<std::string> resolvedSymbols;
    std::set<std::string> inSourceNotInDescriptors;
    bool parseError = false;
    std::string parseErrorMessage;

    bool clean() const
    {
        return !parseError && inSourceNotInDescriptors.empty();
    }
};

/// Extracts every `constexpr std::string_view NAME = "value";` declaration in @p text,
/// mapping declared name to its literal value. Text-based, not a clang-tooling parse:
/// the whole file's constants are collected once, then only the ones actually
/// referenced from `register<Name>Symbols` are kept.
std::map<std::string, std::string> extractStringViewConstants(const std::string& text)
{
    static const std::regex s_constantPattern(
        R"RE(constexpr\s+std::string_view\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*;)RE");
    std::map<std::string, std::string> constants;
    for(auto it = std::sregex_iterator(text.begin(), text.end(), s_constantPattern);
        it != std::sregex_iterator();
        ++it)
    {
        constants.emplace((*it)[1].str(), (*it)[2].str());
    }
    return constants;
}

/// Extracts the body of `register<Name>Symbols(...)` -- the single function every
/// pack's native `.cpp` defines to bind its symbols into a `SymbolScope`. Text-based:
/// finds the matching closing brace by depth-counting from the opening one, so a
/// nested block inside the function does not truncate the match.
std::optional<std::string> extractRegisterSymbolsBody(const std::string& text)
{
    static const std::regex s_signaturePattern(R"(void\s+register\w*Symbols\s*\([^)]*\)\s*\{)");
    std::smatch match;
    if(!std::regex_search(text, match, s_signaturePattern))
    {
        return std::nullopt;
    }
    const size_t bodyStart = static_cast<size_t>(match.position(0)) + match.length(0);
    int depth = 1;
    size_t index = bodyStart;
    for(; index < text.size() && depth > 0; ++index)
    {
        if(text[index] == '{')
        {
            ++depth;
        }
        else if(text[index] == '}')
        {
            --depth;
        }
    }
    if(depth != 0)
    {
        return std::nullopt;
    }
    return text.substr(bodyStart, index - 1 - bodyStart);
}

/// Every identifier passed as `scope.add(...)`'s first argument within
/// `registerBody`, resolving both `scope.add(std::string(NAME), ...)` and the bare
/// `scope.add(NAME, ...)` spelling. There are zero inline `scope.add("literal", ...)`
/// calls anywhere in the tree -- all 11 real registrations pass a named constant -- so
/// only the identifier forms are matched; a literal-string scan would find nothing and
/// silently diff empty-vs-empty.
std::vector<std::string> extractScopeAddArgumentNames(const std::string& registerBody)
{
    static const std::regex s_scopeAddPattern(
        R"(scope\s*\.\s*add\s*\(\s*(?:std::string\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|([A-Za-z_][A-Za-z0-9_]*))\s*,)");
    std::vector<std::string> names;
    for(auto it = std::sregex_iterator(registerBody.begin(), registerBody.end(), s_scopeAddPattern);
        it != std::sregex_iterator();
        ++it)
    {
        const std::string viaStdString = (*it)[1].str();
        names.push_back(viaStdString.empty() ? (*it)[2].str() : viaStdString);
    }
    return names;
}

/// Resolves `--native-source <file.cpp>` against the harvested descriptor symbol set.
/// Finds `register<Name>Symbols`, collects the identifiers passed to `scope.add(...)`,
/// resolves each back to its `constexpr std::string_view NAME = "value";` declaration
/// in the same file, and diffs the resolved values against @p descriptorSymbols.
///
/// A file that yields zero resolved symbols is reported as a parse error, not a clean
/// pass: an empty-vs-empty diff is exactly the false-green this check exists to catch
/// (a regex that finds nothing looks identical to a file that legitimately declares
/// nothing).
NativeSourceCheck checkNativeSource(const std::string& path,
                                    const std::set<std::string>& descriptorSymbols)
{
    NativeSourceCheck check;
    check.sourceFile = path;

    std::ifstream file(path, std::ios::binary);
    if(!file.is_open())
    {
        check.parseError = true;
        check.parseErrorMessage = "failed to open '" + path + "'";
        return check;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    const auto registerBody = extractRegisterSymbolsBody(text);
    if(!registerBody.has_value())
    {
        check.parseError = true;
        check.parseErrorMessage
            = "no 'register<Name>Symbols(...)' function found in '" + path + "'";
        return check;
    }

    const auto constants = extractStringViewConstants(text);
    const auto argumentNames = extractScopeAddArgumentNames(*registerBody);

    for(const auto& name : argumentNames)
    {
        const auto it = constants.find(name);
        if(it == constants.end())
        {
            check.parseError = true;
            if(!check.parseErrorMessage.empty())
            {
                check.parseErrorMessage.append("; ");
            }
            check.parseErrorMessage.append("'scope.add' in ")
                .append(path)
                .append(" references '")
                .append(name)
                .append("', which has no 'constexpr std::string_view ")
                .append(name)
                .append(" = \"...\";' declaration in the same file");
            continue;
        }
        check.resolvedSymbols.insert(it->second);
    }

    if(check.resolvedSymbols.empty())
    {
        check.parseError = true;
        if(check.parseErrorMessage.empty())
        {
            check.parseErrorMessage
                = "'" + path
                  + "' resolved zero native symbols from register<Name>Symbols -- treating "
                    "this as an error rather than an empty-vs-empty pass";
        }
        return check;
    }

    for(const auto& symbol : check.resolvedSymbols)
    {
        if(descriptorSymbols.count(symbol) == 0)
        {
            check.inSourceNotInDescriptors.insert(symbol);
        }
    }
    return check;
}

/// The descriptor-named symbols no supplied `--native-source` file declares, across all
/// of them. Only meaningful once every native source backing the descriptor roots has
/// been passed, so it is reported as a run-level violation rather than pinned on any one
/// file. With no `--native-source` at all this is not computed: absence of the flag means
/// the cross-check was not requested, not that every symbol is unaccounted for.
std::set<std::string>
    descriptorSymbolsNoSourceDeclares(const std::vector<NativeSourceCheck>& checks,
                                      const std::set<std::string>& descriptorSymbols)
{
    std::set<std::string> declared;
    for(const auto& check : checks)
    {
        declared.insert(check.resolvedSymbols.begin(), check.resolvedSymbols.end());
    }

    std::set<std::string> undeclared;
    for(const auto& symbol : descriptorSymbols)
    {
        if(declared.count(symbol) == 0)
        {
            undeclared.insert(symbol);
        }
    }
    return undeclared;
}

struct Options
{
    std::vector<std::string> roots;
    std::vector<std::string> nativeSources;
    std::vector<std::string> expectEngines;
    bool json = false;
    bool showHelp = false;
};

void printHelp(const char* programName)
{
    std::cout << "Usage: " << programName
              << " <root>... [--native-source <cpp>]... [--expect-engine <name>]... [--json]\n"
              << "Loads and validates generic-kernel-ingestor descriptor bundles under one or\n"
              << "more root directories, the same way a real provider would at plugin load\n"
              << "time -- without a GPU and without linking a real provider.\n"
              << "Options:\n"
              << "  <root>                    Descriptor root directory (repeatable)\n"
              << "  --native-source <cpp>     Cross-check a pack's register<Name>Symbols\n"
              << "                            against the descriptors' named symbols "
                 "(repeatable)\n"
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
        if(arg == "--native-source")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: --native-source requires a file argument\n";
                return std::nullopt;
            }
            options.nativeSources.emplace_back(argv[++i]);
        }
        else if(arg == "--expect-engine")
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

    const auto descriptorSymbols = harvested.all();
    std::vector<NativeSourceCheck> nativeSourceChecks;
    nativeSourceChecks.reserve(options->nativeSources.size());
    for(const auto& sourcePath : options->nativeSources)
    {
        nativeSourceChecks.push_back(checkNativeSource(sourcePath, descriptorSymbols));
    }

    // Aggregated across every supplied source, not per file: see NativeSourceCheck.
    const auto undeclaredSymbols
        = nativeSourceChecks.empty()
              ? std::set<std::string>{}
              : descriptorSymbolsNoSourceDeclares(nativeSourceChecks, descriptorSymbols);

    const bool nativeSourceClean
        = std::all_of(nativeSourceChecks.begin(),
                      nativeSourceChecks.end(),
                      [](const NativeSourceCheck& check) { return check.clean(); })
          && undeclaredSymbols.empty();

    const bool success = errorMessages.empty() && missingEngines.empty() && nativeSourceClean;

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

        auto& checksJson = report["native_source_checks"];
        checksJson = nlohmann::json::array();
        for(const auto& check : nativeSourceChecks)
        {
            checksJson.push_back({
                {"source_file", check.sourceFile},
                {"clean", check.clean()},
                {"parse_error", check.parseError},
                {"parse_error_message", check.parseErrorMessage},
                {"resolved_symbols", check.resolvedSymbols},
                {"in_source_not_in_descriptors", check.inSourceNotInDescriptors},
            });
        }

        // Run-level, not per file: the descriptor-named symbols that no supplied
        // --native-source declares. Empty (and meaningless) when the flag was not used.
        report["descriptor_symbols_no_source_declares"] = undeclaredSymbols;

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

        for(const auto& check : nativeSourceChecks)
        {
            if(check.parseError)
            {
                std::cerr << "VIOLATION: native-source parse error in '" << check.sourceFile
                          << "': " << check.parseErrorMessage << "\n";
                continue;
            }
            for(const auto& symbol : check.inSourceNotInDescriptors)
            {
                std::cerr << "VIOLATION: native-source '" << check.sourceFile
                          << "' declares symbol '" << symbol << "' that no descriptor names\n";
            }
        }

        for(const auto& symbol : undeclaredSymbols)
        {
            std::cerr << "VIOLATION: descriptor names symbol '" << symbol
                      << "', which none of the supplied --native-source files declares\n";
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
    // The tool walks the filesystem, runs regexes and parses JSON, all of which throw.
    // Letting one escape `main` gives the caller a terminate() and no diagnostic, which
    // in a validator is indistinguishable from a crash in the thing being validated.
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
