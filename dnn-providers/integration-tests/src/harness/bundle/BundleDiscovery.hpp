// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_integration_tests::bundle
{

// Naming types, kept together. DerivedTestName is the output of deriveTestName()
// for direct graph bundles; DiscoveredBundle is what discoverBundles() returns
// for registration. GTest joins suiteName and testName with '.' to form the full
// registered name.
//
// Template sweeps use the sweep directory as the suite and each cases[].id as a
// logical test. Direct bundles use the relative folder path as the suite and the
// graph .json stem as the test.
struct DerivedTestName
{
    std::string suiteName;
    std::string testName;
};

// A template-sweep case selected from a sweep root: the shared
// graph.template.json and the cases[].id it expands. Present on a
// DiscoveredBundle exactly when that bundle is a template-sweep case.
struct SweepCase
{
    std::filesystem::path templatePath; // graph.template.json shared by the sweep
    std::string caseId; // logical case selected from cases[]
};

// One registerable bundle test. Direct bundles load jsonPath as a graph .json.
// Template-sweep cases load jsonPath as sweep.json and carry a SweepCase naming
// the graph.template.json and the cases[] entry to expand. diagnosticPath() is
// only for logs/errors and names the logical sweep case.
struct DiscoveredBundle
{
    std::filesystem::path jsonPath; // graph .json for single bundles, sweep.json for sweep cases
    std::string suiteName;
    std::string testName;
    std::optional<SweepCase> sweep; // set iff this is a template-sweep case

    bool isTemplateSweepCase() const
    {
        return sweep.has_value();
    }

    /// "Suite.Test": the name GTest filters on and the registry de-duplicates by.
    std::string fullName() const
    {
        return suiteName + "." + testName;
    }

    std::filesystem::path diagnosticPath() const
    {
        if(!isTemplateSweepCase())
        {
            return jsonPath;
        }

        return {jsonPath.string() + "#" + sweep->caseId};
    }
};

inline bool isDescendantOf(const std::filesystem::path& path, const std::filesystem::path& ancestor)
{
    const auto normalizedPath = path.lexically_normal();
    const auto normalizedAncestor = ancestor.lexically_normal();

    auto pathIt = normalizedPath.begin();
    auto ancestorIt = normalizedAncestor.begin();
    for(; pathIt != normalizedPath.end() && ancestorIt != normalizedAncestor.end();
        ++pathIt, ++ancestorIt)
    {
        if(*pathIt != *ancestorIt)
        {
            return false;
        }
    }

    return ancestorIt == normalizedAncestor.end();
}

namespace detail
{

// True when isDescendantOf(path, a) holds for some `a` whose lexically_normal() form
// is in `normalizedAncestors`. The cost is one set lookup per level of `path`.
inline bool hasAncestorIn(const std::filesystem::path& path,
                          const std::set<std::filesystem::path>& normalizedAncestors)
{
    auto current = path.lexically_normal();
    while(true)
    {
        if(normalizedAncestors.count(current) != 0)
        {
            return true;
        }
        auto parent = current.parent_path();
        if(parent == current)
        {
            break;
        }
        current = std::move(parent);
    }
    // The walk stops at the root path, for example "/" or "C:\" on Windows. The element
    // sequence of `path` also starts with its root name ("C:") and with the empty sequence.
    return normalizedAncestors.count(current.root_name()) != 0
           || normalizedAncestors.count(std::filesystem::path{}) != 0;
}

inline std::set<std::filesystem::path>
    lexicallyNormalSet(const std::vector<std::filesystem::path>& paths)
{
    std::set<std::filesystem::path> normalized;
    for(const auto& path : paths)
    {
        normalized.insert(path.lexically_normal());
    }
    return normalized;
}

} // namespace detail

// The directory `path` resolves to, or the path itself when it cannot be resolved,
// for comparing the directories a walk has descended through.
inline std::filesystem::path canonicalDirectory(const std::filesystem::path& path)
{
    std::error_code error;
    auto canonical = std::filesystem::canonical(path, error);
    return error ? path.lexically_normal() : canonical;
}

// False when `directory` cannot be listed for lack of permission. The walk probes
// before descending because the iterator cannot report such a directory and go on:
// without skip_permission_denied the first one throws and ends discovery, and with it
// the directory is dropped without a word.
inline bool canListDirectory(const std::filesystem::path& directory)
{
    std::error_code error;
    const std::filesystem::directory_iterator probe(directory, error);
    return error != std::errc::permission_denied;
}

// Visits every entry at or under `root`, the one walk every discovery scan uses.
// Directory symlinks are followed at any depth, so a bundle root assembled from
// links (for example quick/SdpaFwd linked to an installed tree) is discovered like
// the same tree copied. Entries keep their path through the link, which is what
// test names are derived from.
//
// The walk keeps the canonical directory of every level it is currently inside,
// starting with the root. A directory that resolves to one of them, or to a parent
// of one of them, would lead the walk back to where it already is, so it is a cycle
// and is skipped with a warning rather than descended. That covers a link to its own
// ancestor, a/to_b -> b next to b/to_a -> a, and a link out of the root to one of
// the root's parents (say / or $HOME), which would otherwise walk that whole tree
// before reaching the root again. Two links to the same directory from different
// branches are not a cycle; both are walked, as a copied tree would be.
//
// Only a directory symlink is resolved. A plain subdirectory resolves to the level
// it is listed in plus its own name. No outer level lies at or below that innermost
// level (the root trivially, a link by the check, a plain directory by this same
// argument), so none lies at or below the subdirectory and it cannot lead back. A
// tree with no links therefore costs one canonical() call, for the root.
//
// A directory the walk may not list is skipped with a warning, so one unreadable
// folder under a linked tree does not end discovery.
template <typename Visit>
void forEachBundleTreeEntry(const std::filesystem::path& root, Visit&& visit)
{
    namespace fs = std::filesystem;
    // ancestry[d] is the directory that holds the entries at iterator depth d.
    std::vector<fs::path> ancestry{canonicalDirectory(root)};
    for(auto it
        = fs::recursive_directory_iterator(root, fs::directory_options::follow_directory_symlink);
        it != fs::recursive_directory_iterator();
        ++it)
    {
        ancestry.resize(static_cast<size_t>(it.depth()) + 1);
        if(it->is_directory())
        {
            fs::path directory;
            if(it->is_symlink())
            {
                directory = canonicalDirectory(it->path());
                const bool leadsBack
                    = std::any_of(ancestry.begin(), ancestry.end(), [&](const fs::path& level) {
                          return isDescendantOf(level, directory);
                      });
                if(leadsBack)
                {
                    HIPDNN_PLUGIN_LOG_WARN(
                        "Not following bundle directory that contains one already being walked "
                        "(cycle): "
                        << it->path() << " -> " << directory);
                    it.disable_recursion_pending();
                    continue;
                }
            }
            else
            {
                directory = ancestry.back() / it->path().filename();
            }
            if(!canListDirectory(it->path()))
            {
                HIPDNN_PLUGIN_LOG_WARN(
                    "Skipping bundle directory that cannot be listed (permission denied): "
                    << it->path());
                it.disable_recursion_pending();
                continue;
            }
            ancestry.push_back(std::move(directory));
        }
        visit(*it);
    }
}

// The path of `path` below `root`, derived from the paths alone. Lexical on purpose:
// std::filesystem::relative() resolves symlinks, and a bundle reached through a link
// whose target lies outside the root would get a "../.." name instead of the name of
// the folder it was found in.
inline std::filesystem::path relativeToRoot(const std::filesystem::path& path,
                                            const std::filesystem::path& root)
{
    return path.lexically_normal().lexically_relative(root.lexically_normal());
}

// Generic recursive file scanner. It carries no bundle knowledge; graph vs
// companion vs sweep filtering is layered on top by isGraphFile() and
// discoverBundles().
inline std::vector<std::filesystem::path>
    scanFilesByExtension(const std::filesystem::path& directory, const std::string& extension)
{
    std::vector<std::filesystem::path> paths;
    forEachBundleTreeEntry(directory, [&](const std::filesystem::directory_entry& entry) {
        if(entry.is_regular_file() && entry.path().extension() == extension)
        {
            paths.push_back(entry.path());
        }
    });
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Returns every leaf directory at or under root. Leaf folders with no graph .json
// are warned as likely incomplete bundles, except for recognized sweep roots and
// sweep golden-data leaves.
inline std::vector<std::filesystem::path> findLeafDirectories(const std::filesystem::path& root)
{
    std::set<std::filesystem::path> withSubdir;
    std::set<std::filesystem::path> allDirs;
    allDirs.insert(root);
    forEachBundleTreeEntry(root, [&](const std::filesystem::directory_entry& entry) {
        if(entry.is_directory())
        {
            allDirs.insert(entry.path());
            withSubdir.insert(entry.path().parent_path());
        }
    });

    std::vector<std::filesystem::path> leaves;
    for(const auto& dir : allDirs)
    {
        if(withSubdir.find(dir) == withSubdir.end())
        {
            leaves.push_back(dir);
        }
    }
    return leaves;
}

// Companion file kinds that mark a .json as metadata for a graph instead of a
// graph test. Add future companion suffixes here so discovery has one exclusion
// list.
inline const std::set<std::string>& companionKinds()
{
    static const std::set<std::string> s_kinds = {"meta", "support"};
    return s_kinds;
}

inline bool isSweepTemplateFile(const std::filesystem::path& jsonPath)
{
    return jsonPath.filename() == "graph.template.json";
}

inline bool isSweepManifestFile(const std::filesystem::path& jsonPath)
{
    return jsonPath.filename() == "sweep.json";
}

inline bool isSweepBundleRoot(const std::filesystem::path& directory)
{
    return std::filesystem::exists(directory / "graph.template.json")
           && std::filesystem::exists(directory / "sweep.json");
}

// Discovery predicate: true only for a direct bundle graph .json.
//
// Known companion files are excluded when their whole stem is a companion kind
// ("meta.json") or their final dotted segment is a companion kind
// ("Small.meta.json"). Other dotted names remain valid graphs, e.g.
// "model.fp16.json" or "resnet50.v2.json".
//
// Template-sweep control files ("graph.template.json" and "sweep.json") are not
// direct graph tests; discoverSweepCases() registers logical tests for them.
inline bool isGraphFile(const std::filesystem::path& jsonPath)
{
    if(jsonPath.extension() != ".json")
    {
        return false;
    }
    if(isSweepTemplateFile(jsonPath) || isSweepManifestFile(jsonPath))
    {
        return false;
    }

    const auto stem = jsonPath.stem().string();
    if(companionKinds().count(stem) != 0)
    {
        return false;
    }

    const auto dot = stem.rfind('.');
    return dot == std::string::npos || companionKinds().count(stem.substr(dot + 1)) == 0;
}

// Maps any non-[alnum_] character to '_' so user-provided path segments and case
// ids are legal GTest name components. RegisterTest() does not validate names, so
// this is the safety boundary for ad-hoc customer bundle folders.
inline std::string sanitizeForGtest(const std::string& input)
{
    std::string result;
    result.reserve(input.size());
    for(const char c : input)
    {
        result += (std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_') ? c : '_';
    }
    return result;
}

// Builds the GTest suite from the bundle path relative to the data root by
// sanitizing each segment and joining with '_'. The first segment is the tier
// directory (e.g. "quick", "standard", "full") and passes through as-is so
// bundle suite names match the on-disk folder structure directly.
inline std::string deriveSuiteName(const std::filesystem::path& relativeDir,
                                   const std::filesystem::path& sourcePath)
{
    if(relativeDir.empty())
    {
        throw std::runtime_error(
            "Bundle content must live in a sub-folder of the data root, not at the root itself: "
            + sourcePath.string());
    }

    std::string suite;
    for(const auto& segment : relativeDir)
    {
        if(!suite.empty())
        {
            suite += "_";
        }
        suite += sanitizeForGtest(segment.string());
    }
    return suite;
}

// Derives direct-bundle GTest names from path only:
//   suiteName = relative directory path, sanitized and joined by '_'
//   testName  = graph .json stem, sanitized
//
// This deliberately diverges from RFC 0011's fixed tier/op/layout/dtype folder
// structure. Discovery imposes no semantic folder schema so "drop a folder, it
// runs" works for ad-hoc bundles; structural validation belongs to the bundle
// verifier, not registration.
//
// Op family is the second path segment by convention (tier/OpFamily/…), not
// stored as bundle metadata. Consumers that need it (e.g. the offline support
// matrix renderer) derive it from the canonical directory layout.
inline DerivedTestName deriveTestName(const std::filesystem::path& jsonPath,
                                      const std::filesystem::path& bundleDir)
{
    const auto relative = relativeToRoot(jsonPath, bundleDir);
    const auto relativeDir = relative.parent_path();
    if(relativeDir.empty())
    {
        return {sanitizeForGtest(bundleDir.filename().string()),
                sanitizeForGtest(jsonPath.stem().string())};
    }

    return {deriveSuiteName(relativeDir, jsonPath), sanitizeForGtest(jsonPath.stem().string())};
}

// A sweep root is any directory with both graph.template.json and sweep.json.
// It registers one logical test per cases[].id, and normal graph discovery skips
// every JSON file below it so templates, manifests, and golden metadata do not
// become direct tests.
inline std::vector<std::filesystem::path>
    findSweepDirectories(const std::filesystem::path& bundleDir)
{
    std::set<std::filesystem::path> sweepDirs;
    if(isSweepBundleRoot(bundleDir))
    {
        sweepDirs.insert(bundleDir);
    }

    forEachBundleTreeEntry(bundleDir, [&](const std::filesystem::directory_entry& entry) {
        if(entry.is_directory() && isSweepBundleRoot(entry.path()))
        {
            sweepDirs.insert(entry.path());
        }
    });

    return {sweepDirs.begin(), sweepDirs.end()};
}

namespace detail
{

// Leaf folders at or under `bundleDir` that hold no graph .json. Sweep roots are
// not reported. Golden tensor directories under a sweep are leaves by design; they
// are data, not incomplete bundle folders, so they are not reported either.
inline std::vector<std::filesystem::path>
    findEmptyLeafFolders(const std::filesystem::path& bundleDir,
                         const std::vector<std::filesystem::path>& sweepDirs)
{
    std::vector<std::filesystem::path> goldenDirs;
    goldenDirs.reserve(sweepDirs.size());
    for(const auto& sweepDir : sweepDirs)
    {
        goldenDirs.push_back(sweepDir / "golden");
    }
    const auto normalizedGoldenDirs = lexicallyNormalSet(goldenDirs);

    std::vector<std::filesystem::path> emptyLeaves;
    for(const auto& leaf : findLeafDirectories(bundleDir))
    {
        if(isSweepBundleRoot(leaf) || hasAncestorIn(leaf, normalizedGoldenDirs))
        {
            continue;
        }

        const bool hasGraph
            = std::any_of(std::filesystem::directory_iterator(leaf),
                          std::filesystem::directory_iterator(),
                          [](const std::filesystem::directory_entry& entry) {
                              return entry.is_regular_file() && isGraphFile(entry.path());
                          });
        if(!hasGraph)
        {
            emptyLeaves.push_back(leaf);
        }
    }
    return emptyLeaves;
}

} // namespace detail

inline void warnOnEmptyLeafFolders(const std::filesystem::path& bundleDir,
                                   const std::vector<std::filesystem::path>& sweepDirs)
{
    for(const auto& leaf : detail::findEmptyLeafFolders(bundleDir, sweepDirs))
    {
        HIPDNN_PLUGIN_LOG_WARN("Skipping empty bundle leaf folder (no graph .json): " << leaf);
    }
}

// Reads and validates the registration-time shape of sweep.json. Full case
// validation happens during loading, but discovery needs cases[] ids to be
// strings and unique so GTest names are deterministic and collision checks are
// meaningful.
inline std::vector<std::string> readSweepCaseIds(const std::filesystem::path& sweepPath)
{
    std::ifstream stream(sweepPath);
    if(!stream)
    {
        throw std::runtime_error("Could not open sweep manifest: " + sweepPath.string());
    }

    const auto sweepJson = nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false);
    if(sweepJson.is_discarded())
    {
        throw std::runtime_error("Sweep manifest is not parseable JSON: " + sweepPath.string());
    }
    if(!sweepJson.contains("cases") || !sweepJson.at("cases").is_array())
    {
        throw std::runtime_error("Sweep manifest missing cases[] array: " + sweepPath.string());
    }

    std::unordered_set<std::string> seenIds;
    std::vector<std::string> caseIds;
    caseIds.reserve(sweepJson.at("cases").size());

    for(const auto& caseJson : sweepJson.at("cases"))
    {
        if(!caseJson.is_object() || !caseJson.contains("id") || !caseJson.at("id").is_string())
        {
            throw std::runtime_error("Sweep case missing string id: " + sweepPath.string());
        }

        const auto caseId = caseJson.at("id").get<std::string>();
        if(caseId.empty())
        {
            throw std::runtime_error("Sweep case has empty id in " + sweepPath.string());
        }
        if(!seenIds.insert(caseId).second)
        {
            throw std::runtime_error("Duplicate sweep case id '" + caseId + "' in "
                                     + sweepPath.string());
        }
        caseIds.push_back(caseId);
    }

    return caseIds;
}

namespace detail
{

inline bool isJsonWhitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Counts the elements of the top-level "cases" array of a sweep manifest in one byte
// pass, without a DOM. Returns nullopt when the scan does not accept the shape: the
// text is not one complete object, or "cases" is missing, is not an array, occurs more
// than once, or holds an element that is not an object. The scan compares raw key
// bytes, so an escaped spelling of "cases" also gives nullopt.
inline std::optional<size_t> countSweepCasesWithoutParsing(std::string_view json)
{
    constexpr int OBJECT_DEPTH = 1;
    constexpr int CASES_ARRAY_DEPTH = 2;

    size_t position = 0;
    while(position < json.size() && isJsonWhitespace(json[position]))
    {
        ++position;
    }
    if(position == json.size() || json[position] != '{')
    {
        return std::nullopt;
    }

    int depth = 0;
    bool inString = false;
    bool escaped = false;
    size_t stringStart = 0;
    bool lastStringIsCasesKey = false;
    bool expectCasesArray = false;
    bool casesKeySeen = false;
    bool inCasesArray = false;
    bool casesArraySeen = false;
    bool rootClosed = false;
    size_t caseCount = 0;

    for(; position < json.size(); ++position)
    {
        const char c = json[position];
        if(inString)
        {
            if(escaped)
            {
                escaped = false;
            }
            else if(c == '\\')
            {
                escaped = true;
            }
            else if(c == '"')
            {
                inString = false;
                if(depth == OBJECT_DEPTH)
                {
                    lastStringIsCasesKey
                        = json.substr(stringStart, position - stringStart) == "cases";
                }
            }
            continue;
        }
        if(isJsonWhitespace(c))
        {
            continue;
        }
        if(rootClosed)
        {
            return std::nullopt;
        }

        if(inCasesArray && depth == CASES_ARRAY_DEPTH)
        {
            if(c == '{')
            {
                ++caseCount;
                ++depth;
            }
            else if(c == ']')
            {
                --depth;
                inCasesArray = false;
            }
            else if(c != ',')
            {
                return std::nullopt;
            }
            continue;
        }

        if(depth == OBJECT_DEPTH)
        {
            const bool startsCasesArray = c == '[' && expectCasesArray;
            const bool endsCasesKey = c == ':' && lastStringIsCasesKey;
            lastStringIsCasesKey = false;
            expectCasesArray = false;
            if(endsCasesKey)
            {
                if(casesKeySeen)
                {
                    return std::nullopt;
                }
                casesKeySeen = true;
                expectCasesArray = true;
                continue;
            }
            if(startsCasesArray)
            {
                inCasesArray = true;
                casesArraySeen = true;
            }
        }

        switch(c)
        {
        case '{':
        case '[':
            ++depth;
            break;
        case '}':
        case ']':
            if(depth == 0)
            {
                return std::nullopt;
            }
            --depth;
            rootClosed = depth == 0;
            break;
        case '"':
            inString = true;
            stringStart = position + 1;
            break;
        default:
            break;
        }
    }

    if(inString || !rootClosed || !casesArraySeen)
    {
        return std::nullopt;
    }
    return caseCount;
}

} // namespace detail

// The number of cases[] entries in a sweep manifest. A structural scan gives the count
// without a full parse. Manifests that the scan does not accept go through
// readSweepCaseIds(), which throws for a malformed manifest.
inline size_t countSweepCases(const std::filesystem::path& sweepPath)
{
    std::ifstream stream(sweepPath, std::ios::binary);
    if(!stream)
    {
        throw std::runtime_error("Could not open sweep manifest: " + sweepPath.string());
    }
    std::string content;
    stream.seekg(0, std::ios::end);
    const auto size = static_cast<std::streamoff>(stream.tellg());
    if(size > 0)
    {
        content.resize(static_cast<size_t>(size));
        stream.seekg(0, std::ios::beg);
        stream.read(content.data(), static_cast<std::streamsize>(size));
        content.resize(static_cast<size_t>(stream.gcount()));
    }
    if(const auto count = detail::countSweepCasesWithoutParsing(content))
    {
        return *count;
    }
    return readSweepCaseIds(sweepPath).size();
}

// Converts one sweep root into registerable logical tests. The suite name comes
// from the sweep directory; the test name comes from the sanitized case id.
inline std::vector<DiscoveredBundle> discoverSweepCases(const std::filesystem::path& sweepDir,
                                                        const std::filesystem::path& bundleDir)
{
    const auto sweepPath = sweepDir / "sweep.json";
    const auto templatePath = sweepDir / "graph.template.json";
    const auto suiteName = deriveSuiteName(relativeToRoot(sweepDir, bundleDir), sweepPath);

    std::vector<DiscoveredBundle> bundles;
    for(const auto& caseId : readSweepCaseIds(sweepPath))
    {
        bundles.push_back(
            {sweepPath, suiteName, sanitizeForGtest(caseId), SweepCase{templatePath, caseId}});
    }

    return bundles;
}

// Recursively discovers all registerable bundle tests under the data root.
//
// Direct bundles: every graph .json accepted by isGraphFile() becomes one test.
// Template sweeps: every sweep root becomes one test per cases[].id, and all JSON
// below the sweep root is skipped by direct graph discovery.
//
// Graceful handling:
//   - leaf folders without a graph .json warn and skip, so partial DVC pulls or
//     empty customer folders do not prevent other tests from registering.
//
// Hard errors:
//   - a malformed sweep manifest (unparseable, missing cases[], or a non-string
//     or duplicate case id) throws: a checked-in manifest must be coherent, so a
//     broken sweep fails the run rather than silently dropping its cases.
//   - generated GTest name collisions throw and name both diagnostic paths.
inline std::vector<DiscoveredBundle> discoverBundles(const std::filesystem::path& bundleDir)
{
    std::vector<DiscoveredBundle> bundles;
    std::unordered_map<std::string, std::filesystem::path> nameToPath;

    const auto sweepDirs = findSweepDirectories(bundleDir);
    warnOnEmptyLeafFolders(bundleDir, sweepDirs);

    auto registerBundle = [&](DiscoveredBundle bundle) {
        const auto fullName = bundle.fullName();
        const auto diagnosticPath = bundle.diagnosticPath();
        auto it = nameToPath.find(fullName);
        if(it != nameToPath.end())
        {
            throw std::runtime_error("Bundle name collision: '" + fullName
                                     + "' produced by both:\n  " + it->second.string() + "\n  "
                                     + diagnosticPath.string());
        }
        nameToPath[fullName] = diagnosticPath;
        bundles.push_back(std::move(bundle));
    };

    for(const auto& sweepDir : sweepDirs)
    {
        for(auto& bundle : discoverSweepCases(sweepDir, bundleDir))
        {
            registerBundle(std::move(bundle));
        }
    }

    const auto normalizedSweepDirs = detail::lexicallyNormalSet(sweepDirs);
    for(const auto& jsonPath : scanFilesByExtension(bundleDir, ".json"))
    {
        if(detail::hasAncestorIn(jsonPath, normalizedSweepDirs))
        {
            continue;
        }
        if(!isGraphFile(jsonPath))
        {
            continue;
        }

        const DerivedTestName derived = deriveTestName(jsonPath, bundleDir);
        registerBundle({jsonPath, derived.suiteName, derived.testName, std::nullopt});
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::bundle
