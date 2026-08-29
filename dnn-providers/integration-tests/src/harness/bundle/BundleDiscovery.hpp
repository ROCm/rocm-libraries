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
#include <unordered_map>
#include <unordered_set>
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

    std::filesystem::path diagnosticPath() const
    {
        if(!isTemplateSweepCase())
        {
            return jsonPath;
        }

        return {jsonPath.string() + "#" + sweep->caseId};
    }
};

// Generic recursive file scanner. It carries no bundle knowledge; graph vs
// companion vs sweep filtering is layered on top by isGraphFile() and
// discoverBundles().
inline std::vector<std::filesystem::path>
    scanFilesByExtension(const std::filesystem::path& directory, const std::string& extension)
{
    std::vector<std::filesystem::path> paths;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(directory))
    {
        if(entry.is_regular_file() && entry.path().extension() == extension)
        {
            paths.push_back(entry.path());
        }
    }
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
    for(const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if(entry.is_directory())
        {
            allDirs.insert(entry.path());
            withSubdir.insert(entry.path().parent_path());
        }
    }

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
    const auto relative = std::filesystem::relative(jsonPath, bundleDir);
    const auto relativeDir = relative.parent_path();
    if(relativeDir.empty())
    {
        return {sanitizeForGtest(bundleDir.filename().string()),
                sanitizeForGtest(jsonPath.stem().string())};
    }

    return {deriveSuiteName(relativeDir, jsonPath), sanitizeForGtest(jsonPath.stem().string())};
}

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

    for(const auto& entry : std::filesystem::recursive_directory_iterator(bundleDir))
    {
        if(entry.is_directory() && isSweepBundleRoot(entry.path()))
        {
            sweepDirs.insert(entry.path());
        }
    }

    return {sweepDirs.begin(), sweepDirs.end()};
}

// Golden tensor directories under a sweep are leaves by design; they are data,
// not incomplete bundle folders, so empty-leaf warnings skip them.
inline bool isSweepGoldenLeaf(const std::filesystem::path& leaf,
                              const std::vector<std::filesystem::path>& sweepDirs)
{
    return std::any_of(sweepDirs.begin(), sweepDirs.end(), [&](const auto& sweepDir) {
        return isDescendantOf(leaf, sweepDir / "golden");
    });
}

inline void warnOnEmptyLeafFolders(const std::filesystem::path& bundleDir,
                                   const std::vector<std::filesystem::path>& sweepDirs)
{
    for(const auto& leaf : findLeafDirectories(bundleDir))
    {
        if(isSweepBundleRoot(leaf) || isSweepGoldenLeaf(leaf, sweepDirs))
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
            HIPDNN_PLUGIN_LOG_WARN("Skipping empty bundle leaf folder (no graph .json): " << leaf);
        }
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
        if(!seenIds.insert(caseId).second)
        {
            throw std::runtime_error("Duplicate sweep case id '" + caseId + "' in "
                                     + sweepPath.string());
        }
        caseIds.push_back(caseId);
    }

    return caseIds;
}

// Converts one sweep root into registerable logical tests. The suite name comes
// from the sweep directory; the test name comes from the sanitized case id.
inline std::vector<DiscoveredBundle> discoverSweepCases(const std::filesystem::path& sweepDir,
                                                        const std::filesystem::path& bundleDir)
{
    const auto sweepPath = sweepDir / "sweep.json";
    const auto templatePath = sweepDir / "graph.template.json";
    const auto suiteName
        = deriveSuiteName(std::filesystem::relative(sweepDir, bundleDir), sweepPath);

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
        const auto fullName = bundle.suiteName + "." + bundle.testName;
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

    for(const auto& jsonPath : scanFilesByExtension(bundleDir, ".json"))
    {
        if(std::any_of(sweepDirs.begin(), sweepDirs.end(), [&](const auto& sweepDir) {
               return isDescendantOf(jsonPath, sweepDir);
           }))
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
