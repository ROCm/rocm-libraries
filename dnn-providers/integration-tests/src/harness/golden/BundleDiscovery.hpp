// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_integration_tests::golden
{

// Naming types, kept together. DerivedTestName is the output of deriveTestName()
// (naming only); DiscoveredBundle is what discoverBundles() returns for
// registration — the same two name fields plus the graph .json path to load.
// They overlap by design: one is a short-lived intermediate, the other the
// stored record. See deriveTestName() below for how the names are built.

// The two halves of a GTest identifier, as registered via RegisterTest().
// GTest joins them with '.' to form the full name: "{suiteName}.{testName}".
//
//   suiteName — the bundle's directory path relative to the data root, with each
//               segment joined by '_'. Groups bundles that share a folder path.
//   testName  — the graph .json file stem (why the test exists), e.g. "Small".
//
// Example: quick/BatchnormFwdInference/ncdhw/fp32/Small/Small.json
//   suiteName = "quick_BatchnormFwdInference_ncdhw_fp32_Small"
//   testName  = "Small"
//   full GTest name = "quick_BatchnormFwdInference_ncdhw_fp32_Small.Small"
struct DerivedTestName
{
    std::string suiteName;
    std::string testName;
};

// One registerable test bundle: a DerivedTestName plus the bundle's graph .json
// path the harness loads at run time. This is the unit discoverBundles returns —
// one per test that gets RegisterTest'd.
struct DiscoveredBundle
{
    std::filesystem::path jsonPath; // absolute path to the bundle graph .json
    std::string suiteName; // GTest suite, e.g. "quick_BatchnormFwdInference_ncdhw_fp32_Small"
    std::string testName; // GTest test, e.g. "Small"
};

// Generic recursive file scanner: returns every file under `directory` whose
// extension matches `extension` (e.g. ".json"), sorted for deterministic test
// ordering. It carries NO bundle knowledge — graph-vs-companion filtering is
// layered on top by the caller (see isGraphFile / discoverBundles). This is the
// clean split called for in ALMIOPEN-1968: a generic scan, with bundle filtering
// applied separately rather than baked into the directory walk.
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

// Returns every leaf directory at or under `root` (a leaf has no subdirectories),
// including `root` itself when it has no subdirectories. A leaf is where a bundle
// is expected to live, so the caller uses this to reject leaf folders that hold
// no graph .json (an authoring mistake — an empty bundle folder). Intermediate
// directories (those with subdirectories) are just path segments and are not
// returned.
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

// Companion file "kinds" — the suffixes that mark a .json as a companion of a
// graph rather than a graph itself. A companion is either "{Name}.{kind}.json"
// (e.g. Small.meta.json) or the bare "{kind}.json" (e.g. meta.json). When a new
// companion type is added (e.g. per-graph claims, RFC 0015), add its kind here —
// this one list is the single place discovery learns about it.
inline const std::set<std::string>& companionKinds()
{
    static const std::set<std::string> s_kinds = {"meta"};
    return s_kinds;
}

// Discovery predicate (ALLOWLIST): true only for a bundle GRAPH .json — the file
// that should become a test. A .json is a COMPANION (not a graph) when its name
// is "{kind}.json" or "{Name}.{kind}.json" for a known kind in companionKinds();
// everything else is a graph.
//
// We match the trailing dotted segment against the known-kind list rather than
// rejecting "any dotted stem". That distinction matters: a legitimate graph may
// embed dots in its name (e.g. model.fp16.json, resnet50.v2.json), and the
// "drop a folder, it runs" contract for ad-hoc customer bundles means such a
// file must still be discovered as a graph — not silently dropped as if it were
// a companion. Only recognized companion kinds are excluded.
inline bool isGraphFile(const std::filesystem::path& jsonPath)
{
    if(jsonPath.extension() != ".json")
    {
        return false;
    }
    const auto stem = jsonPath.stem().string();

    // Bare companion: "meta.json" (stem == "meta").
    if(companionKinds().count(stem) != 0)
    {
        return false;
    }

    // Suffixed companion: "{Name}.{kind}.json" — check the segment after the
    // last '.'. A graph whose name merely contains dots (kind not recognized)
    // falls through and is treated as a graph.
    const auto dot = stem.rfind('.');
    return dot == std::string::npos || companionKinds().count(stem.substr(dot + 1)) == 0;
}

// Maps any non-[alnum_] char to '_' so a path segment is a legal GTest name
// component. Required, not redundant: bundle tests register via RegisterTest(),
// which (unlike INSTANTIATE_TEST_SUITE_P's IsValidParamName) performs NO name
// validation — this is the only thing keeping bundle test names legal. Repairs
// rather than rejects because folder names legitimately contain '-'/'.' (e.g.
// "resnet50-layer3.v2").
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

// Derives the GTest suite and test names from the bundle's path relative to the
// data root. Convention: a bundle lives in its own folder containing the graph
// .json, at least one folder below the root: {dir...}/{bundle}/{file}.json.
//   suiteName = the relative directory path, segments sanitized and joined by '_'
//   testName  = the .json file stem
// e.g. quick/Batchnorm/ncdhw/fp32/Small/Small.json (rel. to root) ->
//   suite "quick_Batchnorm_ncdhw_fp32_Small", test "Small".
// A flat customer drop case_23421/graph.json -> suite "case_23421", test "graph".
//
// NOTE — divergence from RFC 0011 §4.3: the RFC derives op/layout/dtype from the
// *graph JSON* and uses fixed tier folders. We deliberately impose NO structure:
// any folder containing a graph .json is a runnable bundle, named purely from its
// path. Structural correctness (layout, label-vs-graph consistency, well-formed
// bundles) is enforced out-of-band by the Python bundle verifier, not here, so
// "drop a folder, it runs" works for ad-hoc customer bundles.
//
// Consequence: op/dtype are NOT separable from the joined suite string. Tolerance
// lookup (ALMIOPEN-1969) must read op/dtype from the graph JSON, not the folder.
//
// Throws if the .json sits directly at the data root (no folder above it), since
// that yields an empty suite name.
inline DerivedTestName deriveTestName(const std::filesystem::path& jsonPath,
                                      const std::filesystem::path& bundleDir)
{
    const auto relative = std::filesystem::relative(jsonPath, bundleDir);
    const auto relativeDir = relative.parent_path();

    if(relativeDir.empty())
    {
        throw std::runtime_error(
            "Bundle .json must live in a sub-folder of the data root, not at the root itself: "
            + jsonPath.string() + "; expected {folder}/{file}.json");
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

    const std::string test = sanitizeForGtest(jsonPath.stem().string());

    return {suite, test};
}

// Recursively discovers bundles under the data root. Every graph .json (see
// isGraphFile — i.e. any .json that is not a companion) is a bundle; its GTest
// name comes from its path via deriveTestName. No fixed folder structure is
// required — see deriveTestName for the rationale (structure is validated
// out-of-band by the Python verifier).
//
// Graceful handling:
//   - a leaf folder (no subdirectories) that contains no graph .json is warned
//     and skipped. Partial DVC pulls can leave empty folders; aborting the
//     entire binary would prevent all other tests from running. A completely
//     empty data root returns an empty vector (no bundles).
//
// Hard errors (throw):
//   - a generated test-name collision: two bundles whose paths sanitize to the
//     same GTest name (naming both paths so the clash is obvious).
inline std::vector<DiscoveredBundle> discoverBundles(const std::filesystem::path& bundleDir)
{
    std::vector<DiscoveredBundle> bundles;
    std::unordered_map<std::string, std::filesystem::path> nameToPath;

    // Every leaf directory must hold at least one graph .json (non-meta).
    for(const auto& leaf : findLeafDirectories(bundleDir))
    {
        const bool hasGraph = std::any_of(std::filesystem::directory_iterator(leaf),
                                          std::filesystem::directory_iterator(),
                                          [](const std::filesystem::directory_entry& e) {
                                              return e.is_regular_file() && isGraphFile(e.path());
                                          });
        if(!hasGraph)
        {
            HIPDNN_PLUGIN_LOG_WARN("Skipping empty bundle leaf folder (no graph .json): " << leaf);
        }
    }

    for(const auto& jsonPath : scanFilesByExtension(bundleDir, ".json"))
    {
        if(!isGraphFile(jsonPath))
        {
            continue; // companion (.meta.json, future .claims.json, ...), not a graph
        }

        const DerivedTestName derived = deriveTestName(jsonPath, bundleDir);

        auto fullName = derived.suiteName + "." + derived.testName;
        auto it = nameToPath.find(fullName);
        if(it != nameToPath.end())
        {
            throw std::runtime_error("Bundle name collision: '" + fullName
                                     + "' produced by both:\n  " + it->second.string() + "\n  "
                                     + jsonPath.string());
        }
        nameToPath[fullName] = jsonPath;

        bundles.push_back({jsonPath, derived.suiteName, derived.testName});
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::golden
