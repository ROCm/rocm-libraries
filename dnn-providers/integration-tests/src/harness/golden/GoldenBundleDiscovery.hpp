// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_integration_tests::golden
{

// Naming types, kept together. DerivedTestName is the output of deriveTestName()
// (naming only); DiscoveredBundle is what discoverGoldenBundles() returns for
// registration — the same two name fields plus the graph .json path to load.
// They overlap by design: one is a short-lived intermediate, the other the
// stored record. See deriveTestName() below for how the names are built.

// The two halves of a GTest identifier, as registered via RegisterTest().
// GTest joins them with '.' to form the full name: "{suiteName}.{testName}".
//
//   suiteName — computational identity: [{Tier}/]{op}_{layout}_{dtype}
//               (the tier prefix keeps its trailing '/', empty for the quick
//               tier; see tierPrefix). Shared by every bundle of the same
//               op/layout/dtype.
//   testName  — scenario identity: the bundle directory name (why the test
//               exists), e.g. "resnet50_layer3".
//
// Example: standard/ConvFwd/nhwc/fp16/resnet50_layer3/graph.json
//   suiteName = "Standard/ConvFwd_nhwc_fp16"
//   testName  = "resnet50_layer3"
//   full GTest name = "Standard/ConvFwd_nhwc_fp16.resnet50_layer3"
struct DerivedTestName
{
    std::string suiteName;
    std::string testName;
};

// One registerable golden test: a DerivedTestName plus the bundle's graph .json
// path the harness loads at run time. This is the unit discoverGoldenBundles
// returns — one per test that gets RegisterTest'd.
struct DiscoveredBundle
{
    std::filesystem::path jsonPath; // absolute path to the bundle graph .json
    std::string suiteName; // GTest suite, e.g. "Standard/ConvFwd_nhwc_fp16"
    std::string testName; // GTest test, e.g. "resnet50_layer3"
};

// Generic recursive file scanner: returns every file under `directory` whose
// extension matches `extension` (e.g. ".json"), sorted for deterministic test
// ordering. It carries NO golden-ref knowledge — meta-file exclusion is layered
// on top by the caller (see isGoldenMetaFile / discoverGoldenBundles). This is
// the clean split called for in ALMIOPEN-1968: a generic scan, with golden-ref
// filtering applied separately rather than baked into the directory walk.
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

// Golden-ref filter: true for companion .json files that are NOT bundle graphs
// and must be excluded from discovery. Currently only metadata: a bare
// `meta.json` or any `{Name}.meta.json`.
//
// NOTE: scanTier keeps every .json this returns false for, so this is the single
// chokepoint for "companion, not a graph." Any future non-graph companion (e.g.
// a planned support.json) MUST be added here — otherwise it is misregistered as
// a graph bundle and fails at load time as a spurious test.
inline bool isGoldenMetaFile(const std::filesystem::path& jsonPath)
{
    if(jsonPath.filename() == "meta.json")
    {
        return true;
    }
    const auto stem = jsonPath.stem().string();
    return stem.size() >= 5 && stem.substr(stem.size() - 5) == ".meta";
}

inline constexpr std::array<const char*, 4> K_TIER_NAMES
    = {"quick", "standard", "comprehensive", "full"};

// RFC 0011 §4.3 test-naming scheme: the tier becomes a GTest suite prefix.
// `quick` is the default smoke tier and carries no prefix; the others are
// Capitalized (e.g. `Standard/`). The prefix includes its trailing '/'.
inline std::string tierPrefix(const std::string& tierName)
{
    if(tierName == "quick")
    {
        return "";
    }
    if(tierName == "standard")
    {
        return "Standard/";
    }
    if(tierName == "comprehensive")
    {
        return "Comprehensive/";
    }
    if(tierName == "full")
    {
        return "Full/";
    }
    return "";
}

// Maps any non-[alnum_] char to '_' so a path segment is a legal GTest name
// component. Required, not redundant: golden tests register via RegisterTest(),
// which (unlike INSTANTIATE_TEST_SUITE_P's IsValidParamName) performs NO name
// validation — this is the only thing keeping golden test names legal. Repairs
// rather than rejects because folder names legitimately contain '-'/'.' (e.g.
// "resnet50-layer3.v2"). Assumes non-empty input; deriveTestName() rejects
// empty path segments upstream.
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

// Derives the GTest suite and test names from the bundle's folder path.
// Path convention: {tier}/{op}/{layout}/{dtype}/{bundle_name}/{file}.json
//
// NOTE — divergence from RFC 0011 §4.3: the RFC derives op/layout/dtype from
// the *graph JSON* ("suite name derived from graph content"). We deliberately
// derive them from the *folder path* instead, moving the responsibility for
// computational identity out of C++ and onto the directory layout — adding a
// bundle is purely "drop files in the right folders," no graph parsing here.
// This also handles mixed-precision graphs (e.g. fp16 in / fp8 weight / fp32
// out) cleanly: the author encodes the signature as a folder label (e.g.
// "fp16_fp8_fp32"), which a single graph dtype enum could not express.
//
// Trade-off: folder labels are NOT validated against the graph's actual
// dtype/layout, so a bundle can be mislabeled (an fp16 graph dropped under
// "fp32/") and the test name will silently lie about its content.
// TODO: add a discovery-time check that the {op}/{layout}/{dtype} labels agree
// with the graph JSON (cheap to fold into the tolerance-lookup story, which
// already parses op+dtype from the graph).
inline DerivedTestName deriveTestName(const std::filesystem::path& jsonPath,
                                      const std::string& tierName)
{
    const auto bundleDir = jsonPath.parent_path();
    const auto op = bundleDir.parent_path().parent_path().parent_path().filename().string();
    const auto layout = bundleDir.parent_path().parent_path().filename().string();
    const auto dtype = bundleDir.parent_path().filename().string();
    const auto bundleName = bundleDir.filename().string();

    if(op.empty() || layout.empty() || dtype.empty() || bundleName.empty())
    {
        throw std::runtime_error(
            "Golden bundle path has empty segment(s): " + jsonPath.string()
            + "; expected {tier}/{op}/{layout}/{dtype}/{bundle_name}/{file}.json");
    }

    const std::string suite = tierPrefix(tierName) + sanitizeForGtest(op) + "_"
                              + sanitizeForGtest(layout) + "_" + sanitizeForGtest(dtype);
    const std::string test = sanitizeForGtest(bundleName);

    return {suite, test};
}

// Scans a single tier directory for bundle .json files: a recursive scan with
// the golden-ref meta-file filter layered on top. This is the "recursive .json
// scan per tier" the ticket (ALMIOPEN-1968) describes. It deliberately does NOT
// own the root-level rules (stray-dir rejection, all-tiers-exist, cross-tier
// collision) — those need visibility across all tiers and live in
// discoverGoldenBundles, which is why that entry point takes the data root
// rather than a single tierDir.
inline std::vector<std::filesystem::path> scanTier(const std::filesystem::path& tierDir)
{
    std::vector<std::filesystem::path> jsonPaths;
    for(auto& p : scanFilesByExtension(tierDir, ".json"))
    {
        if(!isGoldenMetaFile(p))
        {
            jsonPaths.push_back(std::move(p));
        }
    }
    return jsonPaths;
}

// Recursively discovers golden bundles across every tier directory.
//
// Per ALMIOPEN-1968, structural problems are hard errors (throw), not warnings:
//   - a stray top-level directory that is not one of the four tiers
//   - a tier directory that is missing or empty
//   - a bundle placed at the wrong directory depth
//   - a generated test-name collision
//
// The caller registers tests only on success, so any throw aborts startup and
// surfaces the authoring mistake loudly rather than silently dropping coverage.
inline std::vector<DiscoveredBundle>
    discoverGoldenBundles(const std::filesystem::path& goldenDataDir)
{
    std::vector<DiscoveredBundle> bundles;
    std::unordered_map<std::string, std::filesystem::path> nameToPath;

    // Reject stray top-level directories that are not recognized tiers.
    for(const auto& entry : std::filesystem::directory_iterator(goldenDataDir))
    {
        if(!entry.is_directory())
        {
            continue;
        }
        auto dirName = entry.path().filename().string();
        const bool isTier = std::any_of(K_TIER_NAMES.begin(),
                                        K_TIER_NAMES.end(),
                                        [&](const char* tier) { return dirName == tier; });
        if(!isTier)
        {
            throw std::runtime_error("Unexpected top-level directory '" + dirName
                                     + "' in golden reference data at " + goldenDataDir.string()
                                     + "; expected one of: quick, standard, comprehensive, full");
        }
    }

    for(const auto& tierName : K_TIER_NAMES)
    {
        auto tierDir = goldenDataDir / tierName;
        if(!std::filesystem::exists(tierDir) || !std::filesystem::is_directory(tierDir))
        {
            throw std::runtime_error(
                "Golden reference tier directory missing: " + tierDir.string()
                + "; every tier (quick, standard, comprehensive, full) must exist");
        }

        const auto jsonPaths = scanTier(tierDir);
        if(jsonPaths.empty())
        {
            throw std::runtime_error("Golden reference tier directory is empty: " + tierDir.string()
                                     + "; every tier must contain at least one bundle");
        }

        for(const auto& jsonPath : jsonPaths)
        {
            // Validate path depth: {op}/{layout}/{dtype}/{bundle_name}/{file}.json = 5 components
            const auto relative = std::filesystem::relative(jsonPath, tierDir);
            size_t depth = 0;
            for(auto it = relative.begin(); it != relative.end(); ++it)
            {
                ++depth;
            }
            if(depth < 5)
            {
                throw std::runtime_error(
                    "Golden bundle at wrong directory depth: " + jsonPath.string()
                    + "; expected {tier}/{op}/{layout}/{dtype}/{bundle_name}/{file}.json");
            }

            const DerivedTestName derived = deriveTestName(jsonPath, tierName);

            auto fullName = derived.suiteName + "." + derived.testName;
            auto it = nameToPath.find(fullName);
            if(it != nameToPath.end())
            {
                throw std::runtime_error("Golden bundle name collision: '" + fullName
                                         + "' produced by both:\n  " + it->second.string() + "\n  "
                                         + jsonPath.string());
            }
            nameToPath[fullName] = jsonPath;

            bundles.push_back({jsonPath, derived.suiteName, derived.testName});
        }
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::golden
