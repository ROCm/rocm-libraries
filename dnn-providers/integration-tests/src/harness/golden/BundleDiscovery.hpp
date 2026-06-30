// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_integration_tests::golden
{

struct DerivedTestName
{
    std::string suiteName;
    std::string testName;
};

struct DiscoveredBundle
{
    std::filesystem::path jsonPath; // graph .json for single bundles, sweep.json for sweep cases
    std::string suiteName;
    std::string testName;
    std::filesystem::path templatePath; // set only for template-sweep cases
    std::string caseId; // set only for template-sweep cases

    bool isTemplateSweepCase() const
    {
        return !templatePath.empty();
    }

    std::filesystem::path diagnosticPath() const
    {
        if(!isTemplateSweepCase())
        {
            return jsonPath;
        }

        return {jsonPath.string() + "#" + caseId};
    }
};

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

inline const std::set<std::string>& companionKinds()
{
    static const std::set<std::string> s_kinds = {"meta"};
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
        bundles.push_back({sweepPath, suiteName, sanitizeForGtest(caseId), templatePath, caseId});
    }

    return bundles;
}

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
        try
        {
            for(auto& bundle : discoverSweepCases(sweepDir, bundleDir))
            {
                registerBundle(std::move(bundle));
            }
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_WARN("Skipping template-sweep bundle " << sweepDir << ": "
                                                                     << e.what());
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
        registerBundle({jsonPath, derived.suiteName, derived.testName, {}, {}});
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::golden
