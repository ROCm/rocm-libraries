// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/GoldenOutputProbe.hpp"

#include <exception>
#include <string>
#include <system_error>
#include <utility>

#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle::detail
{

namespace
{

bool directoryHasTensorBlob(const std::filesystem::path& directory)
{
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(directory, error))
    {
        if(entry.path().extension() == ".bin"
           && entry.path().filename().string().rfind("tensor", 0) == 0)
        {
            return true;
        }
    }
    // An unreadable or absent directory is not evidence of absence.
    return static_cast<bool>(error);
}

bool hasOutputBlobSibling(const std::filesystem::path& jsonPath)
{
    const auto prefix = jsonPath.stem().string() + ".tensor";
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(jsonPath.parent_path(), error))
    {
        if(entry.path().extension() != ".bin")
        {
            continue;
        }
        const auto name = entry.path().filename().string();
        if(name.rfind(prefix, 0) == 0)
        {
            return true;
        }
    }
    // An unreadable directory is not evidence of absence.
    return static_cast<bool>(error);
}

} // namespace

bool GoldenOutputProbe::mayCarryGoldenOutputs(const DiscoveredBundle& disc)
{
    return disc.isTemplateSweepCase() ? sweepCaseHasGoldenBlobs(disc)
                                      : hasOutputBlobSibling(disc.jsonPath);
}

const SweepManifest* GoldenOutputProbe::sweepManifest(const std::filesystem::path& path)
{
    auto it = _manifests.find(path);
    if(it == _manifests.end())
    {
        it = _manifests.emplace(path, SweepManifest::load(path)).first;
    }
    return it->second.has_value() ? &*it->second : nullptr;
}

bool GoldenOutputProbe::sweepCaseHasGoldenBlobs(const DiscoveredBundle& disc)
{
    const auto* manifest = sweepManifest(disc.jsonPath);
    if(manifest == nullptr)
    {
        return true;
    }
    const auto* caseJson = manifest->findCase(disc.sweep->caseId);
    if(caseJson == nullptr)
    {
        return true;
    }

    std::optional<std::filesystem::path> goldenDirectory;
    try
    {
        goldenDirectory = detail::resolveSweepGoldenDirectory(disc.jsonPath, *caseJson);
    }
    catch(const std::exception&)
    {
        // A malformed `golden` block is an authoring error the loader reports
        // as INVALID_SWEEP_CASE. Let it through so it says so.
        return true;
    }
    if(!goldenDirectory.has_value())
    {
        return false;
    }
    return directoryHasTensorBlob(*goldenDirectory);
}

} // namespace hipdnn_integration_tests::bundle::detail
