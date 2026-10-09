// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SweepManifestCache.hpp"

#include <utility>

#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

SweepManifest::SweepManifest(nlohmann::json sweepJson)
    : _json(std::move(sweepJson))
{
    if(!_json.contains("cases") || !_json.at("cases").is_array())
    {
        return;
    }

    const auto& cases = _json.at("cases");
    _caseIndexById.reserve(cases.size());
    for(size_t index = 0; index < cases.size(); ++index)
    {
        const auto& caseJson = cases.at(index);
        if(caseJson.is_object() && caseJson.contains("id") && caseJson.at("id").is_string())
        {
            // emplace() leaves an id that is already present alone: the first one wins.
            _caseIndexById.emplace(caseJson.at("id").get<std::string>(), index);
        }
    }
}

std::optional<SweepManifest> SweepManifest::load(const std::filesystem::path& path)
{
    auto json = detail::parseJsonFile(path);
    if(!json.has_value())
    {
        return std::nullopt;
    }
    return SweepManifest(std::move(*json));
}

const nlohmann::json* SweepManifest::findCase(const std::string& caseId) const
{
    const auto it = _caseIndexById.find(caseId);
    return it != _caseIndexById.end() ? &_json.at("cases").at(it->second) : nullptr;
}

const SweepManifestCache::Sweep& SweepManifestCache::get(const DiscoveredBundle& discovered)
{
    if(_sweep.has_value() && _sweepPath == discovered.jsonPath
       && _templatePath == discovered.sweep->templatePath)
    {
        return *_sweep;
    }

    _sweepPath = discovered.jsonPath;
    _templatePath = discovered.sweep->templatePath;
    auto& sweep = _sweep.emplace();
    sweep.templateJson = detail::parseJsonFile(_templatePath);
    sweep.manifest = SweepManifest::load(_sweepPath);
    return sweep;
}

} // namespace hipdnn_integration_tests::bundle
