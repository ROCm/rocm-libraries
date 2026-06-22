// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

#include "harness/golden/BundleDiscovery.hpp"

namespace hipdnn_integration_tests::golden
{

using TensorMap = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct IntegrationTestBundle
{
    flatbuffers::DetachedBuffer graphBuffer;
    hipdnn_test_sdk::utilities::BundleMetadata metadata;
    std::vector<int64_t> outputTensorUids;
    std::optional<TensorMap> tensors;

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper{graphBuffer.data(),
                                                                          graphBuffer.size()};
    }
};

enum class LoadError
{
    MALFORMED_JSON,
    INVALID_GRAPH_SCHEMA,
    MISSING_METADATA,
    TENSOR_LOAD_FAILED,
    INVALID_SWEEP_CASE
};

using LoadResult = std::variant<IntegrationTestBundle, LoadError>;

inline const char* toString(LoadError error)
{
    switch(error)
    {
    case LoadError::MALFORMED_JSON:
        return "graph JSON is not parseable";
    case LoadError::INVALID_GRAPH_SCHEMA:
        return "graph JSON is not a valid graph";
    case LoadError::MISSING_METADATA:
        return "missing or invalid .meta.json companion";
    case LoadError::TENSOR_LOAD_FAILED:
        return "tensor .bin present but failed to load";
    case LoadError::INVALID_SWEEP_CASE:
        return "template-sweep case is invalid";
    default:
        return "unknown load error";
    }
}

namespace detail
{

inline std::optional<nlohmann::json> parseJsonFile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if(!stream)
    {
        return std::nullopt;
    }

    auto json = nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false);
    if(json.is_discarded())
    {
        return std::nullopt;
    }

    return std::make_optional<nlohmann::json>(std::move(json));
}

inline bool buildGraphBuffer(const nlohmann::json& graphJson,
                             flatbuffers::DetachedBuffer& graphBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    try
    {
        auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            builder, graphJson);
        builder.Finish(offset);
    }
    catch(const std::exception&)
    {
        return false;
    }

    graphBuffer = builder.Release();
    return true;
}

template <typename BlobPathFn>
inline bool allTensorBlobsPresent(const nlohmann::json& graphJson, BlobPathFn&& blobPathForUid)
{
    if(!graphJson.contains("tensors") || !graphJson.at("tensors").is_array()
       || graphJson.at("tensors").empty())
    {
        return false;
    }

    for(const auto& tensor : graphJson.at("tensors"))
    {
        if(!tensor.contains("uid") || !tensor.at("uid").is_number_integer())
        {
            return false;
        }

        const auto uid = tensor.at("uid").get<int64_t>();
        if(!std::filesystem::exists(blobPathForUid(uid)))
        {
            return false;
        }
    }

    return true;
}

template <typename BlobPathFn>
inline std::optional<LoadError> loadTensorDataIfPresent(IntegrationTestBundle& bundle,
                                                        const nlohmann::json& graphJson,
                                                        BlobPathFn&& blobPathForUid)
{
    if(!allTensorBlobsPresent(graphJson, blobPathForUid))
    {
        return std::nullopt;
    }

    const auto& graph = *hipdnn_flatbuffers_sdk::data_objects::GetGraph(bundle.graphBuffer.data());

    try
    {
        TensorMap tensorMap;
        for(const auto* attributes : *graph.tensors())
        {
            tensorMap[attributes->uid()] = hipdnn_test_sdk::utilities::tensorFromFileAndAttributes(
                blobPathForUid(attributes->uid()), *attributes);
        }
        bundle.tensors = std::move(tensorMap);
    }
    catch(const std::exception&)
    {
        return LoadError::TENSOR_LOAD_FAILED;
    }

    return std::nullopt;
}

inline std::string firstPathToken(const std::string& path)
{
    const auto dot = path.find('.');
    return dot == std::string::npos ? path : path.substr(0, dot);
}

inline const nlohmann::json* lookupJsonPath(const nlohmann::json& json, const std::string& path)
{
    const nlohmann::json* current = &json;
    std::size_t start = 0;
    while(start < path.size())
    {
        const auto end = path.find('.', start);
        const auto key
            = path.substr(start, end == std::string::npos ? path.size() - start : end - start);
        if(!current->is_object() || !current->contains(key))
        {
            return nullptr;
        }
        current = &current->at(key);
        if(end == std::string::npos)
        {
            return current;
        }
        start = end + 1;
    }

    return current;
}

inline std::optional<std::string> placeholderField(const nlohmann::json& json)
{
    if(!json.is_string())
    {
        return std::nullopt;
    }

    const auto& value = json.get_ref<const std::string&>();
    constexpr std::string_view prefix = "${case.";
    if(value.size() <= prefix.size() + 1 || value.rfind(prefix.data(), 0) != 0
       || value.back() != '}')
    {
        return std::nullopt;
    }

    return value.substr(prefix.size(), value.size() - prefix.size() - 1);
}

inline bool requiresPerTensorValue(const std::string& fieldPath)
{
    const auto field = firstPathToken(fieldPath);
    return field == "dims" || field == "strides" || field == "data_type";
}

struct SweepUseTracker
{
    std::unordered_set<std::string> usedValueKeys;
    std::unordered_map<int64_t, std::unordered_set<std::string>> usedTensorKeys;
};

inline std::unordered_set<int64_t> collectTemplateTensorUids(const nlohmann::json& templateJson)
{
    std::unordered_set<int64_t> uids;
    if(!templateJson.contains("tensors") || !templateJson.at("tensors").is_array())
    {
        return uids;
    }

    for(const auto& tensorJson : templateJson.at("tensors"))
    {
        if(!tensorJson.is_object() || !tensorJson.contains("uid")
           || !tensorJson.at("uid").is_number_integer())
        {
            throw std::runtime_error("Template tensor missing integer uid");
        }
        uids.insert(tensorJson.at("uid").get<int64_t>());
    }

    return uids;
}

inline std::unordered_map<int64_t, const nlohmann::json*>
    buildCaseTensorMap(const nlohmann::json& caseValues,
                       const std::unordered_set<int64_t>& templateTensorUids)
{
    std::unordered_map<int64_t, const nlohmann::json*> caseTensors;

    if(!caseValues.contains("tensors"))
    {
        return caseTensors;
    }
    if(!caseValues.at("tensors").is_array())
    {
        throw std::runtime_error("values.tensors must be an array");
    }

    for(const auto& tensorJson : caseValues.at("tensors"))
    {
        if(!tensorJson.is_object() || !tensorJson.contains("uid")
           || !tensorJson.at("uid").is_number_integer())
        {
            throw std::runtime_error("Case tensor missing integer uid");
        }

        const auto uid = tensorJson.at("uid").get<int64_t>();
        if(templateTensorUids.find(uid) == templateTensorUids.end())
        {
            throw std::runtime_error("Case tensor uid not present in template graph");
        }
        if(!caseTensors.emplace(uid, &tensorJson).second)
        {
            throw std::runtime_error("Duplicate case tensor uid");
        }
    }

    return caseTensors;
}

inline const nlohmann::json&
    resolvePlaceholder(const std::string& fieldPath,
                       const std::optional<int64_t>& currentTensorUid,
                       const nlohmann::json& caseValues,
                       const std::unordered_map<int64_t, const nlohmann::json*>& caseTensors,
                       SweepUseTracker& useTracker)
{
    if(currentTensorUid.has_value())
    {
        auto tensorIt = caseTensors.find(*currentTensorUid);
        if(tensorIt != caseTensors.end())
        {
            if(const auto* tensorValue = lookupJsonPath(*tensorIt->second, fieldPath))
            {
                useTracker.usedTensorKeys[*currentTensorUid].insert(firstPathToken(fieldPath));
                return *tensorValue;
            }
        }

        if(requiresPerTensorValue(fieldPath))
        {
            throw std::runtime_error("Missing per-tensor placeholder value");
        }
    }

    if(const auto* value = lookupJsonPath(caseValues, fieldPath))
    {
        useTracker.usedValueKeys.insert(firstPathToken(fieldPath));
        return *value;
    }

    throw std::runtime_error("Missing placeholder value");
}

inline nlohmann::json
    expandTemplateNode(const nlohmann::json& node,
                       const std::optional<int64_t>& currentTensorUid,
                       const nlohmann::json& caseValues,
                       const std::unordered_map<int64_t, const nlohmann::json*>& caseTensors,
                       SweepUseTracker& useTracker)
{
    if(const auto placeholder = placeholderField(node))
    {
        return resolvePlaceholder(
            *placeholder, currentTensorUid, caseValues, caseTensors, useTracker);
    }

    if(node.is_array())
    {
        nlohmann::json expanded = nlohmann::json::array();
        for(const auto& item : node)
        {
            expanded.push_back(
                expandTemplateNode(item, currentTensorUid, caseValues, caseTensors, useTracker));
        }
        return expanded;
    }

    if(node.is_object())
    {
        auto nextTensorUid = currentTensorUid;
        if(node.contains("uid") && node.at("uid").is_number_integer())
        {
            nextTensorUid = node.at("uid").get<int64_t>();
        }

        auto expanded = nlohmann::json::object();
        for(const auto& [key, value] : node.items())
        {
            expanded[key]
                = expandTemplateNode(value, nextTensorUid, caseValues, caseTensors, useTracker);
        }
        return expanded;
    }

    return node;
}

inline void warnUnusedSweepValues(const std::filesystem::path& diagnosticPath,
                                  const nlohmann::json& caseValues,
                                  SweepUseTracker& useTracker)
{
    if(caseValues.is_object())
    {
        for(const auto& [key, value] : caseValues.items())
        {
            if(key == "tensors"
               || useTracker.usedValueKeys.find(key) != useTracker.usedValueKeys.end())
            {
                continue;
            }
            HIPDNN_SDK_LOG_WARN("Unused sweep value '" << key << "' in " << diagnosticPath);
        }
    }

    if(caseValues.contains("tensors") && caseValues.at("tensors").is_array())
    {
        for(const auto& tensorJson : caseValues.at("tensors"))
        {
            if(!tensorJson.is_object() || !tensorJson.contains("uid")
               || !tensorJson.at("uid").is_number_integer())
            {
                continue;
            }

            const auto uid = tensorJson.at("uid").get<int64_t>();
            const auto& usedKeys = useTracker.usedTensorKeys[uid];
            for(const auto& [key, value] : tensorJson.items())
            {
                if(key == "uid" || usedKeys.find(key) != usedKeys.end())
                {
                    continue;
                }
                HIPDNN_SDK_LOG_WARN("Unused sweep tensor value '" << key << "' for uid " << uid
                                                                  << " in " << diagnosticPath);
            }
        }
    }
}

inline nlohmann::json expandTemplateGraph(const nlohmann::json& templateJson,
                                          const nlohmann::json& caseJson,
                                          const DiscoveredBundle& discovered)
{
    const auto caseValues = caseJson.contains("values") && caseJson.at("values").is_object()
                                ? caseJson.at("values")
                                : nlohmann::json::object();

    auto useTracker = SweepUseTracker{};
    const auto templateTensorUids = collectTemplateTensorUids(templateJson);
    const auto caseTensors = buildCaseTensorMap(caseValues, templateTensorUids);
    auto expanded
        = expandTemplateNode(templateJson, std::nullopt, caseValues, caseTensors, useTracker);
    warnUnusedSweepValues(discovered.diagnosticPath(), caseValues, useTracker);
    return expanded;
}

inline const nlohmann::json* findSweepCase(const nlohmann::json& sweepJson,
                                           const std::string& caseId)
{
    if(!sweepJson.contains("cases") || !sweepJson.at("cases").is_array())
    {
        return nullptr;
    }

    for(const auto& caseJson : sweepJson.at("cases"))
    {
        if(caseJson.is_object() && caseJson.contains("id") && caseJson.at("id").is_string()
           && caseJson.at("id").get<std::string>() == caseId)
        {
            return &caseJson;
        }
    }

    return nullptr;
}

inline std::optional<std::filesystem::path>
    resolveSweepGoldenDirectory(const std::filesystem::path& sweepPath,
                                const nlohmann::json& caseJson)
{
    if(!caseJson.contains("golden") || caseJson.at("golden").is_null())
    {
        return std::nullopt;
    }
    if(!caseJson.at("golden").is_object() || !caseJson.at("golden").contains("path")
       || !caseJson.at("golden").at("path").is_string())
    {
        throw std::runtime_error("Sweep case golden.path is required when golden is present");
    }

    const auto goldenPath
        = sweepPath.parent_path() / caseJson.at("golden").at("path").get<std::string>();
    return goldenPath.parent_path();
}

inline std::optional<hipdnn_test_sdk::utilities::BundleMetadata>
    loadSweepMetadata(const std::filesystem::path& diagnosticPath, const nlohmann::json& caseJson)
{
    if(!caseJson.contains("metadata") || caseJson.at("metadata").is_null())
    {
        return hipdnn_test_sdk::utilities::BundleMetadata{};
    }

    return hipdnn_test_sdk::utilities::parseBundleMetadataJson(caseJson.at("metadata"),
                                                               diagnosticPath.string());
}

} // namespace detail

inline LoadResult loadIntegrationTestBundle(const std::filesystem::path& jsonPath)
{
    const auto graphJson = detail::parseJsonFile(jsonPath);
    if(!graphJson.has_value())
    {
        return LoadError::MALFORMED_JSON;
    }

    flatbuffers::DetachedBuffer graphBuffer;
    if(!detail::buildGraphBuffer(*graphJson, graphBuffer))
    {
        return LoadError::INVALID_GRAPH_SCHEMA;
    }

    auto metadata = hipdnn_test_sdk::utilities::loadBundleMetadata(jsonPath);
    if(!metadata.has_value())
    {
        return LoadError::MISSING_METADATA;
    }

    IntegrationTestBundle bundle;
    bundle.graphBuffer = std::move(graphBuffer);
    bundle.metadata = std::move(*metadata);
    bundle.outputTensorUids = hipdnn_test_sdk::utilities::getOutputTensorUidsFromGraph(*graphJson);

    auto basePath = jsonPath;
    basePath.replace_extension();
    if(const auto loadError = detail::loadTensorDataIfPresent(
           bundle,
           *graphJson,
           [&](int64_t uid) {
               return std::filesystem::path(basePath.string() + ".tensor" + std::to_string(uid)
                                            + ".bin");
           });
       loadError.has_value())
    {
        return *loadError;
    }

    return bundle;
}

inline LoadResult loadIntegrationTestBundle(const DiscoveredBundle& discovered)
{
    if(!discovered.isTemplateSweepCase())
    {
        return loadIntegrationTestBundle(discovered.jsonPath);
    }

    const auto templateJson = detail::parseJsonFile(discovered.templatePath);
    const auto sweepJson = detail::parseJsonFile(discovered.jsonPath);
    if(!templateJson.has_value() || !sweepJson.has_value())
    {
        return LoadError::MALFORMED_JSON;
    }

    const auto* caseJson = detail::findSweepCase(*sweepJson, discovered.caseId);
    if(caseJson == nullptr)
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    const auto metadata = detail::loadSweepMetadata(discovered.diagnosticPath(), *caseJson);
    if(!metadata.has_value())
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    nlohmann::json expandedGraph;
    std::optional<std::filesystem::path> goldenDirectory;
    try
    {
        expandedGraph = detail::expandTemplateGraph(*templateJson, *caseJson, discovered);
        goldenDirectory = detail::resolveSweepGoldenDirectory(discovered.jsonPath, *caseJson);
    }
    catch(const std::exception&)
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    flatbuffers::DetachedBuffer graphBuffer;
    if(!detail::buildGraphBuffer(expandedGraph, graphBuffer))
    {
        return LoadError::INVALID_GRAPH_SCHEMA;
    }

    IntegrationTestBundle bundle;
    bundle.graphBuffer = std::move(graphBuffer);
    bundle.metadata = std::move(*metadata);
    bundle.outputTensorUids
        = hipdnn_test_sdk::utilities::getOutputTensorUidsFromGraph(expandedGraph);

    if(goldenDirectory.has_value())
    {
        if(const auto loadError = detail::loadTensorDataIfPresent(
               bundle,
               expandedGraph,
               [&](int64_t uid) {
                   return *goldenDirectory / ("tensor" + std::to_string(uid) + ".bin");
               });
           loadError.has_value())
        {
            return *loadError;
        }
    }

    return bundle;
}

} // namespace hipdnn_integration_tests::golden
