// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

#include "harness/bundle/BundleDiscovery.hpp"

namespace hipdnn_integration_tests::bundle
{

// Loaded tensors keyed by tensor UID. Inputs carry their data. Outputs carry
// expected golden values only when output blobs are present; otherwise the
// harness verifies outputs against a reference executor.
using TensorMap = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// One test's worth of bundle data loaded from disk.
//
//   graphBuffer      — the parsed graph, as a flatbuffer. Always present in a
//                      loaded bundle; the engine deserializes it (from_binary)
//                      and the harness walks it (GraphWrapper) for dtypes and
//                      tolerances. A bundle that cannot produce a graph is a
//                      LoadError, not a bundle.
//   metadata         — .meta.json contents for direct bundles, or inline sweep
//                      metadata for template-sweep cases. Metadata is mandatory
//                      only when golden output blobs are present; graph-only and
//                      reference-verified bundles default to empty metadata.
//   outputTensorUids — UIDs of the graph's output tensors, derived from the
//                      graph. Always available, even for graph-only bundles, so
//                      the harness knows which tensors to compare or allocate.
//   tensors          — loaded tensor data, keyed by uid. Present when input blobs
//                      are available. If present and hasGoldenOutputs is false,
//                      it carries inputs only and outputs are reference-verified.
//                      Absent means the bundle is graph-only; the harness may
//                      fill inputs, otherwise it skips the case.
//   hasGoldenOutputs — true iff every output tensor's .bin blob was present and
//                      loaded into `tensors`. When false, engine output must be
//                      checked against a reference executor instead of golden data.
struct IntegrationTestBundle
{
    flatbuffers::DetachedBuffer graphBuffer;
    hipdnn_integration_tests::BundleMetadata metadata;
    std::vector<int64_t> outputTensorUids;
    std::optional<TensorMap> tensors;
    bool hasGoldenOutputs = false;

    // View over the graph flatbuffer, valid as long as this bundle lives.
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper{graphBuffer.data(),
                                                                          graphBuffer.size()};
    }
};

// Why a load did NOT produce a bundle. These are authoring failures in the
// bundle or sweep case. A valid graph-only bundle is still a loaded bundle and is
// skipped later only if the harness cannot fill inputs.
enum class LoadError
{
    MALFORMED_JSON, // graph/template/sweep JSON is unreadable or syntactically invalid
    INVALID_GRAPH_SCHEMA, // expanded graph JSON cannot build a valid graph flatbuffer
    MISSING_METADATA, // direct golden bundle is missing valid .meta.json metadata
    TENSOR_LOAD_FAILED, // a present tensor blob is unreadable, wrong-sized, or unsupported
    INVALID_SWEEP_CASE // sweep case id, placeholders, metadata, or golden path are invalid
};

// A load either yields a bundle or explains why it could not. std::visit at the
// call site forces both cases to be handled.
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

inline std::filesystem::path tensorBlobPath(const std::filesystem::path& jsonPath, int64_t uid)
{
    auto basePath = jsonPath;
    basePath.replace_extension();
    return {basePath.string() + ".tensor" + std::to_string(uid) + ".bin"};
}

template <typename BlobPathFn>
inline bool blobsPresentFor(const std::vector<int64_t>& uids, BlobPathFn&& blobPathForUid)
{
    for(const int64_t uid : uids)
    {
        if(!std::filesystem::exists(blobPathForUid(uid)))
        {
            return false;
        }
    }
    return true;
}

inline std::vector<int64_t> allTensorUids(const nlohmann::json& graphJson)
{
    std::vector<int64_t> uids;
    if(!graphJson.contains("tensors") || !graphJson.at("tensors").is_array())
    {
        return uids;
    }
    for(const auto& tensor : graphJson.at("tensors"))
    {
        if(tensor.contains("uid"))
        {
            uids.push_back(tensor.at("uid").get<int64_t>());
        }
    }
    return uids;
}

// Replaces test_sdk's getOutputTensorUidsFromGraph(), which assumes every
// node wraps output UIDs in an "outputs" sub-object. ReductionAttributes,
// ResampleFwdAttributes, and CustomOpAttributes emit flat keys instead,
// causing node.at("outputs") to throw for those ops.
inline std::vector<int64_t> extractOutputUidsFromJson(const nlohmann::json& graphJson)
{
    std::vector<int64_t> uids;
    if(!graphJson.contains("nodes") || !graphJson.at("nodes").is_array())
    {
        return uids;
    }

    for(const auto& node : graphJson.at("nodes"))
    {
        // Standard: outputs wrapped in "outputs" sub-object
        if(node.contains("outputs") && node.at("outputs").is_object())
        {
            for(auto& [name, value] : node.at("outputs").items())
            {
                if(name.find("_tensor_uid") != std::string::npos && !value.is_null()
                   && value.is_number_integer())
                {
                    uids.push_back(value.get<int64_t>());
                }
            }
            continue;
        }

        // Flat vector: CustomOp uses "output_tensor_uids" array
        if(node.contains("output_tensor_uids") && node.at("output_tensor_uids").is_array())
        {
            for(const auto& v : node.at("output_tensor_uids"))
            {
                if(v.is_number_integer())
                {
                    uids.push_back(v.get<int64_t>());
                }
            }
            continue;
        }

        // Flat scalar fallback: Reduction ("out_tensor_uid"),
        // ResampleFwd ("y_tensor_uid")
        for(const auto& key : {"out_tensor_uid", "y_tensor_uid"})
        {
            if(node.contains(key) && node.at(key).is_number_integer())
            {
                uids.push_back(node.at(key).get<int64_t>());
            }
        }
    }
    return uids;
}

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

// Thrown only by validateRuntimePassByValueTensors(), distinct from the generic
// schema-conversion failures buildGraphBuffer() otherwise collapses to
// LoadError::INVALID_GRAPH_SCHEMA. Kept as its own type so callers can single
// this one contradiction out for a hard failure instead of a quiet skip — see
// BundleRegistration.hpp's classifyBundle().
class RuntimePassByValueInvariantError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Invariant: a tensor with is_runtime_pass_by_value=true must not also carry a baked
// value_type or value — if it does, both the CPU reference and the provider silently
// short-circuit to the baked value and the runtime path is never exercised. Checked here
// rather than in applyTensorPatches so it covers every path that reaches a graph buffer,
// not just the template-sweep path that patches tensors.
inline void validateRuntimePassByValueTensors(const nlohmann::json& graphJson)
{
    if(!graphJson.contains("tensors") || !graphJson.at("tensors").is_array())
    {
        return;
    }

    for(const auto& tensor : graphJson.at("tensors"))
    {
        if(tensor.value("is_runtime_pass_by_value", false)
           && (tensor.contains("value_type") || tensor.contains("value")))
        {
            const auto uid = tensor.value("uid", int64_t{-1});
            throw RuntimePassByValueInvariantError(
                "tensor uid " + std::to_string(uid)
                + " has is_runtime_pass_by_value=true but still carries value_type or value;"
                  " remove them from the tensor JSON, or from the sweep case's tensor_patches"
                  " 'remove' list or template");
        }
    }
}

// Validates the PBV invariant outside the try block below so its distinct
// RuntimePassByValueInvariantError propagates to the caller uncaught, instead
// of being collapsed into the generic false/INVALID_GRAPH_SCHEMA result used
// for ordinary schema-conversion failures.
inline bool buildGraphBuffer(const nlohmann::json& graphJson,
                             flatbuffers::DetachedBuffer& graphBuffer)
{
    validateRuntimePassByValueTensors(graphJson);

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
inline std::optional<LoadError> loadTensorDataIfPresent(IntegrationTestBundle& bundle,
                                                        const nlohmann::json& graphJson,
                                                        BlobPathFn&& blobPathForUid)
{
    const std::vector<int64_t> allUids = allTensorUids(graphJson);
    const std::set<int64_t> outputUidSet(bundle.outputTensorUids.begin(),
                                         bundle.outputTensorUids.end());

    std::vector<int64_t> inputUids;
    inputUids.reserve(allUids.size());
    for(const int64_t uid : allUids)
    {
        if(outputUidSet.count(uid) == 0)
        {
            inputUids.push_back(uid);
        }
    }

    const bool inputsPresent = !inputUids.empty() && blobsPresentFor(inputUids, blobPathForUid);
    const bool outputsPresent = !bundle.outputTensorUids.empty()
                                && blobsPresentFor(bundle.outputTensorUids, blobPathForUid);
    if(!inputsPresent)
    {
        return std::nullopt;
    }

    const auto& graph = *hipdnn_flatbuffers_sdk::data_objects::GetGraph(bundle.graphBuffer.data());
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        attrByUid;
    for(const auto* attributes : *graph.tensors())
    {
        attrByUid[attributes->uid()] = attributes;
    }

    const auto loadUids = [&](const std::vector<int64_t>& uids, TensorMap& into) {
        for(const int64_t uid : uids)
        {
            const auto it = attrByUid.find(uid);
            if(it == attrByUid.end())
            {
                continue;
            }
            into[uid] = hipdnn_test_sdk::utilities::tensorFromFileAndAttributes(blobPathForUid(uid),
                                                                                *it->second);
        }
    };

    try
    {
        TensorMap tensorMap;
        loadUids(inputUids, tensorMap);
        if(outputsPresent)
        {
            loadUids(bundle.outputTensorUids, tensorMap);
            bundle.hasGoldenOutputs = true;
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
    constexpr std::string_view PREFIX = "${case.";
    if(value.size() <= PREFIX.size() + 1
       || value.compare(0, PREFIX.size(), PREFIX.data(), PREFIX.size()) != 0 || value.back() != '}')
    {
        return std::nullopt;
    }

    return value.substr(PREFIX.size(), value.size() - PREFIX.size() - 1);
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

// Applies structural tensor patches declared in a sweep case's "tensor_patches" array.
// Each patch targets a tensor by uid and supports "set" (upsert fields) and "remove" (erase
// fields). Called after expandTemplateGraph so placeholders are already resolved, and before
// buildGraphBuffer so the flatbuffer has not yet been sealed.
inline void applyTensorPatches(nlohmann::json& expandedGraph, const nlohmann::json& caseJson)
{
    if(!caseJson.contains("tensor_patches") || !caseJson.at("tensor_patches").is_array())
    {
        return;
    }

    if(!expandedGraph.contains("tensors") || !expandedGraph.at("tensors").is_array())
    {
        return;
    }

    for(const auto& patch : caseJson.at("tensor_patches"))
    {
        if(!patch.is_object() || !patch.contains("uid") || !patch.at("uid").is_number_integer())
        {
            throw std::runtime_error("tensor_patch entry is missing an integer uid");
        }

        const auto targetUid = patch.at("uid").get<int64_t>();
        bool found = false;

        for(auto& tensor : expandedGraph.at("tensors"))
        {
            if(!tensor.contains("uid") || tensor.at("uid").get<int64_t>() != targetUid)
            {
                continue;
            }

            if(patch.contains("set") && patch.at("set").is_object())
            {
                for(const auto& [key, val] : patch.at("set").items())
                {
                    tensor[key] = val;
                }
            }

            if(patch.contains("remove") && patch.at("remove").is_array())
            {
                for(const auto& key : patch.at("remove"))
                {
                    if(key.is_string())
                    {
                        tensor.erase(key.get<std::string>());
                    }
                }
            }

            found = true;
            break;
        }

        if(!found)
        {
            throw std::runtime_error("tensor_patch uid " + std::to_string(targetUid)
                                     + " not found in expanded graph");
        }
    }
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

} // namespace detail

// Load a direct bundle from its graph .json path, classifying the outcome.
//
// This deliberately does NOT call test_sdk's loadGraphAndTensors(), whose
// all-or-nothing contract ("graph AND at least one tensor, or throw") conflicts
// with graph-only bundles being legitimate. Instead it composes the same
// primitives under this policy:
//
//   * graph .json not parseable            -> LoadError::MALFORMED_JSON
//   * parseable but not a valid graph      -> LoadError::INVALID_GRAPH_SCHEMA
//   * golden outputs present, no metadata  -> LoadError::MISSING_METADATA
//   * no golden outputs, no metadata       -> bundle with empty metadata
//   * valid graph, input blobs absent      -> bundle with tensors == nullopt
//   * present blob fails to load           -> LoadError::TENSOR_LOAD_FAILED
//   * inputs present, outputs absent       -> bundle verified against reference
//   * inputs and outputs present           -> bundle verified against golden data
//
// Inputs and outputs are loaded independently. Output uids come from the graph;
// every other declared tensor is treated as input. Every failure above is
// reported through the return value, except one: a
// RuntimePassByValueInvariantError from buildGraphBuffer() propagates
// uncaught, since callers (see BundleRegistration.hpp's classifyBundle())
// deliberately treat that one contradiction as a hard failure rather than a
// quiet skip.
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

    IntegrationTestBundle bundle;
    bundle.graphBuffer = std::move(graphBuffer);

    {
        auto allOutputUids = detail::extractOutputUidsFromJson(*graphJson);
        const auto wrapper = bundle.graphWrapper();
        const auto& tensorMap = wrapper.getTensorMap();
        for(const int64_t uid : allOutputUids)
        {
            auto it = tensorMap.find(uid);
            if(it == tensorMap.end() || !it->second->virtual_())
            {
                bundle.outputTensorUids.push_back(uid);
            }
        }
    }

    const auto blobPathForUid = [&](int64_t uid) { return detail::tensorBlobPath(jsonPath, uid); };
    const bool goldenOutputsPresent
        = !bundle.outputTensorUids.empty()
          && detail::blobsPresentFor(bundle.outputTensorUids, blobPathForUid);

    auto metadata = hipdnn_integration_tests::loadBundleMetadata(jsonPath);
    if(!metadata.has_value())
    {
        if(goldenOutputsPresent)
        {
            return LoadError::MISSING_METADATA;
        }
        metadata.emplace();
    }
    bundle.metadata = std::move(*metadata);

    if(const auto loadError = detail::loadTensorDataIfPresent(bundle, *graphJson, blobPathForUid);
       loadError.has_value())
    {
        return *loadError;
    }

    return bundle;
}

// Load either a direct bundle or one logical template-sweep case.
//
// Sweep cases parse graph.template.json plus sweep.json, locate the discovered
// case id, expand `${case...}` placeholders, load inline metadata, and resolve an
// optional golden directory. Sweep authoring errors are reported as
// INVALID_SWEEP_CASE; an expanded graph that still fails schema conversion is
// INVALID_GRAPH_SCHEMA. As with the direct-bundle overload, a
// RuntimePassByValueInvariantError from buildGraphBuffer() is the one
// exception that propagates uncaught rather than being folded into
// INVALID_GRAPH_SCHEMA.
inline LoadResult loadIntegrationTestBundle(const DiscoveredBundle& discovered)
{
    if(!discovered.isTemplateSweepCase())
    {
        return loadIntegrationTestBundle(discovered.jsonPath);
    }

    const auto templateJson = detail::parseJsonFile(discovered.sweep->templatePath);
    const auto sweepJson = detail::parseJsonFile(discovered.jsonPath);
    if(!templateJson.has_value() || !sweepJson.has_value())
    {
        return LoadError::MALFORMED_JSON;
    }

    const auto* caseJson = detail::findSweepCase(*sweepJson, discovered.sweep->caseId);
    if(caseJson == nullptr)
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    nlohmann::json expandedGraph;
    std::optional<std::filesystem::path> goldenDirectory;
    try
    {
        expandedGraph = detail::expandTemplateGraph(*templateJson, *caseJson, discovered);
        goldenDirectory = detail::resolveSweepGoldenDirectory(discovered.jsonPath, *caseJson);
        detail::applyTensorPatches(expandedGraph, *caseJson);
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

    // Mirror the direct-bundle path: extractOutputUidsFromJson handles flat-uid
    // ops (Reduction/ResampleFwd/CustomOp) that getOutputTensorUidsFromGraph throws on.
    {
        auto allOutputUids = detail::extractOutputUidsFromJson(expandedGraph);
        const auto wrapper = bundle.graphWrapper();
        const auto& tensorMap = wrapper.getTensorMap();
        for(const int64_t uid : allOutputUids)
        {
            auto it = tensorMap.find(uid);
            if(it == tensorMap.end() || !it->second->virtual_())
            {
                bundle.outputTensorUids.push_back(uid);
            }
        }
    }

    // Every sweep case must carry a metadata block. Metadata (arch lock,
    // ROCm/GPU version, seed, VRAM guard) is what validates golden data and
    // anchors the case, so an absent block is MISSING_METADATA and a present
    // but unparseable block is an authoring error (INVALID_SWEEP_CASE).
    if(!caseJson->contains("metadata") || caseJson->at("metadata").is_null())
    {
        return LoadError::MISSING_METADATA;
    }
    auto metadata = hipdnn_integration_tests::parseBundleMetadataJson(
        caseJson->at("metadata"), discovered.diagnosticPath().string());
    if(!metadata.has_value())
    {
        return LoadError::INVALID_SWEEP_CASE;
    }
    bundle.metadata = std::move(*metadata);

    if(goldenDirectory.has_value())
    {
        const auto blobPathForUid = [&](int64_t uid) {
            return *goldenDirectory / ("tensor" + std::to_string(uid) + ".bin");
        };
        if(const auto loadError
           = detail::loadTensorDataIfPresent(bundle, expandedGraph, blobPathForUid);
           loadError.has_value())
        {
            return *loadError;
        }
    }

    return bundle;
}

} // namespace hipdnn_integration_tests::bundle
