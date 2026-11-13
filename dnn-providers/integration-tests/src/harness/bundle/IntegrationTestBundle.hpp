// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLengthUids.hpp>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/SweepManifestCache.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/TensorAliases.hpp"

namespace hipdnn_integration_tests::bundle
{

// Tensors keyed by tensor UID. Inputs carry their data. Outputs carry expected golden
// values only when output blobs are present; otherwise the harness verifies outputs
// against a reference executor.
using TensorMap = std::unordered_map<int64_t, std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// Where a bundle's tensor blobs sit on disk. Nothing is read until a test asks for it.
struct TensorBlobs
{
    std::vector<int64_t> inputUids; // every declared tensor that is not an output
    std::vector<int64_t> outputUids; // golden outputs; empty unless hasGoldenOutputs
    std::function<std::filesystem::path(int64_t)> pathForUid;
};

// One test's worth of bundle data located on disk.
//
//   graphBuffer      — the parsed graph, as a flatbuffer. Always present in a
//                      loaded bundle; the engine deserializes it (from_binary)
//                      and the harness walks it (GraphWrapper) for dtypes and
//                      tolerances. A bundle that cannot produce a graph is a
//                      LoadError, not a bundle.
//   metadata         — .meta.json contents for direct bundles, or inline sweep
//                      metadata for template-sweep cases. Metadata is mandatory
//                      only when golden output blobs are present; graph-only and
//                      reference-verified bundles without a .meta.json default to
//                      empty metadata. Present-but-malformed metadata throws
//                      BundleMetadataError rather than loading.
//   outputTensorUids — UIDs of the graph's output tensors, derived from the
//                      graph. Always available, even for graph-only bundles, so
//                      the harness knows which tensors to compare or allocate.
//   blobs            — where the tensor data is, present when every input blob
//                      exists. If present and hasGoldenOutputs is false, it covers
//                      inputs only and outputs are reference-verified. Absent means
//                      the bundle is graph-only; the harness may fill inputs,
//                      otherwise it skips the case.
//   hasGoldenOutputs — true iff every output tensor's .bin blob is present. When
//                      false, engine output must be checked against a reference
//                      executor instead of golden data.
//
// A registered bundle lives for the whole process, so it holds no tensor data itself:
// reading every golden blob at registration kept all of it in memory for the run.
// Each test reads what it needs with loadTensors(), or under --benchmark-reference
// with loadInputs() and, when golden outputs verify it, loadGoldenOutputs(), and
// owns the result.
struct IntegrationTestBundle
{
    flatbuffers::DetachedBuffer graphBuffer;
    hipdnn_integration_tests::BundleMetadata metadata;
    std::vector<int64_t> outputTensorUids;
    std::optional<TensorBlobs> blobs;
    bool hasGoldenOutputs = false;

    // View over the graph flatbuffer, valid as long as this bundle lives.
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper{graphBuffer.data(),
                                                                          graphBuffer.size()};
    }

    // Reads the bundle's blobs into fresh tensors: the inputs, plus the golden outputs
    // when it has them. Empty for a graph-only bundle. The caller owns the result, so
    // the memory is freed with it. Throws if a blob is unreadable, the wrong size, or
    // of a type that has no tensor.
    TensorMap loadTensors() const
    {
        auto tensors = loadInputs();
        loadGoldenOutputs(tensors);
        return tensors;
    }

    // Reads only the input blobs, as loadTensors() does. Under --benchmark-reference
    // the golden outputs are read separately, inside the reference timing.
    TensorMap loadInputs() const
    {
        TensorMap tensors;
        if(blobs.has_value())
        {
            loadTensorUids(tensors, blobs->inputUids);
        }
        return tensors;
    }

    // Adds the golden output blobs to `tensors`, which must already hold the inputs:
    // a ragged output takes its offset tensor from them. Does nothing for a graph-only
    // bundle. Throws as loadTensors() does.
    void loadGoldenOutputs(TensorMap& tensors) const
    {
        if(blobs.has_value())
        {
            loadTensorUids(tensors, blobs->outputUids);
        }
    }

private:
    // Callers check that the bundle has blobs.
    void loadTensorUids(TensorMap& tensors, const std::vector<int64_t>& uids) const
    {
        const auto& graph = *hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer.data());
        std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
            attrByUid;
        for(const auto* attributes : *graph.tensors())
        {
            attrByUid[attributes->uid()] = attributes;
        }

        // Ragged tensors are constructed from their offset tensor and, when the node
        // declares one, their seq_len tensor, so they load last. Both are always
        // inputs, so an output's offset and seq_len are already loaded.
        const auto lengthUids = hipdnn_test_sdk::utilities::raggedLengthUids(graph);
        const auto loadUids = [&] {
            std::vector<int64_t> raggedUids;
            for(const int64_t uid : uids)
            {
                const auto it = attrByUid.find(uid);
                if(it == attrByUid.end())
                {
                    continue;
                }
                if(it->second->ragged_offset_tensor_uid().has_value())
                {
                    raggedUids.push_back(uid);
                    continue;
                }
                tensors[uid] = hipdnn_test_sdk::utilities::tensorFromFileAndAttributes(
                    blobs->pathForUid(uid), *it->second);
            }
            for(const int64_t uid : raggedUids)
            {
                const auto* attributes = attrByUid.at(uid);
                const int64_t offsetUid = attributes->ragged_offset_tensor_uid().value();
                const auto offsetIt = tensors.find(offsetUid);
                if(offsetIt == tensors.end())
                {
                    throw std::runtime_error("ragged tensor " + std::to_string(uid)
                                             + " references missing offset tensor "
                                             + std::to_string(offsetUid));
                }
                const auto lengths
                    = hipdnn_test_sdk::utilities::findRaggedLengths(lengthUids, tensors, uid);
                tensors[uid] = hipdnn_test_sdk::utilities::raggedTensorFromFileAndAttributes(
                    blobs->pathForUid(uid), *attributes, offsetIt->second, lengths);
            }
        };

        loadUids();
    }
};

// Why a load did NOT produce a bundle. These are authoring failures in the
// bundle or sweep case. A valid graph-only bundle is still a loaded bundle and is
// skipped later only if the harness cannot fill inputs.
//
// UNVALIDATABLE_GOLDEN_DATA is deliberately separate from MISSING_METADATA: it is
// the one load failure that gets louder as the checkout gets more complete. Golden
// blobs are on disk but nothing describes how they were produced, so the bundle
// would be dropped at exactly the moment its data became available. Callers
// (classifyBundle()) turn that one red; the rest stay log-and-skip.
//
// Metadata that is present but malformed is not a LoadError at all: the loader
// lets BundleMetadataError propagate so its detail reaches the failure message.
enum class LoadError
{
    MALFORMED_JSON, // graph/template/sweep JSON is unreadable or syntactically invalid
    INVALID_GRAPH_SCHEMA, // expanded graph JSON cannot build a valid graph flatbuffer
    MISSING_METADATA, // metadata absent, and no golden data that would need validating
    UNVALIDATABLE_GOLDEN_DATA, // golden blobs present but their metadata is absent
    INVALID_SWEEP_CASE // sweep case id, placeholders, or golden path are invalid
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
        return "metadata is missing";
    case LoadError::UNVALIDATABLE_GOLDEN_DATA:
        return "golden tensor .bin files are present but their metadata is missing, "
               "so the data cannot be validated";
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

template <typename BlobPathFn>
inline bool anyBlobPresentFor(const std::vector<int64_t>& uids, BlobPathFn&& blobPathForUid)
{
    for(const int64_t uid : uids)
    {
        if(std::filesystem::exists(blobPathForUid(uid)))
        {
            return true;
        }
    }
    return false;
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
// ResampleFwdAttributes, ResampleBwdAttributes, and CustomOpAttributes emit flat keys instead,
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
        // ResampleFwd ("y_tensor_uid"), ResampleBwd ("dx_tensor_uid")
        for(const auto& key : {"out_tensor_uid", "y_tensor_uid", "dx_tensor_uid"})
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

// Records where a bundle's tensor blobs are, without reading them. Inputs are every
// declared tensor that is not an output. The bundle carries blobs only when all inputs
// are present, and golden outputs only when all outputs are too.
//
// `blobPathForUid` is stored, so it must own what it captures.
inline void describeTensorBlobs(IntegrationTestBundle& bundle,
                                const nlohmann::json& graphJson,
                                std::function<std::filesystem::path(int64_t)> blobPathForUid)
{
    const std::vector<int64_t> allUids = allTensorUids(graphJson);
    const std::set<int64_t> outputUidSet(bundle.outputTensorUids.begin(),
                                         bundle.outputTensorUids.end());

    TensorBlobs blobs;
    blobs.inputUids.reserve(allUids.size());
    for(const int64_t uid : allUids)
    {
        if(outputUidSet.count(uid) == 0)
        {
            blobs.inputUids.push_back(uid);
        }
    }

    const bool inputsPresent
        = !blobs.inputUids.empty() && blobsPresentFor(blobs.inputUids, blobPathForUid);
    if(!inputsPresent)
    {
        return;
    }

    if(!bundle.outputTensorUids.empty() && blobsPresentFor(bundle.outputTensorUids, blobPathForUid))
    {
        blobs.outputUids = bundle.outputTensorUids;
        bundle.hasGoldenOutputs = true;
    }

    blobs.pathForUid = std::move(blobPathForUid);
    bundle.blobs = std::move(blobs);
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

// The error for a bad patch field of a sweep case. The message names the case, so a failure
// in a sweep of thousands of cases can be found, and then the cause.
inline std::runtime_error sweepPatchError(const nlohmann::json& caseJson, const std::string& cause)
{
    std::string id = "<no id>";
    if(caseJson.is_object() && caseJson.contains("id") && caseJson.at("id").is_string())
    {
        id = caseJson.at("id").get<std::string>();
    }
    return std::runtime_error("sweep case '" + id + "': " + cause);
}

inline bool graphHasTensorUid(const nlohmann::json& tensors, int64_t uid)
{
    return std::any_of(tensors.begin(), tensors.end(), [uid](const nlohmann::json& tensor) {
        return tensor.is_object() && tensor.contains("uid") && tensor.at("uid").is_number_integer()
               && tensor.at("uid").get<int64_t>() == uid;
    });
}

// Appends the literal tensor objects of a sweep case's "add_tensors" array to the expanded
// graph's tensors[]. The entries are not template-expanded: they are final JSON. A uid that
// the graph already has (from the template or from an earlier entry) is an error, and so is
// an entry that is not an object with an integer uid. Runs before node_patches and
// tensor_patches, so both can refer to the added uids.
inline void applyAddTensors(nlohmann::json& expandedGraph, const nlohmann::json& caseJson)
{
    if(!caseJson.contains("add_tensors"))
    {
        return;
    }

    const auto& added = caseJson.at("add_tensors");
    if(!added.is_array())
    {
        throw sweepPatchError(caseJson, "add_tensors must be an array");
    }

    if(added.empty())
    {
        return;
    }

    if(!expandedGraph.contains("tensors") || !expandedGraph.at("tensors").is_array())
    {
        throw sweepPatchError(caseJson, "add_tensors needs a tensors array in the expanded graph");
    }

    auto& tensors = expandedGraph.at("tensors");
    for(size_t index = 0; index < added.size(); ++index)
    {
        const auto& tensor = added.at(index);
        const auto where = "add_tensors[" + std::to_string(index) + "]";
        if(!tensor.is_object() || !tensor.contains("uid") || !tensor.at("uid").is_number_integer())
        {
            throw sweepPatchError(caseJson, where + " must be an object with an integer uid");
        }

        const auto uid = tensor.at("uid").get<int64_t>();
        if(graphHasTensorUid(tensors, uid))
        {
            throw sweepPatchError(
                caseJson, where + ": uid " + std::to_string(uid) + " already exists in the graph");
        }

        tensors.push_back(tensor);
    }
}

// The node a node_patches entry addresses, and how to name it in an error message.
struct PatchedNode
{
    nlohmann::json* node = nullptr;
    std::string label;
};

// Finds a node by its "name" (which must be unique in nodes[]) or by its index in nodes[].
inline PatchedNode findPatchedNode(nlohmann::json& expandedGraph,
                                   const nlohmann::json& caseJson,
                                   const nlohmann::json& selector,
                                   const std::string& where)
{
    if(!expandedGraph.contains("nodes") || !expandedGraph.at("nodes").is_array())
    {
        throw sweepPatchError(caseJson, where + ": the expanded graph has no nodes array");
    }

    auto& nodes = expandedGraph.at("nodes");
    if(selector.is_number_integer())
    {
        const auto index = selector.get<int64_t>();
        if(index < 0 || static_cast<size_t>(index) >= nodes.size())
        {
            throw sweepPatchError(caseJson,
                                  where + ": node index " + std::to_string(index)
                                      + " is out of range (nodes[] has "
                                      + std::to_string(nodes.size()) + " entries)");
        }
        return {&nodes.at(static_cast<size_t>(index)), "node index " + std::to_string(index)};
    }

    if(!selector.is_string())
    {
        throw sweepPatchError(caseJson, where + ": node must be a node name or an index");
    }

    const auto name = selector.get<std::string>();
    nlohmann::json* match = nullptr;
    size_t matches = 0;
    for(auto& node : nodes)
    {
        if(node.is_object() && node.contains("name") && node.at("name").is_string()
           && node.at("name").get<std::string>() == name)
        {
            match = &node;
            ++matches;
        }
    }

    if(matches == 0)
    {
        throw sweepPatchError(caseJson,
                              where + ": node '" + name + "' not found in expanded graph");
    }
    if(matches > 1)
    {
        throw sweepPatchError(caseJson,
                              where + ": node name '" + name + "' is ambiguous ("
                                  + std::to_string(matches)
                                  + " nodes have it); address it by index");
    }
    return {match, "node '" + name + "'"};
}

// Applies the entries of a sweep case's "node_patches" array to the expanded graph. An entry
// is {"node": <name|index>, "section": <section>, "set": {...}, "remove": [...]}, where the
// section is the node object that holds the keys: "attributes" or "parameters" for the flags
// and scalars, "inputs" or "outputs" for the tensor uids. The entry edits that one object:
//
//   * "set" overwrites keys that the section already has. A key it lacks is an error rather
//     than an insert, because the JSON-to-flatbuffer converters ignore keys they do not know,
//     so a misspelled key would silently do nothing. Templates list every key (null
//     included), so every legal key is present.
//   * "remove" then erases keys. Removing a key the section lacks is an error.
//
// A node or a section that does not exist is an error and is never created. Runs after
// applyAddTensors, so an entry may wire in an added uid, and before applyTensorPatches.
inline void applyNodePatches(nlohmann::json& expandedGraph, const nlohmann::json& caseJson)
{
    if(!caseJson.contains("node_patches"))
    {
        return;
    }

    const auto& patches = caseJson.at("node_patches");
    if(!patches.is_array())
    {
        throw sweepPatchError(caseJson, "node_patches must be an array");
    }

    constexpr std::array<std::string_view, 4> SECTIONS{
        "attributes", "parameters", "inputs", "outputs"};
    constexpr std::array<std::string_view, 4> PATCH_KEYS{"node", "section", "set", "remove"};

    for(size_t index = 0; index < patches.size(); ++index)
    {
        const auto& patch = patches.at(index);
        const auto where = "node_patches[" + std::to_string(index) + "]";
        if(!patch.is_object() || !patch.contains("node") || !patch.contains("section"))
        {
            throw sweepPatchError(caseJson, where + " must be an object with node and section");
        }

        for(const auto& item : patch.items())
        {
            if(std::find(PATCH_KEYS.begin(), PATCH_KEYS.end(), item.key()) == PATCH_KEYS.end())
            {
                throw sweepPatchError(caseJson, where + ": unknown key '" + item.key() + "'");
            }
        }

        const auto& sectionName = patch.at("section");
        if(!sectionName.is_string()
           || std::find(SECTIONS.begin(), SECTIONS.end(), sectionName.get<std::string>())
                  == SECTIONS.end())
        {
            throw sweepPatchError(caseJson,
                                  where + ": unknown section " + sectionName.dump()
                                      + " (expected attributes, parameters, inputs or outputs)");
        }

        const bool hasSet = patch.contains("set");
        const bool hasRemove = patch.contains("remove");
        if(!hasSet && !hasRemove)
        {
            throw sweepPatchError(caseJson, where + " has neither set nor remove");
        }
        if(hasSet && !patch.at("set").is_object())
        {
            throw sweepPatchError(caseJson, where + ": set must be an object");
        }
        if(hasRemove && !patch.at("remove").is_array())
        {
            throw sweepPatchError(caseJson, where + ": remove must be an array of key names");
        }

        const auto target = findPatchedNode(expandedGraph, caseJson, patch.at("node"), where);
        const auto section = sectionName.get<std::string>();
        if(!target.node->is_object() || !target.node->contains(section)
           || !target.node->at(section).is_object())
        {
            std::string message = where;
            message.append(": ")
                .append(target.label)
                .append(" has no '")
                .append(section)
                .append("' section");
            throw sweepPatchError(caseJson, message);
        }

        auto& keys = target.node->at(section);
        if(hasSet)
        {
            for(const auto& [key, value] : patch.at("set").items())
            {
                if(!keys.contains(key))
                {
                    std::string message = where;
                    message.append(": set key '")
                        .append(key)
                        .append("' is not in the '")
                        .append(section)
                        .append("' section of ")
                        .append(target.label)
                        .append(" (set only overwrites existing keys)");
                    throw sweepPatchError(caseJson, message);
                }
                keys[key] = value;
            }
        }

        if(hasRemove)
        {
            for(const auto& key : patch.at("remove"))
            {
                if(!key.is_string() || !keys.contains(key.get<std::string>()))
                {
                    std::string message = where;
                    message.append(": remove key ")
                        .append(key.dump())
                        .append(" is not in the '")
                        .append(section)
                        .append("' section of ")
                        .append(target.label);
                    throw sweepPatchError(caseJson, message);
                }
                keys.erase(key.get<std::string>());
            }
        }
    }
}

// Applies structural tensor patches declared in a sweep case's "tensor_patches" array.
// Each patch targets a tensor by uid and supports "set" (upsert fields) and "remove" (erase
// fields). Called after expandTemplateGraph, applyAddTensors and applyNodePatches, so
// placeholders are already resolved and an added uid can be patched, and before
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
            throw sweepPatchError(caseJson, "tensor_patch entry is missing an integer uid");
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
            throw sweepPatchError(caseJson,
                                  "tensor_patch uid " + std::to_string(targetUid)
                                      + " not found in expanded graph");
        }
    }
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

// The rules that tie a bundle's metadata flags to the golden data it declares.
// "Declared" is a fact about the checked-in files, not about the checkout: the
// `.tensors.dvc` pointer (or a sweep case's `golden.path`) is in git, the blobs it
// names are only there after `dvc pull`. Reading it from the pointer keeps these
// rules the same on a machine that has not pulled.
//
//   * Both references disabled (not on an expected rejection, which ignores them):
//     nothing but golden data could ever verify the
//     bundle, so it must declare some.
//   * An expected-rejection bundle is never executed, so golden data beside it is
//     meaningless, and would also pull the bundle into the golden-data binary.
//
// `where` names the bundle or sweep case in the thrown message.
inline void checkGoldenDeclarationRules(const hipdnn_integration_tests::BundleMetadata& metadata,
                                        bool goldenDeclared,
                                        const std::string& where)
{
    if(metadata.expectedRejection)
    {
        // The reference flags are ignored on an expected rejection (nothing is ever
        // compared), so the both-off rule below must not apply to it.
        if(goldenDeclared)
        {
            throw hipdnn_integration_tests::BundleMetadataError(
                where
                + " is an expected_rejection bundle but declares golden data; it is never"
                  " executed, so there is nothing to compare against");
        }
        return;
    }
    if(metadata.noCpuReference && metadata.noGpuReference && !goldenDeclared)
    {
        throw hipdnn_integration_tests::BundleMetadataError(
            where
            + " disables both references (no_cpu_reference and no_gpu_reference) but declares no"
              " golden data; add golden data ({Name}.tensors.dvc or output blobs, or the sweep"
              " case's golden.path) or enable a reference");
    }
}

// The rules that tie a bundle's `aliases` to its graph: checkTensorAliases() against the
// graph's tensors and its input fill, and no tensor data on disk (golden data declared
// or input blobs present), because blobs store each tensor apart. `where` names the
// bundle or sweep case in the thrown message.
inline void checkAliasRules(const hipdnn_integration_tests::BundleMetadata& metadata,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph,
                            const std::vector<int64_t>& outputTensorUids,
                            bool tensorDataOnDisk,
                            const std::string& where)
{
    if(metadata.aliases.empty())
    {
        return;
    }
    try
    {
        hipdnn_integration_tests::checkTensorAliases(
            metadata.aliases,
            graph.getTensorMap(),
            outputTensorUids,
            hipdnn_integration_tests::fillDerivesInputs(graph.getGraph()));
    }
    catch(const std::invalid_argument& e)
    {
        throw hipdnn_integration_tests::BundleMetadataError(where + ": " + e.what());
    }
    if(tensorDataOnDisk)
    {
        throw hipdnn_integration_tests::BundleMetadataError(
            where + ": "
            + hipdnn_integration_tests::tensor_alias_detail::describe(metadata.aliases.front())
            + ": the bundle has golden data or tensor blobs, which store each tensor apart");
    }
}

// The rules that tie a bundle's `start_offsets` to its graph. Each tensor is a graph
// tensor that holds its own data: not virtual, not passed by value at run time, not a
// constant, not the view of an alias and not of a sub-byte type. A tensor that declares
// an alignment greater than 0 must have a start_offset that is a multiple of it. Throws
// BundleMetadataError naming `where`, the tensor and the rule.
inline void
    checkStartOffsetRules(const hipdnn_integration_tests::BundleMetadata& metadata,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph,
                          const std::string& where)
{
    using hipdnn_integration_tests::BundleMetadataError;
    using hipdnn_integration_tests::bundle_metadata_detail::describeStartOffset;

    if(metadata.startOffsets.empty())
    {
        return;
    }
    const auto& tensors = graph.getTensorMap();
    const auto error = [&](int64_t uid, const std::string& rule) {
        std::string message = where;
        message.append(": ").append(describeStartOffset(uid)).append(": ").append(rule);
        return message;
    };

    for(const auto& entry : metadata.startOffsets)
    {
        const auto it = tensors.find(entry.uid);
        if(it == tensors.end() || it->second == nullptr)
        {
            throw BundleMetadataError(error(entry.uid, "the tensor is not a tensor of the graph"));
        }
        const auto& tensor = *it->second;
        if(tensor.virtual_())
        {
            throw BundleMetadataError(error(entry.uid, "the tensor is virtual"));
        }
        if(tensor.is_runtime_pass_by_value())
        {
            throw BundleMetadataError(
                error(entry.uid, "the tensor is passed by value at run time"));
        }
        if(tensor.value_type() != hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
        {
            throw BundleMetadataError(error(entry.uid, "the tensor has a constant value"));
        }
        for(const auto& alias : metadata.aliases)
        {
            if(alias.uid == entry.uid)
            {
                throw BundleMetadataError(
                    error(entry.uid,
                          "the tensor is "
                              + hipdnn_integration_tests::tensor_alias_detail::describe(alias)
                              + "; a view moves with its owner"));
            }
        }
        if(hipdnn_integration_tests::tensor_alias_detail::isSubByte(tensor.data_type()))
        {
            throw BundleMetadataError(error(entry.uid,
                                            std::string("the tensor has the sub-byte data type ")
                                                + EnumNameDataType(tensor.data_type())));
        }
        const int64_t alignment = tensor.alignment();
        if(alignment > 0 && entry.startOffset % alignment != 0)
        {
            throw BundleMetadataError(error(entry.uid,
                                            "start_offset " + std::to_string(entry.startOffset)
                                                + " is not a multiple of the tensor's alignment "
                                                + std::to_string(alignment)));
        }
    }
}

// The rules that tie a bundle's `live_rows` to its graph. Each output is a compared
// output of a floating-point type that is not ragged. A `lengths` entry has an axis from
// 1 to the output's rank minus 1, and names a graph input of type INT32 or INT64 that
// holds its own data and has one element per batch of the output. A `like` entry names
// a ragged graph tensor that is not virtual and has as many batches as the output.
// Throws BundleMetadataError naming `where`, the output and the rule.
inline void
    checkLiveRowRules(const hipdnn_integration_tests::BundleMetadata& metadata,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph,
                      const std::vector<int64_t>& outputTensorUids,
                      const std::string& where)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_integration_tests::BundleMetadataError;
    using hipdnn_integration_tests::bundle_metadata_detail::describeLiveRows;

    if(metadata.liveRows.empty())
    {
        return;
    }
    const auto& tensors = graph.getTensorMap();
    const std::unordered_set<int64_t> outputs(outputTensorUids.begin(), outputTensorUids.end());
    const auto error = [&](int64_t uid, const std::string& rule) {
        std::string message = where;
        message.append(": ").append(describeLiveRows(uid)).append(": ").append(rule);
        return message;
    };
    const auto find = [&](int64_t uid) {
        const auto it = tensors.find(uid);
        return it == tensors.end() ? nullptr : it->second;
    };
    const auto dimsOf = [](const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor) {
        std::vector<int64_t> dims;
        if(tensor.dims() != nullptr)
        {
            dims.assign(tensor.dims()->begin(), tensor.dims()->end());
        }
        return dims;
    };
    const auto isFloatingPoint = [](DataType type) {
        return type == DataType::FLOAT || type == DataType::HALF || type == DataType::BFLOAT16
               || type == DataType::DOUBLE || type == DataType::FP8_E4M3
               || type == DataType::FP8_E5M2 || type == DataType::FP8_E4M3_FNUZ
               || type == DataType::FP8_E5M2_FNUZ;
    };

    for(const auto& entry : metadata.liveRows)
    {
        const auto* output = find(entry.uid);
        if(output == nullptr)
        {
            throw BundleMetadataError(error(entry.uid, "the output is not a tensor of the graph"));
        }
        if(output->virtual_() || outputs.count(entry.uid) == 0)
        {
            throw BundleMetadataError(error(entry.uid, "the tensor is not a compared output"));
        }
        if(output->ragged_offset_tensor_uid().has_value())
        {
            throw BundleMetadataError(
                error(entry.uid, "the output is ragged; a ragged output has its own live rows"));
        }
        if(!isFloatingPoint(output->data_type()))
        {
            throw BundleMetadataError(error(entry.uid,
                                            std::string("the output has the data type ")
                                                + EnumNameDataType(output->data_type())
                                                + ", which is not a floating-point type"));
        }
        const auto outputDims = dimsOf(*output);
        if(outputDims.empty())
        {
            throw BundleMetadataError(error(entry.uid, "the output has no dims"));
        }

        if(entry.likeUid.has_value())
        {
            const int64_t likeUid = *entry.likeUid;
            const auto* like = find(likeUid);
            if(like == nullptr)
            {
                throw BundleMetadataError(error(entry.uid,
                                                "like tensor " + std::to_string(likeUid)
                                                    + " is not a tensor of the graph"));
            }
            if(like->virtual_())
            {
                throw BundleMetadataError(
                    error(entry.uid, "like tensor " + std::to_string(likeUid) + " is virtual"));
            }
            if(!like->ragged_offset_tensor_uid().has_value())
            {
                throw BundleMetadataError(
                    error(entry.uid, "like tensor " + std::to_string(likeUid) + " is not ragged"));
            }
            const auto likeDims = dimsOf(*like);
            if(likeDims.empty() || likeDims[0] != outputDims[0])
            {
                throw BundleMetadataError(
                    error(entry.uid,
                          "like tensor " + std::to_string(likeUid)
                              + " does not have the same dims[0] as the output"));
            }
            continue;
        }

        const int64_t axis = entry.axis.value_or(0);
        const auto rank = static_cast<int64_t>(outputDims.size());
        if(axis < 1 || axis >= rank)
        {
            throw BundleMetadataError(error(entry.uid,
                                            "axis " + std::to_string(axis)
                                                + " is not in the range 1 to "
                                                + std::to_string(rank - 1)));
        }
        const int64_t lengthsUid = entry.lengthsUid.value_or(0);
        const std::string lengthsName = "lengths tensor " + std::to_string(lengthsUid);
        const auto* lengths = find(lengthsUid);
        if(lengths == nullptr)
        {
            throw BundleMetadataError(
                error(entry.uid, lengthsName + " is not a tensor of the graph"));
        }
        if(lengths->virtual_() || outputs.count(lengthsUid) != 0)
        {
            throw BundleMetadataError(error(
                entry.uid, lengthsName + " is not a graph input (it is virtual or an output)"));
        }
        if(lengths->is_runtime_pass_by_value())
        {
            throw BundleMetadataError(
                error(entry.uid, lengthsName + " is passed by value at run time"));
        }
        for(const auto& alias : metadata.aliases)
        {
            if(alias.uid == lengthsUid)
            {
                throw BundleMetadataError(
                    error(entry.uid,
                          lengthsName + " is "
                              + hipdnn_integration_tests::tensor_alias_detail::describe(alias)));
            }
        }
        if(lengths->data_type() != DataType::INT32 && lengths->data_type() != DataType::INT64)
        {
            throw BundleMetadataError(error(entry.uid,
                                            lengthsName + " has the data type "
                                                + EnumNameDataType(lengths->data_type())
                                                + ", not INT32 or INT64"));
        }
        int64_t elementCount = 1;
        for(const int64_t dim : dimsOf(*lengths))
        {
            elementCount *= dim;
        }
        if(elementCount != outputDims[0])
        {
            throw BundleMetadataError(error(
                entry.uid,
                lengthsName + " has " + std::to_string(elementCount) + " elements; the output has "
                    + std::to_string(outputDims[0]) + " batches (dims[0])"));
        }
    }
}

// Runs `step` and rethrows any exception that is not a BundleMetadataError as a
// BundleMetadataError that names `where`.
template <typename Step>
inline decltype(auto) rethrowAsBundleMetadataError(const std::string& where, Step&& step)
{
    try
    {
        return std::forward<Step>(step)();
    }
    catch(const hipdnn_integration_tests::BundleMetadataError&)
    {
        throw;
    }
    catch(const std::exception& e)
    {
        std::string message = where;
        message += ": ";
        message += e.what();
        throw hipdnn_integration_tests::BundleMetadataError(message);
    }
}

// The sweep-level `version` of the sweep at `sweepPath`: 1 when the key is absent, else
// the integer 1 or 2. `where` names the sweep case in the thrown message.
inline int checkSweepVersion(const nlohmann::json* version,
                             const std::filesystem::path& sweepPath,
                             const std::string& where)
{
    if(version == nullptr)
    {
        return 1;
    }
    if(version->is_number_integer() && (*version == 1 || *version == 2))
    {
        return version->get<int>();
    }
    std::string message = where;
    message += ": sweep version ";
    message += version->dump();
    message += " in ";
    message += sweepPath.string();
    message += " is not 1 or 2";
    throw hipdnn_integration_tests::BundleMetadataError(message);
}

inline std::string formatUidList(const std::vector<int64_t>& uids)
{
    std::string text = "[";
    for(size_t i = 0; i < uids.size(); ++i)
    {
        if(i != 0)
        {
            text += ", ";
        }
        text += std::to_string(uids[i]);
    }
    text += ']';
    return text;
}

// A node port is a key that ends in "_tensor_uid", directly under the "inputs" or
// "outputs" object of a node.
inline bool isNodePortKey(const std::string& key)
{
    constexpr std::string_view SUFFIX = "_tensor_uid";
    return key.size() >= SUFFIX.size()
           && key.compare(key.size() - SUFFIX.size(), SUFFIX.size(), SUFFIX.data(), SUFFIX.size())
                  == 0;
}

// The uid a JSON value holds: an integer, or a float with an integral value. A float
// such as 3.0 converts to the same uid when the graph is built.
inline std::optional<int64_t> uidOf(const nlohmann::json& value)
{
    if(value.is_number_integer())
    {
        return value.get<int64_t>();
    }
    if(!value.is_number_float())
    {
        return std::nullopt;
    }
    constexpr double INT64_BOUND = 9223372036854775808.0;
    const auto number = value.get<double>();
    if(std::trunc(number) != number || number < -INT64_BOUND || number >= INT64_BOUND)
    {
        return std::nullopt;
    }
    return static_cast<int64_t>(number);
}

inline bool holdsUid(const nlohmann::json& value, const std::unordered_set<int64_t>& uids)
{
    const auto uid = uidOf(value);
    return uid.has_value() && uids.count(*uid) != 0;
}

// Removes the tensors in `uids` from `graphJson` and sets each node port that holds one
// of them to null. Nothing else changes, and no uid is renumbered.
inline nlohmann::json pruneAbsentTensors(nlohmann::json graphJson, const std::vector<int64_t>& uids)
{
    const std::unordered_set<int64_t> absent(uids.begin(), uids.end());

    if(graphJson.contains("tensors") && graphJson.at("tensors").is_array())
    {
        auto kept = nlohmann::json::array();
        for(auto& tensor : graphJson.at("tensors"))
        {
            if(!tensor.is_object() || !tensor.contains("uid")
               || !holdsUid(tensor.at("uid"), absent))
            {
                kept.push_back(std::move(tensor));
            }
        }
        graphJson["tensors"] = std::move(kept);
    }

    if(graphJson.contains("nodes") && graphJson.at("nodes").is_array())
    {
        for(auto& node : graphJson.at("nodes"))
        {
            for(const char* side : {"inputs", "outputs"})
            {
                if(!node.is_object() || !node.contains(side) || !node.at(side).is_object())
                {
                    continue;
                }
                auto& ports = node.at(side);
                for(auto it = ports.begin(); it != ports.end(); ++it)
                {
                    if(isNodePortKey(it.key()) && holdsUid(it.value(), absent))
                    {
                        it.value() = nullptr;
                    }
                }
            }
        }
    }

    return graphJson;
}

// Appends to `found` the path of each integer below `json` that equals `uid` and sits
// under a key whose name holds "uid", directly or through arrays. `path` names `json`;
// `underUidKey` says whether `json` itself sits under such a key.
inline void findUidReferences(const nlohmann::json& json,
                              int64_t uid,
                              bool underUidKey,
                              const std::string& path,
                              std::vector<std::string>& found)
{
    if(json.is_object())
    {
        for(auto it = json.begin(); it != json.end(); ++it)
        {
            std::string childPath = path;
            childPath += '.';
            childPath += it.key();
            findUidReferences(
                it.value(), uid, it.key().find("uid") != std::string::npos, childPath, found);
        }
        return;
    }
    if(json.is_array())
    {
        for(size_t i = 0; i < json.size(); ++i)
        {
            std::string childPath = path;
            childPath += '[';
            childPath += std::to_string(i);
            childPath += ']';
            findUidReferences(json.at(i), uid, underUidKey, childPath, found);
        }
        return;
    }
    if(underUidKey && uidOf(json) == uid)
    {
        found.push_back(path);
    }
}

// Where the nodes of a template hold one uid: the nodes that hold it in a node port, and
// the path of each other place.
struct NodeUidUse
{
    std::set<size_t> portNodes;
    std::vector<std::string> otherPaths;
};

inline NodeUidUse findNodeUidUse(const nlohmann::json& templateJson, int64_t uid)
{
    NodeUidUse use;
    if(!templateJson.contains("nodes") || !templateJson.at("nodes").is_array())
    {
        return use;
    }

    const auto& nodes = templateJson.at("nodes");
    for(size_t index = 0; index < nodes.size(); ++index)
    {
        const auto& node = nodes.at(index);
        std::string nodePath = "nodes[";
        nodePath += std::to_string(index);
        nodePath += ']';
        if(!node.is_object())
        {
            findUidReferences(node, uid, false, nodePath, use.otherPaths);
            continue;
        }
        for(auto field = node.begin(); field != node.end(); ++field)
        {
            std::string fieldPath = nodePath;
            fieldPath += '.';
            fieldPath += field.key();
            const bool portSide = (field.key() == "inputs" || field.key() == "outputs")
                                  && field.value().is_object();
            if(!portSide)
            {
                findUidReferences(field.value(),
                                  uid,
                                  field.key().find("uid") != std::string::npos,
                                  fieldPath,
                                  use.otherPaths);
                continue;
            }
            for(auto port = field.value().begin(); port != field.value().end(); ++port)
            {
                if(isNodePortKey(port.key()) && uidOf(port.value()) == uid)
                {
                    use.portNodes.insert(index);
                    continue;
                }
                std::string portPath = fieldPath;
                portPath += '.';
                portPath += port.key();
                findUidReferences(port.value(),
                                  uid,
                                  port.key().find("uid") != std::string::npos,
                                  portPath,
                                  use.otherPaths);
            }
        }
    }
    return use;
}

// The paths outside the nodes where the template refers to `uid`, other than the "uid"
// key of a tensor entry.
inline std::vector<std::string> findOtherUidReferences(const nlohmann::json& templateJson,
                                                       int64_t uid)
{
    std::vector<std::string> found;
    if(!templateJson.is_object())
    {
        return found;
    }
    for(auto field = templateJson.begin(); field != templateJson.end(); ++field)
    {
        if(field.key() == "nodes")
        {
            continue;
        }
        if(field.key() != "tensors" || !field.value().is_array())
        {
            findUidReferences(field.value(),
                              uid,
                              field.key().find("uid") != std::string::npos,
                              field.key(),
                              found);
            continue;
        }
        const auto& tensors = field.value();
        for(size_t index = 0; index < tensors.size(); ++index)
        {
            std::string tensorPath = "tensors[";
            tensorPath += std::to_string(index);
            tensorPath += ']';
            const auto& tensor = tensors.at(index);
            if(!tensor.is_object())
            {
                findUidReferences(tensor, uid, false, tensorPath, found);
                continue;
            }
            for(auto key = tensor.begin(); key != tensor.end(); ++key)
            {
                if(key.key() == "uid")
                {
                    continue;
                }
                std::string keyPath = tensorPath;
                keyPath += '.';
                keyPath += key.key();
                findUidReferences(
                    key.value(), uid, key.key().find("uid") != std::string::npos, keyPath, found);
            }
        }
    }
    return found;
}

// Throws BundleMetadataError naming `where` for the first node port of the template that
// holds neither null nor a uid, for example a `${case.*}` placeholder. The pruning cannot
// tell whether such a port holds an absent uid.
inline void checkNodePortsAreLiteral(const nlohmann::json& templateJson, const std::string& where)
{
    if(!templateJson.contains("nodes") || !templateJson.at("nodes").is_array())
    {
        return;
    }
    const auto& nodes = templateJson.at("nodes");
    for(size_t index = 0; index < nodes.size(); ++index)
    {
        const auto& node = nodes.at(index);
        for(const char* side : {"inputs", "outputs"})
        {
            if(!node.is_object() || !node.contains(side) || !node.at(side).is_object())
            {
                continue;
            }
            const auto& ports = node.at(side);
            for(auto port = ports.begin(); port != ports.end(); ++port)
            {
                if(!isNodePortKey(port.key()) || port.value().is_null()
                   || uidOf(port.value()).has_value())
                {
                    continue;
                }
                std::string message = where;
                message += ": absent_tensors needs a uid or null in each node port, and nodes[";
                message += std::to_string(index);
                message += "].";
                message += side;
                message += '.';
                message += port.key();
                message += " holds ";
                message += port.value().dump();
                throw hipdnn_integration_tests::BundleMetadataError(message);
            }
        }
    }
}

inline std::string
    absentTensorMessage(const std::string& where, int64_t uid, const std::string& rule)
{
    std::string message = where;
    message += ": absent tensor ";
    message += std::to_string(uid);
    message += ' ';
    message += rule;
    return message;
}

// A sweep case with `absent_tensors`, checked against its template: the template with
// the absent tensors removed, the case metadata and the absent uids.
struct AbsentTensorCase
{
    nlohmann::json graphTemplate;
    hipdnn_integration_tests::BundleMetadata metadata;
    std::vector<int64_t> uids;
};

// The rules that tie a case's names to its absent tensors. A case names an absent tensor
// in none of: values.tensors, a key of another case value (`${case.<field>.<uid>}`),
// tensor_patches, the metadata `inputs`, the metadata `aliases`, the metadata
// `start_offsets`, the metadata `live_rows`.
inline void
    checkCaseDoesNotNameAbsentTensors(const nlohmann::json& caseJson,
                                      const hipdnn_integration_tests::BundleMetadata& metadata,
                                      const std::unordered_set<int64_t>& absent,
                                      const std::vector<int64_t>& uids,
                                      const std::string& where)
{
    using hipdnn_integration_tests::BundleMetadataError;

    if(caseJson.contains("values") && caseJson.at("values").is_object())
    {
        const auto& values = caseJson.at("values");
        if(values.contains("tensors"))
        {
            const auto& tensors = values.at("tensors");
            if(!tensors.is_array())
            {
                throw std::runtime_error("values.tensors must be an array");
            }
            for(const auto& entry : tensors)
            {
                const auto uid = entry.at("uid").get<int64_t>();
                if(absent.count(uid) != 0)
                {
                    throw BundleMetadataError(
                        absentTensorMessage(where, uid, "has a values.tensors entry"));
                }
            }
        }
        for(const int64_t uid : uids)
        {
            const auto uidKey = std::to_string(uid);
            for(auto field = values.begin(); field != values.end(); ++field)
            {
                if(field.key() == "tensors" || !field.value().is_object()
                   || !field.value().contains(uidKey))
                {
                    continue;
                }
                std::string rule = "is a key of the case value values.";
                rule += field.key();
                rule += '.';
                rule += uidKey;
                throw BundleMetadataError(absentTensorMessage(where, uid, rule));
            }
        }
    }

    if(caseJson.contains("tensor_patches") && caseJson.at("tensor_patches").is_array())
    {
        for(const auto& patch : caseJson.at("tensor_patches"))
        {
            if(!patch.is_object() || !patch.contains("uid") || !patch.at("uid").is_number_integer())
            {
                throw std::runtime_error("tensor_patch entry is missing an integer uid");
            }

            const auto uid = patch.at("uid").get<int64_t>();
            if(absent.count(uid) != 0)
            {
                throw BundleMetadataError(
                    absentTensorMessage(where, uid, "has a tensor_patches entry"));
            }
        }
    }

    for(const int64_t uid : uids)
    {
        if(metadata.inputs.has_value() && metadata.inputs->count(uid) != 0)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, uid, "has a metadata inputs entry"));
        }
    }

    for(const auto& alias : metadata.aliases)
    {
        if(absent.count(alias.uid) != 0)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, alias.uid, "is the view of a metadata aliases entry"));
        }
        if(absent.count(alias.ofUid) != 0)
        {
            throw BundleMetadataError(absentTensorMessage(
                where, alias.ofUid, "is the owner of a metadata aliases entry"));
        }
    }

    for(const auto& entry : metadata.startOffsets)
    {
        if(absent.count(entry.uid) != 0)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, entry.uid, "has a metadata start_offsets entry"));
        }
    }

    for(const auto& entry : metadata.liveRows)
    {
        for(const auto& uid : {std::optional<int64_t>(entry.uid), entry.lengthsUid, entry.likeUid})
        {
            if(uid.has_value() && absent.count(*uid) != 0)
            {
                throw BundleMetadataError(
                    absentTensorMessage(where, *uid, "is named by a metadata live_rows entry"));
            }
        }
    }
}

// Checks a sweep case that has `absent_tensors` against its template and returns the
// pruned template. Every rule throws BundleMetadataError naming `where`, and the absent
// uid where there is one. Runs before placeholder expansion: the placeholders of an
// absent tensor have no case value.
inline AbsentTensorCase prepareAbsentTensors(const nlohmann::json& templateJson,
                                             const nlohmann::json& caseJson,
                                             int sweepVersion,
                                             const std::string& where)
{
    using hipdnn_integration_tests::BundleMetadataError;

    if(!caseJson.contains("metadata") || caseJson.at("metadata").is_null())
    {
        throw BundleMetadataError(where
                                  + ": absent_tensors needs case metadata with format_version 2");
    }
    auto metadata
        = hipdnn_integration_tests::parseBundleMetadataJson(caseJson.at("metadata"), where);
    if(metadata.formatVersion < 2)
    {
        throw BundleMetadataError(where + ": absent_tensors needs format_version 2");
    }

    if(sweepVersion != 2)
    {
        throw BundleMetadataError(where + ": absent_tensors needs sweep version 2, not "
                                  + std::to_string(sweepVersion));
    }

    const auto& list = caseJson.at("absent_tensors");
    if(!list.is_array() || list.empty())
    {
        throw BundleMetadataError(
            where + ": absent_tensors must be a non-empty array of integer tensor uids");
    }
    std::vector<int64_t> uids;
    std::unordered_set<int64_t> absent;
    for(const auto& entry : list)
    {
        if(!entry.is_number_integer())
        {
            throw BundleMetadataError(
                where + ": absent_tensors must be a non-empty array of integer tensor uids");
        }
        const auto uid = entry.get<int64_t>();
        if(!absent.insert(uid).second)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, uid, "is named twice in absent_tensors"));
        }
        uids.push_back(uid);
    }

    const auto templateUids = collectTemplateTensorUids(templateJson);
    for(const int64_t uid : uids)
    {
        if(templateUids.count(uid) == 0)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, uid, "is not a tensor of the template"));
        }
    }

    checkCaseDoesNotNameAbsentTensors(caseJson, metadata, absent, uids, where);

    if(caseJson.contains("golden") && !caseJson.at("golden").is_null())
    {
        throw BundleMetadataError(where
                                  + ": absent_tensors and golden data do not mix; remove the "
                                    "golden pointer or the absent tensors");
    }

    for(const auto& tensor : templateJson.at("tensors"))
    {
        const auto uid = tensor.at("uid").get<int64_t>();
        if(absent.count(uid) == 0 || !tensor.contains("virtual") || tensor.at("virtual") == false)
        {
            continue;
        }
        throw BundleMetadataError(absentTensorMessage(where,
                                                      uid,
                                                      tensor.at("virtual") == true
                                                          ? "is virtual"
                                                          : "has a virtual flag that is not a "
                                                            "literal boolean in the template"));
    }

    checkNodePortsAreLiteral(templateJson, where);
    for(const int64_t uid : uids)
    {
        const auto use = findNodeUidUse(templateJson, uid);
        if(use.portNodes.size() > 1)
        {
            throw BundleMetadataError(
                absentTensorMessage(where, uid, "is held by node ports of more than one node"));
        }
        if(!use.otherPaths.empty())
        {
            throw BundleMetadataError(absentTensorMessage(
                where, uid, "is held by a node outside a node port, at " + use.otherPaths.front()));
        }
        const auto references = findOtherUidReferences(templateJson, uid);
        if(!references.empty())
        {
            throw BundleMetadataError(
                absentTensorMessage(where, uid, "is referenced at " + references.front()));
        }
    }

    auto graphTemplate = pruneAbsentTensors(templateJson, uids);
    return {std::move(graphTemplate), std::move(metadata), std::move(uids)};
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
//   * golden outputs present, no metadata  -> LoadError::UNVALIDATABLE_GOLDEN_DATA
//   * no golden outputs, no metadata       -> bundle with empty metadata
//   * metadata present but malformed       -> throws BundleMetadataError
//   * metadata flags that contradict the golden data it declares (both references
//     off with none, expected_rejection with some) -> throws BundleMetadataError
//   * aliases that the graph or the tensor data on disk rule out -> throws BundleMetadataError
//   * valid graph, input blobs absent      -> bundle with blobs == nullopt
//   * inputs present, outputs absent       -> bundle verified against reference
//   * inputs and outputs present           -> bundle verified against golden data
//
// Inputs and outputs are loaded independently. Output uids come from the graph;
// every other declared tensor is treated as input. Every failure above is
// reported through the return value, except two that propagate uncaught: a
// BundleMetadataError from malformed metadata, and a
// RuntimePassByValueInvariantError from buildGraphBuffer(). Callers (see
// BundleRegistration.hpp's classifyBundle()) deliberately treat both as hard
// failures rather than quiet skips, and the exception carries the detail.
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

    const auto blobPathForUid
        = [jsonPath](int64_t uid) { return detail::tensorBlobPath(jsonPath, uid); };
    const bool goldenOutputsPresent
        = !bundle.outputTensorUids.empty()
          && detail::blobsPresentFor(bundle.outputTensorUids, blobPathForUid);

    // An absent .meta.json is fine for a graph-only bundle (default metadata) but
    // not next to golden blobs. A present-but-malformed one is an authoring error
    // either way: loadBundleMetadata() throws BundleMetadataError, which is left
    // to propagate so it never falls back to default metadata.
    auto metadata = hipdnn_integration_tests::loadBundleMetadata(jsonPath);
    if(!metadata.has_value())
    {
        if(goldenOutputsPresent)
        {
            return LoadError::UNVALIDATABLE_GOLDEN_DATA;
        }
        metadata.emplace();
    }
    bundle.metadata = std::move(*metadata);

    // The `.tensors.dvc` pointer counts as declaring golden data even when the blobs
    // it names have not been pulled.
    auto pointerPath = jsonPath;
    pointerPath.replace_extension(".tensors.dvc");
    const bool goldenDeclared = goldenOutputsPresent || std::filesystem::exists(pointerPath);
    detail::checkGoldenDeclarationRules(bundle.metadata, goldenDeclared, jsonPath.string());

    detail::describeTensorBlobs(bundle, *graphJson, blobPathForUid);
    detail::checkStartOffsetRules(bundle.metadata, bundle.graphWrapper(), jsonPath.string());
    detail::checkLiveRowRules(
        bundle.metadata, bundle.graphWrapper(), bundle.outputTensorUids, jsonPath.string());
    // One blob is enough: a bundle that leaves out the view's blob still stores its
    // other tensors apart.
    detail::checkAliasRules(
        bundle.metadata,
        bundle.graphWrapper(),
        bundle.outputTensorUids,
        goldenDeclared
            || detail::anyBlobPresentFor(detail::allTensorUids(*graphJson), blobPathForUid),
        jsonPath.string());

    return bundle;
}

// Load either a direct bundle or one logical template-sweep case.
//
// Sweep cases take graph.template.json plus sweep.json from `sweeps`, locate the
// discovered case id, expand `${case...}` placeholders, load inline metadata, and
// resolve an optional golden directory. Sweep authoring errors are reported as
// INVALID_SWEEP_CASE; an expanded graph that still fails schema conversion is
// INVALID_GRAPH_SCHEMA. As with the direct-bundle overload, two exceptions
// propagate uncaught: a BundleMetadataError from a malformed metadata block, and
// a RuntimePassByValueInvariantError from buildGraphBuffer(). A sweep `version`
// other than 1 or 2 also throws BundleMetadataError, and so does a case that
// breaks an `absent_tensors` rule, has an unbuildable graph or has no output left.
// A case with `absent_tensors` loads from a copy of the template without those
// tensors, made before the placeholders are expanded. An expansion error in such a
// case is still INVALID_SWEEP_CASE.
inline LoadResult loadIntegrationTestBundle(const DiscoveredBundle& discovered,
                                            SweepManifestCache& sweeps)
{
    if(!discovered.isTemplateSweepCase())
    {
        return loadIntegrationTestBundle(discovered.jsonPath);
    }

    const auto& sweep = sweeps.get(discovered);
    const auto& templateJson = sweep.templateJson;
    if(!templateJson.has_value() || !sweep.manifest.has_value())
    {
        return LoadError::MALFORMED_JSON;
    }

    const auto* caseJson = sweep.manifest->findCase(discovered.sweep->caseId);
    if(caseJson == nullptr)
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    const auto where = discovered.diagnosticPath().string();
    const int sweepVersion = detail::rethrowAsBundleMetadataError(where, [&] {
        return detail::checkSweepVersion(sweep.manifest->version(), discovered.jsonPath, where);
    });

    std::optional<detail::AbsentTensorCase> absent;
    if(caseJson->contains("absent_tensors"))
    {
        absent.emplace(detail::rethrowAsBundleMetadataError(where, [&] {
            return detail::prepareAbsentTensors(*templateJson, *caseJson, sweepVersion, where);
        }));
    }
    const nlohmann::json& graphTemplate
        = absent.has_value() ? absent->graphTemplate : *templateJson;

    nlohmann::json expandedGraph;
    std::optional<std::filesystem::path> goldenDirectory;
    try
    {
        expandedGraph = detail::expandTemplateGraph(graphTemplate, *caseJson, discovered);
        goldenDirectory = detail::resolveSweepGoldenDirectory(discovered.jsonPath, *caseJson);
        detail::applyAddTensors(expandedGraph, *caseJson);
        detail::applyNodePatches(expandedGraph, *caseJson);
        detail::applyTensorPatches(expandedGraph, *caseJson);
    }
    catch(const std::exception&)
    {
        return LoadError::INVALID_SWEEP_CASE;
    }

    flatbuffers::DetachedBuffer graphBuffer;
    if(!detail::buildGraphBuffer(expandedGraph, graphBuffer))
    {
        if(absent.has_value())
        {
            throw hipdnn_integration_tests::BundleMetadataError(
                where + ": the graph with absent tensors " + detail::formatUidList(absent->uids)
                + " does not build; a removed tensor fills a port the op requires");
        }
        return LoadError::INVALID_GRAPH_SCHEMA;
    }

    IntegrationTestBundle bundle;
    bundle.graphBuffer = std::move(graphBuffer);

    // Mirror the direct-bundle path: extractOutputUidsFromJson handles flat-uid
    // ops (Reduction/ResampleFwd/ResampleBwd/CustomOp) that getOutputTensorUidsFromGraph
    // throws on.
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

    if(absent.has_value() && bundle.outputTensorUids.empty())
    {
        throw hipdnn_integration_tests::BundleMetadataError(where + ": removing "
                                                            + detail::formatUidList(absent->uids)
                                                            + " leaves no output to compare");
    }

    // Golden blobs without metadata cannot be validated, so that combination is
    // UNVALIDATABLE_GOLDEN_DATA and hard-fails downstream (see LoadError). Mirrors
    // the direct-bundle path above.
    const bool goldenOutputsPresent
        = goldenDirectory.has_value() && !bundle.outputTensorUids.empty()
          && detail::blobsPresentFor(bundle.outputTensorUids, [&](int64_t uid) {
                 return *goldenDirectory / ("tensor" + std::to_string(uid) + ".bin");
             });

    // Every sweep case must carry a metadata block. Metadata (arch lock,
    // ROCm/GPU version, seed, VRAM guard) is what validates golden data and
    // anchors the case, so an absent block is MISSING_METADATA. A present but
    // malformed block is an authoring error: parseBundleMetadataJson() throws
    // BundleMetadataError, which is left to propagate (a typo in the metadata
    // block must not quietly delete the case). A case with absent tensors already
    // has its metadata parsed.
    if(absent.has_value())
    {
        bundle.metadata = std::move(absent->metadata);
    }
    else
    {
        if(!caseJson->contains("metadata") || caseJson->at("metadata").is_null())
        {
            return goldenOutputsPresent ? LoadError::UNVALIDATABLE_GOLDEN_DATA
                                        : LoadError::MISSING_METADATA;
        }
        bundle.metadata
            = hipdnn_integration_tests::parseBundleMetadataJson(caseJson->at("metadata"), where);
    }

    const bool goldenDeclared = goldenOutputsPresent || goldenDirectory.has_value();
    detail::checkGoldenDeclarationRules(bundle.metadata, goldenDeclared, where);

    // One blob is enough: a case that leaves out the view's blob still stores its
    // other tensors apart. Mirrors the direct-bundle path above.
    bool anyBlobPresent = false;
    if(goldenDirectory.has_value())
    {
        const auto blobPathForUid = [goldenDir = *goldenDirectory](int64_t uid) {
            return goldenDir / ("tensor" + std::to_string(uid) + ".bin");
        };
        detail::describeTensorBlobs(bundle, expandedGraph, blobPathForUid);
        anyBlobPresent
            = detail::anyBlobPresentFor(detail::allTensorUids(expandedGraph), blobPathForUid);
    }
    detail::checkStartOffsetRules(bundle.metadata, bundle.graphWrapper(), where);
    detail::checkLiveRowRules(
        bundle.metadata, bundle.graphWrapper(), bundle.outputTensorUids, where);
    detail::checkAliasRules(bundle.metadata,
                            bundle.graphWrapper(),
                            bundle.outputTensorUids,
                            goldenDeclared || anyBlobPresent,
                            where);

    return bundle;
}

// Loads one bundle on its own. A pass over many sweep cases should share one
// SweepManifestCache instead, so each manifest is parsed once.
inline LoadResult loadIntegrationTestBundle(const DiscoveredBundle& discovered)
{
    SweepManifestCache sweeps;
    return loadIntegrationTestBundle(discovered, sweeps);
}

} // namespace hipdnn_integration_tests::bundle
