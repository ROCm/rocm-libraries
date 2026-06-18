// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

namespace hipdnn_integration_tests::golden
{

// Loaded tensors keyed by tensor UID. Holds every tensor declared by the graph —
// inputs carry their data, output tensors carry their expected (golden) values as
// loaded from the .bin blobs. The harness saves the outputs as golden and zeroes
// them before execution.
using TensorMap = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// One test's worth of bundle data loaded from disk.
//
//   graphBuffer      — the parsed graph, as a flatbuffer. Always present in a
//                      loaded bundle; the engine deserializes it (from_binary)
//                      and the harness walks it (GraphWrapper) for dtypes and
//                      tolerances. A bundle that cannot even produce a graph is a
//                      LoadError, not a bundle.
//   metadata         — .meta.json contents (VRAM / arch guards). MANDATORY: a
//                      bundle without a valid .meta.json is a LoadError, so a
//                      loaded bundle always carries real metadata.
//   outputTensorUids — UIDs of the graph's output tensors, derived from the
//                      graph. Always available (even for a graph-only bundle),
//                      so the harness knows which tensors to compare / allocate.
//   tensors          — the loaded tensor data. Absent (nullopt) for a graph-only
//                      bundle (no "tensors" in the graph, or .bin data not pulled
//                      via DVC); such a bundle cannot be executed/compared and
//                      the harness SKIPs it.
struct IntegrationTestBundle
{
    flatbuffers::DetachedBuffer graphBuffer;
    hipdnn_test_sdk::utilities::BundleMetadata metadata;
    std::vector<int64_t> outputTensorUids;
    std::optional<TensorMap> tensors;

    // View over the graph flatbuffer, valid as long as this bundle lives.
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper{graphBuffer.data(),
                                                                          graphBuffer.size()};
    }
};

// Why a load did NOT produce a bundle. These are the FAIL outcomes — an authoring
// error in the bundle itself. (A bundle that loads but lacks tensor data is a
// successfully-loaded graph-only bundle, not a LoadError; the harness SKIPs it.)
enum class LoadError
{
    MALFORMED_JSON, // the graph .json is not syntactically valid JSON
    INVALID_GRAPH_SCHEMA, // valid JSON, but not a valid graph (cannot build flatbuffer)
    MISSING_METADATA, // required .meta.json companion is absent or invalid
    TENSOR_LOAD_FAILED // a tensor .bin is present but failed to load (wrong size,
    // unreadable, unsupported dtype, ...)
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
    default:
        return "unknown load error";
    }
}

namespace detail
{

// True iff every tensor declared in the graph has its companion .bin blob on
// disk. The blob path is "{stem}.tensor{uid}.bin", matching the loader's own
// derivation. A graph with no "tensors" array is graph-only -> returns false.
inline bool allTensorBlobsPresent(const nlohmann::json& graphJson,
                                  const std::filesystem::path& jsonPath)
{
    if(!graphJson.contains("tensors") || !graphJson.at("tensors").is_array()
       || graphJson.at("tensors").empty())
    {
        return false;
    }

    auto basePath = jsonPath;
    basePath.replace_extension();
    for(const auto& tensor : graphJson.at("tensors"))
    {
        if(!tensor.contains("uid"))
        {
            return false;
        }
        const auto uid = tensor.at("uid").get<int64_t>();
        const auto binPath
            = std::filesystem::path(basePath.string() + ".tensor" + std::to_string(uid) + ".bin");
        if(!std::filesystem::exists(binPath))
        {
            return false;
        }
    }
    return true;
}

} // namespace detail

// Load a bundle from its graph .json path, classifying the outcome.
//
// This deliberately does NOT call test_sdk's loadGraphAndTensors(), whose
// all-or-nothing contract ("graph AND at least one tensor, or throw") conflicts
// with our design where a graph-only bundle is legitimate. Instead it composes
// the same test_sdk primitives (json -> flatbuffer graph, per-tensor blob load)
// under our own policy:
//
//   * graph .json not parseable           -> LoadError::MALFORMED_JSON      (FAIL)
//   * parseable but not a valid graph     -> LoadError::INVALID_GRAPH_SCHEMA(FAIL)
//   * valid graph, no .meta.json companion-> LoadError::MISSING_METADATA    (FAIL)
//   * valid graph, tensor .bin data absent-> bundle with tensors == nullopt (SKIP)
//   * valid graph, .bin present but broken-> LoadError::TENSOR_LOAD_FAILED  (FAIL)
//   * valid graph, all .bin present       -> fully loaded bundle            (RUN)
//
// The function is total: it never lets an exception escape. Every outcome is
// either a loaded bundle or a classified LoadError.
inline LoadResult loadIntegrationTestBundle(const std::filesystem::path& jsonPath)
{
    // 1. Read and parse the graph .json. Unreadable or unparseable -> FAIL.
    std::ifstream stream(jsonPath);
    if(!stream)
    {
        return LoadError::MALFORMED_JSON;
    }

    const auto graphJson = nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false);
    if(graphJson.is_discarded())
    {
        return LoadError::MALFORMED_JSON;
    }

    // 2. Verify the graph by building the flatbuffer. A structurally invalid
    //    graph throws -> INVALID_GRAPH_SCHEMA.
    flatbuffers::FlatBufferBuilder builder;
    try
    {
        auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            builder, graphJson);
        builder.Finish(offset);
    }
    catch(const std::exception&)
    {
        return LoadError::INVALID_GRAPH_SCHEMA;
    }

    // 3. Metadata is MANDATORY: every valid-graph bundle must ship a valid
    //    .meta.json companion. loadBundleMetadata returns nullopt both when the
    //    file is absent and when it is present but invalid (bad JSON / bad
    //    format_version) — either way it is an authoring error -> FAIL.
    auto metadata = hipdnn_test_sdk::utilities::loadBundleMetadata(jsonPath);
    if(!metadata.has_value())
    {
        return LoadError::MISSING_METADATA;
    }

    // 4. Graph + metadata verified: capture them and the output UIDs (always
    //    available, even for a graph-only bundle).
    IntegrationTestBundle bundle;
    bundle.graphBuffer = builder.Release();
    bundle.metadata = std::move(*metadata);
    bundle.outputTensorUids = hipdnn_test_sdk::utilities::getOutputTensorUidsFromGraph(graphJson);

    // 5. Load tensor .bin data if every blob is present; otherwise leave
    //    tensors == nullopt (graph-only bundle -> harness SKIPs). A blob that is
    //    present but fails to load (wrong size, unreadable, unsupported dtype)
    //    throws inside tensorFromFileAndAttributes; we catch it here and classify
    //    it as TENSOR_LOAD_FAILED so the loader is total (every outcome is either
    //    a bundle or a named LoadError, never a raw escaping exception).
    if(detail::allTensorBlobsPresent(graphJson, jsonPath))
    {
        const auto& graph
            = *hipdnn_flatbuffers_sdk::data_objects::GetGraph(bundle.graphBuffer.data());
        auto basePath = jsonPath;
        basePath.replace_extension();

        try
        {
            TensorMap tensorMap;
            for(const auto* attributes : *graph.tensors())
            {
                const auto tensorPath
                    = basePath.string() + ".tensor" + std::to_string(attributes->uid()) + ".bin";
                tensorMap[attributes->uid()]
                    = hipdnn_test_sdk::utilities::tensorFromFileAndAttributes(tensorPath,
                                                                              *attributes);
            }
            bundle.tensors = std::move(tensorMap);
        }
        catch(const std::exception&)
        {
            return LoadError::TENSOR_LOAD_FAILED;
        }
    }

    return bundle;
}

} // namespace hipdnn_integration_tests::golden
