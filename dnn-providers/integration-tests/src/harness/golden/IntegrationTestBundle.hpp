// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
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
//   metadata         — .meta.json contents (VRAM / arch guards). Mandatory ONLY
//                      for golden bundles (those shipping output .bin blobs);
//                      metadata validates golden data, so a bundle without it is
//                      a LoadError. For a no-golden bundle (graph-only, or
//                      inputs-only verified against a reference) absent metadata
//                      is valid and this is default-constructed (all fields
//                      empty); the optional-aware consumers fall back to defaults.
//   outputTensorUids — UIDs of the graph's output tensors, derived from the
//                      graph. Always available (even for a graph-only bundle),
//                      so the harness knows which tensors to compare / allocate.
//   tensors          — the loaded tensor data, keyed by uid. Holds the INPUT
//                      tensors (with their data) whenever they are present on
//                      disk, plus the OUTPUT tensors carrying their golden values
//                      iff every output blob is present (see hasGoldenOutputs).
//                      Absent (nullopt) only when the input blobs themselves are
//                      not on disk — a true graph-only bundle. The harness may
//                      still synthesize inputs for such a bundle (tier 3); if it
//                      cannot, it SKIPs.
//   hasGoldenOutputs — true iff every output tensor's .bin blob was present and
//                      loaded into `tensors`. When false, `tensors` (if present)
//                      carries inputs only — the engine output must be verified
//                      against a reference executor, not golden data.
struct IntegrationTestBundle
{
    flatbuffers::DetachedBuffer graphBuffer;
    hipdnn_test_sdk::utilities::BundleMetadata metadata;
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

// Why a load did NOT produce a bundle. These are the FAIL outcomes — an authoring
// error in the bundle itself. (A bundle that loads but lacks tensor data is a
// successfully-loaded graph-only bundle, not a LoadError; the harness SKIPs it.)
enum class LoadError
{
    MALFORMED_JSON, // the graph .json is not syntactically valid JSON
    INVALID_GRAPH_SCHEMA, // valid JSON, but not a valid graph (cannot build flatbuffer)
    MISSING_METADATA, // golden bundle's .meta.json companion is absent or invalid
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

// The on-disk blob path for a tensor: "{stem}.tensor{uid}.bin", matching the
// loader's own derivation.
inline std::filesystem::path tensorBlobPath(const std::filesystem::path& jsonPath, int64_t uid)
{
    auto basePath = jsonPath;
    basePath.replace_extension();
    return {basePath.string() + ".tensor" + std::to_string(uid) + ".bin"};
}

// True iff every uid in `uids` has its companion .bin blob on disk. An empty
// `uids` set returns true (vacuously) — callers handle "no such tensors"
// separately (e.g. a graph with no inputs, or no outputs).
inline bool blobsPresentFor(const std::vector<int64_t>& uids, const std::filesystem::path& jsonPath)
{
    for(const int64_t uid : uids)
    {
        if(!std::filesystem::exists(tensorBlobPath(jsonPath, uid)))
        {
            return false;
        }
    }
    return true;
}

// The uids of every tensor declared in the graph's "tensors" array. Empty if the
// array is absent/empty (a graph-only bundle). Tensors without a "uid" are
// skipped (malformed entries are caught later when building the flatbuffer).
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
//   * golden bundle, no/invalid .meta.json-> LoadError::MISSING_METADATA    (FAIL)
//   * no-golden bundle, no .meta.json      -> bundle, metadata default-constructed
//   * valid graph, input .bin data absent -> bundle, tensors == nullopt     (tier-3:
//                                            harness may synthesize, else SKIP)
//   * valid graph, .bin present but broken-> LoadError::TENSOR_LOAD_FAILED  (FAIL)
//   * valid graph, inputs present,
//       outputs absent                    -> bundle, tensors set,
//                                            hasGoldenOutputs == false (verify via ref)
//   * valid graph, inputs + outputs present-> bundle, hasGoldenOutputs == true (golden)
//
// Inputs and outputs are loaded INDEPENDENTLY (partial loading): a bundle that
// ships input blobs but no output (golden) blobs is legitimate — its engine
// output is verified against a reference executor instead of golden data. Output
// uids come from getOutputTensorUidsFromGraph; everything else declared in the
// graph is treated as an input.
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

    // 3. Capture the graph and derive the output UIDs (always available, even
    //    for a graph-only bundle).
    IntegrationTestBundle bundle;
    bundle.graphBuffer = builder.Release();
    bundle.outputTensorUids = hipdnn_test_sdk::utilities::getOutputTensorUidsFromGraph(graphJson);

    // 4. Metadata is mandatory ONLY for golden bundles — those shipping output
    //    .bin blobs. Metadata (arch lock, provenance, seed) exists to validate
    //    golden data; a bundle with no golden outputs (pure graph-only, or
    //    inputs-only verified against a reference) has nothing for it to
    //    validate, so absent metadata is fine and we default-construct it.
    //
    //    loadBundleMetadata returns nullopt both when the .meta.json is absent
    //    and when it is present but invalid (bad JSON / bad format_version). For
    //    a golden bundle either case is an authoring error -> FAIL.
    const bool goldenOutputsPresent = !bundle.outputTensorUids.empty()
                                      && detail::blobsPresentFor(bundle.outputTensorUids, jsonPath);

    auto metadata = hipdnn_test_sdk::utilities::loadBundleMetadata(jsonPath);
    if(!metadata.has_value())
    {
        if(goldenOutputsPresent)
        {
            return LoadError::MISSING_METADATA;
        }
        metadata.emplace(); // graph-only / no-golden: empty metadata is valid.
    }
    bundle.metadata = std::move(*metadata);

    // 5. Load tensor .bin data, inputs and outputs INDEPENDENTLY.
    //
    //    Output uids are the graph's outputs; every other declared tensor is an
    //    input. We load inputs only if all input blobs are present, and outputs
    //    (golden) only if all output blobs are present:
    //
    //      * all input blobs present  -> tensors gets the inputs
    //      * all output blobs present -> tensors also gets the golden outputs and
    //                                    hasGoldenOutputs = true
    //      * input blobs absent       -> tensors stays nullopt (tier-3: harness
    //                                    may synthesize inputs, else SKIP)
    //
    //    A blob that is present but fails to load (wrong size, unreadable,
    //    unsupported dtype) throws inside tensorFromFileAndAttributes; we catch it
    //    and classify it as TENSOR_LOAD_FAILED so the loader stays total.
    {
        const std::vector<int64_t> allUids = detail::allTensorUids(graphJson);

        const std::set<int64_t> outputUidSet(bundle.outputTensorUids.begin(),
                                             bundle.outputTensorUids.end());
        std::vector<int64_t> inputUids;
        for(const int64_t uid : allUids)
        {
            if(outputUidSet.count(uid) == 0)
            {
                inputUids.push_back(uid);
            }
        }

        // A graph with no declared inputs cannot be fed; treat as graph-only.
        const bool inputsPresent
            = !inputUids.empty() && detail::blobsPresentFor(inputUids, jsonPath);
        const bool outputsPresent = goldenOutputsPresent; // computed in step 4

        if(inputsPresent)
        {
            const auto& graph
                = *hipdnn_flatbuffers_sdk::data_objects::GetGraph(bundle.graphBuffer.data());

            // uid -> attributes, so we can load a chosen subset of tensors.
            std::unordered_map<int64_t,
                               const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
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
                    into[uid] = hipdnn_test_sdk::utilities::tensorFromFileAndAttributes(
                        detail::tensorBlobPath(jsonPath, uid), *it->second);
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
        }
    }

    return bundle;
}

} // namespace hipdnn_integration_tests::golden
