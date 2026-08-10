// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

/**
 * @file KernelIngestorTestFixtures.hpp
 * @brief Shared fixtures for the ingestor's SDK-level tests.
 *
 * Models the pack shape a real engine ships: two kernels that differ only in a ranking
 * axis, plus one the kernel-scoped matcher prunes. That is the smallest catalog where
 * matching, pruning, and ranking each have something to do.
 */
namespace hipdnn_plugin_sdk::ingestor::testing
{

constexpr const char* BLOCK_SIZE = "block_size";
constexpr const char* DTYPE = "dtype";
constexpr const char* GRAPH_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.graph_match";
constexpr const char* KERNEL_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.kernel_match";
constexpr const char* SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.score";

/// A minimal IGraph. The SDK-side machinery reads only the graph's identity, so the rest
/// of the interface throws: a test that starts depending on graph contents should fail
/// loudly rather than silently match against an empty graph.
class TestGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    /// @param graphId The identity to carry, or nullopt to model a legacy or unfinalized
    ///        graph that has none.
    explicit TestGraph(std::optional<GraphId> graphId = std::nullopt)
    {
        flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Graph> graph;
        if(graphId.has_value())
        {
            const auto uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*graphId);
            auto name = _builder.CreateString("test_graph");
            hipdnn_flatbuffers_sdk::data_objects::GraphBuilder graphBuilder(_builder);
            graphBuilder.add_name(name);
            graphBuilder.add_id(&uuid);
            graph = graphBuilder.Finish();
        }
        else
        {
            graph = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(_builder, "test_graph");
        }
        _builder.Finish(graph);
    }

    const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
    {
        return *flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _builder.GetBufferPointer());
    }

    bool isValid() const override
    {
        return true;
    }

    uint32_t nodeCount() const override
    {
        return 0;
    }

    bool hasOnlySupportedAttributes(
        std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supported*/) const override
    {
        return true;
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t /*index*/) const override
    {
        throw std::logic_error("TestGraph carries no nodes");
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&
        getNodeWrapper(uint32_t /*index*/) const override
    {
        throw std::logic_error("TestGraph carries no nodes");
    }

    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&
        nodeWrappers() const override
    {
        throw std::logic_error("TestGraph carries no nodes");
    }

    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        getTensorMap() const override
    {
        return _tensors;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        _tensors;
};

inline GraphId makeGraphId(uint8_t seed)
{
    GraphId id{};
    id.fill(seed);
    return id;
}

inline hipDeviceProp_t testDeviceProperties()
{
    hipDeviceProp_t properties{};
    properties.warpSize = 64;
    return properties;
}

inline bool acceptAnyGraph(const MatchContext& /*context*/)
{
    return true;
}

/// Accepts only FLOAT kernels, so a pack carrying a HALF kernel is pruned down.
inline bool acceptFloatKernels(const MatchContext& /*context*/, const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(DTYPE) == "FLOAT";
}

/// Bigger block size scores higher, so ranking has a defined winner.
inline double scoreByBlockSize(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    return static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
}

/// Registers this fixture's symbols for a test's duration, so tests sharing the
/// process-wide registry stay independent of one another's registrations.
class ScopedTestSymbols
{
public:
    ScopedTestSymbols()
    {
        GraphMatcherRegistry::registerSymbol(GRAPH_MATCH_SYMBOL, &acceptAnyGraph);
        KernelMatcherRegistry::registerSymbol(KERNEL_MATCH_SYMBOL, &acceptFloatKernels);
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
    }

    ~ScopedTestSymbols()
    {
        GraphMatcherRegistry::unregisterSymbol(GRAPH_MATCH_SYMBOL);
        KernelMatcherRegistry::unregisterSymbol(KERNEL_MATCH_SYMBOL);
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
    }

    ScopedTestSymbols(const ScopedTestSymbols&) = delete;
    ScopedTestSymbols& operator=(const ScopedTestSymbols&) = delete;
};

inline KernelDescriptor
    makeTestKernel(const std::string& id, int64_t blockSize, const std::string& dtype)
{
    KernelDescriptor kernel;
    kernel.id = id;
    kernel.name = id;
    kernel.sourceFile = "Test.cpp";
    kernel.entryPoint = "TestKernel";
    kernel.metadata = {{BLOCK_SIZE, blockSize}, {DTYPE, dtype}};
    return kernel;
}

/// Two FLOAT kernels differing only in block size, plus a HALF kernel the kernel-scoped
/// matcher prunes: a catalog of two survivors with a defined ranking.
inline std::unique_ptr<KernelIngestorStateManager<int>>
    makeTestStateManager(size_t cacheCapacity
                         = KernelIngestorStateManager<int>::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    MetadataSchema schema;
    schema.id = "kmd";
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, int64_t{64}}, {DTYPE, std::string{"FLOAT"}}};

    EngineDescriptor engine;
    engine.id = "ued";
    engine.name = "test:engine";
    engine.heuristicId = "uhd";
    engine.metadataSchemaId = "kmd";
    engine.knobs = {BLOCK_SIZE};

    std::vector<MatchDescriptor> matchers{
        {"umd_graph", "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL},
        {"umd_kernel", "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}};
    std::vector<DispatchDescriptor> dispatches{
        {"udd", "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}};

    KernelDescriptorPack pack;
    pack.id = "kdp";
    pack.name = "test pack";
    pack.matcherIds = {"umd_graph", "umd_kernel"};
    pack.engineId = "ued";
    pack.dispatchId = "udd";
    pack.kernels = {makeTestKernel("kernel_64_float", 64, "FLOAT"),
                    makeTestKernel("kernel_256_float", 256, "FLOAT"),
                    makeTestKernel("kernel_64_half", 64, "HALF")};

    return std::make_unique<KernelIngestorStateManager<int>>(
        std::move(engine),
        std::move(schema),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(&scoreByBlockSize),
        cacheCapacity);
}

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
