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

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

/**
 * @file KernelIngestorTestFixtures.hpp
 * @brief Shared fixtures for the ingestor's SDK-level tests, split across
 *        `plugin_sdk/tests/ingestor/`.
 *
 * Everything here is `inline`: this header is included by several translation units in
 * the same test binary, so a plain free function or namespace-scope object would violate
 * the one-definition rule the moment two of those files linked together.
 *
 * Two catalog shapes recur across the split files and are built here once:
 *  - The `int`-handle shape (`TestHandle`/`StateManager`/`makeStateManager()`): a pack of
 *    two FLOAT kernels differing in block size plus one HALF kernel a kernel-scoped
 *    matcher prunes, wired to counting matchers so a test can assert how often each
 *    scope ran.
 *  - The `int`-handle shape from `makeTestStateManager()`: the same catalog, wired to
 *    fixed (non-counting) matchers, for tests that only need a stable answer.
 *  - The `StubHandle` shape (`StubSettings`/`StubContext`/`makeStubStateManager()`):
 *    minimal stand-ins for the provider types `GenericEngine`/`GenericPlanBuilder` are
 *    parameterized on, for tests that construct those templates directly.
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

inline bool acceptAnyGraph(const MatchContext& /*context*/, BoundTokens& /*bound*/)
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

/// A device resolver for tests that construct an engine. Reports one fixed device, since
/// nothing here exercises multi-device behavior.
class TestDeviceResolver : public IDeviceResolver<int>
{
public:
    DeviceId deviceId(const int& /*handle*/) const override
    {
        return 0;
    }

    const hipDeviceProp_t& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    hipDeviceProp_t _properties = testDeviceProperties();
};

/// A descriptor id from a short seed, so a fixture can name ids readably while the type
/// stays the real 128-bit one.
inline DescriptorId testId(uint8_t seed)
{
    DescriptorId id{};
    id.fill(seed);
    return id;
}

inline const DescriptorId ENGINE_ID = testId(0xE0);
inline const DescriptorId SCHEMA_ID = testId(0xE1);
inline const DescriptorId HEURISTIC_ID = testId(0xE2);
inline const DescriptorId GRAPH_MATCHER_ID = testId(0xE3);
inline const DescriptorId KERNEL_MATCHER_ID = testId(0xE4);
inline const DescriptorId DISPATCH_ID = testId(0xE5);
inline const DescriptorId PACK_ID = testId(0xE6);

inline KernelDescriptor makeTestKernel(const DescriptorId& id,
                                       const std::string& name,
                                       int64_t blockSize,
                                       const std::string& dtype)
{
    KernelDescriptor kernel;
    kernel.id = id;
    kernel.name = name;
    kernel.source.sourceFile = "Test.cpp";
    kernel.source.entryPoint = "TestKernel";
    kernel.metadata = {{BLOCK_SIZE, MetadataValue{blockSize}}, {DTYPE, MetadataValue{dtype}}};
    return kernel;
}

/// The UED the fixture engine is built from. Separate from the state manager, which
/// holds only what selection reads.
inline EngineDescriptor makeTestEngine()
{
    EngineDescriptor engine;
    engine.id = ENGINE_ID;
    engine.name = "test:engine";
    engine.heuristicId = HEURISTIC_ID;
    engine.metadataSchemaId = SCHEMA_ID;
    engine.knobs = {BLOCK_SIZE};
    return engine;
}

/// Two FLOAT kernels differing only in block size, plus a HALF kernel the kernel-scoped
/// matcher prunes: a catalog of two survivors with a defined ranking.
inline std::unique_ptr<KernelIngestorStateManager<int>>
    makeTestStateManager(size_t cacheCapacity
                         = KernelIngestorStateManager<int>::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    // block_size defaults; dtype is mandatory, so every kernel below states it.
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    std::vector<MatchDescriptor> matchers{
        {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL},
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}};
    std::vector<DispatchDescriptor> dispatches{
        {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = {GRAPH_MATCHER_ID, KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_256_float", 256, "FLOAT"),
                    makeTestKernel(testId(0x66), "kernel_64_half", 64, "HALF")};

    return std::make_unique<KernelIngestorStateManager<int>>(
        std::move(schema),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        cacheCapacity);
}

// ---------------------------------------------------------------------------
// Counting matchers: the `TestHandle = int` catalog wired to observe how often each
// matcher scope ran. Shared by TestKernelIngestorStateManager.cpp and
// TestGenericPlanBuilder.cpp.
// ---------------------------------------------------------------------------

/// Counts matcher invocations, so a test can assert that a graph-scoped matcher runs
/// once per (graph, device) while a kernel-scoped one runs once per surviving kernel.
struct MatcherCounters
{
    int graphCalls = 0;
    int kernelCalls = 0;

    void reset()
    {
        graphCalls = 0;
        kernelCalls = 0;
    }
};

inline MatcherCounters& counters()
{
    static MatcherCounters s_counters;
    return s_counters;
}

/// An arbitrary value acceptGraph binds, so a test can prove it reached dispatch intact.
constexpr int64_t BOUND_TOKEN_VALUE = 4242;

inline bool acceptGraph(const MatchContext& /*context*/, BoundTokens& bound)
{
    ++counters().graphCalls;
    // Stands in for the tensor uids and dimensions a real matcher resolves: the point
    // under test is that whatever matching bound reaches dispatch without a re-match.
    bound["test.bound_token"] = BOUND_TOKEN_VALUE;
    return true;
}

inline bool rejectGraph(const MatchContext& /*context*/, BoundTokens& /*bound*/)
{
    ++counters().graphCalls;
    return false;
}

inline bool countingFloatKernels(const MatchContext& context, const KernelDefinition& kernel)
{
    ++counters().kernelCalls;
    return acceptFloatKernels(context, kernel);
}

/// Every kernel scores the same, so ranking falls through to the tie-break.
constexpr const char* CONSTANT_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.constant_score";

inline double scoreConstant(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 1.0;
}

/// Registers the constant scorer for a test's duration. The heuristic resolves its
/// symbol on first use, so the registration has to outlive the ranking call.
class ScopedConstantScore
{
public:
    ScopedConstantScore()
    {
        ScoreRegistry::registerSymbol(CONSTANT_SCORE_SYMBOL, &scoreConstant);
    }

    ~ScopedConstantScore()
    {
        ScoreRegistry::unregisterSymbol(CONSTANT_SCORE_SYMBOL);
    }

    ScopedConstantScore(const ScopedConstantScore&) = delete;
    ScopedConstantScore& operator=(const ScopedConstantScore&) = delete;
};

inline MetadataSchema makeSchema()
{
    return {SCHEMA_ID,
            "test schema",
            {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
             {DTYPE, MetadataType::STRING, std::nullopt}}};
}

inline KernelDescriptor makeKernel(const DescriptorId& id,
                                   const std::string& name,
                                   int64_t blockSize,
                                   const std::string& dtype,
                                   int64_t priority = 0)
{
    auto kernel = makeTestKernel(id, name, blockSize, dtype);
    kernel.priority = priority;
    return kernel;
}

/// The pack shape a real engine ships, wired to the counting matchers above.
inline KernelDescriptorPack makePack(const std::vector<DescriptorId>& matcherIds)
{
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = matcherIds;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeKernel(testId(0x65), "kernel_256_float", 256, "FLOAT"),
                    makeKernel(testId(0x66), "kernel_64_half", 64, "HALF")};
    return pack;
}

inline std::vector<MatchDescriptor> makeTestMatchers()
{
    return {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
}

inline std::vector<DispatchDescriptor> makeTestDispatches()
{
    return {{DISPATCH_ID, "test dispatch", "test.dispatch"}};
}

/// A catalog entry, for the ranking and plan tests that build one directly.
inline KernelDefinition
    makeDefinition(const DescriptorId& id, int64_t blockSize, int64_t priority = 0)
{
    return {id,
            PACK_ID,
            DISPATCH_ID,
            KernelSource{KernelSourceKind::EMBEDDED_SOURCE, "Test.cpp", "TestKernel"},
            {{BLOCK_SIZE, MetadataValue{blockSize}}},
            priority};
}

/// Registers counting matchers under this file's own symbol names, so these tests
/// observe invocation counts without disturbing the shared fixture's registrations.
class ScopedSymbols
{
public:
    ScopedSymbols(std::string graphSymbol,
                  GraphMatcherFn graphFn,
                  std::string kernelSymbol,
                  KernelMatcherFn kernelFn)
        : _graphSymbol(std::move(graphSymbol))
        , _kernelSymbol(std::move(kernelSymbol))
    {
        GraphMatcherRegistry::registerSymbol(_graphSymbol, graphFn);
        KernelMatcherRegistry::registerSymbol(_kernelSymbol, kernelFn);
        // The heuristic resolves its symbol on the first call that needs an order, not
        // at construction, so ranking tests need it registered for their duration too.
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
        counters().reset();
    }

    ~ScopedSymbols()
    {
        GraphMatcherRegistry::unregisterSymbol(_graphSymbol);
        KernelMatcherRegistry::unregisterSymbol(_kernelSymbol);
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
    }

    ScopedSymbols(const ScopedSymbols&) = delete;
    ScopedSymbols& operator=(const ScopedSymbols&) = delete;

private:
    std::string _graphSymbol;
    std::string _kernelSymbol;
};

using TestHandle = int;
using StateManager = KernelIngestorStateManager<TestHandle>;

inline std::unique_ptr<StateManager>
    makeStateManager(const std::string& scoreSymbol = SCORE_SYMBOL,
                     size_t cacheCapacity = StateManager::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    std::vector<MatchDescriptor> matchers{
        {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
    std::vector<DispatchDescriptor> dispatches{{DISPATCH_ID, "test dispatch", "test.dispatch"}};

    return std::make_unique<StateManager>(
        makeSchema(),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
        std::make_shared<NativeKernelHeuristic>(scoreSymbol),
        cacheCapacity);
}

// ---------------------------------------------------------------------------
// A no-op dispatch handler, generic over the handle type: sufficient wherever a test
// needs buildPlan() or initializeExecutionContext() to succeed without asserting on the
// launch itself. Shared by TestGenericEngine.cpp and TestGenericPlanBuilder.cpp.
// ---------------------------------------------------------------------------

template <typename THandle>
class NoopDispatchHandler : public IKernelDispatchHandler<THandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const THandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

/// Registers @p handler under @p symbol in DispatchRegistry<THandle> for a test's
/// duration.
template <typename THandle>
class ScopedDispatchRegistration
{
public:
    ScopedDispatchRegistration(std::string symbol, const IKernelDispatchHandler<THandle>& handler)
        : _symbol(std::move(symbol))
    {
        DispatchRegistry<THandle>::registerSymbol(_symbol, &handler);
    }

    ~ScopedDispatchRegistration()
    {
        DispatchRegistry<THandle>::unregisterSymbol(_symbol);
    }

    ScopedDispatchRegistration(const ScopedDispatchRegistration&) = delete;
    ScopedDispatchRegistration& operator=(const ScopedDispatchRegistration&) = delete;

private:
    std::string _symbol;
};

// ---------------------------------------------------------------------------
// StubHandle: minimal stand-ins for the provider types GenericEngine/GenericPlanBuilder
// are parameterized on. Only the members those templates actually touch are present, so
// a change that starts depending on more of a real handle or context fails here rather
// than compiling silently.
// ---------------------------------------------------------------------------

struct StubHandle
{
    void storeEngineDetailsDetachedBuffer(const void* /*ptr*/,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _buffers.push_back(std::move(buffer));
    }

private:
    std::vector<std::unique_ptr<flatbuffers::DetachedBuffer>> _buffers;
};

/// Every TSettings GenericPlanBuilder is instantiated over must carry this field (see
/// KnobFilter's doc): initializeExecutionSettings() populates it, getMaxWorkspaceSize()
/// reads it back.
struct StubSettings
{
    KnobFilter ingestorKnobFilter;
};

struct StubContext
{
    void setExecutionSettings(const StubSettings& /*settings*/) {}

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> plan)
    {
        _plan = std::move(plan);
    }

    bool hasPlan() const
    {
        return _plan != nullptr;
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> _plan;
};

/// A device resolver over StubHandle, for the engine-level tests. Reports one fixed
/// device; tests needing per-handle device resolution use IngestorMocks.hpp's
/// MockDeviceResolver instead.
class StubDeviceResolver : public IDeviceResolver<StubHandle>
{
public:
    DeviceId deviceId(const StubHandle& /*handle*/) const override
    {
        return 0;
    }

    const hipDeviceProp_t& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    hipDeviceProp_t _properties = testDeviceProperties();
};

/// A state manager over StubHandle: one FLOAT kernel behind one graph-scoped matcher,
/// for the GenericEngine-level tests that only need construction to succeed.
inline std::unique_ptr<KernelIngestorStateManager<StubHandle>> makeStubStateManager()
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = {GRAPH_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    return std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL}},
        std::vector<DispatchDescriptor>{
            {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}},
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
}

inline EngineDescriptor makeEngineWithKnobs(std::vector<std::string> knobs)
{
    EngineDescriptor engine;
    engine.id = ENGINE_ID;
    engine.name = "test:engine";
    engine.heuristicId = HEURISTIC_ID;
    engine.metadataSchemaId = SCHEMA_ID;
    engine.knobs = std::move(knobs);
    return engine;
}

/// An IEngineConfig setting `knobName` to `value`, built the same way
/// TestEngineConfigWrapper.cpp builds one: a real flatbuffer, not a mock, so tests using
/// this exercise the same parsing path a real caller's setAttribute() eventually
/// produces.
inline hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper makeIntKnobEngineConfig(
    flatbuffers::FlatBufferBuilder& builder, const std::string& knobName, int64_t value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::vector<flatbuffers::Offset<KnobSetting>> knobSettings;
    knobSettings.push_back(CreateKnobSettingDirect(
        builder, knobName.c_str(), KnobValue::IntValue, CreateIntValue(builder, value).Union()));
    auto knobsVector = builder.CreateVector(knobSettings);
    builder.Finish(CreateEngineConfig(builder, ENGINE_ID.front(), knobsVector));

    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
        builder.GetBufferPointer(), builder.GetSize());
}

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
