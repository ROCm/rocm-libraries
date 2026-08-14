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
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
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
 * @brief Shared, `inline` fixtures for the ingestor's SDK-level tests.
 */
namespace hipdnn_plugin_sdk::ingestor::testing
{

constexpr const char* BLOCK_SIZE = "block_size";
constexpr const char* DTYPE = "dtype";
constexpr const char* GRAPH_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.graph_match";
constexpr const char* KERNEL_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.kernel_match";
constexpr const char* SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.score";

class TestGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    explicit TestGraph(std::optional<GraphId> graphId = std::nullopt,
                       std::optional<hipdnn_data_sdk::utilities::Version> schemaFloor
                       = std::nullopt)
    {
        flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Graph> graph;
        hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion version{};
        if(schemaFloor.has_value())
        {
            version = hipdnn_plugin_sdk::toEngineApiVersion(*schemaFloor);
        }
        const auto* versionPtr = schemaFloor.has_value() ? &version : nullptr;

        if(graphId.has_value())
        {
            const auto uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*graphId);
            auto name = _builder.CreateString("test_graph");
            hipdnn_flatbuffers_sdk::data_objects::GraphBuilder graphBuilder(_builder);
            graphBuilder.add_name(name);
            graphBuilder.add_id(&uuid);
            graphBuilder.add_min_required_engine_api_version(versionPtr);
            graph = graphBuilder.Finish();
        }
        else
        {
            auto name = _builder.CreateString("test_graph");
            hipdnn_flatbuffers_sdk::data_objects::GraphBuilder graphBuilder(_builder);
            graphBuilder.add_name(name);
            graphBuilder.add_min_required_engine_api_version(versionPtr);
            graph = graphBuilder.Finish();
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
    id[6] = static_cast<uint8_t>((id[6] & 0x0fU) | 0x40U);
    id[8] = static_cast<uint8_t>((id[8] & 0x3fU) | 0x80U);
    return id;
}

inline GraphId makeNonV4GraphId(uint8_t seed)
{
    GraphId id{};
    id.fill(seed);
    id[6] = static_cast<uint8_t>(id[6] & 0x0fU);
    return id;
}

inline GraphId makeNilGraphId()
{
    return GraphId{};
}

inline DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx000";
    properties.warpSize = 64;
    return properties;
}

inline bool acceptAnyGraph(const MatchContext& /*context*/, BoundTokens& /*bound*/)
{
    return true;
}

inline bool acceptFloatKernels(const MatchContext& /*context*/, const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(DTYPE) == "FLOAT";
}

inline double scoreByBlockSize(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    return static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
}

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

/// Answers nothing; stands behind the UDD for tests that never dispatch (dispatch
/// symbols resolve at manager construction, so one is always needed).
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
        return nullptr;
    }

    void launch(const THandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

/// Ensures @p symbol resolves so a fixture-built manager constructs; global, idempotent.
/// Never overwrites: a test that installed its own handler under this symbol, via
/// ScopedDispatchRegistration, keeps it.
template <typename THandle>
inline void ensureNoopDispatchRegistered(const std::string& symbol = "test.dispatch")
{
    static const NoopDispatchHandler<THandle> s_handler;
    if(DispatchRegistry<THandle>::tryResolve(symbol) == nullptr)
    {
        DispatchRegistry<THandle>::registerSymbol(symbol, &s_handler);
    }
}

class TestDeviceResolver : public IDeviceResolver<int>
{
public:
    DeviceId deviceId(const int& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = testDeviceProperties();
};

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

inline std::unique_ptr<KernelIngestorStateManager<int>>
    makeTestStateManager(size_t cacheCapacity
                         = KernelIngestorStateManager<int>::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    std::vector<MatchDescriptor> matchers{
        {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL},
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}};
    ensureNoopDispatchRegistered<int>("hipdnn.kernel_ingestor.test.dispatch");
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

constexpr int64_t BOUND_TOKEN_VALUE = 4242;

inline bool acceptGraph(const MatchContext& /*context*/, BoundTokens& bound)
{
    ++counters().graphCalls;
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

constexpr const char* CONSTANT_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.constant_score";

inline double scoreConstant(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 1.0;
}

/// RAII: must outlive the heuristic naming this scorer, which resolves at construction.
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

/// RAII: registers only the block-size scorer, for hand-wired matchers.
class ScopedBlockSizeScore
{
public:
    ScopedBlockSizeScore()
    {
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
    }

    ~ScopedBlockSizeScore()
    {
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
    }

    ScopedBlockSizeScore(const ScopedBlockSizeScore&) = delete;
    ScopedBlockSizeScore& operator=(const ScopedBlockSizeScore&) = delete;
};

/// Registers one graph matcher for the returned object's lifetime. Unregisters even when
/// the test body throws, which a trailing unregisterSymbol() call does not. @p symbol must
/// outlive the guard; every call site passes a string literal.
inline hipdnn_data_sdk::utilities::ScopedResource<const char*>
    scopedGraphMatcher(const char* symbol, GraphMatcherFn matcher)
{
    GraphMatcherRegistry::registerSymbol(symbol, matcher);
    return {symbol,
            [](const char* registered) { GraphMatcherRegistry::unregisterSymbol(registered); }};
}

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

inline KernelDescriptorPack makePack(const std::vector<DescriptorId>& matcherIds,
                                     const std::vector<std::string>& arch = {})
{
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = matcherIds;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.arch = arch;
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

template <typename THandle = int>
inline std::vector<DispatchDescriptor> makeTestDispatches()
{
    ensureNoopDispatchRegistered<THandle>();
    return {{DISPATCH_ID, "test dispatch", "test.dispatch"}};
}

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

/// RAII: registers matchers under caller-supplied names; construct before any state
/// manager naming them, since symbols resolve eagerly.
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
    ensureNoopDispatchRegistered<TestHandle>();
    std::vector<DispatchDescriptor> dispatches{{DISPATCH_ID, "test dispatch", "test.dispatch"}};

    return std::make_unique<StateManager>(
        makeSchema(),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
        std::make_shared<NativeKernelHeuristic>(scoreSymbol),
        cacheCapacity);
}

/// Installs @p handler under @p symbol for the object's lifetime, replacing
/// makeTestDispatches()'s no-op and restoring it after.
template <typename THandle>
class ScopedDispatchRegistration
{
public:
    ScopedDispatchRegistration(std::string symbol, const IKernelDispatchHandler<THandle>& handler)
        : _symbol(std::move(symbol))
        , _previous(DispatchRegistry<THandle>::replaceSymbol(_symbol, &handler))
    {
    }

    ~ScopedDispatchRegistration()
    {
        if(_previous != nullptr)
        {
            static_cast<void>(DispatchRegistry<THandle>::replaceSymbol(_symbol, _previous));
        }
        else
        {
            DispatchRegistry<THandle>::unregisterSymbol(_symbol);
        }
    }

    ScopedDispatchRegistration(const ScopedDispatchRegistration&) = delete;
    ScopedDispatchRegistration& operator=(const ScopedDispatchRegistration&) = delete;

private:
    std::string _symbol;
    const IKernelDispatchHandler<THandle>* _previous = nullptr;
};

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

class StubDeviceResolver : public IDeviceResolver<StubHandle>
{
public:
    DeviceId deviceId(const StubHandle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = testDeviceProperties();
};

class StubWorkspaceHandler : public IKernelDispatchHandler<StubHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(kernel.getIntMetadata(BLOCK_SIZE));
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const StubHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

inline std::vector<DispatchDescriptor> makeStubDispatches()
{
    // Checked per call, not once: a ScopedDispatchRegistration that ran before the first
    // call here restores a null previous entry by unregistering, so a one-shot static
    // would leave every later caller with an unresolvable symbol.
    ensureNoopDispatchRegistered<StubHandle>("hipdnn.kernel_ingestor.test.dispatch");
    return {{DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}};
}

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
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
}

inline EngineDescriptor
    makeEngineWithKnobs(std::vector<std::string> knobs,
                        std::optional<hipdnn_data_sdk::utilities::Version> sdkVersion
                        = std::nullopt)
{
    EngineDescriptor engine;
    engine.id = ENGINE_ID;
    engine.name = "test:engine";
    engine.heuristicId = HEURISTIC_ID;
    engine.metadataSchemaId = SCHEMA_ID;
    engine.knobs = std::move(knobs);
    if(sdkVersion.has_value())
    {
        engine.sdkVersion = *sdkVersion;
    }
    return engine;
}

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

inline hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper makeFloatKnobEngineConfig(
    flatbuffers::FlatBufferBuilder& builder, const std::string& knobName, double value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::vector<flatbuffers::Offset<KnobSetting>> knobSettings;
    knobSettings.push_back(CreateKnobSettingDirect(builder,
                                                   knobName.c_str(),
                                                   KnobValue::FloatValue,
                                                   CreateFloatValue(builder, value).Union()));
    auto knobsVector = builder.CreateVector(knobSettings);
    builder.Finish(CreateEngineConfig(builder, ENGINE_ID.front(), knobsVector));

    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
        builder.GetBufferPointer(), builder.GetSize());
}

inline hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper makeStringKnobEngineConfig(
    flatbuffers::FlatBufferBuilder& builder, const std::string& knobName, const std::string& value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::vector<flatbuffers::Offset<KnobSetting>> knobSettings;
    knobSettings.push_back(
        CreateKnobSettingDirect(builder,
                                knobName.c_str(),
                                KnobValue::StringValue,
                                CreateStringValueDirect(builder, value.c_str()).Union()));
    auto knobsVector = builder.CreateVector(knobSettings);
    builder.Finish(CreateEngineConfig(builder, ENGINE_ID.front(), knobsVector));

    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
        builder.GetBufferPointer(), builder.GetSize());
}

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
