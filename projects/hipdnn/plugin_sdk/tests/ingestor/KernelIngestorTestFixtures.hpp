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
 * @brief Shared fixtures for the ingestor's SDK-level tests under `plugin_sdk/tests/ingestor/`.
 *
 * Everything here is `inline`: this header is included by several translation units in
 * the same test binary.
 *
 * Two catalog shapes recur:
 *  - `int`-handle (`TestHandle`/`StateManager`/`makeStateManager()`): two FLOAT kernels
 *    differing in block size plus one HALF kernel a kernel-scoped matcher prunes, wired
 *    to counting matchers.
 *  - `int`-handle via `makeTestStateManager()`: same catalog, wired to fixed
 *    (non-counting) matchers, for tests needing only a stable answer.
 *  - `StubHandle` (`StubSettings`/`StubContext`/`makeStubStateManager()`): minimal
 *    stand-ins for the provider types `GenericEngine`/`GenericPlanBuilder` are
 *    parameterized on.
 */
namespace hipdnn_plugin_sdk::ingestor::testing
{

constexpr const char* BLOCK_SIZE = "block_size";
constexpr const char* DTYPE = "dtype";
constexpr const char* GRAPH_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.graph_match";
constexpr const char* KERNEL_MATCH_SYMBOL = "hipdnn.kernel_ingestor.test.kernel_match";
constexpr const char* SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.score";

/// Minimal IGraph exposing only identity; every other member throws so a test depending
/// on graph contents fails loudly.
class TestGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    /// @param graphId Identity to carry, or nullopt for a legacy/unfinalized graph.
    /// @param schemaFloor Graph schema version this graph's contents require
    ///        (min_required_engine_api_version), or nullopt to leave it unstamped as a
    ///        writer that never populated the field would.
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

/// Test graph id distinct per seed, shaped as a valid v4 UUID.
inline GraphId makeGraphId(uint8_t seed)
{
    GraphId id{};
    id.fill(seed);
    id[6] = static_cast<uint8_t>((id[6] & 0x0fU) | 0x40U);
    id[8] = static_cast<uint8_t>((id[8] & 0x3fU) | 0x80U);
    return id;
}

/// Distinct-per-seed id shaped to fail the v4 check, for tests needing a non-nil
/// "no identity" id.
inline GraphId makeNonV4GraphId(uint8_t seed)
{
    GraphId id{};
    id.fill(seed);
    id[6] = static_cast<uint8_t>(id[6] & 0x0fU);
    return id;
}

/// The nil UUID: all-zero bytes.
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

/// Accepts only FLOAT kernels; a HALF kernel in the pack is pruned.
inline bool acceptFloatKernels(const MatchContext& /*context*/, const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(DTYPE) == "FLOAT";
}

/// Bigger block size scores higher, giving ranking a defined winner.
inline double scoreByBlockSize(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    return static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
}

/// RAII: registers this fixture's symbols for the object's lifetime, so tests sharing
/// the process-wide registry stay independent.
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

/// A handler that answers nothing, for the many tests that build a state manager but
/// never dispatch through it.
///
/// Needed because dispatch symbols resolve when the manager is constructed: a
/// descriptor naming an unregistered symbol is a load error, which is the whole point,
/// but it means every manager needs *some* handler behind its UDD.
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

/// Ensures "test.dispatch" resolves, so a fixture-built state manager constructs.
///
/// Process-lifetime and idempotent: registration is global, and a test wanting real
/// dispatch behaviour installs its own with ScopedDispatchRegistration, which replaces
/// this for its scope and restores it after.
template <typename THandle>
inline void ensureNoopDispatchRegistered(const std::string& symbol = "test.dispatch")
{
    static const NoopDispatchHandler<THandle> s_handler;
    // replaceSymbol, not registerSymbol: idempotent across repeated calls and across
    // the several symbols the fixtures use, without a per-symbol once flag.
    static_cast<void>(DispatchRegistry<THandle>::replaceSymbol(symbol, &s_handler));
}

/// Device resolver reporting one fixed device; no multi-device behavior is exercised.
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

/// Descriptor id built from a short seed, keeping the real 128-bit type readable.
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

/// Two FLOAT kernels differing in block size, plus a HALF kernel the kernel-scoped
/// matcher prunes: two survivors with a defined ranking.
inline std::unique_ptr<KernelIngestorStateManager<int>>
    makeTestStateManager(size_t cacheCapacity
                         = KernelIngestorStateManager<int>::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    // block_size has a default; dtype is mandatory, so every kernel below sets it.
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    std::vector<MatchDescriptor> matchers{
        {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL},
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}};
    // Same symbol name as the stub fixture, but this manager is over TestHandle, so it
    // needs the TestHandle registry populated; DispatchRegistry is per handle type.
    // int, not TestHandle: that alias is declared further down this file.
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

// Counting matchers: `TestHandle = int` catalog instrumented to count matcher-scope
// calls. Shared by TestKernelIngestorStateManager.cpp and TestGenericPlanBuilder.cpp.

/// Counts matcher calls: graph-scoped runs once per (graph, device); kernel-scoped runs
/// once per surviving kernel.
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

/// Value acceptGraph binds, so a test can verify it reaches dispatch intact.
constexpr int64_t BOUND_TOKEN_VALUE = 4242;

inline bool acceptGraph(const MatchContext& /*context*/, BoundTokens& bound)
{
    ++counters().graphCalls;
    // Stands in for the tensor uids/dimensions a real matcher would resolve.
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

/// RAII: registers the constant scorer. Must outlive the heuristic naming it, which
/// resolves at construction.
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

/// RAII: registers only the block-size scorer, for tests that wire their own matchers
/// by hand but still need a heuristic to be constructible.
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
///
/// @param arch Supported GFX targets; empty (the default) is arch-independent, which
///        is what every test not about the arch gate wants.
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

/// @note Registers the no-op handler as a side effect, so the descriptor this returns
///       always resolves. Call it before constructing the manager that uses it.
template <typename THandle = int>
inline std::vector<DispatchDescriptor> makeTestDispatches()
{
    ensureNoopDispatchRegistered<THandle>();
    return {{DISPATCH_ID, "test dispatch", "test.dispatch"}};
}

/// A catalog entry built directly by ranking and plan tests.
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

/// RAII: registers matchers under caller-supplied symbol names without disturbing the
/// shared fixture's registrations.
///
/// Must be constructed *before* any state manager or heuristic naming these symbols:
/// both resolve eagerly, so a manager built while this is out of scope throws.
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
        // The heuristic resolves at construction, so a manager built under this scope
        // needs the scorer registered before it, not merely before ranking.
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

/// Installs @p handler under @p symbol in DispatchRegistry<THandle> for the object's
/// lifetime, replacing whatever was registered and restoring it afterwards.
///
/// Replace rather than add, because makeTestDispatches() keeps a process-lifetime no-op
/// under "test.dispatch" so a fixture-built manager can construct at all. A test
/// wanting real dispatch behaviour takes the symbol over for its scope and hands it
/// back, so tests stay order-independent.
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

// StubHandle: minimal stand-ins for the types GenericEngine/GenericPlanBuilder are
// parameterized on. Only members those templates touch are present, so a template
// depending on more fails to compile here.

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

/// Every TSettings GenericPlanBuilder is instantiated over must carry this field;
/// initializeExecutionSettings() populates it, getMaxWorkspaceSize() reads it back.
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

    /// True once a plan builder has installed a plan, which nothing else can do.
    bool hasPlan() const
    {
        return _plan != nullptr;
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> _plan;
};

/// Device resolver over StubHandle reporting one fixed device; use IngestorMocks.hpp's
/// MockDeviceResolver for per-handle resolution.
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

/// Reports block_size as its workspace, so a workspace answer is traceable to a
/// specific kernel rather than to a default. The StubHandle counterpart of
/// TestGenericPlanBuilder.cpp's handler of the same shape.
class StubWorkspaceHandler : public IKernelDispatchHandler<StubHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(kernel.getIntMetadata(BLOCK_SIZE));
    }

    /// A real object, not nullptr: GenericPlanBuilder rejects a handler that prepares
    /// no launch, so returning nullptr would fail before a plan reaches the context.
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

/// The StubHandle equivalent of makeTestDispatches().
///
/// DispatchRegistry is per handle type, so registering into the TestHandle one leaves
/// a StubHandle manager unable to resolve. Same side effect, different registry.
inline std::vector<DispatchDescriptor> makeStubDispatches()
{
    static const NoopDispatchHandler<StubHandle> s_handler;
    static const bool s_registered = [] {
        DispatchRegistry<StubHandle>::replaceSymbol("hipdnn.kernel_ingestor.test.dispatch",
                                                    &s_handler);
        return true;
    }();
    static_cast<void>(s_registered);
    return {{DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}};
}

/// State manager over StubHandle: one FLOAT kernel behind one graph-scoped matcher.
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

/// @param sdkVersion Graph schema the engine declares; nullopt leaves
///        EngineDescriptor's baseline default in place.
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

/// IEngineConfig setting `knobName` to an int `value`, built as a real flatbuffer (the
/// same parsing path a real caller's setAttribute() eventually produces).
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

/// FloatValue knob setting, to reach readKnobFilter()'s type-rejection branch (other
/// builders here produce IntValue).
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

/// StringValue counterpart of makeFloatKnobEngineConfig().
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
