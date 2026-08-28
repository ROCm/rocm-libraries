// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <limits>
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

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView bytes() const override
    {
        return {_builder.GetBufferPointer(), _builder.GetSize()};
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

inline std::optional<BoundTokens> acceptAnyGraph(const MatchContext& /*context*/)
{
    return BoundTokens{};
}

inline bool acceptFloatKernels(const MatchContext& /*context*/,
                               const BoundTokens& /*bound*/,
                               const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(DTYPE) == "FLOAT";
}

inline double scoreByBlockSize(const MatchContext& /*context*/,
                               const BoundTokens& /*bound*/,
                               const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
}

class ScopedTestSymbols
{
public:
    ScopedTestSymbols()
    {
        GraphMatchRegistry::registerSymbol(GRAPH_MATCH_SYMBOL, &acceptAnyGraph);
        KernelMatcherRegistry::registerSymbol(KERNEL_MATCH_SYMBOL, &acceptFloatKernels);
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
    }

    ~ScopedTestSymbols()
    {
        GraphMatchRegistry::unregisterSymbol(GRAPH_MATCH_SYMBOL);
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

/// The minimal handle the state-manager and plan-builder tests pass around. It carries
/// a stream because GenericPlanBuilder requires one of any ingestor handle, and an
/// equality operator so per-handle device resolution can be asserted. Implicitly
/// convertible from int so tests can keep identifying handles by a bare literal.
struct TestHandle
{
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    TestHandle(int handleId = 0)
        : id(handleId)
    {
    }

    int id = 0;

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return nullptr;
    }

    friend bool operator==(const TestHandle& lhs, const TestHandle& rhs)
    {
        return lhs.id == rhs.id;
    }
};

class TestDeviceResolver : public IDeviceResolver<TestHandle>
{
public:
    DeviceId deviceId(const TestHandle& /*handle*/) const override
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

inline std::unique_ptr<KernelIngestorStateManager<TestHandle>>
    makeTestStateManager(size_t cacheCapacity
                         = KernelIngestorStateManager<TestHandle>::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    std::vector<MatchDescriptor> matchers{
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}};
    ensureNoopDispatchRegistered<TestHandle>("hipdnn.kernel_ingestor.test.dispatch");
    std::vector<DispatchDescriptor> dispatches{
        {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = {KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_256_float", 256, "FLOAT"),
                    makeTestKernel(testId(0x66), "kernel_64_half", 64, "HALF")};

    return std::make_unique<KernelIngestorStateManager<TestHandle>>(
        std::move(schema),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        GRAPH_MATCH_SYMBOL,
        "engine 'test fixture'",
        cacheCapacity);
}

struct MatcherCounters
{
    int graphMatchCalls = 0;
    int graphCalls = 0;
    int kernelCalls = 0;

    void reset()
    {
        graphMatchCalls = 0;
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

inline std::optional<BoundTokens> acceptGraph(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    BoundTokens bound;
    bound["test.bound_token"] = BOUND_TOKEN_VALUE;
    return bound;
}

inline std::optional<BoundTokens> rejectGraph(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    return std::nullopt;
}

/// Graph-scoped criteria. These read the tokens the engine's graph match produced and
/// count separately from it, so a test can tell which stage ran.
inline bool acceptCriterion(const MatchContext& /*context*/, const BoundTokens& /*bound*/)
{
    ++counters().graphCalls;
    return true;
}

inline bool rejectCriterion(const MatchContext& /*context*/, const BoundTokens& /*bound*/)
{
    ++counters().graphCalls;
    return false;
}

inline bool countingFloatKernels(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel)
{
    ++counters().kernelCalls;
    return acceptFloatKernels(context, bound, kernel);
}

constexpr const char* CONSTANT_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.constant_score";

inline double scoreConstant(const MatchContext& /*context*/,
                            const BoundTokens& /*bound*/,
                            const KernelDefinition& /*kernel*/)
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

constexpr const char* NAN_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.nan_score";

/// Scores the largest block size NaN and everything else by block size, modeling one
/// pack poisoning the whole ranking: a comparator that mishandles NaN misorders the
/// finite kernels too, not just the NaN-scored one.
inline double scoreNanForLargestBlock(const MatchContext& /*context*/,
                                      const BoundTokens& /*bound*/,
                                      const KernelDefinition& kernel)
{
    return kernel.getIntMetadata(BLOCK_SIZE) == 4096
               ? std::numeric_limits<double>::quiet_NaN()
               : static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
}

/// RAII: must outlive the heuristic naming this scorer, which resolves at construction.
class ScopedNanScore
{
public:
    ScopedNanScore()
    {
        ScoreRegistry::registerSymbol(NAN_SCORE_SYMBOL, &scoreNanForLargestBlock);
    }

    ~ScopedNanScore()
    {
        ScoreRegistry::unregisterSymbol(NAN_SCORE_SYMBOL);
    }

    ScopedNanScore(const ScopedNanScore&) = delete;
    ScopedNanScore& operator=(const ScopedNanScore&) = delete;
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

/// Registers one graph criterion for the returned object's lifetime. Unregisters even
/// when the test body throws, which a trailing unregisterSymbol() call does not. @p
/// symbol must outlive the guard; every call site passes a string literal.
inline hipdnn_data_sdk::utilities::ScopedResource<const char*>
    scopedGraphMatcher(const char* symbol, GraphCriterionFn matcher)
{
    GraphCriterionRegistry::registerSymbol(symbol, matcher);
    return {symbol,
            [](const char* registered) { GraphCriterionRegistry::unregisterSymbol(registered); }};
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

/// The graph-scoped criterion here resolves through GraphCriterionRegistry, so a test
/// using it registers one with scopedGraphMatcher("test.graph_criterion", ...).
inline std::vector<MatchDescriptor> makeTestMatchers()
{
    return {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph_criterion"},
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
}

template <typename THandle = TestHandle>
inline std::vector<DispatchDescriptor> makeTestDispatches()
{
    ensureNoopDispatchRegistered<THandle>();
    return {{DISPATCH_ID, "test dispatch", "test.dispatch"}};
}

inline KernelSource makeEmbeddedSource(const std::string& sourceFile = "Test.cpp",
                                       const std::string& entryPoint = "TestKernel")
{
    KernelSource source;
    source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    source.sourceFile = sourceFile;
    source.entryPoint = entryPoint;
    return source;
}

/// Defaults spell the shape the descriptor packager emits.
inline KernelSource makeKpackSource(const std::string& library
                                    = "kpack/hip_kernel_provider_gfx942.kpack",
                                    const std::string& tocKey = "test-toc-key",
                                    const std::string& symbol = "TestKernel",
                                    const std::string& sha256 = std::string(64, 'a'))
{
    KernelSource source;
    source.kind = KernelSourceKind::KPACK;
    source.library = library;
    source.tocKey = tocKey;
    source.symbol = symbol;
    source.sha256 = sha256;
    return source;
}

inline KernelDefinition makeDefinition(const DescriptorId& id,
                                       int64_t blockSize,
                                       int64_t priority = 0,
                                       const std::vector<std::string>& arch = {})
{
    KernelDefinition definition;
    definition.kernelId = id;
    definition.packId = PACK_ID;
    definition.dispatchId = DISPATCH_ID;
    definition.source = makeEmbeddedSource();
    definition.metadata = {{BLOCK_SIZE, MetadataValue{blockSize}}};
    definition.priority = priority;
    definition.arch = arch;
    return definition;
}

/// @p originDirectory is what `source.library` resolves against; a test that only reads
/// the coordinates can leave it empty.
inline KernelDefinition makeKpackDefinition(const DescriptorId& id,
                                            int64_t blockSize,
                                            const std::filesystem::path& originDirectory = {},
                                            const KernelSource& source = makeKpackSource(),
                                            int64_t priority = 0,
                                            const std::vector<std::string>& arch = {})
{
    KernelDefinition definition = makeDefinition(id, blockSize, priority, arch);
    definition.source = source;
    definition.originDirectory = originDirectory;
    definition.name = "kpack kernel";
    return definition;
}

/// RAII: registers the engine's graph match and one kernel matcher under
/// caller-supplied names; construct before any state manager naming them, since symbols
/// resolve eagerly. Graph criteria are registered separately with scopedGraphMatcher().
class ScopedSymbols
{
public:
    ScopedSymbols(std::string graphSymbol,
                  GraphMatchFn graphFn,
                  std::string kernelSymbol,
                  KernelMatcherFn kernelFn)
        : _graphSymbol(std::move(graphSymbol))
        , _kernelSymbol(std::move(kernelSymbol))
    {
        GraphMatchRegistry::registerSymbol(_graphSymbol, graphFn);
        KernelMatcherRegistry::registerSymbol(_kernelSymbol, kernelFn);
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
        counters().reset();
    }

    ~ScopedSymbols()
    {
        GraphMatchRegistry::unregisterSymbol(_graphSymbol);
        KernelMatcherRegistry::unregisterSymbol(_kernelSymbol);
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
    }

    ScopedSymbols(const ScopedSymbols&) = delete;
    ScopedSymbols& operator=(const ScopedSymbols&) = delete;

private:
    std::string _graphSymbol;
    std::string _kernelSymbol;
};

using StateManager = KernelIngestorStateManager<TestHandle>;

/// The default engine: a graph match plus one kernel-scoped criterion, and no
/// graph-scoped criterion. Tests that exercise graph criteria register their own with
/// scopedGraphMatcher() and list GRAPH_MATCHER_ID in their pack.
inline std::unique_ptr<StateManager>
    makeStateManager(const std::string& scoreSymbol = SCORE_SYMBOL,
                     size_t cacheCapacity = StateManager::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    std::vector<MatchDescriptor> matchers{
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
    ensureNoopDispatchRegistered<TestHandle>();
    std::vector<DispatchDescriptor> dispatches{{DISPATCH_ID, "test dispatch", "test.dispatch"}};

    return std::make_unique<StateManager>(
        makeSchema(),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{makePack({KERNEL_MATCHER_ID})},
        std::make_shared<NativeKernelHeuristic>(scoreSymbol),
        "test.graph",
        "engine 'test fixture'",
        cacheCapacity);
}

/// The same engine as makeStateManager(), but carrying @p engineName so its on-disk
/// winner-cache shard resolves. makeStateManager() leaves the name empty, which
/// disables the disk cache, so every test that does not opt in stays in-memory only.
inline std::unique_ptr<StateManager> makeNamedStateManager(const std::string& engineName)
{
    std::vector<MatchDescriptor> matchers{
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
    ensureNoopDispatchRegistered<TestHandle>();
    std::vector<DispatchDescriptor> dispatches{{DISPATCH_ID, "test dispatch", "test.dispatch"}};

    return std::make_unique<StateManager>(
        makeSchema(),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{makePack({KERNEL_MATCHER_ID})},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph",
        "engine 'test fixture'",
        StateManager::DEFAULT_CATALOG_CACHE_CAPACITY,
        engineName);
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

/// Models a real provider handle: every shipped handle exposes getStream(), and
/// GenericPlanBuilder static_asserts it, since benchmarking times candidates on that
/// stream.
struct StubHandle
{
    void storeEngineDetailsDetachedBuffer(const void* /*ptr*/,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _buffers.push_back(std::move(buffer));
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return nullptr;
    }

private:
    std::vector<std::unique_ptr<flatbuffers::DetachedBuffer>> _buffers;
};

struct StubSettings
{
    IngestorSettings ingestorSettings;
};

struct StubContext
{
    void setExecutionSettings(const StubSettings& settings)
    {
        _settings = settings;
    }

    const StubSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> plan)
    {
        _plan = std::move(plan);
    }

    bool hasPlan() const
    {
        return _plan != nullptr;
    }

private:
    StubSettings _settings;
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
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    return std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        GRAPH_MATCH_SYMBOL);
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
