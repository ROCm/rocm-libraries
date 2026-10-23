// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// One pack's contract as the test side sees it: the strings its descriptors carry and
/// its native file implements.
struct PackSymbols
{
    std::string_view engineName;
    std::string_view graphMatcher;
    /// The graph-scoped matcher that admits only this pack's operation, empty for a
    /// single-pack engine whose graph matcher checks the operation itself.
    std::string_view operationMatcher;
    std::string_view kernelMatcher;
    std::string_view score;
    std::string_view dispatch;
    std::string_view inputAToken;
    std::string_view inputBToken;
    std::string_view outputToken;
};

/// The descriptor set this provider ships for @p engineName. Asserting against the
/// loaded set rather than a hand-written twin is what makes these tests fail if the
/// descriptors stop being installed.
inline const hipdnn_plugin_sdk::ingestor::DescriptorSet& loadedSet(std::string_view engineName)
{
    const auto& sets = discoverDescriptorSets();
    const auto match = std::find_if(sets.begin(), sets.end(), [engineName](const auto& set) {
        return set.engine.name == engineName;
    });

    // Fatal rather than a returned optional: every caller would only dereference it.
    if(match == sets.end())
    {
        throw std::runtime_error("no descriptor set loaded for engine '" + std::string(engineName)
                                 + "'");
    }
    return *match;
}

/// How many distinct pack ids @p set holds.
///
/// The packer emits one copy of a pack per architecture. Every copy keeps the authored
/// pack id. Count the ids to get the number of authored packs. That count does not
/// change with the number of architectures.
inline std::size_t distinctPackIdCount(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set)
{
    std::set<hipdnn_plugin_sdk::ingestor::DescriptorId> ids;
    for(const auto& pack : set.packs)
    {
        ids.insert(pack.id);
    }
    return ids.size();
}

/// KMD fields both reference packs vary along. Shared because the *schema* shape is
/// what a pack author copies, unlike the symbol names, which must differ per pack.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
/// The KMD field that discriminates the three Pointwise packs (ADD/MUL/SUB); ConvFwd's
/// KMD declares no such field, since it has only one operation.
constexpr std::string_view OPERATION_FIELD = "operation";

/// A pack's native functions, reached by the symbol name its descriptors carry.
/// Resolving (not calling directly) surfaces a descriptor naming a symbol nothing
/// implements.
inline hipdnn_plugin_sdk::ingestor::GraphMatchFn graphMatcher(const PackSymbols& pack)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(std::string(pack.graphMatcher));
}

inline hipdnn_plugin_sdk::ingestor::KernelMatcherFn kernelMatcher(const PackSymbols& pack)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(pack.kernelMatcher));
}

inline hipdnn_plugin_sdk::ingestor::ScoreFn scorer(const PackSymbols& pack)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(pack.score));
}

inline const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>&
    dispatchHandler(const PackSymbols& pack)
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(pack.dispatch));
    return *handler;
}

/// Runs the engine's graph match: the sole producer of bound tokens. nullopt means the
/// engine does not serve this graph.
inline std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    matchesGraph(const PackSymbols& pack, const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    return graphMatcher(pack)(context);
}

/// Runs the graph-scoped criterion that admits only @p pack's operation.
///
/// Separate from matchesGraph() because the split is the contract: the engine's graph
/// match says "this engine could serve this graph", this one says "this pack is the one".
/// A pack passes only if both do. A criterion reads the tokens the match bound and
/// never writes.
inline bool matchesOperation(const PackSymbols& pack,
                             const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                             const hipdnn_plugin_sdk::ingestor::BoundTokens& bound)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::GraphCriterionRegistry::resolve(
        std::string(pack.operationMatcher))(context, bound);
}

inline bool matchesKernel(const PackSymbols& pack,
                          const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    const auto bound = matchesGraph(pack, context);
    return bound.has_value() && kernelMatcher(pack)(context, *bound, kernel);
}

inline double scoreKernel(const PackSymbols& pack,
                          const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                          const hipdnn_plugin_sdk::ingestor::BoundTokens& bound = {})
{
    return scorer(pack)(context, bound, kernel);
}

/// A fixed, warp-64 device, for CPU-only matcher tests that never compile or launch.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties testDeviceProperties()
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx000";
    properties.warpSize = 64;
    return properties;
}

/// The real current device's properties, queried once; zeroed if no device is current.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties currentDeviceProperties()
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties resolved;
    hipDeviceProp_t properties{};
    int deviceId = 0;
    if(hipGetDevice(&deviceId) == hipSuccess
       && hipGetDeviceProperties(&properties, deviceId) == hipSuccess)
    {
        resolved.gcnArchName = properties.gcnArchName;
        resolved.warpSize = properties.warpSize;
        resolved.multiProcessorCount = properties.multiProcessorCount;
    }
    return resolved;
}

/// Apply changes to a graph fixture, then serialize it.
template <typename Mutate>
inline flatbuffers::FlatBufferBuilder transformGraph(flatbuffers::FlatBufferBuilder builder,
                                                     Mutate mutate)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    const std::unique_ptr<data_objects::GraphT> graph(
        data_objects::GetGraph(builder.GetBufferPointer())->UnPack());
    mutate(*graph);
    builder.Clear();
    builder.Finish(data_objects::Graph::Pack(builder, graph.get()));
    return builder;
}

/// Wraps a built graph buffer so a test reads it the way an engine does.
class GraphFixture
{
public:
    explicit GraphFixture(flatbuffers::FlatBufferBuilder builder,
                          hipdnn_plugin_sdk::ingestor::DeviceProperties properties
                          = testDeviceProperties())
        : _builder(std::move(builder))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(std::move(properties))
    {
    }

    hipdnn_plugin_sdk::ingestor::MatchContext context() const
    {
        return hipdnn_plugin_sdk::ingestor::MatchContext{_graph, 0, _properties};
    }

    const hipdnn_plugin_sdk::ingestor::DeviceProperties& deviceProperties() const
    {
        return _properties;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    hipdnn_plugin_sdk::ingestor::DeviceProperties _properties;
};

/// A KernelDefinition for a reference pack's kernel.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(int64_t blockSize,
                                                                const std::string& dtype,
                                                                const std::string& entryPoint
                                                                = "PointwiseAdd")
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000003");
    // The key the compiled-in source table holds, which is what the staged descriptor of
    // this kernel carries.
    kernel.source.sourceFile = "kernels/" + entryPoint + ".cpp";
    kernel.source.entryPoint = entryPoint;
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

/// A `graph_match.nodes` pattern for one binary pointwise node. The node id matches the
/// native attribute root, so both arms publish the same `pointwise.*` keys. @p swapInputs
/// binds `$input_a` to the node's `in_1` edge and `$input_b` to its `in_0` edge. The
/// registry cannot catch this authoring mistake because both are operands of the same op.
inline nlohmann::json pointwisePattern(bool swapInputs = false)
{
    const auto* inputAEdge = swapInputs ? "in_1" : "in_0";
    const auto* inputBEdge = swapInputs ? "in_0" : "in_1";
    return nlohmann::json::array(
        {{{"kind", "op"},
          {"id", "pointwise"},
          {"op", "pointwise"},
          {"operands", {{inputAEdge, "$input_a"}, {inputBEdge, "$input_b"}}},
          {"results", {{"out_0", "$output"}}}}});
}

/// A `graph_match.nodes` pattern for one forward convolution. The node id matches the
/// native attribute root, so both arms publish the same `conv.*` keys. @p swapXAndW binds
/// `$x` to the node's `w` edge and `$w` to its `x` edge. The registry cannot catch this
/// authoring mistake because both are required operands of the same op.
inline nlohmann::json convPattern(bool swapXAndW = false)
{
    const auto* xEdge = swapXAndW ? "w" : "x";
    const auto* wEdge = swapXAndW ? "x" : "w";
    return nlohmann::json::array({{{"kind", "op"},
                                   {"id", "conv"},
                                   {"op", "convolution"},
                                   {"operands", {{xEdge, "$x"}, {wEdge, "$w"}}},
                                   {"results", {{"y", "$y"}}}}});
}

/// Compiles @p nodes and binds @p context through it, as a UED's declarative arm does.
inline std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    matchPattern(const nlohmann::json& nodes,
                 const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    return hipdnn_plugin_sdk::ingestor::matchGraphPattern(
        *hipdnn_plugin_sdk::ingestor::compileGraphPattern(nodes, "test pattern"), context);
}

/// Asserts @p declarative publishes exactly the tokens @p native does, with the same
/// values, so criteria and dispatch written against the native binding work under either
/// arm.
///
/// A native hook also publishes op-specific heuristic features (shape roots, flop and byte
/// counts) that a pattern cannot derive. Tokens whose name starts with one of
/// @p derivedPrefixes are such features and are left out of the comparison.
inline void expectSameBinding(const hipdnn_plugin_sdk::ingestor::BoundTokens& native,
                              const hipdnn_plugin_sdk::ingestor::BoundTokens& declarative,
                              std::initializer_list<std::string_view> derivedPrefixes = {})
{
    const auto isDerived = [&](const std::string& name) {
        for(const auto prefix : derivedPrefixes)
        {
            if(name.rfind(prefix, 0) == 0)
            {
                return true;
            }
        }
        return false;
    };

    std::set<std::string> nativeNames;
    for(const auto& token : native)
    {
        if(!isDerived(token.first))
        {
            nativeNames.insert(token.first);
        }
    }
    std::set<std::string> declarativeNames;
    for(const auto& token : declarative)
    {
        declarativeNames.insert(token.first);
    }

    ASSERT_EQ(declarativeNames, nativeNames);
    for(const auto& name : nativeNames)
    {
        EXPECT_TRUE(declarative.at(name) == native.at(name)) << name;
    }
}

/// An operand state no pack dispatches from: a virtual tensor has no buffer, and a
/// pass-by-value or scalar-valued tensor carries a host value. Every pack refuses each of
/// these on every operand.
struct UnservableOperandState
{
    const char* name;
    bool isVirtual;
    bool isRuntimePassByValue;
    bool hasScalarValue;
};

inline constexpr std::array<UnservableOperandState, 4> UNSERVABLE_OPERAND_STATES{{
    {"Virtual", true, false, false},
    {"RuntimePassByValue", false, true, false},
    {"ScalarValue", false, false, true},
    {"RuntimePassByValueScalar", false, true, true},
}};

/// Puts @p tensor in @p state.
inline void applyOperandState(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensor,
                              const UnservableOperandState& state)
{
    tensor.virtual_ = state.isVirtual;
    tensor.is_runtime_pass_by_value = state.isRuntimePassByValue;
    if(state.hasScalarValue)
    {
        tensor.value.Set(hipdnn_flatbuffers_sdk::data_objects::Float32Value(2.0f));
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
