// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// The pack's contract, restated for the test side exactly as PointwiseAddNative.cpp and
// PointwiseAddDescriptors.cpp each restate it. There is deliberately no shared header:
// after ALMIOPEN-2401 the descriptors are a data file, so a test asserting the pack's
// behaviour has to name these strings itself, the same way an operator authoring a
// descriptor would. A mismatch here fails at resolve() with the descriptor named.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.pointwise_add.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.pointwise_add.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.pointwise_add.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.pointwise_add.dispatch";
constexpr std::string_view ENGINE_NAME = "hipkernel:PointwiseAdd";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";

/// Tokens the graph matcher binds for dispatch to read back.
constexpr std::string_view INPUT_A_TOKEN = "pointwise_add.input_a.uid";
constexpr std::string_view INPUT_B_TOKEN = "pointwise_add.input_b.uid";
constexpr std::string_view OUTPUT_TOKEN = "pointwise_add.output.uid";

/// This pack's native functions, reached the only way anything reaches them now: by
/// the symbol name its descriptors carry. Each registers the provider's packs first,
/// which is idempotent and is exactly what Container's constructor does, so a test
/// needs no fixture and no ordering discipline to reach a matcher.
///
/// Resolving here rather than calling a declared function is also what keeps these
/// tests honest after ALMIOPEN-2401: a descriptor naming a symbol nothing implements
/// is the failure mode the string-valued contract introduces, and it surfaces here.
inline hipdnn_plugin_sdk::ingestor::GraphMatcherFn graphMatcher()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
}

inline hipdnn_plugin_sdk::ingestor::KernelMatcherFn kernelMatcher()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
}

inline hipdnn_plugin_sdk::ingestor::ScoreFn scorer()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
}

/// This pack's dispatch handler, from the registry that owns its process lifetime.
inline const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>& dispatchHandler()
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    return *handler;
}

/// Runs this pack's graph matcher, binding into @p bound.
inline bool matchesGraph(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                         hipdnn_plugin_sdk::ingestor::BoundTokens& bound)
{
    return graphMatcher()(context, bound);
}

/// Runs this pack's kernel-scoped matcher against one candidate.
inline bool matchesKernel(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    return kernelMatcher()(context, kernel);
}

/// Scores one candidate with this pack's scorer.
inline double scoreKernel(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                          const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    return scorer()(kernel, context);
}

/// Tensor uids the builders below use, in argument order.
constexpr int64_t INPUT_A_UID = 1;
constexpr int64_t INPUT_B_UID = 2;
constexpr int64_t OUTPUT_UID = 3;
/// A real third operand, added when `includeThirdOperand` is set.
constexpr int64_t INPUT_C_UID = 4;
/// Uid named by `in_1_tensor_uid` when `danglingInputBUid` is not overridden; guaranteed
/// absent from every graph this file builds.
constexpr int64_t DEFAULT_DANGLING_UID = 999;

/// A fixed, warp-64 device, for CPU-only matcher tests that never compile or launch.
inline hipDeviceProp_t testDeviceProperties()
{
    hipDeviceProp_t properties{};
    properties.warpSize = 64;
    return properties;
}

/// The real current device's properties, queried once; zeroed if no device is current.
inline hipDeviceProp_t currentDeviceProperties()
{
    hipDeviceProp_t properties{};
    int deviceId = 0;
    if(hipGetDevice(&deviceId) == hipSuccess)
    {
        static_cast<void>(hipGetDeviceProperties(&properties, deviceId));
    }
    return properties;
}

/**
 * @brief Builds a single-node binary-pointwise-add graph, parameterized on everything
 *        this pack's matchers gate: operation, dtype (uniform and per-operand), shape,
 *        arity, and operand validity.
 *
 * @param inputBDataType Overrides input B's dtype away from @p dataType.
 * @param includeThirdOperand Adds a real third tensor (`INPUT_C_UID`) and sets
 *        `in_2_tensor_uid`, producing a ternary op.
 * @param danglingInputBUid When set, `in_1_tensor_uid` names this value instead of
 *        `INPUT_B_UID`, and no tensor is inserted for it.
 * @param inputAVirtual Marks input A virtual.
 * @param inputAIsRuntimePassByValue Marks input A runtime-pass-by-value.
 */
inline flatbuffers::FlatBufferBuilder buildPointwiseGraph(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation
    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    const std::vector<int64_t>& dims = {1, 1, 1, 1},
    std::optional<hipdnn_flatbuffers_sdk::utilities::UuidBytes> graphId = std::nullopt,
    bool binary = true,
    const std::optional<std::vector<int64_t>>& explicitStrides = std::nullopt,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> inputBDataType = std::nullopt,
    bool includeThirdOperand = false,
    std::optional<int64_t> danglingInputBUid = std::nullopt,
    bool inputAVirtual = false,
    bool inputAIsRuntimePassByValue = false)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    // Packed strides by default; an explicit set describes a view into a larger buffer.
    const std::vector<int64_t> strides
        = explicitStrides.has_value() ? *explicitStrides : std::vector<int64_t>(dims.size(), 1);
    const auto resolvedInputBDataType = inputBDataType.value_or(dataType);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 INPUT_A_UID,
                                                                 nullptr,
                                                                 dataType,
                                                                 &strides,
                                                                 &dims,
                                                                 inputAVirtual,
                                                                 data_objects::TensorValue::NONE,
                                                                 0,
                                                                 inputAIsRuntimePassByValue));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, INPUT_B_UID, nullptr, resolvedInputBDataType, &strides, &dims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, OUTPUT_UID, nullptr, dataType, &strides, &dims, false));
    if(includeThirdOperand)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, INPUT_C_UID, nullptr, dataType, &strides, &dims, false));
    }

    data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_operation(operation);
    attributesBuilder.add_in_0_tensor_uid(INPUT_A_UID);
    if(binary)
    {
        attributesBuilder.add_in_1_tensor_uid(danglingInputBUid.value_or(INPUT_B_UID));
    }
    if(includeThirdOperand)
    {
        attributesBuilder.add_in_2_tensor_uid(INPUT_C_UID);
    }
    attributesBuilder.add_out_0_tensor_uid(OUTPUT_UID);
    auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "pointwise",
                                       dataType,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("pointwise_add_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);

    // Held for the duration of the GraphBuilder: add_id stores a pointer to it.
    data_objects::Uuid uuid{};
    if(graphId.has_value())
    {
        uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*graphId);
        graphBuilder.add_id(&uuid);
    }
    builder.Finish(graphBuilder.Finish());

    return builder;
}

/// @brief A graph with two pointwise nodes, which no prebuilt single-op kernel serves.
inline flatbuffers::FlatBufferBuilder buildTwoNodePointwiseGraph()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    constexpr int64_t INTERMEDIATE_UID = 4;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto uid : {INPUT_A_UID, INPUT_B_UID, OUTPUT_UID, INTERMEDIATE_UID})
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     uid,
                                                                     nullptr,
                                                                     data_objects::DataType::FLOAT,
                                                                     &strides,
                                                                     &dims,
                                                                     uid == INTERMEDIATE_UID));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(const auto& [in0, in1, out] : {std::tuple{INPUT_A_UID, INPUT_B_UID, INTERMEDIATE_UID},
                                       std::tuple{INTERMEDIATE_UID, INPUT_B_UID, OUTPUT_UID}})
    {
        data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_operation(data_objects::PointwiseMode::ADD);
        attributesBuilder.add_in_0_tensor_uid(in0);
        attributesBuilder.add_in_1_tensor_uid(in1);
        attributesBuilder.add_out_0_tensor_uid(out);
        auto attributes = attributesBuilder.Finish();

        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "pointwise",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           attributes.Union()));
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "two_node_pointwise",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   &tensors,
                                                   &nodes));

    return builder;
}

/// @brief A distinct graph identity, so cache-keyed tests do not collide.
inline hipdnn_flatbuffers_sdk::utilities::UuidBytes makeGraphId(uint8_t seed)
{
    hipdnn_flatbuffers_sdk::utilities::UuidBytes id{};
    id.fill(seed);
    return id;
}

/**
 * @brief Wraps a built graph buffer so a test reads it the way an engine does.
 *
 * Parameterized on the device-properties source: the matcher tests are CPU-only and use
 * a fixed testDeviceProperties(), while the dispatch-handler tests compile through
 * hiprtc and need currentDeviceProperties() for the device they launch on.
 */
class GraphFixture
{
public:
    explicit GraphFixture(flatbuffers::FlatBufferBuilder builder,
                          hipDeviceProp_t properties = testDeviceProperties())
        : _builder(std::move(builder))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(properties)
    {
    }

    hipdnn_plugin_sdk::ingestor::MatchContext context() const
    {
        return hipdnn_plugin_sdk::ingestor::MatchContext{_graph, 0, _properties};
    }

    const hipDeviceProp_t& deviceProperties() const
    {
        return _properties;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    hipDeviceProp_t _properties;
};

/// @brief A KernelDefinition for this pack's kernel, for tests that never need a real
/// descriptor set.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(int64_t blockSize,
                                                                const std::string& dtype)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000003");
    kernel.source.sourceFile = "PointwiseAdd.cpp";
    kernel.source.entryPoint = "PointwiseAdd";
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
