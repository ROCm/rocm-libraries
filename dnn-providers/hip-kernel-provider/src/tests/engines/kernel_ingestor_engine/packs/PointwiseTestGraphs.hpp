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
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
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

/// The three packs of the multi-pack engine. Everything but `operationMatcher` is
/// deliberately identical: sharing by id is what the three-pack topology exists to show.
inline constexpr PackSymbols POINTWISE_ADD{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.add_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

inline constexpr PackSymbols POINTWISE_MUL{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.mul_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

inline constexpr PackSymbols POINTWISE_SUB{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.sub_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

/// The second engine, split from Pointwise by graph node type rather than operation.
/// One pack, so `operationMatcher` is empty -- its graph matcher both admits the node
/// type and validates the shape in one pass, unlike the three Pointwise packs above.
inline constexpr PackSymbols CONV_FWD{"hipkernel:ConvFwd",
                                      "hipkernel.conv_fwd.graph_match",
                                      "",
                                      "hipkernel.conv_fwd.kernel_match",
                                      "hipkernel.conv_fwd.score",
                                      "hipkernel.conv_fwd.dispatch",
                                      "conv_fwd.x.uid",
                                      "conv_fwd.w.uid",
                                      "conv_fwd.y.uid"};

/// The descriptor set this provider ships for @p engineName, read from the same files
/// the provider reads. Asserting on a loaded set rather than a hand-written twin is
/// what makes these tests fail if the descriptors stop being installed.
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

/// KMD fields both reference packs vary along. Shared because the *schema* shape is
/// what a pack author copies, unlike the symbol names, which must differ per pack.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";

/// A pack's native functions, reached by the symbol name its descriptors carry.
/// Resolving (not calling directly) surfaces a descriptor naming a symbol nothing
/// implements.
inline hipdnn_plugin_sdk::ingestor::GraphMatcherFn graphMatcher(const PackSymbols& pack)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry::resolve(
        std::string(pack.graphMatcher));
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

inline bool matchesGraph(const PackSymbols& pack,
                         const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                         hipdnn_plugin_sdk::ingestor::BoundTokens& bound)
{
    return graphMatcher(pack)(context, bound);
}

/// Runs the graph-scoped matcher that admits only @p pack's operation.
///
/// Separate from matchesGraph() because the split is the contract: the shared matcher
/// says "this engine could serve this graph", this one says "this pack is the one".
/// A pack passes only if both do.
inline bool matchesOperation(const PackSymbols& pack,
                             const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                             hipdnn_plugin_sdk::ingestor::BoundTokens& bound)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry::resolve(
        std::string(pack.operationMatcher))(context, bound);
}

/// Runs a pack's kernel-scoped matcher against one candidate.
inline bool matchesKernel(const PackSymbols& pack,
                          const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    return kernelMatcher(pack)(context, kernel);
}

inline double scoreKernel(const PackSymbols& pack,
                          const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                          const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    return scorer(pack)(kernel, context);
}

/// Tensor uids the builders below use, in argument order.
constexpr int64_t INPUT_A_UID = 1;
constexpr int64_t INPUT_B_UID = 2;
constexpr int64_t OUTPUT_UID = 3;
/// A real third operand, added when `includeThirdOperand` is set.
constexpr int64_t INPUT_C_UID = 4;
/// Uid named by `in_1_tensor_uid` unless `danglingInputBUid` overrides it; never inserted.
constexpr int64_t DEFAULT_DANGLING_UID = 999;

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

/**
 * @brief Builds a single-node binary-pointwise-add graph, parameterized on everything
 *        this pack's matchers gate.
 *
 * @param includeThirdOperand Adds a real third tensor (`INPUT_C_UID`), producing a
 *        ternary op.
 * @param danglingInputBUid When set, `in_1_tensor_uid` names this value instead of
 *        `INPUT_B_UID`, and no tensor is inserted for it.
 * @param omitStrides Builds every tensor with no strides vector at all: applicability
 *        runs before anything has validated a caller-supplied graph.
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
    bool inputAIsRuntimePassByValue = false,
    bool outputVirtual = false,
    bool omitStrides = false)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    // An explicit set describes a view into a larger buffer.
    const std::vector<int64_t> strides
        = explicitStrides.has_value() ? *explicitStrides : std::vector<int64_t>(dims.size(), 1);
    // Null, not empty: the field is omitted entirely, so strides() returns nullptr.
    const std::vector<int64_t>* const stridesPtr = omitStrides ? nullptr : &strides;
    const auto resolvedInputBDataType = inputBDataType.value_or(dataType);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 INPUT_A_UID,
                                                                 nullptr,
                                                                 dataType,
                                                                 stridesPtr,
                                                                 &dims,
                                                                 inputAVirtual,
                                                                 data_objects::TensorValue::NONE,
                                                                 0,
                                                                 inputAIsRuntimePassByValue));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, INPUT_B_UID, nullptr, resolvedInputBDataType, stridesPtr, &dims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, OUTPUT_UID, nullptr, dataType, stridesPtr, &dims, outputVirtual));
    if(includeThirdOperand)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, INPUT_C_UID, nullptr, dataType, stridesPtr, &dims, false));
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

/// @brief Row-major packed strides for @p dims -- the layout the conv kernel's flat
/// index arithmetic assumes, since it takes no stride arguments of its own.
inline std::vector<int64_t> packedRowMajorStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// Tensor uids buildConvFwdGraph() uses, in kernel argument order.
constexpr int64_t CONV_X_UID = 1;
constexpr int64_t CONV_W_UID = 2;
constexpr int64_t CONV_Y_UID = 3;

/**
 * @brief Builds a single-node conv-forward graph, parameterized on everything this
 *        pack's matcher gates: mode, stride, dilation, padding, and dtype.
 *
 * Defaults to the one shape the engine's naive kernel can serve: unit stride and
 * dilation, no padding, cross-correlation, packed NCHW/KCRS/NKPQ, uniform dtype.
 * @p wDims and @p yDims default from @p xDims assuming a 1-in/1-out-channel, 2x2
 * kernel with no padding (P = H - R + 1, Q = W - S + 1), so a caller overriding only
 * @p xDims for a refusal case does not also have to keep w/y consistent by hand.
 *
 * @param wDataType Overrides w's dtype away from @p dataType, for the cross-operand
 *        dtype-mismatch refusal.
 * @param xStridesOverride Overrides x's strides away from packed row-major, for the
 *        non-packed-layout refusal; the kernel takes no strides of its own.
 */
inline flatbuffers::FlatBufferBuilder
    buildConvFwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                      = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                      hipdnn_flatbuffers_sdk::data_objects::ConvMode convMode
                      = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION,
                      const std::vector<int64_t>& stride = {1, 1},
                      const std::vector<int64_t>& dilation = {1, 1},
                      const std::vector<int64_t>& prePadding = {0, 0},
                      const std::vector<int64_t>& postPadding = {0, 0},
                      const std::vector<int64_t>& xDims = {1, 1, 3, 3},
                      const std::optional<std::vector<int64_t>>& wDims = std::nullopt,
                      const std::optional<std::vector<int64_t>>& yDims = std::nullopt,
                      std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> wDataType
                      = std::nullopt,
                      const std::optional<std::vector<int64_t>>& xStridesOverride = std::nullopt)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const auto resolvedWDims = wDims.value_or(std::vector<int64_t>{1, xDims[1], 2, 2});
    const auto resolvedYDims
        = yDims.value_or(std::vector<int64_t>{xDims[0],
                                              resolvedWDims[0],
                                              xDims[2] - resolvedWDims[2] + 1,
                                              xDims[3] - resolvedWDims[3] + 1});
    const auto resolvedWDataType = wDataType.value_or(dataType);

    const auto xStrides = xStridesOverride.value_or(packedRowMajorStrides(xDims));
    const auto wStrides = packedRowMajorStrides(resolvedWDims);
    const auto yStrides = packedRowMajorStrides(resolvedYDims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_X_UID, nullptr, dataType, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_W_UID, nullptr, resolvedWDataType, &wStrides, &resolvedWDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_Y_UID, nullptr, dataType, &yStrides, &resolvedYDims));

    auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                         CONV_X_UID,
                                                                         CONV_W_UID,
                                                                         CONV_Y_UID,
                                                                         &prePadding,
                                                                         &postPadding,
                                                                         &stride,
                                                                         &dilation,
                                                                         convMode);

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "conv_fwd",
                                       dataType,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("conv_fwd_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());

    return builder;
}

/// @brief A distinct graph identity, so cache-keyed tests do not collide.
inline hipdnn_flatbuffers_sdk::utilities::UuidBytes makeGraphId(uint8_t seed)
{
    hipdnn_flatbuffers_sdk::utilities::UuidBytes id{};
    id.fill(seed);
    return id;
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
    kernel.source.sourceFile = entryPoint + ".cpp";
    kernel.source.entryPoint = entryPoint;
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
