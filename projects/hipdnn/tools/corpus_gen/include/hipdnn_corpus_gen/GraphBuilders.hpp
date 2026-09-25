// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <string>
#include <optional>
#include <vector>

/// @file GraphBuilders.hpp
/// @brief Constructing a graph from an explicit description of a problem.
///
/// These exist because the test SDK's `FlatbufferGraphTestUtils.hpp` builders answer a
/// different question. Theirs is "give me a valid graph of this operation", and they answer it
/// well: 102 of their 135 call sites pass no arguments at all and take a default shape --
/// a 4x4x4x4 convolution with a 1x1 filter, a 4x8 by 8x5 matmul. Those are the smallest shapes
/// that exercise a code path, and the parameters exist so the minority of tests that care can
/// override them.
///
/// Corpus generation asks something stricter: give me *exactly* the graph this problem
/// describes. Under that contract those builders are unfit in ways that are invisible under
/// theirs -- LayerNorm and RMSNorm accept `inputDataType` and `computeDataType` and ignore
/// them for the graph header, BatchNorm has no dtype argument at all, Reduction had no
/// parameters, MoE takes only a mode. None of that is a defect in a fixture. All of it
/// silently mislabels a corpus row, which records the parameters that were *asked for* rather
/// than the ones that reached the hardware.
///
/// So the contract here is:
///
///  - **No defaults.** Every value a graph depends on is a parameter. A builder that can be
///    called with no arguments will be, and the shape it invents will end up in a corpus.
///  - **Every parameter reaches the graph.** Enforced by test: changing any argument must
///    change the emitted bytes. That is the check that would have caught the ignored dtypes.
///  - **Nothing is hardcoded that a problem might vary**, including the graph-level dtypes,
///    which is where the LayerNorm fixture goes wrong.
///
/// The cost is duplicated construction. The benefit is that a corpus's correctness stops
/// depending on a file whose purpose is something else, and which is edited freely for reasons
/// that have nothing to do with us.
namespace hipdnn_corpus_gen::builders
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// Serialized graph bytes.
using GraphBytes = std::vector<uint8_t>;

/// One tensor of a problem. Strides are explicit rather than derived, because a layout is part
/// of a problem: an engine may serve NCHW and refuse the same extents in NHWC.
struct TensorSpec
{
    int64_t uid = 0;
    std::string name;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    fb::DataType dataType = fb::DataType::FLOAT;

    /// An intermediate of a multi-node graph: produced by one node and consumed by the next,
    /// never allocated by the caller.
    bool isVirtual = false;

    /// Set for a pass-by-value tensor -- a scalar such as a norm's epsilon, whose value is part
    /// of the graph. The frontend refuses one given as an ordinary tensor.
    std::optional<float> scalarValue;
};

/// The dtypes a graph declares. Three fields because the schema has three, and conflating them
/// is exactly the LayerNorm fixture's error.
struct GraphTypes
{
    fb::DataType io = fb::DataType::FLOAT;
    fb::DataType intermediate = fb::DataType::FLOAT;
    fb::DataType compute = fb::DataType::FLOAT;

    /// The common case: one element type throughout.
    static GraphTypes uniform(fb::DataType type)
    {
        return {type, type, type};
    }
};

namespace detail
{

inline flatbuffers::Offset<fb::TensorAttributes> addTensor(flatbuffers::FlatBufferBuilder& builder,
                                                           const TensorSpec& tensor)
{
    if(tensor.scalarValue.has_value())
    {
        const fb::Float32Value value(*tensor.scalarValue);
        return fb::CreateTensorAttributesDirect(builder,
                                                tensor.uid,
                                                tensor.name.c_str(),
                                                tensor.dataType,
                                                &tensor.strides,
                                                &tensor.dims,
                                                tensor.isVirtual,
                                                fb::TensorValue::Float32Value,
                                                builder.CreateStruct(value).Union());
    }
    return fb::CreateTensorAttributesDirect(builder,
                                            tensor.uid,
                                            tensor.name.c_str(),
                                            tensor.dataType,
                                            &tensor.strides,
                                            &tensor.dims,
                                            tensor.isVirtual);
}

inline GraphBytes finish(flatbuffers::FlatBufferBuilder& builder,
                         const std::string& name,
                         const GraphTypes& types,
                         std::vector<flatbuffers::Offset<fb::TensorAttributes>>& tensors,
                         std::vector<flatbuffers::Offset<fb::Node>>& nodes)
{
    // Named rather than positional on purpose. The generated signature is
    // (name, compute, intermediate, io) -- not the io-first order the struct lists -- and while
    // every graph used one type throughout, passing them the wrong way round produced byte
    // identical output. It stayed wrong until a declaration asked for fp16 operands with fp32
    // accumulate, which is the ordinary mixed-precision case.
    const auto graph = fb::CreateGraphDirect(builder,
                                             name.c_str(),
                                             /*compute_data_type=*/types.compute,
                                             /*intermediate_data_type=*/types.intermediate,
                                             /*io_data_type=*/types.io,
                                             &tensors,
                                             &nodes);
    builder.Finish(graph);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

} // namespace detail

/// Spatial parameters shared by the three convolution directions.
struct ConvGeometry
{
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    fb::ConvMode mode = fb::ConvMode::CROSS_CORRELATION;
};

/// @brief Forward convolution.
///
/// Group count is not a parameter here because it is not one in the schema: the frontend
/// derives it as `x.dims[1] / w.dims[1]`, so a depthwise convolution is expressed by giving
/// the weight tensor one input channel. A `groups` argument would be a second way to say the
/// same thing, and the two could disagree.
inline GraphBytes convolutionForward(const TensorSpec& x,
                                     const TensorSpec& w,
                                     const TensorSpec& y,
                                     const ConvGeometry& geometry,
                                     const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, w),
        detail::addTensor(builder, y)};

    const auto attributes = fb::CreateConvolutionFwdAttributesDirect(builder,
                                                                     x.uid,
                                                                     w.uid,
                                                                     y.uid,
                                                                     &geometry.prePadding,
                                                                     &geometry.postPadding,
                                                                     &geometry.stride,
                                                                     &geometry.dilation,
                                                                     geometry.mode);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv_fwd",
                             types.compute,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             attributes.Union())};
    return detail::finish(builder, "conv_fwd", types, tensors, nodes);
}

/// @brief Convolution, optional bias add, activation: the fusion MIOpen runs as one plan.
///
/// Three nodes when @p bias is given (conv -> ADD bias -> activation), two otherwise. The
/// intermediates are virtual fp32 tensors, and the convolution and activation compute in fp32
/// while the bias add computes in the bias tensor's type -- the contract MIOpen's
/// ConvFwdBiasActiv builder checks node by node.
inline GraphBytes convolutionBiasActivation(const TensorSpec& x,
                                            const TensorSpec& w,
                                            const std::optional<TensorSpec>& bias,
                                            const TensorSpec& y,
                                            const ConvGeometry& geometry,
                                            fb::PointwiseMode activation,
                                            const GraphTypes& types)
{
    constexpr int64_t CONV_OUT_UID = 101;
    constexpr int64_t BIAS_OUT_UID = 102;
    TensorSpec convOut = y;
    convOut.uid = CONV_OUT_UID;
    convOut.name = "conv_out";
    convOut.dataType = fb::DataType::FLOAT;
    convOut.isVirtual = true;
    TensorSpec biasOut = convOut;
    biasOut.uid = BIAS_OUT_UID;
    biasOut.name = "bias_out";

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, w),
        detail::addTensor(builder, convOut), detail::addTensor(builder, y)};
    if(bias.has_value())
    {
        tensors.push_back(detail::addTensor(builder, *bias));
        tensors.push_back(detail::addTensor(builder, biasOut));
    }

    std::vector<flatbuffers::Offset<fb::Node>> nodes;
    const auto conv = fb::CreateConvolutionFwdAttributesDirect(builder,
                                                               x.uid,
                                                               w.uid,
                                                               convOut.uid,
                                                               &geometry.prePadding,
                                                               &geometry.postPadding,
                                                               &geometry.stride,
                                                               &geometry.dilation,
                                                               geometry.mode);
    nodes.push_back(fb::CreateNodeDirect(builder, "conv_fwd", fb::DataType::FLOAT,
                                         fb::NodeAttributes::ConvolutionFwdAttributes,
                                         conv.Union()));
    int64_t activationIn = convOut.uid;
    if(bias.has_value())
    {
        const auto add = fb::CreatePointwiseAttributes(builder,
                                                       fb::PointwiseMode::ADD,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt, // axis
                                                       convOut.uid,
                                                       bias->uid,
                                                       flatbuffers::nullopt, // in_2
                                                       biasOut.uid);
        nodes.push_back(fb::CreateNodeDirect(builder, "bias", bias->dataType,
                                             fb::NodeAttributes::PointwiseAttributes,
                                             add.Union()));
        activationIn = biasOut.uid;
    }
    const auto activate = fb::CreatePointwiseAttributes(builder,
                                                        activation,
                                                        flatbuffers::nullopt,
                                                        flatbuffers::nullopt,
                                                        flatbuffers::nullopt,
                                                        flatbuffers::nullopt, // axis
                                                        activationIn,
                                                        flatbuffers::nullopt, // in_1
                                                        flatbuffers::nullopt, // in_2
                                                        y.uid);
    nodes.push_back(fb::CreateNodeDirect(builder, "activation", fb::DataType::FLOAT,
                                         fb::NodeAttributes::PointwiseAttributes,
                                         activate.Union()));
    return detail::finish(builder, "conv_bias_activation", types, tensors, nodes);
}

/// @brief Convolution data gradient. dx's extents are a parameter because they cannot be
///        derived: several inputs give the same output under a stride.
inline GraphBytes convolutionBackwardData(const TensorSpec& dy,
                                          const TensorSpec& w,
                                          const TensorSpec& dx,
                                          const ConvGeometry& geometry,
                                          const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy), detail::addTensor(builder, w),
        detail::addTensor(builder, dx)};

    const auto attributes = fb::CreateConvolutionBwdAttributesDirect(builder,
                                                                     dy.uid,
                                                                     w.uid,
                                                                     dx.uid,
                                                                     &geometry.prePadding,
                                                                     &geometry.postPadding,
                                                                     &geometry.stride,
                                                                     &geometry.dilation,
                                                                     geometry.mode);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv_dgrad",
                             types.compute,
                             fb::NodeAttributes::ConvolutionBwdAttributes,
                             attributes.Union())};
    return detail::finish(builder, "conv_dgrad", types, tensors, nodes);
}

/// @brief Convolution weight gradient.
inline GraphBytes convolutionBackwardWeights(const TensorSpec& x,
                                             const TensorSpec& dy,
                                             const TensorSpec& dw,
                                             const ConvGeometry& geometry,
                                             const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, dy),
        detail::addTensor(builder, dw)};

    const auto attributes = fb::CreateConvolutionWrwAttributesDirect(builder,
                                                                     x.uid,
                                                                     dy.uid,
                                                                     dw.uid,
                                                                     &geometry.prePadding,
                                                                     &geometry.postPadding,
                                                                     &geometry.stride,
                                                                     &geometry.dilation,
                                                                     geometry.mode);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv_wgrad",
                             types.compute,
                             fb::NodeAttributes::ConvolutionWrwAttributes,
                             attributes.Union())};
    return detail::finish(builder, "conv_wgrad", types, tensors, nodes);
}

/// @brief Matrix multiply, C = A x B.
inline GraphBytes matmul(const TensorSpec& a,
                         const TensorSpec& b,
                         const TensorSpec& c,
                         const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, a), detail::addTensor(builder, b),
        detail::addTensor(builder, c)};

    const auto attributes = fb::CreateMatmulAttributes(builder, a.uid, b.uid, c.uid);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "matmul", types.compute, fb::NodeAttributes::MatmulAttributes,
        attributes.Union())};
    return detail::finish(builder, "matmul", types, tensors, nodes);
}

/// @brief A matmul followed by an epilogue: a bias add, an activation, or both in that order.
///
/// Two or three nodes. The matmul's output and the biased intermediate are virtual fp32
/// tensors, and every node computes in fp32. @p bias, when given, is added to the matmul's
/// output; @p activation, when given, is applied last. At least one is required -- with
/// neither this is just `matmul`.
inline GraphBytes matmulEpilogue(const TensorSpec& a,
                                 const TensorSpec& b,
                                 const std::optional<TensorSpec>& bias,
                                 const std::optional<fb::PointwiseMode>& activation,
                                 const TensorSpec& c,
                                 const GraphTypes& types)
{
    constexpr int64_t MATMUL_OUT_UID = 101;
    constexpr int64_t BIAS_OUT_UID = 102;
    const auto intermediate = [&c](int64_t uid, const char* name) {
        TensorSpec spec = c;
        spec.uid = uid;
        spec.name = name;
        spec.dataType = fb::DataType::FLOAT;
        spec.isVirtual = true;
        return spec;
    };
    // The last node writes c; every earlier output is virtual.
    const auto matmulOut = (bias || activation) ? intermediate(MATMUL_OUT_UID, "matmul_out") : c;
    const auto biasOut = activation ? intermediate(BIAS_OUT_UID, "bias_out") : c;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, a), detail::addTensor(builder, b), detail::addTensor(builder, c)};
    if(bias || activation)
    {
        tensors.push_back(detail::addTensor(builder, matmulOut));
    }
    if(bias)
    {
        tensors.push_back(detail::addTensor(builder, *bias));
        if(activation)
        {
            tensors.push_back(detail::addTensor(builder, biasOut));
        }
    }

    std::vector<flatbuffers::Offset<fb::Node>> nodes;
    const auto mm = fb::CreateMatmulAttributes(builder, a.uid, b.uid, matmulOut.uid);
    nodes.push_back(fb::CreateNodeDirect(builder, "matmul", fb::DataType::FLOAT,
                                         fb::NodeAttributes::MatmulAttributes, mm.Union()));
    int64_t next = matmulOut.uid;
    if(bias)
    {
        const auto add = fb::CreatePointwiseAttributes(builder,
                                                       fb::PointwiseMode::ADD,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt, // axis
                                                       matmulOut.uid,
                                                       bias->uid,
                                                       flatbuffers::nullopt, // in_2
                                                       biasOut.uid);
        nodes.push_back(fb::CreateNodeDirect(builder, "bias", fb::DataType::FLOAT,
                                             fb::NodeAttributes::PointwiseAttributes,
                                             add.Union()));
        next = biasOut.uid;
    }
    if(activation)
    {
        const auto act = fb::CreatePointwiseAttributes(builder,
                                                       *activation,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt, // axis
                                                       next,
                                                       flatbuffers::nullopt, // in_1
                                                       flatbuffers::nullopt, // in_2
                                                       c.uid);
        nodes.push_back(fb::CreateNodeDirect(builder, "activation", fb::DataType::FLOAT,
                                             fb::NodeAttributes::PointwiseAttributes,
                                             act.Union()));
    }
    return detail::finish(builder, "matmul_epilogue", types, tensors, nodes);
}

/// @brief C = dequantize(A, scaleA) x dequantize(B, scaleB): a block-scaled (MX) matmul.
///
/// Three nodes, as the graph states it: each operand is dequantized by its per-block scale into
/// a virtual fp32 tensor, and the matmul consumes those. Every node computes in fp32. Whether an
/// engine can run it -- MX formats are a property of the device -- is the engine's answer; the
/// graph is the same on every architecture.
inline GraphBytes blockScaledMatmul(const TensorSpec& a,
                                    const TensorSpec& aScale,
                                    const TensorSpec& b,
                                    const TensorSpec& bScale,
                                    const TensorSpec& c,
                                    const std::vector<int32_t>& blockSize,
                                    const GraphTypes& types)
{
    constexpr int64_t A_DEQUANTIZED_UID = 101;
    constexpr int64_t B_DEQUANTIZED_UID = 102;
    const auto dequantized = [](const TensorSpec& operand, int64_t uid, const char* name) {
        TensorSpec spec = operand;
        spec.uid = uid;
        spec.name = name;
        spec.dataType = fb::DataType::FLOAT;
        spec.isVirtual = true;
        return spec;
    };
    const auto aDeq = dequantized(a, A_DEQUANTIZED_UID, "a_dequantized");
    const auto bDeq = dequantized(b, B_DEQUANTIZED_UID, "b_dequantized");

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, a),    detail::addTensor(builder, aScale),
        detail::addTensor(builder, b),    detail::addTensor(builder, bScale),
        detail::addTensor(builder, aDeq), detail::addTensor(builder, bDeq),
        detail::addTensor(builder, c)};

    std::vector<flatbuffers::Offset<fb::Node>> nodes;
    for(const auto* operand : {&a, &b})
    {
        const auto& scale = operand == &a ? aScale : bScale;
        const auto& out = operand == &a ? aDeq : bDeq;
        const auto deq = fb::CreateBlockScaleDequantizeAttributesDirect(
            builder, operand->uid, scale.uid, out.uid, &blockSize, /*is_negative_scale=*/false);
        nodes.push_back(fb::CreateNodeDirect(builder, "block_scale_dequantize",
                                             fb::DataType::FLOAT,
                                             fb::NodeAttributes::BlockScaleDequantizeAttributes,
                                             deq.Union()));
    }
    const auto mm = fb::CreateMatmulAttributes(builder, aDeq.uid, bDeq.uid, c.uid);
    nodes.push_back(fb::CreateNodeDirect(builder, "matmul", fb::DataType::FLOAT,
                                         fb::NodeAttributes::MatmulAttributes, mm.Union()));
    return detail::finish(builder, "block_scaled_matmul", types, tensors, nodes);
}

/// @brief Binary elementwise pointwise.
///
/// The optional tensor uids are left null rather than zero. The schema declares them
/// `= null`, so a literal 0 references tensor uid 0 -- a tensor a binary pointwise does not
/// carry -- and the graph then fails to deserialize rather than simply describing something
/// unusual.
/// Mode-specific scalars. §12.6 lists them as parameters of a pointwise problem, and they are:
/// a ReLU with a non-zero lower-clip slope is a leaky ReLU and a different kernel.
/// Written only when set. The schema leaves each of these null by default, and a present value
/// changes the operation: a relu_upper_clip of 0 makes ReLU a clamp to [0, 0], which MIOpen
/// accepts as a clamp and computes as one.
struct PointwiseScalars
{
    flatbuffers::Optional<float> reluLowerClip = flatbuffers::nullopt;
    flatbuffers::Optional<float> reluUpperClip = flatbuffers::nullopt;
    flatbuffers::Optional<float> reluLowerClipSlope = flatbuffers::nullopt;
    flatbuffers::Optional<float> swishBeta = flatbuffers::nullopt;
    flatbuffers::Optional<float> eluAlpha = flatbuffers::nullopt;
    flatbuffers::Optional<float> softplusBeta = flatbuffers::nullopt;
};

inline GraphBytes pointwiseBinary(const TensorSpec& inA,
                                  const TensorSpec& inB,
                                  const TensorSpec& out,
                                  fb::PointwiseMode mode,
                                  const PointwiseScalars& scalars,
                                  const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, inA), detail::addTensor(builder, inB),
        detail::addTensor(builder, out)};

    const auto attributes = fb::CreatePointwiseAttributes(builder,
                                                          mode,
                                                          scalars.reluLowerClip,
                                                          scalars.reluUpperClip,
                                                          scalars.reluLowerClipSlope,
                                                          flatbuffers::nullopt, // axis
                                                          inA.uid,
                                                          inB.uid,
                                                          flatbuffers::nullopt, // in_2
                                                          out.uid,
                                                          scalars.swishBeta,
                                                          scalars.eluAlpha,
                                                          scalars.softplusBeta);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "pointwise", types.compute, fb::NodeAttributes::PointwiseAttributes,
        attributes.Union())};
    return detail::finish(builder, "pointwise", types, tensors, nodes);
}

/// @brief One operand in, one result out: an activation or a unary math function.
///
/// The same node as pointwiseBinary with in_1 left null. Engines that run activations check
/// for exactly that -- MIOpen's activation builder takes a single-input pointwise node.
inline GraphBytes pointwiseUnary(const TensorSpec& in,
                                 const TensorSpec& out,
                                 fb::PointwiseMode mode,
                                 const PointwiseScalars& scalars,
                                 const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, in), detail::addTensor(builder, out)};

    const auto attributes = fb::CreatePointwiseAttributes(builder,
                                                          mode,
                                                          scalars.reluLowerClip,
                                                          scalars.reluUpperClip,
                                                          scalars.reluLowerClipSlope,
                                                          flatbuffers::nullopt, // axis
                                                          in.uid,
                                                          flatbuffers::nullopt, // in_1
                                                          flatbuffers::nullopt, // in_2
                                                          out.uid,
                                                          scalars.swishBeta,
                                                          scalars.eluAlpha,
                                                          scalars.softplusBeta);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "pointwise", types.compute, fb::NodeAttributes::PointwiseAttributes,
        attributes.Union())};
    return detail::finish(builder, "pointwise_unary", types, tensors, nodes);
}

/// @brief Reduction. The output extents are a parameter because they *are* the statement of
///        which axes reduce; nothing infers them.
inline GraphBytes reduction(const TensorSpec& in,
                            const TensorSpec& out,
                            fb::ReductionMode mode,
                            bool deterministic,
                            const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, in), detail::addTensor(builder, out)};

    const auto attributes
        = fb::CreateReductionAttributes(builder, mode, in.uid, out.uid, deterministic);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "reduction", types.compute, fb::NodeAttributes::ReductionAttributes,
        attributes.Union())};
    return detail::finish(builder, "reduction", types, tensors, nodes);
}

/// @brief LayerNorm forward.
///
/// The graph dtypes come from @p types like everything else. The test-SDK equivalent accepts
/// input and compute types and then writes io=FLOAT, intermediate=HALF, compute=BFLOAT16
/// regardless, which is why graphs built from it could not be deserialized whatever the
/// declaration asked for.
inline GraphBytes layernormForward(const TensorSpec& x,
                                   const TensorSpec& scale,
                                   const TensorSpec& bias,
                                   const TensorSpec& epsilon,
                                   const TensorSpec& y,
                                   int64_t normalizedDimCount,
                                   fb::NormFwdPhase phase,
                                   const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, scale),
        detail::addTensor(builder, bias), detail::addTensor(builder, epsilon),
        detail::addTensor(builder, y)};

    const auto attributes = fb::CreateLayernormAttributes(builder,
                                                          x.uid,
                                                          scale.uid,
                                                          bias.uid,
                                                          epsilon.uid,
                                                          y.uid,
                                                          normalizedDimCount,
                                                          flatbuffers::nullopt, // mean
                                                          flatbuffers::nullopt, // inv_variance
                                                          phase);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "layernorm", types.compute, fb::NodeAttributes::LayernormAttributes,
        attributes.Union())};
    return detail::finish(builder, "layernorm_fwd", types, tensors, nodes);
}

/// @brief RMSNorm forward. Bias is optional in the schema and omitted here; a declaration that
///        needs it wants a separate entry rather than a flag, since it changes the tensor set.
inline GraphBytes rmsNormForward(const TensorSpec& x,
                                 const TensorSpec& scale,
                                 const TensorSpec& epsilon,
                                 const TensorSpec& y,
                                 fb::NormFwdPhase phase,
                                 const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, scale),
        detail::addTensor(builder, epsilon), detail::addTensor(builder, y)};

    const auto attributes = fb::CreateRMSNormAttributes(builder,
                                                        x.uid,
                                                        scale.uid,
                                                        epsilon.uid,
                                                        y.uid,
                                                        flatbuffers::nullopt, // bias
                                                        flatbuffers::nullopt, // inv_rms
                                                        phase);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "rmsnorm", types.compute, fb::NodeAttributes::RMSNormAttributes,
        attributes.Union())};
    return detail::finish(builder, "rmsnorm_fwd", types, tensors, nodes);
}


// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Optional behaviour of an attention problem. These are part of the problem, not of the
/// kernel: a causal attention and a full one do different work and are served by different
/// kernels, so a corpus that fixed them would be a corpus of one regime.
struct SdpaOptions
{
    bool causalMask = false;
    bool paddingMask = false;
    bool alibiMask = false;
    bool generateStats = false;

    /// Softmax scale. Part of the problem: a kernel may fold a known scale into its epilogue.
    float attnScale = 0.0F;

    /// Dropout rate. Non-zero changes the kernel: an RNG and a mask are generated.
    float dropoutProbability = 0.0F;

    /// Sliding-window attention, as rocKE's shape files carry it. -1 means unbounded, which is
    /// full attention; a finite bound is a different kernel with different work per query.
    int64_t leftBound = -1;
    int64_t rightBound = -1;

    /// Which corner the causal diagonal is anchored at. Only meaningful under a causal mask,
    /// and then it is not a detail: at seqlen_q < seqlen_k the two anchors mask different
    /// triangles, so they are different work -- and an engine that serves one anchor and
    /// refuses the other is removed from a comparison by the corpus rather than by a
    /// measurement. Defaults to the schema's default so an unset option writes what a graph
    /// built before this field did.
    fb::DiagonalAlignment diagonalAlignment = fb::DiagonalAlignment::TOP_LEFT;
};

/// @brief Scaled dot-product attention, forward.
///
/// SdpaAttributes declares twenty-eight optional tensor uids -- paged KV, dropout, descale
/// factors, sinks. All are left null here. Each is a different problem rather than a variation
/// on this one, and giving them a uid they do not have is how a graph stops deserializing.
///
/// With `generateStats` -- the training forward -- @p stats receives the softmax statistics the
/// backward pass consumes; the flag without the tensor is a graph no engine can run.
inline GraphBytes sdpaForward(const TensorSpec& q,
                              const TensorSpec& k,
                              const TensorSpec& v,
                              const TensorSpec& o,
                              const SdpaOptions& options,
                              const GraphTypes& types,
                              const std::optional<TensorSpec>& stats = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, q), detail::addTensor(builder, k),
        detail::addTensor(builder, v), detail::addTensor(builder, o)};
    const bool withStats = options.generateStats && stats.has_value();
    if(withStats)
    {
        tensors.push_back(detail::addTensor(builder, *stats));
    }

    fb::SdpaAttributesBuilder attributes(builder);
    attributes.add_q_tensor_uid(q.uid);
    attributes.add_k_tensor_uid(k.uid);
    attributes.add_v_tensor_uid(v.uid);
    attributes.add_o_tensor_uid(o.uid);
    // Causality is written as the bounds it means -- right_bound 0, left unbounded -- with the
    // anchor in diagonal_alignment, and the deprecated causal_mask flag left false. The flag is
    // not a synonym: providers give it precedence over the bounds and read it as TOP-LEFT
    // whatever the alignment says (SdpaPlanUtils::getMaskType), so a "bottom-right causal"
    // graph built with it is top-left, and an engine whose causal kernels are all bottom-right
    // -- AITER on gfx942 -- declined every causal problem in the corpus.
    const bool causal = options.causalMask && options.rightBound < 0;
    attributes.add_causal_mask(false);
    attributes.add_padding_mask(options.paddingMask);
    attributes.add_alibi_mask(options.alibiMask);
    attributes.add_generate_stats(withStats);
    if(withStats)
    {
        attributes.add_stats_tensor_uid(stats->uid);
    }
    if(options.attnScale != 0.0F)
    {
        attributes.add_attn_scale_value(options.attnScale);
    }
    if(options.dropoutProbability != 0.0F)
    {
        attributes.add_dropout_probability(options.dropoutProbability);
    }
    if(options.leftBound >= 0)
    {
        attributes.add_left_bound(options.leftBound);
    }
    if(options.rightBound >= 0)
    {
        attributes.add_right_bound(options.rightBound);
    }
    else if(causal)
    {
        attributes.add_right_bound(0);
    }
    attributes.add_diagonal_alignment(options.diagonalAlignment);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "sdpa_fwd", types.compute, fb::NodeAttributes::SdpaAttributes, node.Union())};
    return detail::finish(builder, "sdpa_fwd", types, tensors, nodes);
}

/// @brief Scaled dot-product attention, backward. Stats from the forward pass are required,
///        not optional: the backward pass reads them rather than recomputing the softmax.
inline GraphBytes sdpaBackward(const TensorSpec& q,
                               const TensorSpec& k,
                               const TensorSpec& v,
                               const TensorSpec& o,
                               const TensorSpec& dO,
                               const TensorSpec& stats,
                               const TensorSpec& dq,
                               const TensorSpec& dk,
                               const TensorSpec& dv,
                               const SdpaOptions& options,
                               const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, q),     detail::addTensor(builder, k),
        detail::addTensor(builder, v),     detail::addTensor(builder, o),
        detail::addTensor(builder, dO),    detail::addTensor(builder, stats),
        detail::addTensor(builder, dq),    detail::addTensor(builder, dk),
        detail::addTensor(builder, dv)};

    fb::SdpaBackwardAttributesBuilder attributes(builder);
    attributes.add_q_tensor_uid(q.uid);
    attributes.add_k_tensor_uid(k.uid);
    attributes.add_v_tensor_uid(v.uid);
    attributes.add_o_tensor_uid(o.uid);
    attributes.add_do_tensor_uid(dO.uid);
    attributes.add_stats_tensor_uid(stats.uid);
    attributes.add_dq_tensor_uid(dq.uid);
    attributes.add_dk_tensor_uid(dk.uid);
    attributes.add_dv_tensor_uid(dv.uid);
    // Written exactly as sdpaForward writes them, for the same reasons: causality as bounds with
    // the anchor in diagonal_alignment (the deprecated flag reads as top-left whatever the
    // alignment says), and the scale stated rather than left to a provider's guess.
    const bool causal = options.causalMask && options.rightBound < 0;
    attributes.add_causal_mask(false);
    attributes.add_padding_mask(options.paddingMask);
    attributes.add_alibi_mask(options.alibiMask);
    if(options.attnScale != 0.0F)
    {
        attributes.add_attn_scale_value(options.attnScale);
    }
    if(options.dropoutProbability != 0.0F)
    {
        attributes.add_dropout_probability(options.dropoutProbability);
    }
    if(options.leftBound >= 0)
    {
        attributes.add_left_bound(options.leftBound);
    }
    if(options.rightBound >= 0)
    {
        attributes.add_right_bound(options.rightBound);
    }
    else if(causal)
    {
        attributes.add_right_bound(0);
    }
    attributes.add_diagonal_alignment(options.diagonalAlignment);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "sdpa_bwd", types.compute, fb::NodeAttributes::SdpaBackwardAttributes,
        node.Union())};
    return detail::finish(builder, "sdpa_bwd", types, tensors, nodes);
}

// ---------------------------------------------------------------------------
// Normalization, backward
// ---------------------------------------------------------------------------

/// @brief LayerNorm backward.
inline GraphBytes layernormBackward(const TensorSpec& dy,
                                    const TensorSpec& x,
                                    const TensorSpec& scale,
                                    const TensorSpec& dx,
                                    const TensorSpec& dscale,
                                    const TensorSpec& dbias,
                                    int64_t normalizedDimCount,
                                    const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy),     detail::addTensor(builder, x),
        detail::addTensor(builder, scale),  detail::addTensor(builder, dx),
        detail::addTensor(builder, dscale), detail::addTensor(builder, dbias)};

    fb::LayernormBackwardAttributesBuilder attributes(builder);
    attributes.add_dy_tensor_uid(dy.uid);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_dx_tensor_uid(dx.uid);
    attributes.add_dscale_tensor_uid(dscale.uid);
    attributes.add_dbias_tensor_uid(dbias.uid);
    attributes.add_normalized_dim_count(normalizedDimCount);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "layernorm_bwd", types.compute,
        fb::NodeAttributes::LayernormBackwardAttributes, node.Union())};
    return detail::finish(builder, "layernorm_bwd", types, tensors, nodes);
}

/// @brief RMSNorm backward. inv_rms is required: it carries the forward pass's normalizer.
inline GraphBytes rmsNormBackward(const TensorSpec& dy,
                                  const TensorSpec& x,
                                  const TensorSpec& scale,
                                  const TensorSpec& invRms,
                                  const TensorSpec& dx,
                                  const TensorSpec& dscale,
                                  const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy),     detail::addTensor(builder, x),
        detail::addTensor(builder, scale),  detail::addTensor(builder, invRms),
        detail::addTensor(builder, dx),     detail::addTensor(builder, dscale)};

    fb::RMSNormBackwardAttributesBuilder attributes(builder);
    attributes.add_dy_tensor_uid(dy.uid);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_inv_rms_tensor_uid(invRms.uid);
    attributes.add_dx_tensor_uid(dx.uid);
    attributes.add_dscale_tensor_uid(dscale.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "rmsnorm_bwd", types.compute, fb::NodeAttributes::RMSNormBackwardAttributes,
        node.Union())};
    return detail::finish(builder, "rmsnorm_bwd", types, tensors, nodes);
}


// ---------------------------------------------------------------------------
// Batch normalization
// ---------------------------------------------------------------------------

/// @brief BatchNorm training forward, with mean and inverse variance produced.
///
/// `peer_stats_tensor_uid` is a vector in the schema, for multi-GPU statistic exchange. It is
/// left empty: a peer-reduced batchnorm is a different problem, and an empty list says so
/// rather than implying one peer.
/// Batchnorm training's optional running statistics: previous and next running mean and
/// variance, blended by a pass-by-value momentum. All five or none.
struct BatchnormRunningStats
{
    TensorSpec prevMean;
    TensorSpec prevVariance;
    TensorSpec momentum;
    TensorSpec nextMean;
    TensorSpec nextVariance;
};

namespace detail
{
/// A virtual fp32 intermediate shaped like @p like: one node's output, the next node's input.
inline TensorSpec intermediateLike(const TensorSpec& like, int64_t uid, const char* name)
{
    TensorSpec spec = like;
    spec.uid = uid;
    spec.name = name;
    spec.dataType = fb::DataType::FLOAT;
    spec.isVirtual = true;
    return spec;
}

/// The activation node that ends a fused batchnorm: @p in to @p out, parameters unset.
inline flatbuffers::Offset<fb::Node> activationNode(flatbuffers::FlatBufferBuilder& builder,
                                                    fb::PointwiseMode mode,
                                                    int64_t in,
                                                    int64_t out)
{
    const auto act = fb::CreatePointwiseAttributes(builder,
                                                   mode,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt, // axis
                                                   in,
                                                   flatbuffers::nullopt, // in_1
                                                   flatbuffers::nullopt, // in_2
                                                   out);
    return fb::CreateNodeDirect(builder, "activation", fb::DataType::FLOAT,
                                fb::NodeAttributes::PointwiseAttributes, act.Union());
}
} // namespace detail

/// Batchnorm training, optionally with running statistics and optionally followed by an
/// activation. With an activation, batchnorm writes a virtual fp32 tensor and the activation
/// writes @p y.
inline GraphBytes batchnormForwardTraining(const TensorSpec& x,
                                           const TensorSpec& scale,
                                           const TensorSpec& bias,
                                           const TensorSpec& epsilon,
                                           const TensorSpec& y,
                                           const TensorSpec& mean,
                                           const TensorSpec& invVariance,
                                           const GraphTypes& types,
                                           const std::optional<BatchnormRunningStats>& running
                                           = std::nullopt,
                                           const std::optional<fb::PointwiseMode>& activation
                                           = std::nullopt)
{
    const auto bnOut = activation ? detail::intermediateLike(y, 101, "bn_out") : y;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x),    detail::addTensor(builder, scale),
        detail::addTensor(builder, bias), detail::addTensor(builder, epsilon),
        detail::addTensor(builder, y),    detail::addTensor(builder, mean),
        detail::addTensor(builder, invVariance)};
    if(activation)
    {
        tensors.push_back(detail::addTensor(builder, bnOut));
    }
    if(running)
    {
        for(const auto* t : {&running->prevMean, &running->prevVariance, &running->momentum,
                             &running->nextMean, &running->nextVariance})
        {
            tensors.push_back(detail::addTensor(builder, *t));
        }
    }

    const std::vector<int64_t> noPeers;
    const auto peers = builder.CreateVector(noPeers);

    fb::BatchnormAttributesBuilder attributes(builder);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_bias_tensor_uid(bias.uid);
    attributes.add_epsilon_tensor_uid(epsilon.uid);
    attributes.add_peer_stats_tensor_uid(peers);
    attributes.add_y_tensor_uid(bnOut.uid);
    attributes.add_mean_tensor_uid(mean.uid);
    attributes.add_inv_variance_tensor_uid(invVariance.uid);
    if(running)
    {
        attributes.add_prev_running_mean_tensor_uid(running->prevMean.uid);
        attributes.add_prev_running_variance_tensor_uid(running->prevVariance.uid);
        attributes.add_momentum_tensor_uid(running->momentum.uid);
        attributes.add_next_running_mean_tensor_uid(running->nextMean.uid);
        attributes.add_next_running_variance_tensor_uid(running->nextVariance.uid);
    }
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "batchnorm", types.compute, fb::NodeAttributes::BatchnormAttributes,
        node.Union())};
    if(activation)
    {
        nodes.push_back(detail::activationNode(builder, *activation, bnOut.uid, y.uid));
    }
    return detail::finish(builder, "batchnorm_training", types, tensors, nodes);
}

/// @brief BatchNorm inference. Statistics are inputs here rather than outputs, which is what
///        distinguishes it from the training pass and gives it different kernels.
/// Batchnorm inference, optionally followed by an activation (batchnorm then writes a virtual
/// fp32 tensor and the activation writes @p y).
inline GraphBytes batchnormInference(const TensorSpec& x,
                                     const TensorSpec& mean,
                                     const TensorSpec& invVariance,
                                     const TensorSpec& scale,
                                     const TensorSpec& bias,
                                     const TensorSpec& y,
                                     const GraphTypes& types,
                                     const std::optional<fb::PointwiseMode>& activation
                                     = std::nullopt)
{
    const auto bnOut = activation ? detail::intermediateLike(y, 101, "bn_out") : y;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x),     detail::addTensor(builder, mean),
        detail::addTensor(builder, invVariance), detail::addTensor(builder, scale),
        detail::addTensor(builder, bias),  detail::addTensor(builder, y)};
    if(activation)
    {
        tensors.push_back(detail::addTensor(builder, bnOut));
    }

    fb::BatchnormInferenceAttributesBuilder attributes(builder);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_mean_tensor_uid(mean.uid);
    attributes.add_inv_variance_tensor_uid(invVariance.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_bias_tensor_uid(bias.uid);
    attributes.add_y_tensor_uid(bnOut.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "batchnorm_inference", types.compute,
        fb::NodeAttributes::BatchnormInferenceAttributes, node.Union())};
    if(activation)
    {
        nodes.push_back(detail::activationNode(builder, *activation, bnOut.uid, y.uid));
    }
    return detail::finish(builder, "batchnorm_inference", types, tensors, nodes);
}

/// @brief BatchNorm backward.
inline GraphBytes batchnormBackward(const TensorSpec& dy,
                                    const TensorSpec& x,
                                    const TensorSpec& scale,
                                    const TensorSpec& dx,
                                    const TensorSpec& dscale,
                                    const TensorSpec& dbias,
                                    const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy),     detail::addTensor(builder, x),
        detail::addTensor(builder, scale),  detail::addTensor(builder, dx),
        detail::addTensor(builder, dscale), detail::addTensor(builder, dbias)};

    const std::vector<int64_t> noPeers;
    const auto peers = builder.CreateVector(noPeers);

    fb::BatchnormBackwardAttributesBuilder attributes(builder);
    attributes.add_dy_tensor_uid(dy.uid);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_peer_stats_tensor_uid(peers);
    attributes.add_dx_tensor_uid(dx.uid);
    attributes.add_dscale_tensor_uid(dscale.uid);
    attributes.add_dbias_tensor_uid(dbias.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "batchnorm_bwd", types.compute,
        fb::NodeAttributes::BatchnormBackwardAttributes, node.Union())};
    return detail::finish(builder, "batchnorm_bwd", types, tensors, nodes);
}

/// @brief The backward pass through a fused batchnorm inference and activation.
///
/// Three nodes: batchnorm inference recomputes y (virtual); the activation's backward mode takes
/// the incoming gradient @p dy and that y, and writes the gradient batchnorm backward consumes
/// (virtual); batchnorm backward then writes dx, dscale and dbias, reusing inference's x, scale,
/// mean and inverse variance.
inline GraphBytes batchnormInferenceActivationBackward(const TensorSpec& x,
                                                       const TensorSpec& mean,
                                                       const TensorSpec& invVariance,
                                                       const TensorSpec& scale,
                                                       const TensorSpec& bias,
                                                       const TensorSpec& dy,
                                                       const TensorSpec& dx,
                                                       const TensorSpec& dscale,
                                                       const TensorSpec& dbias,
                                                       fb::PointwiseMode activationBackward,
                                                       const GraphTypes& types)
{
    const auto y = detail::intermediateLike(dy, 101, "bn_out");
    const auto dyBn = detail::intermediateLike(dy, 102, "dy_bn");

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x),      detail::addTensor(builder, mean),
        detail::addTensor(builder, invVariance), detail::addTensor(builder, scale),
        detail::addTensor(builder, bias),   detail::addTensor(builder, dy),
        detail::addTensor(builder, dx),     detail::addTensor(builder, dscale),
        detail::addTensor(builder, dbias),  detail::addTensor(builder, y),
        detail::addTensor(builder, dyBn)};

    std::vector<flatbuffers::Offset<fb::Node>> nodes;
    {
        fb::BatchnormInferenceAttributesBuilder attributes(builder);
        attributes.add_x_tensor_uid(x.uid);
        attributes.add_mean_tensor_uid(mean.uid);
        attributes.add_inv_variance_tensor_uid(invVariance.uid);
        attributes.add_scale_tensor_uid(scale.uid);
        attributes.add_bias_tensor_uid(bias.uid);
        attributes.add_y_tensor_uid(y.uid);
        const auto node = attributes.Finish();
        nodes.push_back(fb::CreateNodeDirect(builder, "batchnorm_inference", types.compute,
                                             fb::NodeAttributes::BatchnormInferenceAttributes,
                                             node.Union()));
    }
    {
        const auto act = fb::CreatePointwiseAttributes(builder,
                                                       activationBackward,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt,
                                                       flatbuffers::nullopt, // axis
                                                       dy.uid,               // in_0: gradient
                                                       y.uid,                // in_1: forward output
                                                       flatbuffers::nullopt, // in_2
                                                       dyBn.uid);
        nodes.push_back(fb::CreateNodeDirect(builder, "activation_bwd", fb::DataType::FLOAT,
                                             fb::NodeAttributes::PointwiseAttributes,
                                             act.Union()));
    }
    {
        const std::vector<int64_t> noPeers;
        const auto peers = builder.CreateVector(noPeers);
        fb::BatchnormBackwardAttributesBuilder attributes(builder);
        attributes.add_dy_tensor_uid(dyBn.uid);
        attributes.add_x_tensor_uid(x.uid);
        attributes.add_mean_tensor_uid(mean.uid);
        attributes.add_inv_variance_tensor_uid(invVariance.uid);
        attributes.add_scale_tensor_uid(scale.uid);
        attributes.add_peer_stats_tensor_uid(peers);
        attributes.add_dx_tensor_uid(dx.uid);
        attributes.add_dscale_tensor_uid(dscale.uid);
        attributes.add_dbias_tensor_uid(dbias.uid);
        const auto node = attributes.Finish();
        nodes.push_back(fb::CreateNodeDirect(builder, "batchnorm_bwd", types.compute,
                                             fb::NodeAttributes::BatchnormBackwardAttributes,
                                             node.Union()));
    }
    return detail::finish(builder, "batchnorm_activation_bwd", types, tensors, nodes);
}

// ---------------------------------------------------------------------------
// Resample
// ---------------------------------------------------------------------------

/// Window, stride and padding of a pooling problem.
struct ResampleGeometry
{
    std::vector<int64_t> window;
    std::vector<int64_t> stride;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    fb::ResampleMode mode = fb::ResampleMode::MAXPOOL;
    fb::PaddingMode paddingMode = fb::PaddingMode::ZERO_PAD;
};

/// @brief Resample forward (pooling).
///
/// @p index, when given, is written with the position of each window's maximum (max pooling's
/// generate_index), for a backward pass to route gradients through.
inline GraphBytes resampleForward(const TensorSpec& x,
                                  const TensorSpec& y,
                                  const ResampleGeometry& geometry,
                                  const GraphTypes& types,
                                  const std::optional<TensorSpec>& index = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, y)};
    if(index)
    {
        tensors.push_back(detail::addTensor(builder, *index));
    }

    const auto node = fb::CreateResampleFwdAttributesDirect(
        builder,
        x.uid,
        y.uid,
        index ? flatbuffers::Optional<int64_t>(index->uid) : flatbuffers::nullopt,
        &geometry.prePadding,
        &geometry.postPadding,
        &geometry.stride,
        &geometry.window,
        geometry.mode,
        geometry.paddingMode,
        index ? flatbuffers::Optional<bool>(true) : flatbuffers::nullopt);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "resample_fwd", types.compute, fb::NodeAttributes::ResampleFwdAttributes,
        node.Union())};
    return detail::finish(builder, "resample_fwd", types, tensors, nodes);
}

/// @brief Resample backward.
///
/// @p index, when given, is the forward pass's max positions; max pooling's gradient needs it.
inline GraphBytes resampleBackward(const TensorSpec& dy,
                                   const TensorSpec& dx,
                                   const ResampleGeometry& geometry,
                                   const GraphTypes& types,
                                   const std::optional<TensorSpec>& index = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy), detail::addTensor(builder, dx)};
    if(index)
    {
        tensors.push_back(detail::addTensor(builder, *index));
    }

    const auto node = fb::CreateResampleBwdAttributesDirect(
        builder,
        dy.uid,
        dx.uid,
        index ? flatbuffers::Optional<int64_t>(index->uid) : flatbuffers::nullopt,
        &geometry.prePadding,
                                                            &geometry.postPadding,
                                                            &geometry.stride,
                                                            &geometry.window,
                                                            geometry.mode,
                                                            geometry.paddingMode);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "resample_bwd", types.compute, fb::NodeAttributes::ResampleBwdAttributes,
        node.Union())};
    return detail::finish(builder, "resample_bwd", types, tensors, nodes);
}


// ---------------------------------------------------------------------------
// Block scaling
// ---------------------------------------------------------------------------

/// @brief Block-scale quantize. `blockSize` is a scalar here and a vector on the dequantize
///        side; that asymmetry is the schema's, not a transcription slip.
inline GraphBytes blockScaleQuantize(const TensorSpec& x,
                                     const TensorSpec& y,
                                     const TensorSpec& scale,
                                     int32_t blockSize,
                                     bool transpose,
                                     const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, y),
        detail::addTensor(builder, scale)};

    fb::BlockScaleQuantizeAttributesBuilder attributes(builder);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_y_tensor_uid(y.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_block_size(blockSize);
    attributes.add_transpose(transpose);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "block_scale_quantize", types.compute,
        fb::NodeAttributes::BlockScaleQuantizeAttributes, node.Union())};
    return detail::finish(builder, "block_scale_quantize", types, tensors, nodes);
}

/// @brief Block-scale dequantize.
inline GraphBytes blockScaleDequantize(const TensorSpec& x,
                                       const TensorSpec& scale,
                                       const TensorSpec& y,
                                       const std::vector<int32_t>& blockSize,
                                       bool negativeScale,
                                       const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, scale),
        detail::addTensor(builder, y)};

    const auto node = fb::CreateBlockScaleDequantizeAttributesDirect(
        builder, x.uid, scale.uid, y.uid, &blockSize, negativeScale);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "block_scale_dequantize", types.compute,
        fb::NodeAttributes::BlockScaleDequantizeAttributes, node.Union())};
    return detail::finish(builder, "block_scale_dequantize", types, tensors, nodes);
}

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

/// @brief MoE grouped matmul.
///
/// The routing lives in the *contents* of `firstTokenOffset` and `tokenIndex`, not in any
/// extent: how many tokens each expert receives decides the size of every grouped GEMM. Two
/// problems with byte-identical graphs and different routing are different problems, which is
/// why the corpus must declare those contents (see TensorFillers.hpp) rather than leave them
/// to whatever a benchmark happens to allocate.
inline GraphBytes moeGroupedMatmul(const TensorSpec& token,
                                   const TensorSpec& weight,
                                   const TensorSpec& firstTokenOffset,
                                   const TensorSpec& output,
                                   fb::MoeGroupedMatmulMode mode,
                                   int32_t topK,
                                   const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, token), detail::addTensor(builder, weight),
        detail::addTensor(builder, firstTokenOffset), detail::addTensor(builder, output)};

    fb::MoeGroupedMatmulAttributesBuilder attributes(builder);
    attributes.add_token_tensor_uid(token.uid);
    attributes.add_weight_tensor_uid(weight.uid);
    attributes.add_first_token_offset_tensor_uid(firstTokenOffset.uid);
    attributes.add_output_tensor_uid(output.uid);
    attributes.add_mode(mode);
    attributes.add_top_k(topK);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "moe_grouped_matmul", types.compute,
        fb::NodeAttributes::MoeGroupedMatmulAttributes, node.Union())};
    return detail::finish(builder, "moe_grouped_matmul", types, tensors, nodes);
}

/// @brief MoE grouped matmul, weight gradient.
inline GraphBytes moeGroupedMatmulBackward(const TensorSpec& dOutput,
                                           const TensorSpec& token,
                                           const TensorSpec& firstTokenOffset,
                                           const TensorSpec& dWeight,
                                           const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dOutput), detail::addTensor(builder, token),
        detail::addTensor(builder, firstTokenOffset), detail::addTensor(builder, dWeight)};

    fb::MoeGroupedMatmulBwdAttributesBuilder attributes(builder);
    attributes.add_doutput_tensor_uid(dOutput.uid);
    attributes.add_token_tensor_uid(token.uid);
    attributes.add_first_token_offset_tensor_uid(firstTokenOffset.uid);
    attributes.add_dweight_tensor_uid(dWeight.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "moe_grouped_matmul_bwd", types.compute,
        fb::NodeAttributes::MoeGroupedMatmulBwdAttributes, node.Union())};
    return detail::finish(builder, "moe_grouped_matmul_bwd", types, tensors, nodes);
}

// ---------------------------------------------------------------------------
// Custom operation
// ---------------------------------------------------------------------------

/// @brief A custom operation, identified by id with opaque payload.
///
/// Included for completeness of the NodeAttributes union, but a custom op has no declared
/// parameter space -- its shape is whatever its author decided, and `data` is bytes this tool
/// cannot generate meaningfully. A corpus for one is only possible if its author supplies the
/// parameterization; there is nothing here to discover.
inline GraphBytes customOperation(const std::string& customOpId,
                                  const std::vector<TensorSpec>& inputs,
                                  const std::vector<TensorSpec>& outputs,
                                  const std::vector<uint8_t>& payload,
                                  const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors;
    std::vector<int64_t> inputUids;
    std::vector<int64_t> outputUids;
    for(const auto& tensor : inputs)
    {
        tensors.push_back(detail::addTensor(builder, tensor));
        inputUids.push_back(tensor.uid);
    }
    for(const auto& tensor : outputs)
    {
        tensors.push_back(detail::addTensor(builder, tensor));
        outputUids.push_back(tensor.uid);
    }

    const auto node = fb::CreateCustomOpAttributesDirect(
        builder, customOpId.c_str(), &inputUids, &outputUids, &payload);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "custom_op", types.compute, fb::NodeAttributes::CustomOpAttributes,
        node.Union())};
    return detail::finish(builder, "custom_op", types, tensors, nodes);
}

} // namespace hipdnn_corpus_gen::builders
