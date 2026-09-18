// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <string>
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
    return fb::CreateTensorAttributesDirect(builder,
                                            tensor.uid,
                                            tensor.name.c_str(),
                                            tensor.dataType,
                                            &tensor.strides,
                                            &tensor.dims,
                                            /*virtual=*/false);
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

/// @brief Binary elementwise pointwise.
///
/// The optional tensor uids are left null rather than zero. The schema declares them
/// `= null`, so a literal 0 references tensor uid 0 -- a tensor a binary pointwise does not
/// carry -- and the graph then fails to deserialize rather than simply describing something
/// unusual.
/// Mode-specific scalars. §12.6 lists them as parameters of a pointwise problem, and they are:
/// a ReLU with a non-zero lower-clip slope is a leaky ReLU and a different kernel.
struct PointwiseScalars
{
    float reluLowerClip = 0.0F;
    float reluUpperClip = 0.0F;
    float reluLowerClipSlope = 0.0F;
    float swishBeta = 0.0F;
    float eluAlpha = 0.0F;
    float softplusBeta = 0.0F;
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
};

/// @brief Scaled dot-product attention, forward.
///
/// SdpaAttributes declares twenty-eight optional tensor uids -- paged KV, dropout, descale
/// factors, sinks. All are left null here. Each is a different problem rather than a variation
/// on this one, and giving them a uid they do not have is how a graph stops deserializing.
inline GraphBytes sdpaForward(const TensorSpec& q,
                              const TensorSpec& k,
                              const TensorSpec& v,
                              const TensorSpec& o,
                              const SdpaOptions& options,
                              const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, q), detail::addTensor(builder, k),
        detail::addTensor(builder, v), detail::addTensor(builder, o)};

    fb::SdpaAttributesBuilder attributes(builder);
    attributes.add_q_tensor_uid(q.uid);
    attributes.add_k_tensor_uid(k.uid);
    attributes.add_v_tensor_uid(v.uid);
    attributes.add_o_tensor_uid(o.uid);
    attributes.add_causal_mask(options.causalMask);
    attributes.add_padding_mask(options.paddingMask);
    attributes.add_alibi_mask(options.alibiMask);
    attributes.add_generate_stats(options.generateStats);
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
    attributes.add_causal_mask(options.causalMask);
    attributes.add_padding_mask(options.paddingMask);
    attributes.add_alibi_mask(options.alibiMask);
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
inline GraphBytes batchnormForwardTraining(const TensorSpec& x,
                                           const TensorSpec& scale,
                                           const TensorSpec& bias,
                                           const TensorSpec& epsilon,
                                           const TensorSpec& y,
                                           const TensorSpec& mean,
                                           const TensorSpec& invVariance,
                                           const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x),    detail::addTensor(builder, scale),
        detail::addTensor(builder, bias), detail::addTensor(builder, epsilon),
        detail::addTensor(builder, y),    detail::addTensor(builder, mean),
        detail::addTensor(builder, invVariance)};

    const std::vector<int64_t> noPeers;
    const auto peers = builder.CreateVector(noPeers);

    fb::BatchnormAttributesBuilder attributes(builder);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_bias_tensor_uid(bias.uid);
    attributes.add_epsilon_tensor_uid(epsilon.uid);
    attributes.add_peer_stats_tensor_uid(peers);
    attributes.add_y_tensor_uid(y.uid);
    attributes.add_mean_tensor_uid(mean.uid);
    attributes.add_inv_variance_tensor_uid(invVariance.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "batchnorm", types.compute, fb::NodeAttributes::BatchnormAttributes,
        node.Union())};
    return detail::finish(builder, "batchnorm_training", types, tensors, nodes);
}

/// @brief BatchNorm inference. Statistics are inputs here rather than outputs, which is what
///        distinguishes it from the training pass and gives it different kernels.
inline GraphBytes batchnormInference(const TensorSpec& x,
                                     const TensorSpec& mean,
                                     const TensorSpec& invVariance,
                                     const TensorSpec& scale,
                                     const TensorSpec& bias,
                                     const TensorSpec& y,
                                     const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x),     detail::addTensor(builder, mean),
        detail::addTensor(builder, invVariance), detail::addTensor(builder, scale),
        detail::addTensor(builder, bias),  detail::addTensor(builder, y)};

    fb::BatchnormInferenceAttributesBuilder attributes(builder);
    attributes.add_x_tensor_uid(x.uid);
    attributes.add_mean_tensor_uid(mean.uid);
    attributes.add_inv_variance_tensor_uid(invVariance.uid);
    attributes.add_scale_tensor_uid(scale.uid);
    attributes.add_bias_tensor_uid(bias.uid);
    attributes.add_y_tensor_uid(y.uid);
    const auto node = attributes.Finish();

    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "batchnorm_inference", types.compute,
        fb::NodeAttributes::BatchnormInferenceAttributes, node.Union())};
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
inline GraphBytes resampleForward(const TensorSpec& x,
                                  const TensorSpec& y,
                                  const ResampleGeometry& geometry,
                                  const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, x), detail::addTensor(builder, y)};

    const auto node = fb::CreateResampleFwdAttributesDirect(builder,
                                                            x.uid,
                                                            y.uid,
                                                            flatbuffers::nullopt, // index
                                                            &geometry.prePadding,
                                                            &geometry.postPadding,
                                                            &geometry.stride,
                                                            &geometry.window,
                                                            geometry.mode,
                                                            geometry.paddingMode);
    std::vector<flatbuffers::Offset<fb::Node>> nodes{fb::CreateNodeDirect(
        builder, "resample_fwd", types.compute, fb::NodeAttributes::ResampleFwdAttributes,
        node.Union())};
    return detail::finish(builder, "resample_fwd", types, tensors, nodes);
}

/// @brief Resample backward.
inline GraphBytes resampleBackward(const TensorSpec& dy,
                                   const TensorSpec& dx,
                                   const ResampleGeometry& geometry,
                                   const GraphTypes& types)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        detail::addTensor(builder, dy), detail::addTensor(builder, dx)};

    const auto node = fb::CreateResampleBwdAttributesDirect(builder,
                                                            dy.uid,
                                                            dx.uid,
                                                            flatbuffers::nullopt, // index
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
