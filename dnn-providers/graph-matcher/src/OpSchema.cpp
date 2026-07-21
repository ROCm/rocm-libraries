// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_graph_matcher/OpSchema.hpp>

namespace hipdnn::graph_matcher {

namespace data = hipdnn_flatbuffers_sdk::data_objects;

// Role name = the field base (e.g. "q" for q_tensor_uid). The getter is always
// <base>_tensor_uid() except for the raw-named CustomOp vectors.
#define REQ(T, base)                                                                \
    EdgeRole {                                                                      \
        #base, Arity::Required, &readRequired<data::T, &data::T::base##_tensor_uid> \
    }
#define OPT(T, base)                                                                \
    EdgeRole {                                                                      \
        #base, Arity::Optional, &readOptional<data::T, &data::T::base##_tensor_uid> \
    }
#define VAR(T, base)                                                                \
    EdgeRole {                                                                      \
        #base, Arity::Variadic, &readVariadic<data::T, &data::T::base##_tensor_uid> \
    }
#define VAR_RAW(T, getter, name)                                        \
    EdgeRole {                                                          \
        name, Arity::Variadic, &readVariadic<data::T, &data::T::getter> \
    }
#define ATTR(T, field, Ret)                                    \
    AttrAccessor {                                             \
        #field, &readAttrScalar<data::T, Ret, &data::T::field> \
    }

OpSchemaRegistry::OpSchemaRegistry() {
    using NA = NodeAttributes;

    _schemas = {
        // --- Pointwise ---
        // axis_tensor_uid is a plain axis index (per schema), not an edge.
        {NA::PointwiseAttributes,
         "pointwise",
         {REQ(PointwiseAttributes, in_0), OPT(PointwiseAttributes, in_1),
          OPT(PointwiseAttributes, in_2)},
         {REQ(PointwiseAttributes, out_0)},
         {ATTR(PointwiseAttributes, operation, data::PointwiseMode)}},

        // --- Matmul ---
        {NA::MatmulAttributes,
         "matmul",
         {REQ(MatmulAttributes, a), REQ(MatmulAttributes, b)},
         {REQ(MatmulAttributes, c)}},

        // --- Reduction ---
        {NA::ReductionAttributes,
         "reduction",
         {REQ(ReductionAttributes, in)},
         {REQ(ReductionAttributes, out)},
         {ATTR(ReductionAttributes, mode, data::ReductionMode)}},

        // --- Convolution ---
        {NA::ConvolutionFwdAttributes,
         "conv_fwd",
         {REQ(ConvolutionFwdAttributes, x), REQ(ConvolutionFwdAttributes, w)},
         {REQ(ConvolutionFwdAttributes, y)}},
        {NA::ConvolutionBwdAttributes,
         "conv_bwd",
         {REQ(ConvolutionBwdAttributes, dy), REQ(ConvolutionBwdAttributes, w)},
         {REQ(ConvolutionBwdAttributes, dx)}},
        {NA::ConvolutionWrwAttributes,
         "conv_wrw",
         {REQ(ConvolutionWrwAttributes, x), REQ(ConvolutionWrwAttributes, dy)},
         {REQ(ConvolutionWrwAttributes, dw)}},

        // --- Resample ---
        {NA::ResampleFwdAttributes,
         "resample_fwd",
         {REQ(ResampleFwdAttributes, x)},
         {REQ(ResampleFwdAttributes, y), OPT(ResampleFwdAttributes, index)}},

        // --- Normalizations ---
        {NA::LayernormAttributes,
         "layernorm",
         {REQ(LayernormAttributes, x), REQ(LayernormAttributes, scale),
          REQ(LayernormAttributes, bias), REQ(LayernormAttributes, epsilon)},
         {REQ(LayernormAttributes, y), OPT(LayernormAttributes, mean),
          OPT(LayernormAttributes, inv_variance)}},
        {NA::LayernormBackwardAttributes,
         "layernorm_bwd",
         {REQ(LayernormBackwardAttributes, dy), REQ(LayernormBackwardAttributes, x),
          REQ(LayernormBackwardAttributes, scale), OPT(LayernormBackwardAttributes, mean),
          OPT(LayernormBackwardAttributes, inv_variance),
          OPT(LayernormBackwardAttributes, epsilon)},
         {REQ(LayernormBackwardAttributes, dx), REQ(LayernormBackwardAttributes, dscale),
          REQ(LayernormBackwardAttributes, dbias)}},
        {NA::RMSNormAttributes,
         "rmsnorm",
         {REQ(RMSNormAttributes, x), REQ(RMSNormAttributes, scale), REQ(RMSNormAttributes, epsilon),
          OPT(RMSNormAttributes, bias)},
         {REQ(RMSNormAttributes, y), OPT(RMSNormAttributes, inv_rms)}},
        {NA::RMSNormBackwardAttributes,
         "rmsnorm_bwd",
         {REQ(RMSNormBackwardAttributes, dy), REQ(RMSNormBackwardAttributes, x),
          REQ(RMSNormBackwardAttributes, scale), REQ(RMSNormBackwardAttributes, inv_rms)},
         {REQ(RMSNormBackwardAttributes, dx), REQ(RMSNormBackwardAttributes, dscale),
          OPT(RMSNormBackwardAttributes, dbias)}},

        // --- Batchnorm ---
        {NA::BatchnormInferenceAttributes,
         "batchnorm_inference",
         {REQ(BatchnormInferenceAttributes, x), REQ(BatchnormInferenceAttributes, mean),
          REQ(BatchnormInferenceAttributes, inv_variance), REQ(BatchnormInferenceAttributes, scale),
          REQ(BatchnormInferenceAttributes, bias)},
         {REQ(BatchnormInferenceAttributes, y)}},
        {NA::BatchnormInferenceAttributesVarianceExt,
         "batchnorm_inference_variance_ext",
         {REQ(BatchnormInferenceAttributesVarianceExt, x),
          REQ(BatchnormInferenceAttributesVarianceExt, mean),
          REQ(BatchnormInferenceAttributesVarianceExt, variance),
          REQ(BatchnormInferenceAttributesVarianceExt, scale),
          REQ(BatchnormInferenceAttributesVarianceExt, bias),
          REQ(BatchnormInferenceAttributesVarianceExt, epsilon)},
         {REQ(BatchnormInferenceAttributesVarianceExt, y)}},
        {NA::BatchnormAttributes,
         "batchnorm",
         {REQ(BatchnormAttributes, x), REQ(BatchnormAttributes, scale),
          REQ(BatchnormAttributes, bias), REQ(BatchnormAttributes, epsilon),
          VAR(BatchnormAttributes, peer_stats), OPT(BatchnormAttributes, prev_running_mean),
          OPT(BatchnormAttributes, prev_running_variance), OPT(BatchnormAttributes, momentum)},
         {REQ(BatchnormAttributes, y), OPT(BatchnormAttributes, mean),
          OPT(BatchnormAttributes, inv_variance), OPT(BatchnormAttributes, next_running_mean),
          OPT(BatchnormAttributes, next_running_variance)}},
        {NA::BatchnormBackwardAttributes,
         "batchnorm_bwd",
         {REQ(BatchnormBackwardAttributes, dy), REQ(BatchnormBackwardAttributes, x),
          OPT(BatchnormBackwardAttributes, mean), OPT(BatchnormBackwardAttributes, inv_variance),
          REQ(BatchnormBackwardAttributes, scale), VAR(BatchnormBackwardAttributes, peer_stats)},
         {REQ(BatchnormBackwardAttributes, dx), REQ(BatchnormBackwardAttributes, dscale),
          REQ(BatchnormBackwardAttributes, dbias)}},

        // --- Block-scale quant/dequant ---
        {NA::BlockScaleQuantizeAttributes,
         "block_scale_quantize",
         {REQ(BlockScaleQuantizeAttributes, x)},
         {REQ(BlockScaleQuantizeAttributes, y), REQ(BlockScaleQuantizeAttributes, scale)},
         {ATTR(BlockScaleQuantizeAttributes, transpose, bool)}},
        {NA::BlockScaleDequantizeAttributes,
         "block_scale_dequantize",
         {REQ(BlockScaleDequantizeAttributes, x), REQ(BlockScaleDequantizeAttributes, scale)},
         {REQ(BlockScaleDequantizeAttributes, y)}},

        // --- SDPA ---
        {NA::SdpaAttributes,
         "sdpa_fwd",
         {REQ(SdpaAttributes, q),
          REQ(SdpaAttributes, k),
          REQ(SdpaAttributes, v),
          OPT(SdpaAttributes, attn_mask),
          OPT(SdpaAttributes, scale),
          OPT(SdpaAttributes, seq_len_q),
          OPT(SdpaAttributes, seq_len_kv),
          OPT(SdpaAttributes, seed),
          OPT(SdpaAttributes, offset),
          OPT(SdpaAttributes, dropout_mask),
          OPT(SdpaAttributes, dropout_scale),
          OPT(SdpaAttributes, page_table_k),
          OPT(SdpaAttributes, page_table_v),
          OPT(SdpaAttributes, block_mask),
          OPT(SdpaAttributes, sink_token),
          OPT(SdpaAttributes, descale_q),
          OPT(SdpaAttributes, descale_k),
          OPT(SdpaAttributes, descale_v),
          OPT(SdpaAttributes, descale_s),
          OPT(SdpaAttributes, scale_s),
          OPT(SdpaAttributes, scale_o)},
         {REQ(SdpaAttributes, o), OPT(SdpaAttributes, stats), OPT(SdpaAttributes, max),
          OPT(SdpaAttributes, sum_exp), OPT(SdpaAttributes, rng_dump), OPT(SdpaAttributes, amax_s),
          OPT(SdpaAttributes, amax_o)},
         {ATTR(SdpaAttributes, causal_mask, bool), ATTR(SdpaAttributes, padding_mask, bool),
          ATTR(SdpaAttributes, alibi_mask, bool),
          ATTR(SdpaAttributes, diagonal_alignment, data::DiagonalAlignment)}},
        {NA::SdpaBackwardAttributes,
         "sdpa_bwd",
         {REQ(SdpaBackwardAttributes, q), REQ(SdpaBackwardAttributes, k),
          REQ(SdpaBackwardAttributes, v), REQ(SdpaBackwardAttributes, o),
          REQ(SdpaBackwardAttributes, do), REQ(SdpaBackwardAttributes, stats),
          OPT(SdpaBackwardAttributes, scale), OPT(SdpaBackwardAttributes, attn_mask),
          OPT(SdpaBackwardAttributes, seq_len_q), OPT(SdpaBackwardAttributes, seq_len_kv),
          OPT(SdpaBackwardAttributes, seed), OPT(SdpaBackwardAttributes, offset),
          OPT(SdpaBackwardAttributes, dropout_mask), OPT(SdpaBackwardAttributes, dropout_scale),
          OPT(SdpaBackwardAttributes, dropout_scale_inv)},
         {REQ(SdpaBackwardAttributes, dq), REQ(SdpaBackwardAttributes, dk),
          REQ(SdpaBackwardAttributes, dv), OPT(SdpaBackwardAttributes, dbias)}},

        // --- Custom op (generic UID vectors) ---
        {NA::CustomOpAttributes,
         "custom_op",
         {VAR_RAW(CustomOpAttributes, input_tensor_uids, "inputs")},
         {VAR_RAW(CustomOpAttributes, output_tensor_uids, "outputs")}},
    };

    for (const auto& schema : _schemas) {
        _byType[static_cast<size_t>(schema.type)] = &schema;
    }
}

#undef REQ
#undef OPT
#undef VAR
#undef VAR_RAW
#undef ATTR

const OpSchemaRegistry& OpSchemaRegistry::builtin() {
    static const OpSchemaRegistry registry;
    return registry;
}

const OpSchema* OpSchemaRegistry::find(NodeAttributes type) const noexcept {
    const auto index = static_cast<size_t>(type);
    if (index >= _byType.size()) {
        return nullptr;
    }
    return _byType[index];
}

const OpSchema* OpSchemaRegistry::forNode(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node) const noexcept {
    return find(node.attributesType());
}

const OpSchema* OpSchemaRegistry::findByOpcode(std::string_view opcode) const noexcept {
    for (const auto& schema : _schemas) {
        if (schema.opcode == opcode) {
            return &schema;
        }
    }
    return nullptr;
}

}  // namespace hipdnn::graph_matcher
