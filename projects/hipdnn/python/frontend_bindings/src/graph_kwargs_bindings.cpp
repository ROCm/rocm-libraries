// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"
#include "cudnn_compat.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;
using namespace hipdnn_frontend;

namespace
{

using TensorPtr = std::shared_ptr<graph::TensorAttributes>;
using TensorList = std::vector<TensorPtr>;

// cuDNN's two convolution shapes differ only in how padding is given, so each
// conv op registers both from one body. The symmetric form is registered first,
// matching cuDNN's order.
template <typename AttributesT>
AttributesT makeConvAttributes(const std::vector<int64_t>& prePadding,
                               const std::vector<int64_t>& postPadding,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& dilation,
                               ConvolutionMode convolutionMode,
                               DataType computeDataType,
                               const std::string& name)
{
    AttributesT attributes;
    attributes.set_pre_padding(prePadding)
        .set_post_padding(postPadding)
        .set_stride(stride)
        .set_dilation(dilation)
        .set_convolution_mode(convolutionMode)
        .set_compute_data_type(computeDataType)
        .set_name(name);
    return attributes;
}

template <typename AttributesT, typename OpT>
void defConv(nb::class_<graph::Graph>& cls,
             const char* opName,
             const char* firstName,
             const char* secondName,
             OpT op)
{
    cls.def(
        opName,
        [op](graph::Graph& graph,
             TensorPtr first,
             TensorPtr second,
             const std::vector<int64_t>& padding,
             const std::vector<int64_t>& stride,
             const std::vector<int64_t>& dilation,
             ConvolutionMode convolutionMode,
             DataType computeDataType,
             const std::string& name) {
            return op(
                graph,
                std::move(first),
                std::move(second),
                makeConvAttributes<AttributesT>(
                    padding, padding, stride, dilation, convolutionMode, computeDataType, name));
        },
        nb::arg(firstName),
        nb::arg(secondName),
        nb::arg("padding") = std::vector<int64_t>{},
        nb::arg("stride") = std::vector<int64_t>{},
        nb::arg("dilation") = std::vector<int64_t>{},
        nb::arg("convolution_mode") = ConvolutionMode::CROSS_CORRELATION,
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form; padding applies to both sides.");

    cls.def(
        opName,
        [op](graph::Graph& graph,
             TensorPtr first,
             TensorPtr second,
             const std::vector<int64_t>& prePadding,
             const std::vector<int64_t>& postPadding,
             const std::vector<int64_t>& stride,
             const std::vector<int64_t>& dilation,
             ConvolutionMode convolutionMode,
             DataType computeDataType,
             const std::string& name) {
            return op(graph,
                      std::move(first),
                      std::move(second),
                      makeConvAttributes<AttributesT>(prePadding,
                                                      postPadding,
                                                      stride,
                                                      dilation,
                                                      convolutionMode,
                                                      computeDataType,
                                                      name));
        },
        nb::arg(firstName),
        nb::arg(secondName),
        nb::arg("pre_padding") = std::vector<int64_t>{},
        nb::arg("post_padding") = std::vector<int64_t>{},
        nb::arg("stride") = std::vector<int64_t>{},
        nb::arg("dilation") = std::vector<int64_t>{},
        nb::arg("convolution_mode") = ConvolutionMode::CROSS_CORRELATION,
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form with asymmetric padding.");
}

#ifdef HIPDNN_ENABLE_SDPA
using AttnScale = std::variant<float, TensorPtr>;

template <typename AttributesT>
void setAttnScale(AttributesT& attributes, const std::optional<AttnScale>& attnScale)
{
    if(!attnScale)
    {
        return;
    }
    if(const auto* value = std::get_if<float>(&*attnScale))
    {
        attributes.set_attn_scale(*value);
    }
    else
    {
        attributes.set_attn_scale(std::get<TensorPtr>(*attnScale));
    }
}

// Forward sdpa only, as in cuDNN: the deprecated causal flags become an alignment
// plus a zero right bound, so they conflict with an explicit right bound, and
// sliding_window_length is the deprecated spelling of the left bound.
struct DiagonalBand
{
    DiagonalAlignment alignment;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
};

DiagonalBand resolveDiagonalBand(bool useCausalMask,
                                 bool useCausalMaskBottomRight,
                                 std::optional<int64_t> slidingWindowLength,
                                 DiagonalAlignment diagonalAlignment,
                                 std::optional<int64_t> leftBound,
                                 std::optional<int64_t> rightBound)
{
    if(useCausalMask && useCausalMaskBottomRight)
    {
        throw std::runtime_error(
            "use_causal_mask and use_causal_mask_bottom_right cannot both be true");
    }
    if(useCausalMask && rightBound)
    {
        throw std::runtime_error(
            "use_causal_mask and diagonal_band_right_bound cannot be set at the same time");
    }
    if(useCausalMaskBottomRight && rightBound)
    {
        throw std::runtime_error("use_causal_mask_bottom_right and diagonal_band_right_bound "
                                 "cannot be set at the same time");
    }
    if(slidingWindowLength && leftBound)
    {
        throw std::runtime_error("sliding window and left_bound cannot be set at the same time");
    }

    DiagonalBand band{
        diagonalAlignment, slidingWindowLength ? slidingWindowLength : leftBound, rightBound};
    if(useCausalMask)
    {
        band.alignment = DiagonalAlignment::TOP_LEFT;
        band.rightBound = 0;
    }
    if(useCausalMaskBottomRight)
    {
        band.alignment = DiagonalAlignment::BOTTOM_RIGHT;
        band.rightBound = 0;
    }
    return band;
}

template <typename AttributesT>
void setDiagonalBand(AttributesT& attributes, const DiagonalBand& band)
{
    attributes.set_diagonal_alignment(band.alignment);
    if(band.leftBound)
    {
        attributes.set_diagonal_band_left_bound(*band.leftBound);
    }
    if(band.rightBound)
    {
        attributes.set_diagonal_band_right_bound(*band.rightBound);
    }
}

void requireNone(const nb::handle& value, const char* parameter)
{
    if(!value.is_none())
    {
        hipdnn_python::throwUnsupportedParameter(parameter, "None");
    }
}

void requireNullTensor(const TensorPtr& value, const char* parameter)
{
    if(value)
    {
        hipdnn_python::throwUnsupportedParameter(parameter, "None");
    }
}

// Philox form (probability, seed, offset) or explicit mask form.
using SdpaDropout
    = std::variant<std::tuple<float, TensorPtr, TensorPtr>, std::tuple<TensorPtr, TensorPtr>>;
using SdpaBackwardDropout = std::variant<std::tuple<float, TensorPtr, TensorPtr>,
                                         std::tuple<TensorPtr, TensorPtr, TensorPtr>>;

void sdpaBindings(nb::class_<graph::Graph>& cls)
{
    cls.def(
        "sdpa",
        [](graph::Graph& graph,
           TensorPtr q,
           TensorPtr k,
           TensorPtr v,
           std::optional<bool> isInference,
           const std::optional<AttnScale>& attnScale,
           const TensorPtr& bias,
           const TensorPtr& blockMask,
           bool useAlibiMask,
           bool usePaddingMask,
           const TensorPtr& seqLenQ,
           const TensorPtr& seqLenKv,
           bool useCausalMask,
           bool useCausalMaskBottomRight,
           std::optional<int64_t> slidingWindowLength,
           DiagonalAlignment diagonalAlignment,
           std::optional<int64_t> diagonalBandLeftBound,
           std::optional<int64_t> diagonalBandRightBound,
           const std::optional<SdpaDropout>& dropout,
           const TensorPtr& rngDump,
           const TensorPtr& pagedAttentionKTable,
           const TensorPtr& pagedAttentionVTable,
           std::optional<int32_t> pagedAttentionMaxSeqLenKv,
           DataType computeDataType,
           const std::string& name,
           const nb::object& scoreMod,
           std::optional<bool> generateStats,
           AttentionImplementation implementation,
           const TensorPtr& scoreMax,
           const TensorPtr& scoreSumExp,
           const TensorPtr& sinkToken,
           bool unfuseFma) {
            requireNone(scoreMod, "score_mod");
            if(generateStats.has_value() == isInference.has_value())
            {
                throw std::runtime_error("Exactly one of {generate_stats, is_inference} must be "
                                         "set (prefer generate_stats).");
            }

            graph::SdpaAttributes attributes;
            attributes.set_generate_stats(generateStats ? *generateStats : !*isInference)
                .set_alibi_mask(useAlibiMask)
                .set_padding_mask(usePaddingMask)
                .set_implementation(implementation)
                .set_unfuse_fma(unfuseFma)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            setAttnScale(attributes, attnScale);
            setDiagonalBand(attributes,
                            resolveDiagonalBand(useCausalMask,
                                                useCausalMaskBottomRight,
                                                slidingWindowLength,
                                                diagonalAlignment,
                                                diagonalBandLeftBound,
                                                diagonalBandRightBound));
            if(bias)
            {
                attributes.set_bias(bias);
            }
            if(blockMask)
            {
                attributes.set_block_mask(blockMask);
            }
            if(seqLenQ)
            {
                attributes.set_seq_len_q(seqLenQ);
            }
            if(seqLenKv)
            {
                attributes.set_seq_len_kv(seqLenKv);
            }
            if(pagedAttentionKTable)
            {
                attributes.set_paged_attention_k_table(pagedAttentionKTable);
            }
            if(pagedAttentionVTable)
            {
                attributes.set_paged_attention_v_table(pagedAttentionVTable);
            }
            if(pagedAttentionMaxSeqLenKv)
            {
                attributes.set_paged_attention_max_seq_len_kv(*pagedAttentionMaxSeqLenKv);
            }
            if(scoreMax)
            {
                attributes.set_logit_max(scoreMax);
            }
            if(scoreSumExp)
            {
                attributes.set_score_sum_exp(scoreSumExp);
            }
            if(sinkToken)
            {
                attributes.set_sink_token(sinkToken);
            }
            // As in cuDNN, rng_dump only applies to the Philox dropout form.
            if(dropout)
            {
                if(const auto* philox = std::get_if<0>(&*dropout))
                {
                    attributes.set_dropout(
                        std::get<0>(*philox), std::get<1>(*philox), std::get<2>(*philox));
                    if(rngDump)
                    {
                        attributes.set_rng_dump(rngDump);
                    }
                }
                else
                {
                    const auto& mask = std::get<1>(*dropout);
                    attributes.set_dropout(std::get<0>(mask), std::get<1>(mask));
                }
            }

            const auto outputs
                = graph.sdpa(std::move(q), std::move(k), std::move(v), std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1]);
        },
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("is_inference") = nb::none(),
        nb::arg("attn_scale") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("block_mask") = nb::none(),
        nb::arg("use_alibi_mask") = false,
        nb::arg("use_padding_mask") = false,
        nb::arg("seq_len_q") = nb::none(),
        nb::arg("seq_len_kv") = nb::none(),
        nb::arg("use_causal_mask") = false,
        nb::arg("use_causal_mask_bottom_right") = false,
        nb::arg("sliding_window_length") = nb::none(),
        nb::arg("diagonal_alignment") = DiagonalAlignment::TOP_LEFT,
        nb::arg("diagonal_band_left_bound") = nb::none(),
        nb::arg("diagonal_band_right_bound") = nb::none(),
        nb::arg("dropout") = nb::none(),
        nb::arg("rng_dump") = nb::none(),
        nb::arg("paged_attention_k_table") = nb::none(),
        nb::arg("paged_attention_v_table") = nb::none(),
        nb::arg("paged_attention_max_seq_len_kv") = nb::none(),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        nb::arg("score_mod") = nb::none(),
        nb::arg("generate_stats") = nb::none(),
        nb::arg("implementation") = AttentionImplementation::AUTO,
        nb::arg("score_max") = nb::none(),
        nb::arg("score_sum_exp") = nb::none(),
        nb::arg("sink_token") = nb::none(),
        nb::arg("unfuse_fma") = false,
        "cuDNN-compatible keyword form. Returns (o, stats); exactly one of generate_stats "
        "and is_inference must be set.");

    cls.def(
        "sdpa_backward",
        [](graph::Graph& graph,
           TensorPtr q,
           TensorPtr k,
           TensorPtr v,
           TensorPtr o,
           TensorPtr dO,
           TensorPtr stats,
           const std::optional<AttnScale>& attnScale,
           const TensorPtr& bias,
           const TensorPtr& dBias,
           bool useAlibiMask,
           bool usePaddingMask,
           const TensorPtr& seqLenQ,
           const TensorPtr& seqLenKv,
           std::optional<int64_t> maxTotalSeqLenQ,
           std::optional<int64_t> maxTotalSeqLenKv,
           bool useCausalMask,
           bool useCausalMaskBottomRight,
           std::optional<int64_t> slidingWindowLength,
           DiagonalAlignment diagonalAlignment,
           std::optional<int64_t> diagonalBandLeftBound,
           std::optional<int64_t> diagonalBandRightBound,
           const std::optional<SdpaBackwardDropout>& dropout,
           const TensorPtr& rngDump,
           bool useDeterministicAlgorithm,
           DataType computeDataType,
           const std::string& name,
           const nb::object& scoreMod,
           const nb::object& scoreModBprop,
           const TensorPtr& sinkToken,
           const TensorPtr& dSinkToken) {
            if(maxTotalSeqLenQ)
            {
                hipdnn_python::throwUnsupportedParameter("max_total_seq_len_q", "None");
            }
            if(maxTotalSeqLenKv)
            {
                hipdnn_python::throwUnsupportedParameter("max_total_seq_len_kv", "None");
            }
            requireNullTensor(rngDump, "rng_dump");
            if(useDeterministicAlgorithm)
            {
                hipdnn_python::throwUnsupportedParameter("use_deterministic_algorithm", "False");
            }
            requireNone(scoreMod, "score_mod");
            requireNone(scoreModBprop, "score_mod_bprop");
            requireNullTensor(sinkToken, "sink_token");
            requireNullTensor(dSinkToken, "dSink_token");

            graph::SdpaBackwardAttributes attributes;
            attributes.set_alibi_mask(useAlibiMask)
                .set_padding_mask(usePaddingMask)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            setAttnScale(attributes, attnScale);
            // cuDNN's backward applies these in order with no conflict checks, so a later
            // bound overrides what a causal flag set; the forward overload rejects such mixes.
            attributes.set_diagonal_alignment(diagonalAlignment)
                .set_causal_mask(useCausalMask)
                .set_causal_mask_bottom_right(useCausalMaskBottomRight);
            if(slidingWindowLength)
            {
                attributes.set_diagonal_band_left_bound(*slidingWindowLength);
            }
            if(diagonalBandLeftBound)
            {
                attributes.set_diagonal_band_left_bound(*diagonalBandLeftBound);
            }
            if(diagonalBandRightBound)
            {
                attributes.set_diagonal_band_right_bound(*diagonalBandRightBound);
            }
            if(bias)
            {
                attributes.set_bias(bias);
            }
            if(dBias)
            {
                attributes.set_dbias(dBias);
            }
            if(seqLenQ)
            {
                attributes.set_seq_len_q(seqLenQ);
            }
            if(seqLenKv)
            {
                attributes.set_seq_len_kv(seqLenKv);
            }
            if(dropout)
            {
                if(const auto* philox = std::get_if<0>(&*dropout))
                {
                    attributes.set_dropout(
                        std::get<0>(*philox), std::get<1>(*philox), std::get<2>(*philox));
                }
                else
                {
                    const auto& mask = std::get<1>(*dropout);
                    attributes.set_dropout(std::get<0>(mask), std::get<1>(mask), std::get<2>(mask));
                }
            }

            const auto outputs = graph.sdpa_backward(std::move(q),
                                                     std::move(k),
                                                     std::move(v),
                                                     std::move(o),
                                                     std::move(dO),
                                                     std::move(stats),
                                                     std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
        },
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("o"),
        nb::arg("dO"),
        nb::arg("stats"),
        nb::arg("attn_scale") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("dBias") = nb::none(),
        nb::arg("use_alibi_mask") = false,
        nb::arg("use_padding_mask") = false,
        nb::arg("seq_len_q") = nb::none(),
        nb::arg("seq_len_kv") = nb::none(),
        nb::arg("max_total_seq_len_q") = nb::none(),
        nb::arg("max_total_seq_len_kv") = nb::none(),
        nb::arg("use_causal_mask") = false,
        nb::arg("use_causal_mask_bottom_right") = false,
        nb::arg("sliding_window_length") = nb::none(),
        nb::arg("diagonal_alignment") = DiagonalAlignment::TOP_LEFT,
        nb::arg("diagonal_band_left_bound") = nb::none(),
        nb::arg("diagonal_band_right_bound") = nb::none(),
        nb::arg("dropout") = nb::none(),
        nb::arg("rng_dump") = nb::none(),
        nb::arg("use_deterministic_algorithm") = false,
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        nb::arg("score_mod") = nb::none(),
        nb::arg("score_mod_bprop") = nb::none(),
        nb::arg("sink_token") = nb::none(),
        nb::arg("dSink_token") = nb::none(),
        "cuDNN-compatible keyword form. Returns (dQ, dK, dV).");
}
#endif

} // namespace

void graphKwargsBindings(nb::class_<graph::Graph>& cls)
{
    cls.def(
        "batchnorm",
        [](graph::Graph& graph,
           TensorPtr input,
           TensorPtr scale,
           TensorPtr bias,
           const TensorPtr& inRunningMean,
           const TensorPtr& inRunningVar,
           const TensorPtr& epsilon,
           const TensorPtr& momentum,
           const TensorList& peerStats,
           DataType computeDataType,
           const std::string& name) {
            graph::BatchnormAttributes attributes;
            attributes.set_epsilon(epsilon)
                .set_peer_stats(peerStats)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            // Running statistics are optional in hipDNN; leave absent ones unset rather than
            // recording null inputs.
            if(inRunningMean)
            {
                attributes.set_prev_running_mean(inRunningMean);
            }
            if(inRunningVar)
            {
                attributes.set_prev_running_variance(inRunningVar);
            }
            if(momentum)
            {
                attributes.set_momentum(momentum);
            }
            return graph.batchnorm(
                std::move(input), std::move(scale), std::move(bias), std::move(attributes));
        },
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("bias"),
        nb::arg("in_running_mean").none(),
        nb::arg("in_running_var").none(),
        nb::arg("epsilon"),
        nb::arg("momentum").none(),
        nb::arg("peer_stats") = TensorList{},
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns [y, mean, inv_variance, next_running_mean, "
        "next_running_var].");

    cls.def(
        "batchnorm_backward",
        [](graph::Graph& graph,
           TensorPtr grad,
           TensorPtr input,
           TensorPtr scale,
           const TensorPtr& mean,
           const TensorPtr& invVariance,
           const TensorList& peerStats,
           DataType computeDataType,
           const std::string& name) {
            graph::BatchnormBackwardAttributes attributes;
            attributes.set_peer_stats(peerStats)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            if(mean)
            {
                attributes.set_mean(mean);
            }
            if(invVariance)
            {
                attributes.set_inv_variance(invVariance);
            }
            return graph.batchnorm_backward(
                std::move(grad), std::move(input), std::move(scale), std::move(attributes));
        },
        nb::arg("grad"),
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("mean").none(),
        nb::arg("inv_variance").none(),
        nb::arg("peer_stats") = TensorList{},
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns [dx, dscale, dbias].");

    cls.def(
        "batchnorm_inference",
        [](graph::Graph& graph,
           TensorPtr input,
           TensorPtr mean,
           TensorPtr invVariance,
           TensorPtr scale,
           TensorPtr bias,
           DataType computeDataType,
           const std::string& name) {
            graph::BatchnormInferenceAttributes attributes;
            attributes.set_compute_data_type(computeDataType).set_name(name);
            return graph.batchnorm_inference(std::move(input),
                                             std::move(mean),
                                             std::move(invVariance),
                                             std::move(scale),
                                             std::move(bias),
                                             std::move(attributes));
        },
        nb::arg("input"),
        nb::arg("mean"),
        nb::arg("inv_variance"),
        nb::arg("scale"),
        nb::arg("bias"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form.");

    defConv<graph::ConvFpropAttributes>(
        cls,
        "conv_fprop",
        "image",
        "weight",
        [](graph::Graph& graph,
           TensorPtr image,
           TensorPtr weight,
           graph::ConvFpropAttributes attributes) {
            return graph.conv_fprop(std::move(image), std::move(weight), std::move(attributes));
        });
    defConv<graph::ConvWgradAttributes>(
        cls,
        "conv_wgrad",
        "image",
        "loss",
        [](graph::Graph& graph,
           TensorPtr image,
           TensorPtr loss,
           graph::ConvWgradAttributes attributes) {
            return graph.conv_wgrad(std::move(loss), std::move(image), std::move(attributes));
        });
    defConv<graph::ConvDgradAttributes>(
        cls,
        "conv_dgrad",
        "loss",
        "filter",
        [](graph::Graph& graph,
           TensorPtr loss,
           TensorPtr filter,
           graph::ConvDgradAttributes attributes) {
            return graph.conv_dgrad(std::move(loss), std::move(filter), std::move(attributes));
        });

    cls.def(
        "matmul",
        [](graph::Graph& graph,
           TensorPtr a,
           TensorPtr b,
           DataType computeDataType,
           double padding,
           const std::string& name) {
            if(padding != 0.0)
            {
                hipdnn_python::throwUnsupportedParameter("padding", "0.0");
            }
            graph::MatmulAttributes attributes;
            attributes.set_compute_data_type(computeDataType).set_name(name);
            return graph.matmul(std::move(a), std::move(b), std::move(attributes));
        },
        nb::arg("A"),
        nb::arg("B"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("padding") = 0.0,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form; padding accepts only 0.0.");

    cls.def(
        "layernorm",
        [](graph::Graph& graph,
           NormFwdPhase normForwardPhase,
           TensorPtr input,
           TensorPtr scale,
           TensorPtr bias,
           const TensorPtr& epsilon,
           DataType computeDataType,
           const std::string& name) {
            graph::LayernormAttributes attributes;
            attributes.set_forward_phase(normForwardPhase)
                .set_epsilon(epsilon)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            const auto outputs = graph.layernorm(
                std::move(input), std::move(scale), std::move(bias), std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
        },
        nb::arg("norm_forward_phase"),
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("bias"),
        nb::arg("epsilon"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns (y, mean, inv_variance).");

    cls.def(
        "layernorm_backward",
        [](graph::Graph& graph,
           TensorPtr grad,
           TensorPtr input,
           TensorPtr scale,
           const TensorPtr& mean,
           const TensorPtr& invVariance,
           DataType computeDataType,
           const std::string& name,
           const TensorPtr& epsilon,
           std::optional<int64_t> normalizedDimCount) {
            graph::LayernormBackwardAttributes attributes;
            attributes.set_compute_data_type(computeDataType).set_name(name);
            if(mean)
            {
                attributes.set_mean(mean);
            }
            if(invVariance)
            {
                attributes.set_inv_variance(invVariance);
            }
            if(epsilon)
            {
                attributes.set_epsilon(epsilon);
            }
            if(normalizedDimCount)
            {
                attributes.set_normalized_dim_count(*normalizedDimCount);
            }
            const auto outputs = graph.layernorm_backward(
                std::move(grad), std::move(input), std::move(scale), std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
        },
        nb::arg("grad"),
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("mean") = nb::none(),
        nb::arg("inv_variance") = nb::none(),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        nb::kw_only(),
        nb::arg("epsilon") = nb::none(),
        nb::arg("normalized_dim_count") = nb::none(),
        "cuDNN-compatible keyword form. mean and inv_variance may both be omitted; epsilon "
        "and normalized_dim_count are hipDNN extensions. Returns (dx, dscale, dbias).");

    // cuDNN places the required epsilon after the defaulted bias. nanobind, like
    // pybind11, accepts that order, so positional cuDNN calls line up unchanged.
    cls.def(
        "rmsnorm",
        [](graph::Graph& graph,
           NormFwdPhase normForwardPhase,
           TensorPtr input,
           TensorPtr scale,
           const TensorPtr& bias,
           const TensorPtr& epsilon,
           DataType computeDataType,
           const std::string& name) {
            graph::RMSNormAttributes attributes;
            attributes.set_forward_phase(normForwardPhase)
                .set_epsilon(epsilon)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            if(bias)
            {
                attributes.set_bias(bias);
            }
            const auto outputs
                = graph.rmsnorm(std::move(input), std::move(scale), std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1]);
        },
        nb::arg("norm_forward_phase"),
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("bias") = nb::none(),
        nb::arg("epsilon"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns (y, inv_rms).");

    cls.def(
        "rmsnorm_backward",
        [](graph::Graph& graph,
           TensorPtr grad,
           TensorPtr input,
           TensorPtr scale,
           TensorPtr invVariance,
           bool hasDbias,
           DataType computeDataType,
           const std::string& name) {
            graph::RMSNormBackwardAttributes attributes;
            attributes.set_compute_dbias(hasDbias)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            const auto outputs = graph.rmsnorm_backward(std::move(grad),
                                                        std::move(input),
                                                        std::move(scale),
                                                        std::move(invVariance),
                                                        std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
        },
        nb::arg("grad"),
        nb::arg("input"),
        nb::arg("scale"),
        nb::arg("inv_variance"),
        nb::arg("has_dbias"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns (dx, dscale, dbias).");

    cls.def(
        "block_scale_dequantize",
        [](graph::Graph& graph,
           TensorPtr input,
           TensorPtr descale,
           const std::vector<int32_t>& blockSize,
           bool isNegativeScale,
           DataType computeDataType,
           const std::string& name) {
            graph::BlockScaleDequantizeAttributes attributes;
            attributes.set_block_size(blockSize)
                .set_is_negative_scale(isNegativeScale)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            return graph.block_scale_dequantize(
                std::move(input), std::move(descale), std::move(attributes));
        },
        nb::arg("input"),
        nb::arg("descale"),
        nb::arg("block_size"),
        nb::arg("is_negative_scale") = false,
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form.");

    cls.def(
        "block_scale_quantize",
        [](graph::Graph& graph,
           TensorPtr input,
           int32_t blockSize,
           std::optional<int64_t> axis,
           bool transpose,
           DataType computeDataType,
           const std::string& name) {
            graph::BlockScaleQuantizeAttributes attributes;
            attributes.set_block_size(blockSize)
                .set_transpose(transpose)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            if(axis)
            {
                attributes.set_axis(*axis);
            }
            const auto outputs
                = graph.block_scale_quantize(std::move(input), std::move(attributes));
            return nb::make_tuple(outputs[0], outputs[1]);
        },
        nb::arg("input"),
        nb::arg("block_size"),
        nb::arg("axis") = nb::none(),
        nb::arg("transpose") = false,
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form. Returns (y, scale).");

    cls.def(
        "reduction",
        [](graph::Graph& graph,
           TensorPtr input,
           ReductionMode mode,
           DataType computeDataType,
           const std::string& name) {
            graph::ReductionAttributes attributes;
            attributes.set_mode(mode).set_compute_data_type(computeDataType).set_name(name);
            return graph.reduction(std::move(input), std::move(attributes));
        },
        nb::arg("input"),
        nb::arg("mode"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form.");

    cls.def(
        "moe_grouped_matmul",
        [](graph::Graph& graph,
           TensorPtr token,
           TensorPtr weight,
           TensorPtr firstTokenOffset,
           TensorPtr tokenIndex,
           TensorPtr tokenKs,
           MoeGroupedMatmulMode mode,
           DataType computeDataType,
           int32_t topK,
           const std::string& name) {
            graph::MoeGroupedMatmulAttributes attributes;
            attributes.set_mode(mode)
                .set_top_k(topK)
                .set_compute_data_type(computeDataType)
                .set_name(name);
            return graph.moe_grouped_matmul(std::move(token),
                                            std::move(weight),
                                            std::move(firstTokenOffset),
                                            std::move(tokenIndex),
                                            std::move(tokenKs),
                                            std::move(attributes));
        },
        nb::arg("token"),
        nb::arg("weight"),
        nb::arg("first_token_offset"),
        nb::arg("token_index") = nb::none(),
        nb::arg("token_ks") = nb::none(),
        nb::arg("mode") = MoeGroupedMatmulMode::NONE,
        nb::arg("compute_data_type") = DataType::FLOAT,
        nb::arg("top_k") = 0,
        nb::arg("name") = "",
        "cuDNN-compatible keyword form.");

#ifdef HIPDNN_ENABLE_SDPA
    sdpaBindings(cls);
#endif
}
