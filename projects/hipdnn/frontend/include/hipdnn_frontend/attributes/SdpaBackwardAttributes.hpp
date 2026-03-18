// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file SdpaBackwardAttributes.hpp
 * @brief Attributes for scaled dot-product attention (SDPA) backward pass
 *
 * This file defines the SdpaBackwardAttributes class used to configure
 * the backward pass (gradient computation) of scaled dot-product attention.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class SdpaBackwardAttributes
 * @brief Configuration attributes for scaled dot-product attention backward pass
 *
 * SdpaBackwardAttributes configures the backward pass of scaled dot-product
 * attention, computing gradients with respect to Q, K, and V:
 * @code
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * @endcode
 *
 * **Required inputs:**
 * - Q: Query tensor from forward pass [B, H, S_q, D]
 * - K: Key tensor from forward pass [B, H, S_kv, D]
 * - V: Value tensor from forward pass [B, H, S_kv, D_v]
 * - O: Output tensor from forward pass [B, H, S_q, D_v]
 * - dO: Gradient of loss w.r.t. output [B, H, S_q, D_v]
 * - Stats: Softmax statistics from forward pass [B, H, S_q, 1]
 *
 * **Outputs:**
 * - dQ: Gradient w.r.t. query [B, H, S_q, D]
 * - dK: Gradient w.r.t. key [B, H, S_kv, D]
 * - dV: Gradient w.r.t. value [B, H, S_kv, D_v]
 *
 * **Optional features:**
 * - Causal masking and diagonal band bounds
 * - Additive attention bias gradient (dBias)
 * - Dropout (probability + seed/offset tensors, or explicit mask)
 * - ALiBi positional encoding
 * - Attention scale override (attn_scale_value)
 *
 * @code{.cpp}
 * SdpaBackwardAttributes attr;
 * attr.set_attn_scale_value(1.0f / std::sqrt(static_cast<float>(d_k)))
 *     .set_causal_mask(true);
 *
 * auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, do_, stats, attr);
 * @endcode
 *
 * @see SdpaAttributes for the forward pass
 */
class SdpaBackwardAttributes : public Attributes<SdpaBackwardAttributes>
{
public:
    // NOLINTBEGIN(readability-identifier-naming)
    enum class InputNames
    {
        Q = 0,
        K = 1,
        V = 2,
        O = 3,
        dO = 4, // Gradient of output (dO)
        Stats = 5, // Softmax statistics from forward pass
        Attn_scale = 6, // Attention scale tensor
        Bias = 7, // Additive attention bias (Bias)
        SEQ_LEN_Q = 8,
        SEQ_LEN_KV = 9,
        Seed = 10, // Dropout seed
        Offset = 11, // Dropout offset
        Dropout_mask = 12,
        Dropout_scale = 13,
        Dropout_scale_inv = 14,
    };
    typedef InputNames input_names;

    enum class OutputNames
    {
        dQ = 0, // Gradient w.r.t. query
        dK = 1, // Gradient w.r.t. key
        dV = 2, // Gradient w.r.t. value
        dBias = 3, // Gradient w.r.t. additive attention bias (optional)
    };
    typedef OutputNames output_names;
    // NOLINTEND(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTBEGIN(readability-identifier-naming)
    // Boolean flags
    bool alibi_mask = false;
    bool padding_mask = false;
    bool causal_mask = false; // Deprecated
    bool causal_mask_bottom_right = false; // Deprecated

    // Scalar attributes
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;
    std::optional<int64_t> left_bound;
    std::optional<int64_t> right_bound;

    // Enum attributes
    DiagonalAlignment diagonal_alignment = DiagonalAlignment::TOP_LEFT;
    // NOLINTEND(readability-identifier-naming)

    // -- Input tensor getters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_q() const
    {
        return getInput(InputNames::Q);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_k() const
    {
        return getInput(InputNames::K);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_v() const
    {
        return getInput(InputNames::V);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_o() const
    {
        return getInput(InputNames::O);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_do() const
    {
        return getInput(InputNames::dO);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_stats() const
    {
        return getInput(InputNames::Stats);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_attn_scale() const
    {
        return getInput(InputNames::Attn_scale);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::Bias);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seq_len_q() const
    {
        return getInput(InputNames::SEQ_LEN_Q);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seq_len_kv() const
    {
        return getInput(InputNames::SEQ_LEN_KV);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seed() const
    {
        return getInput(InputNames::Seed);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_offset() const
    {
        return getInput(InputNames::Offset);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_mask() const
    {
        return getInput(InputNames::Dropout_mask);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_scale() const
    {
        return getInput(InputNames::Dropout_scale);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_scale_inv() const
    {
        return getInput(InputNames::Dropout_scale_inv);
    }

    // -- Output tensor getters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dq() const
    {
        return getOutput(OutputNames::dQ);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dk() const
    {
        return getOutput(OutputNames::dK);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dv() const
    {
        return getOutput(OutputNames::dV);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::dBias);
    }

    // -- Input tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::K, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::K, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::V, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::V, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::O, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::O, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_do(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::dO, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_do(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::dO, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_stats(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Stats, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_stats(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Stats, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Attn_scale, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Attn_scale, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Bias, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Bias, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_kv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_kv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seed(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Seed, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seed(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Seed, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Offset, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Offset, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_mask(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Dropout_mask, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_mask(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Dropout_mask, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Dropout_scale, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Dropout_scale, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale_inv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Dropout_scale_inv, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale_inv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Dropout_scale_inv, std::move(value));
    }

    // -- Output tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dq(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::dQ, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dq(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::dQ, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dk(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::dK, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dk(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::dK, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::dV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::dV, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::dBias, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::dBias, std::move(value));
    }

    // -- Scalar/flag setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_alibi_mask(bool value)
    {
        alibi_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_padding_mask(bool value)
    {
        padding_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_causal_mask(bool value)
    {
        causal_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_causal_mask_bottom_right(bool value)
    {
        causal_mask_bottom_right = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout(float probability,
                                        const std::shared_ptr<TensorAttributes>& seed,
                                        const std::shared_ptr<TensorAttributes>& offset)
    {
        dropout_probability = probability;
        setInput(InputNames::Seed, seed);
        setInput(InputNames::Offset, offset);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_scale_value(float value)
    {
        attn_scale_value = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_band_left_bound(int64_t value)
    {
        left_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_band_right_bound(int64_t value)
    {
        right_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_alignment(DiagonalAlignment value)
    {
        diagonal_alignment = value;
        return *this;
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        const auto optUid
            = [](const std::shared_ptr<TensorAttributes>& t) -> flatbuffers::Optional<int64_t> {
            return t ? flatbuffers::Optional<int64_t>(t->get_uid()) : flatbuffers::nullopt;
        };

        return hipdnn_data_sdk::data_objects::CreateSdpaBackwardAttributes(
            builder,
            get_q()->get_uid(),
            get_k()->get_uid(),
            get_v()->get_uid(),
            get_o()->get_uid(),
            get_do()->get_uid(),
            get_stats()->get_uid(),
            get_dq()->get_uid(),
            get_dk()->get_uid(),
            get_dv()->get_uid(),
            optUid(get_attn_scale()),
            optUid(get_bias()),
            optUid(get_seq_len_q()),
            optUid(get_seq_len_kv()),
            optUid(get_seed()),
            optUid(get_offset()),
            optUid(get_dropout_mask()),
            optUid(get_dropout_scale()),
            optUid(get_dropout_scale_inv()),
            optUid(get_dbias()),
            alibi_mask,
            padding_mask,
            causal_mask,
            causal_mask_bottom_right,
            dropout_probability,
            attn_scale_value,
            left_bound,
            right_bound,
            toSdkType(diagonal_alignment));
    }

    static SdpaBackwardAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::SdpaBackwardAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        SdpaBackwardAttributes attr;

        attr.set_q(tensorMap.at(fb->q_tensor_uid()));
        attr.set_k(tensorMap.at(fb->k_tensor_uid()));
        attr.set_v(tensorMap.at(fb->v_tensor_uid()));
        attr.set_o(tensorMap.at(fb->o_tensor_uid()));
        attr.set_do(tensorMap.at(fb->do_tensor_uid()));
        attr.set_stats(tensorMap.at(fb->stats_tensor_uid()));
        attr.set_dq(tensorMap.at(fb->dq_tensor_uid()));
        attr.set_dk(tensorMap.at(fb->dk_tensor_uid()));
        attr.set_dv(tensorMap.at(fb->dv_tensor_uid()));

        if(fb->scale_tensor_uid().has_value())
        {
            attr.set_attn_scale(tensorMap.at(fb->scale_tensor_uid().value()));
        }
        if(fb->attn_mask_tensor_uid().has_value())
        {
            attr.set_bias(tensorMap.at(fb->attn_mask_tensor_uid().value()));
        }
        if(fb->seq_len_q_tensor_uid().has_value())
        {
            attr.set_seq_len_q(tensorMap.at(fb->seq_len_q_tensor_uid().value()));
        }
        if(fb->seq_len_kv_tensor_uid().has_value())
        {
            attr.set_seq_len_kv(tensorMap.at(fb->seq_len_kv_tensor_uid().value()));
        }
        if(fb->seed_tensor_uid().has_value())
        {
            attr.set_seed(tensorMap.at(fb->seed_tensor_uid().value()));
        }
        if(fb->offset_tensor_uid().has_value())
        {
            attr.set_offset(tensorMap.at(fb->offset_tensor_uid().value()));
        }
        if(fb->dropout_mask_tensor_uid().has_value())
        {
            attr.set_dropout_mask(tensorMap.at(fb->dropout_mask_tensor_uid().value()));
        }
        if(fb->dropout_scale_tensor_uid().has_value())
        {
            attr.set_dropout_scale(tensorMap.at(fb->dropout_scale_tensor_uid().value()));
        }
        if(fb->dropout_scale_inv_tensor_uid().has_value())
        {
            attr.set_dropout_scale_inv(tensorMap.at(fb->dropout_scale_inv_tensor_uid().value()));
        }
        if(fb->dbias_tensor_uid().has_value())
        {
            attr.set_dbias(tensorMap.at(fb->dbias_tensor_uid().value()));
        }

        attr.alibi_mask = fb->alibi_mask();
        attr.padding_mask = fb->padding_mask();
        attr.causal_mask = fb->causal_mask();
        attr.causal_mask_bottom_right = fb->causal_mask_bottom_right();

        if(fb->dropout_probability().has_value())
        {
            attr.dropout_probability = fb->dropout_probability();
        }
        if(fb->attn_scale_value().has_value())
        {
            attr.attn_scale_value = fb->attn_scale_value();
        }
        if(fb->left_bound().has_value())
        {
            attr.left_bound = fb->left_bound();
        }
        if(fb->right_bound().has_value())
        {
            attr.right_bound = fb->right_bound();
        }

        attr.diagonal_alignment = fromSdkType(fb->diagonal_alignment());

        return attr;
    }
};

typedef SdpaBackwardAttributes SDPA_backward_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
