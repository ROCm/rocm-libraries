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
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <string>
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
 * attr.set_attn_scale(1.0f / std::sqrt(static_cast<float>(d_k)))
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
    SdpaBackwardAttributes() = default;

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
    SdpaBackwardAttributes& set_attn_scale(float value)
    {
        attn_scale_value = value;
        return *this;
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

    // -- cuDNN parity setters --
    // Each accepts the cuDNN frontend spelling/overload/semantics that is more
    // than a straight rename of a native setter (overload merge, one-to-many
    // split, or a capability hipDNN lacks).

    // cuDNN set_sliding_window_length(n) forwards to the left diagonal-band bound.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_sliding_window_length(int value)
    {
        return set_diagonal_band_left_bound(value);
    }
    // cuDNN backward set_dropout(mask, scale, scale_inv) is one fused call
    // carrying the extra inverse-scale; hipDNN sets the three tensors separately.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout(std::shared_ptr<TensorAttributes> mask,
                                        std::shared_ptr<TensorAttributes> scale,
                                        std::shared_ptr<TensorAttributes> scaleInv)
    {
        set_dropout_mask(std::move(mask));
        set_dropout_scale(std::move(scale));
        set_dropout_scale_inv(std::move(scaleInv));
        return *this;
    }
    // cuDNN's programmable score modifier and its backprop are callbacks over the
    // graph; hipDNN has no equivalent. Accepted for source compatibility,
    // recorded as unsupported so the graph fails loudly at validate(). Templated
    // to accept any callable without naming the cuDNN std::function type.
    template <typename ScoreModifier>
    SdpaBackwardAttributes&
        set_score_mod(ScoreModifier&& value) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA score modifier is unsupported by hipDNN");
    }
    template <typename ScoreModifier>
    SdpaBackwardAttributes&
        set_score_mod_bprop(ScoreModifier&& value) // NOLINT(readability-identifier-naming)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA score-modifier backprop is unsupported by hipDNN");
    }
    // cuDNN nested-tensor / ragged-batch max-total-seq-len hints have no hipDNN
    // equivalent yet.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_max_total_seq_len_q(int64_t value)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA max_total_seq_len_q is unsupported by hipDNN");
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_max_total_seq_len_kv(int64_t value)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA max_total_seq_len_kv is unsupported by hipDNN");
    }
    // Determinism is correctness-critical; hipDNN cannot guarantee it, so a true
    // request fails loudly rather than silently running a non-deterministic
    // kernel. Ignoring false is safe.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_deterministic_algorithm(bool value)
    {
        if(value)
        {
            return recordUnsupported("Deterministic SDPA backward is unsupported by hipDNN");
        }
        return *this;
    }
    // cuDNN backward exposes rng_dump; hipDNN's backward attributes do not.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_rng_dump(const std::shared_ptr<TensorAttributes>& value)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA backward RNG dump is unsupported by hipDNN");
    }
    // Attention-sink tokens (fwd input + bwd gradient) have no hipDNN backward
    // equivalent.
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_sink_token(const std::shared_ptr<TensorAttributes>& value)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA backward sink token is unsupported by hipDNN");
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dsink_token(const std::shared_ptr<TensorAttributes>& value)
    {
        static_cast<void>(value);
        return recordUnsupported("SDPA backward sink-token gradient is unsupported by hipDNN");
    }
};

typedef SdpaBackwardAttributes SDPA_backward_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
