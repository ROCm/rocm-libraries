// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file SdpaAttributes.hpp
 * @brief Attributes for scaled dot-product attention (SDPA) operation
 *
 * This file defines the SdpaAttributes class used to configure
 * scaled dot-product attention operations, computing
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class SdpaAttributes
 * @brief Configuration attributes for scaled dot-product attention
 *
 * SdpaAttributes configures a scaled dot-product attention operation:
 * @code
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * @endcode
 *
 * **Tensor Shapes (BHSD ordering):**
 * - **Q** (query): `(B, H, S_q, D)` — batch, heads, query sequence length, head dimension
 * - **K** (key): `(B, H_k, S_kv, D)` — batch, key heads, key/value sequence length, head dimension
 * - **V** (value): `(B, H_v, S_kv, D_v)` — batch, value heads, key/value sequence length, value head dimension
 * - **O** (output): `(B, H, S_q, D_v)` — batch, heads, query sequence length, value head dimension
 * - **Stats** (optional): `(B, H, S_q, 1)` — softmax statistics (when generate_stats is true)
 *
 * Memory layout is controlled via strides. Use `TensorLayout::BHSD` for row-major or
 * `TensorLayout::BSHD` for sequence-major layout.
 *
 * **Required inputs:**
 * - Q: Query tensor
 * - K: Key tensor (num_heads must divide evenly into Q's num_heads for GQA/MQA)
 * - V: Value tensor (seq_kv must match K)
 *
 * **Outputs:**
 * - O: Attention output tensor
 * - Stats: Softmax statistics (optional, when generate_stats is set)
 *
 * **Optional features:**
 * - Causal masking and diagonal band bounds
 * - Additive attention bias (attn_mask)
 * - Dropout (probability + seed/offset tensors)
 * - ALiBi positional encoding
 * - Paged attention (page_table_k, page_table_v)
 * - FP8 quantization (descale/scale tensors)
 * - Attention scale override (attn_scale_value)
 *
 * @code{.cpp}
 * SdpaAttributes attr;
 * attr.set_attn_scale_value(1.0f / std::sqrt(static_cast<float>(d_k)))
 *     .set_causal_mask(true);
 *
 * auto [o, stats] = graph.sdpa(q, k, v, attr);
 * @endcode
 */
class SdpaAttributes : public Attributes<SdpaAttributes>
{
public:
    SdpaAttributes() = default;

    // NOLINTBEGIN(readability-identifier-naming)
    enum class InputNames
    {
        Q = 0,
        K = 1,
        V = 2,
        Attn_scale = 3, // Attention scale tensor
        Bias = 4, // Additive attention bias
        SEQ_LEN_Q = 5,
        SEQ_LEN_KV = 6,
        Seed = 7, // Dropout seed
        Offset = 8, // Dropout offset
        Dropout_mask = 9,
        Dropout_scale = 10,
        Page_table_K = 11,
        Page_table_V = 12,
        Block_mask = 13,
        Descale_Q = 14,
        Descale_K = 15,
        Descale_V = 16,
        Descale_S = 17,
        Scale_S = 18,
        Scale_O = 19,
        SINK_TOKEN = 20,
    };
    typedef InputNames input_names;

    enum class OutputNames
    {
        O = 0,
        Stats = 1,
        Max = 2,
        Sum_exp = 3,
        RNG_DUMP = 4,
        Amax_S = 5,
        Amax_O = 6,
    };
    typedef OutputNames output_names;
    // NOLINTEND(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTBEGIN(readability-identifier-naming)
    // Boolean flags
    std::optional<bool> generate_stats;
    bool alibi_mask = false;
    bool padding_mask = false;
    bool causal_mask = false; // Deprecated
    bool causal_mask_bottom_right = false; // Deprecated

    // Scalar attributes
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;
    std::optional<int64_t> left_bound;
    std::optional<int64_t> right_bound;
    std::optional<int32_t> max_seq_len_kv;

    // Enum attributes
    DiagonalAlignment diagonal_alignment = DiagonalAlignment::TOP_LEFT;
    DataType mma_core_mode = DataType::NOT_SET;
    AttentionImplementation implementation = AttentionImplementation::AUTO;
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
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::Bias);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_attn_scale() const
    {
        return getInput(InputNames::Attn_scale);
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
    std::shared_ptr<TensorAttributes> get_page_table_k() const
    {
        return getInput(InputNames::Page_table_K);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_page_table_v() const
    {
        return getInput(InputNames::Page_table_V);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_block_mask() const
    {
        return getInput(InputNames::Block_mask);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_sink_token() const
    {
        return getInput(InputNames::SINK_TOKEN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_descale_q() const
    {
        return getInput(InputNames::Descale_Q);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_descale_k() const
    {
        return getInput(InputNames::Descale_K);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_descale_v() const
    {
        return getInput(InputNames::Descale_V);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_descale_s() const
    {
        return getInput(InputNames::Descale_S);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale_s() const
    {
        return getInput(InputNames::Scale_S);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale_o() const
    {
        return getInput(InputNames::Scale_O);
    }

    // -- Output tensor getters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_o() const
    {
        return getOutput(OutputNames::O);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_stats() const
    {
        return getOutput(OutputNames::Stats);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_max() const
    {
        return getOutput(OutputNames::Max);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_sum_exp() const
    {
        return getOutput(OutputNames::Sum_exp);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_rng_dump() const
    {
        return getOutput(OutputNames::RNG_DUMP);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_amax_s() const
    {
        return getOutput(OutputNames::Amax_S);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_amax_o() const
    {
        return getOutput(OutputNames::Amax_O);
    }

    // -- Input tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::K, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::K, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::V, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::V, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Bias, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Bias, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_attn_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Attn_scale, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_attn_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Attn_scale, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seq_len_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seq_len_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seq_len_kv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seq_len_kv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seed(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Seed, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_seed(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Seed, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Offset, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Offset, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout_mask(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Dropout_mask, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout_mask(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Dropout_mask, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Dropout_scale, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Dropout_scale, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_page_table_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Page_table_K, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_page_table_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Page_table_K, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_page_table_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Page_table_V, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_page_table_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Page_table_V, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_block_mask(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Block_mask, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_block_mask(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Block_mask, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_sink_token(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SINK_TOKEN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_sink_token(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SINK_TOKEN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Descale_Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Descale_Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Descale_K, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Descale_K, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Descale_V, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Descale_V, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_s(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Descale_S, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_descale_s(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Descale_S, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_scale_s(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Scale_S, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_scale_s(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Scale_S, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_scale_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Scale_O, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_scale_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Scale_O, std::move(value));
    }

    // -- Output tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::O, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::O, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_stats(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Stats, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_stats(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Stats, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_max(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Max, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_max(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Max, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_sum_exp(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Sum_exp, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_sum_exp(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Sum_exp, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_rng_dump(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::RNG_DUMP, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_rng_dump(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::RNG_DUMP, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_amax_s(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Amax_S, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_amax_s(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Amax_S, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_amax_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Amax_O, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_amax_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Amax_O, std::move(value));
    }

    // -- Scalar/flag setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_generate_stats(bool value)
    {
        generate_stats = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_alibi_mask(bool value)
    {
        alibi_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_padding_mask(bool value)
    {
        padding_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_causal_mask(bool value)
    {
        causal_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_causal_mask_bottom_right(bool value)
    {
        causal_mask_bottom_right = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout(float probability,
                                const std::shared_ptr<TensorAttributes>& seed,
                                const std::shared_ptr<TensorAttributes>& offset)
    {
        dropout_probability = probability;
        setInput(InputNames::Seed, seed);
        setInput(InputNames::Offset, offset);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_dropout_probability(float probability)
    {
        dropout_probability = probability;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_attn_scale_value(float value)
    {
        attn_scale_value = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_diagonal_band_left_bound(int64_t value)
    {
        left_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_diagonal_band_right_bound(int64_t value)
    {
        right_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_paged_attention_max_seq_len_kv(int32_t value)
    {
        max_seq_len_kv = value;
        return *this;
    }
    /// @brief Set the diagonal alignment for causal masking
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_diagonal_alignment(DiagonalAlignment value)
    {
        diagonal_alignment = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_mma_core_mode(DataType value)
    {
        mma_core_mode = value;
        return *this;
    }
    /// @brief Set the execution strategy for SDPA
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaAttributes& set_implementation(AttentionImplementation value)
    {
        implementation = value;
        return *this;
    }
};

typedef SdpaAttributes SDPA_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
