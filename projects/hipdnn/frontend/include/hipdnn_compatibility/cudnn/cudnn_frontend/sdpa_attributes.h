// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h), used under the MIT license.

/**
 * @file sdpa_attributes.h
 * @brief cuDNN-shaped SDPA attribute wrappers for the hipDNN compatibility shim.
 *
 * Unlike most of the shim, the SDPA attribute types are NOT zero-cost `using`
 * aliases: the cuDNN FE `SDPA_attributes` / `SDPA_backward_attributes` surface
 * diverges from hipDNN's `SdpaAttributes` / `SdpaBackwardAttributes` by method
 * name, by semantics, and by a few methods hipDNN lacks entirely. So each type
 * is a header-only composition wrapper holding the hipDNN attrs by value and
 * re-exposing the cuDNN spelling (a residual composition wrapper).
 *
 * DIVERGENCE INVENTORY — every wrapper setter that is not a straight 1:1 forward
 * carries an in-code `SHIM-DIVERGENCE(<category>)` tag; grep for it to get the
 * full list. Categories: RENAME (same behavior, different name), SEMANTIC
 * (behavior remap), SPLIT (one cuDNN call → several hipDNN calls), MISSING (no
 * hipDNN equivalent; shim records an error or warns). Each is a candidate for
 * changing hipDNN's own frontend so the wrapper can collapse back to an alias —
 * that upstream change is deliberately NOT made here, only flagged.
 *
 * SDPA_attributes divergences:
 *   RENAME   set_attn_scale(float)          -> set_attn_scale_value(float)
 *   RENAME   set_logit_max                  -> set_max
 *   RENAME   set_score_sum_exp              -> set_sum_exp
 *   RENAME   set_paged_attention_k_table    -> set_page_table_k
 *   RENAME   set_paged_attention_v_table    -> set_page_table_v
 *   RENAME   set_sliding_window_length(int) -> set_diagonal_band_left_bound
 *   RENAME   _set_mma_core_mode             -> set_mma_core_mode
 *   SEMANTIC set_is_inference(bool)         -> set_generate_stats(!value) [deprecated]
 *   SPLIT    set_dropout(mask, scale)       -> set_dropout_mask + set_dropout_scale
 *   MISSING  set_score_mod                  -> (no hipDNN equivalent) record error
 *   MISSING  set_unfuse_fma                 -> (perf hint) warn + ignore
 *
 * SDPA_backward_attributes divergences:
 *   RENAME   set_attn_scale(float)          -> set_attn_scale_value(float)
 *   RENAME   set_sliding_window_length(int) -> set_diagonal_band_left_bound
 *   SPLIT    set_dropout(mask, scale, inv)  -> set_dropout_mask + _scale + _scale_inv
 *   MISSING  set_score_mod / _bprop         -> record error
 *   MISSING  set_max_total_seq_len_q/kv     -> record error
 *   MISSING  set_deterministic_algorithm    -> record error when true
 *   MISSING  set_rng_dump                   -> record error
 *   MISSING  set_sink_token / set_dsink_token -> record error
 *
 * @note Internal-to-shim; pulled in by the umbrella under HIPDNN_ENABLE_SDPA.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

#ifdef HIPDNN_ENABLE_SDPA

#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming)

class Graph; // completed in detail/graph_wrapper.h; friended for underlying-attrs access

// cuDNN FE's programmable score modifier is a std::function over the graph
// object. hipDNN has no equivalent, so the shim accepts the type for source
// compatibility but records an error when it is actually used (see set_score_mod).
using AttentionScoreModifier_t = std::function<std::shared_ptr<Tensor_attributes>(
    std::shared_ptr<Graph>, std::shared_ptr<Tensor_attributes>)>;

class SDPA_attributes
{
public:
    SDPA_attributes& set_name(const std::string& value)
    {
        _attrs.set_name(value);
        return *this;
    }

    SDPA_attributes& set_compute_data_type(DataType_t value)
    {
        _attrs.set_compute_data_type(value);
        return *this;
    }

    SDPA_attributes& set_q(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_q(std::move(value));
        return *this;
    }

    SDPA_attributes& set_k(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_k(std::move(value));
        return *this;
    }

    SDPA_attributes& set_v(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_v(std::move(value));
        return *this;
    }

    SDPA_attributes& set_bias(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_bias(std::move(value));
        return *this;
    }

    SDPA_attributes& set_attn_scale(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_attn_scale(std::move(value));
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_attn_scale(float); hipDNN spells the
    // scalar overload set_attn_scale_value(float). hipDNN could add the float
    // overload of set_attn_scale so this collapses to a straight forward.
    SDPA_attributes& set_attn_scale(float value)
    {
        _attrs.set_attn_scale_value(value);
        return *this;
    }

    SDPA_attributes& set_generate_stats(bool value)
    {
        _attrs.set_generate_stats(value);
        return *this;
    }

    // SHIM-DIVERGENCE(SEMANTIC): cuDNN [[deprecated]] set_is_inference(b) means
    // "no stats"; maps to hipDNN set_generate_stats(!b). PyTorch still emits this
    // on CUDNN_FRONTEND_VERSION <= 11200 paths. hipDNN could add a deprecated
    // set_is_inference for source parity.
    [[deprecated("use set_generate_stats(!value)")]] SDPA_attributes& set_is_inference(bool value)
    {
        _attrs.set_generate_stats(!value);
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_logit_max; hipDNN spells it set_max.
    SDPA_attributes& set_logit_max(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_max(std::move(value));
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_score_sum_exp; hipDNN spells it set_sum_exp.
    SDPA_attributes& set_score_sum_exp(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_sum_exp(std::move(value));
        return *this;
    }

    SDPA_attributes& set_block_mask(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_block_mask(std::move(value));
        return *this;
    }

    SDPA_attributes& set_alibi_mask(bool value)
    {
        _attrs.set_alibi_mask(value);
        return *this;
    }

    SDPA_attributes& set_padding_mask(bool value)
    {
        _attrs.set_padding_mask(value);
        return *this;
    }

    SDPA_attributes& set_causal_mask(bool value)
    {
        _attrs.set_causal_mask(value);
        return *this;
    }

    SDPA_attributes& set_causal_mask_bottom_right(bool value)
    {
        _attrs.set_causal_mask_bottom_right(value);
        return *this;
    }

    SDPA_attributes& set_seq_len_q(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_seq_len_q(std::move(value));
        return *this;
    }

    SDPA_attributes& set_seq_len_kv(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_seq_len_kv(std::move(value));
        return *this;
    }

    SDPA_attributes& set_diagonal_alignment(DiagonalAlignment_t value)
    {
        _attrs.set_diagonal_alignment(value);
        return *this;
    }

    SDPA_attributes& set_diagonal_band_left_bound(int value)
    {
        _attrs.set_diagonal_band_left_bound(value);
        return *this;
    }

    SDPA_attributes& set_diagonal_band_right_bound(int value)
    {
        _attrs.set_diagonal_band_right_bound(value);
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_sliding_window_length(n) forwards, per
    // upstream, to the left diagonal-band bound. hipDNN has no sliding-window
    // spelling; it could add one that maps to the same bound.
    SDPA_attributes& set_sliding_window_length(int value)
    {
        _attrs.set_diagonal_band_left_bound(value);
        return *this;
    }

    SDPA_attributes& set_dropout(float probability,
                                 std::shared_ptr<Tensor_attributes> seed,
                                 std::shared_ptr<Tensor_attributes> offset)
    {
        _attrs.set_dropout(probability, std::move(seed), std::move(offset));
        return *this;
    }

    // SHIM-DIVERGENCE(SPLIT): cuDNN set_dropout(mask, scale) is one call; hipDNN
    // sets the mask and scale separately. hipDNN could add the fused overload.
    SDPA_attributes& set_dropout(std::shared_ptr<Tensor_attributes> mask,
                                 std::shared_ptr<Tensor_attributes> scale)
    {
        _attrs.set_dropout_mask(std::move(mask));
        _attrs.set_dropout_scale(std::move(scale));
        return *this;
    }

    SDPA_attributes& set_rng_dump(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_rng_dump(std::move(value));
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_paged_attention_k_table; hipDNN spells
    // it set_page_table_k.
    SDPA_attributes& set_paged_attention_k_table(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_page_table_k(std::move(value));
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_paged_attention_v_table; hipDNN spells
    // it set_page_table_v.
    SDPA_attributes& set_paged_attention_v_table(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_page_table_v(std::move(value));
        return *this;
    }

    SDPA_attributes& set_paged_attention_max_seq_len_kv(int value)
    {
        _attrs.set_paged_attention_max_seq_len_kv(value);
        return *this;
    }

    SDPA_attributes& set_sink_token(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_sink_token(std::move(value));
        return *this;
    }

    SDPA_attributes& set_implementation(AttentionImplementation_t value)
    {
        _attrs.set_implementation(value);
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN's internal _set_mma_core_mode; hipDNN spells
    // it set_mma_core_mode.
    SDPA_attributes& _set_mma_core_mode(DataType_t value)
    {
        _attrs.set_mma_core_mode(value);
        return *this;
    }

    // SHIM-DIVERGENCE(MISSING): cuDNN's programmable score modifier has no hipDNN
    // equivalent. Accepted for source compatibility; records an error so the
    // graph fails loudly at validate() rather than silently dropping the callback.
    SDPA_attributes& set_score_mod(AttentionScoreModifier_t value)
    {
        static_cast<void>(value);
        return record("SDPA score modifier is unsupported by this shim");
    }

    // SHIM-DIVERGENCE(MISSING): cuDNN FMA-unfuse perf hint; hipDNN has no such
    // knob. Advisory and safe to drop, so warn and ignore.
    SDPA_attributes& set_unfuse_fma(bool value)
    {
        static_cast<void>(value);
        CUDNN_FE_LOG_LABEL("Ignoring SDPA unfuse-FMA hint; hipDNN selects fusion internally");
        return *this;
    }

private:
    friend class Graph;

    SDPA_attributes& record(const char* message)
    {
        if(!_recordedError.has_value())
        {
            _recordedError = error_t{error_code_t::INVALID_VALUE, message};
        }
        return *this;
    }

    hipdnn_frontend::graph::SdpaAttributes _attrs;
    std::optional<error_t> _recordedError;
};

class SDPA_backward_attributes
{
public:
    SDPA_backward_attributes& set_name(const std::string& value)
    {
        _attrs.set_name(value);
        return *this;
    }

    SDPA_backward_attributes& set_compute_data_type(DataType_t value)
    {
        _attrs.set_compute_data_type(value);
        return *this;
    }

    SDPA_backward_attributes& set_attn_scale(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_attn_scale(std::move(value));
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_attn_scale(float); hipDNN spells the
    // scalar overload set_attn_scale_value(float).
    SDPA_backward_attributes& set_attn_scale(float value)
    {
        _attrs.set_attn_scale_value(value);
        return *this;
    }

    SDPA_backward_attributes& set_bias(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_bias(std::move(value));
        return *this;
    }

    SDPA_backward_attributes& set_dbias(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_dbias(std::move(value));
        return *this;
    }

    SDPA_backward_attributes& set_alibi_mask(bool value)
    {
        _attrs.set_alibi_mask(value);
        return *this;
    }

    SDPA_backward_attributes& set_padding_mask(bool value)
    {
        _attrs.set_padding_mask(value);
        return *this;
    }

    SDPA_backward_attributes& set_causal_mask(bool value)
    {
        _attrs.set_causal_mask(value);
        return *this;
    }

    SDPA_backward_attributes& set_causal_mask_bottom_right(bool value)
    {
        _attrs.set_causal_mask_bottom_right(value);
        return *this;
    }

    SDPA_backward_attributes& set_seq_len_q(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_seq_len_q(std::move(value));
        return *this;
    }

    SDPA_backward_attributes& set_seq_len_kv(std::shared_ptr<Tensor_attributes> value)
    {
        _attrs.set_seq_len_kv(std::move(value));
        return *this;
    }

    SDPA_backward_attributes& set_diagonal_alignment(DiagonalAlignment_t value)
    {
        _attrs.set_diagonal_alignment(value);
        return *this;
    }

    SDPA_backward_attributes& set_diagonal_band_left_bound(int value)
    {
        _attrs.set_diagonal_band_left_bound(value);
        return *this;
    }

    SDPA_backward_attributes& set_diagonal_band_right_bound(int value)
    {
        _attrs.set_diagonal_band_right_bound(value);
        return *this;
    }

    // SHIM-DIVERGENCE(RENAME): cuDNN set_sliding_window_length(n) forwards to the
    // left diagonal-band bound, per upstream.
    SDPA_backward_attributes& set_sliding_window_length(int value)
    {
        _attrs.set_diagonal_band_left_bound(value);
        return *this;
    }

    SDPA_backward_attributes& set_dropout(float probability,
                                          std::shared_ptr<Tensor_attributes> seed,
                                          std::shared_ptr<Tensor_attributes> offset)
    {
        _attrs.set_dropout(probability, std::move(seed), std::move(offset));
        return *this;
    }

    // SHIM-DIVERGENCE(SPLIT): cuDNN backward set_dropout(mask, scale, scale_inv)
    // is one call carrying the extra inverse-scale (vs the 2-arg forward form);
    // hipDNN sets the three tensors separately.
    SDPA_backward_attributes& set_dropout(std::shared_ptr<Tensor_attributes> mask,
                                          std::shared_ptr<Tensor_attributes> scale,
                                          std::shared_ptr<Tensor_attributes> scaleInv)
    {
        _attrs.set_dropout_mask(std::move(mask));
        _attrs.set_dropout_scale(std::move(scale));
        _attrs.set_dropout_scale_inv(std::move(scaleInv));
        return *this;
    }

    // SHIM-DIVERGENCE(MISSING): programmable score modifier (fwd + bprop) — no
    // hipDNN equivalent; record an error so use fails loudly.
    SDPA_backward_attributes& set_score_mod(AttentionScoreModifier_t value)
    {
        static_cast<void>(value);
        return record("SDPA score modifier is unsupported by this shim");
    }

    SDPA_backward_attributes& set_score_mod_bprop(AttentionScoreModifier_t value)
    {
        static_cast<void>(value);
        return record("SDPA score-modifier backprop is unsupported by this shim");
    }

    // SHIM-DIVERGENCE(MISSING): nested-tensor max-total-seq-len hints have no
    // hipDNN equivalent yet. hipDNN could add them to support ragged batches.
    SDPA_backward_attributes& set_max_total_seq_len_q(int64_t value)
    {
        static_cast<void>(value);
        return record("SDPA max_total_seq_len_q is unsupported by this shim");
    }

    SDPA_backward_attributes& set_max_total_seq_len_kv(int64_t value)
    {
        static_cast<void>(value);
        return record("SDPA max_total_seq_len_kv is unsupported by this shim");
    }

    // SHIM-DIVERGENCE(MISSING): determinism is correctness-critical — the shim
    // cannot guarantee it, so it errors when requested rather than silently
    // running a non-deterministic kernel. Ignoring false is safe.
    SDPA_backward_attributes& set_deterministic_algorithm(bool value)
    {
        if(value)
        {
            return record("Deterministic SDPA backward is unsupported by this shim");
        }
        return *this;
    }

    // SHIM-DIVERGENCE(MISSING): cuDNN backward exposes rng_dump; hipDNN's
    // backward attributes do not.
    SDPA_backward_attributes& set_rng_dump(std::shared_ptr<Tensor_attributes> value)
    {
        static_cast<void>(value);
        return record("SDPA backward RNG dump is unsupported by this shim");
    }

    // SHIM-DIVERGENCE(MISSING): attention-sink tokens (fwd input + bwd gradient)
    // have no hipDNN backward equivalent.
    SDPA_backward_attributes& set_sink_token(std::shared_ptr<Tensor_attributes> value)
    {
        static_cast<void>(value);
        return record("SDPA backward sink token is unsupported by this shim");
    }

    SDPA_backward_attributes& set_dsink_token(std::shared_ptr<Tensor_attributes> value)
    {
        static_cast<void>(value);
        return record("SDPA backward sink-token gradient is unsupported by this shim");
    }

private:
    friend class Graph;

    SDPA_backward_attributes& record(const char* message)
    {
        if(!_recordedError.has_value())
        {
            _recordedError = error_t{error_code_t::INVALID_VALUE, message};
        }
        return *this;
    }

    hipdnn_frontend::graph::SdpaBackwardAttributes _attrs;
    std::optional<error_t> _recordedError;
};

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph

#endif // HIPDNN_ENABLE_SDPA
