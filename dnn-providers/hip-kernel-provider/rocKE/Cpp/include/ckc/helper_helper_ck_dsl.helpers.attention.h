/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_ck_dsl.helpers.attention.h -- C99 port of a second selection of
 * symbols from ck_dsl/helpers/attention.py (companion to
 * helper_ck_dsl.helpers.attention.h, which ports the mask / inv-l / stage-count
 * surface). The two ports share no symbol.
 *
 * PORTED SYMBOLS (this phase):
 *   - ckc_apply_softcap_log2        (apply_softcap_log2)
 *   - ckc_binary_search_seq_idx     (binary_search_seq_idx)
 *   - ckc_mfma_16x16x16_for_dtype   (mfma_16x16x16_for_dtype)
 *   - ckc_wave64_reduce_max         (wave64_reduce_max)
 *   - ckc_wave64_reduce_sum         (wave64_reduce_sum)
 *
 * Each IR-emitting helper reproduces its Python counterpart's ckc_b_* builder-
 * call sequence byte-faithfully (same ops, same order, same operands), binding
 * only to ckc/ir.h's public surface plus the companion helper header.
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef CKC_HELPER_HELPER_CK_DSL_HELPERS_ATTENTION_H
#define CKC_HELPER_HELPER_CK_DSL_HELPERS_ATTENTION_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------- softcap (log2-domain)
 *
 * apply_softcap_log2(b, score_log2, softcap) ->
 *     softcap * tanh(score_natural / softcap) computed via exp2 only.
 * Returns NULL only if the builder's sticky error is already/becomes set. */
ckc_value_t*
    ckc_apply_softcap_log2(ckc_ir_builder_t* b, ckc_value_t* score_log2, ckc_value_t* softcap);

/* ------------------------------------------------------- MFMA dtype dispatch
 *
 * mfma_16x16x16_for_dtype(b, dtype, a, bv, c): dispatch
 * mfma_f32_16x16x16_<dtype> for f16 / bf16. Any other dtype sets the builder's
 * sticky error (CKC_ERR_VALUE) and returns NULL. `dtype` must be non-NULL. */
ckc_value_t* ckc_mfma_16x16x16_for_dtype(
    ckc_ir_builder_t* b, const ckc_type_t* dtype, ckc_value_t* a, ckc_value_t* bv, ckc_value_t* c);

/* ------------------------------------------------- wave64 cross-lane reduction
 *
 * Six-stage XOR butterfly (masks 1,2,4,8,16,32) over all 64 lanes of a wave.
 * After the call every lane holds the wave-wide max / sum. */
ckc_value_t* ckc_wave64_reduce_max(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_wave64_reduce_sum(ckc_ir_builder_t* b, ckc_value_t* v);

/* ----------------------------------------------- binary search on cu_q
 *
 * Triton-style binary search for the seq_idx owning q_block_global_idx, mirroring
 * aiter's find_seq_idx. `block_q` is the BLOCK_Q divisor; `iterations` is the
 * fixed loop trip count specialized from the batch size; `per_token` selects the
 * per-token comparison mode (cu_q[s] <= q_token). Returns the result Value
 * (loop.results[0] - 1), or NULL if the loop op failed to materialize. */
ckc_value_t* ckc_binary_search_seq_idx(ckc_ir_builder_t* b,
                                       ckc_value_t* cu_q,
                                       ckc_value_t* q_block_global_idx,
                                       ckc_value_t* num_seqs,
                                       int block_q,
                                       int iterations,
                                       bool per_token);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_HELPER_CK_DSL_HELPERS_ATTENTION_H */
