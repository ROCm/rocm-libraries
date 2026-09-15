/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_rocke.helpers.attention.h -- C99 port of a second selection of
 * symbols from rocke/helpers/attention.py (companion to
 * helper_rocke.helpers.attention.h, which ports the mask / inv-l / stage-count
 * surface). The two ports share no symbol.
 *
 * PORTED SYMBOLS (this phase):
 *   - rocke_apply_softcap_log2        (apply_softcap_log2)
 *   - rocke_binary_search_seq_idx     (binary_search_seq_idx)
 *   - rocke_mfma_16x16x16_for_dtype   (mfma_16x16x16_for_dtype)
 *   - rocke_wave64_reduce_max         (wave64_reduce_max)
 *   - rocke_wave64_reduce_sum         (wave64_reduce_sum)
 *
 * Each IR-emitting helper reproduces its Python counterpart's rocke_b_* builder-
 * call sequence byte-faithfully (same ops, same order, same operands), binding
 * only to rocke/ir.h's public surface plus the companion helper header.
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef ROCKE_HELPER_HELPER_ROCKE_HELPERS_ATTENTION_H
#define ROCKE_HELPER_HELPER_ROCKE_HELPERS_ATTENTION_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------- softcap (log2-domain)
 *
 * apply_softcap_log2(b, score_log2, softcap) ->
 *     softcap * tanh(score_natural / softcap) computed via exp2 only.
 * Returns NULL only if the builder's sticky error is already/becomes set. */
rocke_value_t* rocke_apply_softcap_log2(rocke_ir_builder_t* b,
                                        rocke_value_t* score_log2,
                                        rocke_value_t* softcap);

/* ------------------------------------------------------- MFMA dtype dispatch
 *
 * mfma_16x16x16_for_dtype(b, dtype, a, bv, c): dispatch
 * mfma_f32_16x16x16_<dtype> for f16 / bf16. Any other dtype sets the builder's
 * sticky error (ROCKE_ERR_VALUE) and returns NULL. `dtype` must be non-NULL. */
rocke_value_t* rocke_mfma_16x16x16_for_dtype(rocke_ir_builder_t* b,
                                             const rocke_type_t* dtype,
                                             rocke_value_t* a,
                                             rocke_value_t* bv,
                                             rocke_value_t* c);

/* ------------------------------------------------- wave64 cross-lane reduction
 *
 * Six-stage XOR butterfly (masks 1,2,4,8,16,32) over all 64 lanes of a wave.
 * After the call every lane holds the wave-wide max / sum. */
rocke_value_t* rocke_wave64_reduce_max(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_wave64_reduce_sum(rocke_ir_builder_t* b, rocke_value_t* v);

/* ----------------------------------------------- binary search on cu_q
 *
 * Triton-style binary search for the seq_idx owning q_block_global_idx, mirroring
 * aiter's find_seq_idx. `block_q` is the BLOCK_Q divisor; `iterations` is the
 * fixed loop trip count specialized from the batch size; `per_token` selects the
 * per-token comparison mode (cu_q[s] <= q_token). Returns the result Value
 * (loop.results[0] - 1), or NULL if the loop op failed to materialize. */
rocke_value_t* rocke_binary_search_seq_idx(rocke_ir_builder_t* b,
                                           rocke_value_t* cu_q,
                                           rocke_value_t* q_block_global_idx,
                                           rocke_value_t* num_seqs,
                                           int block_q,
                                           int iterations,
                                           bool per_token);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_HELPER_ROCKE_HELPERS_ATTENTION_H */
