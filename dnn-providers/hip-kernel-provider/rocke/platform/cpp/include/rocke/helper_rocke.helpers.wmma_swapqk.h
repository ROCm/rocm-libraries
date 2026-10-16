/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helpers/wmma_swapqk.py: the transposed-QK WMMA FMHA-forward inner
 * body for gfx1151 (CK gfx11 ``qr_ks_vs`` design), FP16/BF16, dense, with
 * runtime tensor strides.
 *
 * Python                              C99 (this header)
 * ----------------------------------  ---------------------------------------
 * wmma_swapqk_fwd_inner_body(...)     rocke_wmma_swapqk_fwd_inner_body(...)
 *
 * SCOPE. The validated D64/D128 configuration supports FP16 and BF16, one
 * query tile per wave, mask_mode NONE/CAUSAL, d-outer QK, lazy online-softmax
 * rescale, raw exp2, and buffer-descriptor D16 V gathers against original
 * dense storage. block_n (32 or 64) and n_waves (1 or 2) are the only
 * tunables.
 *
 * PARAMS. Reuses the existing rocke_mfma_attn_params_t (see
 * rocke/helper_rocke.helpers.mfma_attention.h) so the caller does not need a
 * second params type; only the fields the Python keyword signature lists are
 * read (Q, K, V, O, head_size, dtype, seqlen_k, q_tile_base, q_pos_base,
 * head_idx, kv_head_idx, the eight stride fields, scale_log2, batch offsets,
 * mask mode/context, and optional LSE output fields).
 * The caller has already declared all
 * kernel params and decoded the (q_group, head, batch) grid ids / GQA head
 * mapping (mirroring the Python docstring): this function only emits the
 * QK/softmax/PV/epilogue body (including the O stores). It does not declare
 * params, decode block_id_{x,y,z}, or emit ret().
 *
 * An optional causal_ctx_offset shifts the diagonal, normally by Sk-Sq for
 * bottom-right alignment. NULL preserves legacy top-left emission. Offset
 * masking uses true -inf and a safe empty-row softmax shift; mask_neg_inf can
 * reuse the caller's -inf Value instead of emitting another constant.
 *
 * FIDELITY. Every rocke_b_* call sequence reproduces the Python builder-call
 * order, operands and compile-time constants op-for-op so the emitted IR is
 * byte-identical to the Python helper's emission.
 *
 * BINDINGS.
 *   - IR builder primitives: rocke/ir.h's rocke_b_* entry points.
 *   - WMMA atom / lane layout: rocke/arch_target.h's dtype-matched
 *     wmma_f32_16x16x16_{f16,bf16} op and rocke_layout_map_coord.
 *   - Tensor views / tile windows: rocke/helper_rocke.helpers.tensor_view.h.
 *   - Attention mask: rocke/helper_rocke.helpers.attention.h's
 *     rocke_apply_attention_mask.
 *
 * ERROR MODEL. Mirrors the rest of the C port: the sticky-error builder
 * (rocke_b_*) stands in for `raise`. Each Python `raise ValueError` maps onto
 * a rocke_i_set_err(ROCKE_ERR_VALUE) sticky error + early return.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_WMMA_SWAPQK_H
#define ROCKE_HELPER_ROCKE_HELPERS_WMMA_SWAPQK_H

#include "rocke/helper_rocke.helpers.mfma_attention.h" /* rocke_mfma_attn_params_t */
#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------ wmma_swapqk_fwd_inner_body *
 *
 * Python:
 *
 *     def wmma_swapqk_fwd_inner_body(b, *, Q, K, V, O, head_size, seqlen_k,
 *                                    q_tile_base, q_pos_base, head_idx,
 *                                    kv_head_idx, stride_q_token, ...,
 *                                    scale_log2, k_token_offset_elems,
 *                                    v_token_offset_elems, mask_mode, block_n,
 *                                    n_waves, arch="gfx1151") -> None
 *
 * Emits one transposed-QK WMMA FMHA-forward wave body (S^T = K @ Q^T,
 * register P-transpose, O^T = V @ P) into the caller's already-open kernel
 * region. `block_n` and `n_waves` are the only algorithm tunables besides the
 * params; every other lever from the wider exploration (V pre-transpose,
 * LDS-staged K/V, software pipelining, f16 O-carry, persistent launch, ...) is
 * out of scope and rejected.
 *
 * Validates: arch must be "gfx1151" (p->arch, NULL => "gfx1151"); head_size
 * must be 64 or 128; block_n must be 32 or 64; n_waves must
 * be 1 or 2; mask_mode (p->mask_mode) must be NONE or CAUSAL. Returns
 * ROCKE_ERR_VALUE (builder sticky error set) on any violation, or if the
 * gfx1151 wmma_f32_16x16x16_f16 atom / verified lane layout is unavailable.
 * `p` must be non-NULL.
 *
 * The position and batch-element-offset fields listed above are required;
 * callers supply zero Values rather than NULL for absent batch offsets. */
rocke_status_t rocke_wmma_swapqk_fwd_inner_body(rocke_ir_builder_t* b,
                                                const rocke_mfma_attn_params_t* p,
                                                int block_n,
                                                int n_waves);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_WMMA_SWAPQK_H */
