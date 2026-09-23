/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.qk_scale.h -- C99 port of rocke.helpers.qk_scale.
 *
 * Per-block Q-scale / K-scale loader for Sage attention (CK Tile
 * 49_sageattention). Q and K are quantised into per-block (or per-head) f32
 * scales + a low-precision mantissa buffer; the inner attention loop applies
 * those scales to the score before softmax. This helper provides the canonical
 * "load + apply" chain for two scale-buffer layouts:
 *
 *   * per-head  -- one f32 scale per (batch, head) pair. One load per CTA.
 *   * per-block -- one f32 scale per (batch, head, q_block) (or
 *                  (batch, head, k_block)), where q_block / k_block slices the
 *                  sequence dim into blocks of size scale_block.
 *
 * Both layouts share a QkScaleSpec descriptor capturing
 * (layout, scale_block, stride_*) so the kernel author picks either at
 * kernel-build time.
 *
 * Scope of THIS file (all five requested symbols):
 *   - QkScaleLayout            -> rocke_qk_scale_layout_t
 *   - QkScaleSpec              -> rocke_qk_scale_spec_t (+ init/validate)
 *   - apply_qk_scales          -> rocke_b_apply_qk_scales
 *   - load_k_scale_for_block   -> rocke_b_load_k_scale_for_block
 *   - load_q_scale_for_block   -> rocke_b_load_q_scale_for_block
 *
 * The module-private helpers _scale_ptr_validate and _scale_offset_for_block
 * are ported as static internals of the .c (they are pure ValueError guards /
 * b.add/b.mul chains that the public functions inline).
 *
 * The Python raises ValueError in __post_init__, _scale_ptr_validate, and
 * apply_qk_scales. C99 has no exceptions, so:
 *
 *   * QkScaleSpec validation is exposed as rocke_qk_scale_spec_validate (returns
 *     1/0 + optional *out_reason), mirroring the sibling validate_io port. The
 *     load helpers also re-check the spec/pointer on the builder's sticky-error
 *     model: an unsupported pointer or score type records ROCKE_ERR_VALUE + the
 *     Python-matching message on the builder and returns NULL.
 *   * An already-errored builder makes every rocke_b_* call a NULL no-op.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_QK_SCALE_H
#define ROCKE_HELPER_ROCKE_HELPERS_QK_SCALE_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C99 port of QkScaleLayout = Literal["per_head", "per_block"].
 *
 * The Python literal is a string; the C port uses an enum discriminant. The
 * canonical string spellings ("per_head"/"per_block") are exposed via
 * rocke_qk_scale_layout_name for any consumer that needs the exact Python text
 * (e.g. error messages). */
typedef enum rocke_qk_scale_layout
{
    ROCKE_QK_SCALE_PER_HEAD = 0, /* "per_head"  */
    ROCKE_QK_SCALE_PER_BLOCK = 1 /* "per_block" */
} rocke_qk_scale_layout_t;

/* Return the canonical Python string for a layout ("per_head"/"per_block"),
 * or "?" for an out-of-range value. Static storage; never NULL. */
const char* rocke_qk_scale_layout_name(rocke_qk_scale_layout_t layout);

/* C99 port of the frozen dataclass QkScaleSpec:
 *
 *     @dataclass(frozen=True)
 *     class QkScaleSpec:
 *         layout: QkScaleLayout
 *         scale_block: int = 0
 *         stride_batch: int = 0
 *         stride_head: int = 0
 *         stride_block: int = 1
 *
 * Per-Sage-tensor scale-buffer layout descriptor. `scale_block` is the block
 * size in tokens (the sequence-dim slice each scale covers); ignored for
 * per_head, required (>0) for per_block. The stride_* are element strides into
 * the scale tensor; stride_block defaults to 1 for the common contiguous case.
 */
typedef struct rocke_qk_scale_spec
{
    rocke_qk_scale_layout_t layout;
    int scale_block; /* default 0  */
    int stride_batch; /* default 0  */
    int stride_head; /* default 0  */
    int stride_block; /* default 1  */
} rocke_qk_scale_spec_t;

/* Initialise `spec` with the dataclass field defaults and the required
 * `layout`. After this call scale_block/stride_batch/stride_head are 0 and
 * stride_block is 1, matching the Python defaults. Note this does NOT run the
 * __post_init__ validation; call rocke_qk_scale_spec_validate for that. */
void rocke_qk_scale_spec_init(rocke_qk_scale_spec_t* spec, rocke_qk_scale_layout_t layout);

/* C99 port of QkScaleSpec.__post_init__:
 *
 *     if self.layout not in ("per_head", "per_block"):
 *         raise ValueError("QkScaleSpec.layout must be 'per_head' or "
 *                          "'per_block', got {layout!r}")
 *     if self.layout == "per_block" and self.scale_block <= 0:
 *         raise ValueError("per_block scale_block must be > 0 "
 *                          "(got {scale_block})")
 *
 * Returns 1 if the spec is valid, 0 otherwise. *out_reason (if non-NULL)
 * receives a static / Python-matching message on rejection, or "" on success.
 * Takes no builder: pure validation, no IR side effects. */
int rocke_qk_scale_spec_validate(const rocke_qk_scale_spec_t* spec, const char** out_reason);

/* C99 port of load_q_scale_for_block:
 *
 *     def load_q_scale_for_block(b, q_scale_ptr, *, spec, batch_idx,
 *                                head_idx, q_block_idx) -> Value:
 *         _scale_ptr_validate(q_scale_ptr)
 *         off = _scale_offset_for_block(b, spec=spec, batch_idx=batch_idx,
 *                                       head_idx=head_idx, block_idx=q_block_idx)
 *         return b.global_load_f32(q_scale_ptr, off)
 *
 * Loads the f32 Q-scale for one (batch, head, q_block) tuple. For per_head the
 * q_block_idx is ignored (it does not enter the offset). q_scale_ptr must be a
 * ptr<f32>; a non-typed-ptr / non-ptr<f32> operand records ROCKE_ERR_VALUE + the
 * Python-matching message and returns NULL. Returns the loaded f32 Value. */
rocke_value_t* rocke_b_load_q_scale_for_block(rocke_ir_builder_t* b,
                                              rocke_value_t* q_scale_ptr,
                                              const rocke_qk_scale_spec_t* spec,
                                              rocke_value_t* batch_idx,
                                              rocke_value_t* head_idx,
                                              rocke_value_t* q_block_idx);

/* C99 port of load_k_scale_for_block:
 *
 *     def load_k_scale_for_block(b, k_scale_ptr, *, spec, batch_idx,
 *                                head_idx, k_block_idx) -> Value:
 *         _scale_ptr_validate(k_scale_ptr)
 *         off = _scale_offset_for_block(b, spec=spec, batch_idx=batch_idx,
 *                                       head_idx=head_idx, block_idx=k_block_idx)
 *         return b.global_load_f32(k_scale_ptr, off)
 *
 * Symmetric to rocke_b_load_q_scale_for_block; used per K-iteration in the Sage
 * inner loop. Same validation / sticky-error / return contract. */
rocke_value_t* rocke_b_load_k_scale_for_block(rocke_ir_builder_t* b,
                                              rocke_value_t* k_scale_ptr,
                                              const rocke_qk_scale_spec_t* spec,
                                              rocke_value_t* batch_idx,
                                              rocke_value_t* head_idx,
                                              rocke_value_t* k_block_idx);

/* C99 port of apply_qk_scales:
 *
 *     def apply_qk_scales(b, score_log2, *, q_scale, k_scale) -> Value:
 *         if score_log2.type.name != "f32":
 *             raise ValueError("apply_qk_scales expects an f32 score, got "
 *                              "{score_log2.type.name}")
 *         qk_scale = b.fmul(q_scale, k_scale)
 *         return b.fmul(score_log2, qk_scale)
 *
 * Applies q_scale * k_scale to a log2-domain score:
 * out = score_log2 * (q_scale * k_scale). A non-f32 score records
 * ROCKE_ERR_VALUE + the Python-matching message and returns NULL; otherwise
 * returns the scaled f32 Value. */
rocke_value_t* rocke_b_apply_qk_scales(rocke_ir_builder_t* b,
                                       rocke_value_t* score_log2,
                                       rocke_value_t* q_scale,
                                       rocke_value_t* k_scale);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_QK_SCALE_H */
