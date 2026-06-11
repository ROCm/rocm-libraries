/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.qk_scale.c -- C99 port of ck_dsl.helpers.qk_scale:
 *   QkScaleLayout, QkScaleSpec, apply_qk_scales,
 *   load_k_scale_for_block, load_q_scale_for_block.
 *
 * Each public builder helper reproduces the Python builder-call sequence in the
 * exact same order so the emitted IR (and SSA value numbering) stays
 * byte-identical. Where Python evaluates the arguments of a binary builder call
 * left-to-right (and each sub-emission consumes one SSA value-counter tick),
 * the C port sequences those sub-emissions explicitly into locals, because C's
 * argument-evaluation order is unspecified.
 */

#include "ckc/helper_ck_dsl.helpers.qk_scale.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

/* -------------------------------------------------------- QkScaleLayout */

const char* ckc_qk_scale_layout_name(ckc_qk_scale_layout_t layout)
{
    switch (layout)
    {
    case CKC_QK_SCALE_PER_HEAD:
        return "per_head";
    case CKC_QK_SCALE_PER_BLOCK:
        return "per_block";
    default:
        return "?";
    }
}

/* ---------------------------------------------------------- QkScaleSpec */

void ckc_qk_scale_spec_init(ckc_qk_scale_spec_t* spec, ckc_qk_scale_layout_t layout)
{
    if (spec == NULL)
    {
        return;
    }
    /* Python field defaults:
     *   scale_block: int = 0
     *   stride_batch: int = 0
     *   stride_head: int = 0
     *   stride_block: int = 1 */
    spec->layout = layout;
    spec->scale_block = 0;
    spec->stride_batch = 0;
    spec->stride_head = 0;
    spec->stride_block = 1;
}

int ckc_qk_scale_spec_validate(const ckc_qk_scale_spec_t* spec, const char** out_reason)
{
    /* Python __post_init__:
     *
     *     if self.layout not in ("per_head", "per_block"):
     *         raise ValueError("QkScaleSpec.layout must be 'per_head' or "
     *                          "'per_block', got {layout!r}")
     *     if self.layout == "per_block" and self.scale_block <= 0:
     *         raise ValueError("per_block scale_block must be > 0 "
     *                          "(got {scale_block})")
     */
    if (out_reason != NULL)
    {
        *out_reason = "";
    }
    if (spec == NULL)
    {
        if (out_reason != NULL)
        {
            *out_reason = "QkScaleSpec is NULL";
        }
        return 0;
    }
    if (spec->layout != CKC_QK_SCALE_PER_HEAD && spec->layout != CKC_QK_SCALE_PER_BLOCK)
    {
        if (out_reason != NULL)
        {
            *out_reason =
                "QkScaleSpec.layout must be 'per_head' or 'per_block'";
        }
        return 0;
    }
    if (spec->layout == CKC_QK_SCALE_PER_BLOCK && spec->scale_block <= 0)
    {
        if (out_reason != NULL)
        {
            *out_reason = "per_block scale_block must be > 0";
        }
        return 0;
    }
    return 1;
}

/* ------------------------------------------------- module-private helpers */

/* C99 port of _scale_ptr_validate:
 *
 *     def _scale_ptr_validate(ptr):
 *         if not isinstance(ptr.type, PtrType):
 *             raise ValueError("Q/K scale pointer must be a typed pointer; "
 *                              "got {ptr.type.name}")
 *         if ptr.type.pointee != F32:
 *             raise ValueError("Q/K scale tensors must be ptr<f32>, "
 *                              "got ptr<{ptr.type.pointee.name}>")
 *
 * Returns 1 if valid; on rejection records the matching ValueError text on the
 * builder (CKC_ERR_VALUE) and returns 0. */
static int ckc_qk__scale_ptr_validate(ckc_ir_builder_t* b, ckc_value_t* ptr)
{
    const ckc_type_t* pty = (ptr != NULL) ? ptr->type : NULL;

    /* if not isinstance(ptr.type, PtrType): raise ValueError(...) */
    if (pty == NULL || pty->kind != CKC_TYPE_PTR)
    {
        const char* tn = (pty != NULL && pty->name != NULL) ? pty->name : "None";
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "Q/K scale pointer must be a typed pointer; got %s", tn);
        return 0;
    }

    /* if ptr.type.pointee != F32: raise ValueError(...) */
    if (pty->pointee == NULL || pty->pointee->name == NULL ||
        strcmp(pty->pointee->name, "f32") != 0)
    {
        const char* pn = (pty->pointee != NULL && pty->pointee->name != NULL)
                             ? pty->pointee->name
                             : "None";
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "Q/K scale tensors must be ptr<f32>, got ptr<%s>", pn);
        return 0;
    }
    return 1;
}

/* C99 port of _scale_offset_for_block:
 *
 *     def _scale_offset_for_block(b, *, spec, batch_idx, head_idx, block_idx):
 *         off = b.add(
 *             b.mul(batch_idx, b.const_i32(spec.stride_batch)),
 *             b.mul(head_idx, b.const_i32(spec.stride_head)),
 *         )
 *         if spec.layout == "per_block":
 *             off = b.add(off, b.mul(block_idx, b.const_i32(spec.stride_block)))
 *         return off
 *
 * Python evaluates each binary call's args left-to-right; every const_i32 / mul
 * sub-emission consumes one SSA tick, so the sequence is pinned explicitly. */
static ckc_value_t* ckc_qk__scale_offset_for_block(ckc_ir_builder_t* b,
                                                   const ckc_qk_scale_spec_t* spec,
                                                   ckc_value_t* batch_idx,
                                                   ckc_value_t* head_idx,
                                                   ckc_value_t* block_idx)
{
    ckc_value_t* off;

    /* off = b.add(b.mul(batch_idx, b.const_i32(stride_batch)),
     *             b.mul(head_idx,  b.const_i32(stride_head)))
     *
     * Inner-left arg of b.add is emitted first: its own arg const_i32 emits,
     * then the mul. Then the inner-right arg: const_i32, then mul. Then b.add. */
    {
        ckc_value_t* mul_batch;
        ckc_value_t* mul_head;
        {
            ckc_value_t* c_sb = ckc_b_const_i32(b, spec->stride_batch);
            mul_batch = ckc_b_mul(b, batch_idx, c_sb);
        }
        {
            ckc_value_t* c_sh = ckc_b_const_i32(b, spec->stride_head);
            mul_head = ckc_b_mul(b, head_idx, c_sh);
        }
        off = ckc_b_add(b, mul_batch, mul_head);
    }

    /* if spec.layout == "per_block":
     *     off = b.add(off, b.mul(block_idx, b.const_i32(stride_block))) */
    if (spec->layout == CKC_QK_SCALE_PER_BLOCK)
    {
        ckc_value_t* mul_block;
        {
            ckc_value_t* c_blk = ckc_b_const_i32(b, spec->stride_block);
            mul_block = ckc_b_mul(b, block_idx, c_blk);
        }
        off = ckc_b_add(b, off, mul_block);
    }
    return off;
}

/* ------------------------------------------------------- public loaders */

ckc_value_t* ckc_b_load_q_scale_for_block(ckc_ir_builder_t* b,
                                          ckc_value_t* q_scale_ptr,
                                          const ckc_qk_scale_spec_t* spec,
                                          ckc_value_t* batch_idx,
                                          ckc_value_t* head_idx,
                                          ckc_value_t* q_block_idx)
{
    ckc_value_t* off;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if (!ckc_i_live(b))
    {
        return NULL;
    }
    if (spec == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE,
                                           "load_q_scale_for_block: spec is NULL");
    }

    /* _scale_ptr_validate(q_scale_ptr) */
    if (!ckc_qk__scale_ptr_validate(b, q_scale_ptr))
    {
        return NULL;
    }

    /* off = _scale_offset_for_block(b, spec=spec, batch_idx=batch_idx,
     *                               head_idx=head_idx, block_idx=q_block_idx) */
    off = ckc_qk__scale_offset_for_block(b, spec, batch_idx, head_idx, q_block_idx);

    /* return b.global_load_f32(q_scale_ptr, off)
     * Python passes no align kwarg -> default (4); the C global_load_f32 maps a
     * non-positive align to its f32 default, so 0 reproduces the default. */
    return ckc_b_global_load_f32(b, q_scale_ptr, off, 0);
}

ckc_value_t* ckc_b_load_k_scale_for_block(ckc_ir_builder_t* b,
                                          ckc_value_t* k_scale_ptr,
                                          const ckc_qk_scale_spec_t* spec,
                                          ckc_value_t* batch_idx,
                                          ckc_value_t* head_idx,
                                          ckc_value_t* k_block_idx)
{
    ckc_value_t* off;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if (!ckc_i_live(b))
    {
        return NULL;
    }
    if (spec == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE,
                                           "load_k_scale_for_block: spec is NULL");
    }

    /* _scale_ptr_validate(k_scale_ptr) */
    if (!ckc_qk__scale_ptr_validate(b, k_scale_ptr))
    {
        return NULL;
    }

    /* off = _scale_offset_for_block(b, spec=spec, batch_idx=batch_idx,
     *                               head_idx=head_idx, block_idx=k_block_idx) */
    off = ckc_qk__scale_offset_for_block(b, spec, batch_idx, head_idx, k_block_idx);

    /* return b.global_load_f32(k_scale_ptr, off) -- default align (see above). */
    return ckc_b_global_load_f32(b, k_scale_ptr, off, 0);
}

/* ------------------------------------------------------ apply_qk_scales */

ckc_value_t* ckc_b_apply_qk_scales(ckc_ir_builder_t* b,
                                   ckc_value_t* score_log2,
                                   ckc_value_t* q_scale,
                                   ckc_value_t* k_scale)
{
    ckc_value_t* qk_scale;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* if score_log2.type.name != "f32": raise ValueError(...) */
    if (score_log2 == NULL || score_log2->type == NULL ||
        score_log2->type->name == NULL ||
        strcmp(score_log2->type->name, "f32") != 0)
    {
        const char* tn = (score_log2 != NULL && score_log2->type != NULL &&
                          score_log2->type->name != NULL)
                             ? score_log2->type->name
                             : "None";
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "apply_qk_scales expects an f32 score, got %s", tn);
    }

    /* qk_scale = b.fmul(q_scale, k_scale) */
    qk_scale = ckc_b_fmul(b, q_scale, k_scale);
    /* return b.fmul(score_log2, qk_scale) */
    return ckc_b_fmul(b, score_log2, qk_scale);
}
