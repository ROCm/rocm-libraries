// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.gather_scatter.c -- C99 port of
 * rocke.helpers.gather_scatter.load_sorted_token_id and
 * load_sorted_topk_weight.
 *
 * Faithful translation of the two indirect-load helpers used by the fused-MoE
 * gather (gate / up input) and topk-weighted scatter (down-projection reduce).
 * Each is a single global load from a moe-sort output buffer, guarded by a
 * pointer-type / pointee-name check.
 *
 * Python:
 *
 *     def load_sorted_token_id(b, sorted_token_ids, bucket_idx) -> Value:
 *         if not isinstance(sorted_token_ids.type, PtrType):
 *             raise ValueError("load_sorted_token_id expects a pointer to "
 *                              "int32 (the SortedTokenIds buffer from moe_sort)")
 *         if sorted_token_ids.type.pointee.name != "i32":
 *             raise ValueError(f"SortedTokenIds must be ptr<i32>, got "
 *                              f"ptr<{sorted_token_ids.type.pointee.name}>")
 *         return b.global_load_i32(sorted_token_ids, bucket_idx)
 *
 *     def load_sorted_topk_weight(b, sorted_weights, bucket_idx) -> Value:
 *         if not isinstance(sorted_weights.type, PtrType):
 *             raise ValueError("load_sorted_topk_weight expects a pointer to "
 *                              "f32 (the SortedWeights buffer from moe_sort)")
 *         if sorted_weights.type.pointee.name != "f32":
 *             raise ValueError(f"SortedWeights must be ptr<f32>, got "
 *                              f"ptr<{sorted_weights.type.pointee.name}>")
 *         return b.global_load_f32(sorted_weights, bucket_idx)
 */

#include "rocke/helper_rocke.helpers.gather_scatter.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

rocke_value_t* rocke_b_load_sorted_token_id(rocke_ir_builder_t* b,
                                            rocke_value_t* sorted_token_ids,
                                            rocke_value_t* bucket_idx)
{
    const rocke_type_t* pty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* if not isinstance(sorted_token_ids.type, PtrType): raise ValueError(...) */
    pty = sorted_token_ids ? sorted_token_ids->type : NULL;
    if(pty == NULL || pty->kind != ROCKE_TYPE_PTR)
    {
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "load_sorted_token_id expects a pointer to int32 "
                                               "(the SortedTokenIds buffer from moe_sort)");
    }

    /* if sorted_token_ids.type.pointee.name != "i32": raise ValueError(...) */
    if(pty->pointee == NULL || pty->pointee->name == NULL || strcmp(pty->pointee->name, "i32") != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "SortedTokenIds must be ptr<i32>, got ptr<%s>",
            (pty->pointee && pty->pointee->name) ? pty->pointee->name : "None");
    }

    /* return b.global_load_i32(sorted_token_ids, bucket_idx)
     * Python passes no align kwarg -> default; 0 matches global_load_i32. */
    return rocke_b_global_load_i32(b, sorted_token_ids, bucket_idx, 0);
}

rocke_value_t* rocke_b_load_sorted_topk_weight(rocke_ir_builder_t* b,
                                               rocke_value_t* sorted_weights,
                                               rocke_value_t* bucket_idx)
{
    const rocke_type_t* pty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* if not isinstance(sorted_weights.type, PtrType): raise ValueError(...) */
    pty = sorted_weights ? sorted_weights->type : NULL;
    if(pty == NULL || pty->kind != ROCKE_TYPE_PTR)
    {
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "load_sorted_topk_weight expects a pointer to f32 "
                                               "(the SortedWeights buffer from moe_sort)");
    }

    /* if sorted_weights.type.pointee.name != "f32": raise ValueError(...) */
    if(pty->pointee == NULL || pty->pointee->name == NULL || strcmp(pty->pointee->name, "f32") != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "SortedWeights must be ptr<f32>, got ptr<%s>",
            (pty->pointee && pty->pointee->name) ? pty->pointee->name : "None");
    }

    /* return b.global_load_f32(sorted_weights, bucket_idx)
     * Python passes no align kwarg -> default; 0 matches global_load_f32. */
    return rocke_b_global_load_f32(b, sorted_weights, bucket_idx, 0);
}
