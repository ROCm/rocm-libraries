/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.gather_scatter.h -- C99 port of
 * rocke.helpers.gather_scatter.
 *
 * Scope of THIS file: the two indirect-load address helpers
 *   - load_sorted_token_id
 *   - load_sorted_topk_weight
 * are ported here. The address-arithmetic helpers gather_row_offset /
 * scatter_token_offset from the same Python module are out of scope for this
 * phase (they are pure b.add/b.mul chains and can be added later alongside).
 *
 * Both helpers are one global load from a moe-sort output buffer, guarded by a
 * pointer-type / pointee-name check that mirrors the Python ValueError raises.
 * The C port follows the rest of the helper ports' sticky-error model: an
 * unsupported pointer type records ROCKE_ERR_VALUE + the Python-matching message
 * on the builder and returns NULL; an already-errored builder is a NULL no-op.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_GATHER_SCATTER_H
#define ROCKE_HELPER_ROCKE_HELPERS_GATHER_SCATTER_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C99 port of rocke.helpers.gather_scatter.load_sorted_token_id:
 *
 *     def load_sorted_token_id(b, sorted_token_ids, bucket_idx) -> Value:
 *         if not isinstance(sorted_token_ids.type, PtrType):
 *             raise ValueError(
 *                 "load_sorted_token_id expects a pointer to int32 "
 *                 "(the SortedTokenIds buffer from moe_sort)")
 *         if sorted_token_ids.type.pointee.name != "i32":
 *             raise ValueError(
 *                 f"SortedTokenIds must be ptr<i32>, got "
 *                 f"ptr<{sorted_token_ids.type.pointee.name}>")
 *         return b.global_load_i32(sorted_token_ids, bucket_idx)
 *
 * One i32 global load from the moe-sort SortedTokenIds buffer. The Python
 * passes no align kwarg -> align defaults; this port passes 0 to match
 * global_load_i32's default. Returns NULL on a non-ptr / non-ptr<i32> operand
 * (recording the matching ValueError text) or on an already-errored builder. */
rocke_value_t* rocke_b_load_sorted_token_id(rocke_ir_builder_t* b,
                                            rocke_value_t* sorted_token_ids,
                                            rocke_value_t* bucket_idx);

/* C99 port of rocke.helpers.gather_scatter.load_sorted_topk_weight:
 *
 *     def load_sorted_topk_weight(b, sorted_weights, bucket_idx) -> Value:
 *         if not isinstance(sorted_weights.type, PtrType):
 *             raise ValueError(
 *                 "load_sorted_topk_weight expects a pointer to f32 "
 *                 "(the SortedWeights buffer from moe_sort)")
 *         if sorted_weights.type.pointee.name != "f32":
 *             raise ValueError(
 *                 f"SortedWeights must be ptr<f32>, got "
 *                 f"ptr<{sorted_weights.type.pointee.name}>")
 *         return b.global_load_f32(sorted_weights, bucket_idx)
 *
 * One f32 global load of the per-bucket topk weight from the moe-sort
 * SortedWeights buffer. The Python passes no align kwarg -> align defaults;
 * this port passes 0 to match global_load_f32's default. Returns NULL on a
 * non-ptr / non-ptr<f32> operand (recording the matching ValueError text) or
 * on an already-errored builder. */
rocke_value_t* rocke_b_load_sorted_topk_weight(rocke_ir_builder_t* b,
                                               rocke_value_t* sorted_weights,
                                               rocke_value_t* bucket_idx);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_GATHER_SCATTER_H */
