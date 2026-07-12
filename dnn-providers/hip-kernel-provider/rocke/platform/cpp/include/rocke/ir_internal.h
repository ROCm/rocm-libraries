/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/ir_internal.h -- PRIVATE shared declarations for the C99 port of
 * rocke.core.ir. NOT a public API: only the ir_*.c translation units of the
 * builder include this. The public contract is rocke/ir.h.
 *
 * Everything here is a cross-bucket helper shared by the parallel body files
 * (ir_core.c, ir_arith.c, ir_mem.c, ir_tile.c, ir_flow.c). The DEFINITIONS of
 * all functions declared here live in bucket 0 (ir_core.c). The other buckets
 * only call them.
 *
 * Naming: internal helpers are prefixed rocke_i_ (i = internal) to keep them out
 * of the public rocke_ / rocke_b_ namespace.
 */
#ifndef ROCKE_IR_INTERNAL_H
#define ROCKE_IR_INTERNAL_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- error model */

/* Set the builder's sticky error (first failure wins) and return NULL. Used by
 * the *_t*-returning helpers via rocke_i_fail / by op builders that return a
 * pointer. `fmt` is printf-style; the message is copied into builder->err
 * (truncated to ROCKE_ERR_MSG_CAP). If the builder is already failed, the
 * existing status/message are preserved. Always returns NULL. */
#if defined(__cplusplus)
[[noreturn]]
#endif
void* rocke_i_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...);

/* Translate a thrown ckc::Error (already caught at a public entry boundary) into
 * the builder's sticky status + err message, then return NULL. This is the
 * boundary shim used by the extern "C" entry points: internal code throws a
 * ckc::Error where the Python reference would `raise`, the entry point catches
 * it and funnels it here so the C ABI (status code + builder->err) is unchanged.
 * `code` and `msg` are taken from the caught exception. Unlike rocke_i_set_err
 * this records the message even if the builder is already in an error state
 * (the throw is the authoritative failure). Always returns NULL. */
void* rocke_i_set_err_msg(rocke_ir_builder_t* b, rocke_status_t code, const char* msg);

/* True if the builder is in the OK state (status == ROCKE_OK). Inline-able fast
 * path that every builder entry point calls first; a failed builder makes all
 * subsequent calls no-ops returning NULL / the zero handle. */
bool rocke_i_live(const rocke_ir_builder_t* b);

/* ------------------------------------------------------ value / op plumbing */

/* Allocate a fresh SSA Value (arena-owned) named "%<prefix><counter>" with the
 * given type. Bumps builder->counter. Mirrors Python IRBuilder._fresh + Value
 * construction. Returns NULL on OOM (and sets the sticky error). */
rocke_value_t*
    rocke_i_new_value(rocke_ir_builder_t* b, const char* prefix, const rocke_type_t* type);

/* Allocate a Value with an explicit, already-formed name (with leading '%'),
 * e.g. params "%foo" and loop induction vars "%k0". Does NOT bump the counter.
 */
rocke_value_t*
    rocke_i_value_named(rocke_ir_builder_t* b, const char* name, const rocke_type_t* type);

/* The single shared implementation behind the public rocke_b_op: build an Op of
 * `opcode`, copy operands/result_types/attrs/regions into arena arrays, create
 * one fresh result Value per result type (named with result_name_hint), link
 * results back to the op, append it to the current region, and return it.
 * `attrs`/`regions` may be NULL. This is IRBuilder._op. Every op-emitting
 * helper in every bucket funnels through here. Returns NULL on failure. */
rocke_op_t* rocke_i_op(rocke_ir_builder_t* b,
                       rocke_opcode_t opcode,
                       rocke_value_t* const* operands,
                       int num_operands,
                       const rocke_type_t* const* result_types,
                       int num_results,
                       const rocke_attr_map_t* attrs,
                       rocke_region_t* const* regions,
                       int num_regions,
                       const char* result_name_hint,
                       const char* loc);

/* Append `op` to the current (top-of-stack) region. Mirrors IRBuilder._emit.
 * No-op on a failed builder. */
void rocke_i_emit(rocke_ir_builder_t* b, rocke_op_t* op);

/* Allocate an empty Region with the given label (arena-owned copy). */
rocke_region_t* rocke_i_new_region(rocke_ir_builder_t* b, const char* label);

/* ------------------------------------------------- common emission shorthands */

/* Build a 1-result op and return its single result Value (the common
 * `self._op(...).result` Python idiom). Thin wrapper over rocke_i_op. Returns
 * NULL on failure. */
rocke_value_t* rocke_i_op1(rocke_ir_builder_t* b,
                           rocke_opcode_t opcode,
                           rocke_value_t* const* operands,
                           int num_operands,
                           const rocke_type_t* result_type,
                           const rocke_attr_map_t* attrs,
                           const char* result_name_hint);

/* Build a 0-result (void / effect-only) op. Mirrors `self._op(...)` with no
 * result_types. Returns the op (mostly for chaining) or NULL on failure. */
rocke_op_t* rocke_i_op0(rocke_ir_builder_t* b,
                        rocke_opcode_t opcode,
                        rocke_value_t* const* operands,
                        int num_operands,
                        const rocke_attr_map_t* attrs);

/* Convenience: build a same-result-type binary op `(a, b) -> a->type`, the
 * dominant arith/vector pattern. Validates a/b non-NULL on a live builder. */
rocke_value_t* rocke_i_binop(rocke_ir_builder_t* b,
                             rocke_opcode_t opcode,
                             rocke_value_t* a,
                             rocke_value_t* bb,
                             const char* result_name_hint);

/* Convenience: build a unary op `(a) -> a->type`. */
rocke_value_t* rocke_i_unop(rocke_ir_builder_t* b,
                            rocke_opcode_t opcode,
                            rocke_value_t* a,
                            const char* result_name_hint);

/* ----------------------------------------------------- type-system helpers */

/* Is `t` a scalar of the given canonical name ("i32","f16",...)? NULL-safe. */
bool rocke_i_type_is(const rocke_type_t* t, const char* name);

/* Is `t` a VectorType<elem_name x count>? Pass elem_name=NULL to match any
 * element type, count<0 to match any lane count. NULL-safe. */
bool rocke_i_is_vector(const rocke_type_t* t, const char* elem_name, int count);

/* Element type of a vector, or the type itself for a scalar (Python
 * `v.type.elem if isinstance(...,VectorType) else v.type`). */
const rocke_type_t* rocke_i_elem_of(const rocke_type_t* t);

/* Lane count of a vector type, or 1 for a scalar. */
int rocke_i_count_of(const rocke_type_t* t);

/* ------------------------------------------------------------- attr helpers */

/* Build a small attr map IN the arena and return it by value (the map's
 * entries array is arena-owned). Used to assemble the `attrs={...}` literals
 * the Python ops pass to _op. These mutate via rocke_attr_set_* (public). */
rocke_attr_map_t rocke_i_attrs(rocke_ir_builder_t* b);

/* Deep-copy an attr map into the arena (for rocke_b_op which takes a borrowed
 * attrs pointer; Op.attrs is dict(attrs or {}) in Python). */
void rocke_i_attrs_copy(rocke_ir_builder_t* b, rocke_attr_map_t* dst, const rocke_attr_map_t* src);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_IR_INTERNAL_H */
