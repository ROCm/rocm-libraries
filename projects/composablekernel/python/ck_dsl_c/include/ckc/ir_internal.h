/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/ir_internal.h -- PRIVATE shared declarations for the C99 port of
 * ck_dsl.core.ir. NOT a public API: only the ir_*.c translation units of the
 * builder include this. The public contract is ckc/ir.h.
 *
 * Everything here is a cross-bucket helper shared by the parallel body files
 * (ir_core.c, ir_arith.c, ir_mem.c, ir_tile.c, ir_flow.c). The DEFINITIONS of
 * all functions declared here live in bucket 0 (ir_core.c). The other buckets
 * only call them.
 *
 * Naming: internal helpers are prefixed ckc_i_ (i = internal) to keep them out
 * of the public ckc_ / ckc_b_ namespace.
 */
#ifndef CKC_IR_INTERNAL_H
#define CKC_IR_INTERNAL_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- error model */

/* Set the builder's sticky error (first failure wins) and return NULL. Used by
 * the *_t*-returning helpers via ckc_i_fail / by op builders that return a
 * pointer. `fmt` is printf-style; the message is copied into builder->err
 * (truncated to CKC_ERR_MSG_CAP). If the builder is already failed, the
 * existing status/message are preserved. Always returns NULL. */
#if defined(__cplusplus)
[[noreturn]]
#endif
void* ckc_i_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* fmt, ...);

/* Translate a thrown ckc::Error (already caught at a public entry boundary) into
 * the builder's sticky status + err message, then return NULL. This is the
 * boundary shim used by the extern "C" entry points: internal code throws a
 * ckc::Error where the Python reference would `raise`, the entry point catches
 * it and funnels it here so the C ABI (status code + builder->err) is unchanged.
 * `code` and `msg` are taken from the caught exception. Unlike ckc_i_set_err
 * this records the message even if the builder is already in an error state
 * (the throw is the authoritative failure). Always returns NULL. */
void* ckc_i_set_err_msg(ckc_ir_builder_t* b, ckc_status_t code, const char* msg);

/* True if the builder is in the OK state (status == CKC_OK). Inline-able fast
 * path that every builder entry point calls first; a failed builder makes all
 * subsequent calls no-ops returning NULL / the zero handle. */
bool ckc_i_live(const ckc_ir_builder_t* b);

/* ------------------------------------------------------ value / op plumbing */

/* Allocate a fresh SSA Value (arena-owned) named "%<prefix><counter>" with the
 * given type. Bumps builder->counter. Mirrors Python IRBuilder._fresh + Value
 * construction. Returns NULL on OOM (and sets the sticky error). */
ckc_value_t* ckc_i_new_value(ckc_ir_builder_t* b, const char* prefix, const ckc_type_t* type);

/* Allocate a Value with an explicit, already-formed name (with leading '%'),
 * e.g. params "%foo" and loop induction vars "%k0". Does NOT bump the counter.
 */
ckc_value_t* ckc_i_value_named(ckc_ir_builder_t* b, const char* name, const ckc_type_t* type);

/* The single shared implementation behind the public ckc_b_op: build an Op of
 * `opcode`, copy operands/result_types/attrs/regions into arena arrays, create
 * one fresh result Value per result type (named with result_name_hint), link
 * results back to the op, append it to the current region, and return it.
 * `attrs`/`regions` may be NULL. This is IRBuilder._op. Every op-emitting
 * helper in every bucket funnels through here. Returns NULL on failure. */
ckc_op_t* ckc_i_op(ckc_ir_builder_t* b,
                   ckc_opcode_t opcode,
                   ckc_value_t* const* operands,
                   int num_operands,
                   const ckc_type_t* const* result_types,
                   int num_results,
                   const ckc_attr_map_t* attrs,
                   ckc_region_t* const* regions,
                   int num_regions,
                   const char* result_name_hint,
                   const char* loc);

/* Append `op` to the current (top-of-stack) region. Mirrors IRBuilder._emit.
 * No-op on a failed builder. */
void ckc_i_emit(ckc_ir_builder_t* b, ckc_op_t* op);

/* Allocate an empty Region with the given label (arena-owned copy). */
ckc_region_t* ckc_i_new_region(ckc_ir_builder_t* b, const char* label);

/* ------------------------------------------------- common emission shorthands */

/* Build a 1-result op and return its single result Value (the common
 * `self._op(...).result` Python idiom). Thin wrapper over ckc_i_op. Returns
 * NULL on failure. */
ckc_value_t* ckc_i_op1(ckc_ir_builder_t* b,
                       ckc_opcode_t opcode,
                       ckc_value_t* const* operands,
                       int num_operands,
                       const ckc_type_t* result_type,
                       const ckc_attr_map_t* attrs,
                       const char* result_name_hint);

/* Build a 0-result (void / effect-only) op. Mirrors `self._op(...)` with no
 * result_types. Returns the op (mostly for chaining) or NULL on failure. */
ckc_op_t* ckc_i_op0(ckc_ir_builder_t* b,
                    ckc_opcode_t opcode,
                    ckc_value_t* const* operands,
                    int num_operands,
                    const ckc_attr_map_t* attrs);

/* Convenience: build a same-result-type binary op `(a, b) -> a->type`, the
 * dominant arith/vector pattern. Validates a/b non-NULL on a live builder. */
ckc_value_t* ckc_i_binop(ckc_ir_builder_t* b,
                         ckc_opcode_t opcode,
                         ckc_value_t* a,
                         ckc_value_t* bb,
                         const char* result_name_hint);

/* Convenience: build a unary op `(a) -> a->type`. */
ckc_value_t*
ckc_i_unop(ckc_ir_builder_t* b, ckc_opcode_t opcode, ckc_value_t* a, const char* result_name_hint);

/* ----------------------------------------------------- type-system helpers */

/* Is `t` a scalar of the given canonical name ("i32","f16",...)? NULL-safe. */
bool ckc_i_type_is(const ckc_type_t* t, const char* name);

/* Is `t` a VectorType<elem_name x count>? Pass elem_name=NULL to match any
 * element type, count<0 to match any lane count. NULL-safe. */
bool ckc_i_is_vector(const ckc_type_t* t, const char* elem_name, int count);

/* Element type of a vector, or the type itself for a scalar (Python
 * `v.type.elem if isinstance(...,VectorType) else v.type`). */
const ckc_type_t* ckc_i_elem_of(const ckc_type_t* t);

/* Lane count of a vector type, or 1 for a scalar. */
int ckc_i_count_of(const ckc_type_t* t);

/* ------------------------------------------------------------- attr helpers */

/* Build a small attr map IN the arena and return it by value (the map's
 * entries array is arena-owned). Used to assemble the `attrs={...}` literals
 * the Python ops pass to _op. These mutate via ckc_attr_set_* (public). */
ckc_attr_map_t ckc_i_attrs(ckc_ir_builder_t* b);

/* Deep-copy an attr map into the arena (for ckc_b_op which takes a borrowed
 * attrs pointer; Op.attrs is dict(attrs or {}) in Python). */
void ckc_i_attrs_copy(ckc_ir_builder_t* b, ckc_attr_map_t* dst, const ckc_attr_map_t* src);

#ifdef __cplusplus
}
#endif

#endif /* CKC_IR_INTERNAL_H */
