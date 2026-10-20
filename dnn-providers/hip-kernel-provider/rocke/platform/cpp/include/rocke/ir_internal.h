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

/* Whether `t`'s elements are packed-only (i4, u4): `t` is such a scalar, a vector
 * of one or an LDS tile of one. The twin of core/ir.py::is_packed_only. NULL is
 * not. */
bool rocke_i_is_packed_only(const rocke_type_t* t);

/* Throw ckc::TypeError when rocke_i_is_packed_only(t): the twin of core/ir.py::_refuse_packed_only_value,
 * which states the rule and the message. rocke_i_op calls it for every result
 * and rocke_b_param for every parameter, so no builder makes such a value.
 * NULL-safe. */
void rocke_i_refuse_packed_only_value(const char* op_name, const rocke_type_t* t);

/* Throw ckc::TypeError when `t` is unsigned (rocke_type_is_unsigned) and
 * `op_name` treats its operands as signed; `instead` names the unsigned op to
 * use, or is NULL when there is none. The twin of core/ir.py::_refuse_unsigned,
 * with the same message. NULL-safe. */
void rocke_i_refuse_unsigned(const char* op_name, const rocke_type_t* t, const char* instead);

/* rocke_i_refuse_unsigned for every operand of `op`, a signed op, under the
 * op's dotted name. The builders refuse such an op when it is built; the
 * lowerings call this for IR that did not come through them (hand-assembled
 * or deserialized), so no signed op lowers on unsigned data. The twin of
 * core/ir.py::_refuse_unsigned_operands. */
void rocke_i_refuse_unsigned_operands(const rocke_op_t* op, const char* instead);

/* Lane count of a vector type, or 1 for a scalar. */
int rocke_i_count_of(const rocke_type_t* t);

/* VectorType(elem, count) allocated in `arena`, for callers that hold an arena
 * but no builder (the lowerers). `elem` must be non-NULL; NULL on OOM. The
 * name is the one rocke_vector_type gives the same type. */
const rocke_type_t*
    rocke_i_vector_type_in_arena(rocke_arena_t* arena, const rocke_type_t* elem, int count);

/* The logical IR type of `dtype` (any accepted spelling), as
 * rocke_dtype_to_ir_type returns it (Python core/ir.py::dtype_to_ir_type).
 * Throws ckc::ValueError "unknown dtype '<dtype>'" for an unknown name and
 * "no logical IR type for dtype '<canonical name>'" for an encoding with no
 * IR type. */
const rocke_type_t* rocke_require_ir_type(const char* dtype);

/* The number of bits a value of type `t` occupies (Python core/ir.py::type_bits):
 * a scalar's format width, a vector's element width times its count, a
 * pointer's address-space width. Throws ckc::TypeError for an LDS allocation
 * (memory, not a value) and ckc::ValueError "unknown dtype '<name>'" for a
 * scalar with no format. */
int rocke_i_type_bits(const rocke_type_t* t);

/* Refuse a bitcast of a `source` value to `target` unless both are values of
 * the same bit width (Python core/ir.py::_check_bitcast_width). `verb` names
 * the builder ("bitcast", "vec_bitcast") and `hint` says what to do instead of
 * changing the width. Throws ckc::TypeError when either type is a pointer
 * (LLVM does not bitcast pointers) or an LDS allocation, and ckc::ValueError
 * naming both types and widths when the widths differ. Builders call it before
 * creating the op, so a refused cast leaves the kernel unchanged. */
void rocke_i_check_bitcast_width(const char* verb,
                                 const rocke_type_t* source,
                                 const rocke_type_t* target,
                                 const char* hint);

/* ------------------------------------------------------------- attr helpers */

/* Build a small attr map IN the arena and return it by value (the map's
 * entries array is arena-owned). Used to assemble the `attrs={...}` literals
 * the Python ops pass to _op. These mutate via rocke_attr_set_* (public). */
rocke_attr_map_t rocke_i_attrs(rocke_ir_builder_t* b);

/* Deep-copy an attr map into the arena (for rocke_b_op which takes a borrowed
 * attrs pointer; Op.attrs is dict(attrs or {}) in Python). */
void rocke_i_attrs_copy(rocke_ir_builder_t* b, rocke_attr_map_t* dst, const rocke_attr_map_t* src);

/* ------------------------------------------------------------- DPP helpers */

/* DPP control word that lowers tile.dpp_xor with `xor_mask` (Python
 * rocke.core.ir.dpp_xor_ctrl). has_row_xmask (rocke_arch_has_dpp_row_xmask):
 * 0x160 | mask for any mask 1..15. Otherwise (gfx9 CDNA) 1/2/3 -> quad_perm
 * 0xB1/0x4E/0x1B, 7 -> row_half_mirror 0x141, 8 -> row_ror:8 0x128,
 * 15 -> row_mirror 0x140. Returns -1 for a mask outside 1..15 or with no
 * encoding on the target; the lowerers turn that into ROCKE_ERR_VALUE. */
int64_t rocke_i_dpp_xor_ctrl(int64_t xor_mask, bool has_row_xmask);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_IR_INTERNAL_H */
