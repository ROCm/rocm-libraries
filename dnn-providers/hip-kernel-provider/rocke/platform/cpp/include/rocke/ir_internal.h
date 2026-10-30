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
#include "rocke/strbuf.h"

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

/* Throw ckc::TypeError unless the source of `op`, a u8 -> float conversion, is
 * a u8 scalar. rocke_b_uitofp_f32 checks its input when the op is built; the
 * lowerings call this for IR that did not come through it (hand-assembled or
 * deserialized), because they would read any other source differently: LLVM's
 * uitofp reads an i8 0xff as 255.0, the HIP cast of an int8_t as -1.0. The twin
 * of core/ir.py::_refuse_non_u8_source, with the same message. */
void rocke_i_refuse_non_u8_source(const rocke_op_t* op);

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

/* The bytes one buffer access of `count` `elem` elements moves (Python
 * core/ir.py::_buffer_access_bytes): `elem` is a scalar whose width is whole
 * bytes and the access is 1, 2, 4, 8 or 16 bytes. Otherwise returns 0 and
 * writes the Python engine's refusal, prefixed with `op_name`, into `err`. */
int rocke_i_buffer_access_bytes(
    const char* op_name, const rocke_type_t* elem, int count, char* err, size_t err_cap);

/* The bytes a buffer access whose `role` ("result" of a load, "value" of a
 * store) has type `t` moves (Python core/ir.py::_buffer_vector_bytes): `t` is a
 * vector, checked as rocke_i_buffer_access_bytes. Otherwise returns 0 and
 * writes the Python engine's refusal into `err`. */
int rocke_i_buffer_vector_bytes(
    const char* op_name, const char* role, const rocke_type_t* t, char* err, size_t err_cap);

/* rocke_i_buffer_vector_bytes for a built tile.buffer_load_vN (three operands,
 * one result: the loaded vector) or tile.buffer_store_vN (four operands, the
 * last the stored vector, no result), after its operand and result counts:
 * serialized IR reaches the verifier and the lowerings without passing the
 * builder (Python core/ir.py::_buffer_op_bytes). Otherwise returns 0 and writes
 * the Python engine's refusal into `err`. */
int rocke_i_buffer_op_bytes(const rocke_op_t* op, char* err, size_t err_cap);

/* The error for a global_ptr_offset whose pointer, offset and (once built,
 * else NULL) result have these types, or NULL when they are valid (Python
 * core/ir.py::_global_ptr_offset_error, same texts): a global pointer, an i64
 * offset, a result of the pointer's type. A returned error is written to `buf`. */
const char* rocke_i_global_ptr_offset_error(const rocke_type_t* ptr_type,
                                            const rocke_type_t* byte_off_type,
                                            const rocke_type_t* result_type,
                                            char* buf,
                                            size_t cap);

/* rocke_i_global_ptr_offset_error for a built tile.global_ptr_offset op, after
 * its operand and result counts (Python core/ir.py::_global_ptr_offset_op_error).
 * The verifier and both lowerings use it. */
const char* rocke_i_global_ptr_offset_op_error(const rocke_op_t* op, char* buf, size_t cap);

/* The error for a global_ptr_to_i64 whose pointer and (once built, else NULL)
 * result have these types, or NULL when they are valid (Python
 * core/ir.py::_global_ptr_to_i64_error, same texts): a global pointer and an
 * i64 result. A returned error is written to `buf`. */
const char* rocke_i_global_ptr_to_i64_error(const rocke_type_t* ptr_type,
                                            const rocke_type_t* result_type,
                                            char* buf,
                                            size_t cap);

/* rocke_i_global_ptr_to_i64_error for a built tile.global_ptr_to_i64 op, after
 * its operand and result counts (Python core/ir.py::_global_ptr_to_i64_op_error).
 * The verifier and both lowerings use it. */
const char* rocke_i_global_ptr_to_i64_op_error(const rocke_op_t* op, char* buf, size_t cap);

/* The error for a built tile.trap op that has operands, results or regions,
 * or NULL when it is bare (Python core/ir.py::_trap_op_error, same text): the
 * lowerings emit only the trap call, so a result would reach its uses
 * undefined. A returned error is written to `buf`. The verifier and both
 * lowerings use it. */
const char* rocke_i_trap_op_error(const rocke_op_t* op, char* buf, size_t cap);

/* Whether an scf.if_else can carry a result `index` of type `result_type`
 * (Python core/ir.py::_if_else_result_type_error): not an LDS allocation.
 * Otherwise returns false and writes the Python engine's refusal into `err`.
 * The builder and rocke_i_if_else_branch_yield use it. */
bool rocke_i_if_else_result_type_ok(int index,
                                    const rocke_type_t* result_type,
                                    char* err,
                                    size_t err_cap);

/* The scf.yield that gives the scf.if_else `op`'s results their values on
 * branch `branch` (0 then, 1 else), in `*out_yield`; NULL when `op` has no
 * results (Python core/ir.py::_if_else_branch_yield). Every if/else has one
 * condition and two branches, checked first; no result is an LDS allocation;
 * a branch of an if/else with results ends in an scf.yield of one value per
 * result, of the result's type, and never a buffer descriptor. Otherwise
 * returns false and writes the Python engine's refusal into `err`. */
bool rocke_i_if_else_branch_yield(
    const rocke_op_t* op, int branch, const rocke_op_t** out_yield, char* err, size_t err_cap);

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

/* ------------------------------------------------- loop-carried descriptors */

/* Mirrors ir._is_buffer_descriptor: true when `value` is the result of a
 * tile.buffer_rsrc op. */
bool rocke_i_is_buffer_descriptor(const rocke_value_t* value);

/* Mirrors ir._loop_carried_descriptor_message: writes into `buf` the refusal
 * for a buffer descriptor as scf.for iter_arg `index` (its initial value or the
 * value an iteration yields to it) and returns `buf`. A loop cannot carry a
 * descriptor: the phi or HIP variable holding the carried value cannot hold the
 * lowered descriptor (an opaque ptr addrspace(8) / rsrc_t). */
const char* rocke_i_loop_carried_descriptor_message(int index, char* buf, size_t cap);

/* Mirrors ir._scf_for_carried_descriptor_error: the refusal for the scf.for
 * `op` when an iter_arg init (operands[3:]) or a value an scf.yield directly in
 * its body yields is a buffer descriptor, written into `buf`; NULL otherwise.
 * Accepts any operand / region counts. The builder, the verifier and both
 * lowerings use it. */
const char* rocke_i_scf_for_carried_descriptor_error(const rocke_op_t* op, char* buf, size_t cap);

/* ------------------------------------------------------------- attr helpers */

/* Build a small attr map IN the arena and return it by value (the map's
 * entries array is arena-owned). Used to assemble the `attrs={...}` literals
 * the Python ops pass to _op. These mutate via rocke_attr_set_* (public). */
rocke_attr_map_t rocke_i_attrs(rocke_ir_builder_t* b);

/* Deep-copy an attr map into the arena (for rocke_b_op which takes a borrowed
 * attrs pointer; Op.attrs is dict(attrs or {}) in Python). */
void rocke_i_attrs_copy(rocke_ir_builder_t* b, rocke_attr_map_t* dst, const rocke_attr_map_t* src);

/* Mirrors ir._int_add_wrap_error: the error for arith.add's "wrap" attr on
 * operands of type `operand_type`, or NULL when it is absent (meaning nsw) or
 * valid; "nuw" takes i32 or i64 operands. A formatted message is written into
 * `buf`, which the result then points to. */
const char* rocke_i_add_wrap_error(const rocke_attr_map_t* attrs,
                                   const rocke_type_t* operand_type,
                                   char* buf,
                                   size_t cap);

/* Mirrors ir._int_cmp_pred_error: the same for arith.cmp's "pred" attr
 * (absent means lt; "ult" takes i32 or i64 operands). */
const char* rocke_i_cmp_pred_error(const rocke_attr_map_t* attrs,
                                   const rocke_type_t* operand_type,
                                   char* buf,
                                   size_t cap);

/* ------------------------------------------------------- parameter facts */

/* Receives one message from rocke_param_fact_errors. */
typedef void (*rocke_param_fact_sink_t)(void* ctx, const char* message);

/* Mirrors ir.param_fact_errors: hands `sink` every rule the declaration
 * params[index] breaks, in order, with the Python texts, and returns their
 * count. `params` is the kernel's parameter list in declaration order (a
 * relation may name only parameters before `index`). rocke_b_param raises the
 * first message; rocke_verify reports all of them. (cpp/core/ir/param_facts.cpp) */
int rocke_param_fact_errors(const rocke_param_t* const* params,
                            int count,
                            int index,
                            rocke_param_fact_sink_t sink,
                            void* ctx);

/* Mirrors ir_print._format_param's facts: appends " {ge = %N, le = 4096}" for
 * the facts `p` states, nothing when it states none. */
void rocke_param_facts_print(const rocke_param_t* p, rocke_strbuf_t* out);

/* The canonical JSON (sorted keys, no spaces) of the relation stored in
 * `text`, with each param leaf renamed through `rename` (NULL keeps names),
 * allocated in `arena`; NULL when `text` holds no relation. */
char* rocke_relation_canonical(const char* text,
                               rocke_arena_t* arena,
                               const char* (*rename)(void* ctx, const char* name),
                               void* ctx);

/* Mirrors ir._declaration_text: `p`'s declaration as the author writes it,
 * b.param("M", I32, ge=1, le=1048576), appended to `out`. */
void rocke_param_declaration_text(const rocke_param_t* p, rocke_strbuf_t* out);

/* Mirrors intexpr.relation_text with plain leaf names: the relation stored in
 * `text` as the author writes it (product_of(M, ldc)), or `text` itself when it
 * holds no relation, appended to `out`. */
void rocke_relation_display(const char* text, rocke_strbuf_t* out);

/* ------------------------------------------------------- range marker */

/* Mirrors ir._assume_range_errors: hands `sink` every rule an arith.assume_range
 * marker's attrs break, in order, with the Python texts, and returns their
 * count. `num_operands` counts the marked value and its Value ends.
 * (cpp/core/ir/range_marker.cpp) */
int rocke_assume_range_errors(const rocke_attr_map_t* attrs,
                              int num_operands,
                              rocke_param_fact_sink_t sink,
                              void* ctx);

/* Mirrors ir.assume_range_bound_operands: the marker's ge and lt operands, NULL
 * for an end it does not state as a Value. */
void rocke_assume_range_bound_operands(const rocke_op_t* op,
                                       rocke_value_t** ge,
                                       rocke_value_t** lt);

/* Mirrors ir.one_unsigned_compare_checks: whether 0 <= v < end is exactly
 * v u< end for an i32 or i64 `t` and an end known to lie in [least, most].
 * (cpp/core/ir/gpu_check.cpp) */
bool rocke_i_one_unsigned_compare_checks(int64_t least, int64_t most, const rocke_type_t* t);

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
