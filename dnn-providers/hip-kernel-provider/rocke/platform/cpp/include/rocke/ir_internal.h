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

/* Throw ckc::ValueError when an author-chosen value name is not an identifier,
 * is a C++ keyword, or ends in '_' and digits (the form builder-made names
 * take). `what` names the kind of name ("kernel parameter", ...). The twin of
 * core/ir.py::_refuse_declared_name, with the same messages. */
void rocke_i_refuse_declared_name(rocke_ir_builder_t* b, const char* what, const char* name);

/* Allocate a fresh SSA Value (arena-owned) named "%<prefix>_<counter>" with the
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

/* Element count an LDS vector access of width `vec` can actually claim as its
 * alignment granularity. The natural answer is `vec`, but an LDS tile advances
 * by its innermost dimension per row, so when that dimension is not a multiple
 * of `vec` -- what a bank-conflict pad such as lds_k_pad does -- alternate rows
 * start off the natural boundary. Returns gcd(vec, innermost_dim), or `vec`
 * when the shape is unavailable. Multiply by the element size for bytes.
 * Mirrors lower_llvm._smem_vec_align / ir.py smem_store_vN. */
int rocke_i_smem_vec_align_elems(const rocke_type_t* smem_type, int vec);
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

/* The error for a built tile.gpu_check_failed op that has operands, results or
 * regions, or NULL when it is bare (Python
 * core/ir.py::_gpu_check_failed_op_error, same text). A returned error is
 * written to `buf`. The verifier and both lowerings use it. */
const char* rocke_i_gpu_check_failed_op_error(const rocke_op_t* op, char* buf, size_t cap);

/* The code-object symbol of `kernel_name`'s GPU-check word, allocated in
 * `arena` (Python core/ir.py::gpu_check_word_symbol, same spelling). A wave
 * whose gpu_checked_value fact is false stores 1 to it and returns; the host reads
 * it after the launch. */
const char* rocke_i_gpu_check_word_symbol(rocke_arena_t* arena, const char* kernel_name);

/* Whether `kernel` holds a tile.gpu_check_failed anywhere in its body, nested
 * regions included: a kernel with a gpu_checked_value, whose false fact only a
 * host reading the GPU-check word sees. */
bool rocke_i_kernel_checks_on_gpu(const rocke_kernel_def_t* kernel);

/* The `by` of the range marker rocke_b_gpu_checked_value writes (Python
 * core/ir.py::GPU_CHECKED_VALUE_WRITER). */
#define ROCKE_GPU_CHECKED_VALUE_WRITER "gpu_checked_value"

/* Why such a kernel must not be launched from its code object alone (Python
 * core/ir.py::GPU_CHECK_UNREAD_REASON, same words). */
#define ROCKE_GPU_CHECK_UNREAD_REASON                                      \
    "only rocke's runtime reads this kernel's GPU-check word, so a false " \
    "gpu_checked_value fact would go unseen"

/* Why a kernel whose parameters state facts must not be launched from a launch
 * plan (Python core/ir.py::PARAM_FACTS_UNCHECKED_REASON, same words). */
#define ROCKE_PARAM_FACTS_UNCHECKED_REASON                                        \
    "parameters carry facts, which a launch from a code object or from a launch " \
    "plan does not check"

/* What a refused launch plan tells its caller to do instead (Python
 * core/ir.py::LAUNCH_THROUGH_FROM_ARTIFACT, same words). */
#define ROCKE_LAUNCH_THROUGH_FROM_ARTIFACT "launch it through KernelLauncher.from_artifact"

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

/* Mirrors ir._reused_loop_name_message: writes into `buf` the refusal for an
 * scf.for induction variable or carried value named `name` (as the op records
 * it, "%<name>") when that name is taken by another value of the same loop or
 * of a loop it is nested in, and returns `buf`. The serialized IR refers to a
 * value by its name, so a reused name is ambiguous inside the inner body. The
 * builder, the IR parser and the verifier use it. */
const char* rocke_i_reused_loop_name_message(const char* name, char* buf, size_t cap);

/* Mirrors ir._first_reused_loop_name: the first of `names` (the names an
 * scf.for gives its induction variable and carried values, in that order) that
 * is already taken, or NULL when all are free. `taken` are the names of the
 * loops it is nested in; a name is also taken when it repeats an earlier one of
 * `names`. The builder, the IR parser and the verifier all apply this rule and
 * refuse with rocke_i_reused_loop_name_message. */
const char* rocke_i_first_reused_loop_name(const char* const* taken,
                                           size_t n_taken,
                                           const char* const* names,
                                           size_t n_names);

/* Mirrors ir._scf_for_value_names: the names the scf.for `loop` gives its
 * induction variable and carried values, in that order -- the "iv" attr when it
 * is a string, then the "name" of each "iter_args" entry that has one -- stored
 * into `out` when it is non-NULL; returns how many there are. A name is listed
 * whether or not its type is recorded. The builder, the IR parser and the
 * verifier collect a loop's names with it. */
int rocke_i_scf_for_value_names(const rocke_op_t* loop, const char** out);

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

/* Mirrors param_facts.declaration_errors: hands `sink` every rule the declaration
 * params[index] breaks, in order, with the Python texts, and returns their
 * count. `params` is the kernel's parameter list in declaration order (a
 * relation may name only parameters before `index`). rocke_b_param raises the
 * first message; rocke_verify reports all of them. (cpp/core/ir/param_facts.cpp) */
int rocke_param_fact_errors(const rocke_param_t* const* params,
                            int count,
                            int index,
                            rocke_param_fact_sink_t sink,
                            void* ctx);

/* The range of an integer type a fact may be stated on (Python
 * param_facts.int_type_limits): for i8, i16, i32, i64 and u8
 * (param_facts.FACT_INT_TYPES) writes the inclusive [lo, hi], derived from the
 * type's dtype width and signedness, and returns true; returns false for any
 * other type. */
bool rocke_int_type_limits(const rocke_type_t* t, int64_t* lo, int64_t* hi);

/* Whether `p` states an integer fact: one of param_facts.FACT_WORDS (ge, gt, le, lt, eq,
 * multiple_of) is among its attrs. Pointer attributes are not facts. */
bool rocke_param_states_facts(const rocke_param_t* p);

/* Whether `key` is one of param_facts.FACT_WORDS. */
bool rocke_param_is_fact_word(const char* key);

/* One param attr value as a portable-IR document holds it. Each importer turns
 * its own document node into this, so both read a param's attrs through
 * rocke_param_opts_set_attr. `str` is borrowed from the document. */
typedef enum rocke_param_attr_kind
{
    ROCKE_PARAM_ATTR_BOOL,
    ROCKE_PARAM_ATTR_NUMBER,
    ROCKE_PARAM_ATTR_STRING,
    ROCKE_PARAM_ATTR_OTHER
} rocke_param_attr_kind_t;

typedef struct rocke_param_attr_value
{
    rocke_param_attr_kind_t kind;
    bool b;
    double number;
    const char* str;
} rocke_param_attr_value_t;

/* Store attr `key` = `value` of param `param` in *opts: a key rocke_param_opts_t
 * carries (ROCKE_PARAM_ATTR_KEYS) with a value of its kind. Returns NULL, or the
 * refusal, written to `buf`, for an unknown key or a value of the wrong kind; a
 * dropped attr would silently change the lowered kernel. A fact bound is an
 * integer or a relation's canonical intexpr JSON (a string, kept borrowed; the
 * builder checks it). (cpp/core/ir/param_facts.cpp) */
const char* rocke_param_opts_set_attr(rocke_param_opts_t* opts,
                                      const char* param,
                                      const char* key,
                                      const rocke_param_attr_value_t* value,
                                      char* buf,
                                      size_t cap);

/* The refusal for a param whose attrs are not an object; the argument is the
 * param's name. */
#define ROCKE_PARAM_ATTRS_NOT_OBJECT "param '%s' attrs is not an object"

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

/* One end of an arith.assume_range marker (Python IRBuilder._assume_range's ge /
 * lt and ge_plus / lt_plus): an int when `set`, or the Value `end` plus `plus`
 * for an end only known at run time. */
typedef struct rocke_range_end
{
    bool set;
    int64_t value;
    rocke_value_t* end;
    int64_t plus;
} rocke_range_end_t;

/* The facts an arith.assume_range marker records (Python _assume_range's
 * keywords): ge + ge_plus <= v < lt + lt_plus, v == residue (mod multiple_of;
 * 1 when unset), and `by`, the writer that checked them (NULL for none). */
typedef struct rocke_assume_range_opts
{
    rocke_range_end_t ge;
    rocke_range_end_t lt;
    int64_t multiple_of;
    bool multiple_of_set;
    int64_t residue;
    const char* by;
} rocke_assume_range_opts_t;

/* Python IRBuilder._assume_range: return `v` marked with facts its caller has
 * checked or proved (an arith.assume_range marker, which emits nothing).
 * Internal: a fact stated here is never checked, so only a caller that checked
 * it first may state it (rocke_b_gpu_checked_value). Refuses bad facts with the
 * verifier's texts. (cpp/core/ir/range_marker.cpp) */
rocke_value_t* rocke_b_assume_range(rocke_ir_builder_t* b,
                                    rocke_value_t* v,
                                    const rocke_assume_range_opts_t* opts);

/* Mirrors ir._assume_range_errors: hands `sink` every rule an arith.assume_range
 * marker's attrs and Value ends break, in order, with the Python texts, and
 * returns their count. `operands` are the marked value and its Value ends
 * (`num_operands` of them, at least one). The rules include that each Value
 * end has the marked value's type.
 * (cpp/core/ir/range_marker.cpp) */
int rocke_assume_range_errors(const rocke_attr_map_t* attrs,
                              rocke_value_t* const* operands,
                              int num_operands,
                              rocke_param_fact_sink_t sink,
                              void* ctx);

/* Mirrors ir.assume_range_bound_operands: the marker's ge and lt operands, NULL
 * for an end it does not state as a Value. */
void rocke_assume_range_bound_operands(const rocke_op_t* op,
                                       rocke_value_t** ge,
                                       rocke_value_t** lt);

/* Whether an op with `opcode` reads memory, so its result is a value no check
 * has seen: the ops of Python value_sources.MEMORY_READ_OPS (global, buffer,
 * LDS and cluster loads, LDS transposed reads, atomics). Listed by opcode, not
 * inferred from purity. (cpp/core/ir/gpu_check.cpp) */
bool rocke_i_opcode_reads_memory(rocke_opcode_t opcode);

/* Mirrors ir.one_unsigned_compare_checks: whether 0 <= v < end is exactly
 * v u< end for an i32 or i64 `t` and an end known to lie in [least, most].
 * Throws ckc::ValueError, with the Python text, for any other `t`.
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

#include <string>
#include <utility>
#include <vector>

/* `t`'s name in upper case, as a declaration text writes a type (I32): the
 * Python texts' `t.name.upper()`. (cpp/core/ir/param_facts.cpp) */
std::string rocke_i_upper_type_name(const rocke_type_t* t);

/* The end of a value a bound word states: Python param_facts.LOWER_WORDS or
 * UPPER_WORDS, or both for eq. */
typedef enum rocke_int_bound_side
{
    ROCKE_INT_BOUND_LOWER,
    ROCKE_INT_BOUND_UPPER,
    ROCKE_INT_BOUND_BOTH
} rocke_int_bound_side_t;

/* A fact word that takes a bound, with the rocke_int_facts_t field it fills and
 * the end it states. */
struct rocke_int_bound_word_t
{
    const char* word;
    rocke_int_bound_t rocke_int_facts_t::*bound;
    rocke_int_bound_side_t side;
};

/* The fact words that take a bound, in Python param_facts._BOUND_WORDS order:
 * the one table from a word to its field and its end, which the param builder,
 * the param-fact rules, the GPU-check builder and the param-attr reader all read. */
inline constexpr rocke_int_bound_word_t ROCKE_INT_BOUND_WORDS[]
    = {{"ge", &rocke_int_facts_t::ge, ROCKE_INT_BOUND_LOWER},
       {"gt", &rocke_int_facts_t::gt, ROCKE_INT_BOUND_LOWER},
       {"le", &rocke_int_facts_t::le, ROCKE_INT_BOUND_UPPER},
       {"lt", &rocke_int_facts_t::lt, ROCKE_INT_BOUND_UPPER},
       {"eq", &rocke_int_facts_t::eq, ROCKE_INT_BOUND_BOTH}};

/* "states two lower bounds (ge, gt) -- state one" for each end whose two words
 * `stated(word)` holds for, the lower end first: a fact states each end once.
 * The caller prefixes its subject. Python param_facts.two_bounds_refusals. */
template <typename Stated>
std::vector<std::string> rocke_int_two_bounds_refusals(Stated stated)
{
    std::vector<std::string> refusals;
    for(const auto& [side, end] :
        {std::pair{ROCKE_INT_BOUND_LOWER, "lower"}, std::pair{ROCKE_INT_BOUND_UPPER, "upper"}})
    {
        std::string words;
        bool all_stated = true;
        for(const auto& bound_word : ROCKE_INT_BOUND_WORDS)
        {
            if(bound_word.side != side)
                continue;
            words += (words.empty() ? "" : ", ") + std::string(bound_word.word);
            all_stated = all_stated && stated(bound_word.word);
        }
        if(all_stated)
            refusals.push_back(std::string("states two ") + end + " bounds (" + words
                               + ") -- state one");
    }
    return refusals;
}
#endif

#endif /* ROCKE_IR_INTERNAL_H */
