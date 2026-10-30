// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Matrix intrinsic calls described as data. NOT a public API.
 *
 * Mirrors core/isa/matrix_calls.py and core/isa/matrix_operand_forms.py. One
 * rocke_matrix_call_t says everything the lowering of one MMA instruction needs:
 * the intrinsic, the order of its arguments (the A, B and C operands and literal
 * immediates), the form each operand and the result take, and the
 * declaration-table key. The declaration is derived from the same forms the
 * call is rendered from, so a call and its `declare` line cannot disagree at any
 * LLVM flavor.
 *
 * The forms (Python AsIs / ZeroPadded / EvenHalves):
 *   AS_IS        the value has the register type, or the same bits in another
 *                shape (one bitcast; rocke_mma_operand_needs_reinterpret)
 *   ZERO_PADDED  a packed <n x i32> payload widened to <m x i32>, zero high words
 *   EVEN_HALVES  n logical 16-bit values in the even half-slots of a 2n register
 *
 * The emitter is rocke_ll_emit_matrix_call (cpp/core/lower_llvm/matrix_call.cpp).
 */
#ifndef ROCKE_MATRIX_CALLS_INTERNAL_H
#define ROCKE_MATRIX_CALLS_INTERNAL_H

#include "rocke/arena.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

namespace ckc
{

/* One register type: the IR scalar named `elem` ("i32", "f16", ...), or a
 * vector of `count` of them. count == 0 means the scalar itself. */
typedef struct rocke_matrix_type
{
    const char* elem;
    int count;
} rocke_matrix_type_t;

/* The type a register changed to at `flavor` (one Python FlavorType override). */
typedef struct rocke_matrix_flavor_override
{
    const char* flavor;
    rocke_matrix_type_t type;
} rocke_matrix_flavor_override_t;

/* A register type, possibly per flavor (Python Type | FlavorType): `base` at the
 * oldest flavor; a flavor uses the newest override at or below it. */
typedef struct rocke_matrix_register
{
    rocke_matrix_type_t base;
    const rocke_matrix_flavor_override_t* overrides;
    int num_overrides;
} rocke_matrix_register_t;

typedef enum rocke_matrix_form_kind
{
    ROCKE_MATRIX_FORM_AS_IS = 0,
    ROCKE_MATRIX_FORM_ZERO_PADDED,
    ROCKE_MATRIX_FORM_EVEN_HALVES
} rocke_matrix_form_kind_t;

/* How one operand, or the result, meets its register type. */
typedef struct rocke_matrix_form
{
    rocke_matrix_form_kind_t kind;
    rocke_matrix_register_t register_type;
} rocke_matrix_form_t;

/* One call argument: an MMA operand (A, B, C) or a literal immediate. */
typedef enum rocke_matrix_arg_kind
{
    ROCKE_MATRIX_ARG_A = 0,
    ROCKE_MATRIX_ARG_B,
    ROCKE_MATRIX_ARG_C,
    ROCKE_MATRIX_ARG_IMM
} rocke_matrix_arg_kind_t;

typedef struct rocke_matrix_arg
{
    rocke_matrix_arg_kind_t kind;
    const char* imm_type; /* "i1" / "i16" / "i32" / "i64" (ROCKE_MATRIX_ARG_IMM)  */
    long long imm_value;
    bool immarg; /* the declaration marks the parameter `immarg`      */
} rocke_matrix_arg_t;

/* One MMA instruction's call. `decl_key` NULL means the intrinsic name. */
typedef struct rocke_matrix_call
{
    const char* intrinsic;
    const rocke_matrix_arg_t* args;
    int num_args;
    rocke_matrix_form_t a;
    rocke_matrix_form_t b;
    rocke_matrix_form_t c;
    rocke_matrix_form_t result;
    const char* decl_key;
} rocke_matrix_call_t;

/* One table row: an op_id and its call (Python MATRIX_CALLS item). */
typedef struct rocke_matrix_call_entry
{
    const char* op_id;
    rocke_matrix_call_t call;
} rocke_matrix_call_entry_t;

/* The table, in Python MATRIX_CALLS order. */
const rocke_matrix_call_entry_t* rocke_matrix_calls(int* out_count);

/* The call for `op_id`, or NULL when a hand-written handler lowers it. */
const rocke_matrix_call_t* rocke_matrix_call_for(const char* op_id);

/* The declaration-table key a call needs (Python MatrixCall.key). */
const char* rocke_matrix_call_key(const rocke_matrix_call_t* call);

/* Throw ckc::ValueError / ckc::TypeError, with the Python text, for a call the
 * Python schema would refuse to construct: A, B and C not passed exactly once,
 * a ZERO_PADDED result, a register type its form cannot fill, an immediate of
 * another type or out of range, an override naming an unknown flavor. */
void rocke_matrix_call_check(const rocke_matrix_call_t* call);

/* The register type of `form` at `flavor` (Python _Form.register_type_at),
 * built in `arena`. */
const rocke_type_t* rocke_matrix_form_type_at(rocke_arena_t* arena,
                                              const rocke_matrix_form_t* form,
                                              rocke_llvm_flavor_t flavor);

/* The `declare` line of `call` at `flavor` (Python MatrixCall.declaration).
 * Interned: the text lives as long as the process. */
const char* rocke_matrix_call_declaration(const rocke_matrix_call_t* call,
                                          rocke_llvm_flavor_t flavor);

typedef struct rocke_lower rocke_lower_t;

/* Lower MMA op `op` (operands A, B, C) with op_id `op_id` through `call`
 * (Python emit_matrix_call): each operand through its form, the immediates as
 * written, the call, the result through its form. The call's declaration is
 * the one the lowerer resolves for its key; a call outside the table (one a
 * test builds) registers its derived declaration dynamically, and a resolved
 * text that differs from the derived one is refused. */
void rocke_ll_emit_matrix_call(rocke_lower_t* L,
                               const rocke_op_t* op,
                               const char* op_id,
                               const rocke_matrix_call_t* call);

} /* namespace ckc */

#endif /* ROCKE_MATRIX_CALLS_INTERNAL_H */
