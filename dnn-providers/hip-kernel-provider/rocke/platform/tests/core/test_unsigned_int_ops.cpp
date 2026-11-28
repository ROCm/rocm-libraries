// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_unsigned_int_ops.cpp -- the C++ engine's unsigned compare (cmp_ult), its
 * no-unsigned-wrap add (add_nuw), its logical shift right (lshr) and the
 * arith.assume_range range marker.
 *
 * Run with no argument it checks, through the C API: the LLVM and HIP text of
 * add_nuw and cmp_ult, the HIP text of lshr at every integer width (a shift
 * through the unsigned type of the operand's own width), the builders', the
 * verifier's and the lowerings' integer-element rule for the bitwise ops,
 * arith.not and the shifts (an integer or i1 element) and operand rule for the
 * same-type binary ops (two operands of one type; texts identical to the
 * Python engine's), the verifier's, LLVM's and HIP's refusals of a compare the
 * IR rule refuses (a pred, operands of two types, or an operand type the
 * compare does not take), the builders' operand-type and compare refusals,
 * the verifier's acceptance of the new attrs and its refusals of bad marker
 * attrs (texts identical to the Python verifier), and the constant folder
 * (unsigned compare at the operand width; constants read at their width; a
 * marker on a constant folds to the constant; signed div / mod rounded toward
 * zero, undefined ones and ops of two operand types left unfolded), and the
 * verifier's i32 / i64 rule for the nuw and ult forms.
 *
 * Run with --print-llvm it prints the LLVM text of the two-op function built
 * through rocke_b_add_nuw / rocke_b_cmp_ult, so the Python test can compare it
 * with the same function built by the Python builder.
 *
 * Any other argument list prints the usage to stderr and exits 2.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/passes.h"
#include "rocke/strbuf.h"
#include "rocke/verify.h"

namespace
{

int g_failures = 0;

void fail(const std::string& what, int line)
{
    fprintf(stderr, "FAIL (line %d): %s\n", line, what.c_str());
    ++g_failures;
}

#define EXPECT(cond, what)          \
    do                              \
    {                               \
        if(!(cond))                 \
            fail((what), __LINE__); \
    } while(0)

/* The two-op function the Python test builds too: params a, b of `type`, then
 * add_nuw(a, b) and cmp_ult(a, b). */
void build_two_op(rocke_ir_builder_t* b, const rocke_type_t* type)
{
    rocke_value_t* a = rocke_b_param(b, "a", type, nullptr);
    rocke_value_t* c = rocke_b_param(b, "b", type, nullptr);
    rocke_b_add_nuw(b, a, c);
    rocke_b_cmp_ult(b, a, c);
    rocke_b_ret(b);
}

/* Lower b's kernel to LLVM IR (llvm20, gfx942); the text on success, else ""
 * with the status returned and the lowerer's message in *error. */
rocke_status_t lower_llvm_status(rocke_ir_builder_t* b, std::string* llvm, std::string* error)
{
    char* text = nullptr;
    char message[ROCKE_ERR_MSG_CAP];
    message[0] = '\0';
    const rocke_status_t status = rocke_lower_kernel_to_llvm_ex(rocke_ir_builder_kernel(b),
                                                                ROCKE_LLVM_FLAVOR_LLVM20,
                                                                "gfx942",
                                                                &text,
                                                                message,
                                                                sizeof message);
    llvm->assign(status == ROCKE_OK && text ? text : "");
    error->assign(message);
    free(text);
    return status;
}

std::string lower_llvm(rocke_ir_builder_t* b)
{
    std::string llvm;
    std::string error;
    lower_llvm_status(b, &llvm, &error);
    return llvm;
}

/* Lower b's kernel to HIP for gfx942; the text on success, else "" with the
 * status (and the builder's error message) left for the caller. */
rocke_status_t lower_hip_status(rocke_ir_builder_t* b, std::string* hip)
{
    rocke_strbuf_t out;
    rocke_strbuf_init(&out, 0);
    rocke_lower_hip_opts_t opts{};
    opts.arch = "gfx942";
    const rocke_status_t status
        = rocke_lower_kernel_to_hip(b, rocke_ir_builder_kernel(b), &opts, &out);
    hip->assign(status == ROCKE_OK ? rocke_strbuf_cstr(&out) : "");
    rocke_strbuf_free(&out);
    return status;
}

std::string lower_hip(rocke_ir_builder_t* b)
{
    std::string hip;
    lower_hip_status(b, &hip);
    return hip;
}

/* A range marker op built directly (the C API has no builder entry for it):
 * operands [v] or [v, lt]; attrs set by `set_attrs`. */
template <typename SetAttrs>
rocke_value_t*
    marker(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* lt, SetAttrs set_attrs)
{
    rocke_value_t* operands[2] = {v, lt};
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    set_attrs(&attrs);
    return rocke_i_op1(
        b, ROCKE_OP_ARITH_ASSUME_RANGE, operands, lt ? 2 : 1, v->type, &attrs, "range");
}

std::string verify_errors(rocke_ir_builder_t* b)
{
    rocke_diag_t* diags = nullptr;
    size_t count = 0;
    std::string joined;
    if(rocke_verify(rocke_ir_builder_kernel(b), &diags, &count) != ROCKE_OK)
        return "verifier failed to run";
    for(size_t i = 0; i < count; ++i)
        if(diags[i].severity == ROCKE_DIAG_ERROR)
            joined += std::string(diags[i].message) + "\n";
    rocke_diags_free(diags, count);
    return joined;
}

void case_llvm_text()
{
    for(const rocke_type_t* type : {rocke_i32(), rocke_i64()})
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops");
        build_two_op(&b, type);
        const std::string ll = lower_llvm(&b);
        const std::string ty = type->name;
        EXPECT(ll.find("= add nuw " + ty + " %a, %b") != std::string::npos, "add nuw line");
        EXPECT(ll.find("= icmp ult " + ty + " %a, %b") != std::string::npos, "icmp ult line");
        rocke_ir_builder_free(&b);
    }
}

void case_hip_text()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_hip");
    rocke_value_t* a = rocke_b_param(&b, "a", rocke_i32(), nullptr);
    rocke_value_t* c = rocke_b_param(&b, "b", rocke_i32(), nullptr);
    rocke_value_t* sum = rocke_b_add_nuw(&b, a, c);
    rocke_value_t* below = rocke_b_cmp_ult(&b, a, c);
    rocke_value_t* range = marker(&b, a, nullptr, [&](rocke_attr_map_t* m) {
        rocke_attr_set_int(&b, m, "multiple_of", 1);
        rocke_attr_set_int(&b, m, "residue", 0);
    });
    const std::string hip = lower_hip(&b);
    const std::string s = std::string(sum->name + 1);
    const std::string u = std::string(below->name + 1);
    const std::string r = std::string(range->name + 1);
    EXPECT(hip.find("int " + s + " = (int32_t)((uint32_t)a + (uint32_t)b);") != std::string::npos,
           "HIP add_nuw form");
    EXPECT(hip.find("bool " + u + " = (uint32_t)a < (uint32_t)b;") != std::string::npos,
           "HIP cmp_ult form");
    EXPECT(hip.find("int " + r + " = a;") != std::string::npos, "HIP marker copy");
    rocke_ir_builder_free(&b);
}

/* One `opcode` of a `value_type` parameter and, unless `amount_type` is NULL
 * (a unary op), an `amount_type` parameter, of `value_type`, built directly:
 * the builders refuse a non-integer operand of a bitwise op or shift, and
 * operands of different types, and hand-assembled or deserialized IR can still
 * reach the verifier and the lowerings. */
void build_op_past_the_builder(rocke_ir_builder_t* b,
                               rocke_opcode_t opcode,
                               const rocke_type_t* value_type,
                               const rocke_type_t* amount_type)
{
    rocke_value_t* operands[2] = {rocke_b_param(b, "value", value_type, nullptr), nullptr};
    if(amount_type)
        operands[1] = rocke_b_param(b, "amount", amount_type, nullptr);
    rocke_i_op1(b, opcode, operands, amount_type ? 2 : 1, value_type, nullptr, "op");
    rocke_b_ret(b);
}

/* lshr reads its operand through the unsigned type of the operand's own width,
 * as LLVM's lshr does: a negative i8 or i16 shifts in no sign bits and an i64
 * keeps its high word. The same text as the Python engine (test_lshr.py). */
void case_lshr_hip_text()
{
    struct LshrCase
    {
        const rocke_type_t* type;
        const char* hip_type;
        const char* unsigned_type;
    };
    const LshrCase cases[] = {
        {rocke_i8(), "int8_t", "uint8_t"},
        {rocke_i16(), "int16_t", "uint16_t"},
        {rocke_i32(), "int", "uint32_t"},
        {rocke_i64(), "int64_t", "uint64_t"},
    };
    for(const LshrCase& c : cases)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_lshr");
        rocke_value_t* a = rocke_b_param(&b, "a", c.type, nullptr);
        rocke_value_t* s = rocke_b_param(&b, "s", c.type, nullptr);
        rocke_value_t* shifted = rocke_b_lshr(&b, a, s);
        rocke_b_ret(&b);
        const std::string hip = lower_hip(&b);
        const std::string name = std::string(shifted->name + 1);
        const std::string want = std::string(c.hip_type) + " " + name + " = (" + c.hip_type + ")(("
                                 + c.unsigned_type + ")a >> s);";
        EXPECT(hip.find(want) != std::string::npos,
               std::string("HIP lshr form for ") + c.type->name + ": want '" + want + "'");
        rocke_ir_builder_free(&b);
    }
}

/* The builder must throw a TypeError with exactly `want` from `build`. */
template <typename Build>
void expect_builder_type_error(Build build, const std::string& want)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_integer_op_refusal");
    bool refused = false;
    std::string got = "no refusal";
    try
    {
        build(&b);
    }
    catch(const ckc::Error& error)
    {
        got = error.what();
        refused = error.code() == ROCKE_ERR_TYPE && want == got;
    }
    EXPECT(refused, "want TypeError '" + want + "', got '" + got + "'");
    rocke_ir_builder_free(&b);
}

/* A two-operand and a one-operand builder entry point of the C API. */
using binary_builder_t = rocke_value_t* (*)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*);
using unary_builder_t = rocke_value_t* (*)(rocke_ir_builder_t*, rocke_value_t*);

/* Every op whose element type must be an integer or i1, with the builder entry
 * point that makes it (binary, or unary for arith.not); mirrors
 * test_lshr.py's _INTEGER_ELEMENT_BUILDERS. */
struct IntegerElementOp
{
    rocke_opcode_t opcode;
    binary_builder_t build_binary;
    unary_builder_t build_unary;
    bool vector;
};

const IntegerElementOp k_integer_element_ops[] = {
    {ROCKE_OP_ARITH_AND, rocke_b_land, nullptr, false},
    {ROCKE_OP_ARITH_OR, rocke_b_lor, nullptr, false},
    {ROCKE_OP_ARITH_XOR, rocke_b_xor, nullptr, false},
    {ROCKE_OP_ARITH_NOT, nullptr, rocke_b_lnot, false},
    {ROCKE_OP_ARITH_SHL, rocke_b_shl, nullptr, false},
    {ROCKE_OP_ARITH_LSHR, rocke_b_lshr, nullptr, false},
    {ROCKE_OP_VECTOR_AND, rocke_b_vector_and, nullptr, true},
    {ROCKE_OP_VECTOR_OR, rocke_b_vector_or, nullptr, true},
    {ROCKE_OP_VECTOR_SHL, rocke_b_vector_shl, nullptr, true},
    {ROCKE_OP_VECTOR_LSHR, rocke_b_vector_lshr, nullptr, true},
};

/* `op` built by its builder entry point on `value` (as both operands of a
 * binary op). */
rocke_value_t* build_integer_element_op(rocke_ir_builder_t* b,
                                        const IntegerElementOp& op,
                                        rocke_value_t* value)
{
    return op.build_binary ? op.build_binary(b, value, value) : op.build_unary(b, value);
}

/* `op` of `element` (two lanes of it for a vector op), and of a second operand
 * of that type for a binary op, built past the builders. */
void build_integer_element_op_past_the_builder(rocke_ir_builder_t* b,
                                               const IntegerElementOp& op,
                                               const rocke_type_t* element)
{
    const rocke_type_t* type = op.vector ? rocke_vector_type(b, element, 2) : element;
    build_op_past_the_builder(b, op.opcode, type, op.build_binary ? type : nullptr);
}

std::string integer_element_message(rocke_opcode_t opcode)
{
    return std::string(rocke_opcode_name(opcode))
           + " failed - element type='f32', expected an integer or i1 type";
}

/* LLVM's bitwise ops and shifts take integers only (arith.not lowers to an
 * xor): the builders refuse an f32 operand and accept an i1 one, scalar or
 * lane by lane. */
void case_integer_element_builder_rule()
{
    for(const IntegerElementOp& op : k_integer_element_ops)
    {
        expect_builder_type_error(
            [&](rocke_ir_builder_t* b) {
                const rocke_type_t* type
                    = op.vector ? rocke_vector_type(b, rocke_f32(), 2) : rocke_f32();
                build_integer_element_op(b, op, rocke_b_param(b, "a", type, nullptr));
            },
            integer_element_message(op.opcode));
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_integer_op_i1");
        const rocke_type_t* flag_type
            = op.vector ? rocke_vector_type(&b, rocke_i1(), 2) : rocke_i1();
        rocke_value_t* flag = rocke_b_param(&b, "flag", flag_type, nullptr);
        EXPECT(build_integer_element_op(&b, op, flag) != nullptr,
               std::string("an i1 ") + rocke_opcode_name(op.opcode) + " is built");
        rocke_b_ret(&b);
        const std::string errors = verify_errors(&b);
        EXPECT(errors.empty(),
               std::string("verifier rejected an i1 ") + rocke_opcode_name(op.opcode) + ": "
                   + errors);
        rocke_ir_builder_free(&b);
    }
}

/* IR written past the builders still meets the rule: the verifier reports an
 * f32 bitwise op or shift, scalar or lane by lane, with the Python verifier's
 * text. */
void case_integer_element_verifier_rule()
{
    for(const IntegerElementOp& op : k_integer_element_ops)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_integer_op_verify");
        build_integer_element_op_past_the_builder(&b, op, rocke_f32());
        const std::string want = integer_element_message(op.opcode) + "\n";
        const std::string errors = verify_errors(&b);
        EXPECT(errors == want, "want '" + want + "', got '" + errors + "'");
        rocke_ir_builder_free(&b);
    }
}

/* The status and message of b's kernel lowered to LLVM and to HIP, as
 * "LLVM: status N 'text'" and "HIP: status N 'text'". */
std::string lowering_refusals(rocke_ir_builder_t* b)
{
    std::string llvm;
    std::string llvm_error;
    const rocke_status_t llvm_status = lower_llvm_status(b, &llvm, &llvm_error);
    std::string hip;
    const rocke_status_t hip_status = lower_hip_status(b, &hip);
    const char* hip_error = rocke_ir_builder_error(b);
    return "LLVM: status " + std::to_string((int)llvm_status) + " '" + llvm_error
           + "'; HIP: status " + std::to_string((int)hip_status) + " '"
           + (hip_error ? hip_error : "") + "'";
}

/* lowering_refusals' text for both lowerings refusing with `status` and
 * `message`. */
std::string both_lowerings_refuse(rocke_status_t status, const std::string& message)
{
    const std::string refusal = "status " + std::to_string((int)status) + " '" + message + "'";
    return "LLVM: " + refusal + "; HIP: " + refusal;
}

/* IR written past the builders reaches the lowerings without the verifier:
 * the LLVM and HIP lowerings refuse an f32 bitwise op or shift, scalar or lane
 * by lane, with ROCKE_ERR_TYPE and the builders' text (the same texts as
 * test_lshr.py). */
void case_integer_element_lowering_rule()
{
    for(const IntegerElementOp& op : k_integer_element_ops)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_integer_op_lower");
        build_integer_element_op_past_the_builder(&b, op, rocke_f32());
        const std::string want
            = both_lowerings_refuse(ROCKE_ERR_TYPE, integer_element_message(op.opcode));
        const std::string got = lowering_refusals(&b);
        EXPECT(got == want, "want " + want + ", got " + got);
        rocke_ir_builder_free(&b);
    }
}

/* Every same-type binary op (rocke_i_is_same_type_binary_op), with the builder
 * entry point that makes it and two operand types that differ; mirrors
 * test_lshr.py's _MIXED_TYPE_OPERANDS. */
struct MixedOperandsCase
{
    rocke_opcode_t opcode;
    binary_builder_t build;
    const rocke_type_t* lhs_element;
    int lhs_lanes; /* 0 for a scalar */
    const rocke_type_t* rhs_element;
    int rhs_lanes;
};

const rocke_type_t* lanes_of(rocke_ir_builder_t* b, const rocke_type_t* element, int lanes)
{
    return lanes ? rocke_vector_type(b, element, lanes) : element;
}

/* The operand rule's text for `opcode` of `lhs_type` and `rhs_type`. */
std::string operand_types_message(rocke_opcode_t opcode,
                                  const rocke_type_t* lhs_type,
                                  const rocke_type_t* rhs_type)
{
    return std::string(rocke_opcode_name(opcode)) + " failed - second operand type='"
           + rhs_type->name + "', expected the first operand's type '" + lhs_type->name + "'";
}

/* The builders, the verifier and the LLVM and HIP lowerings refuse a binary op
 * whose operands differ in type, scalar or lane by lane, with ROCKE_ERR_TYPE
 * and one text: the lowerings emit both operands at the first one's type. */
void case_mixed_operand_types()
{
    const MixedOperandsCase cases[] = {
        {ROCKE_OP_ARITH_ADD, rocke_b_add, rocke_i32(), 0, rocke_i64(), 0},
        {ROCKE_OP_ARITH_SUB, rocke_b_sub, rocke_i64(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_MUL, rocke_b_mul, rocke_i64(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_DIV, rocke_b_div, rocke_i32(), 0, rocke_i16(), 0},
        {ROCKE_OP_ARITH_MOD, rocke_b_mod, rocke_i32(), 0, rocke_i64(), 0},
        {ROCKE_OP_ARITH_FADD, rocke_b_fadd, rocke_f32(), 0, rocke_f16(), 0},
        {ROCKE_OP_ARITH_FSUB, rocke_b_fsub, rocke_f16(), 0, rocke_f32(), 0},
        {ROCKE_OP_ARITH_FMUL, rocke_b_fmul, rocke_f32(), 0, rocke_f16(), 0},
        {ROCKE_OP_ARITH_FDIV, rocke_b_fdiv, rocke_f32(), 0, rocke_f16(), 0},
        {ROCKE_OP_ARITH_FMAX, rocke_b_fmax, rocke_f16(), 0, rocke_f32(), 0},
        {ROCKE_OP_ARITH_FMIN, rocke_b_fmin, rocke_f32(), 0, rocke_f16(), 0},
        {ROCKE_OP_ARITH_AND, rocke_b_land, rocke_i32(), 0, rocke_i16(), 0},
        {ROCKE_OP_ARITH_OR, rocke_b_lor, rocke_i1(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_XOR, rocke_b_xor, rocke_i32(), 0, rocke_i64(), 0},
        {ROCKE_OP_ARITH_SHL, rocke_b_shl, rocke_i32(), 0, rocke_i64(), 0},
        {ROCKE_OP_ARITH_LSHR, rocke_b_lshr, rocke_i64(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_SMAX, rocke_b_smax, rocke_i32(), 0, rocke_i64(), 0},
        {ROCKE_OP_ARITH_SMIN, rocke_b_smin, rocke_i64(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_UMAX, rocke_b_umax, rocke_u8(), 0, rocke_i32(), 0},
        {ROCKE_OP_ARITH_UMIN, rocke_b_umin, rocke_u8(), 0, rocke_i32(), 0},
        {ROCKE_OP_VECTOR_ADD, rocke_b_vector_add, rocke_i32(), 2, rocke_i32(), 4},
        {ROCKE_OP_VECTOR_SUB, rocke_b_vector_sub, rocke_i32(), 2, rocke_i16(), 2},
        {ROCKE_OP_VECTOR_MUL, rocke_b_vector_mul, rocke_f32(), 2, rocke_f16(), 2},
        {ROCKE_OP_VECTOR_AND, rocke_b_vector_and, rocke_i32(), 2, rocke_i32(), 4},
        {ROCKE_OP_VECTOR_OR, rocke_b_vector_or, rocke_i32(), 2, rocke_i16(), 2},
        {ROCKE_OP_VECTOR_SHL, rocke_b_vector_shl, rocke_i32(), 2, rocke_i16(), 2},
        {ROCKE_OP_VECTOR_LSHR, rocke_b_vector_lshr, rocke_i32(), 2, rocke_i16(), 2},
        {ROCKE_OP_VECTOR_SMAX, rocke_b_vector_smax, rocke_i32(), 2, rocke_i32(), 4},
        {ROCKE_OP_VECTOR_SMIN, rocke_b_vector_smin, rocke_i16(), 2, rocke_i32(), 2},
        {ROCKE_OP_VECTOR_MAX, rocke_b_vector_max, rocke_f32(), 2, rocke_f32(), 4},
    };
    int covered = 0;
    for(int opcode = ROCKE_OP_INVALID + 1; opcode < ROCKE_OP__COUNT; ++opcode)
        covered += rocke_i_is_same_type_binary_op((rocke_opcode_t)opcode) ? 1 : 0;
    EXPECT(covered == (int)(sizeof cases / sizeof cases[0]),
           "one case per same-type binary op: " + std::to_string(covered) + " ops");
    for(const MixedOperandsCase& c : cases)
    {
        EXPECT(rocke_i_is_same_type_binary_op(c.opcode),
               std::string(rocke_opcode_name(c.opcode)) + " is a same-type binary op");
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_mixed_operands");
        rocke_value_t* operands[2]
            = {rocke_b_param(&b, "lhs", lanes_of(&b, c.lhs_element, c.lhs_lanes), nullptr),
               rocke_b_param(&b, "rhs", lanes_of(&b, c.rhs_element, c.rhs_lanes), nullptr)};
        const std::string want
            = operand_types_message(c.opcode, operands[0]->type, operands[1]->type);
        expect_builder_type_error(
            [&](rocke_ir_builder_t* builder) {
                c.build(
                    builder,
                    rocke_b_param(
                        builder, "lhs", lanes_of(builder, c.lhs_element, c.lhs_lanes), nullptr),
                    rocke_b_param(
                        builder, "rhs", lanes_of(builder, c.rhs_element, c.rhs_lanes), nullptr));
            },
            want);
        rocke_i_op1(&b, c.opcode, operands, 2, operands[0]->type, nullptr, "mixed");
        rocke_b_ret(&b);
        const std::string errors = verify_errors(&b);
        EXPECT(errors == want + "\n", "verifier: want '" + want + "', got '" + errors + "'");
        const std::string lowered_want = both_lowerings_refuse(ROCKE_ERR_TYPE, want);
        const std::string lowered = lowering_refusals(&b);
        EXPECT(lowered == lowered_want, "want " + lowered_want + ", got " + lowered);
        rocke_ir_builder_free(&b);
    }
}

/* A raw value (rocke_i_value_named given no type) has no type: the operand rule
 * refuses it as ROCKE_ERR_TYPE naming the operand, before it reads any type's
 * name, with the Python builder's text (test_lshr.py). */
void case_builder_refuses_an_operand_with_no_type()
{
    for(int untyped_index : {0, 1})
    {
        expect_builder_type_error(
            [untyped_index](rocke_ir_builder_t* b) {
                rocke_value_t* typed = rocke_b_param(b, "typed", rocke_i32(), nullptr);
                rocke_value_t* untyped = rocke_i_value_named(b, "%untyped", nullptr);
                if(untyped_index == 0)
                    rocke_b_add(b, untyped, typed);
                else
                    rocke_b_add(b, typed, untyped);
            },
            "arith.add operand " + std::to_string(untyped_index)
                + " failed - type=None, expected an IR type");
    }
}

/* A lane-wise op takes vectors: its builder refuses a scalar first operand by
 * its type, ROCKE_ERR_TYPE, with the Python builder's text. */
void case_vector_builders_refuse_a_scalar_first_operand()
{
    expect_builder_type_error(
        [](rocke_ir_builder_t* b) {
            rocke_value_t* scalar = rocke_b_param(b, "scalar", rocke_i32(), nullptr);
            rocke_b_vector_and(b, scalar, scalar);
        },
        "vector.and failed - first operand type='i32', expected a vector type");
    expect_builder_type_error(
        [](rocke_ir_builder_t* b) {
            rocke_value_t* scalar = rocke_b_param(b, "scalar", rocke_i32(), nullptr);
            rocke_b_vector_cmp(b, "lt", scalar, scalar);
        },
        "vector.cmp failed - first operand type='i32', expected a vector type");
    expect_builder_type_error(
        [](rocke_ir_builder_t* b) {
            rocke_value_t* scalar = rocke_b_param(b, "scalar", rocke_i32(), nullptr);
            rocke_b_vector_fma(b, scalar, scalar, scalar);
        },
        "vector.fma failed - first operand type='i32', expected a vector type");
}

/* The lane-wise FMA takes three vectors of one type. Its builder refuses the
 * first operand that differs, by its ordinal, with the operand rule's text;
 * written past the builder, the verifier reports it and the LLVM and HIP
 * lowerings refuse it with ROCKE_ERR_TYPE and the same text, instead of
 * emitting an llvm.fmuladd.v2f32 of a <2 x half> (test_lshr.py). */
void case_vector_fma_refuses_operands_of_more_than_one_type()
{
    const struct
    {
        const rocke_type_t* other_element;
        int other_lanes;
        int other_index;
        const char* want;
    } cases[] = {
        {rocke_f16(),
         2,
         1,
         "vector.fma failed - second operand type='vec<f16x2>', expected the first operand's "
         "type 'vec<f32x2>'"},
        {rocke_f32(),
         4,
         2,
         "vector.fma failed - third operand type='vec<f32x4>', expected the first operand's "
         "type 'vec<f32x2>'"},
    };
    for(const auto& c : cases)
    {
        const auto operands_of = [&c](rocke_ir_builder_t* builder, rocke_value_t** operands) {
            rocke_value_t* first = rocke_b_param(
                builder, "first", rocke_vector_type(builder, rocke_f32(), 2), nullptr);
            rocke_value_t* other
                = rocke_b_param(builder,
                                "other",
                                rocke_vector_type(builder, c.other_element, c.other_lanes),
                                nullptr);
            for(int index = 0; index < 3; ++index)
                operands[index] = index == c.other_index ? other : first;
        };
        expect_builder_type_error(
            [&](rocke_ir_builder_t* builder) {
                rocke_value_t* operands[3];
                operands_of(builder, operands);
                rocke_b_vector_fma(builder, operands[0], operands[1], operands[2]);
            },
            c.want);
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_vector_fma_mixed");
        rocke_value_t* operands[3];
        operands_of(&b, operands);
        rocke_i_op1(&b, ROCKE_OP_VECTOR_FMA, operands, 3, operands[0]->type, nullptr, "vfma");
        rocke_b_ret(&b);
        const std::string errors = verify_errors(&b);
        EXPECT(errors == std::string(c.want) + "\n",
               "verifier: want '" + std::string(c.want) + "', got '" + errors + "'");
        const std::string lowered_want = both_lowerings_refuse(ROCKE_ERR_TYPE, c.want);
        const std::string lowered = lowering_refusals(&b);
        EXPECT(lowered == lowered_want, "want " + lowered_want + ", got " + lowered);
        rocke_ir_builder_free(&b);
    }
}

/* A lane-wise FMA with two operands, written past the builder, is refused by
 * the verifier, and by the LLVM and HIP lowerings with ROCKE_ERR_VALUE before
 * they read the missing third operand, with the arity rule's text
 * (test_lshr.py). */
void case_vector_fma_arity_rule()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_vector_fma_two_operands");
    const rocke_type_t* type = rocke_vector_type(&b, rocke_f32(), 2);
    rocke_value_t* operands[2]
        = {rocke_b_param(&b, "first", type, nullptr), rocke_b_param(&b, "second", type, nullptr)};
    rocke_i_op1(&b, ROCKE_OP_VECTOR_FMA, operands, 2, type, nullptr, "vfma");
    rocke_b_ret(&b);
    const std::string want = "vector.fma failed - operands / results=2 / 1, expected 3 / 1";
    const std::string errors = verify_errors(&b);
    EXPECT(errors == want + "\n", "verifier: want '" + want + "', got '" + errors + "'");
    const std::string lowered_want = both_lowerings_refuse(ROCKE_ERR_VALUE, want);
    const std::string lowered = lowering_refusals(&b);
    EXPECT(lowered == lowered_want, "want " + lowered_want + ", got " + lowered);
    rocke_ir_builder_free(&b);
}

/* An i1 is a C++ bool in HIP, where ~ promotes to int (~true is -2, still
 * true): the HIP lowering negates an i1 with ! and keeps ~ for the other
 * integers (test_lshr.py). */
void case_hip_not_is_logical_on_i1()
{
    for(const rocke_type_t* type : {rocke_i1(), rocke_i32()})
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_hip_not");
        rocke_value_t* negated = rocke_b_lnot(&b, rocke_b_param(&b, "value", type, nullptr));
        rocke_b_ret(&b);
        const bool is_i1 = type == rocke_i1();
        const std::string want = std::string(is_i1 ? "bool " : "int ") + (negated->name + 1) + " = "
                                 + (is_i1 ? "!" : "~") + "value;";
        const std::string hip = lower_hip(&b);
        EXPECT(hip.find(want) != std::string::npos,
               std::string("HIP not of ") + type->name + ": want '" + want + "'");
        rocke_ir_builder_free(&b);
    }
}

bool folded_store_value(rocke_ir_builder_t* b, int64_t* out);

/* A binary op or compare with one operand, written past the builders, is
 * refused by the verifier, and by the LLVM and HIP lowerings with
 * ROCKE_ERR_VALUE before they read the missing operand, with the arity rule's
 * text; the folder leaves a one-operand add of a constant unfolded. The same
 * cases as test_unsigned_int_ops.py. */
void case_binary_arity_rule()
{
    const struct
    {
        rocke_opcode_t opcode;
        const char* pred; /* nullptr: no pred attr */
        bool vector;
        bool float_operand;
    } cases[] = {
        {ROCKE_OP_ARITH_ADD, nullptr, false, false},
        {ROCKE_OP_ARITH_CMP, "lt", false, false},
        {ROCKE_OP_ARITH_FCMP, "olt", false, true},
        {ROCKE_OP_VECTOR_CMP, "lt", true, false},
    };
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_one_operand");
        const rocke_type_t* element = c.float_operand ? rocke_f32() : rocke_i32();
        const rocke_type_t* type = c.vector ? rocke_vector_type(&b, element, 2) : element;
        const bool compare = rocke_i_is_compare_op(c.opcode);
        const rocke_type_t* result_type
            = !compare ? type : (c.vector ? rocke_vector_type(&b, rocke_i1(), 2) : rocke_i1());
        rocke_value_t* operands[1] = {rocke_b_param(&b, "value", type, nullptr)};
        rocke_attr_map_t attrs = rocke_i_attrs(&b);
        if(c.pred)
            rocke_attr_set_str(&b, &attrs, "pred", c.pred);
        rocke_i_op1(&b, c.opcode, operands, 1, result_type, &attrs, "one");
        rocke_b_ret(&b);
        const std::string want = std::string(rocke_opcode_name(c.opcode))
                                 + " failed - operands / results=1 / 1, expected 2 / 1";
        const std::string errors = verify_errors(&b);
        EXPECT(errors == want + "\n", "verifier: want '" + want + "', got '" + errors + "'");
        const std::string lowered_want = both_lowerings_refuse(ROCKE_ERR_VALUE, want);
        const std::string lowered = lowering_refusals(&b);
        EXPECT(lowered == lowered_want, "want " + lowered_want + ", got " + lowered);
        rocke_ir_builder_free(&b);
    }
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_fold_one_operand");
    rocke_value_t* out
        = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i32(), "global"), nullptr);
    rocke_value_t* operands[1] = {rocke_b_const_i32(&b, 1)};
    rocke_value_t* value
        = rocke_i_op1(&b, ROCKE_OP_ARITH_ADD, operands, 1, rocke_i32(), nullptr, "one");
    rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), value, 4);
    int64_t folded = -2;
    EXPECT(!folded_store_value(&b, &folded), "a one-operand arith.add was folded");
    rocke_ir_builder_free(&b);
}

/* add_nuw or cmp_ult of `ta` and `tb` must be refused with `status` and
 * exactly `want`. */
void expect_builder_refusal(bool add,
                            const rocke_type_t* ta,
                            const rocke_type_t* tb,
                            rocke_status_t status,
                            const std::string& want)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_refusal");
    rocke_value_t* a = rocke_b_param(&b, "a", ta, nullptr);
    rocke_value_t* c = rocke_b_param(&b, "b", tb, nullptr);
    std::string got = "no refusal";
    bool refused = false;
    try
    {
        add ? rocke_b_add_nuw(&b, a, c) : rocke_b_cmp_ult(&b, a, c);
    }
    catch(const ckc::Error& error)
    {
        got = error.what();
        refused = error.code() == status && want == got;
    }
    EXPECT(refused, "want '" + want + "', got '" + got + "'");
    rocke_ir_builder_free(&b);
}

/* The unsigned forms take i32 or i64 operands (the nuw and ult rules,
 * ROCKE_ERR_VALUE, checked first) of one type (the operand rule,
 * ROCKE_ERR_TYPE), with the Python builders' texts. */
void case_builder_refusals()
{
    expect_builder_refusal(true,
                           rocke_f32(),
                           rocke_f32(),
                           ROCKE_ERR_VALUE,
                           "arith.add: wrap 'nuw' needs i32 or i64 operands, got f32");
    expect_builder_refusal(false,
                           rocke_i32(),
                           rocke_i64(),
                           ROCKE_ERR_TYPE,
                           "arith.cmp failed - second operand type='i64', expected the first "
                           "operand's type 'i32'");
    expect_builder_refusal(true,
                           rocke_i32(),
                           rocke_i64(),
                           ROCKE_ERR_TYPE,
                           "arith.add failed - second operand type='i64', expected the first "
                           "operand's type 'i32'");
    expect_builder_refusal(true,
                           rocke_i1(),
                           rocke_i1(),
                           ROCKE_ERR_VALUE,
                           "arith.add: wrap 'nuw' needs i32 or i64 operands, got i1");
    expect_builder_refusal(false,
                           rocke_f32(),
                           rocke_f32(),
                           ROCKE_ERR_VALUE,
                           "arith.cmp: pred 'ult' needs i32 or i64 operands, got f32");
    expect_builder_refusal(true,
                           rocke_i8(),
                           rocke_i32(),
                           ROCKE_ERR_VALUE,
                           "arith.add: wrap 'nuw' needs i32 or i64 operands, got i8");
    expect_builder_refusal(false,
                           rocke_i8(),
                           rocke_i32(),
                           ROCKE_ERR_VALUE,
                           "arith.cmp: pred 'ult' needs i32 or i64 operands, got i8");
}

/* The operand rule names operands by ordinal, so it refuses more operands than
 * it names before it reads any type (Python test_lshr.py). */
void case_operand_rule_refuses_more_operands_than_it_names()
{
    const rocke_type_t* const operand_types[4]
        = {rocke_f32(), rocke_f32(), rocke_f32(), rocke_f32()};
    char error[ROCKE_ERR_MSG_CAP];
    const char* got
        = rocke_i_operand_types_error("vector.fma", operand_types, 4, error, sizeof error);
    const std::string want = "vector.fma operand rule failed - operands=4, expected at most 3";
    EXPECT(got && want == got, "want '" + want + "', got '" + (got ? got : "") + "'");
}

void case_verifier_accepts()
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_valid");
    rocke_value_t* a = rocke_b_param(&b, "a", rocke_i32(), nullptr);
    rocke_value_t* n = rocke_b_param(&b, "n", rocke_i32(), nullptr);
    rocke_b_add_nuw(&b, a, n);
    rocke_b_cmp_ult(&b, a, n);
    marker(&b, a, n, [&](rocke_attr_map_t* m) {
        rocke_attr_set_int(&b, m, "multiple_of", 4);
        rocke_attr_set_int(&b, m, "residue", 3);
        rocke_attr_set_int(&b, m, "ge", 0);
        rocke_attr_set_int(&b, m, "lt_plus", 1);
    });
    marker(&b, a, nullptr, [&](rocke_attr_map_t* m) {
        rocke_attr_set_int(&b, m, "multiple_of", 1);
        rocke_attr_set_int(&b, m, "residue", 0);
        rocke_attr_set_int(&b, m, "ge", 0);
        rocke_attr_set_int(&b, m, "lt", 64);
    });
    rocke_b_ret(&b);
    const std::string errors = verify_errors(&b);
    EXPECT(errors.empty(), "verifier rejected valid attrs: " + errors);
    rocke_ir_builder_free(&b);
}

/* One bad marker per case; the verifier must report exactly `want`. */
template <typename SetAttrs>
void expect_marker_rejected(bool runtime_end, SetAttrs set_attrs, const char* want)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_bad_marker");
    rocke_value_t* a = rocke_b_param(&b, "a", rocke_i32(), nullptr);
    rocke_value_t* n = rocke_b_param(&b, "n", rocke_i32(), nullptr);
    marker(&b, a, runtime_end ? n : nullptr, [&](rocke_attr_map_t* m) { set_attrs(&b, m); });
    rocke_b_ret(&b);
    const std::string errors = verify_errors(&b);
    EXPECT(errors == std::string(want) + "\n",
           std::string("want '") + want + "', got '" + errors + "'");
    rocke_ir_builder_free(&b);
}

void case_verifier_rejects_marker_attrs()
{
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 0);
            rocke_attr_set_int(b, m, "residue", 0);
        },
        "arith.assume_range: multiple_of must be >= 1, got 0");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 4);
            rocke_attr_set_int(b, m, "residue", 4);
        },
        "arith.assume_range: residue must be in [0, 4), got 4");
    expect_marker_rejected(
        true,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_int(b, m, "lt", 8);
            rocke_attr_set_int(b, m, "lt_plus", 0);
        },
        "arith.assume_range: lt is given both as an attr and as an operand; "
        "state one upper end");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_int(b, m, "ge", 8);
            rocke_attr_set_int(b, m, "lt", 8);
        },
        "arith.assume_range: ge (8) must be below lt (8)");
    /* A value that is not an integer is refused before any value rule reads it. */
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_str(b, m, "multiple_of", "4");
            rocke_attr_set_int(b, m, "residue", 0);
        },
        "arith.assume_range: multiple_of must be an integer");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_bool(b, m, "ge", true);
        },
        "arith.assume_range: ge must be an integer");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_float(b, m, "lt", 8.5);
        },
        "arith.assume_range: lt must be an integer");
    expect_marker_rejected(
        true,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_bool(b, m, "lt_plus", false);
        },
        "arith.assume_range: lt_plus must be an integer");
}

/* nuw and ult take i32 or i64 operands (add_nuw's and cmp_ult's forms); the
 * verifier refuses raw IR of another type with the Python texts. */
void case_unsigned_forms_need_i32_or_i64()
{
    for(const rocke_type_t* type : {rocke_i1(), rocke_i8()})
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_narrow");
        rocke_value_t* a = rocke_b_param(&b, "a", type, nullptr);
        rocke_value_t* operands[2] = {a, a};
        rocke_attr_map_t add_attrs = rocke_i_attrs(&b);
        rocke_attr_set_str(&b, &add_attrs, "wrap", "nuw");
        rocke_i_op1(&b, ROCKE_OP_ARITH_ADD, operands, 2, type, &add_attrs, "sum");
        rocke_attr_map_t cmp_attrs = rocke_i_attrs(&b);
        rocke_attr_set_str(&b, &cmp_attrs, "pred", "ult");
        rocke_i_op1(&b, ROCKE_OP_ARITH_CMP, operands, 2, rocke_i1(), &cmp_attrs, "below");
        rocke_b_ret(&b);
        const std::string name = type->name;
        const std::string want = "arith.add: wrap 'nuw' needs i32 or i64 operands, got " + name
                                 + "\narith.cmp: pred 'ult' needs i32 or i64 operands, got " + name
                                 + "\n";
        const std::string errors = verify_errors(&b);
        EXPECT(errors == want, "want '" + want + "', got '" + errors + "'");
        rocke_ir_builder_free(&b);
    }
}

/* A compare's operand type in the cases below; the vector and pointer types
 * are made per builder. */
enum class CompareOperand
{
    i8,
    i32,
    i64,
    f16,
    f32,
    ptr_i32,
    vec_i16x2,
    vec_i32x2,
    vec_i32x4,
    vec_f32x2
};

const rocke_type_t* compare_operand_type(rocke_ir_builder_t* b, CompareOperand operand)
{
    switch(operand)
    {
    case CompareOperand::i8:
        return rocke_i8();
    case CompareOperand::i32:
        return rocke_i32();
    case CompareOperand::i64:
        return rocke_i64();
    case CompareOperand::f16:
        return rocke_f16();
    case CompareOperand::f32:
        return rocke_f32();
    case CompareOperand::ptr_i32:
        return rocke_ptr_type(b, rocke_i32(), "global");
    case CompareOperand::vec_i16x2:
        return rocke_vector_type(b, rocke_i16(), 2);
    case CompareOperand::vec_i32x2:
        return rocke_vector_type(b, rocke_i32(), 2);
    case CompareOperand::vec_i32x4:
        return rocke_vector_type(b, rocke_i32(), 4);
    case CompareOperand::vec_f32x2:
        return rocke_vector_type(b, rocke_f32(), 2);
    }
    return nullptr;
}

/* A compare's pred attr in the cases below: absent, or of another kind than a
 * string (the int 1), when the case names no pred string. */
const char* const kAbsentPred = "(absent)";

/* The verifier and the LLVM and HIP lowerings refuse a compare the IR rule
 * refuses, all in the rule's words, the lowerings with the rule's status: an
 * arith.cmp pred is required and checked first ("ult" takes i32 or i64
 * operands), then its operands must have one type, an integer or i1 (operands
 * of two types, or a float or pointer one, are not lowered to an integer
 * compare: ROCKE_ERR_TYPE); vector.cmp takes the signed orderings and equality
 * only, on two vectors of one type with integer or i1 lanes; arith.fcmp
 * requires its own predicates, on two f16, bf16, f32 or f64 operands of one
 * type; an unknown or non-string pred is neither read as a default nor passed
 * through. The same cases and texts as test_unsigned_int_ops.py. */
void case_compares_refused_by_the_ir_rule()
{
    struct CompareCase
    {
        rocke_opcode_t opcode;
        const char* pred; /* nullptr: the int attr 1; kAbsentPred: no attr */
        CompareOperand lhs;
        CompareOperand rhs;
        rocke_status_t status;
        const char* want;
    };
    const CompareCase cases[] = {
        {ROCKE_OP_ARITH_CMP,
         "ult",
         CompareOperand::f32,
         CompareOperand::f32,
         ROCKE_ERR_VALUE,
         "arith.cmp: pred 'ult' needs i32 or i64 operands, got f32"},
        {ROCKE_OP_ARITH_CMP,
         "ult",
         CompareOperand::i8,
         CompareOperand::i8,
         ROCKE_ERR_VALUE,
         "arith.cmp: pred 'ult' needs i32 or i64 operands, got i8"},
        {ROCKE_OP_ARITH_CMP,
         "ugt",
         CompareOperand::i32,
         CompareOperand::i32,
         ROCKE_ERR_VALUE,
         "arith.cmp: unknown pred 'ugt'"},
        {ROCKE_OP_ARITH_CMP,
         nullptr,
         CompareOperand::i32,
         CompareOperand::i32,
         ROCKE_ERR_VALUE,
         "arith.cmp: pred must be a string"},
        {ROCKE_OP_ARITH_CMP,
         kAbsentPred,
         CompareOperand::i32,
         CompareOperand::i32,
         ROCKE_ERR_VALUE,
         "op 'arith.cmp' missing required attr 'pred'"},
        {ROCKE_OP_ARITH_CMP,
         "lt",
         CompareOperand::i32,
         CompareOperand::i64,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - second operand type='i64', expected the first operand's type 'i32'"},
        {ROCKE_OP_ARITH_CMP,
         "lt",
         CompareOperand::f32,
         CompareOperand::f32,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - operand type='f32', expected an integer or i1 type (compare "
         "floats with fcmp)"},
        {ROCKE_OP_ARITH_CMP,
         "eq",
         CompareOperand::ptr_i32,
         CompareOperand::ptr_i32,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - operand type='ptr<i32,global>', expected an integer or i1 type "
         "(compare floats with fcmp)"},
        {ROCKE_OP_VECTOR_CMP,
         "ult",
         CompareOperand::vec_i32x2,
         CompareOperand::vec_i32x2,
         ROCKE_ERR_VALUE,
         "vector.cmp: unknown pred 'ult'"},
        {ROCKE_OP_VECTOR_CMP,
         nullptr,
         CompareOperand::vec_i32x2,
         CompareOperand::vec_i32x2,
         ROCKE_ERR_VALUE,
         "vector.cmp: pred must be a string"},
        {ROCKE_OP_VECTOR_CMP,
         "lt",
         CompareOperand::vec_i32x2,
         CompareOperand::vec_i16x2,
         ROCKE_ERR_TYPE,
         "vector.cmp failed - second operand type='vec<i16x2>', expected the first operand's "
         "type 'vec<i32x2>'"},
        {ROCKE_OP_VECTOR_CMP,
         "lt",
         CompareOperand::vec_f32x2,
         CompareOperand::vec_f32x2,
         ROCKE_ERR_TYPE,
         "vector.cmp failed - element type='f32', expected an integer or i1 type"},
        {ROCKE_OP_ARITH_FCMP,
         "lt",
         CompareOperand::f32,
         CompareOperand::f32,
         ROCKE_ERR_VALUE,
         "arith.fcmp: unknown pred 'lt'"},
        {ROCKE_OP_ARITH_FCMP,
         nullptr,
         CompareOperand::f32,
         CompareOperand::f32,
         ROCKE_ERR_VALUE,
         "arith.fcmp: pred must be a string"},
        {ROCKE_OP_ARITH_FCMP,
         kAbsentPred,
         CompareOperand::f32,
         CompareOperand::f32,
         ROCKE_ERR_VALUE,
         "op 'arith.fcmp' missing required attr 'pred'"},
        {ROCKE_OP_ARITH_FCMP,
         "olt",
         CompareOperand::f32,
         CompareOperand::f16,
         ROCKE_ERR_TYPE,
         "arith.fcmp failed - second operand type='f16', expected the first operand's type "
         "'f32'"},
        {ROCKE_OP_ARITH_FCMP,
         "olt",
         CompareOperand::i32,
         CompareOperand::i32,
         ROCKE_ERR_TYPE,
         "arith.fcmp failed - operand type='i32', expected f16, bf16, f32 or f64 (compare "
         "integers with cmp)"},
    };
    const auto status_text = [](rocke_status_t status, const std::string& message) {
        return "status " + std::to_string((int)status) + " '" + message + "'";
    };
    for(const CompareCase& c : cases)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "refused_compare");
        const rocke_type_t* type = compare_operand_type(&b, c.lhs);
        rocke_value_t* operands[2]
            = {rocke_b_param(&b, "lhs", type, nullptr),
               rocke_b_param(&b, "rhs", compare_operand_type(&b, c.rhs), nullptr)};
        rocke_attr_map_t attrs = rocke_i_attrs(&b);
        if(!c.pred)
        {
            rocke_attr_set_int(&b, &attrs, "pred", 1);
        }
        else if(c.pred != kAbsentPred)
        {
            rocke_attr_set_str(&b, &attrs, "pred", c.pred);
        }
        const rocke_type_t* result_type = c.opcode == ROCKE_OP_VECTOR_CMP
                                              ? rocke_vector_type(&b, rocke_i1(), type->count)
                                              : rocke_i1();
        rocke_i_op1(&b, c.opcode, operands, 2, result_type, &attrs, "cmp");
        rocke_b_ret(&b);

        const std::string verifier_want = std::string(c.want) + "\n";
        const std::string verifier_got = verify_errors(&b);
        EXPECT(verifier_got == verifier_want,
               "verifier: want '" + verifier_want + "', got '" + verifier_got + "'");

        const std::string want = status_text(c.status, c.want);
        std::string llvm;
        std::string llvm_error;
        const rocke_status_t llvm_status = lower_llvm_status(&b, &llvm, &llvm_error);
        const std::string llvm_got = status_text(llvm_status, llvm_error);
        EXPECT(llvm_got == want, "LLVM: want " + want + ", got " + llvm_got);

        std::string hip;
        const rocke_status_t hip_status = lower_hip_status(&b, &hip);
        const char* hip_error = rocke_ir_builder_error(&b);
        const std::string hip_got = status_text(hip_status, hip_error ? hip_error : "");
        EXPECT(hip_got == want, "HIP: want " + want + ", got " + hip_got);
        rocke_ir_builder_free(&b);
    }
}

/* The C API compare builders refuse what the IR rule refuses, with the rule's
 * status and text (the same texts as the Python builders). */
void case_compare_builders_refuse_what_the_ir_rule_refuses()
{
    enum class Builder
    {
        cmp_lt,
        cmp_eq,
        fcmp,
        fcmp_of_integers,
        vector_cmp,
        vector_cmp_of_floats,
        cmp_of_i32_and_i64,
        fcmp_of_f32_and_f16,
        vector_cmp_of_2_and_4_lanes
    };
    const struct
    {
        Builder builder;
        rocke_status_t status;
        const char* want;
    } cases[] = {
        {Builder::cmp_lt,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - operand type='f32', expected an integer or i1 type (compare "
         "floats with fcmp)"},
        {Builder::cmp_eq,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - operand type='f32', expected an integer or i1 type (compare "
         "floats with fcmp)"},
        {Builder::fcmp, ROCKE_ERR_VALUE, "arith.fcmp: unknown pred 'lt'"},
        {Builder::fcmp_of_integers,
         ROCKE_ERR_TYPE,
         "arith.fcmp failed - operand type='i32', expected f16, bf16, f32 or f64 (compare "
         "integers with cmp)"},
        {Builder::vector_cmp, ROCKE_ERR_VALUE, "vector.cmp: unknown pred 'ult'"},
        {Builder::vector_cmp_of_floats,
         ROCKE_ERR_TYPE,
         "vector.cmp failed - element type='f32', expected an integer or i1 type"},
        {Builder::cmp_of_i32_and_i64,
         ROCKE_ERR_TYPE,
         "arith.cmp failed - second operand type='i64', expected the first operand's type 'i32'"},
        {Builder::fcmp_of_f32_and_f16,
         ROCKE_ERR_TYPE,
         "arith.fcmp failed - second operand type='f16', expected the first operand's type "
         "'f32'"},
        {Builder::vector_cmp_of_2_and_4_lanes,
         ROCKE_ERR_TYPE,
         "vector.cmp failed - second operand type='vec<i32x4>', expected the first operand's "
         "type 'vec<i32x2>'"},
    };
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "refused_compare_builder");
        rocke_value_t* f = rocke_b_param(&b, "f", rocke_f32(), nullptr);
        rocke_value_t* i = rocke_b_param(&b, "i", rocke_i32(), nullptr);
        rocke_value_t* v = rocke_b_param(&b, "v", rocke_vector_type(&b, rocke_i32(), 2), nullptr);
        rocke_value_t* fv = rocke_b_param(&b, "fv", rocke_vector_type(&b, rocke_f32(), 2), nullptr);
        rocke_value_t* wide = rocke_b_param(&b, "wide", rocke_i64(), nullptr);
        rocke_value_t* half = rocke_b_param(&b, "half", rocke_f16(), nullptr);
        rocke_value_t* v4 = rocke_b_param(&b, "v4", rocke_vector_type(&b, rocke_i32(), 4), nullptr);
        bool refused = false;
        std::string got;
        try
        {
            rocke_value_t* r = nullptr;
            switch(c.builder)
            {
            case Builder::cmp_lt:
                r = rocke_b_cmp_lt(&b, f, f);
                break;
            case Builder::cmp_eq:
                r = rocke_b_cmp_eq(&b, f, f);
                break;
            case Builder::fcmp:
                r = rocke_b_fcmp(&b, "lt", f, f);
                break;
            case Builder::fcmp_of_integers:
                r = rocke_b_fcmp(&b, "olt", i, i);
                break;
            case Builder::vector_cmp:
                r = rocke_b_vector_cmp(&b, "ult", v, v);
                break;
            case Builder::vector_cmp_of_floats:
                r = rocke_b_vector_cmp(&b, "lt", fv, fv);
                break;
            case Builder::cmp_of_i32_and_i64:
                r = rocke_b_cmp_lt(&b, i, wide);
                break;
            case Builder::fcmp_of_f32_and_f16:
                r = rocke_b_fcmp(&b, "olt", f, half);
                break;
            case Builder::vector_cmp_of_2_and_4_lanes:
                r = rocke_b_vector_cmp(&b, "lt", v, v4);
                break;
            }
            const char* error = rocke_ir_builder_error(&b);
            got = error ? error : "";
            refused = r == nullptr && rocke_ir_builder_status(&b) == c.status && got == c.want;
        }
        catch(const ckc::Error& error)
        {
            got = error.what();
            refused = error.code() == c.status && got == c.want;
        }
        EXPECT(refused, std::string("want '") + c.want + "', got '" + got + "'");
        rocke_ir_builder_free(&b);
    }
}

/* A wrap or pred of another attr kind is refused, not read as the default
 * (nsw, lt): by the verifier, and for the wrap by both lowerers (the
 * lowerers' refusal of such a pred is case_compares_refused_by_the_ir_rule). */
void case_non_string_wrap_and_pred()
{
    const char* wrap_error = "arith.add: wrap must be the string 'nsw' or 'nuw'";
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_non_string");
    rocke_value_t* a = rocke_b_param(&b, "a", rocke_i32(), nullptr);
    rocke_value_t* out
        = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i32(), "global"), nullptr);
    rocke_value_t* operands[2] = {a, a};
    rocke_attr_map_t add_attrs = rocke_i_attrs(&b);
    rocke_attr_set_int(&b, &add_attrs, "wrap", 1);
    rocke_value_t* sum
        = rocke_i_op1(&b, ROCKE_OP_ARITH_ADD, operands, 2, rocke_i32(), &add_attrs, "sum");
    rocke_attr_map_t cmp_attrs = rocke_i_attrs(&b);
    rocke_attr_set_int(&b, &cmp_attrs, "pred", 1);
    rocke_i_op1(&b, ROCKE_OP_ARITH_CMP, operands, 2, rocke_i1(), &cmp_attrs, "cmp");
    rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), sum, 1);
    rocke_b_ret(&b);
    const std::string errors = verify_errors(&b);
    const std::string want = std::string(wrap_error) + "\narith.cmp: pred must be a string\n";
    EXPECT(errors == want, "want '" + want + "', got '" + errors + "'");
    EXPECT(lower_llvm(&b).empty(), "LLVM lowering accepted a non-string wrap");
    EXPECT(lower_hip(&b).empty(), "HIP lowering accepted a non-string wrap");
    rocke_ir_builder_free(&b);
}

/* The folder's value for the op stored by the kernel's only store. */
bool folded_store_value(rocke_ir_builder_t* b, int64_t* out)
{
    rocke_kernel_def_t* k = rocke_ir_builder_kernel(b);
    rocke_optimize_kernel(b, k, 0);
    for(int i = 0; i < k->body->num_ops; ++i)
    {
        const rocke_op_t* op = k->body->ops[i];
        if(op->opcode != ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED)
            continue;
        const rocke_op_t* producer = op->operands[2]->op;
        return producer && producer->opcode == ROCKE_OP_ARITH_CONSTANT
               && rocke_attr_get_int(&producer->attrs, "value", out);
    }
    return false;
}

void case_constant_folding()
{
    /* A compare of constants that the IR rule refuses (no pred, or an unknown
     * one) is not folded: the verifier and the lowerings refuse it. The same
     * cases as test_a_compare_the_ir_rule_refuses_is_not_folded. */
    for(const char* pred : {kAbsentPred, "ugt"})
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_fold_refused");
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i1(), "global"), nullptr);
        rocke_value_t* operands[2] = {rocke_b_const_i32(&b, 1), rocke_b_const_i32(&b, 2)};
        rocke_attr_map_t attrs = rocke_i_attrs(&b);
        if(pred != kAbsentPred)
            rocke_attr_set_str(&b, &attrs, "pred", pred);
        rocke_value_t* compare
            = rocke_i_op1(&b, ROCKE_OP_ARITH_CMP, operands, 2, rocke_i1(), &attrs, "cmp");
        rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), compare, 1);
        int64_t folded = -2;
        EXPECT(!folded_store_value(&b, &folded),
               std::string("a compare with pred ") + pred + " was folded");
        rocke_ir_builder_free(&b);
    }

    /* An op of an i32 and an i64 constant breaks the operand rule, so it is not
     * folded into a constant that hides it. The same cases as
     * test_a_binary_op_of_different_types_is_not_folded. */
    for(rocke_opcode_t opcode : {ROCKE_OP_ARITH_CMP, ROCKE_OP_ARITH_ADD})
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops_fold_mixed");
        const bool compare = opcode == ROCKE_OP_ARITH_CMP;
        const rocke_type_t* result_type = compare ? rocke_i1() : rocke_i32();
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, result_type, "global"), nullptr);
        rocke_value_t* operands[2] = {rocke_b_const_i32(&b, 1), rocke_b_const_i64(&b, 2)};
        rocke_attr_map_t attrs = rocke_i_attrs(&b);
        if(compare)
            rocke_attr_set_str(&b, &attrs, "pred", "lt");
        rocke_value_t* value = rocke_i_op1(&b, opcode, operands, 2, result_type, &attrs, "mixed");
        rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), value, 1);
        int64_t folded = -2;
        EXPECT(!folded_store_value(&b, &folded),
               std::string("a mixed-type ") + rocke_opcode_name(opcode) + " was folded");
        rocke_ir_builder_free(&b);
    }

    const int64_t n = 7;
    for(bool wide : {false, true})
        for(int64_t x : {int64_t{-1}, int64_t{0}, n - 1, n})
        {
            rocke_ir_builder_t b;
            rocke_ir_builder_init(&b, "unsigned_int_ops_fold");
            rocke_value_t* out
                = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i1(), "global"), nullptr);
            rocke_value_t* xv = wide ? rocke_b_const_i64(&b, x) : rocke_b_const_i32(&b, x);
            rocke_value_t* nv = wide ? rocke_b_const_i64(&b, n) : rocke_b_const_i32(&b, n);
            rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), rocke_b_cmp_ult(&b, xv, nv), 1);
            int64_t got = -2;
            const int64_t want = (x >= 0 && x < n) ? 1 : 0;
            EXPECT(folded_store_value(&b, &got) && got == want,
                   "cmp_ult fold of " + std::to_string(x) + (wide ? " (i64)" : " (i32)"));
            rocke_ir_builder_free(&b);
        }

    /* A folded add_nuw holds the value the hardware computes: the i32 register
     * holds 0x80000000 (-2^31 read as signed), a signed compare sees it below
     * 0, and widening it as unsigned keeps 2^31. Same values as the Python
     * test. */
    enum class NuwUse
    {
        sum,
        signed_compare,
        zext
    };
    for(NuwUse use : {NuwUse::sum, NuwUse::signed_compare, NuwUse::zext})
    {
        rocke_ir_builder_t nb;
        rocke_ir_builder_init(&nb, "unsigned_int_ops_fold_nuw");
        const rocke_type_t* out_type = use == NuwUse::zext ? rocke_i64() : rocke_i32();
        rocke_value_t* out
            = rocke_b_param(&nb, "out", rocke_ptr_type(&nb, out_type, "global"), nullptr);
        rocke_value_t* sum
            = rocke_b_add_nuw(&nb, rocke_b_const_i32(&nb, INT32_MAX), rocke_b_const_i32(&nb, 1));
        rocke_value_t* stored = sum;
        int64_t want = INT32_MIN;
        const char* label = "sum";
        if(use == NuwUse::signed_compare)
        {
            stored = rocke_b_select(&nb,
                                    rocke_b_cmp_lt(&nb, sum, rocke_b_const_i32(&nb, 0)),
                                    rocke_b_const_i32(&nb, 1),
                                    rocke_b_const_i32(&nb, 2));
            want = 1;
            label = "signed compare";
        }
        else if(use == NuwUse::zext)
        {
            stored = rocke_b_zext(&nb, sum, rocke_i64());
            want = int64_t{1} << 31;
            label = "zext";
        }
        rocke_b_global_store(&nb, out, rocke_b_const_i32(&nb, 0), stored, 1);
        int64_t got = -2;
        EXPECT(folded_store_value(&nb, &got) && got == want,
               std::string("folded add_nuw ") + label + " gave " + std::to_string(got) + ", want "
                   + std::to_string(want));
        rocke_ir_builder_free(&nb);
    }

    /* A literal written in unsigned form is the same i32 bits as a folded sum:
     * 0x80000000 + 0 equals the literal 0x80000000. */
    {
        rocke_ir_builder_t lb;
        rocke_ir_builder_init(&lb, "unsigned_int_ops_fold_literal");
        rocke_value_t* out
            = rocke_b_param(&lb, "out", rocke_ptr_type(&lb, rocke_i32(), "global"), nullptr);
        rocke_value_t* sum
            = rocke_b_add(&lb, rocke_b_const_i32(&lb, 0x80000000LL), rocke_b_const_i32(&lb, 0));
        rocke_value_t* same = rocke_b_cmp_eq(&lb, sum, rocke_b_const_i32(&lb, 0x80000000LL));
        rocke_value_t* stored
            = rocke_b_select(&lb, same, rocke_b_const_i32(&lb, 1), rocke_b_const_i32(&lb, 2));
        rocke_b_global_store(&lb, out, rocke_b_const_i32(&lb, 0), stored, 1);
        int64_t got = -2;
        EXPECT(folded_store_value(&lb, &got) && got == 1,
               "folded 0x80000000 + 0 == 0x80000000 gave " + std::to_string(got) + ", want 1");
        rocke_ir_builder_free(&lb);
    }

    /* A folded div / mod holds what sdiv / srem compute (quotient rounded
     * toward zero); a zero divisor or a quotient past the signed range is not
     * folded. Same cases as the Python test. */
    struct DivModCase
    {
        bool is_mod;
        bool wide;
        int64_t a, d;
        bool folds;
        int64_t want;
    };
    const DivModCase div_mod_cases[] = {
        {true, false, 3, 0x80000000LL, true, 3},
        {false, false, 3, 0x80000000LL, true, 0},
        {true, false, -7, 2, true, -1},
        {false, false, -7, 2, true, -3},
        {true, false, 7, -2, true, 1},
        {false, false, 7, -2, true, -3},
        {true, false, -7, -2, true, -1},
        {false, false, INT32_MIN, 3, true, -715827882},
        {false, true, (int64_t{1} << 62) + 1, 1, true, (int64_t{1} << 62) + 1},
        {true, true, INT64_MIN, 3, true, -2},
        {false, false, INT32_MIN, -1, false, 0},
        {true, false, INT32_MIN, -1, false, 0},
        {false, true, INT64_MIN, -1, false, 0},
        {true, true, INT64_MIN, -1, false, 0},
        {false, false, 5, 0, false, 0},
        {true, false, 5, 0, false, 0},
    };
    for(const DivModCase& c : div_mod_cases)
    {
        rocke_ir_builder_t db;
        rocke_ir_builder_init(&db, "unsigned_int_ops_fold_div_mod");
        const rocke_type_t* type = c.wide ? rocke_i64() : rocke_i32();
        rocke_value_t* out
            = rocke_b_param(&db, "out", rocke_ptr_type(&db, type, "global"), nullptr);
        rocke_value_t* av = c.wide ? rocke_b_const_i64(&db, c.a) : rocke_b_const_i32(&db, c.a);
        rocke_value_t* dv = c.wide ? rocke_b_const_i64(&db, c.d) : rocke_b_const_i32(&db, c.d);
        rocke_value_t* q = c.is_mod ? rocke_b_mod(&db, av, dv) : rocke_b_div(&db, av, dv);
        rocke_b_global_store(&db, out, rocke_b_const_i32(&db, 0), q, 1);
        int64_t got = -2;
        const bool folded = folded_store_value(&db, &got);
        const std::string label = std::string(c.is_mod ? "mod " : "div ") + std::to_string(c.a)
                                  + ", " + std::to_string(c.d);
        EXPECT(folded == c.folds, label + (c.folds ? " did not fold" : " folded"));
        EXPECT(!c.folds || got == c.want,
               label + " gave " + std::to_string(got) + ", want " + std::to_string(c.want));
        rocke_ir_builder_free(&db);
    }

    /* A marker on a constant folds to the constant, even with a runtime end. */
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_fold_marker");
    rocke_value_t* out
        = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i32(), "global"), nullptr);
    rocke_value_t* n_param = rocke_b_param(&b, "n", rocke_i32(), nullptr);
    rocke_value_t* range = marker(&b, rocke_b_const_i32(&b, 5), n_param, [&](rocke_attr_map_t* m) {
        rocke_attr_set_int(&b, m, "multiple_of", 1);
        rocke_attr_set_int(&b, m, "residue", 0);
        rocke_attr_set_int(&b, m, "lt_plus", 0);
    });
    rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), range, 1);
    int64_t got = -2;
    EXPECT(folded_store_value(&b, &got) && got == 5, "marker on a constant folds");
    rocke_ir_builder_free(&b);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc == 2 && strcmp(argv[1], "--print-llvm") == 0)
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "unsigned_int_ops");
        build_two_op(&b, rocke_i32());
        const std::string ll = lower_llvm(&b);
        rocke_ir_builder_free(&b);
        if(ll.empty())
            return 1;
        fputs(ll.c_str(), stdout);
        return 0;
    }
    if(argc != 1)
    {
        /* A mistyped mode must not fall through to the self-test and pass. */
        fprintf(stderr, "usage: %s [--print-llvm]\n", argv[0]);
        return 2;
    }
    case_llvm_text();
    case_hip_text();
    case_lshr_hip_text();
    case_integer_element_builder_rule();
    case_integer_element_verifier_rule();
    case_integer_element_lowering_rule();
    case_mixed_operand_types();
    case_builder_refuses_an_operand_with_no_type();
    case_vector_builders_refuse_a_scalar_first_operand();
    case_vector_fma_refuses_operands_of_more_than_one_type();
    case_vector_fma_arity_rule();
    case_hip_not_is_logical_on_i1();
    case_binary_arity_rule();
    case_operand_rule_refuses_more_operands_than_it_names();
    case_compares_refused_by_the_ir_rule();
    case_compare_builders_refuse_what_the_ir_rule_refuses();
    case_builder_refusals();
    case_verifier_accepts();
    case_verifier_rejects_marker_attrs();
    case_non_string_wrap_and_pred();
    case_unsigned_forms_need_i32_or_i64();
    case_constant_folding();
    if(g_failures)
    {
        fprintf(stderr, "%d unsigned integer op check(s) failed\n", g_failures);
        return 1;
    }
    return 0;
}
