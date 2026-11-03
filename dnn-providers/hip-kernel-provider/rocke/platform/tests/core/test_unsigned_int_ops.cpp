// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_unsigned_int_ops.cpp -- the C++ engine's unsigned compare (cmp_ult), its
 * no-unsigned-wrap add (add_nuw) and the arith.assume_range range marker.
 *
 * Run with no argument it checks, through the C API: the LLVM and HIP text of
 * both ops, the builders' operand-type refusals, the verifier's acceptance of
 * the new attrs and its refusals of bad marker attrs (texts identical to the
 * Python verifier), and the constant folder (unsigned compare at the operand
 * width; constants read at their width; a marker on a constant folds to the
 * constant; signed div / mod rounded toward zero, undefined ones left
 * unfolded), and the verifier's i32 / i64 rule for the nuw and ult forms.
 *
 * Run with --print-llvm it prints the LLVM text of the two-op function built
 * through rocke_b_add_nuw / rocke_b_cmp_ult, so the Python test can compare it
 * with the same function built by the Python builder.
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

std::string lower_llvm(rocke_ir_builder_t* b)
{
    char* text = nullptr;
    std::string out;
    if(rocke_lower_kernel_to_llvm(
           rocke_ir_builder_kernel(b), ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &text)
           == ROCKE_OK
       && text)
        out = text;
    free(text);
    return out;
}

std::string lower_hip(rocke_ir_builder_t* b)
{
    rocke_strbuf_t out;
    rocke_strbuf_init(&out, 0);
    rocke_lower_hip_opts_t opts{};
    opts.arch = "gfx942";
    std::string hip;
    if(rocke_lower_kernel_to_hip(b, rocke_ir_builder_kernel(b), &opts, &out) == ROCKE_OK)
        hip = rocke_strbuf_cstr(&out);
    rocke_strbuf_free(&out);
    return hip;
}

/* A range marker op built directly (the C API has no builder entry for it):
 * operands [v] or [v, hi_value]; attrs set by `set_attrs`. */
template <typename SetAttrs>
rocke_value_t*
    marker(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* hi_value, SetAttrs set_attrs)
{
    rocke_value_t* operands[2] = {v, hi_value};
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    set_attrs(&attrs);
    return rocke_i_op1(
        b, ROCKE_OP_ARITH_ASSUME_RANGE, operands, hi_value ? 2 : 1, v->type, &attrs, "range");
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

void expect_builder_refusal(bool add, const rocke_type_t* ta, const rocke_type_t* tb)
{
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "unsigned_int_ops_refusal");
    rocke_value_t* a = rocke_b_param(&b, "a", ta, nullptr);
    rocke_value_t* c = rocke_b_param(&b, "b", tb, nullptr);
    const char* op_name = add ? "add_nuw" : "cmp_ult";
    const std::string want = std::string(op_name) + " expects two i32 or two i64 operands, got "
                             + ta->name + " / " + tb->name;
    bool refused = false;
    try
    {
        rocke_value_t* r = add ? rocke_b_add_nuw(&b, a, c) : rocke_b_cmp_ult(&b, a, c);
        refused = r == nullptr && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                  && want == rocke_ir_builder_error(&b);
    }
    catch(const ckc::Error& error)
    {
        refused = error.code() == ROCKE_ERR_VALUE && want == error.what();
    }
    EXPECT(refused, want);
    rocke_ir_builder_free(&b);
}

void case_builder_refusals()
{
    expect_builder_refusal(true, rocke_f32(), rocke_f32());
    expect_builder_refusal(false, rocke_i32(), rocke_i64());
    expect_builder_refusal(true, rocke_i1(), rocke_i1());
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
        rocke_attr_set_int(&b, m, "lo", 0);
        rocke_attr_set_int(&b, m, "hi_plus", 1);
    });
    marker(&b, a, nullptr, [&](rocke_attr_map_t* m) {
        rocke_attr_set_int(&b, m, "multiple_of", 1);
        rocke_attr_set_int(&b, m, "residue", 0);
        rocke_attr_set_int(&b, m, "lo", 0);
        rocke_attr_set_int(&b, m, "hi", 64);
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
            rocke_attr_set_int(b, m, "hi", 8);
            rocke_attr_set_int(b, m, "hi_plus", 0);
        },
        "arith.assume_range: hi and hi_value are both given; state at most one end");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_int(b, m, "lo", 8);
            rocke_attr_set_int(b, m, "hi", 8);
        },
        "arith.assume_range: lo (8) must be below hi (8)");
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
            rocke_attr_set_bool(b, m, "lo", true);
        },
        "arith.assume_range: lo must be an integer");
    expect_marker_rejected(
        false,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_float(b, m, "hi", 8.5);
        },
        "arith.assume_range: hi must be an integer");
    expect_marker_rejected(
        true,
        [](rocke_ir_builder_t* b, rocke_attr_map_t* m) {
            rocke_attr_set_int(b, m, "multiple_of", 1);
            rocke_attr_set_int(b, m, "residue", 0);
            rocke_attr_set_bool(b, m, "hi_plus", false);
        },
        "arith.assume_range: hi_plus must be an integer");
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

/* A wrap or pred of another attr kind is refused, not read as the default
 * (nsw, lt): by the verifier, and for the wrap by both lowerers. */
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
        rocke_attr_set_int(&b, m, "hi_plus", 0);
    });
    rocke_b_global_store(&b, out, rocke_b_const_i32(&b, 0), range, 1);
    int64_t got = -2;
    EXPECT(folded_store_value(&b, &got) && got == 5, "marker on a constant folds");
    rocke_ir_builder_free(&b);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc > 1 && strcmp(argv[1], "--print-llvm") == 0)
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
    case_llvm_text();
    case_hip_text();
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
