// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/matrix_call_lowering.cpp -- host unit test for the C++ half of
 * matrix intrinsic calls described as data.
 *
 * The same cases as tests/core/test_matrix_calls.py, against the C++ engine:
 * the per-flavor declaration a call derives, each operand form (as-is,
 * zero-padded words, even half-slots) and its refusal, the schema checks, and
 * the table's entries. Each case lowers a call this test builds into a lowerer
 * set up the way the engine's entry point sets one up, and compares the
 * emitted lines and error texts with the Python engine's, written out here.
 *
 * Plain executable: returns non-zero when a check fails. Registered via
 * tests/CMakeLists.txt.
 */
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/matrix_calls_internal.h"

using namespace ckc;

static int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do                                                                         \
    {                                                                          \
        if(!(cond))                                                            \
        {                                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); \
            ++g_failures;                                                      \
        }                                                                      \
    } while(0)

static void check_equal(const std::string& got, const std::string& want, const char* what)
{
    if(got != want)
    {
        fprintf(stderr, "FAIL: %s\n  got:  %s\n  want: %s\n", what, got.c_str(), want.c_str());
        ++g_failures;
    }
}

/* ------------------------------------------------------------------ fixtures */

/* A lowerer for one target and flavor, holding an entry block, set up as the
 * engine's entry point sets one up before it walks a kernel. */
struct Lowerer
{
    rocke_lower_t L;

    Lowerer(const char* gfx, rocke_llvm_flavor_t flavor)
    {
        memset(&L, 0, sizeof L);
        rocke_arena_init(&L.arena, 0);
        L.err = (char*)rocke_arena_calloc(&L.arena, ROCKE_ERR_MSG_CAP);
        rocke_vec_init(&L.blocks);
        rocke_vec_init(&L.needs);
        rocke_vec_init(&L.dyn_decls);
        L.flavor = flavor;
        /* The mode an unset ROCKE_ARCH_DOMAIN gives; the zeroed struct would
         * read as error mode. */
        L.arch_domain_mode = ROCKE_ARCH_DOMAIN_WARN;
        rocke_status_t st = ROCKE_OK;
        L.backend = rocke_ll_backend_for(gfx, &st);
        rocke_ll_new_block(&L, "entry");
    }
    ~Lowerer()
    {
        rocke_arena_destroy(&L.arena);
    }

    const rocke_type_t* vec(const rocke_type_t* elem, int count)
    {
        return rocke_i_vector_type_in_arena(&L.arena, elem, count);
    }

    /* The lines emitted so far, stripped of their indent. */
    std::vector<std::string> lines()
    {
        std::vector<std::string> out;
        rocke_ll_block_t* blk = rocke_ll_current(&L);
        for(size_t i = 0; i < blk->lines.len; ++i)
        {
            const char* line = blk->lines.data[i];
            while(*line == ' ')
                ++line;
            out.push_back(line);
        }
        return out;
    }

    /* The declaration the lowerer would emit for `key`. */
    std::string declared(const char* key)
    {
        for(size_t i = 0; i < L.dyn_decls.len; ++i)
        {
            if(strcmp(L.dyn_decls.data[i].key, key) == 0)
                return L.dyn_decls.data[i].decl;
        }
        const char* text = rocke_ll_decl_text(L.flavor, key);
        return text ? text : "";
    }
};

/* An MMA op on values %a, %b, %c with result %d. */
struct MmaOp
{
    rocke_value_t a, b, c, d;
    rocke_value_t* operands[3];
    rocke_value_t* results[1];
    rocke_op_t op;

    MmaOp(const rocke_type_t* ta,
          const rocke_type_t* tb,
          const rocke_type_t* tc,
          const rocke_type_t* td)
        : a{"%a", ta, NULL}
        , b{"%b", tb, NULL}
        , c{"%c", tc, NULL}
        , d{"%d", td, NULL}
    {
        operands[0] = &a;
        operands[1] = &b;
        operands[2] = &c;
        results[0] = &d;
        memset(&op, 0, sizeof op);
        op.opcode = ROCKE_OP_TILE_MMA;
        op.name = "tile.mma";
        op.operands = operands;
        op.num_operands = 3;
        op.results = results;
        op.num_results = 1;
    }
};

/* The message `fn` throws, or "" when it does not throw. */
static std::string error_of(const std::function<void()>& fn)
{
    try
    {
        fn();
    }
    catch(const Error& e)
    {
        return e.what();
    }
    return "";
}

static rocke_matrix_register_t reg(const char* elem, int count)
{
    return rocke_matrix_register_t{{elem, count}, NULL, 0};
}

static rocke_matrix_form_t as_is(const char* elem, int count)
{
    return rocke_matrix_form_t{ROCKE_MATRIX_FORM_AS_IS, reg(elem, count)};
}

static const rocke_matrix_arg_t kA = {ROCKE_MATRIX_ARG_A, NULL, 0, false};
static const rocke_matrix_arg_t kB = {ROCKE_MATRIX_ARG_B, NULL, 0, false};
static const rocke_matrix_arg_t kC = {ROCKE_MATRIX_ARG_C, NULL, 0, false};

static rocke_matrix_arg_t imm(const char* type, long long value, bool immarg = true)
{
    return rocke_matrix_arg_t{ROCKE_MATRIX_ARG_IMM, type, value, immarg};
}

/* -------------------------------------------------------------- the calls */

/* The gfx1250 K=64 fp8 WMMA described as data (Python _FP8_WMMA). */
static const rocke_matrix_arg_t k_fp8_args[]
    = {kA, kB, imm("i16", 0), kC, imm("i1", 0), imm("i1", 0)};
static const rocke_matrix_call_t k_fp8_wmma = {
    "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
    k_fp8_args,
    6,
    as_is("i32", 8),
    as_is("i32", 8),
    as_is("f32", 8),
    as_is("f32", 8),
    "wmma.gfx1250.f32.16x16x64.fp8.fp8",
};

/* An operand whose register type changed at LLVM 22 (Python _PER_FLAVOR). */
static const rocke_matrix_flavor_override_t k_i64_at_llvm22[] = {{"llvm22", {"i64", 0}}};
static const rocke_matrix_arg_t k_per_flavor_args[]
    = {kA, kB, kC, imm("i32", 0), imm("i32", 0), imm("i32", 0)};
static const rocke_matrix_form_t k_two_words_or_i64
    = {ROCKE_MATRIX_FORM_AS_IS, {{"i32", 2}, k_i64_at_llvm22, 1}};
static const rocke_matrix_call_t k_per_flavor = {
    "llvm.amdgcn.example.per.flavor",
    k_per_flavor_args,
    6,
    k_two_words_or_i64,
    k_two_words_or_i64,
    as_is("f32", 4),
    as_is("f32", 4),
    NULL,
};

static const rocke_matrix_arg_t k_abc[] = {kA, kB, kC};

static const rocke_matrix_call_t k_i16_accumulator = {
    "llvm.amdgcn.example.i16.acc",
    k_abc,
    3,
    as_is("i32", 8),
    as_is("i32", 8),
    as_is("i16", 8),
    as_is("f32", 4),
    NULL,
};

static const rocke_matrix_call_t k_padded = {
    "llvm.amdgcn.example.padded",
    k_abc,
    3,
    {ROCKE_MATRIX_FORM_ZERO_PADDED, reg("i32", 8)},
    {ROCKE_MATRIX_FORM_ZERO_PADDED, reg("i32", 8)},
    as_is("f32", 4),
    as_is("f32", 4),
    NULL,
};

static const rocke_matrix_arg_t k_even_halves_args[] = {kA, kB, kC, imm("i1", 0)};
static const rocke_matrix_call_t k_even_halves = {
    "llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16",
    k_even_halves_args,
    4,
    as_is("f16", 16),
    as_is("f16", 16),
    {ROCKE_MATRIX_FORM_EVEN_HALVES, reg("f16", 16)},
    {ROCKE_MATRIX_FORM_EVEN_HALVES, reg("f16", 16)},
    NULL,
};

static const rocke_llvm_flavor_t k_flavors[]
    = {ROCKE_LLVM_FLAVOR_LLVM20, ROCKE_LLVM_FLAVOR_LLVM22, ROCKE_LLVM_FLAVOR_LLVM23};

/* -------------------------------------------------------------- per flavor */

static void test_the_declaration_spells_each_flavors_register_type()
{
    check_equal(rocke_matrix_call_declaration(&k_per_flavor, ROCKE_LLVM_FLAVOR_LLVM20),
                "declare <4 x float> @llvm.amdgcn.example.per.flavor(<2 x i32>, <2 x i32>, "
                "<4 x float>, i32 immarg, i32 immarg, i32 immarg)",
                "llvm20 declaration");
    check_equal(rocke_matrix_call_declaration(&k_per_flavor, ROCKE_LLVM_FLAVOR_LLVM23),
                "declare <4 x float> @llvm.amdgcn.example.per.flavor(i64, i64, "
                "<4 x float>, i32 immarg, i32 immarg, i32 immarg)",
                "llvm23 inherits the llvm22 register type");
}

static void test_a_derived_declaration_equals_the_hand_written_one()
{
    for(rocke_llvm_flavor_t flavor : k_flavors)
        check_equal(rocke_matrix_call_declaration(&k_fp8_wmma, flavor),
                    rocke_ll_decl_text(flavor, "wmma.gfx1250.f32.16x16x64.fp8.fp8"),
                    rocke_llvm_flavor_name(flavor));
}

static void test_the_call_and_its_declaration_take_the_same_types()
{
    for(rocke_llvm_flavor_t flavor : k_flavors)
    {
        Lowerer lw("gfx942", flavor);
        const bool i64 = flavor != ROCKE_LLVM_FLAVOR_LLVM20;
        const rocke_type_t* operand = i64 ? rocke_i64() : lw.vec(rocke_i32(), 2);
        const rocke_type_t* f32x4 = lw.vec(rocke_f32(), 4);
        MmaOp m(operand, operand, f32x4, f32x4);
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "example_per_flavor", &k_per_flavor);
        const std::string text = i64 ? "i64" : "<2 x i32>";
        check_equal(lw.lines().at(0),
                    "%d = call <4 x float> @llvm.amdgcn.example.per.flavor(" + text + " %a, " + text
                        + " %b, <4 x float> %c, i32 0, i32 0, i32 0)",
                    "per-flavor call");
        check_equal(lw.declared("llvm.amdgcn.example.per.flavor"),
                    rocke_matrix_call_declaration(&k_per_flavor, flavor),
                    "per-flavor declare");
    }
}

/* ------------------------------------------------------------------- forms */

static void test_as_is()
{
    {
        Lowerer lw("gfx1250", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t *i32x8 = lw.vec(rocke_i32(), 8), *f32x8 = lw.vec(rocke_f32(), 8);
        MmaOp m(i32x8, i32x8, f32x8, f32x8);
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "wmma_gfx1250_f32_16x16x64_fp8_fp8", &k_fp8_wmma);
        check_equal(lw.lines().at(0),
                    "%d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32("
                    "<8 x i32> %a, <8 x i32> %b, i16 0, <8 x float> %c, i1 0, i1 0)",
                    "as-is passes the register type unchanged");
    }
    {
        Lowerer lw("gfx1250", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* fp8x32 = lw.vec(rocke_fp8e4m3(), 32);
        const rocke_type_t* f32x8 = lw.vec(rocke_f32(), 8);
        MmaOp m(fp8x32, fp8x32, f32x8, f32x8);
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "wmma_gfx1250_f32_16x16x64_fp8_fp8", &k_fp8_wmma);
        std::vector<std::string> lines = lw.lines();
        check_equal(lines.at(0), "%reg.1 = bitcast <32 x i8> %a to <8 x i32>", "fp8 a bitcast");
        check_equal(lines.at(1), "%reg.2 = bitcast <32 x i8> %b to <8 x i32>", "fp8 b bitcast");
        CHECK(lines.at(2).find("(<8 x i32> %reg.1, <8 x i32> %reg.2, i16 0,") != std::string::npos,
              "fp8 operands reach the call reinterpreted");
    }
    {
        Lowerer lw("gfx1250", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t *i32x8 = lw.vec(rocke_i32(), 8), *f32x4 = lw.vec(rocke_f32(), 4);
        MmaOp m(i32x8, i32x8, f32x4, f32x4);
        rocke_ll_emit_matrix_call(
            &lw.L, &m.op, "wmma_gfx1250_f32_16x16x64_fp8_fp8", &k_i16_accumulator);
        check_equal(lw.lines().at(0),
                    "%reg.1 = bitcast <4 x float> %c to <8 x i16>",
                    "the accumulator follows the operand rule with the catalog's c dtype");
    }
    {
        Lowerer lw("gfx1250", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* i32x8 = lw.vec(rocke_i32(), 8);
        MmaOp m(i32x8, i32x8, lw.vec(rocke_f16(), 8), lw.vec(rocke_f32(), 4));
        check_equal(error_of([&] {
                        rocke_ll_emit_matrix_call(
                            &lw.L, &m.op, "wmma_gfx1250_f32_16x16x64_fp8_fp8", &k_i16_accumulator);
                    }),
                    "MMA operand c of 'wmma_gfx1250_f32_16x16x64_fp8_fp8' holds f16 values; the "
                    "instruction takes f32 or i16 -- convert the values first",
                    "an accumulator of another number format");
    }
}

static void test_as_is_results()
{
    static const rocke_matrix_call_t k_i16_result = {
        "llvm.amdgcn.example.i16.result",
        k_abc,
        3,
        as_is("i16", 8),
        as_is("i16", 8),
        as_is("i16", 8),
        as_is("i16", 8),
        NULL,
    };
    {
        Lowerer lw("gfx1201", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* i16x8 = lw.vec(rocke_i16(), 8);
        MmaOp m(i16x8, i16x8, i16x8, lw.vec(rocke_bf16(), 8));
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "example", &k_i16_result);
        std::vector<std::string> lines = lw.lines();
        check_equal(lines.at(0),
                    "%mma_d.1 = call <8 x i16> @llvm.amdgcn.example.i16.result("
                    "<8 x i16> %a, <8 x i16> %b, <8 x i16> %c)",
                    "result call");
        check_equal(lines.at(1), "%d = bitcast <8 x i16> %mma_d.1 to <8 x bfloat>", "result back");
    }
    {
        static const rocke_matrix_call_t k_f32 = {
            "llvm.amdgcn.example",
            k_abc,
            3,
            as_is("f32", 4),
            as_is("f32", 4),
            as_is("f32", 4),
            as_is("f32", 4),
            NULL,
        };
        Lowerer lw("gfx942", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* f32x4 = lw.vec(rocke_f32(), 4);
        MmaOp m(f32x4, f32x4, f32x4, lw.vec(rocke_f32(), 8));
        check_equal(error_of([&] { rocke_ll_emit_matrix_call(&lw.L, &m.op, "example", &k_f32); }),
                    "MMA result of 'example' is vec<f32x8> (256 bits); the instruction returns "
                    "vec<f32x4> (128 bits)",
                    "a result of another bit width");
    }
}

static void test_zero_padded()
{
    {
        Lowerer lw("gfx950", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* f32x4 = lw.vec(rocke_f32(), 4);
        MmaOp m(lw.vec(rocke_i32(), 4), lw.vec(rocke_i32(), 8), f32x4, f32x4);
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "example", &k_padded);
        std::vector<std::string> lines = lw.lines();
        check_equal(lines.at(0),
                    "%mma_pad.1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, "
                    "<8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>",
                    "zero padding");
        check_equal(lines.at(1),
                    "%d = call <4 x float> @llvm.amdgcn.example.padded("
                    "<8 x i32> %mma_pad.1, <8 x i32> %b, <4 x float> %c)",
                    "padded call");
    }
    Lowerer types("gfx950", ROCKE_LLVM_FLAVOR_LLVM20);
    const rocke_type_t* refused[]
        = {types.vec(rocke_f16(), 8), types.vec(rocke_i32(), 12), rocke_i32()};
    for(const rocke_type_t* operand_b : refused)
    {
        Lowerer lw("gfx950", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* f32x4 = lw.vec(rocke_f32(), 4);
        MmaOp m(lw.vec(rocke_i32(), 8), operand_b, f32x4, f32x4);
        check_equal(error_of([&] {
                        rocke_ll_emit_matrix_call(
                            &lw.L, &m.op, "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1", &k_padded);
                    }),
                    std::string("MMA operand b of 'mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1' is ")
                        + operand_b->name
                        + "; the instruction takes a packed payload of whole 32-bit words in "
                          "<8 x i32> -- pass the operand's packed words",
                    "zero padding refuses anything but whole words that fit");
    }
}

static void test_even_halves()
{
    {
        Lowerer lw("gfx1151", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t *f16x16 = lw.vec(rocke_f16(), 16), *f16x8 = lw.vec(rocke_f16(), 8);
        MmaOp m(f16x16, f16x16, f16x8, f16x8);
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "wmma_f16_16x16x16_f16", &k_even_halves);
        std::vector<std::string> lines = lw.lines();
        check_equal(lines.at(0),
                    "%mma_spread.1 = shufflevector <8 x half> %c, <8 x half> zeroinitializer, "
                    "<16 x i32> <i32 0, i32 8, i32 1, i32 8, i32 2, i32 8, i32 3, i32 8, "
                    "i32 4, i32 8, i32 5, i32 8, i32 6, i32 8, i32 7, i32 8>",
                    "spread");
        check_equal(lines.at(1),
                    "%mma_d.2 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16("
                    "<16 x half> %a, <16 x half> %b, <16 x half> %mma_spread.1, i1 0)",
                    "even-halves call");
        check_equal(lines.at(2),
                    "%d = shufflevector <16 x half> %mma_d.2, <16 x half> poison, "
                    "<8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>",
                    "gather");
    }
    {
        Lowerer lw("gfx1151", ROCKE_LLVM_FLAVOR_LLVM20);
        const rocke_type_t* f16x16 = lw.vec(rocke_f16(), 16);
        MmaOp m(f16x16, f16x16, f16x16, lw.vec(rocke_f16(), 8));
        check_equal(error_of([&] {
                        rocke_ll_emit_matrix_call(
                            &lw.L, &m.op, "wmma_f16_16x16x16_f16", &k_even_halves);
                    }),
                    "MMA operand c of 'wmma_f16_16x16x16_f16' is vec<f16x16>; the instruction "
                    "keeps 8 results in the even halves of <16 x half> -- pass the 8 logical "
                    "accumulator values",
                    "even halves refuses an accumulator of the register width");
    }
}

/* -------------------------------------------------------------- the schema */

static void test_the_schema()
{
    static const rocke_matrix_arg_t k_abb[] = {kA, kB, kB};
    const rocke_matrix_call_t twice_b = {"llvm.amdgcn.x",
                                         k_abb,
                                         3,
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&twice_b); }),
                "MatrixCall 'llvm.amdgcn.x' must pass each of A, B and C exactly once, got "
                "['A', 'B', 'B']",
                "each operand exactly once");

    const rocke_matrix_call_t padded_result = {"llvm.amdgcn.x",
                                               k_abc,
                                               3,
                                               as_is("f32", 0),
                                               as_is("f32", 0),
                                               as_is("f32", 0),
                                               {ROCKE_MATRIX_FORM_ZERO_PADDED, reg("i32", 4)},
                                               NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&padded_result); }),
                "MatrixCall 'llvm.amdgcn.x' result form must be AsIs or EvenHalves, got "
                "ZeroPadded",
                "a result is never zero padded");

    const rocke_matrix_call_t padded_halves = {"llvm.amdgcn.x",
                                               k_abc,
                                               3,
                                               {ROCKE_MATRIX_FORM_ZERO_PADDED, reg("f16", 8)},
                                               as_is("f32", 0),
                                               as_is("f32", 0),
                                               as_is("f32", 0),
                                               NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&padded_halves); }),
                "ZeroPadded register type must be a vector of i32 words, got vec<f16x8>",
                "zero padding needs i32 words");

    const rocke_matrix_call_t halves_of_f32 = {"llvm.amdgcn.x",
                                               k_abc,
                                               3,
                                               as_is("f32", 0),
                                               as_is("f32", 0),
                                               {ROCKE_MATRIX_FORM_EVEN_HALVES, reg("f32", 8)},
                                               as_is("f32", 0),
                                               NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&halves_of_f32); }),
                "EvenHalves register type must be a vector of an even number of 16-bit "
                "elements, got vec<f32x8>",
                "even halves needs 16-bit elements");

    static const rocke_matrix_arg_t k_bad_imm[]
        = {kA, kB, kC, {ROCKE_MATRIX_ARG_IMM, "i1", 2, true}};
    const rocke_matrix_call_t bad_imm = {"llvm.amdgcn.x",
                                         k_bad_imm,
                                         4,
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         as_is("f32", 0),
                                         NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&bad_imm); }),
                "2 does not fit an i1 immediate",
                "an immediate fits its type");

    static const rocke_matrix_arg_t k_float_imm[]
        = {kA, kB, kC, {ROCKE_MATRIX_ARG_IMM, "f32", 0, true}};
    const rocke_matrix_call_t float_imm = {"llvm.amdgcn.x",
                                           k_float_imm,
                                           4,
                                           as_is("f32", 0),
                                           as_is("f32", 0),
                                           as_is("f32", 0),
                                           as_is("f32", 0),
                                           NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&float_imm); }),
                "an immediate is one of i1, i16, i32, i64, got f32",
                "an immediate is an integer");

    static const rocke_matrix_flavor_override_t k_unknown[] = {{"llvm99", {"i64", 0}}};
    const rocke_matrix_call_t unknown_flavor
        = {"llvm.amdgcn.x",
           k_abc,
           3,
           {ROCKE_MATRIX_FORM_AS_IS, {{"i32", 2}, k_unknown, 1}},
           as_is("f32", 0),
           as_is("f32", 0),
           as_is("f32", 0),
           NULL};
    check_equal(error_of([&] { rocke_matrix_call_check(&unknown_flavor); }),
                "FlavorType names unknown LLVM flavor 'llvm99'; expected one of "
                "('llvm20', 'llvm22', 'llvm23', 'llvm24')",
                "an override names a known flavor");
}

static void test_a_table_entry_must_match_the_hand_written_declaration_it_names()
{
    const rocke_matrix_call_t wrong = {
        "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
        k_abc,
        3,
        as_is("i32", 8),
        as_is("i32", 8),
        as_is("f32", 8),
        as_is("f32", 8),
        "wmma.gfx1250.f32.16x16x64.fp8.fp8",
    };
    Lowerer lw("gfx1250", ROCKE_LLVM_FLAVOR_LLVM20);
    const rocke_type_t *i32x8 = lw.vec(rocke_i32(), 8), *f32x8 = lw.vec(rocke_f32(), 8);
    MmaOp m(i32x8, i32x8, f32x8, f32x8);
    const std::string message = error_of([&] {
        rocke_ll_emit_matrix_call(&lw.L, &m.op, "wmma_gfx1250_f32_16x16x64_fp8_fp8", &wrong);
    });
    CHECK(message.rfind("matrix call for 'wmma_gfx1250_f32_16x16x64_fp8_fp8' declares "
                        "'wmma.gfx1250.f32.16x16x64.fp8.fp8' as ",
                        0)
              == 0,
          message.c_str());
}

/* ------------------------------------------------------- the table, entry by entry */

static void test_every_table_entry()
{
    int count = 0;
    const rocke_matrix_call_entry_t* calls = rocke_matrix_calls(&count);
    for(int i = 0; i < count; ++i)
    {
        const rocke_matrix_call_t* call = &calls[i].call;
        check_equal(error_of([&] { rocke_matrix_call_check(call); }), "", calls[i].op_id);
        CHECK(rocke_matrix_call_for(calls[i].op_id) == call, calls[i].op_id);
        for(rocke_llvm_flavor_t flavor : k_flavors)
            check_equal(rocke_ll_decl_text(flavor, rocke_matrix_call_key(call)),
                        rocke_matrix_call_declaration(call, flavor),
                        calls[i].op_id);
    }
}

int main()
{
    test_the_declaration_spells_each_flavors_register_type();
    test_a_derived_declaration_equals_the_hand_written_one();
    test_the_call_and_its_declaration_take_the_same_types();
    test_as_is();
    test_as_is_results();
    test_zero_padded();
    test_even_halves();
    test_the_schema();
    test_a_table_entry_must_match_the_hand_written_declaration_it_names();
    test_every_table_entry();
    if(g_failures)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("matrix_call_lowering: all checks passed\n");
    return 0;
}
