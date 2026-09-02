// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/mma_frag_ssot.cpp -- host unit test for the IR-layer MMA
 * frag-length / accumulator-dtype tables consulted by rocke_b_mma.
 *
 * rocke_b_mma sizes its tile.mma result vector as <d_frag_len x acc_elem>,
 * where d_frag_len comes from the op_id frag-length table and acc_elem is i32
 * for integer WMMA atoms (else f32). This test pins that mapping for a
 * representative set of atoms and checks the unknown-op_id error path, so a
 * table edit that changes a result width/dtype is caught here.
 *
 * Plain executable: returns non-zero on the first failed check (a clean run is
 * the pass criterion). Registered via tests/CMakeLists.txt so it is installed
 * into the provider test artifact and run under ctest by TheRock CI.
 */
#include <cstdio>
#include <cstring>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/ir.h"

/* Private universal-GEMM helper under test. Keep the core test independent of
 * the large instance_gemm_internal.h closure-state surface. */
extern "C" rocke_value_t* rocke_gemm_emit_zero_acc_op(rocke_ir_builder_t* b,
                                                      const rocke_mma_op_t* op);

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

/* Emit rocke_b_mma(op_id) and assert the result is a vec<expect_elem x
 * expect_frag>. expect_int selects i32 vs f32 for the accumulator element. */
static void check_atom(rocke_ir_builder_t* b, const char* op_id, int expect_frag, bool expect_int)
{
    rocke_value_t* a = rocke_b_const_i32(b, 0);
    rocke_value_t* bb = rocke_b_const_i32(b, 0);
    rocke_value_t* c = rocke_b_const_i32(b, 0);
    rocke_value_t* r = rocke_b_mma(b, op_id, a, bb, c, NULL, 0);

    CHECK(r != NULL, op_id);
    if(!r)
    {
        return;
    }
    CHECK(r->type != NULL && r->type->kind == ROCKE_TYPE_VECTOR, op_id);
    if(!r->type || r->type->kind != ROCKE_TYPE_VECTOR)
    {
        return;
    }
    CHECK(r->type->count == expect_frag, op_id);
    CHECK(r->type->elem != NULL, op_id);
    if(r->type->elem)
    {
        rocke_scalar_kind_t want = expect_int ? ROCKE_SCALAR_I32 : ROCKE_SCALAR_F32;
        CHECK(r->type->elem->scalar == want, op_id);
    }
}

int main(void)
{
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx("gfx942");
    CHECK(target != NULL, "gfx942 target exists");
    if(target)
    {
        const rocke_mma_op_t* op
            = rocke_mma_catalog_by_op_id(&target->mma, "mfma_f32_16x16x16_f16");
        CHECK(op != NULL, "four-role catalog atom exists");
        if(op)
        {
            CHECK(strcmp(op->c_dtype, "fp32") == 0, "C input dtype is explicit");
            CHECK(strcmp(op->d_dtype, "fp32") == 0, "D result dtype is explicit");
            CHECK(op->c_frag_len == 4, "C input fragment length is explicit");
            CHECK(op->d_frag_len == 4, "D result fragment length is explicit");
            CHECK(op->c_layout != op->d_layout, "C and D layouts are distinct objects");
            CHECK(op->c_layout && op->c_layout->role == ROCKE_MMA_ROLE_C,
                  "C layout carries the C role");
            CHECK(op->d_layout && op->d_layout->role == ROCKE_MMA_ROLE_D,
                  "D layout carries the D role");
        }
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "rocke_mma_frag_ssot") != ROCKE_OK)
    {
        fprintf(stderr, "rocke_ir_builder_init failed\n");
        return 1;
    }

    /* MFMA float accumulators: 16x16 -> 4, 32x32 -> 16 (f32). */
    check_atom(&b, "mfma_f32_16x16x16_f16", 4, false);
    check_atom(&b, "mfma_f32_32x32x8_f16", 16, false);
    check_atom(&b, "mfma_f32_16x16x32_bf16", 4, false);
    /* WMMA float accumulator: 8 (f32). */
    check_atom(&b, "wmma_f32_16x16x16_f16", 8, false);
    /* Integer WMMA: 8-wide i32 accumulator. */
    check_atom(&b, "wmma_i32_16x16x16_iu8", 8, true);
    check_atom(&b, "wmma_i32_16x16x16_iu4", 8, true);

    /* Initial accumulator construction is a C-role operation. A synthetic
     * unequal contract proves that zero_acc does not borrow D's dtype/width,
     * and that recurrent helpers reject D -> C rather than emitting an invalid
     * loop-carried value. */
    {
        const rocke_mfma_atom_t unequal
            = {1, 1, 1, 1, 1, 2, 3, "f16", "i32", "f32", "synthetic_mfma"};
        rocke_value_t* zero = rocke_mfma_atom_zero_acc(&b, &unequal);
        CHECK(zero != NULL, "C-shaped synthetic zero exists");
        if(zero && zero->type)
        {
            CHECK(zero->type->kind == ROCKE_TYPE_VECTOR, "synthetic zero is a vector");
            CHECK(zero->type->count == 2, "synthetic zero uses c_per_lane");
            CHECK(zero->type->elem && zero->type->elem->scalar == ROCKE_SCALAR_I32,
                  "synthetic zero uses dtype_c");
        }
    }
    {
        rocke_mma_op_t equal = {};
        equal.c_dtype = "i32";
        equal.d_dtype = "i32";
        equal.c_frag_len = 2;
        equal.d_frag_len = 2;
        rocke_value_t* zero = rocke_gemm_emit_zero_acc_op(&b, &equal);
        CHECK(zero != NULL, "universal GEMM C-shaped zero exists");
        if(zero && zero->type)
        {
            CHECK(zero->type->count == 2, "universal GEMM zero uses c_frag_len");
            CHECK(zero->type->elem && zero->type->elem->scalar == ROCKE_SCALAR_I32,
                  "universal GEMM zero uses c_dtype");
        }
    }

    /* Unknown op_id must be rejected. The engine's error path either returns
     * NULL with a sticky builder error or raises (ckc::ValueError) depending on
     * build config, so accept either form of rejection. */
    bool rejected = false;
    try
    {
        rocke_value_t* a = rocke_b_const_i32(&b, 0);
        rocke_value_t* bad = rocke_b_mma(&b, "not_a_real_op_id", a, a, a, NULL, 0);
        rejected = (bad == NULL);
    }
    catch(...)
    {
        rejected = true;
    }
    CHECK(rejected, "unknown op_id must be rejected");

    rocke_ir_builder_free(&b);

    {
        const rocke_mfma_atom_t unequal
            = {1, 1, 1, 1, 1, 2, 3, "f16", "i32", "f32", "synthetic_mfma"};
        bool rejected = false;
        if(rocke_ir_builder_init(&b, "rocke_mma_recurrence_reject") != ROCKE_OK)
        {
            fprintf(stderr, "rocke_ir_builder_init failed\n");
            return 1;
        }
        try
        {
            rejected = rocke_mfma_atom_require_recurrence(&b, &unequal, "test") == ROCKE_ERR_VALUE;
        }
        catch(...)
        {
            rejected = true;
        }
        CHECK(rejected, "unequal C/D recurrence must be rejected");
        rocke_ir_builder_free(&b);
    }

    {
        rocke_mma_op_t unequal = {};
        unequal.c_dtype = "i32";
        unequal.d_dtype = "fp32";
        unequal.c_frag_len = 2;
        unequal.d_frag_len = 3;
        bool rejected = false;
        if(rocke_ir_builder_init(&b, "rocke_gemm_recurrence_reject") != ROCKE_OK)
        {
            fprintf(stderr, "rocke_ir_builder_init failed\n");
            return 1;
        }
        try
        {
            rejected = rocke_gemm_emit_zero_acc_op(&b, &unequal) == NULL;
        }
        catch(...)
        {
            rejected = true;
        }
        CHECK(rejected, "universal GEMM unequal C/D recurrence must be rejected");
        rocke_ir_builder_free(&b);
    }

    {
        const rocke_mfma_atom_t equal
            = {1, 1, 1, 1, 1, 2, 2, "f16", "f32", "f32", "synthetic_mfma"};
        if(rocke_ir_builder_init(&b, "rocke_mma_recurrence_accept") != ROCKE_OK)
        {
            fprintf(stderr, "rocke_ir_builder_init failed\n");
            return 1;
        }
        CHECK(rocke_mfma_atom_require_recurrence(&b, &equal, "test") == ROCKE_OK,
              "equal C/D recurrence must be accepted");
        rocke_ir_builder_free(&b);
    }

    if(g_failures)
    {
        fprintf(stderr, "rocke_mma_frag_ssot: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_mma_frag_ssot: all checks passed.\n");
    return 0;
}
