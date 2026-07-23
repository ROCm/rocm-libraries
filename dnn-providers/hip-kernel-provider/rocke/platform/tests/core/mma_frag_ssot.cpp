// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/mma_frag_ssot.cpp -- host unit test for the IR-layer MMA
 * frag-length / accumulator-dtype tables consulted by rocke_b_mma.
 *
 * rocke_b_mma sizes its tile.mma result vector as <c_frag_len x acc_elem>,
 * where c_frag_len comes from the op_id frag-length table and acc_elem is i32
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

#include "rocke/ir.h"

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

    if(g_failures)
    {
        fprintf(stderr, "rocke_mma_frag_ssot: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_mma_frag_ssot: all checks passed.\n");
    return 0;
}
