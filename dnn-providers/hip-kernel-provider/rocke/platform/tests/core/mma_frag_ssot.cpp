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

#include "rocke/arch_target.h"
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

/* Emit rocke_b_mma(op_id) on the given operands and assert the result is a
 * vec<expect_elem x expect_frag>. expect_int selects i32 vs f32 for the
 * accumulator element. */
static void check_atom_on(rocke_ir_builder_t* b,
                          const char* op_id,
                          rocke_value_t* a,
                          rocke_value_t* bb,
                          rocke_value_t* c,
                          int expect_frag,
                          bool expect_int)
{
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

/* Placeholder scalar operands: this check is about the result's fragment, not the
 * operands, which an ordinary atom's builder does not inspect. */
static void check_atom(rocke_ir_builder_t* b, const char* op_id, int expect_frag, bool expect_int)
{
    rocke_value_t* a = rocke_b_const_i32(b, 0);
    rocke_value_t* bb = rocke_b_const_i32(b, 0);
    rocke_value_t* c = rocke_b_const_i32(b, 0);
    check_atom_on(b, op_id, a, bb, c, expect_frag, expect_int);
}

/* An emulated TF32 atom is verified on the way in: its operands are vec<tf32xN>
 * (an fp32 payload wrapped by a bitcast) and its accumulator a vec<f32xM>. N and M
 * are the physical atom's fragment lengths, which the atom borrows. */
static void check_emulated_atom(rocke_ir_builder_t* b, const char* op_id, int ab_len, int c_len)
{
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    const rocke_type_t* tf32_vec = rocke_vector_type(b, rocke_tf32(), ab_len);
    rocke_value_t* a = rocke_b_bitcast(b, rocke_b_vector_splat(b, zero, ab_len), tf32_vec);
    rocke_value_t* bb = rocke_b_bitcast(b, rocke_b_vector_splat(b, zero, ab_len), tf32_vec);
    rocke_value_t* c = rocke_b_vector_splat(b, zero, c_len);
    check_atom_on(b, op_id, a, bb, c, c_len, false);
}

static void check_catalog_fragments(const rocke_mma_catalog_t* catalog, const char* op_id)
{
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(catalog, op_id);
    CHECK(op != NULL, op_id);
    if(!op)
    {
        return;
    }
    CHECK(op->a_frag_len == 16, op_id);
    CHECK(op->b_frag_len == 16, op_id);
    CHECK(op->c_frag_len == 8, op_id);
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
    /* Emulated TF32 atoms have no frag-table row of their own; the logical op_id
     * sizes through the bf16 atom it expands into, so rocke_b_mma can build it. */
    check_emulated_atom(&b, "mfma_f32_16x16x32_xf32", 8, 4);
    check_emulated_atom(&b, "mfma_f32_32x32x16_xf32", 8, 16);

    /* Keep the C catalog in parity with Python _MMA_FRAGMENT_INFO. The scaled
     * matrix operands are <16 x i32>, so fragment length is 16 elements, not
     * the equivalent 64-byte storage width. */
    const rocke_arch_target_t* gfx1250 = rocke_arch_target_from_gfx("gfx1250");
    CHECK(gfx1250 != NULL, "gfx1250 target");
    if(gfx1250)
    {
        int scaled = 0;
        for(int i = 0; i < gfx1250->mma.num_ops; ++i)
        {
            const auto& op = gfx1250->mma.ops[i];
            if(strcmp(op.family, "wmma_scaled") != 0)
                continue;
            check_catalog_fragments(&gfx1250->mma, op.op_id);
            check_atom(&b, op.op_id, 8, false);
            ++scaled;
        }
        CHECK(scaled == 86, "all scaled atoms must have accumulator metadata");
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

    if(g_failures)
    {
        fprintf(stderr, "rocke_mma_frag_ssot: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_mma_frag_ssot: all checks passed.\n");
    return 0;
}
