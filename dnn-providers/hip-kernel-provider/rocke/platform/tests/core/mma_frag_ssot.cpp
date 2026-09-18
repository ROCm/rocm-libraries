// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/mma_frag_ssot.cpp -- host unit test for the IR-layer MMA
 * frag-length / accumulator-dtype tables consulted by rocke_b_mma.
 *
 * rocke_b_mma sizes its tile.mma result vector as <dst_frag_len x acc_elem>,
 * where dst_frag_len comes from the op_id frag-length table and acc_elem is i32
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
#include <initializer_list>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
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

static void check_catalog_fragments(const rocke_mma_catalog_t* catalog,
                                    const char* op_id,
                                    int scale_block_size)
{
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(catalog, op_id);
    CHECK(op != NULL, op_id);
    if(!op)
    {
        return;
    }
    CHECK(op->srcs[0].frag_len == 16, op_id);
    CHECK(op->srcs[1].frag_len == 16, op_id);
    CHECK(op->srcs[2].frag_len == 8, op_id);
    CHECK(op->dst.frag_len == 8, op_id);
    for(int i = 0; i < 2; ++i)
    {
        CHECK(strcmp(op->srcs[i].dtype, "fp8e4m3") == 0, op_id);
        CHECK(op->srcs[i].scale_dtype != NULL, op_id);
        if(op->srcs[i].scale_dtype)
        {
            CHECK(strcmp(op->srcs[i].scale_dtype, "e8m0") == 0, op_id);
        }
        CHECK(op->srcs[i].scale_block_size == scale_block_size, op_id);
    }
    CHECK(op->srcs[2].scale_dtype == NULL && op->srcs[2].scale_block_size == 0, op_id);
}

static void check_contract_queries()
{
    rocke_mma_op_t rows[10] = {};
    rows[0].family = "wmma_scaled";
    rows[0].srcs[0] = {"fp8e4m3", 16, NULL, "e8m0", 32};
    rows[0].srcs[1] = {"bf8e5m2", 16, NULL, "e8m0", 32};
    rows[0].srcs[2] = {"fp32", 8, NULL, NULL, 0};
    rows[0].dst = {"fp32", 8, NULL};
    rows[0].m = rows[0].n = 16;
    rows[0].k = 128;
    for(int i = 1; i < 10; ++i)
        rows[i] = rows[0];
    for(int src = 0; src < 2; ++src)
    {
        rows[1 + src * 3].srcs[src].scale_dtype = "e4m3";
        rows[2 + src * 3].srcs[src].scale_block_size = 16;
        rows[3 + src * 3].srcs[src].scale_dtype = NULL;
        rows[3 + src * 3].srcs[src].scale_block_size = 0;
    }
    rows[7].srcs[2].dtype = "fp16";
    rows[8].dst.dtype = "fp16";
    for(int i = 0; i < 2; ++i)
    {
        rows[9].srcs[i].scale_dtype = NULL;
        rows[9].srcs[i].scale_block_size = 0;
    }
    const rocke_mma_catalog_t catalog = {rows, 10};
    char ids[10][256];
    for(int i = 0; i < 10; ++i)
    {
        rocke_mma_op_t* row = &rows[i];
        CHECK(rocke_mma_op_semantic_id(row, ids[i], sizeof(ids[i])) != NULL, "format contract ID");
        row->op_id = ids[i];
        for(int j = 0; j < i; ++j)
            CHECK(strcmp(ids[i], ids[j]) != 0, "different contracts have different IDs");
        const char* dtypes[3] = {"fp8", "bf8", row->srcs[2].dtype};
        rocke_mma_scale_operand_t scales[3];
        for(int j = 0; j < 3; ++j)
            scales[j] = {row->srcs[j].scale_dtype, row->srcs[j].scale_block_size};
        // Exercise the scale-format alias through every query operation.
        if(i == 1)
            scales[0].dtype = "fp8e4m3";
        const rocke_mma_op_t* out[10];
        CHECK(rocke_mma_catalog_enumerate_indexed(
                  &catalog, row->family, dtypes, row->dst.dtype, scales, 16, 16, out, 10)
                      == 1
                  && out[0] == row,
              "enumeration distinguishes every operand contract");
        CHECK(rocke_mma_catalog_has_shape_indexed(
                  &catalog, row->family, dtypes, row->dst.dtype, scales, 16, 16, 128),
              "full contract exists");
        CHECK(rocke_mma_catalog_op_for_shape_indexed(
                  &catalog, row->family, dtypes, row->dst.dtype, scales, 16, 16, 128)
                  == row,
              "exact contract selection");
        CHECK(rocke_mma_catalog_select_largest_k_indexed(
                  &catalog, row->family, dtypes, row->dst.dtype, scales, 16, 16, -1)
                  == row,
              "largest K contract selection");
    }
    const char* dtypes[3] = {"fp8", "bf8", "fp32"};
    CHECK(rocke_mma_catalog_enumerate_indexed(
              &catalog, "wmma_scaled", dtypes, "fp32", NULL, 16, 16, NULL, 0)
              == 8,
          "omitted scales are wildcard");
    for(bool largest : {false, true})
    {
        bool rejected = false;
        try
        {
            if(largest)
                rocke_mma_catalog_select_largest_k_indexed(
                    &catalog, "wmma_scaled", dtypes, "fp32", NULL, 16, 16, -1);
            else
                rocke_mma_catalog_op_for_shape_indexed(
                    &catalog, "wmma_scaled", dtypes, "fp32", NULL, 16, 16, 128);
        }
        catch(const ckc::Error&)
        {
            rejected = true;
        }
        CHECK(rejected, "underspecified selection rejects ambiguity");
    }
    rocke_mma_scale_operand_t scales[3] = {{"e5m3", 32}, {"e5m3", 32}, {NULL, 0}};
    CHECK(rocke_mma_catalog_op_for_shape_indexed(
              &catalog, "wmma_scaled", dtypes, "fp32", scales, 16, 16, 128)
              == NULL,
          "unsupported contract absent");
}

int main(void)
{
    check_contract_queries();
    const char* cpu_targets[] = {"gfx950", "gfx1250"};
    for(const char* gfx : cpu_targets)
    {
        const rocke_arch_target_t* target = rocke_arch_target_from_gfx(gfx);
        CHECK(target != NULL, "CPU catalog target exists");
        if(!target)
        {
            continue;
        }
        for(int i = 0; i < target->mma.num_ops; ++i)
        {
            const rocke_mma_op_t* op = &target->mma.ops[i];
            const char* src_dtypes[3] = {op->srcs[0].dtype, op->srcs[1].dtype, op->srcs[2].dtype};
            rocke_mma_scale_operand_t scales[3];
            for(int j = 0; j < 3; ++j)
                scales[j] = {op->srcs[j].scale_dtype, op->srcs[j].scale_block_size};
            if(strcmp(op->family, "wmma_scaled") == 0)
            {
                char id[256];
                CHECK(rocke_mma_op_semantic_id(op, id, sizeof(id)) != NULL, "semantic ID fits");
                CHECK(strcmp(id, op->op_id) == 0, "semantic ID agrees with catalog");
            }
            CHECK(rocke_mma_catalog_op_for_shape_indexed(&target->mma,
                                                         op->family,
                                                         src_dtypes,
                                                         op->dst.dtype,
                                                         scales,
                                                         op->m,
                                                         op->n,
                                                         op->k)
                      != NULL,
                  "indexed query resolves CPU catalog key");
        }
    }

    {
        rocke_mma_op_t distinct = {};
        const rocke_layout_map_t src2_layout = {ROCKE_MMA_ROLE_SRC2, 3, 32, NULL};
        const rocke_layout_map_t dst_layout = {ROCKE_MMA_ROLE_DST, 7, 32, NULL};
        distinct.family = "mma";
        distinct.srcs[0].dtype = "xf32";
        distinct.srcs[1].dtype = "xf32";
        distinct.srcs[0].scale_dtype = "e8m0";
        distinct.srcs[0].scale_block_size = 32;
        distinct.srcs[1].scale_dtype = "e4m3";
        distinct.srcs[1].scale_block_size = 16;
        distinct.srcs[2].dtype = "fp32";
        distinct.srcs[2].frag_len = 3;
        distinct.srcs[2].layout = &src2_layout;
        distinct.dst.dtype = "i32";
        distinct.dst.frag_len = 7;
        distinct.dst.layout = &dst_layout;
        distinct.m = 16;
        distinct.n = 16;
        distinct.k = 8;
        distinct.op_id = "synthetic_distinct_dst";
        const rocke_mma_catalog_t catalog = {&distinct, 1};
        const char* src_dtypes[3] = {"xf32", "xf32", "fp32"};
        CHECK(rocke_mma_catalog_op_for_shape_indexed(
                  &catalog, "mma", src_dtypes, "i32", NULL, 16, 16, 8)
                  == &distinct,
              "indexed query distinguishes src2 and dst");
        CHECK(rocke_mma_catalog_op_for_shape(&catalog, "mma", "xf32", "xf32", "fp32", 16, 16, 8)
                  == NULL,
              "legacy query defaults dst to src2");
        CHECK(rocke_mma_op_src_layout(&distinct, 2, NULL) == &src2_layout,
              "indexed src2 layout remains independent");
        CHECK(rocke_mma_op_dst_layout(&distinct, NULL) == &dst_layout,
              "dst layout remains independent");
        CHECK(rocke_mma_op_c_layout(&distinct, NULL) == &dst_layout,
              "historical C layout projects dst");
        const rocke_mma_op_t* selected
            = rocke_mma_catalog_by_op_id(&catalog, "synthetic_distinct_dst");
        CHECK(selected != NULL, "catalog preserves scaled operand metadata");
        if(selected)
        {
            CHECK(strcmp(selected->srcs[0].scale_dtype, "e8m0") == 0
                      && selected->srcs[0].scale_block_size == 32,
                  "src0 scale retains its value format and block size");
            CHECK(strcmp(selected->srcs[1].scale_dtype, "e4m3") == 0
                      && selected->srcs[1].scale_block_size == 16,
                  "src1 scale is independent of src0");
            CHECK(selected->srcs[2].scale_dtype == NULL && selected->srcs[2].scale_block_size == 0,
                  "src2 remains unscaled");
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

    /* Keep the C catalog in parity with Python _MMA_FRAGMENT_INFO. The scaled
     * matrix operands are <16 x i32>, so fragment length is 16 elements, not
     * the equivalent 64-byte storage width. */
    const rocke_arch_target_t* gfx1250 = rocke_arch_target_from_gfx("gfx1250");
    CHECK(gfx1250 != NULL, "gfx1250 target");
    if(gfx1250)
    {
        check_catalog_fragments(
            &gfx1250->mma,
            "wmma.scaled.16x16x128.src0_fp8e4m3_e8m0_b32.src1_fp8e4m3_e8m0_b32.src2_fp32.dst_fp32",
            32);
        check_catalog_fragments(
            &gfx1250->mma,
            "wmma.scaled.16x16x128.src0_fp8e4m3_e8m0_b16.src1_fp8e4m3_e8m0_b16.src2_fp32.dst_fp32",
            16);
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
