// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/cdna_mfma_ops.cpp -- the C++ engine's catalog rows and lane maps
 * for the CDNA MFMA instructions each target gained beyond f16 / bf16 / fp8 /
 * bf8: i8 and the bf16 MFMAs without the _1k suffix on gfx908; f32, i8, f64
 * and the same bf16 MFMAs on gfx90a; i8 and mixed fp8 x bf8 / bf8 x fp8 on
 * gfx942 and gfx950, plus gfx950's f64 and wider i8 forms (gfx942 already had
 * f64; its row is pinned with the others).
 *
 * For each one: the catalog row (dtypes, shape, fragment lengths), the
 * tile.mma result rocke_b_mma builds from the bare op_id, and the lane maps,
 * evaluated on every (lane, slot) of a wave: A and B follow the dense MFMA
 * closed form (lane l holds row l % M of A and the K run starting at
 * (l / M) * (K per lane)), and the accumulator places every element once (the
 * f64 one with rows four apart). Mirrors tests/core/test_cdna_mfma_ops.py.
 *
 * Plain executable: returns non-zero when a check fails.
 */
#include <cstdio>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/ir.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                                 \
    do                                                                                    \
    {                                                                                     \
        if(!(cond))                                                                       \
        {                                                                                 \
            fprintf(stderr, "FAIL: %s: %s (%s:%d)\n", (what), #cond, __FILE__, __LINE__); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while(0)

struct Case
{
    const char* gfx;
    const char* op_id;
    const char* a;
    const char* b;
    const char* c;
    int m, n, k;
    int a_frag, b_frag, c_frag;
};

static const Case k_cases[] = {
    {"gfx908", "mfma_i32_16x16x16_i8", "i8", "i8", "i32", 16, 16, 16, 4, 4, 4},
    {"gfx908", "mfma_i32_32x32x8_i8", "i8", "i8", "i32", 32, 32, 8, 4, 4, 16},
    {"gfx908", "mfma_f32_16x16x8_bf16", "bf16", "bf16", "fp32", 16, 16, 8, 2, 2, 4},
    {"gfx908", "mfma_f32_32x32x4_bf16", "bf16", "bf16", "fp32", 32, 32, 4, 2, 2, 16},
    {"gfx90a", "mfma_f32_16x16x4_f32", "fp32", "fp32", "fp32", 16, 16, 4, 1, 1, 4},
    {"gfx90a", "mfma_f32_32x32x2_f32", "fp32", "fp32", "fp32", 32, 32, 2, 1, 1, 16},
    {"gfx90a", "mfma_i32_16x16x16_i8", "i8", "i8", "i32", 16, 16, 16, 4, 4, 4},
    {"gfx90a", "mfma_i32_32x32x8_i8", "i8", "i8", "i32", 32, 32, 8, 4, 4, 16},
    {"gfx90a", "mfma_f64_16x16x4_f64", "fp64", "fp64", "fp64", 16, 16, 4, 1, 1, 4},
    {"gfx90a", "mfma_f32_16x16x8_bf16", "bf16", "bf16", "fp32", 16, 16, 8, 2, 2, 4},
    {"gfx90a", "mfma_f32_32x32x4_bf16", "bf16", "bf16", "fp32", 32, 32, 4, 2, 2, 16},
    {"gfx942", "mfma_i32_16x16x32_i8", "i8", "i8", "i32", 16, 16, 32, 8, 8, 4},
    {"gfx942", "mfma_i32_32x32x16_i8", "i8", "i8", "i32", 32, 32, 16, 8, 8, 16},
    {"gfx942", "mfma_f64_16x16x4_f64", "fp64", "fp64", "fp64", 16, 16, 4, 1, 1, 4},
    {"gfx942", "mfma_f32_16x16x32_fp8_bf8", "fp8e4m3", "bf8e5m2", "fp32", 16, 16, 32, 8, 8, 4},
    {"gfx942", "mfma_f32_16x16x32_bf8_fp8", "bf8e5m2", "fp8e4m3", "fp32", 16, 16, 32, 8, 8, 4},
    {"gfx942", "mfma_f32_32x32x16_fp8_bf8", "fp8e4m3", "bf8e5m2", "fp32", 32, 32, 16, 8, 8, 16},
    {"gfx942", "mfma_f32_32x32x16_bf8_fp8", "bf8e5m2", "fp8e4m3", "fp32", 32, 32, 16, 8, 8, 16},
    {"gfx950", "mfma_i32_16x16x32_i8", "i8", "i8", "i32", 16, 16, 32, 8, 8, 4},
    {"gfx950", "mfma_i32_32x32x16_i8", "i8", "i8", "i32", 32, 32, 16, 8, 8, 16},
    {"gfx950", "mfma_f64_16x16x4_f64", "fp64", "fp64", "fp64", 16, 16, 4, 1, 1, 4},
    {"gfx950", "mfma_f32_16x16x32_fp8_bf8", "fp8e4m3", "bf8e5m2", "fp32", 16, 16, 32, 8, 8, 4},
    {"gfx950", "mfma_f32_16x16x32_bf8_fp8", "bf8e5m2", "fp8e4m3", "fp32", 16, 16, 32, 8, 8, 4},
    {"gfx950", "mfma_f32_32x32x16_fp8_bf8", "fp8e4m3", "bf8e5m2", "fp32", 32, 32, 16, 8, 8, 16},
    {"gfx950", "mfma_f32_32x32x16_bf8_fp8", "bf8e5m2", "fp8e4m3", "fp32", 32, 32, 16, 8, 8, 16},
    {"gfx950", "mfma_i32_16x16x64_i8", "i8", "i8", "i32", 16, 16, 64, 16, 16, 4},
    {"gfx950", "mfma_i32_32x32x32_i8", "i8", "i8", "i32", 32, 32, 32, 16, 16, 16},
};

/* The integer a lane map's arithmetic gives when `lane` is `lane_value`: the
 * constants, add, mul, div and mod the maps emit, evaluated. */
static long long evaluate(const rocke_value_t* v, const rocke_value_t* lane, long long lane_value)
{
    if(v == lane)
        return lane_value;
    const rocke_op_t* op = v->op;
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        int64_t value = 0;
        rocke_attr_get_int(&op->attrs, "value", &value);
        return value;
    }
    const long long x = evaluate(op->operands[0], lane, lane_value);
    const long long y = evaluate(op->operands[1], lane, lane_value);
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        return x + y;
    case ROCKE_OP_ARITH_MUL:
        return x * y;
    case ROCKE_OP_ARITH_DIV:
        return x / y;
    case ROCKE_OP_ARITH_MOD:
        return x % y;
    default:
        fprintf(stderr, "unexpected op %s in a lane map\n", op->name);
        ++g_failures;
    }
    return -1;
}

/* Every (lane, slot) -> coordinate pair of `map` over a 64-lane wave, in order. */
static std::vector<std::pair<long long, long long>> placements(const rocke_layout_map_t* map)
{
    std::vector<std::pair<long long, long long>> out;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "cdna_lane_map") != ROCKE_OK)
        return out;
    rocke_value_t* lane = rocke_b_thread_id_x(&b);
    for(int l = 0; l < 64; ++l)
    {
        for(int slot = 0; slot < map->frag_len; ++slot)
        {
            rocke_value_t *x = NULL, *y = NULL;
            if(!rocke_layout_map_coord(map, &b, lane, slot, &x, &y))
                continue;
            out.emplace_back(evaluate(x, lane, l), evaluate(y, lane, l));
        }
    }
    rocke_ir_builder_free(&b);
    return out;
}

/* Whether `got` places each of the rows x cols coordinates exactly once. */
static bool covers_once(const std::vector<std::pair<long long, long long>>& got, int rows, int cols)
{
    std::set<std::pair<long long, long long>> seen(got.begin(), got.end());
    if((int)got.size() != rows * cols || seen.size() != got.size())
        return false;
    for(const auto& p : seen)
    {
        if(p.first < 0 || p.first >= rows || p.second < 0 || p.second >= cols)
            return false;
    }
    return true;
}

static void check_case(const Case& t)
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(t.gfx);
    CHECK(arch != NULL, t.gfx);
    if(!arch)
        return;
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, t.op_id);
    CHECK(op != NULL, t.op_id);
    if(!op)
        return;
    CHECK(strcmp(op->family, "mma") == 0, t.op_id);
    CHECK(strcmp(op->a_dtype, t.a) == 0 && strcmp(op->b_dtype, t.b) == 0
              && strcmp(op->c_dtype, t.c) == 0,
          t.op_id);
    CHECK(op->m == t.m && op->n == t.n && op->k == t.k, t.op_id);
    CHECK(op->a_frag_len == t.a_frag && op->b_frag_len == t.b_frag && op->c_frag_len == t.c_frag
              && op->wave_size == 64,
          t.op_id);

    /* rocke_b_mma sizes the result from the bare op_id: <c_frag x c>. */
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "cdna_result") == ROCKE_OK, t.op_id);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_value_t* d = rocke_b_mma(&b, t.op_id, zero, zero, zero, NULL, 0);
        CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == t.c_frag, t.op_id);
        if(d)
            CHECK(d->type->elem == rocke_dtype_to_ir_type(t.c), t.op_id);
        rocke_ir_builder_free(&b);
    }

    const auto a = placements(rocke_mma_op_a_layout(op, NULL));
    const auto b = placements(rocke_mma_op_b_layout(op, NULL));
    const auto c = placements(rocke_mma_op_c_layout(op, NULL));
    CHECK(covers_once(a, t.m, t.k), t.op_id);
    CHECK(covers_once(c, t.m, t.n), t.op_id);
    bool closed_form = (int)a.size() == 64 * t.a_frag && b.size() == a.size();
    for(int l = 0; closed_form && l < 64; ++l)
    {
        for(int slot = 0; slot < t.a_frag; ++slot)
        {
            const auto& got_a = a[l * t.a_frag + slot];
            const auto& got_b = b[l * t.a_frag + slot];
            const long long row = l % t.m, kk = (l / t.m) * t.a_frag + slot;
            closed_form = closed_form && got_a == std::make_pair(row, kk)
                          && got_b == std::make_pair(kk, row);
        }
    }
    CHECK(closed_form, t.op_id);
    if(strcmp(t.c, "fp64") == 0)
    {
        bool rows_four_apart = (int)c.size() == 64 * 4;
        for(int l = 0; rows_four_apart && l < 64; ++l)
        {
            for(int slot = 0; slot < 4; ++slot)
                rows_four_apart
                    = rows_four_apart
                      && c[l * 4 + slot]
                             == std::make_pair((long long)(4 * slot + l / 16), (long long)(l % 16));
        }
        CHECK(rows_four_apart, t.op_id);
    }
}

int main(void)
{
    for(const Case& t : k_cases)
        check_case(t);
    if(g_failures)
    {
        fprintf(stderr, "rocke_cdna_mfma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_cdna_mfma_ops: all checks passed.\n");
    return 0;
}
