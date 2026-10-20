// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/gfx12_wmma_ops.cpp -- the C++ engine's catalog rows and lane maps
 * for the gfx12 WMMA instructions beyond f16 / bf16 -> f32: fp8 / bf8 -> f32,
 * i8 -> i32, f16 -> f16 and bf16 -> bf16, on gfx12-generic and on gfx1200 and
 * gfx1201, which take them from the generic row.
 *
 * For each one: the catalog row (dtypes, shape, fragment lengths), the
 * tile.mma result rocke_b_mma builds from the bare op_id, and the lane maps,
 * evaluated on every (lane, slot) of a wave: lane l holds row l % 16 of A
 * (column l % 16 of B) and the K run starting at (l / 16) * 8, and result slot
 * i of lane l is (row (l / 16) * 8 + i, column l % 16). Mirrors
 * tests/core/test_gfx12_wmma_ops.py.
 *
 * Plain executable: returns non-zero when a check fails.
 */
#include <cstdio>
#include <cstring>
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
    const char* op_id;
    const char* a;
    const char* b;
    const char* c;
};

static const Case k_cases[] = {
    {"wmma_gfx12_f32_16x16x16_fp8_fp8", "fp8e4m3", "fp8e4m3", "fp32"},
    {"wmma_gfx12_f32_16x16x16_fp8_bf8", "fp8e4m3", "bf8e5m2", "fp32"},
    {"wmma_gfx12_f32_16x16x16_bf8_fp8", "bf8e5m2", "fp8e4m3", "fp32"},
    {"wmma_gfx12_f32_16x16x16_bf8_bf8", "bf8e5m2", "bf8e5m2", "fp32"},
    {"wmma_gfx12_i32_16x16x16_iu8", "i8", "i8", "i32"},
    {"wmma_gfx12_f16_16x16x16_f16", "fp16", "fp16", "fp16"},
    {"wmma_gfx12_bf16_16x16x16_bf16", "bf16", "bf16", "bf16"},
};

/* The rows that hold the instructions: the generic row and the processor that
 * points at it. */
static const char* const k_targets[] = {"gfx12-generic", "gfx1200", "gfx1201"};

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

/* Every (lane, slot) -> coordinate pair of `map` over a 32-lane wave, in order. */
static std::vector<std::pair<long long, long long>> placements(const rocke_layout_map_t* map)
{
    std::vector<std::pair<long long, long long>> out;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "gfx12_lane_map") != ROCKE_OK)
        return out;
    rocke_value_t* lane = rocke_b_thread_id_x(&b);
    for(int l = 0; l < 32; ++l)
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

static void check_case(const char* gfx, const Case& t)
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
    CHECK(arch != NULL, gfx);
    if(!arch)
        return;
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, t.op_id);
    CHECK(op != NULL, t.op_id);
    if(!op)
        return;
    CHECK(strcmp(op->family, "wmma") == 0, t.op_id);
    CHECK(strcmp(op->a_dtype, t.a) == 0 && strcmp(op->b_dtype, t.b) == 0
              && strcmp(op->c_dtype, t.c) == 0,
          t.op_id);
    CHECK(op->m == 16 && op->n == 16 && op->k == 16, t.op_id);
    CHECK(op->a_frag_len == 8 && op->b_frag_len == 8 && op->c_frag_len == 8 && op->wave_size == 32,
          t.op_id);

    /* rocke_b_mma sizes the result from the bare op_id: <8 x c>. */
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "gfx12_result") == ROCKE_OK, t.op_id);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_value_t* d = rocke_b_mma(&b, t.op_id, zero, zero, zero, NULL, 0);
        CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == 8, t.op_id);
        if(d)
            CHECK(d->type->elem == rocke_dtype_to_ir_type(t.c), t.op_id);
        rocke_ir_builder_free(&b);
    }

    const auto a = placements(rocke_mma_op_a_layout(op, NULL));
    const auto b = placements(rocke_mma_op_b_layout(op, NULL));
    const auto c = placements(rocke_mma_op_c_layout(op, NULL));
    bool maps = a.size() == 32 * 8 && b.size() == a.size() && c.size() == a.size();
    for(int l = 0; maps && l < 32; ++l)
    {
        for(int slot = 0; slot < 8; ++slot)
        {
            const long long row = l % 16, kk = (l / 16) * 8 + slot;
            const long long c_row = (l / 16) * 8 + slot, c_col = l % 16;
            maps = maps && a[l * 8 + slot] == std::make_pair(row, kk)
                   && b[l * 8 + slot] == std::make_pair(kk, row)
                   && c[l * 8 + slot] == std::make_pair(c_row, c_col);
        }
    }
    CHECK(maps, t.op_id);
}

int main(void)
{
    for(const char* gfx : k_targets)
        for(const Case& t : k_cases)
            check_case(gfx, t);
    if(g_failures)
    {
        fprintf(stderr, "rocke_gfx12_wmma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_gfx12_wmma_ops: all checks passed.\n");
    return 0;
}
