// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/gfx11_wmma_ops.cpp -- the C++ engine's catalog rows and lane maps
 * for the gfx11 WMMA instructions with a 16-bit result, f16 -> f16 and
 * bf16 -> bf16, on gfx11-generic and on gfx1100 and gfx1151, which take them
 * from the generic row.
 *
 * For each one: the catalog row (dtypes, shape, fragment lengths: sixteen A
 * and B values and eight logical results per lane), the tile.mma result
 * rocke_b_mma builds from the bare op_id (<8 x c>), and the lane maps,
 * evaluated on every (lane, slot) of a wave, equal to those of the same
 * instruction with an f32 result: result slot i of lane l is
 * (row 2 * i + l / 16, column l % 16). Mirrors
 * tests/core/test_gfx11_wmma_ops.py.
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
    const char* dtype; /* of A, B, the accumulator and the result */
    const char* f32_output_op_id; /* the same instruction with an f32 result */
};

static const Case k_cases[] = {
    {"wmma_f16_16x16x16_f16", "fp16", "wmma_f32_16x16x16_f16"},
    {"wmma_bf16_16x16x16_bf16", "bf16", "wmma_f32_16x16x16_bf16"},
};

/* The rows that hold the instructions: the generic row and two processors that
 * point at it. */
static const char* const k_targets[] = {"gfx11-generic", "gfx1100", "gfx1151"};

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
    if(rocke_ir_builder_init(&b, "gfx11_lane_map") != ROCKE_OK)
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
    const rocke_mma_op_t* f32_output = rocke_mma_catalog_by_op_id(&arch->mma, t.f32_output_op_id);
    CHECK(op != NULL && f32_output != NULL, t.op_id);
    if(!op || !f32_output)
        return;
    CHECK(strcmp(op->family, "wmma") == 0, t.op_id);
    CHECK(strcmp(op->a_dtype, t.dtype) == 0 && strcmp(op->b_dtype, t.dtype) == 0
              && strcmp(op->c_dtype, t.dtype) == 0,
          t.op_id);
    CHECK(op->m == 16 && op->n == 16 && op->k == 16, t.op_id);
    CHECK(op->a_frag_len == 16 && op->b_frag_len == 16 && op->c_frag_len == 8
              && op->wave_size == 32,
          t.op_id);

    /* rocke_b_mma sizes the result from the bare op_id: <8 x dtype>. */
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "gfx11_result") == ROCKE_OK, t.op_id);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_value_t* d = rocke_b_mma(&b, t.op_id, zero, zero, zero, NULL, 0);
        CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == 8, t.op_id);
        if(d)
            CHECK(d->type->elem == rocke_dtype_to_ir_type(t.dtype), t.op_id);
        rocke_ir_builder_free(&b);
    }

    const auto a = placements(rocke_mma_op_a_layout(op, NULL));
    const auto b = placements(rocke_mma_op_b_layout(op, NULL));
    const auto c = placements(rocke_mma_op_c_layout(op, NULL));
    CHECK(a == placements(rocke_mma_op_a_layout(f32_output, NULL)), t.op_id);
    CHECK(b == placements(rocke_mma_op_b_layout(f32_output, NULL)), t.op_id);
    CHECK(c == placements(rocke_mma_op_c_layout(f32_output, NULL)), t.op_id);
    bool maps = c.size() == 32 * 8;
    for(int l = 0; maps && l < 32; ++l)
    {
        for(int slot = 0; slot < 8; ++slot)
        {
            const long long row = 2 * slot + l / 16, col = l % 16;
            maps = maps && c[l * 8 + slot] == std::make_pair(row, col);
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
        fprintf(stderr, "rocke_gfx11_wmma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_gfx11_wmma_ops: all checks passed.\n");
    return 0;
}
