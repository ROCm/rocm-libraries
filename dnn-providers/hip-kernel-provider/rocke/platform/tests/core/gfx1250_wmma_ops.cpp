// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/gfx1250_wmma_ops.cpp -- the C++ engine's gfx1250 catalog rows for
 * the WMMA instructions with a 16-bit or integer result (f16 -> f16 and
 * bf16 -> bf16 at K 32, fp8 / bf8 -> f16 at K 64 and K 128, i8 -> i32 at K 64
 * in every operand signedness) and the lane maps of f32 -> f32 at K 4.
 *
 * For each new instruction: the catalog row (dtypes, shape, fragment lengths
 * in logical values per lane), the tile.mma result rocke_b_mma builds from the
 * bare op_id (<8 x c>), and that it has no lane maps yet. For f32 16x16x4: the
 * A and B maps, evaluated on every (lane, slot) of a wave, are CK's -- lane l
 * holds row l % 16 of A (column l % 16 of B) and K 2 * (l / 16) + slot -- and
 * the result map is the gfx12 accumulator's. Mirrors
 * tests/core/test_gfx1250_wmma_ops.py.
 *
 * Plain executable: returns non-zero when a check fails.
 */
#include <cstdio>
#include <cstring>
#include <string>
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
    std::string op_id;
    const char* a;
    const char* b;
    const char* c;
    int k;
    int k_per_lane; /* values of A and of B per lane */
};

static std::vector<Case> cases()
{
    std::vector<Case> out = {
        {"wmma_gfx1250_f16_16x16x32_f16", "fp16", "fp16", "fp16", 32, 16},
        {"wmma_gfx1250_bf16_16x16x32_bf16", "bf16", "bf16", "bf16", 32, 16},
        {"wmma_gfx1250_i32_16x16x64_iu8", "i8", "i8", "i32", 64, 32},
        {"wmma_gfx1250_i32_16x16x64_iu8_u8_i8", "u8", "i8", "i32", 64, 32},
        {"wmma_gfx1250_i32_16x16x64_iu8_i8_u8", "i8", "u8", "i32", 64, 32},
        {"wmma_gfx1250_i32_16x16x64_iu8_u8_u8", "u8", "u8", "i32", 64, 32},
    };
    const char* const names[] = {"fp8", "bf8"};
    const char* const dtypes[] = {"fp8e4m3", "bf8e5m2"};
    for(int k : {64, 128})
        for(int a = 0; a < 2; ++a)
            for(int b = 0; b < 2; ++b)
                out.push_back({"wmma_gfx1250_f16_16x16x" + std::to_string(k) + "_" + names[a] + "_"
                                   + names[b],
                               dtypes[a],
                               dtypes[b],
                               "fp16",
                               k,
                               k / 2});
    return out;
}

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
    if(!map || rocke_ir_builder_init(&b, "gfx1250_lane_map") != ROCKE_OK)
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

static void check_case(const rocke_arch_target_t* arch, const Case& t)
{
    const char* id = t.op_id.c_str();
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, id);
    CHECK(op != NULL, id);
    if(!op)
        return;
    CHECK(strcmp(op->family, "wmma") == 0, id);
    CHECK(strcmp(op->a_dtype, t.a) == 0 && strcmp(op->b_dtype, t.b) == 0
              && strcmp(op->c_dtype, t.c) == 0 && op->dst_dtype == NULL,
          id);
    CHECK(op->m == 16 && op->n == 16 && op->k == t.k, id);
    CHECK(op->a_frag_len == t.k_per_lane && op->b_frag_len == t.k_per_lane && op->c_frag_len == 8
              && op->wave_size == 32,
          id);
    CHECK(op->a_layout == NULL && op->b_layout == NULL && op->c_layout == NULL, id);

    /* rocke_b_mma sizes the result from the bare op_id: <8 x c>. */
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "gfx1250_result") == ROCKE_OK, id);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    rocke_value_t* d = rocke_b_mma(&b, id, zero, zero, zero, NULL, 0);
    CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == 8, id);
    if(d)
        CHECK(d->type->elem == rocke_dtype_to_ir_type(t.c), id);
    rocke_ir_builder_free(&b);
}

static void check_f32_16x16x4_maps(const rocke_arch_target_t* arch)
{
    const char* id = "wmma_gfx1250_f32_16x16x4_f32";
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, id);
    CHECK(op != NULL, id);
    if(!op)
        return;
    const auto a = placements(rocke_mma_op_a_layout(op, NULL));
    const auto b = placements(rocke_mma_op_b_layout(op, NULL));
    const auto c = placements(rocke_mma_op_c_layout(op, NULL));
    bool maps = a.size() == 64 && b.size() == 64 && c.size() == 32 * 8;
    for(int l = 0; maps && l < 32; ++l)
    {
        for(int slot = 0; slot < 2; ++slot)
        {
            const long long index = l % 16, kk = 2 * (l / 16) + slot;
            maps = maps && a[l * 2 + slot] == std::make_pair(index, kk)
                   && b[l * 2 + slot] == std::make_pair(kk, index);
        }
        for(int slot = 0; slot < 8; ++slot)
        {
            const long long c_row = (l / 16) * 8 + slot, c_col = l % 16;
            maps = maps && c[l * 8 + slot] == std::make_pair(c_row, c_col);
        }
    }
    CHECK(maps, id);
}

int main(void)
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx("gfx1250");
    CHECK(arch != NULL, "gfx1250");
    if(arch)
    {
        for(const Case& t : cases())
            check_case(arch, t);
        check_f32_16x16x4_maps(arch);
    }
    if(g_failures)
    {
        fprintf(stderr, "rocke_gfx1250_wmma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_gfx1250_wmma_ops: all checks passed.\n");
    return 0;
}
