// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/gfx950_f8f6f4_mfma_ops.cpp -- the C++ engine's gfx950 f8f6f4
 * 16x16x128 and 32x32x64 MFMA instructions.
 *
 * - Each catalog row (dtypes, shape, fragment lengths), one per narrow format
 *   pair and shape.
 * - Its A and B maps evaluated on every (lane, slot) of a 64-lane wave against
 *   its formats' f8f6f4 operand layouts (four groups of 16 lanes at
 *   16x16x128, two groups of 32 at 32x32x64), and the shape's f32 accumulator
 *   map.
 * - Each call: the nine-argument intrinsic and its declaration key, the format
 *   immediates and the zero scale arguments, A and B forms (as is for an 8-bit
 *   format, zero-padded for a 6-bit or 4-bit one) and the register types; the
 *   tile.mma result rocke_b_mma builds from the bare op_id.
 *
 * Mirrors tests/core/test_gfx950_f8f6f4_mfma_ops.py.
 *
 * Plain executable: returns non-zero when a check fails.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/matrix_calls_internal.h"
#include "rocke/mfma_scale_internal.h"

using namespace ckc;

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

typedef std::vector<std::pair<long long, long long>> Placements;

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
static Placements placements(const rocke_layout_map_t* map)
{
    Placements out;
    rocke_ir_builder_t b;
    if(!map || rocke_ir_builder_init(&b, "gfx950_lane_map") != ROCKE_OK)
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

/* The (index, K) of every (lane, slot) of `layout` over a 64-lane wave. */
static Placements layout_placements(const rocke_matrix_fragment_layout_t& layout)
{
    Placements out;
    for(int l = 0; l < 64; ++l)
    {
        for(uint64_t slot = 0; slot < layout.fragment.count; ++slot)
        {
            uint64_t index = 0, k = 0;
            CHECK(rocke_matrix_fragment_coord(&layout, l, slot, &index, &k), "layout coord");
            out.emplace_back((long long)index, (long long)k);
        }
    }
    return out;
}

/* rocke_b_mma sizes the result from the bare op_id: <count x float>. */
static void check_result(const char* id, int count)
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "gfx950_result") == ROCKE_OK, id);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    rocke_value_t* d = rocke_b_mma(&b, id, zero, zero, zero, NULL, 0);
    CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == count, id);
    if(d)
        CHECK(d->type->elem == rocke_dtype_to_ir_type("fp32"), id);
    rocke_ir_builder_free(&b);
}

static bool is_register(const rocke_matrix_form_t& form, const char* elem, int count)
{
    return form.register_type.num_overrides == 0 && strcmp(form.register_type.base.elem, elem) == 0
           && form.register_type.base.count == count;
}

static bool is_imm(const rocke_matrix_arg_t& arg, int value, bool immarg)
{
    return arg.kind == ROCKE_MATRIX_ARG_IMM && strcmp(arg.imm_type, "i32") == 0
           && arg.imm_value == value && arg.immarg == immarg;
}

struct shape_t
{
    int m, n, k, lane_groups, lanes_per_group, results;
    const char* intrinsic;
    const char* decl_key;
    const char* accumulator_op; /* an f16 op of the shape: its accumulator map */
};

static void check_shape(const rocke_arch_target_t* arch, const shape_t& shape)
{
    const rocke_mma_op_t* acc_op = rocke_mma_catalog_by_op_id(&arch->mma, shape.accumulator_op);
    CHECK(acc_op != NULL, shape.accumulator_op);
    const Placements acc_expected
        = acc_op ? placements(rocke_mma_op_c_layout(acc_op, NULL)) : Placements();
    for(int i = 0; i < k_rocke_narrow_matrix_format_pairs.count; ++i)
    {
        const int a = k_rocke_narrow_matrix_format_pairs.pairs[i].a;
        const int b = k_rocke_narrow_matrix_format_pairs.pairs[i].b;
        const std::string op_id = "mfma_f32_" + std::to_string(shape.m) + "x"
                                  + std::to_string(shape.n) + "x" + std::to_string(shape.k)
                                  + "_f8f6f4_" + k_rocke_matrix_formats[a] + "_"
                                  + k_rocke_matrix_formats[b];
        const char* id = op_id.c_str();
        const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, id);
        CHECK(op != NULL, id);
        if(!op)
            continue;
        CHECK(strcmp(op->family, "mma") == 0 && strcmp(op->a_dtype, k_rocke_matrix_formats[a]) == 0
                  && strcmp(op->b_dtype, k_rocke_matrix_formats[b]) == 0
                  && strcmp(op->c_dtype, "fp32") == 0 && op->dst_dtype == NULL,
              id);
        CHECK(op->m == shape.m && op->n == shape.n && op->k == shape.k, id);
        CHECK(op->a_frag_len == 32 && op->b_frag_len == 32 && op->c_frag_len == shape.results
                  && op->wave_size == 64,
              id);

        /* A is (row, K) where its format's layout puts it; B is (K, column). */
        const Placements a_expected = layout_placements(rocke_matrix_operand_layout(
            k_rocke_matrix_formats[a], 8, shape.lane_groups, shape.lanes_per_group));
        Placements b_expected;
        for(const auto& p : layout_placements(rocke_matrix_operand_layout(
                k_rocke_matrix_formats[b], 8, shape.lane_groups, shape.lanes_per_group)))
            b_expected.emplace_back(p.second, p.first);
        CHECK(placements(rocke_mma_op_a_layout(op, NULL)) == a_expected, id);
        CHECK(placements(rocke_mma_op_b_layout(op, NULL)) == b_expected, id);
        CHECK(placements(rocke_mma_op_c_layout(op, NULL)) == acc_expected, id);

        /* The call: (A, B, C, format A, format B, 0, 0, 0, 0). */
        const rocke_matrix_call_t* call = rocke_matrix_call_for(id);
        CHECK(call != NULL, id);
        if(!call)
            continue;
        CHECK(strcmp(call->intrinsic, shape.intrinsic) == 0, id);
        CHECK(call->decl_key && strcmp(call->decl_key, shape.decl_key) == 0, id);
        CHECK(call->num_args == 9 && call->args[0].kind == ROCKE_MATRIX_ARG_A
                  && call->args[1].kind == ROCKE_MATRIX_ARG_B
                  && call->args[2].kind == ROCKE_MATRIX_ARG_C && is_imm(call->args[3], a, true)
                  && is_imm(call->args[4], b, true) && is_imm(call->args[5], 0, true)
                  && is_imm(call->args[6], 0, false) && is_imm(call->args[7], 0, true)
                  && is_imm(call->args[8], 0, false),
              id);
        CHECK(call->a.kind
                  == (rocke_matrix_format_bits(a) == 8 ? ROCKE_MATRIX_FORM_AS_IS
                                                       : ROCKE_MATRIX_FORM_ZERO_PADDED),
              id);
        CHECK(call->b.kind
                  == (rocke_matrix_format_bits(b) == 8 ? ROCKE_MATRIX_FORM_AS_IS
                                                       : ROCKE_MATRIX_FORM_ZERO_PADDED),
              id);
        CHECK(is_register(call->a, "i32", 8) && is_register(call->b, "i32", 8)
                  && is_register(call->c, "f32", shape.results)
                  && is_register(call->result, "f32", shape.results),
              id);
        check_result(id, shape.results);
    }
}

int main(void)
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx("gfx950");
    CHECK(arch != NULL, "gfx950");
    if(arch)
    {
        check_shape(arch,
                    {16,
                     16,
                     128,
                     4,
                     16,
                     4,
                     "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4",
                     "mfma.f32.16x16x128.fp8.hero",
                     "mfma_f32_16x16x16_f16"});
        check_shape(arch,
                    {32,
                     32,
                     64,
                     2,
                     32,
                     16,
                     "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32",
                     "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32",
                     "mfma_f32_32x32x8_f16"});
    }
    if(g_failures)
    {
        fprintf(stderr, "rocke_gfx950_f8f6f4_mfma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_gfx950_f8f6f4_mfma_ops: all checks passed.\n");
    return 0;
}
