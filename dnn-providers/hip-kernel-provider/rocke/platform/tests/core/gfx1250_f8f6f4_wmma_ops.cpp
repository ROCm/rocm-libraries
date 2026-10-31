// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/gfx1250_f8f6f4_wmma_ops.cpp -- the C++ engine's gfx1250 f8f6f4
 * 16x16x128 and f4 32x16x128 WMMA instructions and the f8f6f4 operand layout.
 *
 * - The narrow format pairs: every (A, B) pair with a 6-bit or 4-bit operand,
 *   A-major in format-code order.
 * - rocke_matrix_operand_layout: K runs of 16 (8-bit) or 32 (6-bit, 4-bit)
 *   taken in turn by the lane groups; the gfx1250 scaled and gfx950 scaled
 *   layouts are calls to it; a non-matrix dtype is refused with the Python
 *   text.
 * - Each f8f6f4 catalog row (dtypes, shape, fragment lengths), its A and B
 *   maps evaluated on every (lane, slot) of a wave against its formats'
 *   layouts, and the gfx12 accumulator map; the f4 32x16x128 row without maps.
 * - Each call: the format immediates, A and B forms (as is for an 8-bit
 *   format, zero-padded for a 6-bit or 4-bit one) and the register types; the
 *   tile.mma result rocke_b_mma builds from the bare op_id.
 *
 * Mirrors tests/core/test_gfx1250_f8f6f4_wmma_ops.py.
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
#include "rocke/wmma_scale_internal.h"

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

/* Every (lane, slot) -> coordinate pair of `map` over a 32-lane wave, in order. */
static Placements placements(const rocke_layout_map_t* map)
{
    Placements out;
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

/* The (index, K) of every (lane, slot) of `layout` over a 32-lane wave. */
static Placements layout_placements(const rocke_matrix_fragment_layout_t& layout)
{
    Placements out;
    for(int l = 0; l < 32; ++l)
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

static bool same_layout(const rocke_matrix_fragment_layout_t& x,
                        const rocke_matrix_fragment_layout_t& y)
{
    return x.fragment.packing.element_bits == y.fragment.packing.element_bits
           && x.fragment.packing.slot_bits == y.fragment.packing.slot_bits
           && x.fragment.count == y.fragment.count
           && x.fragment.carrier_bits == y.fragment.carrier_bits
           && x.fragment.carrier_count == y.fragment.carrier_count
           && x.chunk_elements == y.chunk_elements && x.lane_groups == y.lane_groups
           && x.lanes_per_group == y.lanes_per_group;
}

static void check_pairs()
{
    std::vector<std::pair<int, int>> expected;
    for(int a = 0; a < 5; ++a)
        for(int b = 0; b < 5; ++b)
            if(a >= 2 || b >= 2)
                expected.emplace_back(a, b);
    CHECK(k_rocke_narrow_matrix_format_pairs.count == (int)expected.size(), "pairs");
    for(int i = 0; i < k_rocke_narrow_matrix_format_pairs.count && i < (int)expected.size(); ++i)
        CHECK(k_rocke_narrow_matrix_format_pairs.pairs[i].a == expected[i].first
                  && k_rocke_narrow_matrix_format_pairs.pairs[i].b == expected[i].second,
              "pair order");
}

static void check_operand_layouts()
{
    const auto fp8 = rocke_matrix_operand_layout("fp8e4m3", 16, 2, 16);
    CHECK(fp8.fragment.count == 64 && fp8.chunk_elements == 16, "fp8 layout");
    uint64_t index = 0, k = 0;
    CHECK(rocke_matrix_fragment_coord(&fp8, 19, 16, &index, &k) && index == 3 && k == 48,
          "fp8 lane 19 slot 16");
    for(const char* narrow : {"fp6e2m3", "fp6e3m2", "fp4e2m1"})
    {
        const auto layout = rocke_matrix_operand_layout(narrow, 16, 2, 16);
        CHECK(layout.fragment.count == 64 && layout.chunk_elements == 32, narrow);
        CHECK(rocke_matrix_fragment_coord(&layout, 0, 32, &index, &k) && index == 0 && k == 64,
              narrow);
    }
    for(const char* dtype : {"fp8", "bf8", "fp6", "bf6", "fp4"})
    {
        const char* canonical = rocke_require_dtype_info(dtype)->name;
        CHECK(same_layout(rocke_scaled_matrix_layout(dtype, 16),
                          rocke_matrix_operand_layout(dtype, 16, 2, 16)),
              dtype);
        CHECK(same_layout(rocke_mfma_matrix_layout(canonical),
                          rocke_matrix_operand_layout(canonical, 8, 2, 32)),
              dtype);
    }
    std::string message;
    try
    {
        rocke_matrix_operand_layout("fp16", 16, 2, 16);
    }
    catch(const ckc::ValueError& error)
    {
        message = error.what();
    }
    CHECK(message
              == "'fp16' is not an f8f6f4 matrix format; expected one of fp8e4m3, bf8e5m2, "
                 "fp6e2m3, fp6e3m2, fp4e2m1",
          "non-matrix dtype");
}

/* rocke_b_mma sizes the result from the bare op_id: <count x float>. */
static void check_result(const char* id, int count)
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "gfx1250_result") == ROCKE_OK, id);
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

static void check_f8f6f4(const rocke_arch_target_t* arch)
{
    for(int i = 0; i < k_rocke_narrow_matrix_format_pairs.count; ++i)
    {
        const int a = k_rocke_narrow_matrix_format_pairs.pairs[i].a;
        const int b = k_rocke_narrow_matrix_format_pairs.pairs[i].b;
        const std::string op_id = std::string("wmma_gfx1250_f32_16x16x128_f8f6f4_")
                                  + k_rocke_matrix_formats[a] + "_" + k_rocke_matrix_formats[b];
        const char* id = op_id.c_str();
        const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, id);
        CHECK(op != NULL, id);
        if(!op)
            continue;
        CHECK(strcmp(op->family, "wmma") == 0 && strcmp(op->a_dtype, k_rocke_matrix_formats[a]) == 0
                  && strcmp(op->b_dtype, k_rocke_matrix_formats[b]) == 0
                  && strcmp(op->c_dtype, "fp32") == 0 && op->dst_dtype == NULL,
              id);
        CHECK(op->m == 16 && op->n == 16 && op->k == 128, id);
        CHECK(op->a_frag_len == 64 && op->b_frag_len == 64 && op->c_frag_len == 8
                  && op->wave_size == 32,
              id);

        /* A is (row, K) where its format's layout puts it; B is (K, column). */
        const Placements a_expected
            = layout_placements(rocke_matrix_operand_layout(k_rocke_matrix_formats[a], 16, 2, 16));
        Placements b_expected;
        for(const auto& p :
            layout_placements(rocke_matrix_operand_layout(k_rocke_matrix_formats[b], 16, 2, 16)))
            b_expected.emplace_back(p.second, p.first);
        CHECK(placements(rocke_mma_op_a_layout(op, NULL)) == a_expected, id);
        CHECK(placements(rocke_mma_op_b_layout(op, NULL)) == b_expected, id);
        const Placements c = placements(rocke_mma_op_c_layout(op, NULL));
        bool c_map = c.size() == 32 * 8;
        for(int l = 0; c_map && l < 32; ++l)
            for(int slot = 0; slot < 8; ++slot)
                c_map = c_map
                        && c[l * 8 + slot]
                               == std::make_pair((long long)(l / 16) * 8 + slot, (long long)l % 16);
        CHECK(c_map, id);

        /* The call: (format A, A, format B, B, i16 0, C). */
        const rocke_matrix_call_t* call = rocke_matrix_call_for(id);
        CHECK(call != NULL, id);
        if(!call)
            continue;
        CHECK(strcmp(call->intrinsic, "llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32")
                  == 0,
              id);
        CHECK(call->num_args == 6 && call->args[0].kind == ROCKE_MATRIX_ARG_IMM
                  && strcmp(call->args[0].imm_type, "i32") == 0 && call->args[0].imm_value == a
                  && call->args[1].kind == ROCKE_MATRIX_ARG_A
                  && call->args[2].kind == ROCKE_MATRIX_ARG_IMM && call->args[2].imm_value == b
                  && call->args[3].kind == ROCKE_MATRIX_ARG_B
                  && strcmp(call->args[4].imm_type, "i16") == 0 && call->args[4].imm_value == 0
                  && call->args[5].kind == ROCKE_MATRIX_ARG_C,
              id);
        CHECK(call->a.kind
                  == (rocke_matrix_format_bits(a) == 8 ? ROCKE_MATRIX_FORM_AS_IS
                                                       : ROCKE_MATRIX_FORM_ZERO_PADDED),
              id);
        CHECK(call->b.kind
                  == (rocke_matrix_format_bits(b) == 8 ? ROCKE_MATRIX_FORM_AS_IS
                                                       : ROCKE_MATRIX_FORM_ZERO_PADDED),
              id);
        CHECK(is_register(call->a, "i32", 16) && is_register(call->b, "i32", 16)
                  && is_register(call->c, "f32", 8) && is_register(call->result, "f32", 8),
              id);
        check_result(id, 8);
    }
}

static void check_f4(const rocke_arch_target_t* arch)
{
    const char* id = "wmma_gfx1250_f32_32x16x128_f4";
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, id);
    CHECK(op != NULL, id);
    if(!op)
        return;
    CHECK(strcmp(op->family, "wmma") == 0 && strcmp(op->a_dtype, "fp4e2m1") == 0
              && strcmp(op->b_dtype, "fp4e2m1") == 0 && strcmp(op->c_dtype, "fp32") == 0,
          id);
    CHECK(op->m == 32 && op->n == 16 && op->k == 128, id);
    CHECK(op->a_frag_len == 128 && op->b_frag_len == 64 && op->c_frag_len == 16
              && op->wave_size == 32,
          id);
    CHECK(op->a_layout == NULL && op->b_layout == NULL && op->c_layout == NULL, id);
    const rocke_matrix_call_t* call = rocke_matrix_call_for(id);
    CHECK(call != NULL, id);
    if(call)
    {
        CHECK(strcmp(call->intrinsic, "llvm.amdgcn.wmma.f32.32x16x128.f4.v16f32.v16i32.v8i32") == 0,
              id);
        CHECK(call->a.kind == ROCKE_MATRIX_FORM_ZERO_PADDED
                  && call->b.kind == ROCKE_MATRIX_FORM_ZERO_PADDED,
              id);
        CHECK(is_register(call->a, "i32", 16) && is_register(call->b, "i32", 8)
                  && is_register(call->c, "f32", 16) && is_register(call->result, "f32", 16),
              id);
    }
    check_result(id, 16);
}

int main(void)
{
    check_pairs();
    check_operand_layouts();
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx("gfx1250");
    CHECK(arch != NULL, "gfx1250");
    if(arch)
    {
        check_f8f6f4(arch);
        check_f4(arch);
    }
    if(g_failures)
    {
        fprintf(stderr, "rocke_gfx1250_f8f6f4_wmma_ops: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_gfx1250_f8f6f4_wmma_ops: all checks passed.\n");
    return 0;
}
