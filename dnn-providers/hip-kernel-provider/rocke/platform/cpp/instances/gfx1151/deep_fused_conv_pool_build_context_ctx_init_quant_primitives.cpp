// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_Build-context + ctx_init + quant
 * primitives.c -- one bucket of the chunked C99 port of the gfx1151
 * (RDNA3.5 / Strix Halo, wave32, WMMA 16x16x16) genuine-int8/int4 deep-fused
 * conv0 -> conv1 -> maxpool builder
 *   (rocke/instances/gfx1151/deep_fused_conv_pool.py, 3238 LOC).
 *
 * SCOPE OF THIS PART-FILE:
 *   - rocke_gfx1151_dfcp_build_ctx_init        (the Python prologue, lines
 *                                             2688-2778: is_valid gate -> resolve
 *                                             WMMA f16 op + iu8/iu4 op0/op1 ->
 *                                             declare X/W0/Y/W1 params ->
 *                                             WarpGrid.from_atom + bind ->
 *                                             waves_per_eu attrs + SchedulePolicy
 *                                             -> a0/c0/w1/c1 dtype + kpad resolve
 *                                             -> LDS allocs)
 *   - the 17 pure quantization primitives    (Python lines 648-741):
 *       _quant_i8 / _quant_i4 / _i8_to_f32 / _neg_i32 / _clamp_i32 / _relu_i32 /
 *       _round_shift_rne_i32 / _quant_i8_shift / _quant_i4_shift + the _vec_i32
 *       twins (_neg / _clamp / _relu / _round_shift_rne / _quant_i8_shift /
 *       _quant_i4_shift) + _splat_i32
 *   - the requant code-fn selector emitters   (Python conv0/conv1 closures, lines
 *       2786-2793, 2919-2940, 3051-3066): apply_code_fn / apply_vec_code_fn
 *
 * Peer phases (staging, fragment loaders, WMMA GEMMs, the conv0->conv1 register
 * handoff, scatter, maxpool, the persistent / single-tile drivers) live in
 * sibling gfx1151 part-files and are reached only via the internal header; this
 * TU implements ONLY the bucket above and calls peers through their prototypes.
 *
 * Builder call sequence is byte-faithful to the Python prologue + closures.
 */
#include "rocke/instance_gfx1151_deep_fused_conv_pool.h"
#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <string.h> /* memset */

#include "rocke/arena.h" /* rocke_arena_alloc */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx / _op_for_shape / _by_op_id */
#include "rocke/helper_rocke.helpers.epilogues.h" /* rocke_warp_grid_t + rocke_warp_grid_block_size */
#include "rocke/helper_rocke.helpers.schedule.h" /* rocke_schedule_policy_for_pipeline */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ===================================================================== *
 *  QUANTIZATION PRIMITIVES (Python lines 648-741) -- pure IR emitters.
 * ===================================================================== */

/* def _quant_i8(b, vf32, inv_scale):
 *     scaled = b.fmul(vf32, inv_scale)
 *     clamped = b.clamp_f32(scaled, b.const_f32(-127.0), b.const_f32(127.0))
 *     return b.cvt_f32_to_i8_sat(clamped) */
rocke_value_t* rocke_gfx1151_dfcp_quant_i8(rocke_ir_builder_t* b,
                                           rocke_value_t* vf32,
                                           rocke_value_t* inv_scale)
{
    rocke_value_t* scaled;
    rocke_value_t* clamped;
    if(b == NULL)
    {
        return NULL;
    }
    scaled = rocke_b_fmul(b, vf32, inv_scale);
    {
        /* hoist clamp bounds in Python's left-to-right order: lo first, then hi */
        rocke_value_t* lo = rocke_b_const_f32(b, -127.0);
        rocke_value_t* hi = rocke_b_const_f32(b, 127.0);
        clamped = rocke_b_clamp_f32(b, scaled, lo, hi);
    }
    return rocke_b_cvt_f32_to_i8_sat(b, clamped);
}

/* def _quant_i4(b, vf32, inv_scale):
 *     scaled = b.fmul(vf32, inv_scale)
 *     clamped = b.clamp_f32(scaled, b.const_f32(-8.0), b.const_f32(7.0))
 *     return b.cvt_f32_to_i8_sat(clamped) */
rocke_value_t* rocke_gfx1151_dfcp_quant_i4(rocke_ir_builder_t* b,
                                           rocke_value_t* vf32,
                                           rocke_value_t* inv_scale)
{
    rocke_value_t* scaled;
    rocke_value_t* clamped;
    if(b == NULL)
    {
        return NULL;
    }
    scaled = rocke_b_fmul(b, vf32, inv_scale);
    {
        /* hoist clamp bounds in Python's left-to-right order: lo first, then hi */
        rocke_value_t* lo = rocke_b_const_f32(b, -8.0);
        rocke_value_t* hi = rocke_b_const_f32(b, 7.0);
        clamped = rocke_b_clamp_f32(b, scaled, lo, hi);
    }
    return rocke_b_cvt_f32_to_i8_sat(b, clamped);
}

/* def _i8_to_f32(b, qi8):
 *     return b.sitofp_f32(b.sext(qi8, I32)) */
rocke_value_t* rocke_gfx1151_dfcp_i8_to_f32(rocke_ir_builder_t* b, rocke_value_t* qi8)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_sitofp_f32(b, rocke_b_sext(b, qi8, rocke_i32()));
}

/* def _neg_i32(b, x):
 *     return b.sub(b.const_i32(0), x) */
rocke_value_t* rocke_gfx1151_dfcp_neg_i32(rocke_ir_builder_t* b, rocke_value_t* x)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_sub(b, rocke_b_const_i32(b, 0), x);
}

/* def _clamp_i32(b, x, lo, hi):
 *     return b.smin(b.smax(x, b.const_i32(lo)), b.const_i32(hi)) */
rocke_value_t* rocke_gfx1151_dfcp_clamp_i32(rocke_ir_builder_t* b, rocke_value_t* x, int lo, int hi)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_smin(b, rocke_b_smax(b, x, rocke_b_const_i32(b, lo)), rocke_b_const_i32(b, hi));
}

/* def _relu_i32(b, x):
 *     return b.smax(x, b.const_i32(0)) */
rocke_value_t* rocke_gfx1151_dfcp_relu_i32(rocke_ir_builder_t* b, rocke_value_t* x)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_smax(b, x, rocke_b_const_i32(b, 0));
}

/* def _round_shift_rne_i32(b, x, shift):
 *     if shift == 0: return x
 *     sign = b.cmp_lt(x, b.const_i32(0))
 *     ax = b.select(sign, _neg_i32(b, x), x)
 *     floor_q = b.lshr(ax, b.const_i32(shift))
 *     bias = b.add(b.const_i32((1 << (shift - 1)) - 1), b.land(floor_q, b.const_i32(1)))
 *     q = b.lshr(b.add(ax, bias), b.const_i32(shift))
 *     return b.select(sign, _neg_i32(b, q), q) */
rocke_value_t*
    rocke_gfx1151_dfcp_round_shift_rne_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    rocke_value_t* sign;
    rocke_value_t* ax;
    rocke_value_t* floor_q;
    rocke_value_t* bias;
    rocke_value_t* q;
    if(b == NULL)
    {
        return NULL;
    }
    if(shift == 0)
    {
        return x;
    }
    sign = rocke_b_cmp_lt(b, x, rocke_b_const_i32(b, 0));
    ax = rocke_b_select(b, sign, rocke_gfx1151_dfcp_neg_i32(b, x), x);
    floor_q = rocke_b_lshr(b, ax, rocke_b_const_i32(b, shift));
    bias = rocke_b_add(b,
                       rocke_b_const_i32(b, (int64_t)((1 << (shift - 1)) - 1)),
                       rocke_b_land(b, floor_q, rocke_b_const_i32(b, 1)));
    q = rocke_b_lshr(b, rocke_b_add(b, ax, bias), rocke_b_const_i32(b, shift));
    return rocke_b_select(b, sign, rocke_gfx1151_dfcp_neg_i32(b, q), q);
}

/* def _quant_i8_shift(b, x, shift):
 *     q = _clamp_i32(b, _round_shift_rne_i32(b, x, shift), -127, 127)
 *     return b.trunc(q, I8) */
rocke_value_t* rocke_gfx1151_dfcp_quant_i8_shift(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    rocke_value_t* q;
    if(b == NULL)
    {
        return NULL;
    }
    q = rocke_gfx1151_dfcp_clamp_i32(
        b, rocke_gfx1151_dfcp_round_shift_rne_i32(b, x, shift), -127, 127);
    return rocke_b_trunc(b, q, rocke_i8());
}

/* def _quant_i4_shift(b, x, shift):
 *     q = _clamp_i32(b, _round_shift_rne_i32(b, x, shift), -8, 7)
 *     return b.trunc(q, I8) */
rocke_value_t* rocke_gfx1151_dfcp_quant_i4_shift(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    rocke_value_t* q;
    if(b == NULL)
    {
        return NULL;
    }
    q = rocke_gfx1151_dfcp_clamp_i32(b, rocke_gfx1151_dfcp_round_shift_rne_i32(b, x, shift), -8, 7);
    return rocke_b_trunc(b, q, rocke_i8());
}

/* def _splat_i32(b, value, n):
 *     return b.vector_splat(b.const_i32(value), n) */
rocke_value_t* rocke_gfx1151_dfcp_splat_i32(rocke_ir_builder_t* b, int value, int n)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_vector_splat(b, rocke_b_const_i32(b, value), n);
}

/* Helper: x.type.count for an <N x i32> vector operand. */
static int rocke_gfx1151_dfcp_vec_count(const rocke_value_t* x)
{
    if(x == NULL || x->type == NULL)
    {
        return 0;
    }
    return x->type->count;
}

/* def _neg_i32_vec(b, x):
 *     return b.vector_sub(b.zero_vec(I32, x.type.count), x) */
rocke_value_t* rocke_gfx1151_dfcp_neg_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_vector_sub(
        b, rocke_b_zero_vec(b, rocke_i32(), rocke_gfx1151_dfcp_vec_count(x)), x);
}

/* def _clamp_i32_vec(b, x, lo, hi):
 *     n = x.type.count
 *     return b.vector_smin(b.vector_smax(x, _splat_i32(b, lo, n)), _splat_i32(b, hi, n)) */
rocke_value_t*
    rocke_gfx1151_dfcp_clamp_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x, int lo, int hi)
{
    int n;
    if(b == NULL)
    {
        return NULL;
    }
    n = rocke_gfx1151_dfcp_vec_count(x);
    return rocke_b_vector_smin(b,
                               rocke_b_vector_smax(b, x, rocke_gfx1151_dfcp_splat_i32(b, lo, n)),
                               rocke_gfx1151_dfcp_splat_i32(b, hi, n));
}

/* def _relu_i32_vec(b, x):
 *     return b.vector_smax(x, b.zero_vec(I32, x.type.count)) */
rocke_value_t* rocke_gfx1151_dfcp_relu_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_b_vector_smax(
        b, x, rocke_b_zero_vec(b, rocke_i32(), rocke_gfx1151_dfcp_vec_count(x)));
}

/* def _round_shift_rne_i32_vec(b, x, shift):
 *     if shift == 0: return x
 *     n = x.type.count
 *     sign = b.vector_cmp("lt", x, b.zero_vec(I32, n))
 *     ax = b.vector_select(sign, _neg_i32_vec(b, x), x)
 *     floor_q = b.vector_lshr(ax, _splat_i32(b, shift, n))
 *     bias = b.vector_add(
 *         _splat_i32(b, (1 << (shift - 1)) - 1, n),
 *         b.vector_and(floor_q, _splat_i32(b, 1, n)))
 *     q = b.vector_lshr(b.vector_add(ax, bias), _splat_i32(b, shift, n))
 *     return b.vector_select(sign, _neg_i32_vec(b, q), q) */
rocke_value_t*
    rocke_gfx1151_dfcp_round_shift_rne_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    int n;
    rocke_value_t* sign;
    rocke_value_t* ax;
    rocke_value_t* floor_q;
    rocke_value_t* bias;
    rocke_value_t* q;
    if(b == NULL)
    {
        return NULL;
    }
    if(shift == 0)
    {
        return x;
    }
    n = rocke_gfx1151_dfcp_vec_count(x);
    sign = rocke_b_vector_cmp(b, "lt", x, rocke_b_zero_vec(b, rocke_i32(), n));
    ax = rocke_b_vector_select(b, sign, rocke_gfx1151_dfcp_neg_i32_vec(b, x), x);
    floor_q = rocke_b_vector_lshr(b, ax, rocke_gfx1151_dfcp_splat_i32(b, shift, n));
    bias
        = rocke_b_vector_add(b,
                             rocke_gfx1151_dfcp_splat_i32(b, (1 << (shift - 1)) - 1, n),
                             rocke_b_vector_and(b, floor_q, rocke_gfx1151_dfcp_splat_i32(b, 1, n)));
    q = rocke_b_vector_lshr(
        b, rocke_b_vector_add(b, ax, bias), rocke_gfx1151_dfcp_splat_i32(b, shift, n));
    return rocke_b_vector_select(b, sign, rocke_gfx1151_dfcp_neg_i32_vec(b, q), q);
}

/* def _quant_i8_shift_vec_i32(b, x, shift):
 *     return _clamp_i32_vec(b, _round_shift_rne_i32_vec(b, x, shift), -127, 127) */
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i8_shift_vec_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_gfx1151_dfcp_clamp_i32_vec(
        b, rocke_gfx1151_dfcp_round_shift_rne_i32_vec(b, x, shift), -127, 127);
}

/* def _quant_i4_shift_vec_i32(b, x, shift):
 *     return _clamp_i32_vec(b, _round_shift_rne_i32_vec(b, x, shift), -8, 7) */
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i4_shift_vec_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift)
{
    if(b == NULL)
    {
        return NULL;
    }
    return rocke_gfx1151_dfcp_clamp_i32_vec(
        b, rocke_gfx1151_dfcp_round_shift_rne_i32_vec(b, x, shift), -8, 7);
}

/* ===================================================================== *
 *  REQUANT CODE-FN SELECTOR EMITTERS (conv0 / conv1 epilogue closures).
 *
 *  The Python passes the per-slot requant transforms as closures over (b, spec,
 *  c_m0/c_m0b/c_m1, zero_f). In C the scatter / fuse phases pass a selector enum
 *  (rocke_gfx1151_dfcp_code_fn_t) and apply the matching transform to each acc
 *  slot. The constants the f16 closures capture (spec.m0/m0b/m1 + 0.0) are
 *  recomputed here from ctx->spec so the emitted IR matches the closures.
 * ===================================================================== */

/* conv0_code_i8(p0) -- the native-int and fp16 conv0 requant (Python 2919-2931
 * single-tile; 2786-2789 persistent native branch is the native arm of this). */
static rocke_value_t* rocke_gfx1151_dfcp_conv0_code_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                       rocke_value_t* p0)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;

    if(spec->native_int)
    {
        /* q0 = _quant_i8_shift(b, p0, 4)            # m0 = 1/16
         * q0r = _relu_i32(b, b.sext(q0, I32))
         * return _quant_i4_shift(b, q0r, 1)         # m0b = 1/2 */
        rocke_value_t* q0 = rocke_gfx1151_dfcp_quant_i8_shift(b, p0, 4);
        rocke_value_t* q0r = rocke_gfx1151_dfcp_relu_i32(b, rocke_b_sext(b, q0, rocke_i32()));
        return rocke_gfx1151_dfcp_quant_i4_shift(b, q0r, 1);
    }
    else
    {
        /* p0_f32 = b.rint_f32(p0)
         * q0 = _quant_i8(b, p0_f32, c_m0)
         * q0r = b.fmax(_i8_to_f32(b, q0), zero_f)   # ReLU
         * return _quant_i4(b, q0r, c_m0b)
         * c_m0/c_m0b/zero_f are the body-hoisted consts (ctx) created once at
         * the Python closure-definition point, not per-slot. */
        rocke_value_t* c_m0 = ctx->c_m0;
        rocke_value_t* c_m0b = ctx->c_m0b;
        rocke_value_t* zero_f = ctx->zero_f;
        rocke_value_t* p0_f32 = rocke_b_rint_f32(b, p0);
        rocke_value_t* q0 = rocke_gfx1151_dfcp_quant_i8(b, p0_f32, c_m0);
        rocke_value_t* q0r = rocke_b_fmax(b, rocke_gfx1151_dfcp_i8_to_f32(b, q0), zero_f);
        return rocke_gfx1151_dfcp_quant_i4(b, q0r, c_m0b);
    }
}

/* conv0_code_f16(p0) = b.trunc_f32_to_f16(_i8_to_f32(b, conv0_code_i8(p0)))
 * (Python 2933-2934). */
static rocke_value_t* rocke_gfx1151_dfcp_conv0_code_f16(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                        rocke_value_t* p0)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* code = rocke_gfx1151_dfcp_conv0_code_i8(ctx, p0);
    return rocke_b_trunc_f32_to_f16(b, rocke_gfx1151_dfcp_i8_to_f32(b, code));
}

/* conv0_code_vec_i8(acc) -- specialized_rne whole-accumulator conv0 requant
 * (Python 2936-2940):
 *     q0 = _quant_i8_shift_vec_i32(b, acc, 4)   # m0 = 1/16
 *     q0r = _relu_i32_vec(b, q0)
 *     q0b = _quant_i4_shift_vec_i32(b, q0r, 1)  # m0b = 1/2
 *     return b.vector_trunc(q0b, I8) */
static rocke_value_t* rocke_gfx1151_dfcp_conv0_code_vec_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                           rocke_value_t* acc)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* q0 = rocke_gfx1151_dfcp_quant_i8_shift_vec_i32(b, acc, 4);
    rocke_value_t* q0r = rocke_gfx1151_dfcp_relu_i32_vec(b, q0);
    rocke_value_t* q0b = rocke_gfx1151_dfcp_quant_i4_shift_vec_i32(b, q0r, 1);
    return rocke_b_vector_trunc(b, q0b, rocke_i8());
}

/* conv1_code_i8(p1) -- native-int + fp16 conv1 requant (Python 3051-3058;
 * 2791-2793 persistent native branch is the native arm of this). */
static rocke_value_t* rocke_gfx1151_dfcp_conv1_code_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                       rocke_value_t* p1)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec = ctx->spec;

    if(spec->native_int)
    {
        /* q1 = _quant_i4_shift(b, p1, 2)            # m1 = 1/4
         * return b.trunc(_relu_i32(b, b.sext(q1, I32)), I8) */
        rocke_value_t* q1 = rocke_gfx1151_dfcp_quant_i4_shift(b, p1, 2);
        return rocke_b_trunc(
            b, rocke_gfx1151_dfcp_relu_i32(b, rocke_b_sext(b, q1, rocke_i32())), rocke_i8());
    }
    else
    {
        /* p1_f32 = b.rint_f32(p1)
         * q1 = _quant_i4(b, p1_f32, c_m1)
         * q1r = b.fmax(_i8_to_f32(b, q1), zero_f)   # ReLU
         * return b.cvt_f32_to_i8_sat(q1r)
         * c_m1 is the body-hoisted const created after the conv1 GEMM; zero_f is
         * the SAME body const captured by the conv0 closure (Python 2917). */
        rocke_value_t* c_m1 = ctx->c_m1;
        rocke_value_t* zero_f = ctx->zero_f;
        rocke_value_t* p1_f32 = rocke_b_rint_f32(b, p1);
        rocke_value_t* q1 = rocke_gfx1151_dfcp_quant_i4(b, p1_f32, c_m1);
        rocke_value_t* q1r = rocke_b_fmax(b, rocke_gfx1151_dfcp_i8_to_f32(b, q1), zero_f);
        return rocke_b_cvt_f32_to_i8_sat(b, q1r);
    }
}

/* conv1_code_f16(p1) = b.trunc_f32_to_f16(_i8_to_f32(b, conv1_code_i8(p1)))
 * (Python 3060-3061). */
static rocke_value_t* rocke_gfx1151_dfcp_conv1_code_f16(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                        rocke_value_t* p1)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* code = rocke_gfx1151_dfcp_conv1_code_i8(ctx, p1);
    return rocke_b_trunc_f32_to_f16(b, rocke_gfx1151_dfcp_i8_to_f32(b, code));
}

/* conv1_code_vec_i8(acc) -- specialized_rne whole-accumulator conv1 requant
 * (Python 3063-3066):
 *     q1 = _quant_i4_shift_vec_i32(b, acc, 2)   # m1 = 1/4
 *     q1r = _relu_i32_vec(b, q1)
 *     return b.vector_trunc(q1r, I8) */
static rocke_value_t* rocke_gfx1151_dfcp_conv1_code_vec_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                           rocke_value_t* acc)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* q1 = rocke_gfx1151_dfcp_quant_i4_shift_vec_i32(b, acc, 2);
    rocke_value_t* q1r = rocke_gfx1151_dfcp_relu_i32_vec(b, q1);
    return rocke_b_vector_trunc(b, q1r, rocke_i8());
}

/* Apply a scalar conv0/conv1 code function to one acc-slot Value. */
rocke_value_t* rocke_gfx1151_dfcp_apply_code_fn(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                int code_fn,
                                                rocke_value_t* slot)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    switch((rocke_gfx1151_dfcp_code_fn_t)code_fn)
    {
    case ROCKE_GFX1151_DFCP_CODE_CONV0_I8:
        return rocke_gfx1151_dfcp_conv0_code_i8(ctx, slot);
    case ROCKE_GFX1151_DFCP_CODE_CONV0_F16:
        return rocke_gfx1151_dfcp_conv0_code_f16(ctx, slot);
    case ROCKE_GFX1151_DFCP_CODE_CONV1_I8:
        return rocke_gfx1151_dfcp_conv1_code_i8(ctx, slot);
    case ROCKE_GFX1151_DFCP_CODE_CONV1_F16:
        return rocke_gfx1151_dfcp_conv1_code_f16(ctx, slot);
    case ROCKE_GFX1151_DFCP_CODE_CONV0_VEC_I8:
    case ROCKE_GFX1151_DFCP_CODE_CONV1_VEC_I8:
    default:
        /* vector code functions belong to apply_vec_code_fn; a scalar request for
         * one is a builder programming error (matches the Python TypeError that
         * would surface if a vec closure were called with a scalar slot). */
        return (rocke_value_t*)rocke_i_set_err(
            ctx->b, ROCKE_ERR_TYPE, "gfx1151 dfcp: vector code fn %d applied as scalar", code_fn);
    }
}

/* Apply a vector (whole-accumulator) code function. */
rocke_value_t* rocke_gfx1151_dfcp_apply_vec_code_fn(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    int code_fn,
                                                    rocke_value_t* acc)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    switch((rocke_gfx1151_dfcp_code_fn_t)code_fn)
    {
    case ROCKE_GFX1151_DFCP_CODE_CONV0_VEC_I8:
        return rocke_gfx1151_dfcp_conv0_code_vec_i8(ctx, acc);
    case ROCKE_GFX1151_DFCP_CODE_CONV1_VEC_I8:
        return rocke_gfx1151_dfcp_conv1_code_vec_i8(ctx, acc);
    default:
        return (rocke_value_t*)rocke_i_set_err(
            ctx->b, ROCKE_ERR_TYPE, "gfx1151 dfcp: scalar code fn %d applied as vector", code_fn);
    }
}

/* ===================================================================== *
 *  rocke_gfx1151_dfcp_build_ctx_init -- the Python build_deep_fused_conv_pool
 *  prologue (lines 2688-2778).
 * ===================================================================== */
rocke_status_t
    rocke_gfx1151_dfcp_build_ctx_init(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                      rocke_ir_builder_t* b,
                                      const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
                                      const char* arch)
{
    const rocke_fused_conv_pool_problem_t* p;
    const rocke_conv_problem_t* c;
    const rocke_archtarget_t* target;
    char reason[ROCKE_ERR_MSG_CAP];
    int a0_kind;
    int c0_cols;
    int w1_cols;
    int foot_h;
    int foot_w;

    if(ctx == NULL || b == NULL || spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    memset(ctx, 0, sizeof(*ctx));

    /* ok, why = is_valid_spec(spec, arch=arch)
     * if not ok: raise ValueError(f"invalid gfx1151 deep fused conv/pool spec: {why}") */
    if(!rocke_gfx1151_deep_fused_conv_pool_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid gfx1151 deep fused conv/pool spec: %s", reason);
        return ROCKE_ERR_VALUE;
    }

    /* (A) build-time constants ------------------------------------------- */
    ctx->b = b;
    ctx->spec = spec;
    /* arch NULL => "gfx1151" (mirrors the default-arg). */
    ctx->arch = (arch != NULL) ? arch : ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_ARCH;

    /* p = spec.problem; c = p.conv; kpad = spec.kpad */
    p = &spec->problem;
    c = &p->conv;
    ctx->p = p;
    ctx->c = c;
    ctx->kpad = rocke_gfx1151_dfcp_kpad(spec);

    /* target = ArchTarget.from_gfx(arch)
     * op = target.mma.op_for_shape(family="wmma", a="f16", b="f16", c="fp32",
     *                              m=_WMMA, n=_WMMA, k=_WMMA) */
    target = rocke_archtarget_from_gfx(ctx->arch);
    if(target == NULL)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_KEY, "unknown gfx target '%s'", ctx->arch);
        return ROCKE_ERR_KEY;
    }
    ctx->op = rocke_archtarget_op_for_shape(target,
                                            "wmma",
                                            "f16",
                                            "f16",
                                            "fp32",
                                            ROCKE_GFX1151_DFCP_WMMA,
                                            ROCKE_GFX1151_DFCP_WMMA,
                                            ROCKE_GFX1151_DFCP_WMMA);

    /* op0 = target.mma.by_op_id(_OP_ID_IU8) if spec.native_int else op
     * op1 = target.mma.by_op_id(_OP_ID_IU4) if spec.native_int else op */
    if(spec->native_int)
    {
        ctx->op0 = rocke_archtarget_by_op_id(target, ROCKE_GFX1151_DFCP_OP_ID_IU8);
        ctx->op1 = rocke_archtarget_by_op_id(target, ROCKE_GFX1151_DFCP_OP_ID_IU4);
    }
    else
    {
        ctx->op0 = ctx->op;
        ctx->op1 = ctx->op;
    }

    /* ---- kernel params (X / W0 / Y / W1) -------------------------------- *
     * X = b.param("X", PtrType(I8 ,"global"), noalias, readonly, align16)
     * W0= b.param("W0",PtrType(I8 ,"global"), noalias, readonly, align16)
     * Y = b.param("Y", PtrType(I32,"global"), noalias, writeonly,align16)
     * W1= b.param("W1",PtrType(I8 ,"global"), noalias, readonly, align16) */
    {
        rocke_param_opts_t ro;
        rocke_param_opts_t wo;
        const rocke_type_t* ptr_i8 = rocke_ptr_type(b, rocke_i8(), "global");
        const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");

        memset(&ro, 0, sizeof(ro));
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;

        memset(&wo, 0, sizeof(wo));
        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;

        ctx->X = rocke_b_param(b, "X", ptr_i8, &ro);
        ctx->W0 = rocke_b_param(b, "W0", ptr_i8, &ro);
        ctx->Y = rocke_b_param(b, "Y", ptr_i32, &wo);
        ctx->W1 = rocke_b_param(b, "W1", ptr_i8, &ro);
    }

    /* grid = WarpGrid.from_atom(op, tile_m, tile_n, tile_k=_WMMA, warp_m, warp_n,
     *            wave_size=_WAVE).bind(b, block_m_axis="y", block_n_axis="x")
     *
     * from_atom pins warp_tile_{m,n,k} from the fp16 WMMA op (16x16x16 = _WMMA),
     * tile_k = _WMMA, wave_size = _WAVE (=32). bind then emits the exact same
     * const/thread_id/mod/div/mul SSA sequence as WarpGrid.bind (mirrors the
     * conv_implicit_gemm grouped glue port). The grid lives in the builder arena
     * so ctx->grid stays valid for the whole build. */
    {
        rocke_warp_grid_t* grid = (rocke_warp_grid_t*)rocke_arena_alloc(&b->arena, sizeof(*grid));
        rocke_value_t* wave;
        rocke_value_t* c_warps_n;
        rocke_value_t* c_warps_n_warp_m;
        rocke_value_t* c_tile_m;
        rocke_value_t* c_tile_n;
        rocke_value_t* c_tile_k;
        rocke_value_t* tid_v;
        rocke_value_t* lane_v;
        rocke_value_t* warp_id_v;
        rocke_value_t* warp_m_idx_v;
        rocke_value_t* warp_n_idx_v;
        rocke_value_t* warp_k_idx_v;
        rocke_value_t* block_m_off_v;
        rocke_value_t* block_n_off_v;
        rocke_value_t* block_k_off_v;

        if(grid == NULL)
        {
            return ROCKE_ERR_OOM;
        }
        memset(grid, 0, sizeof(*grid));

        /* from_atom: compile-time geometry. */
        grid->tile_m = spec->tile_m;
        grid->tile_n = spec->tile_n;
        grid->tile_k = ROCKE_GFX1151_DFCP_WMMA; /* tile_k=_WMMA */
        grid->warp_m = spec->warp_m;
        grid->warp_n = spec->warp_n;
        grid->warp_k = 1;
        grid->warp_tile_m = rocke_gfx1151_dfcp_warp_tile_m(spec); /* op.m = _WMMA */
        grid->warp_tile_n = rocke_gfx1151_dfcp_warp_tile_n(spec); /* op.n = _WMMA */
        grid->warp_tile_k = rocke_gfx1151_dfcp_warp_tile_k(spec); /* op.k = _WMMA */
        grid->wave_size = rocke_gfx1151_dfcp_block_size(spec) / (spec->warp_m * spec->warp_n);

        /* bind: b.kernel.attrs["max_workgroup_size"] = self.block_size */
        if(b->kernel != NULL)
        {
            rocke_attr_set_int(
                b, &b->kernel->attrs, "max_workgroup_size", rocke_warp_grid_block_size(grid));
        }

        /* bind: const stage (emitted in this exact order). c_warps_n_warp_m and
         * c_tile_k are emitted by bind even though warp_k==1 / block_k_axis==None
         * leave them unused -- kept for byte-identity with the Python SSA stream. */
        wave = rocke_b_const_i32(b, grid->wave_size);
        c_warps_n = rocke_b_const_i32(b, grid->warp_n);
        c_warps_n_warp_m = rocke_b_const_i32(b, grid->warp_n * grid->warp_m);
        c_tile_m = rocke_b_const_i32(b, grid->tile_m);
        c_tile_n = rocke_b_const_i32(b, grid->tile_n);
        c_tile_k = rocke_b_const_i32(b, grid->tile_k);
        (void)c_warps_n_warp_m;
        (void)c_tile_k;

        tid_v = rocke_b_thread_id_x(b);
        lane_v = rocke_b_mod(b, tid_v, wave);
        warp_id_v = rocke_b_div(b, tid_v, wave);

        /* warp_k == 1 path */
        warp_m_idx_v = rocke_b_div(b, warp_id_v, c_warps_n);
        warp_n_idx_v = rocke_b_mod(b, warp_id_v, c_warps_n);
        warp_k_idx_v = rocke_b_const_i32(b, 0);

        /* block_m_axis="y", block_n_axis="x"; block_k_axis=None. */
        block_m_off_v = rocke_b_mul(b, rocke_b_block_id_y(b), c_tile_m);
        block_n_off_v = rocke_b_mul(b, rocke_b_block_id_x(b), c_tile_n);
        block_k_off_v = rocke_b_const_i32(b, 0);

        grid->tid = tid_v;
        grid->lane = lane_v;
        grid->warp_id = warp_id_v;
        grid->warp_m_idx = warp_m_idx_v;
        grid->warp_n_idx = warp_n_idx_v;
        grid->warp_k_idx = warp_k_idx_v;
        grid->block_m_off = block_m_off_v;
        grid->block_n_off = block_n_off_v;
        grid->block_k_off = block_k_off_v;

        ctx->grid = grid;
    }

    /* L1: occupancy launch bound.
     *   if spec.waves_per_eu:
     *       b.kernel.attrs["max_workgroup_size"] = spec.block_size
     *       b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu */
    if(spec->waves_per_eu)
    {
        rocke_attr_set_int(
            b, &b->kernel->attrs, "max_workgroup_size", rocke_gfx1151_dfcp_block_size(spec));
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    /* L2: policy = SchedulePolicy.for_pipeline(spec.sched_policy).
     * The phases treat the policy opaquely (passed to emit_wmma_k_sched). Stage a
     * builder-arena copy and hand its pointer through ctx->policy ("mem" leaves a
     * no-op policy, never NULL, mirroring the Python object). */
    {
        rocke_schedule_policy_t* pol
            = (rocke_schedule_policy_t*)rocke_arena_alloc(&b->arena, sizeof(*pol));
        if(pol == NULL)
        {
            return ROCKE_ERR_OOM;
        }
        *pol = rocke_schedule_policy_for_pipeline(b, spec->sched_policy);
        ctx->policy = pol;
    }

    /* ---- LDS dtype resolution (Python a0/c0/w1/c1 dtype locals) --------- *
     * a0_dtype = I8 if native_int else F16
     * c0_dtype = w1_dtype = c1_dtype = I8 if native_int else F16
     * Stored as the ckc scalar-kind tag. */
    a0_kind = spec->native_int ? ROCKE_SCALAR_I8 : ROCKE_SCALAR_F16;
    ctx->a0_dtype = a0_kind;
    ctx->c0_dtype = a0_kind;
    ctx->w1_dtype = a0_kind;
    ctx->c1_dtype = a0_kind;

    /* ---- LDS tile allocations (Python lines 2745-2778) ----------------- */
    {
        const rocke_type_t* a0_t = spec->native_int ? rocke_i8() : rocke_f16();
        const rocke_type_t* c0_t = a0_t;
        const rocke_type_t* w1_t = a0_t;
        const rocke_type_t* c1_t = a0_t;

        foot_h = rocke_gfx1151_dfcp_foot_h(spec);
        foot_w = rocke_gfx1151_dfcp_foot_w(spec);

        /* if spec.direct_conv0:
         *     a0_smem = b.smem_alloc(a0_dtype, [foot_h*foot_w, c.C], "INP_smem")
         * else:
         *     a0_smem = b.smem_alloc(a0_dtype, [tile_m, kpad], "A0_smem") */
        if(spec->direct_conv0)
        {
            int shape[2];
            shape[0] = foot_h * foot_w;
            shape[1] = c->C;
            ctx->a0_smem = rocke_b_smem_alloc(b, a0_t, shape, 2, "INP_smem");
        }
        else
        {
            int shape[2];
            shape[0] = spec->tile_m;
            shape[1] = ctx->kpad;
            ctx->a0_smem = rocke_b_smem_alloc(b, a0_t, shape, 2, "A0_smem");
        }

        /* w0_smem = b.smem_alloc(a0_dtype, [tile_n, kpad], "W0_smem") */
        {
            int shape[2];
            shape[0] = spec->tile_n;
            shape[1] = ctx->kpad;
            ctx->w0_smem = rocke_b_smem_alloc(b, a0_t, shape, 2, "W0_smem");
        }

        /* c0_cols = c.K // 2 if packed_c0_handoff else c.K
         * c0_smem = None if fused_c0a1
         *           else b.smem_alloc(c0_dtype, [tile_m, c0_cols], "C0_smem") */
        c0_cols = spec->packed_c0_handoff ? (c->K / 2) : c->K;
        if(spec->fused_c0a1)
        {
            ctx->c0_smem = NULL;
        }
        else
        {
            int shape[2];
            shape[0] = spec->tile_m;
            shape[1] = c0_cols;
            ctx->c0_smem = rocke_b_smem_alloc(b, c0_t, shape, 2, "C0_smem");
        }

        /* c0_packed_smem = b.smem_alloc(I8, [tile_m, c.K // 2], "C0pk_smem")
         *                  if repack_c0 else None */
        if(spec->repack_c0)
        {
            int shape[2];
            shape[0] = spec->tile_m;
            shape[1] = c->K / 2;
            ctx->c0_packed_smem = rocke_b_smem_alloc(b, rocke_i8(), shape, 2, "C0pk_smem");
        }
        else
        {
            ctx->c0_packed_smem = NULL;
        }

        /* if native_int: w1_cols = c.K if conv1_int8 else c.K // 2
         * else:          w1_cols = c.K
         * w1_smem = b.smem_alloc(w1_dtype, [tile_n, w1_cols], "W1_smem") */
        if(spec->native_int)
        {
            w1_cols = spec->conv1_int8 ? c->K : (c->K / 2);
        }
        else
        {
            w1_cols = c->K;
        }
        {
            int shape[2];
            shape[0] = spec->tile_n;
            shape[1] = w1_cols;
            ctx->w1_smem = rocke_b_smem_alloc(b, w1_t, shape, 2, "W1_smem");
        }

        /* c1_smem = b.smem_alloc(c1_dtype, [tile_m, tile_n], "C1_smem") */
        {
            int shape[2];
            shape[0] = spec->tile_m;
            shape[1] = spec->tile_n;
            ctx->c1_smem = rocke_b_smem_alloc(b, c1_t, shape, 2, "C1_smem");
        }
    }

    /* (B) accs / a_frags / h_blk / w_blk left zeroed for the phase walk. */

    return rocke_ir_builder_ok(b) ? ROCKE_OK : rocke_ir_builder_status(b);
}
