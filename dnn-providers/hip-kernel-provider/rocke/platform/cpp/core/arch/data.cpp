// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * arch_target_arch_target_data.c -- bucket 0 of the C99 port of
 * rocke.core.arch.target: the frozen SSOT data, the lane-coord IR emitters,
 * and the shared lookup tables/helpers declared in
 * rocke/arch_target_internal.h.
 *
 * This file is a faithful, byte-identical translation of:
 *   - the _mfma / _wmma lane-coord closures                -> static rocke_lane_coord_fn
 *   - LayoutMap.coord                                      -> rocke_layout_map_coord
 *   - _MMA_FRAGMENT_INFO + _build_mma_op (precomputed)     -> per-op_id static
 *                                                            rocke_layout_map_t +
 *                                                            rocke_mma_op_t[] catalogs
 *   - the embedded core/arch/data/arch_specs.json          -> static
 *                                                            rocke_arch_target_t
 *                                                            singletons + registry
 *
 * The lane-coord emitters reproduce EXACTLY the index arithmetic of the Python
 * closures (same op order, same constants) so the emitted IR is byte-identical.
 * All catalog/target descriptors are static read-only storage (program-lifetime
 * borrows, mirroring the Python @lru_cache singletons); the only arena/IR use is
 * inside the layout-map emitters, which create Values in the supplied builder.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target_internal.h"
#include "rocke/wmma_scale_internal.h"

/* =========================================================================
 * lane-coord emitters (static rocke_lane_coord_fn)
 * =========================================================================
 *
 * Each fn reproduces, op-for-op, the Python closure of the same name, emitting
 * arith.* through the builder. On a failed/NULL builder the public rocke_b_*
 * helpers are sticky no-ops returning NULL; we additionally short-circuit so the
 * out params are left NULL (matching the "no-op leaving NULLs" contract).
 */

#define ROCKE_ATI_COORD_GUARD(b, out0, out1)       \
    do                                             \
    {                                              \
        if(out0)                                   \
            *(out0) = NULL;                        \
        if(out1)                                   \
            *(out1) = NULL;                        \
        if((b) == NULL || (b)->status != ROCKE_OK) \
            return;                                \
    } while(0)

/* MFMA 16x16 accumulator: slot i -> (m_blk*4 + i, lane % 16). */
static void _mfma_acc_16x16(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c16, *n_in_atom, *m_blk, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    n_in_atom = rocke_b_mod(b, lane, c16);
    m_blk = rocke_b_div(b, lane, c16);
    /* row = b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* row_mul = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
        row = rocke_b_add(b, row_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 32x32 accumulator:
 *   row = (slot//4)*8 + (lane//32)*4 + (slot%4), col = lane % 32. */
static void _mfma_acc_32x32(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c32, *n_in_atom, *m_blk, *row;
    int rb, ri;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    n_in_atom = rocke_b_mod(b, lane, c32);
    m_blk = rocke_b_div(b, lane, c32);
    rb = slot / 4;
    ri = slot % 4;
    /* row = b.add(b.add(b.const_i32(rb*8), b.mul(m_blk, b.const_i32(4))),
     *             b.const_i32(ri))
     * Python evaluates each add left-to-right: const(rb*8) first, then the mul
     * (const 4 + mul), then the inner add, then const(ri), then the outer add.
     * C call-arg order is unspecified (GCC: right-to-left); sequence into temps
     * to pin Python source-order. */
    {
        rocke_value_t* c_rb8 = rocke_b_const_i32(b, rb * 8);
        rocke_value_t* blk_mul = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
        rocke_value_t* inner = rocke_b_add(b, c_rb8, blk_mul);
        row = rocke_b_add(b, inner, rocke_b_const_i32(b, ri));
    }
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 16x16x16 A operand: (lane % 16, k_blk*4 + slot). */
static void _mfma_a_16x16(rocke_ir_builder_t* b,
                          rocke_value_t* lane,
                          int slot,
                          rocke_value_t** out0,
                          rocke_value_t** out1)
{
    rocke_value_t *c16, *m_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    m_in_atom = rocke_b_mod(b, lane, c16);
    k_blk = rocke_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 4));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 16x16x16 B operand: (k_blk*4 + slot, lane % 16). */
static void _mfma_b_16x16(rocke_ir_builder_t* b,
                          rocke_value_t* lane,
                          int slot,
                          rocke_value_t** out0,
                          rocke_value_t** out1)
{
    rocke_value_t *c16, *n_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    n_in_atom = rocke_b_mod(b, lane, c16);
    k_blk = rocke_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 4));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 32x32x8 A operand: (lane % 32, k_blk*4 + slot). */
static void _mfma_a_32x32x8(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c32, *m_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    m_in_atom = rocke_b_mod(b, lane, c32);
    k_blk = rocke_b_div(b, lane, c32);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 4));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 32x32x8 B operand: (k_blk*4 + slot, lane % 32). */
static void _mfma_b_32x32x8(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c32, *n_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    n_in_atom = rocke_b_mod(b, lane, c32);
    k_blk = rocke_b_div(b, lane, c32);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 4));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 16x16x32 A operand: (lane % 16, k_blk*8 + slot). K-packed CDNA3 atom;
 * the lane's <8 x half> covers the contiguous block K=[k_blk*8 : k_blk*8+8]
 * (per MfmaAtom.f16_16x16x32, hardware-verified to 1e-3; the flat-concat
 * alternative validates only to 1e-2). */
static void _mfma_a_16x16x32(rocke_ir_builder_t* b,
                             rocke_value_t* lane,
                             int slot,
                             rocke_value_t** out0,
                             rocke_value_t** out1)
{
    rocke_value_t *c16, *m_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    m_in_atom = rocke_b_mod(b, lane, c16);
    k_blk = rocke_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_blk, b.const_i32(8)), b.const_i32(slot)). Pin the mul
     * (with its const 8) before const(slot) to match Python's left-to-right
     * arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 16x16x32 B operand: (k_blk*8 + slot, lane % 16). */
static void _mfma_b_16x16x32(rocke_ir_builder_t* b,
                             rocke_value_t* lane,
                             int slot,
                             rocke_value_t** out0,
                             rocke_value_t** out1)
{
    rocke_value_t *c16, *n_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    n_in_atom = rocke_b_mod(b, lane, c16);
    k_blk = rocke_b_div(b, lane, c16);
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 32x32x16 A operand: (lane % 32, k_blk*8 + slot). The K=16 sibling of
 * _mfma_a_32x32x8; same CDNA3 K-packing (contiguous 8-K block per k_blk) as
 * the verified 16x16x32 atom (per MfmaAtom.f16_32x32x16). */
static void _mfma_a_32x32x16(rocke_ir_builder_t* b,
                             rocke_value_t* lane,
                             int slot,
                             rocke_value_t** out0,
                             rocke_value_t** out1)
{
    rocke_value_t *c32, *m_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    m_in_atom = rocke_b_mod(b, lane, c32);
    k_blk = rocke_b_div(b, lane, c32);
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 32x32x16 B operand: (k_blk*8 + slot, lane % 32). */
static void _mfma_b_32x32x16(rocke_ir_builder_t* b,
                             rocke_value_t* lane,
                             int slot,
                             rocke_value_t** out0,
                             rocke_value_t** out1)
{
    rocke_value_t *c32, *n_in_atom, *k_blk, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    n_in_atom = rocke_b_mod(b, lane, c32);
    k_blk = rocke_b_div(b, lane, c32);
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 16x16x4 fp32 / f64 A operand (#8348): one scalar per lane.
 * (lane % 16, lane // 16); slot is always 0 (a_frag_len == 1). */
static void _mfma_a_16x16x4(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c16, *m_in_atom, *k;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    m_in_atom = rocke_b_mod(b, lane, c16);
    k = rocke_b_div(b, lane, c16);
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 16x16x4 fp32 / f64 B operand (#8348): one scalar per lane.
 * (lane // 16, lane % 16); slot is always 0 (b_frag_len == 1). */
static void _mfma_b_16x16x4(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c16, *n_in_atom, *k;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    n_in_atom = rocke_b_mod(b, lane, c16);
    k = rocke_b_div(b, lane, c16);
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 32x32x2 fp32 A operand (#8348): scalar float per lane.
 * (lane % 32, lane // 32); slot is always 0 (a_frag_len == 1). */
static void _mfma_a_32x32x2_f32(rocke_ir_builder_t* b,
                                rocke_value_t* lane,
                                int slot,
                                rocke_value_t** out0,
                                rocke_value_t** out1)
{
    rocke_value_t *c32, *m_in_atom, *k;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    m_in_atom = rocke_b_mod(b, lane, c32);
    k = rocke_b_div(b, lane, c32);
    if(out0)
        *out0 = m_in_atom;
    if(out1)
        *out1 = k;
}

/* MFMA 32x32x2 fp32 B operand (#8348): scalar float per lane.
 * (lane // 32, lane % 32); slot is always 0 (b_frag_len == 1). */
static void _mfma_b_32x32x2_f32(rocke_ir_builder_t* b,
                                rocke_value_t* lane,
                                int slot,
                                rocke_value_t** out0,
                                rocke_value_t** out1)
{
    rocke_value_t *c32, *n_in_atom, *k;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c32 = rocke_b_const_i32(b, 32);
    n_in_atom = rocke_b_mod(b, lane, c32);
    k = rocke_b_div(b, lane, c32);
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 16x16x4 f64 accumulator: slot i -> (i*4 + lane // 16, lane % 16). */
static void _mfma_acc_16x16x4_f64(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int slot,
                                  rocke_value_t** out0,
                                  rocke_value_t** out1)
{
    rocke_value_t *c16, *n_in_atom, *m_blk, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    n_in_atom = rocke_b_mod(b, lane, c16);
    m_blk = rocke_b_div(b, lane, c16);
    row = rocke_b_add(b, rocke_b_const_i32(b, slot * 4), m_blk);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = n_in_atom;
}

/* MFMA 4x4x4 f64 B operand (4 blocks stacked along rows, one scalar per lane):
 * lane 16*k + 4*blk + n -> (blk*4 + k, n). */
static void _mfma_b_4x4x4_f64(rocke_ir_builder_t* b,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_value_t *c4, *n_in_blk, *blk, *k, *row;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c4 = rocke_b_const_i32(b, 4);
    n_in_blk = rocke_b_mod(b, lane, c4);
    blk = rocke_b_mod(b, rocke_b_div(b, lane, c4), c4);
    k = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    row = rocke_b_add(b, rocke_b_mul(b, blk, c4), k);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = n_in_blk;
}

/* MFMA 4x4x4 f64 accumulator (4 blocks stacked along rows, one scalar per lane):
 * lane 16*m + 4*blk + n -> (blk*4 + m, n). */
static void _mfma_acc_4x4x4_f64(rocke_ir_builder_t* b,
                                rocke_value_t* lane,
                                int slot,
                                rocke_value_t** out0,
                                rocke_value_t** out1)
{
    rocke_value_t *c4, *n_in_blk, *blk, *m, *row;
    (void)slot;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c4 = rocke_b_const_i32(b, 4);
    n_in_blk = rocke_b_mod(b, lane, c4);
    blk = rocke_b_mod(b, rocke_b_div(b, lane, c4), c4);
    m = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    row = rocke_b_add(b, rocke_b_mul(b, blk, c4), m);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = n_in_blk;
}

/* WMMA 16x16x16 accumulator (wave32): (2*slot + lane//16, lane % 16). */
static void _wmma_acc_16x16(rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    rocke_value_t *c16, *col, *half, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    half = rocke_b_div(b, lane, c16);
    row = rocke_b_add(b, rocke_b_const_i32(b, 2 * slot), half);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = col;
}

/* WMMA 16x16x16 A operand (wave32): (lane % 16, K=slot). */
static void _wmma_a_16x16(rocke_ir_builder_t* b,
                          rocke_value_t* lane,
                          int slot,
                          rocke_value_t** out0,
                          rocke_value_t** out1)
{
    rocke_value_t *c16, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    row = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = rocke_b_const_i32(b, slot);
}

/* WMMA 16x16x16 B operand (wave32): (K=slot, lane % 16). */
static void _wmma_b_16x16(rocke_ir_builder_t* b,
                          rocke_value_t* lane,
                          int slot,
                          rocke_value_t** out0,
                          rocke_value_t** out1)
{
    rocke_value_t *c16, *col;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = rocke_b_const_i32(b, slot);
    if(out1)
        *out1 = col;
}

/* iu8 WMMA A operand (wave32): (lane % 16, k_base = 4*slot). */
static void _wmma_a_16x16_iu8(rocke_ir_builder_t* b,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_value_t *c16, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    row = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = rocke_b_const_i32(b, 4 * slot);
}

/* iu8 WMMA B operand (wave32): (k_base = 4*slot, lane % 16). */
static void _wmma_b_16x16_iu8(rocke_ir_builder_t* b,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_value_t *c16, *col;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = rocke_b_const_i32(b, 4 * slot);
    if(out1)
        *out1 = col;
}

/* iu4 WMMA A operand (wave32): (lane % 16, k_base = 8*slot). */
static void _wmma_a_16x16_iu4(rocke_ir_builder_t* b,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_value_t *c16, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    row = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = rocke_b_const_i32(b, 8 * slot);
}

/* iu4 WMMA B operand (wave32): (k_base = 8*slot, lane % 16). */
static void _wmma_b_16x16_iu4(rocke_ir_builder_t* b,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_value_t *c16, *col;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    if(out0)
        *out0 = rocke_b_const_i32(b, 8 * slot);
    if(out1)
        *out1 = col;
}

/* RDNA4 WMMA 16x16x16 accumulator (wave32): ((lane//16)*8 + slot, lane % 16). */
static void _wmma_gfx12_acc_16x16(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int slot,
                                  rocke_value_t** out0,
                                  rocke_value_t** out1)
{
    rocke_value_t *c16, *col, *half, *row;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    half = rocke_b_div(b, lane, c16);
    /* row = b.add(b.mul(half, b.const_i32(8)), b.const_i32(slot)). Python
     * evaluates the add's args left-to-right (the mul, with its const 8, emits
     * BEFORE const(slot)). C call-arg order is unspecified (GCC: right-to-left),
     * which would emit const(slot) first and shift later SSA names. Hoist the
     * mul into a temp to pin Python source-order. */
    {
        rocke_value_t* row_mul = rocke_b_mul(b, half, rocke_b_const_i32(b, 8));
        row = rocke_b_add(b, row_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = col;
}

/* RDNA4 WMMA 16x16x16 A operand (wave32): (lane % 16, (lane//16)*8 + slot). */
static void _wmma_gfx12_a_16x16(rocke_ir_builder_t* b,
                                rocke_value_t* lane,
                                int slot,
                                rocke_value_t** out0,
                                rocke_value_t** out1)
{
    rocke_value_t *c16, *row, *k_half, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    row = rocke_b_mod(b, lane, c16);
    k_half = rocke_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_half, b.const_i32(8)), b.const_i32(slot)). Pin the mul
     * (with its const 8) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_half, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = k;
}

/* RDNA4 WMMA 16x16x16 B operand (wave32): ((lane//16)*8 + slot, lane % 16). */
static void _wmma_gfx12_b_16x16(rocke_ir_builder_t* b,
                                rocke_value_t* lane,
                                int slot,
                                rocke_value_t** out0,
                                rocke_value_t** out1)
{
    rocke_value_t *c16, *col, *k_half, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    k_half = rocke_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_half, b.const_i32(8)), b.const_i32(slot)). Pin the mul
     * (with its const 8) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_half, rocke_b_const_i32(b, 8));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = col;
}

/* =========================================================================
 * LayoutMap.coord -> rocke_layout_map_coord
 * =========================================================================*/

bool rocke_layout_map_coord(const rocke_layout_map_t* m,
                            rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1)
{
    if(out0)
        *out0 = NULL;
    if(out1)
        *out1 = NULL;
    if(m == NULL || m->fn == NULL)
    {
        return false;
    }
    /* Python LayoutMap.coord: ValueError if not (0 <= slot < frag_len). */
    if(!(slot >= 0 && slot < m->frag_len))
    {
        if(b != NULL)
        {
            const char* role_txt = (m->role == ROCKE_MMA_ROLE_ACC)       ? "acc"
                                   : (m->role == ROCKE_MMA_ROLE_A)       ? "a"
                                   : (m->role == ROCKE_MMA_ROLE_B)       ? "b"
                                   : (m->role == ROCKE_MMA_ROLE_A_SCALE) ? "a_scale"
                                                                         : "b_scale";
            b->status = ROCKE_ERR_VALUE;
            snprintf(b->err,
                     ROCKE_ERR_MSG_CAP,
                     "fragment slot %d out of range [0, %d) for '%s' layout map",
                     slot,
                     m->frag_len,
                     role_txt);
        }
        return false;
    }
    m->fn(b, lane, slot, out0, out1);
    if(b != NULL && b->status != ROCKE_OK)
    {
        return false;
    }
    return true;
}

/* Scale slots describe logical K groups; both half-waves duplicate them. */
static void _wmma_gfx1250_a_scale(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int slot,
                                  rocke_value_t** out0,
                                  rocke_value_t** out1)
{
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* row = rocke_b_mod(b, lane, c16);
    rocke_value_t* group = rocke_b_const_i32(b, slot);
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = group;
}

static void _wmma_gfx1250_b_scale(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int slot,
                                  rocke_value_t** out0,
                                  rocke_value_t** out1)
{
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* col = rocke_b_mod(b, lane, c16);
    rocke_value_t* group = rocke_b_const_i32(b, slot);
    if(out0)
        *out0 = group;
    if(out1)
        *out1 = col;
}

/* =========================================================================
 * Per-op_id LayoutMap statics (the precomputed _build_mma_op output)
 * =========================================================================
 *
 * In Python _build_mma_op constructs, for each catalog row, up to three
 * LayoutMap instances from the op_id's _FragInfo (role + frag_len + wave_size +
 * fn). Because the maps depend ONLY on op_id (not on the per-arch row), we define
 * one static LayoutMap triplet per op_id here and point every arch's catalog rows
 * at them. A map is omitted (left NULL) when the _FragInfo has no fn for that
 * role or a zero frag_len (the `_mk(...) is None` case).
 *
 * Naming: lm_<opidkey>_{a,b,c}. Only op_ids with at least one verified map need a
 * struct; the rest carry NULL layout pointers in the catalog directly.
 */

static const rocke_layout_map_t lm_wmma_scale_k32_a
    = {ROCKE_MMA_ROLE_A_SCALE, 4, 32, _wmma_gfx1250_a_scale};
static const rocke_layout_map_t lm_wmma_scale_k32_b
    = {ROCKE_MMA_ROLE_B_SCALE, 4, 32, _wmma_gfx1250_b_scale};
static const rocke_layout_map_t lm_wmma_scale_k16_a
    = {ROCKE_MMA_ROLE_A_SCALE, 8, 32, _wmma_gfx1250_a_scale};
static const rocke_layout_map_t lm_wmma_scale_k16_b
    = {ROCKE_MMA_ROLE_B_SCALE, 8, 32, _wmma_gfx1250_b_scale};

/* --- mfma_f32_16x16x16_f16 / _bf16: a/b/c all present (frag 4/4/4, wave64) --- */
static const rocke_layout_map_t lm_mfma_16x16x16_a = {ROCKE_MMA_ROLE_A, 4, 64, _mfma_a_16x16};
static const rocke_layout_map_t lm_mfma_16x16x16_b = {ROCKE_MMA_ROLE_B, 4, 64, _mfma_b_16x16};
static const rocke_layout_map_t lm_mfma_16x16x16_c = {ROCKE_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16};

/* --- mfma_f32_16x16x32_{f16,bf16,fp8,bf8}: a/b/c present (frag 8/8/4, wave64).
 * fp8/bf8 share the f16 operand lane layout (same a_per_lane=8, K-packing). --- */
static const rocke_layout_map_t lm_mfma_16x16x32_a = {ROCKE_MMA_ROLE_A, 8, 64, _mfma_a_16x16x32};
static const rocke_layout_map_t lm_mfma_16x16x32_b = {ROCKE_MMA_ROLE_B, 8, 64, _mfma_b_16x16x32};
static const rocke_layout_map_t lm_mfma_16x16x32_c = {ROCKE_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16};

/* --- mfma_f32_32x32x8_f16: a/b/c present (frag 4/4/16, wave64) --- */
static const rocke_layout_map_t lm_mfma_32x32x8_a = {ROCKE_MMA_ROLE_A, 4, 64, _mfma_a_32x32x8};
static const rocke_layout_map_t lm_mfma_32x32x8_b = {ROCKE_MMA_ROLE_B, 4, 64, _mfma_b_32x32x8};
static const rocke_layout_map_t lm_mfma_32x32x8_c = {ROCKE_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};

/* --- mfma_f32_32x32x16_{f16,bf16,fp8,bf8}: a/b/c present (frag 8/8/16, wave64).
 * fp8/bf8 share the f16 operand lane layout (same a_per_lane=8, K-packing). --- */
static const rocke_layout_map_t lm_mfma_32x32x16_a = {ROCKE_MMA_ROLE_A, 8, 64, _mfma_a_32x32x16};
static const rocke_layout_map_t lm_mfma_32x32x16_b = {ROCKE_MMA_ROLE_B, 8, 64, _mfma_b_32x32x16};
static const rocke_layout_map_t lm_mfma_32x32x16_c = {ROCKE_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};

static void _mfma_a_16x16x8_xf32(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** c0, rocke_value_t** c1)
{
    rocke_value_t* c = rocke_b_const_i32(b, 16);
    rocke_value_t* axis = rocke_b_mod(b, lane, c);
    rocke_value_t* group = rocke_b_div(b, lane, c);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* base = rocke_b_mul(b, group, two);
    rocke_value_t* index = rocke_b_const_i32(b, slot);
    rocke_value_t* k = rocke_b_add(b, base, index);
    *c0 = axis;
    *c1 = k;
}

static void _mfma_b_16x16x8_xf32(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** c0, rocke_value_t** c1)
{
    rocke_value_t* c = rocke_b_const_i32(b, 16);
    rocke_value_t* axis = rocke_b_mod(b, lane, c);
    rocke_value_t* group = rocke_b_div(b, lane, c);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* base = rocke_b_mul(b, group, two);
    rocke_value_t* index = rocke_b_const_i32(b, slot);
    rocke_value_t* k = rocke_b_add(b, base, index);
    *c0 = k;
    *c1 = axis;
}

static void _mfma_a_32x32x4_xf32(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** c0, rocke_value_t** c1)
{
    rocke_value_t* c = rocke_b_const_i32(b, 32);
    rocke_value_t* axis = rocke_b_mod(b, lane, c);
    rocke_value_t* group = rocke_b_div(b, lane, c);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* base = rocke_b_mul(b, group, two);
    rocke_value_t* index = rocke_b_const_i32(b, slot);
    rocke_value_t* k = rocke_b_add(b, base, index);
    *c0 = axis;
    *c1 = k;
}

static void _mfma_b_32x32x4_xf32(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** c0, rocke_value_t** c1)
{
    rocke_value_t* c = rocke_b_const_i32(b, 32);
    rocke_value_t* axis = rocke_b_mod(b, lane, c);
    rocke_value_t* group = rocke_b_div(b, lane, c);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* base = rocke_b_mul(b, group, two);
    rocke_value_t* index = rocke_b_const_i32(b, slot);
    rocke_value_t* k = rocke_b_add(b, base, index);
    *c0 = k;
    *c1 = axis;
}

static const rocke_layout_map_t lm_xf32_16_a = {ROCKE_MMA_ROLE_A, 2, 64, _mfma_a_16x16x8_xf32};
static const rocke_layout_map_t lm_xf32_16_b = {ROCKE_MMA_ROLE_B, 2, 64, _mfma_b_16x16x8_xf32};
static const rocke_layout_map_t lm_xf32_32_a = {ROCKE_MMA_ROLE_A, 2, 64, _mfma_a_32x32x4_xf32};
static const rocke_layout_map_t lm_xf32_32_b = {ROCKE_MMA_ROLE_B, 2, 64, _mfma_b_32x32x4_xf32};

/* --- mfma_f32_16x16x4_f32: a/b/c present (frag 1/1/4, wave64) (#8348) --- */
static const rocke_layout_map_t lm_mfma_16x16x4_a = {ROCKE_MMA_ROLE_A, 1, 64, _mfma_a_16x16x4};
static const rocke_layout_map_t lm_mfma_16x16x4_b = {ROCKE_MMA_ROLE_B, 1, 64, _mfma_b_16x16x4};
static const rocke_layout_map_t lm_mfma_16x16x4_f32_c
    = {ROCKE_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16};

/* --- mfma_f32_32x32x2_f32: a/b/c present (frag 1/1/16, wave64) (#8348) --- */
static const rocke_layout_map_t lm_mfma_32x32x2_f32_a
    = {ROCKE_MMA_ROLE_A, 1, 64, _mfma_a_32x32x2_f32};
static const rocke_layout_map_t lm_mfma_32x32x2_f32_b
    = {ROCKE_MMA_ROLE_B, 1, 64, _mfma_b_32x32x2_f32};
static const rocke_layout_map_t lm_mfma_32x32x2_f32_c
    = {ROCKE_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};

/* --- mfma_f64_16x16x4_f64: a/b shared with fp32 16x16x4, own c (frag 1/1/4, wave64) --- */
static const rocke_layout_map_t lm_mfma_16x16x4_f64_c
    = {ROCKE_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16x4_f64};

/* --- mfma_f64_4x4x4_f64: a shared with 16x16x4 (frag 1/1/1, wave64) --- */
static const rocke_layout_map_t lm_mfma_4x4x4_f64_b = {ROCKE_MMA_ROLE_B, 1, 64, _mfma_b_4x4x4_f64};
static const rocke_layout_map_t lm_mfma_4x4x4_f64_c
    = {ROCKE_MMA_ROLE_ACC, 1, 64, _mfma_acc_4x4x4_f64};

/* --- wmma_f32_16x16x16_{f16,bf16}: a/b/c present (frag 16/16/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_16x16x16_a = {ROCKE_MMA_ROLE_A, 16, 32, _wmma_a_16x16};
static const rocke_layout_map_t lm_wmma_16x16x16_b = {ROCKE_MMA_ROLE_B, 16, 32, _wmma_b_16x16};
static const rocke_layout_map_t lm_wmma_16x16x16_c = {ROCKE_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_i32_16x16x16_iu8: a/b/c present (frag 4/4/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_iu8_a = {ROCKE_MMA_ROLE_A, 4, 32, _wmma_a_16x16_iu8};
static const rocke_layout_map_t lm_wmma_iu8_b = {ROCKE_MMA_ROLE_B, 4, 32, _wmma_b_16x16_iu8};
static const rocke_layout_map_t lm_wmma_iu8_c = {ROCKE_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_i32_16x16x16_iu4: a/b/c present (frag 2/2/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_iu4_a = {ROCKE_MMA_ROLE_A, 2, 32, _wmma_a_16x16_iu4};
static const rocke_layout_map_t lm_wmma_iu4_b = {ROCKE_MMA_ROLE_B, 2, 32, _wmma_b_16x16_iu4};
static const rocke_layout_map_t lm_wmma_iu4_c = {ROCKE_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_gfx12_f32_16x16x16_{f16,bf16}: a/b/c present (frag 8/8/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_gfx12_a = {ROCKE_MMA_ROLE_A, 8, 32, _wmma_gfx12_a_16x16};
static const rocke_layout_map_t lm_wmma_gfx12_b = {ROCKE_MMA_ROLE_B, 8, 32, _wmma_gfx12_b_16x16};
static const rocke_layout_map_t lm_wmma_gfx12_c
    = {ROCKE_MMA_ROLE_ACC, 8, 32, _wmma_gfx12_acc_16x16};

/* gfx1250 WMMA 16x16x4 fp32 A operand (wave32): kABKLane=2, kAK1PerLane=2.
 * row = lane / 2  (M index 0..15);  K = (lane % 2) * 2 + slot  (slot in {0,1}).
 * Mirrors _wmma_gfx1250_a_16x16x4_f32 in target.py. */
static void _wmma_gfx1250_a_16x16x4_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* lane,
                                        int slot,
                                        rocke_value_t** out0,
                                        rocke_value_t** out1)
{
    rocke_value_t *c2, *row, *k_lane, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c2 = rocke_b_const_i32(b, 2);
    row = rocke_b_div(b, lane, c2);
    k_lane = rocke_b_mod(b, lane, c2);
    k = rocke_b_add(b, rocke_b_mul(b, k_lane, c2), rocke_b_const_i32(b, slot));
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = k;
}

/* gfx1250 WMMA 16x16x4 fp32 B operand (wave32): col = lane / 2, K = (lane % 2) * 2 + slot. */
static void _wmma_gfx1250_b_16x16x4_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* lane,
                                        int slot,
                                        rocke_value_t** out0,
                                        rocke_value_t** out1)
{
    rocke_value_t *c2, *col, *k_lane, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c2 = rocke_b_const_i32(b, 2);
    col = rocke_b_div(b, lane, c2);
    k_lane = rocke_b_mod(b, lane, c2);
    k = rocke_b_add(b, rocke_b_mul(b, k_lane, c2), rocke_b_const_i32(b, slot));
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = col;
}

/* --- wmma_gfx1250_f32_16x16x4_f32: a/b/c present (frag 2/2/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_gfx1250_f32_a
    = {ROCKE_MMA_ROLE_A, 2, 32, _wmma_gfx1250_a_16x16x4_f32};
static const rocke_layout_map_t lm_wmma_gfx1250_f32_b
    = {ROCKE_MMA_ROLE_B, 2, 32, _wmma_gfx1250_b_16x16x4_f32};

/* gfx1250 WMMA 16x16x32 A operand (wave32): row = lane % 16,
 * k = (lane / 16) * 16 + slot  (slot in 0..15, a_frag_len=16).
 * Mirrors _wmma_gfx1250_a_16x16x32 in target.py. */
static void _wmma_gfx1250_a_16x16x32(rocke_ir_builder_t* b,
                                     rocke_value_t* lane,
                                     int slot,
                                     rocke_value_t** out0,
                                     rocke_value_t** out1)
{
    rocke_value_t *c16, *row, *k_half, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    row = rocke_b_mod(b, lane, c16);
    k_half = rocke_b_div(b, lane, c16);
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_half, rocke_b_const_i32(b, 16));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = row;
    if(out1)
        *out1 = k;
}

/* gfx1250 WMMA 16x16x32 B operand (wave32): k = (lane / 16) * 16 + slot,
 * col = lane % 16  (slot in 0..15, b_frag_len=16).
 * Mirrors _wmma_gfx1250_b_16x16x32 in target.py. */
static void _wmma_gfx1250_b_16x16x32(rocke_ir_builder_t* b,
                                     rocke_value_t* lane,
                                     int slot,
                                     rocke_value_t** out0,
                                     rocke_value_t** out1)
{
    rocke_value_t *c16, *col, *k_half, *k;
    ROCKE_ATI_COORD_GUARD(b, out0, out1);
    c16 = rocke_b_const_i32(b, 16);
    col = rocke_b_mod(b, lane, c16);
    k_half = rocke_b_div(b, lane, c16);
    {
        rocke_value_t* k_mul = rocke_b_mul(b, k_half, rocke_b_const_i32(b, 16));
        k = rocke_b_add(b, k_mul, rocke_b_const_i32(b, slot));
    }
    if(out0)
        *out0 = k;
    if(out1)
        *out1 = col;
}

/* --- wmma_gfx1250_f32_16x16x32_{f16,bf16}: a/b/c present (frag 16/16/8, wave32) --- */
static const rocke_layout_map_t lm_wmma_gfx1250_32_a
    = {ROCKE_MMA_ROLE_A, 16, 32, _wmma_gfx1250_a_16x16x32};
static const rocke_layout_map_t lm_wmma_gfx1250_32_b
    = {ROCKE_MMA_ROLE_B, 16, 32, _wmma_gfx1250_b_16x16x32};

/* =========================================================================
 * op_id -> accumulator fragment length (the c_frag_len projection of
 * target.py::_MMA_FRAGMENT_INFO).
 * =========================================================================
 *
 * This is the arch SSOT that rocke.core.ir.IRBuilder.mma (rocke_b_mma) consults
 * to size a tile.mma result vector from a bare op_id string; the ir bucket keeps
 * no private copy. It mirrors _MMA_FRAGMENT_INFO's c_frag_len column exactly.
 * Unscaled atoms are listed here; scaled gfx1250 atoms are resolved from the
 * generated catalog below, avoiding a second copy of their IDs.
 *
 * `ati` = arch-target internal (see arch_target_internal.h): the naming prefix
 * reserved for storage private to this arch-target port, kept out of the public
 * rocke_ / rocke_arch_ namespace. This table is file-local static (only the
 * rocke_arch_mma_c_frag_len lookup below reads it), so it is not extern'd in the
 * internal header like the cross-bucket rocke_ati_ symbols are. It holds
 * physical atoms only: an emulated atom's logical op_id has no row and resolves
 * through the physical atom it expands into.
 *
 * NOTE (drift): this is still a hand-maintained mirror of the Python
 * _MMA_FRAGMENT_INFO c_frag_len column, not generated from it.
 * TODO: codegen this table from the Python SSOT (_MMA_FRAGMENT_INFO) at build
 * time so the two engines share a single source of truth. Until then there is
 * no direct mirror-agreement test; each side's frag lengths are pinned
 * independently -- C via tests/core/mma_frag_ssot.cpp (rocke_b_mma) and Python
 * via tests/core/test_mma_frag_tables.py (_MMA_FRAGMENT_INFO) -- and the
 * byte-identity gate (tools/check_byte_identity.py) catches any cross-engine
 * c_frag_len drift, since a wrong length changes the emitted tile.mma result
 * vector width.
 */
typedef struct rocke_ati_mma_frag_row
{
    const char* op_id;
    int c_frag_len;
} rocke_ati_mma_frag_row_t;

/* Layout guard for the rocke_arch_mma_c_frag_len lookup below, which derives its
 * row count from sizeof(table)/sizeof(row). That division is exact for any
 * element size, so padding of rocke_ati_mma_frag_row_t (12 bytes of data padded
 * out to 16 for op_id's 8-byte alignment) cannot desync the count. These asserts
 * pin the field order so a future reorder/resize is caught at compile time. */
static_assert(offsetof(rocke_ati_mma_frag_row_t, op_id) == 0,
              "op_id must be the first field of rocke_ati_mma_frag_row_t");
static_assert(offsetof(rocke_ati_mma_frag_row_t, c_frag_len) == sizeof(const char*),
              "c_frag_len must immediately follow op_id (no leading padding)");

static const rocke_ati_mma_frag_row_t rocke_ati_mma_frag[] = {
    {"mfma_f32_32x32x4_xf32", 16},
    {"mfma_f32_16x16x8_xf32", 4},
    /* --- MFMA fp32 (wave64) --- */
    {"mfma_f32_16x16x4_f32", 4},
    {"mfma_f32_32x32x2_f32", 16},
    /* --- MFMA fp64 (wave64) --- */
    {"mfma_f64_16x16x4_f64", 4},
    {"mfma_f64_4x4x4_f64", 1},
    /* --- MFMA f16 (wave64) --- */
    {"mfma_f32_16x16x16_f16", 4},
    {"mfma_f32_16x16x32_f16", 4},
    {"mfma_f32_32x32x8_f16", 16},
    {"mfma_f32_32x32x16_f16", 16},
    {"mfma_f32_4x4x4_f16", 4},
    {"mfma_f32_4x4x4_bf16", 4},
    /* --- MFMA bf16 (wave64) --- */
    {"mfma_f32_16x16x16_bf16", 4},
    {"mfma_f32_16x16x32_bf16", 4},
    {"mfma_f32_32x32x8_bf16", 16},
    {"mfma_f32_32x32x16_bf16", 16},
    /* --- MFMA fp8 / bf8 (wave64) --- */
    {"mfma_f32_16x16x32_fp8", 4},
    {"mfma_f32_16x16x32_bf8", 4},
    {"mfma_f32_32x32x16_fp8", 16},
    {"mfma_f32_32x32x16_bf8", 16},
    /* --- MFMA MX (wave64), frag lengths only --- */
    {"mfma_f32_16x16x128_fp4", 4},
    {"mfma_f32_16x16x96_fp6", 4},
    {"mfma_f32_16x16x128_fp8", 4},
    {"mfma_scale_f32_16x16x128_f8f6f4", 4},
    /* --- WMMA f16 / bf16 (wave32, RDNA) --- */
    {"wmma_f32_16x16x16_f16", 8},
    {"wmma_f32_16x16x16_bf16", 8},
    /* --- WMMA iu8 / iu4 (wave32, RDNA3/3.5) --- */
    {"wmma_i32_16x16x16_iu8", 8},
    {"wmma_i32_16x16x16_iu4", 8},
    /* --- WMMA f16 / bf16 (wave32, RDNA4 / gfx12) --- */
    {"wmma_gfx12_f32_16x16x16_f16", 8},
    {"wmma_gfx12_f32_16x16x16_bf16", 8},
    /* --- WMMA fp32 / f16 / bf16 / fp8 / bf8 (wave32, gfx1250 CDNA) --- */
    {"wmma_gfx1250_f32_16x16x4_f32", 8},
    {"wmma_gfx1250_f32_16x16x32_f16", 8},
    {"wmma_gfx1250_f32_16x16x32_bf16", 8},
    {"wmma_gfx1250_f32_16x16x64_fp8_fp8", 8},
    {"wmma_gfx1250_f32_16x16x64_fp8_bf8", 8},
    {"wmma_gfx1250_f32_16x16x64_bf8_fp8", 8},
    {"wmma_gfx1250_f32_16x16x64_bf8_bf8", 8},
};

/* The frag-table row registered for op_id itself, or NULL. */
static const rocke_ati_mma_frag_row_t* rocke_ati_mma_frag_row_for(const char* op_id)
{
    size_t i;
    for(i = 0; i < sizeof(rocke_ati_mma_frag) / sizeof(rocke_ati_mma_frag[0]); ++i)
    {
        if(strcmp(rocke_ati_mma_frag[i].op_id, op_id) == 0)
        {
            return &rocke_ati_mma_frag[i];
        }
    }
    return NULL;
}

/* =========================================================================
 * Per-arch rocke_mma_op_t[] catalogs (the embedded arch_specs.json mma rows,
 * enriched with frag lengths + layout-map pointers, mirroring _build_mma_op).
 * =========================================================================
 *
 * Frag lengths come from _MMA_FRAGMENT_INFO[op_id]; op_ids absent from that
 * table carry (0,0,0,64) and NULL maps. dtype fields hold the normalised
 * (canonical) catalog keys from rocke_normalize_dtype, exactly as
 * normalize_dtype(o["a"/"b"/"c"]) would produce.
 *
 * Field order of rocke_mma_op_t:
 *   family, a_dtype, b_dtype, c_dtype, m, n, k, op_id,
 *   a_frag_len, b_frag_len, c_frag_len, wave_size, a_layout, b_layout, c_layout
 *
 * The trailing scale and emulation fields are left to zero-init below: every
 * atom here is unscaled unless it spells its scale metadata, and every one is
 * natively encoded (emulation_recipe NONE, see rocke_arch_mma_op_is_native). A
 * row that is neither must spell its fields positionally through to the last one it needs
 * -- C++ forbids mixing designated and positional initializers in one list.
 *
 * Low-bit spellings normalize to explicit exponent/mantissa catalog keys.
 */

/* ----------------------------- gfx90a (CDNA2) ---------------------------- */
/* MI200 (Aldebaran): wave64, 64 KB LDS, no async_lds, no fp8/bf8 native MFMA.
 * bf16 32x32x8 is hardware-supported (same MFMA atom dimensions as fp16 32x32x8;
 * bf16 arrived in CDNA2 with the same layout maps as fp16 for those shapes). */
static const rocke_mma_op_t k_mma_gfx90a[] = {
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_f16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_f16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_bf16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_bf16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
};

/* ----------------------------- gfx942 (CDNA) ----------------------------- */
static const rocke_mma_op_t k_mma_gfx942[] = {
    {"mma",
     "tf32",
     "tf32",
     "fp32",
     16,
     16,
     8,
     "mfma_f32_16x16x8_xf32",
     2,
     2,
     4,
     64,
     &lm_xf32_16_a,
     &lm_xf32_16_b,
     &lm_mfma_16x16x4_f32_c},
    {"mma",
     "tf32",
     "tf32",
     "fp32",
     32,
     32,
     4,
     "mfma_f32_32x32x4_xf32",
     2,
     2,
     16,
     64,
     &lm_xf32_32_a,
     &lm_xf32_32_b,
     &lm_mfma_32x32x2_f32_c},

    {"mma",
     "fp32",
     "fp32",
     "fp32",
     16,
     16,
     4,
     "mfma_f32_16x16x4_f32",
     1,
     1,
     4,
     64,
     &lm_mfma_16x16x4_a,
     &lm_mfma_16x16x4_b,
     &lm_mfma_16x16x4_f32_c},
    {"mma",
     "fp32",
     "fp32",
     "fp32",
     32,
     32,
     2,
     "mfma_f32_32x32x2_f32",
     1,
     1,
     16,
     64,
     &lm_mfma_32x32x2_f32_a,
     &lm_mfma_32x32x2_f32_b,
     &lm_mfma_32x32x2_f32_c},
    {"mma",
     "fp64",
     "fp64",
     "fp64",
     16,
     16,
     4,
     "mfma_f64_16x16x4_f64",
     1,
     1,
     4,
     64,
     &lm_mfma_16x16x4_a,
     &lm_mfma_16x16x4_b,
     &lm_mfma_16x16x4_f64_c},
    {"mma",
     "fp64",
     "fp64",
     "fp64",
     4,
     4,
     4,
     "mfma_f64_4x4x4_f64",
     1,
     1,
     1,
     64,
     &lm_mfma_16x16x4_a,
     &lm_mfma_4x4x4_f64_b,
     &lm_mfma_4x4x4_f64_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_f16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_f16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_bf16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_bf16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
    {"mma",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_fp8",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_fp8",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
    {"mma",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_bf8",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_bf8",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
};

/* ----------------------------- gfx950 (CDNA) ----------------------------- */
static void mfma_scaled_a(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** row, rocke_value_t** k)
{
    *row = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));
    *k = rocke_b_add(
        b,
        rocke_b_mul(b, rocke_b_div(b, lane, rocke_b_const_i32(b, 32)), rocke_b_const_i32(b, 32)),
        rocke_b_const_i32(b, slot));
}
static void mfma_scaled_f8_a(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** row, rocke_value_t** k)
{
    *row = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));
    *k = rocke_b_add(
        b,
        rocke_b_mul(b, rocke_b_div(b, lane, rocke_b_const_i32(b, 32)), rocke_b_const_i32(b, 16)),
        rocke_b_const_i32(b, (slot / 16) * 32 + slot % 16));
}
static void mfma_scaled_f8_b(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** k, rocke_value_t** col)
{
    mfma_scaled_f8_a(b, lane, slot, col, k);
}
static void mfma_scaled_b(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** k, rocke_value_t** col)
{
    mfma_scaled_a(b, lane, slot, col, k);
}
static void mfma_scaled_scale_a(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** row, rocke_value_t** k)
{
    (void)slot;
    *row = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));
    *k = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
}
static void mfma_scaled_scale_b(
    rocke_ir_builder_t* b, rocke_value_t* lane, int slot, rocke_value_t** k, rocke_value_t** col)
{
    mfma_scaled_scale_a(b, lane, slot, col, k);
}
static const rocke_layout_map_t lm_mfma_scaled_f8_a = {ROCKE_MMA_ROLE_A, 32, 64, mfma_scaled_f8_a};
static const rocke_layout_map_t lm_mfma_scaled_f8_b = {ROCKE_MMA_ROLE_B, 32, 64, mfma_scaled_f8_b};
static const rocke_layout_map_t lm_mfma_scaled_a = {ROCKE_MMA_ROLE_A, 32, 64, mfma_scaled_a};
static const rocke_layout_map_t lm_mfma_scaled_b = {ROCKE_MMA_ROLE_B, 32, 64, mfma_scaled_b};
static const rocke_layout_map_t lm_mfma_scaled_c = {ROCKE_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};
static const rocke_layout_map_t lm_mfma_scaled_sa
    = {ROCKE_MMA_ROLE_A_SCALE, 1, 64, mfma_scaled_scale_a};
static const rocke_layout_map_t lm_mfma_scaled_sb
    = {ROCKE_MMA_ROLE_B_SCALE, 1, 64, mfma_scaled_scale_b};

static const rocke_mma_op_t k_mma_gfx950[] = {
    {"mfma_scaled",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp8_fp8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp8e4m3",
     "bf8e5m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp8_bf8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp8e4m3",
     "fp6e2m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp8_fp6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp8e4m3",
     "fp6e3m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp8_bf6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp8e4m3",
     "fp4e2m1",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp8_fp4_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "bf8e5m2",
     "fp8e4m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf8_fp8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf8_bf8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "bf8e5m2",
     "fp6e2m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf8_fp6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "bf8e5m2",
     "fp6e3m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf8_bf6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "bf8e5m2",
     "fp4e2m1",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf8_fp4_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_f8_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e2m3",
     "fp8e4m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp6_fp8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e2m3",
     "bf8e5m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp6_bf8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e2m3",
     "fp6e2m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp6_fp6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e2m3",
     "fp6e3m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp6_bf6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e2m3",
     "fp4e2m1",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp6_fp4_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e3m2",
     "fp8e4m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf6_fp8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e3m2",
     "bf8e5m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf6_bf8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e3m2",
     "fp6e2m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf6_fp6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e3m2",
     "fp6e3m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf6_bf6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp6e3m2",
     "fp4e2m1",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_bf6_fp4_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp4e2m1",
     "fp8e4m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp4_fp8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp4e2m1",
     "bf8e5m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp4_bf8_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_f8_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp4e2m1",
     "fp6e2m3",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp4_fp6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp4e2m1",
     "fp6e3m2",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp4_bf6_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mfma_scaled",
     "fp4e2m1",
     "fp4e2m1",
     "fp32",
     32,
     32,
     64,
     "mfma_gfx950_f32_32x32x64_fp4_fp4_scale_e8m0_e8m0_k32",
     32,
     32,
     16,
     64,
     &lm_mfma_scaled_a,
     &lm_mfma_scaled_b,
     &lm_mfma_scaled_c,
     "e8m0",
     "e8m0",
     ROCKE_MMA_SCALE_K32,
     1,
     1,
     &lm_mfma_scaled_sa,
     &lm_mfma_scaled_sb},
    {"mma",
     "fp32",
     "fp32",
     "fp32",
     16,
     16,
     4,
     "mfma_f32_16x16x4_f32",
     1,
     1,
     4,
     64,
     &lm_mfma_16x16x4_a,
     &lm_mfma_16x16x4_b,
     &lm_mfma_16x16x4_f32_c},
    {"mma",
     "fp32",
     "fp32",
     "fp32",
     32,
     32,
     2,
     "mfma_f32_32x32x2_f32",
     1,
     1,
     16,
     64,
     &lm_mfma_32x32x2_f32_a,
     &lm_mfma_32x32x2_f32_b,
     &lm_mfma_32x32x2_f32_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_f16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_f16",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_f16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
    {"mma",
     "fp16",
     "fp16",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_f16",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "mfma_f32_16x16x16_bf16",
     4,
     4,
     4,
     64,
     &lm_mfma_16x16x16_a,
     &lm_mfma_16x16x16_b,
     &lm_mfma_16x16x16_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_bf16",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     32,
     32,
     8,
     "mfma_f32_32x32x8_bf16",
     4,
     4,
     16,
     64,
     &lm_mfma_32x32x8_a,
     &lm_mfma_32x32x8_b,
     &lm_mfma_32x32x8_c},
    {"mma",
     "bf16",
     "bf16",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_bf16",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
    {"mma",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_fp8",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_fp8",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
    {"mma",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_bf8",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c},
    {"mma",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_bf8",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c},
    /* Low-bit catalog keys use exponent/mantissa names; op_ids carry frag lengths
     * only (no verified maps), per _MMA_FRAGMENT_INFO. */
    {"mma",
     "fp4e2m1",
     "fp4e2m1",
     "fp32",
     16,
     16,
     128,
     "mfma_f32_16x16x128_fp4",
     16,
     16,
     4,
     64,
     NULL,
     NULL,
     NULL},
    {"mma",
     "fp6e2m3",
     "fp6e2m3",
     "fp32",
     16,
     16,
     96,
     "mfma_f32_16x16x96_fp6",
     12,
     12,
     4,
     64,
     NULL,
     NULL,
     NULL},
    /* TF32 via BF16x3 emulation. No native encoding: each row issues three
     * mfma_*_bf16 over a Dekker-style split of the fp32 payload of its TF32
     * operands. Frag lengths, wave size and layout maps are inherited verbatim
     * from the physical atom -- they are the *same* fragment, differing only in
     * the element type handed to the instruction -- and
     * tests/core/test_xf32_catalog_rows.py pins them equal to the bf16 rows
     * above so the two cannot drift.
     *
     * These two rows must be spelled fully positionally, through
     * emulation_physical_op_id: C++ forbids mixing designated and positional
     * initializers in one braced-init-list, so reaching a trailing member means
     * naming every member before it. The NULL/0/NONE run is the unscaled
     * default that every other row gets from zero-init. */
    {"mma",
     "tf32",
     "tf32",
     "fp32",
     16,
     16,
     32,
     "mfma_f32_16x16x32_xf32",
     8,
     8,
     4,
     64,
     &lm_mfma_16x16x32_a,
     &lm_mfma_16x16x32_b,
     &lm_mfma_16x16x32_c,
     NULL,
     NULL,
     ROCKE_MMA_SCALE_NONE,
     0,
     0,
     NULL,
     NULL,
     ROCKE_MMA_EMULATION_BF16X3,
     "mfma_f32_16x16x32_bf16"},
    {"mma",
     "tf32",
     "tf32",
     "fp32",
     32,
     32,
     16,
     "mfma_f32_32x32x16_xf32",
     8,
     8,
     16,
     64,
     &lm_mfma_32x32x16_a,
     &lm_mfma_32x32x16_b,
     &lm_mfma_32x32x16_c,
     NULL,
     NULL,
     ROCKE_MMA_SCALE_NONE,
     0,
     0,
     NULL,
     NULL,
     ROCKE_MMA_EMULATION_BF16X3,
     "mfma_f32_32x32x16_bf16"},
};

/* ----------------------------- gfx1151 (RDNA3.5) ------------------------- */
static const rocke_mma_op_t k_mma_gfx1151[] = {
    {"wmma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "wmma_f32_16x16x16_f16",
     16,
     16,
     8,
     32,
     &lm_wmma_16x16x16_a,
     &lm_wmma_16x16x16_b,
     &lm_wmma_16x16x16_c},
    {"wmma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "wmma_f32_16x16x16_bf16",
     16,
     16,
     8,
     32,
     &lm_wmma_16x16x16_a,
     &lm_wmma_16x16x16_b,
     &lm_wmma_16x16x16_c},
    {"wmma",
     "i8",
     "i8",
     "i32",
     16,
     16,
     16,
     "wmma_i32_16x16x16_iu8",
     4,
     4,
     8,
     32,
     &lm_wmma_iu8_a,
     &lm_wmma_iu8_b,
     &lm_wmma_iu8_c},
    {"wmma",
     "i4",
     "i4",
     "i32",
     16,
     16,
     16,
     "wmma_i32_16x16x16_iu4",
     2,
     2,
     8,
     32,
     &lm_wmma_iu4_a,
     &lm_wmma_iu4_b,
     &lm_wmma_iu4_c},
};

/* ----------------------------- gfx1201 (RDNA4) -------------------------- */
static const rocke_mma_op_t k_mma_gfx1201[] = {
    {"wmma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "wmma_gfx12_f32_16x16x16_f16",
     8,
     8,
     8,
     32,
     &lm_wmma_gfx12_a,
     &lm_wmma_gfx12_b,
     &lm_wmma_gfx12_c},
    {"wmma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "wmma_gfx12_f32_16x16x16_bf16",
     8,
     8,
     8,
     32,
     &lm_wmma_gfx12_a,
     &lm_wmma_gfx12_b,
     &lm_wmma_gfx12_c},
};

/* --------------------------- gfx11-generic (RDNA3) ---------------------- */
static const rocke_mma_op_t k_mma_gfx11_generic[] = {
    {"wmma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     16,
     "wmma_f32_16x16x16_f16",
     16,
     16,
     8,
     32,
     &lm_wmma_16x16x16_a,
     &lm_wmma_16x16x16_b,
     &lm_wmma_16x16x16_c},
    {"wmma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     16,
     "wmma_f32_16x16x16_bf16",
     16,
     16,
     8,
     32,
     &lm_wmma_16x16x16_a,
     &lm_wmma_16x16x16_b,
     &lm_wmma_16x16x16_c},
    {"wmma",
     "i8",
     "i8",
     "i32",
     16,
     16,
     16,
     "wmma_i32_16x16x16_iu8",
     4,
     4,
     8,
     32,
     &lm_wmma_iu8_a,
     &lm_wmma_iu8_b,
     &lm_wmma_iu8_c},
    {"wmma",
     "i4",
     "i4",
     "i32",
     16,
     16,
     16,
     "wmma_i32_16x16x16_iu4",
     2,
     2,
     8,
     32,
     &lm_wmma_iu4_a,
     &lm_wmma_iu4_b,
     &lm_wmma_iu4_c},
};

/* ----------------------------- gfx1250 (CDNA/GFX12) ---------------------- */
/* gfx1250: wave32, WMMA only (no MFMA). fp32 K=4 atom: A/B are scalar float
 * per lane (same lane layout as mfma_f32_16x16x4_f32 on CDNA — reuses
 * lm_mfma_16x16x4_{a,b}); accumulator is <8 x float> in the gfx12
 * column-distributed layout (lm_wmma_gfx12_c, c_frag_len=8).
 * Primary fp16/bf16 atom is 16x16x32 (K=32, not the gfx12 RDNA 16x16x16).
 * A/B frag_len=16 (16 fp16/bf16 elements per lane); A/B layout maps are NULL
 * pending the gfx1250 ISA backend port (rocke_ll_backend_for); the accumulator
 * reuses lm_wmma_gfx12_c (same 16x16 tile layout). FP8/BF8 K=64 atoms carry
 * frag_len=8 (8 i32 per lane) with NULL A/B maps. */
static constexpr rocke_mma_op_t k_mma_gfx1250_unscaled[] = {
    {"wmma",
     "fp32",
     "fp32",
     "fp32",
     16,
     16,
     4,
     "wmma_gfx1250_f32_16x16x4_f32",
     2,
     2,
     8,
     32,
     &lm_wmma_gfx1250_f32_a,
     &lm_wmma_gfx1250_f32_b,
     &lm_wmma_gfx12_c},
    {"wmma",
     "fp16",
     "fp16",
     "fp32",
     16,
     16,
     32,
     "wmma_gfx1250_f32_16x16x32_f16",
     16,
     16,
     8,
     32,
     &lm_wmma_gfx1250_32_a,
     &lm_wmma_gfx1250_32_b,
     &lm_wmma_gfx12_c},
    {"wmma",
     "bf16",
     "bf16",
     "fp32",
     16,
     16,
     32,
     "wmma_gfx1250_f32_16x16x32_bf16",
     16,
     16,
     8,
     32,
     &lm_wmma_gfx1250_32_a,
     &lm_wmma_gfx1250_32_b,
     &lm_wmma_gfx12_c},
    {"wmma",
     "fp8e4m3",
     "fp8e4m3",
     "fp32",
     16,
     16,
     64,
     "wmma_gfx1250_f32_16x16x64_fp8_fp8",
     8,
     8,
     8,
     32,
     NULL,
     NULL,
     &lm_wmma_gfx12_c},
    {"wmma",
     "fp8e4m3",
     "bf8e5m2",
     "fp32",
     16,
     16,
     64,
     "wmma_gfx1250_f32_16x16x64_fp8_bf8",
     8,
     8,
     8,
     32,
     NULL,
     NULL,
     &lm_wmma_gfx12_c},
    {"wmma",
     "bf8e5m2",
     "fp8e4m3",
     "fp32",
     16,
     16,
     64,
     "wmma_gfx1250_f32_16x16x64_bf8_fp8",
     8,
     8,
     8,
     32,
     NULL,
     NULL,
     &lm_wmma_gfx12_c},
    {"wmma",
     "bf8e5m2",
     "bf8e5m2",
     "fp32",
     16,
     16,
     64,
     "wmma_gfx1250_f32_16x16x64_bf8_bf8",
     8,
     8,
     8,
     32,
     NULL,
     NULL,
     &lm_wmma_gfx12_c},
};

/* Private PODs evaluated at compile time. IDs have separate final static
 * storage: copying the descriptor array cannot leave pointers into temporaries.
 * Keep the same staged order as _gfx1250_scaled_rows; no runtime initialization,
 * allocation, STL, JSON parsing, or Python dependency. */
namespace
{
constexpr int k_scaled_count = 86;
constexpr int k_unscaled_count = 7;
constexpr const char* k_inputs[] = {"fp8", "bf8", "fp6", "bf6", "fp4"};
constexpr const char* k_input_dtypes[] = {"fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"};
constexpr const char* k_scales[] = {"e8m0", "e5m3", "e4m3"};
constexpr int k_blocks[] = {32, 16};
// Native's original E8M0 seed order predates the JSON order; retain both.
constexpr int k_native_e8m0[][2]
    = {{0, 0}, {4, 0}, {2, 0}, {3, 0}, {1, 0}, {0, 1}, {4, 1}, {2, 1}, {3, 1}, {1, 1}};
struct scaled_key
{
    int a, b, sa, sb, mode;
};
struct scaled_keys
{
    scaled_key rows[k_scaled_count];
    int count;
};

consteval scaled_keys make_scaled_keys()
{
    scaled_keys result{};
    bool seen[2][5][5][3][3] = {};
    auto add = [&](int a, int b, int sa, int sb, int mode) {
        if(!rocke_scaled_wmma_formats_valid(a, b, sa, sb))
            return;
        if(!seen[mode][a][b][sa][sb])
        {
            if(result.count == k_scaled_count)
                throw "too many scaled MMA atoms";
            seen[mode][a][b][sa][sb] = true;
            result.rows[result.count++] = {a, b, sa, sb, mode};
        }
    };
    for(const auto& seed : k_native_e8m0)
        add(seed[0], seed[0], 0, 0, seed[1]);
    for(int mode = 0; mode < 2; ++mode)
        for(int a = 0; a < 5; ++a)
            for(int b = 0; b < 5; ++b)
                if(a != b)
                    add(a, b, 0, 0, mode);
    for(int mode = 0; mode < 2; ++mode)
    {
        for(int a = 0; a < 4; ++a)
            for(int sb = 1; sb < 3; ++sb)
                add(a, 4, 0, sb, mode);
        for(int b = 0; b < 4; ++b)
            for(int sa = 1; sa < 3; ++sa)
                add(4, b, sa, 0, mode);
        for(int scale = 1; scale < 3; ++scale)
            add(4, 4, scale, scale, mode);
    }
    return result;
}
constexpr auto k_scaled_keys = make_scaled_keys();
static_assert(k_scaled_keys.count == k_scaled_count);
static_assert(sizeof(k_mma_gfx1250_unscaled) / sizeof(k_mma_gfx1250_unscaled[0])
              == k_unscaled_count);

struct scaled_ids
{
    char rows[k_scaled_count][96];
};
consteval scaled_ids make_scaled_ids()
{
    scaled_ids result{};
    for(int i = 0; i < k_scaled_count; ++i)
    {
        const auto& key = k_scaled_keys.rows[i];
        int length = 0;
        auto append = [&](const char* text) {
            for(int j = 0; text[j]; ++j)
            {
                if(length >= 95)
                    throw "scaled MMA ID too long";
                result.rows[i][length++] = text[j];
            }
        };
        append("wmma_gfx1250_f32_16x16x128_");
        append(k_inputs[key.a]);
        append("_");
        append(k_inputs[key.b]);
        append("_scale_");
        append(k_scales[key.sa]);
        append("_");
        append(k_scales[key.sb]);
        append(key.mode == 0 ? "_k32" : "_k16");
    }
    return result;
}
constexpr auto k_scaled_ids = make_scaled_ids();

struct gfx1250_catalog
{
    rocke_mma_op_t ops[k_unscaled_count + k_scaled_count];
};
consteval gfx1250_catalog make_gfx1250_catalog()
{
    gfx1250_catalog result{};
    for(int i = 0; i < k_unscaled_count; ++i)
        result.ops[i] = k_mma_gfx1250_unscaled[i];
    for(int i = 0; i < k_scaled_count; ++i)
    {
        const auto& key = k_scaled_keys.rows[i];
        const bool k32 = key.mode == 0;
        result.ops[k_unscaled_count + i] = {
            "wmma_scaled",
            k_input_dtypes[key.a],
            k_input_dtypes[key.b],
            "fp32",
            16,
            16,
            128,
            k_scaled_ids.rows[i],
            16,
            16,
            8,
            32,
            NULL,
            NULL,
            &lm_wmma_gfx12_c,
            k_scales[key.sa],
            k_scales[key.sb],
            k32 ? ROCKE_MMA_SCALE_K32 : ROCKE_MMA_SCALE_K16,
            128 / k_blocks[key.mode],
            128 / k_blocks[key.mode],
            k32 ? &lm_wmma_scale_k32_a : &lm_wmma_scale_k16_a,
            k32 ? &lm_wmma_scale_k32_b : &lm_wmma_scale_k16_b,
        };
    }
    return result;
}
constexpr auto k_mma_gfx1250 = make_gfx1250_catalog();
} // namespace

int rocke_arch_mma_c_frag_len(const char* op_id)
{
    if(!op_id)
    {
        return 0;
    }
    const rocke_ati_mma_frag_row_t* row = rocke_ati_mma_frag_row_for(op_id);
    if(!row)
    {
        /* Mirrors target._frag_info: an emulated atom has no fragment of its
         * own and borrows the one belonging to the instruction it issues, so a
         * bare logical op_id resolves through its physical atom. One hop only
         * -- the catalog refuses an expansion chain. */
        const char* physical = rocke_arch_mma_op_id_physical_op_id(op_id);
        row = physical ? rocke_ati_mma_frag_row_for(physical) : NULL;
    }
    if(row)
    {
        return row->c_frag_len;
    }
    for(const auto& op : k_mma_gfx1250.ops)
    {
        if(strcmp(op.op_id, op_id) == 0)
            return op.c_frag_len;
    }
    /* An atom with no fragment row at all -- the gfx950 scaled MFMAs -- takes its
     * accumulator length from its catalog row. */
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx("gfx950");
    const rocke_mma_op_t* atom = rocke_mma_catalog_by_op_id(&target->mma, op_id);
    if(atom)
        return atom->c_frag_len;
    return 0; /* unknown atom: the zero-length _frag_info fallback */
}

/* =========================================================================
 * Per-arch rocke_arch_target_t singletons (the embedded arches[gfx] entries).
 * =========================================================================
 *
 * Field order of rocke_arch_target_t:
 *   gfx, family, target_family, wave_size, lds_capacity_bytes, vmcnt_bits,
 *   mma{ops,num_ops}, memory{has_async_lds,has_async_global_lds,has_ds_read_tr,has_tdm,
 *   buffer_load_max_dwords,async_lds_max_dwords,has_scalar_data_prefetch,
 *   has_global_prefetch,has_cluster_launch,has_multicast_load},
 *   limits{max_threads_per_block,vgprs,agprs,sgprs}
 */

#define K_NUM(arr) ((int)(sizeof(arr) / sizeof((arr)[0])))

static const rocke_arch_target_t k_target_gfx90a = {
    "gfx90a",
    "cdna",
    "gfx9_mfma",
    64,
    65536,
    6,
    {k_mma_gfx90a, K_NUM(k_mma_gfx90a)},
    {false, false, false, false, 4, 0, false, false, false, false},
    {1024, 512, 256, 102},
};

static const rocke_arch_target_t k_target_gfx942 = {
    "gfx942",
    "cdna",
    "gfx9_mfma",
    64,
    65536,
    6,
    {k_mma_gfx942, K_NUM(k_mma_gfx942)},
    {true, true, false, false, 4, 1, false, false, false, false},
    {1024, 512, 256, 102},
};

static const rocke_arch_target_t k_target_gfx950 = {
    "gfx950",
    "cdna",
    "gfx950",
    64,
    163840,
    6,
    {k_mma_gfx950, K_NUM(k_mma_gfx950)},
    {true, true, true, false, 4, 4, false, false, false, false},
    {1024, 512, 256, 102},
};

static const rocke_arch_target_t k_target_gfx1151 = {
    "gfx1151",
    "rdna",
    "gfx11_rdna",
    32,
    65536,
    6,
    {k_mma_gfx1151, K_NUM(k_mma_gfx1151)},
    {false, false, false, false, 4, 0, false, false, false, false},
    {1024, 256, 0, 106},
};

static const rocke_arch_target_t k_target_gfx1201 = {
    "gfx1201",
    "rdna",
    "gfx12_rdna",
    32,
    65536,
    6,
    {k_mma_gfx1201, K_NUM(k_mma_gfx1201)},
    {false, false, false, false, 4, 0, false, false, false, false},
    {1024, 256, 0, 106},
};

static const rocke_arch_target_t k_target_gfx1250 = {
    "gfx1250",
    "cdna",
    "gfx12_cdna",
    32,
    327680,
    6,
    {k_mma_gfx1250.ops, K_NUM(k_mma_gfx1250.ops)},
    {false, true, false, true, 4, 0, true, true, true, true},
    {1024, 256, 0, 106},
};

static const rocke_arch_target_t k_target_gfx11_generic = {
    "gfx11-generic",
    "rdna",
    "gfx11_rdna",
    32,
    65536,
    6,
    {k_mma_gfx11_generic, K_NUM(k_mma_gfx11_generic)},
    {false, false, false, false, 4, 0, false, false, false, false},
    {1024, 256, 0, 106},
};

/* gfx12-generic (RDNA4): the same WMMA atoms as gfx1201, so it shares that
 * catalog. */
static const rocke_arch_target_t k_target_gfx12_generic = {
    "gfx12-generic",
    "rdna",
    "gfx12_rdna",
    32,
    65536,
    6,
    {k_mma_gfx1201, K_NUM(k_mma_gfx1201)},
    {false, false, false, false, 4, 0, false, false, false, false},
    {1024, 256, 0, 106},
};

/* =========================================================================
 * Shared registry + known-arches (sorted by gfx token).
 * =========================================================================
 *
 * Python known_arches() returns tuple(sorted(specs)); sorted() on the gfx
 * strings yields: "gfx11-generic", "gfx1151", "gfx12-generic", "gfx1201",
 * "gfx1250", "gfx90a", "gfx942", "gfx950". ('-' (0x2D) < digits, so a generic
 * name sorts before its family's processors; '0' < '4' so "gfx90a" sorts
 * before "gfx942".)
 * The registry is kept in this same sorted order so rocke_known_arches can return
 * rocke_ati_known_arches directly.
 */

const rocke_ati_arch_row_t rocke_ati_arch_registry[] = {
    {"gfx11-generic", &k_target_gfx11_generic},
    {"gfx1151", &k_target_gfx1151},
    {"gfx12-generic", &k_target_gfx12_generic},
    {"gfx1201", &k_target_gfx1201},
    {"gfx1250", &k_target_gfx1250},
    {"gfx90a", &k_target_gfx90a},
    {"gfx942", &k_target_gfx942},
    {"gfx950", &k_target_gfx950},
    {NULL, NULL}, /* terminator */
};

const int rocke_ati_arch_registry_len = 8;

const char* const rocke_ati_known_arches[] = {
    "gfx11-generic",
    "gfx1151",
    "gfx12-generic",
    "gfx1201",
    "gfx1250",
    "gfx90a",
    "gfx942",
    "gfx950",
    NULL,
};

/* Mirror of arch_specs.json "generic_targets": each catalogued LLVM generic
 * processor followed by the concrete processors that load its code objects. */
static const char* const k_generic_gfx11_members[] = {
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
    "gfx1154",
    NULL,
};

static const char* const k_generic_gfx12_members[] = {
    "gfx1200",
    "gfx1201",
    NULL,
};

const rocke_ati_generic_row_t rocke_ati_generic_targets[] = {
    {"gfx11-generic", k_generic_gfx11_members},
    {"gfx12-generic", k_generic_gfx12_members},
    {NULL, NULL}, /* terminator */
};
