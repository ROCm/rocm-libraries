// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_tiled_common.cpp -- emission helpers shared by the gfx942 and
 * gfx950 forward attention kernels. Mirrors
 * library/kernels/common/_attention_tiled_shared.py.
 *
 * The two 32x32 C helpers reproduce the Python builder-call sequence
 * byte-faithfully: same div/mod constants, same calculate_x ys/ps wiring, same
 * trailing add for the N-tile base. C32_DIST (built once at Python module
 * import from make_static_tile_distribution(make_c_warp_dstr_encoding(
 * MfmaAtom.f16_32x32x16()))) is reproduced as a lazily-built cached
 * distribution off rocke_mfma_atom("f16", 32, 32, 16).
 *
 * Lifetime: every IR node is arena-owned (rocke_ir_builder_t.arena). The cached
 * C32_DIST is built on the first call's builder arena. Because the C
 * builder/arena is freed at the end of each build, the cache is per-build:
 * rocke_attn2d_c32_dist_reset() clears it at every build entry so build N+1
 * never reads build N's freed arena.
 */

#include <stdarg.h>
#include <stdio.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, rocke_make_c_warp_dstr_encoding */
#include "rocke/helper_rocke.helpers.distribution.h" /* rocke_tile_distribution_t, calculate_x */
#include "rocke/instance_attention_tiled_common_internal.h"
#include "rocke/ir.h"

/* ------------------------------------------------------------- error latch */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing
 * `return (T*)rocke_attn_tiled_set_err(...)` call sites valid -- the cast/return is
 * simply never reached. */
[[noreturn]] static void*
    rocke_attn_tiled_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

/* ------------------------------------------------------ _C32_DIST (cached) */
/* make_static_tile_distribution(make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16()))
 * -- a host-side distribution the Python caches at module scope. Built lazily on
 * the first 32x32-C-helper call of a build.
 *
 * Re-entrancy: the cached rocke_tile_distribution_t (and all its inner nodes) is
 * arena-allocated off the *current build's* rocke_ir_builder. When that builder is
 * freed at the end of a build, this pointer dangles; reusing it on the next
 * build feeds freed memory into calculate_x. So the cache is per-build, not
 * process-lifetime: rocke_attn2d_c32_dist_reset() clears it at each build entry
 * (see the gfx942/gfx950 public entries). */
static const rocke_tile_distribution_t* g_c32_dist = NULL;

/* Re-entrancy reset: drop the dangling per-build cache before a new build. */
void rocke_attn2d_c32_dist_reset(void)
{
    g_c32_dist = NULL;
}

static const rocke_tile_distribution_t* rocke_attn2d_c32_dist(rocke_ir_builder_t* b)
{
    const rocke_mfma_atom_t* atom;
    const rocke_tile_distribution_encoding_t* enc;
    const rocke_tile_distribution_t* dist;

    if(g_c32_dist != NULL)
    {
        return g_c32_dist;
    }
    if(b == NULL)
    {
        return NULL;
    }

    atom = rocke_mfma_atom("f16", 32, 32, 16);
    if(atom == NULL)
    {
        rocke_attn_tiled_set_err(b, ROCKE_ERR_VALUE, "_C32_DIST: no f16 32x32x16 MFMA atom");
        return NULL;
    }
    enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return NULL;
    }
    dist = rocke_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return NULL;
    }
    g_c32_dist = dist;
    return g_c32_dist;
}

/* ------------------------------------------------ _mfma_32x32_c_row / _col */

rocke_value_t* rocke__mfma_32x32_c_row(rocke_ir_builder_t* b, rocke_value_t* lane, int elem_idx)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];

    if(b == NULL)
    {
        return NULL;
    }
    /* if not (0 <= elem_idx < 16): raise ValueError */
    if(!(elem_idx >= 0 && elem_idx < 16))
    {
        return (rocke_value_t*)rocke_attn_tiled_set_err(
            b, ROCKE_ERR_VALUE, "mfma_32x32x16 elem_idx must be 0..15, got %d", elem_idx);
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* y0 = const(elem_idx // 4); y1 = const(elem_idx % 4) */
    ys[0] = rocke_b_const_i32(b, (int64_t)(elem_idx / 4));
    ys[1] = rocke_b_const_i32(b, (int64_t)(elem_idx % 4));

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* row, _col = _C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row */
}

rocke_value_t* rocke__mfma_32x32_c_col(rocke_ir_builder_t* b, rocke_value_t* lane, int n_tile32)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];
    rocke_value_t* col;

    if(b == NULL)
    {
        return NULL;
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* ys=[const(0), const(0)] */
    ys[0] = rocke_b_const_i32(b, 0);
    ys[1] = rocke_b_const_i32(b, 0);

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* _row, col = _C32_DIST.calculate_x(b, ys=[0, 0], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    col = out_x[1];

    /* if n_tile32 == 0: return col; else return add(const(n_tile32*32), col) */
    if(n_tile32 == 0)
    {
        return col;
    }
    return rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n_tile32 * 32)), col);
}
