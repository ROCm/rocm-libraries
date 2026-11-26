/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_flex_mfma_internal.hpp -- PRIVATE helpers
 * shared by the C++ port of the gfx950 flex direct forward
 * (conv_direct_fwd_flex_mfma.cpp: spec, validation and the main builder;
 * conv_direct_fwd_flex_mfma_cs4.cpp / _cs2.cpp: the in-block split builders).
 * Declares no public API. The byte-identity contract of
 * rocke/instance_conv_direct_wgrad_internal.hpp applies: one emitting
 * argument per call, Python's evaluation order.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_INTERNAL_HPP
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_INTERNAL_HPP

#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "rocke/instance_conv_direct_fwd_flex_mfma.h"
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

namespace rocke_ffx
{

using rocke_dwg::Em;
using rocke_dwg::V;

constexpr int kCK = 64; /* channels per chunk */
constexpr int kXT = 4; /* 16-pixel MFMA tiles per wave */
constexpr int kPersist = 256; /* blocks of the persistent grid */
constexpr int kRot = 97; /* per-round rotation of the item walk */
constexpr int kZlPx = 64; /* zl: phase pixels of a light zero unit */
constexpr int kZfBits = 26; /* zf: dX pixel bits of the epilogue's LDS table */
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

/* ---- spec geometry (pure host helpers) ---- */

/* One output phase of a ds dgrad along one axis: (taps, input offset, filter
 * position of tap 0). */
struct Phase
{
    int t, pa, f0;
};

/* ds_strides(spec): (row, column) phases (h1: one row phase). */
std::pair<int, int> ds_strides(const rocke_fwd_flex_mfma_spec_t* s);
/* ds_phases(spec): per axis (rows, columns), the phases. */
std::vector<Phase> ds_phases_axis(const rocke_fwd_flex_mfma_spec_t* s, int axis);
/* ds_order(spec): the (a, b) phases the items run, in falling tap count. */
std::vector<std::pair<int, int>> ds_order(const rocke_fwd_flex_mfma_spec_t* s);
/* ds_zero_deltas(spec): zf, offsets to the tapless phases' pixels. */
std::vector<std::pair<int, int>> ds_zero_deltas(const rocke_fwd_flex_mfma_spec_t* s);
/* _tap_grid(spec): rows and columns of the taps a step walks. */
std::pair<int, int> tap_grid(const rocke_fwd_flex_mfma_spec_t* s);
/* _tab_taps(spec): taps of the DMA offset table, rounded up to 8. */
int tab_taps(const rocke_fwd_flex_mfma_spec_t* s);
/* _tab_bytes(spec): bytes of one offset table. */
long long tab_bytes(const rocke_fwd_flex_mfma_spec_t* s);
/* _tab_db(spec, arch): two offset-table buffers. */
bool tab_db(const rocke_fwd_flex_mfma_spec_t* s, const char* arch);
/* _wt_half(spec) / _w_elems(spec) / _nxi(spec). */
int wt_half(const rocke_fwd_flex_mfma_spec_t* s);
long long w_elems(const rocke_fwd_flex_mfma_spec_t* s);
int nxi(const rocke_fwd_flex_mfma_spec_t* s);

/* ---- IR helpers ---- */

/* `_Div`: division by a runtime divisor through a (magic, shift) pair. */
struct Div
{
    V d, m, l;

    V div(rocke_ir_builder_t* b, V n) const
    {
        V h = rocke_b_umul_hi_i32(b, n, m);
        V s = rocke_b_add(b, h, n);
        return rocke_b_lshr(b, s, l);
    }
    V mod(rocke_ir_builder_t* b, V n) const
    {
        V q = div(b, n);
        V p = rocke_b_mul(b, q, d);
        return rocke_b_sub(b, n, p);
    }
};

/* `_divs(b, lane, ds)`: the magic numbers of every divisor in one VALU
 * sequence (lane i for divisor i), read back by v_readlane. */
std::vector<Div> divs(rocke_ir_builder_t* b, V lane, const std::vector<V>& ds);

/* `_dma_lds(b, rsrc, dst, voff, soff)`. */
void dma_lds(rocke_ir_builder_t* b, V rsrc, V dst, V voff, V soff);
/* `_dma_lds_at(b, rsrc, base, imm, voff, soff)`. */
void dma_lds_at(rocke_ir_builder_t* b, V rsrc, V base, int imm, V voff, V soff);
/* b.inline_asm("", cons, [v], I32, sideeffect=True): an opaque copy. */
V opaque(rocke_ir_builder_t* b, V v, const char* cons);

/* `b.scf_if_else(cond, types)` with yields: then_fn / else_fn return the
 * yielded values; returns the op's results (empty on a builder error). */
template <class ThenFn, class ElseFn>
std::vector<V> if_else(rocke_ir_builder_t* b,
                       V cond,
                       const std::vector<const rocke_type_t*>& types,
                       ThenFn&& then_fn,
                       ElseFn&& else_fn)
{
    rocke_if_else_t ife = rocke_b_scf_if_else_results(b, cond, types.data(), (int)types.size());
    if(ife.op == NULL || ife.then_region == NULL || ife.else_region == NULL)
    {
        return std::vector<V>();
    }
    rocke_b_region_enter(b, ife.then_region);
    std::vector<V> t = then_fn();
    rocke_b_scf_yield(b, t.data(), (int)t.size());
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ife.else_region);
    std::vector<V> f = else_fn();
    rocke_b_scf_yield(b, f.data(), (int)f.size());
    rocke_b_region_leave(b);
    return std::vector<V>(ife.op->results, ife.op->results + ife.op->num_results);
}

/* `with b.scf_if_else(cond) as (then_ctx, else_ctx)` without results. */
template <class ThenFn, class ElseFn>
void if_else_void(rocke_ir_builder_t* b, V cond, ThenFn&& then_fn, ElseFn&& else_fn)
{
    rocke_if_else_t ife = rocke_b_scf_if_else(b, cond);
    if(ife.then_region == NULL || ife.else_region == NULL)
    {
        return;
    }
    rocke_b_region_enter(b, ife.then_region);
    then_fn();
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ife.else_region);
    else_fn();
    rocke_b_region_leave(b);
}

/* The in-block split builders (spec validated by the caller). */
rocke_kernel_def_t*
    build_cs4(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch);
rocke_kernel_def_t*
    build_cs2(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch);

} // namespace rocke_ffx

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_INTERNAL_HPP */
