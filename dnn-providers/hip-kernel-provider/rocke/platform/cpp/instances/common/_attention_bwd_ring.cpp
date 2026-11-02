// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_ring.py: the host-side facts
 * (DmaBatch, RingWaits, dma_wait_counts, ring_slot_schedule) and the ring's
 * emission (dma_lane_source, LdsStatsView, QRing, PrefetchStep). See
 * rocke/helper_rocke.instances.common._attention_bwd_ring.h for the mapping.
 *
 * Every Python raise is one ckc::Error throw with the same message (a Python
 * set in a message prints in ascending order here); the C
 * entry points catch it. Python's // and % floor; fdiv / fmod do the same.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"

#include <string.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
namespace
{

long long fdiv(long long a, long long b)
{
    if(b == 0)
        throw ckc::ValueError("integer division or modulo by zero");
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

long long fmod_(long long a, long long b)
{
    return a - fdiv(a, b) * b;
}

void dma_batch_check(const rocke_attn_bwd_dma_batch_t& d)
{
    long long nbytes = (long long)d.block_m * d.head_size * d.elem_bytes;
    if(d.wave_size != 64)
        throw ckc::ValueError("the LDS DMA ring is wave64 only");
    if(fmod_(nbytes, ROCKE_ATTN_BWD_DMA_PIECE_BYTES))
        throw ckc::ValueError("a " + std::to_string(d.block_m) + " x " + std::to_string(d.head_size)
                              + " tile is not whole 1 KiB DMA pieces");
    if(d.block_m > d.wave_size || fmod_(d.block_m, 16))
        throw ckc::ValueError("the stats DMA needs block_m <= 64, a multiple of 16");
    if(d.waves < 1)
        throw ckc::ValueError("waves must be positive");
}

int of_wave_count(const rocke_attn_bwd_dma_batch_t& d, int wave, int* out, int cap)
{
    int n = 0;
    int total = rocke_attn_bwd_dma_batch_n_instructions(&d);
    for(int k = 0; k < total; ++k)
        if(fmod_(k, d.waves) == wave)
        {
            if(out != NULL && n < cap)
                out[n] = k;
            ++n;
        }
    return n;
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_status_t rocke_attn_bwd_dma_batch_init(rocke_attn_bwd_dma_batch_t* out,
                                             int block_m,
                                             int head_size,
                                             int waves,
                                             int wave_size,
                                             int elem_bytes,
                                             rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        rocke_attn_bwd_dma_batch_t d = {block_m, head_size, waves, wave_size, elem_bytes};
        dma_batch_check(d);
        if(out != NULL)
            *out = d;
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
}

int rocke_attn_bwd_dma_batch_pieces(const rocke_attn_bwd_dma_batch_t* d)
{
    return (int)fdiv((long long)d->block_m * d->head_size * d->elem_bytes,
                     ROCKE_ATTN_BWD_DMA_PIECE_BYTES);
}

int rocke_attn_bwd_dma_batch_n_instructions(const rocke_attn_bwd_dma_batch_t* d)
{
    return 2 * rocke_attn_bwd_dma_batch_pieces(d) + 2;
}

bool rocke_attn_bwd_dma_batch_instruction(const rocke_attn_bwd_dma_batch_t* d,
                                          int k,
                                          const char** tensor,
                                          int* index)
{
    int pieces = rocke_attn_bwd_dma_batch_pieces(d);
    const char* t;
    int j;
    if(k < 0 || k >= 2 * pieces + 2)
        return false;
    if(k < pieces)
        t = "q", j = k;
    else if(k < 2 * pieces)
        t = "do", j = k - pieces;
    else
        t = k == 2 * pieces ? "lse2" : "dsum", j = 0;
    if(tensor != NULL)
        *tensor = t;
    if(index != NULL)
        *index = j;
    return true;
}

int rocke_attn_bwd_dma_batch_of_wave(const rocke_attn_bwd_dma_batch_t* d,
                                     int wave,
                                     int* out,
                                     int cap)
{
    return of_wave_count(*d, wave, out, cap);
}

int rocke_attn_bwd_dma_batch_min_per_wave(const rocke_attn_bwd_dma_batch_t* d)
{
    int best = 0;
    for(int w = 0; w < d->waves; ++w)
    {
        int n = of_wave_count(*d, w, NULL, 0);
        if(w == 0 || n < best)
            best = n;
    }
    return best;
}

rocke_status_t rocke_attn_bwd_dma_wait_counts(
    int ring_depth, int n_qd, int n_a, rocke_attn_bwd_ring_waits_t* out, rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    if(!(ring_depth == 1 || ring_depth == 2 || ring_depth == 3) || n_qd < 0 || n_a < 0)
    {
        std::string msg = "bad ring wait inputs (" + std::to_string(ring_depth) + ", "
                          + std::to_string(n_qd) + ", " + std::to_string(n_a) + ")";
        return store_err(err, ROCKE_ERR_VALUE, msg.c_str());
    }
    const int none = ROCKE_ATTN_BWD_NONE;
    const int vmax = ROCKE_ATTN_BWD_RING_VMCNT_MAX;
    rocke_attn_bwd_ring_waits_t w;
    if(ring_depth == 1)
        w = {none, 0, none};
    else if(ring_depth == 2)
        w = {0, none, n_a < vmax ? n_a : vmax};
    else
    {
        long long s = (long long)n_qd + n_a;
        w = {n_qd < vmax ? n_qd : vmax, none, (int)(s < vmax ? s : vmax)};
    }
    if(out != NULL)
        *out = w;
    return ROCKE_OK;
}

} // extern "C"

/* ====================================================================== *
 * emission: slots, lane map, the register and DMA prefetch
 * ====================================================================== */

namespace
{

typedef rocke_value_t* V;
typedef rocke_ir_builder_t B;
typedef rocke_attn_bwd_q_ring_t Ring;
typedef rocke_attn_bwd_prefetch_tile_t Tile;
typedef rocke_attn_bwd_ring_slot_t Slot;

const char* const FENCE = ROCKE_ATTN_BWD_BARRIER_FENCE;
const char* const STATS[2] = {"lse2", "dsum"};
/* M0 holds the LDS destination of an LDS DMA (see the Python module). */
const char* const ASM_X4 = "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dwordx4 $0, $2";
const char* const ASM_X1 = "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dword $0, $2";
const char* const ASM_CONSTRAINTS = "v,s,s,~{m0},~{memory}";
const int CHUNK = 16; /* DmaSlabLayout.CHUNK */

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("attention backward ring: a helper failed");
}

void need(B* b, bool ok)
{
    if(!ok || !live(b))
        raise_builder(b);
}

V needv(B* b, V v)
{
    if(v == NULL || !live(b))
        raise_builder(b);
    return v;
}

void need_st(rocke_status_t st, const rocke_attn_bwd_err_t& e)
{
    if(st != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
}

/* BwdTileGeometry.wave_size / threads inside a guarded body: an unknown arch
 * raises bwd_arch_caps' ValueError (the C entry points return 0 instead). */
int wave_size_of(const rocke_attn_bwd_tile_geometry_t* g)
{
    rocke_attn_bwd_caps_t c;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_arch_caps(g->arch, &c, &e), e);
    return c.wave_size;
}

int threads_of(const rocke_attn_bwd_tile_geometry_t* g)
{
    int waves = g->waves;
    return waves * wave_size_of(g);
}

template <class Fn>
bool guarded(B* b, Fn&& fn)
{
    if(!live(b))
        return false;
    void* r = ckc::guard_builder(b, [&]() -> void* {
        fn();
        return (void*)1;
    });
    return r != NULL && live(b);
}

template <class T>
T* arena(B* b, size_t n)
{
    void* p = rocke_arena_calloc(&b->arena, (n ? n : 1) * sizeof(T));
    if(p == NULL)
        throw ckc::OOMError("attention backward ring: arena allocation failed");
    return (T*)p;
}

V ci(B* b, int64_t x)
{
    return rocke_b_const_i32(b, x);
}

rocke_attn_bwd_iv_t ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

rocke_attn_bwd_iv_t ivi(int64_t i)
{
    return rocke_attn_bwd_iv_int(i);
}

void sync_lds_only(B* b)
{
    rocke_b_sync_lds_only_fence(b, FENCE);
    need(b, true);
}

/* An scf_if region: enter on construction, leave on destruction. */
struct IfScope
{
    B* b;
    IfScope(B* bb, V cond)
        : b(bb)
    {
        rocke_if_t f = rocke_b_scf_if(b, cond);
        if(f.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, f.then_region);
    }
    ~IfScope()
    {
        rocke_b_region_leave(b);
    }
};

int bit_length(int x)
{
    int n = 0;
    while(x > 0)
    {
        ++n;
        x >>= 1;
    }
    return n;
}

rocke_attn_bwd_ring_slots_t slot_schedule(int depth, const char* path)
{
    if(depth == 1)
    {
        if(path == NULL || strcmp(path, "dma") != 0)
            throw ckc::ValueError("ring_depth 1 on the register path is not a ring");
        return rocke_attn_bwd_ring_slots_t{0, 0, 0};
    }
    int ahead = depth - 1;
    int fill = (path != NULL && strcmp(path, "dma") == 0) ? ahead : 1;
    return rocke_attn_bwd_ring_slots_t{0, fill, ahead};
}

/* dma_lane_source on host ints (HostInt). */
void lane_source_host(
    const rocke_attn_bwd_dma_slab_layout_t* layout, int piece, int lane, int* row, int* col)
{
    int eb = layout->elem_bytes;
    int chunk_elems = CHUNK / eb;
    int lanes_per_line = ROCKE_ATTN_BWD_LDS_LINE_BYTES / CHUNK;
    int lines = ROCKE_ATTN_BWD_DMA_PIECE_BYTES / ROCKE_ATTN_BWD_LDS_LINE_BYTES;
    if(layout->rows % lines)
        throw ckc::ValueError("DMA slab tiles need a multiple of 8 rows");
    int l_line = lane >> 3;
    int chunk = (lane & (lanes_per_line - 1)) ^ l_line;
    int row_bytes = rocke_attn_bwd_dma_slab_layout_row_bytes(layout);
    if(row_bytes >= ROCKE_ATTN_BWD_LDS_LINE_BYTES)
    {
        int line_elems = ROCKE_ATTN_BWD_LDS_LINE_BYTES / eb;
        int r_j = (int)fmod_((long long)lines * piece, layout->rows);
        int c_j = (int)fdiv((long long)lines * piece, layout->rows) * line_elems;
        *row = l_line + r_j;
        *col = chunk * chunk_elems + c_j;
        return;
    }
    int fold = rocke_attn_bwd_dma_slab_layout_fold(layout);
    int cpr = row_bytes / CHUNK;
    int r_j = fold * lines * piece;
    *row = l_line * fold + (chunk >> (bit_length(cpr) - 1)) + r_j;
    *col = (chunk & (cpr - 1)) * chunk_elems;
}

void lane_source(
    B* b, const rocke_attn_bwd_dma_slab_layout_t* layout, int piece, V lane, V* row, V* col)
{
    int eb = layout->elem_bytes;
    int chunk_elems = CHUNK / eb;
    int lanes_per_line = ROCKE_ATTN_BWD_LDS_LINE_BYTES / CHUNK;
    int lines = ROCKE_ATTN_BWD_DMA_PIECE_BYTES / ROCKE_ATTN_BWD_LDS_LINE_BYTES;
    if(layout->rows % lines)
        throw ckc::ValueError("DMA slab tiles need a multiple of 8 rows");
    V c3 = ci(b, 3);
    V l_line = rocke_b_lshr(b, lane, c3);
    V cm = ci(b, lanes_per_line - 1);
    V lo = rocke_b_land(b, lane, cm);
    V chunk = rocke_b_xor(b, lo, l_line);
    int row_bytes = rocke_attn_bwd_dma_slab_layout_row_bytes(layout);
    if(row_bytes >= ROCKE_ATTN_BWD_LDS_LINE_BYTES)
    {
        int line_elems = ROCKE_ATTN_BWD_LDS_LINE_BYTES / eb;
        int r_j = (int)fmod_((long long)lines * piece, layout->rows);
        int c_j = (int)fdiv((long long)lines * piece, layout->rows) * line_elems;
        V r = l_line;
        if(r_j)
        {
            V cr = ci(b, r_j);
            r = rocke_b_add(b, l_line, cr);
        }
        V ce = ci(b, chunk_elems);
        V c = rocke_b_mul(b, chunk, ce);
        if(c_j)
        {
            V cc = ci(b, c_j);
            c = rocke_b_add(b, c, cc);
        }
        need(b, true);
        *row = r;
        *col = c;
        return;
    }
    int fold = rocke_attn_bwd_dma_slab_layout_fold(layout);
    int cpr = row_bytes / CHUNK;
    int r_j = fold * lines * piece;
    V cf = ci(b, fold);
    V m = rocke_b_mul(b, l_line, cf);
    V cs = ci(b, bit_length(cpr) - 1);
    V sh = rocke_b_lshr(b, chunk, cs);
    V r = rocke_b_add(b, m, sh);
    if(r_j)
    {
        V cr = ci(b, r_j);
        r = rocke_b_add(b, r, cr);
    }
    V cm2 = ci(b, cpr - 1);
    V a = rocke_b_land(b, chunk, cm2);
    V ce = ci(b, chunk_elems);
    V c = rocke_b_mul(b, a, ce);
    need(b, true);
    *row = r;
    *col = c;
}

/* ----- LdsStatsView ----- */

void stats_row(B* b, const rocke_attn_bwd_lds_stats_view_t* s, V r, V* lse2, V* dsum)
{
    V a = needv(b, rocke_attn_bwd_lds_view_load(b, &s->lse2, ivv(r), ivi(0), rocke_f32(), 1));
    *lse2 = needv(b, rocke_b_vec_extract(b, a, 0));
    V d = needv(b, rocke_attn_bwd_lds_view_load(b, &s->dsum, ivv(r), ivi(0), rocke_f32(), 1));
    *dsum = needv(b, rocke_b_vec_extract(b, d, 0));
}

void lds_stats_rows(
    B* b, const rocke_attn_bwd_lds_stats_view_t* s, V const* rows, int n, V* out_lse2, V* out_dsum)
{
    V inf = rocke_b_const_f32(b, std::numeric_limits<double>::infinity());
    V zero = rocke_b_const_f32(b, 0.0);
    for(int i = 0; i < n; ++i)
    {
        V lse2, dsum;
        stats_row(b, s, rows[i], &lse2, &dsum);
        if(s->rows_left != NULL)
        {
            V ok = rocke_b_cmp_lt(b, rows[i], s->rows_left);
            lse2 = rocke_b_select(b, ok, lse2, inf);
            dsum = rocke_b_select(b, ok, dsum, zero);
        }
        need(b, true);
        out_lse2[i] = lse2;
        out_dsum[i] = dsum;
    }
}

/* ----- QRing ----- */

bool is_dma(const Ring* r)
{
    return r->has_dma;
}

V slot_elems(B* b, const Ring* r, Slot slot)
{
    if(slot.none)
        return NULL;
    if(slot.value == NULL)
        return slot.index ? ci(b, (int64_t)slot.index * r->stride_bytes / 2) : NULL;
    V c = ci(b, r->stride_bytes / 2);
    return needv(b, rocke_b_mul(b, slot.value, c));
}

void ring_views(
    B* b, const Ring* r, Slot slot, rocke_attn_bwd_lds_view_t* v_q, rocke_attn_bwd_lds_view_t* v_do)
{
    V se = slot_elems(b, r, slot);
    *v_q = r->v_q;
    *v_do = r->v_do;
    if(se != NULL)
    {
        v_q->shift = se;
        v_do->shift = se;
    }
}

void stats_views(B* b, const Ring* r, Slot slot, rocke_attn_bwd_lds_view_t out[2])
{
    const rocke_attn_bwd_lds_plan_t* plan = r->lds->plan;
    V se = slot_elems(b, r, slot);
    for(int i = 0; i < 2; ++i)
    {
        rocke_attn_bwd_err_t e;
        const rocke_attn_bwd_lds_buffer_t* buf
            = rocke_attn_bwd_lds_plan_buffer(plan, STATS[i], 0, &e);
        if(buf == NULL)
            ckc::raise_status(e.code, e.msg);
        need(b,
             rocke_attn_bwd_lds_view_init(
                 &out[i], b, r->lds->smem, buf->offset / 2, 1, buf->nbytes / 4, 2, se));
    }
}

V slot_of(B* b, const Ring* r, V it, int offset)
{
    if(r->depth == 1)
        return NULL;
    V x = it;
    if(offset)
    {
        V c = ci(b, offset);
        x = rocke_b_add(b, it, c);
    }
    if(r->depth == 2)
    {
        V c1 = ci(b, 1);
        return needv(b, rocke_b_land(b, x, c1));
    }
    V cd = ci(b, r->depth);
    return needv(b, rocke_b_mod(b, x, cd));
}

Slot slot_v(V v)
{
    return v == NULL ? rocke_attn_bwd_ring_slot_none() : rocke_attn_bwd_ring_slot_val(v);
}

rocke_attn_bwd_lds_stats_view_t lds_stats(B* b, const Ring* r, Slot slot, const Tile* tile)
{
    rocke_attn_bwd_lds_view_t v[2];
    stats_views(b, r, slot, v);
    rocke_attn_bwd_lds_stats_view_t out;
    out.lse2 = v[0];
    out.dsum = v[1];
    out.rows_left = NULL;
    if(is_dma(r))
    {
        rocke_attn_bwd_ws_slab_t sl
            = rocke_attn_bwd_workspace_addr_slab(&r->stats->lse2, ivv(tile->h), ivv(tile->q0));
        need(b, true);
        out.rows_left = sl.rows_left;
    }
    return out;
}

V m0_of(B* b, const Ring* r, int name, int piece, V slot_bytes)
{
    int off = r->offsets[name] + piece * ROCKE_ATTN_BWD_DMA_PIECE_BYTES;
    V m0 = r->lds_base;
    if(off)
    {
        V c = ci(b, off);
        m0 = rocke_b_add(b, r->lds_base, c);
    }
    if(slot_bytes != NULL)
        m0 = rocke_b_add(b, m0, slot_bytes);
    return needv(b, m0);
}

void dma_asm(B* b, const char* tmpl, V voff, V m0, V ptr)
{
    V ops[3] = {voff, m0, ptr};
    rocke_op_t* op = rocke_b_inline_asm(b, tmpl, ASM_CONSTRAINTS, ops, 3, NULL, 0, NULL);
    need(b, op != NULL);
}

void emit_pieces(B* b,
                 const Ring* r,
                 const std::vector<int>& pieces,
                 const Tile* tile,
                 const rocke_attn_bwd_ws_slab_t ws[2],
                 V slot_bytes)
{
    V row_l = r->lane_src[0], col_l = r->lane_src[1];
    for(int k : pieces)
    {
        const char* name;
        int j;
        rocke_attn_bwd_dma_batch_instruction(&r->batch, k, &name, &j);
        int ni = strcmp(name, "q") == 0      ? 0
                 : strcmp(name, "do") == 0   ? 1
                 : strcmp(name, "lse2") == 0 ? 2
                                             : 3;
        V m0 = m0_of(b, r, ni, j, slot_bytes);
        if(ni >= 2)
        {
            const rocke_attn_bwd_ws_slab_t* slab = &ws[ni - 2];
            V idx = needv(b, rocke_attn_bwd_ws_slab_index(slab, r->lane, ivi(0)));
            V c2 = ci(b, 2);
            V voff = rocke_b_shl(b, idx, c2);
            V ptr = needv(b, rocke_attn_bwd_ws_slab_sgpr_ptr(slab));
            dma_asm(b, ASM_X1, voff, m0, ptr);
            continue;
        }
        if(!tile->has_slabs)
            throw ckc::ValueError("'NoneType' object has no attribute 'index'");
        const rocke_attn_bwd_row_slab_t* rslab = ni == 0 ? &tile->q_slab : &tile->do_slab;
        int r_j, c_j;
        lane_source_host(&r->layout, j, 0, &r_j, &c_j);
        V inner = row_l;
        if(r_j)
        {
            V cr = ci(b, r_j);
            inner = rocke_b_add(b, row_l, cr);
        }
        V row = rocke_b_add(b, tile->q0, inner);
        V col = col_l;
        if(c_j)
        {
            V cc = ci(b, c_j);
            col = rocke_b_add(b, col_l, cc);
        }
        V idx = needv(b, rocke_attn_bwd_row_slab_index(rslab, row, ivv(col)));
        V c1 = ci(b, 1);
        V voff = rocke_b_shl(b, idx, c1);
        V ptr = needv(b, rocke_attn_bwd_row_slab_sgpr_ptr(rslab));
        dma_asm(b, ASM_X4, voff, m0, ptr);
    }
}

void issue_dma(B* b, const Ring* r, const Tile* tile, Slot slot)
{
    const rocke_attn_bwd_tile_geometry_t* g = r->geom;
    V slot_bytes = NULL;
    if(!slot.none && !(slot.value == NULL && slot.index == 0))
    {
        V s = slot.value != NULL ? slot.value : ci(b, slot.index);
        V c = ci(b, r->stride_bytes);
        slot_bytes = needv(b, rocke_b_mul(b, s, c));
    }
    rocke_attn_bwd_ws_slab_t ws[2];
    ws[0] = rocke_attn_bwd_workspace_addr_slab(&r->stats->lse2, ivv(tile->h), ivv(tile->q0));
    need(b, true);
    ws[1] = rocke_attn_bwd_workspace_addr_slab(&r->stats->dsum, ivv(tile->h), ivv(tile->q0));
    need(b, true);
    int n_instr = rocke_attn_bwd_dma_batch_n_instructions(&r->batch);
    int n_w = g->waves < n_instr ? g->waves : n_instr;
    for(int w = 0; w < n_w; ++w)
    {
        int cnt = rocke_attn_bwd_dma_batch_of_wave(&r->batch, w, NULL, 0);
        std::vector<int> mine(cnt);
        rocke_attn_bwd_dma_batch_of_wave(&r->batch, w, mine.data(), cnt);
        if(g->waves == 1)
        {
            emit_pieces(b, r, mine, tile, ws, slot_bytes);
            continue;
        }
        V cw = ci(b, w);
        V cond = rocke_b_cmp_eq(b, r->wave, cw);
        IfScope s(b, cond);
        emit_pieces(b, r, mine, tile, ws, slot_bytes);
    }
}

bool packed_ring(const Ring* r)
{
    return r->geom->head_pack;
}

bool joint(const Ring* r, int total)
{
    /* Python's `and` order: threads / wave_size only after the packed test */
    return !packed_ring(r) && total < threads_of(r->geom) && total % wave_size_of(r->geom) == 0;
}

int n_reg_chunks(const Ring* r)
{
    const rocke_attn_bwd_tile_geometry_t* g = r->geom;
    int total = g->k_m0 * g->head_size / 8;
    if(joint(r, total))
        return 1;
    int threads = threads_of(g);
    return 2 * ((total + threads - 1) / threads);
}

/* _chunk_rc */
void chunk_rc(B* b, V chunk, int per_row, V* row, V* col)
{
    if(per_row == 1)
    {
        *row = chunk;
        *col = ci(b, 0);
        return;
    }
    int shift = bit_length(per_row) - 1;
    if(per_row != 1 << shift)
        throw ckc::ValueError("chunks per row must be a power of two");
    V cs = ci(b, shift);
    V r = rocke_b_lshr(b, chunk, cs);
    V cm = ci(b, per_row - 1);
    V a = rocke_b_land(b, chunk, cm);
    V c3 = ci(b, 3);
    V c = rocke_b_shl(b, a, c3);
    need(b, true);
    *row = r;
    *col = c;
}

struct RegChunk
{
    int tensor; /* 0 q, 1 do, 2 joint */
    V r, c, val;
    V guard; /* joint: guard[0] */
    V is_do; /* joint: guard[1] */
};

std::vector<RegChunk> reg_chunks(B* b, const Ring* r, const Tile* tile, bool load)
{
    const rocke_attn_bwd_tile_geometry_t* g = r->geom;
    int d = g->head_size, m0 = g->k_m0;
    int threads = threads_of(g);
    int per_row = d / 8;
    int total = m0 * per_row;
    bool packed = packed_ring(r);
    if(tile != NULL && (tile->packed != NULL) != packed)
        throw ckc::ValueError("a packed tile on an unpacked ring (or the reverse)");
    std::vector<RegChunk> out;
    if(joint(r, total))
    {
        V ct = ci(b, total);
        V is_do = rocke_b_cmp_ge(b, r->tid, ct);
        V ct2 = ci(b, total);
        V c0 = ci(b, 0);
        V sel = rocke_b_select(b, is_do, ct2, c0);
        V chunk = rocke_b_sub(b, r->tid, sel);
        V guard = NULL;
        if(2 * total < threads)
        {
            V c2t = ci(b, 2 * total);
            guard = rocke_b_cmp_lt(b, r->tid, c2t);
            V cl = ci(b, total - 1);
            chunk = rocke_b_smin(b, chunk, cl);
        }
        need(b, true);
        V row, col;
        chunk_rc(b, chunk, per_row, &row, &col);
        V val = NULL;
        if(load)
        {
            if(!tile->has_slabs)
                throw ckc::ValueError("'NoneType' object has no attribute 't_stride'");
            const rocke_attn_bwd_row_slab_t* q = &tile->q_slab;
            const rocke_attn_bwd_row_slab_t* dO = &tile->do_slab;
            V sd = needv(b, rocke_attn_bwd_as_i32(b, dO->t_stride));
            V sq = needv(b, rocke_attn_bwd_as_i32(b, q->t_stride));
            V stride = rocke_b_select(b, is_do, sd, sq);
            V ptr = rocke_b_select(b, is_do, dO->ptr, q->ptr);
            need(b, true);
            rocke_attn_bwd_slab_row_loader_t ctx;
            ctx.b = b;
            ctx.slab = *q;
            ctx.slab.ptr = ptr;
            ctx.slab.t_stride = ivv(stride);
            ctx.slab.has_parts = false; /* parts[2] is no longer ptr */
            ctx.row0 = tile->q0;
            ctx.vec = 8;
            rocke_attn_bwd_tile_loader_t l = rocke_attn_bwd_slab_row_loader(&ctx);
            val = needv(b, l.fn(l.ctx, row, col));
        }
        out.push_back(RegChunk{2, row, col, val, guard, is_do});
        return out;
    }
    for(int t = 0; t < 2; ++t)
    {
        rocke_attn_bwd_slab_row_loader_t sctx;
        rocke_attn_bwd_packed_row_loader_t pctx;
        rocke_attn_bwd_tile_loader_t loader = {NULL, NULL};
        if(load)
        {
            if(packed)
            {
                pctx.b = b;
                pctx.addr = t == 0 ? tile->packed->q_addr : tile->packed->do_addr;
                pctx.packed = tile->packed->rows;
                pctx.vec = 8;
                pctx.clamp = true;
                loader = rocke_attn_bwd_packed_row_loader(&pctx);
            }
            else
            {
                if(!tile->has_slabs)
                    throw ckc::ValueError("'NoneType' object has no attribute 'load_vec'");
                sctx.b = b;
                sctx.slab = t == 0 ? tile->q_slab : tile->do_slab;
                sctx.row0 = tile->q0;
                sctx.vec = 8;
                loader = rocke_attn_bwd_slab_row_loader(&sctx);
            }
        }
        int rounds = (total + threads - 1) / threads;
        for(int it = 0; it < rounds; ++it)
        {
            V chunk = r->tid;
            if(it != 0)
            {
                V c = ci(b, (int64_t)it * threads);
                chunk = rocke_b_add(b, r->tid, c);
            }
            bool tail = (int64_t)(it + 1) * threads > total;
            V src = chunk;
            if(tail)
            {
                V cl = ci(b, total - 1);
                src = rocke_b_smin(b, chunk, cl);
            }
            need(b, true);
            V row, col;
            chunk_rc(b, src, per_row, &row, &col);
            V guard = NULL;
            if(tail)
            {
                V ct = ci(b, total);
                guard = needv(b, rocke_b_cmp_lt(b, chunk, ct));
            }
            V val = load ? needv(b, loader.fn(loader.ctx, row, col)) : NULL;
            out.push_back(RegChunk{t, row, col, val, guard, NULL});
        }
    }
    return out;
}

std::vector<V> load_stats_row(B* b, const Ring* r, const Tile* tile)
{
    V cm = ci(b, r->geom->k_m0 - 1);
    V row = needv(b, rocke_b_smin(b, r->tid, cm));
    V lse2, dsum;
    if(tile->packed == NULL)
        need(b,
             rocke_attn_bwd_stats_view_rows(r->stats, b, tile->h, tile->q0, &row, 1, &lse2, &dsum));
    else
        need(b,
             rocke_attn_bwd_stats_view_packed_rows(
                 r->stats, b, &tile->packed->rows, &row, 1, &lse2, &dsum));
    return std::vector<V>{lse2, dsum};
}

std::vector<V> load_regs(B* b, const Ring* r, const Tile* tile)
{
    std::vector<V> regs;
    for(const RegChunk& c : reg_chunks(b, r, tile, true))
        regs.push_back(c.val);
    std::vector<V> st = load_stats_row(b, r, tile);
    regs.insert(regs.end(), st.begin(), st.end());
    return regs;
}

void store_regs(B* b, const Ring* r, const std::vector<V>& regs, Slot slot)
{
    rocke_attn_bwd_lds_view_t views[2];
    ring_views(b, r, slot, &views[0], &views[1]);
    std::vector<RegChunk> chunks = reg_chunks(b, r, NULL, false);
    if(chunks.size() + 2 != regs.size())
        throw ckc::ValueError("register set does not match the chunk plan");
    {
        V cm = ci(b, r->geom->k_m0);
        V cond = rocke_b_cmp_lt(b, r->tid, cm);
        IfScope s(b, cond);
        rocke_attn_bwd_lds_view_t sv[2];
        stats_views(b, r, slot, sv);
        for(int i = 0; i < 2; ++i)
        {
            rocke_attn_bwd_lds_view_store(
                b, &sv[i], ivv(r->tid), ivi(0), regs[chunks.size() + i], 1);
            need(b, true);
        }
    }
    for(size_t i = 0; i < chunks.size(); ++i)
    {
        const RegChunk& c = chunks[i];
        V val = regs[i];
        if(c.tensor == 2)
        {
            V id_do = needv(b, rocke_attn_bwd_lds_view_index(b, &views[1], ivv(c.r), ivv(c.c)));
            V id_q = needv(b, rocke_attn_bwd_lds_view_index(b, &views[0], ivv(c.r), ivv(c.c)));
            V idx = needv(b, rocke_b_select(b, c.is_do, id_do, id_q));
            if(c.guard == NULL)
                rocke_b_smem_store_vN(b, r->lds->smem, &idx, 1, val, 8);
            else
            {
                IfScope s(b, c.guard);
                rocke_b_smem_store_vN(b, r->lds->smem, &idx, 1, val, 8);
            }
            need(b, true);
            continue;
        }
        if(c.guard == NULL)
            rocke_attn_bwd_lds_view_store(b, &views[c.tensor], ivv(c.r), ivv(c.c), val, 8);
        else
        {
            IfScope s(b, c.guard);
            rocke_attn_bwd_lds_view_store(b, &views[c.tensor], ivv(c.r), ivv(c.c), val, 8);
        }
        need(b, true);
    }
}

void ring_init(B* b,
               Ring* r,
               const rocke_attn_bwd_tile_geometry_t* geom,
               const rocke_attn_bwd_bwd_lds_t* lds,
               V tid,
               V lane,
               V wave,
               const rocke_attn_bwd_stats_view_t* stats,
               int n_atomics)
{
    memset(r, 0, sizeof(*r));
    r->geom = geom;
    r->path = geom->global_path;
    r->depth = geom->ring_depth;
    if(strcmp(r->path, "vgpr") == 0 && r->depth == 1)
        throw ckc::ValueError("(vgpr, 1) is the synchronous staging, not a ring");
    r->lds = lds;
    r->tid = tid;
    r->lane = lane;
    r->wave = wave;
    r->stats = stats;
    r->n_atomics = n_atomics;
    const rocke_attn_bwd_lds_plan_t* plan = lds->plan;
    int d = geom->head_size, m0 = geom->k_m0;
    int wave_size = wave_size_of(geom);
    if(m0 > wave_size)
        throw ckc::ValueError("a ring stats slot holds one row per lane of wave 0");
    const char* names[4] = {"q", "do", "lse2", "dsum"};
    std::vector<int> strides;
    for(const char* n : names)
    {
        rocke_attn_bwd_err_t e;
        int st = rocke_attn_bwd_lds_plan_ring_stride(plan, n, &e);
        if(st < 0)
            ckc::raise_status(e.code, e.msg);
        bool seen = false;
        for(int x : strides)
            seen = seen || x == st;
        if(!seen)
            strides.push_back(st);
    }
    if(strides.size() != 1)
    {
        /* Python prints the set {...}; its order is CPython's hash order,
         * here ascending */
        std::sort(strides.begin(), strides.end());
        std::string set = "{";
        for(size_t i = 0; i < strides.size(); ++i)
            set += (i ? ", " : "") + std::to_string(strides[i]);
        throw ckc::ValueError("ring slots are not evenly spaced: " + set + "}");
    }
    r->stride_bytes = strides[0];
    need(b, rocke_attn_bwd_bwd_lds_view(b, lds, "q", d, 0, &r->v_q));
    need(b, rocke_attn_bwd_bwd_lds_view(b, lds, "do", d, 0, &r->v_do));
    r->schedule = slot_schedule(r->depth, r->path);
    for(int i = 0; i < 4; ++i)
    {
        rocke_attn_bwd_err_t e;
        r->offsets[i] = rocke_attn_bwd_lds_plan_offset(plan, names[i], 0, &e);
        if(r->offsets[i] < 0)
            ckc::raise_status(e.code, e.msg);
    }
    int min_bytes = 0;
    for(int i = 0; i < 2; ++i)
    {
        rocke_attn_bwd_err_t e;
        const rocke_attn_bwd_lds_buffer_t* buf
            = rocke_attn_bwd_lds_plan_buffer(plan, STATS[i], 0, &e);
        if(buf == NULL)
            ckc::raise_status(e.code, e.msg);
        min_bytes = i == 0 || buf->nbytes < min_bytes ? buf->nbytes : min_bytes;
    }
    if(min_bytes < wave_size * 4)
        throw ckc::ValueError("a stats slot must hold one fp32 per lane");
    if(strcmp(r->path, "dma") == 0)
    {
        rocke_attn_bwd::check_lds_dma_width(geom->arch, 4);
        rocke_attn_bwd::check_lds_dma_width(geom->arch, 1);
        if(geom->head_pack || geom->stage_vec != 8)
            throw ckc::ValueError("the DMA ring needs stage_vec = 8 and no head packing");
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_dma_batch_init(&r->batch, m0, d, geom->waves, wave_size, 2, &e), e);
        need_st(rocke_attn_bwd_dma_wait_counts(r->depth,
                                               rocke_attn_bwd_dma_batch_min_per_wave(&r->batch),
                                               n_atomics,
                                               &r->waits,
                                               &e),
                e);
        const rocke_attn_bwd_lds_layout_t* lay = rocke_attn_bwd_lds_plan_layout(plan, "q", &e);
        if(lay == NULL)
            ckc::raise_status(e.code, e.msg);
        if(lay->kind != ROCKE_ATTN_BWD_LAYOUT_DMA_SLAB)
            throw ckc::ValueError("DMA ring buffers must use the slab layout");
        r->layout = lay->dma;
        r->has_dma = true;
        V addr = needv(b, rocke_b_smem_addr_of(b, lds->smem));
        V a32 = needv(b, rocke_b_trunc(b, addr, rocke_i32()));
        r->lds_base = needv(b, rocke_b_to_sgpr_u32(b, a32));
        lane_source(b, &r->layout, 0, lane, &r->lane_src[0], &r->lane_src[1]);
    }
}

std::vector<V>
    prologue(B* b, const Ring* r, V trip, const Tile* tiles, int n_tiles, const char* name)
{
    int want = r->depth - 1 > 0 ? r->depth - 1 : 0;
    if(n_tiles != want)
        throw ckc::ValueError("the prologue of ring depth " + std::to_string(r->depth)
                              + " takes R - 1 tiles");
    V zero = ci(b, 0);
    V has = needv(b, rocke_b_cmp_lt(b, zero, trip));
    std::vector<V> out;
    if(is_dma(r))
    {
        if(r->depth > 1)
        {
            {
                IfScope s(b, has);
                for(int k = 0; k < n_tiles; ++k)
                    issue_dma(b, r, &tiles[k], rocke_attn_bwd_ring_slot_int(k));
                rocke_b_s_waitcnt(b, r->waits.prologue, -1, -1);
            }
            sync_lds_only(b);
        }
        return out;
    }
    if(r->depth == 2)
    {
        {
            IfScope s(b, has);
            store_regs(b, r, load_regs(b, r, &tiles[0]), rocke_attn_bwd_ring_slot_int(0));
        }
        sync_lds_only(b);
        return out;
    }
    /* R = 3: tile 0 into slot 0, tile 1 carried out of a 0 / 1-trip loop */
    V one = ci(b, 1);
    std::vector<V> init;
    int nrc = n_reg_chunks(r);
    for(int i = 0; i < nrc; ++i)
        init.push_back(needv(b, rocke_b_zero_vec(b, r->lds->elem, 8)));
    for(int i = 0; i < 2; ++i)
        init.push_back(rocke_b_const_f32(b, 0.0));
    std::vector<std::string> names(init.size());
    std::vector<rocke_iter_arg_t> args(init.size());
    for(size_t i = 0; i < init.size(); ++i)
    {
        names[i] = std::string(name) + "pr" + std::to_string(i);
        args[i].name = names[i].c_str();
        args[i].init = init[i];
    }
    V hi = rocke_b_select(b, has, one, zero);
    std::string iv_name = std::string(name) + "pit";
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, zero, hi, one, args.data(), (int)args.size(), iv_name.c_str(), false, false);
    need(b, loop.op != NULL);
    rocke_b_region_enter(b, loop.body);
    store_regs(b, r, load_regs(b, r, &tiles[0]), rocke_attn_bwd_ring_slot_int(0));
    std::vector<V> y = load_regs(b, r, &tiles[1]);
    rocke_b_scf_yield(b, y.data(), (int)y.size());
    rocke_b_region_leave(b);
    need(b, true);
    sync_lds_only(b);
    for(int i = 0; i < loop.op->num_results; ++i)
        out.push_back(loop.op->results[i]);
    return out;
}

V* to_arena(B* b, const std::vector<V>& v)
{
    V* a = arena<V>(b, v.size());
    for(size_t i = 0; i < v.size(); ++i)
        a[i] = v[i];
    return a;
}

} // namespace

extern "C" {

rocke_status_t rocke_attn_bwd_ring_slot_schedule(int ring_depth,
                                                 const char* path,
                                                 rocke_attn_bwd_ring_slots_t* out,
                                                 rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        *out = slot_schedule(ring_depth, path);
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
}

bool rocke_attn_bwd_dma_lane_source(rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_dma_slab_layout_t* layout,
                                    int piece,
                                    rocke_value_t* lane,
                                    rocke_value_t** row,
                                    rocke_value_t** col)
{
    return guarded(b, [&]() { lane_source(b, layout, piece, lane, row, col); });
}

rocke_status_t rocke_attn_bwd_dma_lane_source_host(const rocke_attn_bwd_dma_slab_layout_t* layout,
                                                   int piece,
                                                   int lane,
                                                   int* row,
                                                   int* col,
                                                   rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        lane_source_host(layout, piece, lane, row, col);
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
}

bool rocke_attn_bwd_lds_stats_view_rows(const rocke_attn_bwd_lds_stats_view_t* s,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* const* rows,
                                        int n_rows,
                                        rocke_value_t** out_lse2,
                                        rocke_value_t** out_dsum)
{
    return guarded(b, [&]() { lds_stats_rows(b, s, rows, n_rows, out_lse2, out_dsum); });
}

bool rocke_attn_bwd_lds_stats_view_packed_rows(const rocke_attn_bwd_lds_stats_view_t* s,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* const* rows,
                                               int n_rows,
                                               rocke_value_t** out_lse2,
                                               rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        if(s->rows_left != NULL)
            throw ckc::ValueError("the DMA ring does not stage head-packed steps");
        for(int i = 0; i < n_rows; ++i)
            stats_row(b, s, rows[i], &out_lse2[i], &out_dsum[i]);
    });
}

bool rocke_attn_bwd_q_ring_init(rocke_attn_bwd_q_ring_t* r,
                                rocke_ir_builder_t* b,
                                const rocke_attn_bwd_tile_geometry_t* geom,
                                const rocke_attn_bwd_bwd_lds_t* lds,
                                rocke_value_t* tid,
                                rocke_value_t* lane,
                                rocke_value_t* wave,
                                const rocke_attn_bwd_stats_view_t* stats,
                                int n_atomics)
{
    return guarded(b, [&]() { ring_init(b, r, geom, lds, tid, lane, wave, stats, n_atomics); });
}

bool rocke_attn_bwd_q_ring_check_step(const rocke_attn_bwd_q_ring_t* r,
                                      rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_stage_table_t* table)
{
    return guarded(b, [&]() {
        if(!is_dma(r) || r->depth == 1)
            return;
        int after = 0;
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_stage_table_vm_after_prefetch(table, true, &after, &e), e);
        if(after < r->n_atomics)
            throw ckc::ValueError("the step issues " + std::to_string(after)
                                  + " VMEM operations after its DMA copy; the ring's wait "
                                    "immediates assume at least "
                                  + std::to_string(r->n_atomics));
    });
}

rocke_value_t* rocke_attn_bwd_q_ring_slot_of(const rocke_attn_bwd_q_ring_t* r,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* it,
                                             int offset)
{
    V out = NULL;
    guarded(b, [&]() { out = slot_of(b, r, it, offset); });
    return out;
}

rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_none(void)
{
    rocke_attn_bwd_ring_slot_t s = {true, NULL, 0};
    return s;
}

rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_int(int index)
{
    rocke_attn_bwd_ring_slot_t s = {false, NULL, index};
    return s;
}

rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_val(rocke_value_t* value)
{
    rocke_attn_bwd_ring_slot_t s = {false, value, 0};
    return s;
}

bool rocke_attn_bwd_q_ring_views(const rocke_attn_bwd_q_ring_t* r,
                                 rocke_ir_builder_t* b,
                                 rocke_attn_bwd_ring_slot_t slot,
                                 rocke_attn_bwd_lds_view_t* v_q,
                                 rocke_attn_bwd_lds_view_t* v_do)
{
    return guarded(b, [&]() { ring_views(b, r, slot, v_q, v_do); });
}

bool rocke_attn_bwd_q_ring_lds_stats(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     rocke_attn_bwd_ring_slot_t slot,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_attn_bwd_lds_stats_view_t* out)
{
    return guarded(b, [&]() { *out = lds_stats(b, r, slot, tile); });
}

bool rocke_attn_bwd_q_ring_issue_dma(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_attn_bwd_ring_slot_t slot)
{
    return guarded(b, [&]() {
        if(!is_dma(r))
            throw ckc::ValueError("'NoneType' object has no attribute 'instructions'");
        issue_dma(b, r, tile, slot);
    });
}

bool rocke_attn_bwd_q_ring_prologue(const rocke_attn_bwd_q_ring_t* r,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* trip,
                                    const rocke_attn_bwd_prefetch_tile_t* tiles,
                                    int n_tiles,
                                    const char* name,
                                    rocke_value_t*** out,
                                    int* n_out)
{
    return guarded(b, [&]() {
        std::vector<V> v = prologue(b, r, trip, tiles, n_tiles, name);
        *out = to_arena(b, v);
        *n_out = (int)v.size();
    });
}

bool rocke_attn_bwd_q_ring_drain(const rocke_attn_bwd_q_ring_t* r, rocke_ir_builder_t* b)
{
    return guarded(b, [&]() {
        if(is_dma(r))
        {
            rocke_b_s_waitcnt(b, 0, -1, -1);
            sync_lds_only(b);
        }
    });
}

int rocke_attn_bwd_q_ring_n_reg_chunks(const rocke_attn_bwd_q_ring_t* r)
{
    try
    {
        return n_reg_chunks(r);
    }
    catch(const ckc::Error&)
    {
        return 0; /* an unknown arch (a successful _init has rejected it) */
    }
}

bool rocke_attn_bwd_q_ring_load_regs(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_value_t*** out,
                                     int* n_out)
{
    return guarded(b, [&]() {
        std::vector<V> v = load_regs(b, r, tile);
        *out = to_arena(b, v);
        *n_out = (int)v.size();
    });
}

bool rocke_attn_bwd_q_ring_store_regs(const rocke_attn_bwd_q_ring_t* r,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* const* regs,
                                      int n_regs,
                                      rocke_attn_bwd_ring_slot_t slot)
{
    return guarded(b, [&]() { store_regs(b, r, std::vector<V>(regs, regs + n_regs), slot); });
}

rocke_attn_bwd_prefetch_step_t
    rocke_attn_bwd_prefetch_step_make(const rocke_attn_bwd_q_ring_t* ring,
                                      rocke_value_t* it,
                                      const rocke_attn_bwd_prefetch_tile_t* here,
                                      const rocke_attn_bwd_prefetch_tile_t* ahead,
                                      rocke_value_t* const* carried,
                                      int n_carried)
{
    rocke_attn_bwd_prefetch_step_t p;
    memset(&p, 0, sizeof(p));
    p.ring = ring;
    p.it = it;
    p.here = here;
    p.ahead = ahead;
    p.carried = carried;
    p.n_carried = n_carried;
    return p;
}

bool rocke_attn_bwd_prefetch_step_barrier(rocke_attn_bwd_prefetch_step_t* p, rocke_ir_builder_t* b)
{
    (void)p;
    return guarded(b, [&]() { sync_lds_only(b); });
}

static void issue_loads(B* b, rocke_attn_bwd_prefetch_step_t* p)
{
    if(is_dma(p->ring) || p->n_regs)
        throw ckc::ValueError("the register prefetch is issued once per step");
    std::vector<V> v = load_regs(b, p->ring, p->ahead);
    p->regs = to_arena(b, v);
    p->n_regs = (int)v.size();
}

bool rocke_attn_bwd_prefetch_step_begin(rocke_attn_bwd_prefetch_step_t* p,
                                        rocke_ir_builder_t* b,
                                        bool defer_loads,
                                        rocke_attn_bwd_lds_view_t* v_q,
                                        rocke_attn_bwd_lds_view_t* v_do,
                                        rocke_attn_bwd_lds_stats_view_t* stats_src)
{
    return guarded(b, [&]() {
        const Ring* r = p->ring;
        if(defer_loads && is_dma(r))
            throw ckc::ValueError("the DMA copy is issued at the step top");
        V read = slot_of(b, r, p->it, r->schedule.read);
        ring_views(b, r, slot_v(read), v_q, v_do);
        if(is_dma(r))
        {
            if(r->depth == 1)
            {
                issue_dma(b, r, p->here, rocke_attn_bwd_ring_slot_none());
                rocke_b_s_waitcnt(b, r->waits.step_top, -1, -1);
                sync_lds_only(b);
            }
            else
            {
                V fill = slot_of(b, r, p->it, r->schedule.fill);
                issue_dma(b, r, p->ahead, slot_v(fill));
            }
            *stats_src = lds_stats(b, r, slot_v(read), p->here);
            return;
        }
        if(!defer_loads)
        {
            issue_loads(b, p);
            rocke_b_sched_barrier(b, 0);
            need(b, true);
        }
        *stats_src = lds_stats(b, r, slot_v(read), p->here);
    });
}

bool rocke_attn_bwd_prefetch_step_issue_loads(rocke_attn_bwd_prefetch_step_t* p,
                                              rocke_ir_builder_t* b)
{
    return guarded(b, [&]() { issue_loads(b, p); });
}

bool rocke_attn_bwd_prefetch_step_end(rocke_attn_bwd_prefetch_step_t* p, rocke_ir_builder_t* b)
{
    return guarded(b, [&]() {
        const Ring* r = p->ring;
        if(is_dma(r))
        {
            if(r->waits.step_end != ROCKE_ATTN_BWD_NONE)
                rocke_b_s_waitcnt(b, r->waits.step_end, -1, -1);
            sync_lds_only(b);
            return;
        }
        V fill = slot_of(b, r, p->it, r->schedule.fill);
        if(r->depth == 2)
            store_regs(b, r, std::vector<V>(p->regs, p->regs + p->n_regs), slot_v(fill));
        else
        {
            store_regs(b, r, std::vector<V>(p->carried, p->carried + p->n_carried), slot_v(fill));
            p->yields = p->regs;
            p->n_yields = p->n_regs;
        }
        sync_lds_only(b);
    });
}

} // extern "C"
