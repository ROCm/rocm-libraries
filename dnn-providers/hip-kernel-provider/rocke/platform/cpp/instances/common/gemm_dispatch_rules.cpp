/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ GEMM dispatcher (rocke/gemm_dispatch.h): the rule layer of dispatch/gemm (common.py:
 * the native layout forms, the DTL ring / row-interleave / pad / fallback bodies and their
 * admission rules, the g1 adaptation, the K-proxy and the wave-quantisation geometry switch;
 * skinny.py: the gemm_skinny region, plans and the ring split / ring-any rules; fp16_rcr.py /
 * bf16_rcr.py: the candidate registries; core.CandidateRegistry select / combos) and
 * dispatch_gemm_fp16 / dispatch_gemm_bf16 for gfx950.
 */
#include "gemm_dispatch_internal.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace rocke
{
namespace gemm_dispatch
{
namespace detail
{

/* ------------------------------------------------------------------ CU count */

namespace
{
thread_local int64_t g_cus = 256;
}

int64_t current_cus()
{
    return g_cus;
}

CusScope::CusScope(int64_t cus)
    : prev(g_cus)
{
    g_cus = cus;
}

CusScope::~CusScope()
{
    g_cus = prev;
}

DispatchState& dispatch_state()
{
    static thread_local DispatchState s;
    return s;
}

namespace
{

/* ------------------------------------------------------------------ helpers */

std::string upper(std::string s)
{
    for(auto& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

int64_t tiles_of(int64_t M, int64_t N, int64_t tm, int64_t tn)
{
    return cdiv(M, tm) * cdiv(N, tn);
}

Tile mk_tile(int64_t tm,
             int64_t tn,
             int64_t tk,
             int64_t wm,
             int64_t wn,
             int64_t wk,
             int64_t wtm,
             int64_t wtn,
             int64_t wtk)
{
    Tile t;
    t.tile_m = tm;
    t.tile_n = tn;
    t.tile_k = tk;
    t.warp_m = wm;
    t.warp_n = wn;
    t.warp_k = wk;
    t.warp_tile_m = wtm;
    t.warp_tile_n = wtn;
    t.warp_tile_k = wtk;
    return t;
}

Tile ring_tile(int64_t tm, int64_t tn, int64_t tk, int64_t wm, int64_t wn)
{
    return mk_tile(tm, tn, tk, wm, wn, 1, 16, 16, 32);
}

UGemm make_spec(const std::string& name,
                const std::string& dtype,
                const Tile& tile,
                const Trait& trait)
{
    UGemm s;
    s.name = name;
    s.tile = tile;
    s.trait = trait;
    s.data.dtype_a = dtype;
    s.data.dtype_b = dtype;
    s.data.dtype_c = dtype;
    s.data.layout = "RCR";
    s.wave_size = 64;
    s.block_size = 0;
    s.post_init();
    return s;
}

/* --------------------------------------------------------- split_k helpers */

int64_t largest_valid_split_k(int64_t K, int64_t tile_k, int64_t cap)
{
    cap = std::max((int64_t)1, cap);
    int64_t best = 1;
    for(int64_t s = 2; s <= cap; ++s)
        if(pmod(K, s) == 0 && pmod(fdiv(K, s), tile_k) == 0)
            best = s;
    return best;
}

/* helpers.split_k.select_split_k (auto mode, target 256 CTAs) */
int64_t
    select_split_k(int64_t M, int64_t N, int64_t K, int64_t tile_m, int64_t tile_n, int64_t tile_k)
{
    int64_t base_grid = cdiv(M, tile_m) * cdiv(N, tile_n);
    int64_t max_split = std::max((int64_t)1, fdiv(K, tile_k));
    int64_t target_ctas = 256;
    if(base_grid >= target_ctas / 2)
        return 1;
    int64_t raw = K ? pround((double)K / 512.0) : 1;
    raw = std::max((int64_t)1, std::min(raw, max_split));
    int64_t chosen = largest_valid_split_k(K, tile_k, raw);
    return chosen <= 1 ? 1 : chosen;
}

/* ---------------------------------------------------------- common.py knobs */

/* -------------------------------------------------------------------- traits */

/* common._ring_pad_trait(**extra) */
Trait ring_pad_trait()
{
    Trait t;
    t.pipeline = "compv4";
    t.scheduler = "intrawave";
    t.epilogue = "default";
    t.pad_m = true;
    t.pad_n = true;
    t.direct_to_lds = true;
    t.dtl_prefetch = true;
    t.lds_swizzle = true;
    t.dtl_cache_a = 0;
    t.dtl_cache_b = 0;
    t.chiplet_swizzle = true;
    t.chiplet_wgm = 8;
    t.chiplet_chunk_size = 32;
    t.dtl_stages = 2;
    t.interwave_setprio = true;
    t.dtl_hoist = true;
    t.dtl_tight_rsrc = true;
    t.pad_fast_tile = true;
    return t;
}

/* replace(trait, **_S3ILV) */
Trait s3ilv(Trait t)
{
    t.dtl_stages = 3;
    t.dtl_frag_prefetch = true;
    t.mfma_interleave = true;
    t.ilv_merge = true;
    t.ilv_barrier_after = 2;
    return t;
}

/* common.dtl_pingpong_tile_trait(arch) (gfx950, ring levers) */
void dtl_pingpong_tile_trait(Tile* tile, Trait* trait, int64_t chiplet_chunk_size = 64)
{
    *tile = mk_tile(256, 256, 64, 4, 4, 1, 16, 16, 32);
    Trait t;
    t.pipeline = "compv4";
    t.scheduler = "intrawave";
    t.epilogue = "default";
    t.direct_to_lds = true;
    t.dtl_prefetch = true;
    t.lds_swizzle = true;
    t.dtl_cache_a = 0;
    t.dtl_cache_b = 0;
    t.chiplet_swizzle = true;
    t.chiplet_wgm = 8;
    t.chiplet_chunk_size = chiplet_chunk_size;
    t.dtl_stages = 2;
    t.interwave_setprio = true;
    *trait = t;
}

double wave_fill(int64_t M, int64_t N, int64_t tm, int64_t tn)
{
    int64_t tiles = tiles_of(M, N, tm, tn);
    int64_t cus = current_cus();
    double waves = (double)tiles / (double)cus;
    return waves / (double)std::max((int64_t)1, cdiv(tiles, cus));
}

} // namespace

void agpr_ilv_big_grid_tile_trait(const Context& ctx, Tile* tile, Trait* trait)
{
    (void)ctx;
    *tile = mk_tile(256, 256, 32, 2, 2, 1, 16, 16, 32);
    Tile pt;
    Trait base;
    dtl_pingpong_tile_trait(&pt, &base);
    base.chiplet_wgm = 4;
    base.chiplet_chunk_size = 64;
    base.dtl_stages = 3;
    base.interwave_setprio = true;
    base.dtl_frag_prefetch = true;
    base.acc_agpr = true;
    base.mfma_interleave = true;
    base.ilv_merge = true;
    base.ilv_barrier_after = 2;
    base.lds_swizzle_r = 1;
    base.dtl_hoist = true;
    base.persistent_ctas = current_cus();
    *trait = base;
}

namespace
{

/* ------------------------------------------------------------- big grid / rilv */

/* common.dtl_big_grid_body: 0 none, 1 agpr_ilv, 2 ring_w4 */
int dtl_big_grid_body(const Request& req, bool ragged)
{
    if(req.arch != "gfx950")
        return 0;
    int64_t t = tiles_of(req.M, req.N, 256, 256);
    if(t < DTL_BIG_GRID_TILES)
        return 0;
    int64_t K = req.K;
    int64_t footprint = (req.M + req.N) * K * 2;
    if(K <= DTL_BIG_GRID_SMALL_K)
        return 1;
    int64_t wide = ragged ? DTL_BIG_GRID_RAGGED_WIDE_TILES : DTL_BIG_GRID_WIDE_TILES;
    if(!(t >= DTL_BIG_GRID_ALL_K_TILES || footprint > DTL_BIG_GRID_FOOTPRINT_BYTES
         || (t >= wide && K >= DTL_BIG_GRID_WIDE_MIN_K)))
        return 0;
    if(ragged || footprint > DTL_BIG_GRID_MALL_BYTES)
        return 1;
    return 2;
}

bool beat_k_bad(int64_t K)
{
    if(pmod(K, 64) == 0)
        return false;
    return !(pmod(K, G1_DTL_K_TAIL_ALIGN) == 0 && cdiv(K, 64) >= G1_DTL_K_TAIL_MIN_TILES);
}

bool rilv_admits(const Request& req)
{
    if(req.arch != "gfx950")
        return false;
    int64_t M = req.M, N = req.N, K = req.K;
    if(pmod(M, 256) || pmod(N, 256) || beat_k_bad(K) || K < RILV_MIN_K)
        return false;
    return fdiv(M, 256) * fdiv(N, 256) >= RILV_MIN_TILES;
}

bool rilv_hb_schedule(int64_t M, int64_t N, int64_t K)
{
    int64_t tiles = cdiv(M, 256) * cdiv(N, 256);
    return K >= RILV_HB_MIN_K || (tiles >= RILV_HB_BIG_TILES && K >= RILV_HB_BIG_MIN_K);
}

/* common.rilv_tile_trait(arch, hb, persistent, sia) */
void rilv_tile_trait(bool hb, bool persistent, bool sia, Tile* tile, Trait* trait)
{
    *tile = mk_tile(256, 256, 64, 2, 2, 1, 16, 16, 32);
    Tile pt;
    Trait base;
    dtl_pingpong_tile_trait(&pt, &base);
    auto common = [](Trait t) {
        t.lds_swizzle = false;
        t.interwave_setprio = false;
        t.dtl_stages = 2;
        t.acc_agpr = true;
        t.dtl_frag_prefetch = true;
        t.mfma_interleave = true;
        t.dtl_hoist = true;
        t.lds_row_interleave = true;
        t.rilv_pad = RILV_LDS_PAD;
        t.rilv_vec_epilogue = true;
        t.rve_store_nt = true;
        return t;
    };
    Trait t = base;
    if(hb && sia)
    {
        t.chiplet_wgm = 16;
        t.chiplet_chunk_size = 8;
        t.sia_sched = true;
        t.sia_plr = true;
        t.sia_dtl_spread = true;
        t.sia_acq_gap = RILV_SIA_ACQ_GAP;
    }
    else if(hb)
    {
        t.chiplet_wgm = 16;
        t.chiplet_chunk_size = 8;
        t.hb_sched = true;
        t.hb_read_every = 2;
    }
    else if(persistent)
    {
        t.chiplet_wgm = 8;
        t.chiplet_chunk_size = 32;
        t.ilv_merge = true;
        t.persistent_ctas = current_cus();
    }
    else
    {
        t.chiplet_wgm = 16;
        t.chiplet_chunk_size = 8;
        t.ilv_merge = true;
    }
    *trait = common(t);
}

bool rilv_persistent(int64_t M, int64_t N, int64_t K)
{
    int64_t tiles = cdiv(M, 256) * cdiv(N, 256);
    return K <= RILV_PERSIST_MAX_K && tiles > RILV_PERSIST_MIN_WAVES * current_cus();
}

Trait rilv_tuned_trait(Trait t, int64_t kloop_unroll)
{
    t.ilv_merge = false;
    t.sia_sched = false;
    t.sia_plr = false;
    t.sia_dtl_spread = false;
    t.sia_acq_gap = 3;
    t.hb_sched = true;
    t.hb_read_every = 2;
    t.dtl_soff0 = true;
    t.dtl_addr_spread = true;
    t.hb_dtl_delay = 1;
    t.kloop_unroll = kloop_unroll;
    t.hb_skew = RILV_HB_SKEW;
    t.hb_tile_lgkm0 = RILV_HB_TILE_LGKM0;
    t.das_late = RILV_DAS_LATE;
    return t;
}

void rilv_tuned_tile_trait(int64_t M, int64_t N, int64_t K, Tile* tile, Trait* trait)
{
    Trait base;
    rilv_tile_trait(false, false, false, tile, &base);
    if(rilv_persistent(M, N, K))
    {
        base.chiplet_wgm = 8;
        base.chiplet_chunk_size = 32;
        base.persistent_ctas = current_cus();
    }
    else
    {
        /* RILV_LONGK_GRID is () (off) */
        base.chiplet_wgm = 16;
        base.chiplet_chunk_size = 8;
        base.persistent_ctas = 0;
    }
    int64_t ku = K > RILV_PERSIST_MAX_K ? RILV_KLOOP_UNROLL_LONG : RILV_KLOOP_UNROLL_SHORT;
    Trait t = rilv_tuned_trait(base, ku);
    t.mfma_serpentine = true; /* RILV_SERPENTINE */
    *trait = t;
}

/* rilv_spec_tile_trait: rilv_body "tuned", rilv_hb_loop "ir" */
void rilv_spec_tile_trait(const Request& req, Tile* tile, Trait* trait)
{
    rilv_tuned_tile_trait(req.M, req.N, req.K, tile, trait);
}

/* rilv_asm_loop_trait(trait, hb) */
Trait rilv_asm_loop_trait(Trait t, bool hb)
{
    t.ilv_merge = false;
    t.hb_sched = true;
    t.hb_read_every = 1;
    t.asm_loop = "tensile";
    t.asm_loop_skew = 1;
    if(hb)
        t.chiplet_chunk_size = 4;
    return t;
}

bool rilv_shortk_admits(const Request& req)
{
    if(req.arch != "gfx950")
        return false;
    if(req.dtype != "fp16" && req.dtype != "bf16")
        return false;
    int64_t M = req.M, N = req.N, K = req.K;
    if(pmod(M, 256) || pmod(N, 256) || beat_k_bad(K) || !(RILV_SHORTK_MIN_K <= K && K < RILV_MIN_K))
        return false;
    if(fdiv(M, 256) * fdiv(N, 256) < RILV_MIN_TILES)
        return false;
    return wave_fill(M, N, 256, 256) >= RILV_SHORTK_MIN_FILL;
}

void small_tile_128x192_exact_tile_trait(Tile* tile, Trait* trait)
{
    Trait t = ring_pad_trait();
    t.chiplet_swizzle = false;
    t.pad_m = false;
    t.pad_n = false;
    t.dtl_tight_rsrc = false;
    t.pad_fast_tile = false;
    t.dtl_soff0 = true;
    t.epilogue = "cshuffle";
    *tile = ring_tile(128, 192, 64, 4, 2);
    *trait = t;
}

void rilv_shortk_tile_trait(const Request& req, Tile* tile, Trait* trait)
{
    if(rilv_persistent(req.M, req.N, req.K))
    {
        rilv_tuned_tile_trait(req.M, req.N, req.K, tile, trait);
        return;
    }
    if(pmod(req.M, 128) == 0 && pmod(req.N, 192) == 0
       && wave_fill(req.M, req.N, 256, 256) < RILV_SHORTK_SMALL_TILE_MAX_FILL)
    {
        small_tile_128x192_exact_tile_trait(tile, trait);
        return;
    }
    rilv_tile_trait(false, false, false, tile, trait);
    trait->mfma_serpentine = true;
}

bool beat_native_n_any(const Request& req)
{
    std::string lay = upper(req.layout);
    return req.arch == "gfx950" && (lay == "RRR" || lay == "CRR");
}

bool rilv_pad_admits(const Request& req)
{
    if(req.arch != "gfx950")
        return false;
    int64_t M = req.M, N = req.N, K = req.K;
    bool n8 = pmod(N, 8) && !beat_native_n_any(req);
    if(!(pmod(M, 256) || pmod(N, 256)) || n8 || beat_k_bad(K) || K < RILV_MIN_K)
        return false;
    if(std::max(M, N) * K * 2 >= (int64_t(1) << 31) || M * N * 2 >= (int64_t(1) << 32))
        return false;
    if(cdiv(M, 256) * cdiv(N, 256) < RILV_MIN_TILES)
        return false;
    return wave_fill(M, N, 256, 256) >= DTL_RING_SMALL_MAX_FILL;
}

void rilv_pad_tile_trait(const Request& req, Tile* tile, Trait* trait)
{
    int64_t M = req.M, N = req.N, K = req.K;
    bool hb = rilv_hb_schedule(M, N, K);
    int64_t tiles = cdiv(M, 256) * cdiv(N, 256);
    bool persistent = K <= RILV_PERSIST_MAX_K || tiles >= RILV_HB_BIG_TILES;
    Trait t;
    rilv_tile_trait(hb, persistent, hb, tile, &t);
    if(hb)
    {
        /* RILV_PAD_HB_GRID */
        t.chiplet_wgm = RILV_PAD_HB_GRID_WGM;
        t.chiplet_chunk_size = RILV_PAD_HB_GRID_CHUNK;
        /* RILV_PAD_HB_TUNED */
        t.sia_sched = false;
        t.sia_plr = false;
        t.sia_dtl_spread = false;
        t.sia_acq_gap = 3;
        t.hb_sched = true;
        t.hb_read_every = 2;
        t.dtl_addr_spread = true;
        t.dtl_tight_ragged_only = true;
        t.hb_dtl_delay = 1;
        t.hb_skew = RILV_HB_SKEW;
        t.kloop_unroll = RILV_KLOOP_UNROLL_LONG;
    }
    if(!hb)
    {
        t.ilv_merge = false;
        t.hb_sched = true;
        t.hb_read_every = 2;
        t.chiplet_wgm = persistent ? 8 : 16;
        t.chiplet_chunk_size = persistent ? 32 : 8;
        t.persistent_ctas = persistent ? current_cus() : 0;
        if(persistent)
        {
            /* RILV_PAD_PERSIST_TUNED */
            t.dtl_addr_spread = true;
            t.dtl_tight_ragged_only = true;
            t.hb_dtl_delay = 1;
            t.hb_skew = RILV_HB_SKEW;
            t.kloop_unroll
                = K > RILV_PERSIST_MAX_K ? RILV_KLOOP_UNROLL_LONG : RILV_KLOOP_UNROLL_SHORT;
        }
    }
    t.pad_m = pmod(M, 256) != 0;
    t.pad_n = pmod(N, 256) != 0;
    t.dtl_tight_rsrc = true;
    t.pad_fast_tile = tiles >= RILV_PAD_FAST_TILE_MIN_TILES;
    t.rve_pad = 2;
    t.mfma_serpentine = true; /* tuned and RILV_SERPENTINE */
    /* RILV_PAD_HB_TILE_LGKM0 / RILV_PAD_DAS_LATE off */
    *trait = t;
}

void ring_w4_tile_trait(Tile* tile, Trait* trait)
{
    dtl_pingpong_tile_trait(tile, trait);
    trait->chiplet_wgm = 4;
    trait->iglp_opt = 1;
}

/* big_grid_tile_trait (false: None) */
bool big_grid_tile_trait(const Request& req, const Context& ctx, Tile* tile, Trait* trait)
{
    int b = dtl_big_grid_body(req, false);
    if(b == 1)
    {
        agpr_ilv_big_grid_tile_trait(ctx, tile, trait);
        return true;
    }
    if(b == 2)
    {
        ring_w4_tile_trait(tile, trait);
        return true;
    }
    return false;
}

/* small_tile_shortk_tile_trait (false: None) */
bool small_tile_shortk_tile_trait(int64_t M, int64_t N, int64_t K, Tile* tile, Trait* trait)
{
    if(!(SMALL_TILE_SHORTK_MIN_K <= K && K <= SMALL_TILE_SHORTK_MAX_K))
        return false;
    int64_t cus = current_cus();
    Tile t;
    Trait tr;
    if(pmod(K, 64) == 0)
    {
        if(tiles_of(M, N, 192, 128) <= cus)
        {
            if(pmod(M, 192) || pmod(N, 128))
            {
                if(pmod(M, 128) == 0 && pmod(N, 192) == 0)
                    t = ring_tile(128, 192, 64, 4, 2);
                else
                    t = ring_tile(192, 128, 64, 2, 4);
            }
            else
                t = ring_tile(192, 128, 64, 2, 4);
            tr = s3ilv(ring_pad_trait());
            tr.ilv_barrier_after = 0;
        }
        else
        {
            if(pmod(M, 128) || pmod(N, 192))
                t = (pmod(M, 192) == 0 && pmod(N, 128) == 0) ? ring_tile(192, 128, 64, 2, 4)
                                                             : ring_tile(128, 192, 64, 4, 2);
            else
                t = ring_tile(128, 192, 64, 4, 2);
            tr = ring_pad_trait();
            tr.chiplet_swizzle = false;
            if(wave_fill(M, N, t.tile_m, t.tile_n) < SMALL_TILE_SHORTK_MIN_FILL)
                return false;
        }
    }
    else if(pmod(K, 32) == 0)
    {
        t = ring_tile(256, 128, 32, 4, 1);
        tr = s3ilv(ring_pad_trait());
        tr.acc_agpr = true;
        tr.interwave_setprio = false;
    }
    else
        return false;
    if(tiles_of(M, N, t.tile_m, t.tile_n) > SMALL_TILE_SHORTK_MAX_WAVES * cus)
        return false;
    if((double)tiles_of(M, N, t.tile_m, t.tile_n) < DTL_RING_SMALL_MIN_TILES_PER_CU * (double)cus)
        return false;
    if(pmod(M, t.tile_m) == 0 && pmod(N, t.tile_n) == 0)
    {
        tr.pad_m = false;
        tr.pad_n = false;
        tr.dtl_tight_rsrc = false;
        tr.pad_fast_tile = false;
        tr.dtl_soff0 = true;
        tr.epilogue = "cshuffle";
    }
    *tile = t;
    *trait = tr;
    return true;
}

/* dtl_ring_small_tile_trait(arch, K, M, N, dtype) (all given) */
void dtl_ring_small_tile_trait(int64_t M, int64_t N, int64_t K, Tile* tile, Trait* trait)
{
    if(small_tile_shortk_tile_trait(M, N, K, tile, trait))
        return;
    if(pmod(K, 64))
    {
        bool poor = tiles_of(M, N, 256, 128) <= DTL_RING_SMALL_K32_C128_MAX_WAVES * current_cus();
        if(poor)
        {
            *tile = mk_tile(128, 128, 32, 2, 2, 1, 16, 16, 32);
            *trait = s3ilv(ring_pad_trait());
            return;
        }
        *tile = mk_tile(256, 128, 32, 2, 2, 1, 16, 16, 32);
        Trait t = ring_pad_trait();
        t.acc_agpr = true;
        *trait = s3ilv(t);
        return;
    }
    *tile = mk_tile(128, 128, 64, 2, 4, 1, 16, 16, 32);
    if(K >= DTL_RING_SMALL_S3_MIN_K && tiles_of(M, N, 128, 128) <= current_cus())
    {
        *trait = s3ilv(ring_pad_trait());
        return;
    }
    *trait = ring_pad_trait();
}

void dtl_ring_pad_tile_trait(const Context& ctx, bool big_grid, Tile* tile, Trait* trait)
{
    if(big_grid)
    {
        agpr_ilv_big_grid_tile_trait(ctx, tile, trait);
        trait->pad_m = true;
        trait->pad_n = true;
        trait->dtl_tight_rsrc = true;
        trait->pad_fast_tile = true;
        return;
    }
    Trait dummy;
    dtl_pingpong_tile_trait(tile, &dummy);
    Trait t = ring_pad_trait();
    t.persistent_ctas = current_cus();
    *trait = t;
}

/* --------------------------------------------------------------- admissions */

/* the default (cshuffle) candidate's calibrated tile: 128 x 128 x 32 on gfx950 */
bool ring_split_k_reason(const Request& req)
{
    return select_split_k(req.M, req.N, req.K, 128, 128, 32) > 1;
}

bool aligned_to_default(const Request& req)
{
    return pmod(req.M, 128) == 0 && pmod(req.N, 128) == 0 && pmod(req.K, 32) == 0;
}

bool ragged_small_k(const Request& req)
{
    return !(pmod(req.M, 256) == 0 && pmod(req.N, 256) == 0) && req.K <= DTL_RING_SMALL_RAGGED_MAX_K
           && tiles_of(req.M, req.N, 256, 256) < DTL_RING_SMALL_RAGGED_MAX_TILES;
}

/* DTL_RING_ADMIT_RULES k_lo of (gfx950, dtype) */
int64_t ring_rule_k_lo(const std::string& dtype)
{
    return dtype == "bf16" ? 64 : 128;
}

bool dtl_ring_small_admit(const Request& req, const UGemm& spec)
{
    if(rilv_shortk_admits(req))
        return false;
    if(ring_split_k_reason(req))
        return false;
    double fill = wave_fill(req.M, req.N, 256, 256);
    bool rsk = ragged_small_k(req);
    if(fill >= DTL_RING_SMALL_MAX_FILL && !rsk)
        return false;
    const Tile& t = spec.tile;
    int64_t tiles = tiles_of(req.M, req.N, t.tile_m, t.tile_n);
    double need = DTL_RING_SMALL_MIN_TILES_PER_CU * (double)current_cus();
    if((double)tiles < need)
        return false;
    if(aligned_to_default(req))
    {
        int64_t k_min;
        if(t.tile_k == 64)
            k_min = ring_rule_k_lo(req.dtype);
        else if(spec.trait.acc_agpr)
            k_min = DTL_RING_SMALL_K32_AGPR_MIN_K;
        else
            k_min = DTL_RING_SMALL_K32_MIN_K;
        if(req.K < k_min)
            return false;
    }
    return true;
}

bool dtl_ring_pad_admit(const Request& req, const UGemm& spec)
{
    const Tile& t = spec.tile;
    if(pmod(req.M, t.tile_m) == 0 && pmod(req.N, t.tile_n) == 0)
        return false;
    if(ring_split_k_reason(req))
        return false;
    double fill = wave_fill(req.M, req.N, t.tile_m, t.tile_n);
    if(fill < DTL_RING_SMALL_MAX_FILL)
        return false;
    if(ragged_small_k(req))
        return false;
    return true;
}

/* TileCostModel.cost */
double tile_cost(int64_t tm,
                 int64_t tn,
                 double a,
                 double b,
                 double theta,
                 double gamma,
                 int64_t M,
                 int64_t N,
                 int64_t K,
                 int64_t cus)
{
    int64_t tiles = cdiv(M, tm) * cdiv(N, tn);
    int64_t n = cdiv(tiles, cus);
    double u = std::min(1.0, (double)tiles / (double)cus);
    return a * (double)n + b * (double)K * std::pow((double)n, theta) * std::pow(u, gamma);
}

bool dtl_pingpong_admit(const Request& req, const UGemm& spec)
{
    /* policy "auto" */
    if(rilv_shortk_admits(req))
        return true;
    bool ring = spec.trait.dtl_stages >= 2;
    /* (gfx950, dtype, body) cost models: (256x256 candidate, 128x128x32 default) */
    double fa, fb, fth, fga, da, db, dth, dga;
    if(req.dtype == "fp16")
    {
        if(ring)
        {
            fa = 399.6, fb = 1.1499, fth = 0.95, fga = 0.15;
            da = 83.8, db = 1.0, dth = 0.75, dga = 0.0;
        }
        else
        {
            fa = 422.5, fb = 1.6093, fth = 0.90, fga = 0.20;
            da = 88.4, db = 1.0, dth = 0.80, dga = 0.0;
        }
    }
    else
    {
        if(ring)
        {
            fa = 375.6, fb = 1.0786, fth = 0.95, fga = 0.15;
            da = 91.3, db = 1.0, dth = 0.80, dga = 0.0;
        }
        else
        {
            fa = 413.2, fb = 1.5272, fth = 0.90, fga = 0.20;
            da = 102.3, db = 1.0, dth = 0.85, dga = 0.0;
        }
    }
    const Tile& t = spec.tile;
    if(!(t.tile_m == 256 && t.tile_n == 256))
        return false;
    if(select_split_k(req.M, req.N, req.K, 128, 128, 32) > 1)
        return false;
    if(ring)
    {
        int64_t k_lo = req.dtype == "bf16" ? 64 : 128;
        int64_t k_hi = req.dtype == "bf16" ? 768 : 1024;
        int64_t t_min = req.dtype == "bf16" ? 144 : 100;
        int64_t tiles = cdiv(req.M, t.tile_m) * cdiv(req.N, t.tile_n);
        return req.K >= k_hi || (req.K >= k_lo && tiles >= t_min);
    }
    int64_t cus = current_cus();
    double margin = 0.95; /* (gfx950, pingpong) */
    double t_f = tile_cost(256, 256, fa, fb, fth, fga, req.M, req.N, req.K, cus);
    double t_d = tile_cost(128, 128, da, db, dth, dga, req.M, req.N, req.K, cus);
    (void)dth;
    return !(t_f > margin * t_d);
}

bool g1_ring_any_admit(const Request& req)
{
    if(pmod(req.K, G1_DTL_K_TAIL_ALIGN))
        return false;
    int64_t t128 = tiles_of(req.M, req.N, 128, 128);
    if(t128 < G1_RING_ANY_SPLIT_TMAX && req.K >= G1_RING_ANY_SPLIT_KMIN)
        return false;
    return true;
}

bool g1_ring_any_uses_pad_ring(const Request& req)
{
    return wave_fill(req.M, req.N, 256, 256) >= DTL_RING_SMALL_MAX_FILL
           && tiles_of(req.M, req.N, 256, 256) >= current_cus();
}

UGemm g1_ring_any_retile(const Request& req, const UGemm& spec)
{
    static const struct
    {
        int64_t tm, tn;
        double eff;
    } opts[] = {{128, 128, 1.0}, {128, 64, 0.6}, {64, 128, 0.6}, {64, 64, 0.4}};
    int64_t M = req.M, N = req.N;
    int64_t cus = current_cus();
    const Tile& t = spec.tile;
    bool have = false;
    double best_cost = 0.0;
    UGemm best;
    for(const auto& o : opts)
    {
        int64_t tiles = cdiv(M, o.tm) * cdiv(N, o.tn);
        double cost = (double)(cdiv(tiles, cus) * o.tm * o.tn) / o.eff;
        if(have && cost >= best_cost)
            continue;
        bool got = false;
        UGemm cand;
        const int64_t wl[6][2] = {{t.warp_m, t.warp_n}, {2, 4}, {2, 2}, {4, 2}, {1, 4}, {4, 1}};
        for(const auto& w : wl)
        {
            if(o.tm == t.tile_m && o.tn == t.tile_n && w[0] == t.warp_m && w[1] == t.warp_n)
            {
                cand = spec;
                got = true;
                break;
            }
            UGemm s2 = spec;
            s2.tile.tile_m = o.tm;
            s2.tile.tile_n = o.tn;
            s2.tile.warp_m = w[0];
            s2.tile.warp_n = w[1];
            s2.block_size = 0;
            s2.post_init();
            if(is_valid_spec_x(s2, req.arch, false, false, nullptr))
            {
                cand = s2;
                got = true;
                break;
            }
        }
        if(got)
        {
            have = true;
            best_cost = cost;
            best = cand;
        }
    }
    return have ? best : spec;
}

/* g1_pad_tile_trait (gfx950) */
void g1_pad_tile_trait(const Request& req, Tile* tile, Trait* trait)
{
    int64_t M = req.M, N = req.N, K = req.K;
    Tile t;
    if(M <= G1_PAD_SKINNY || N <= G1_PAD_SKINNY)
    {
        if(M <= N)
            t = mk_tile(16, 64, 32, 1, 4, 1, 16, 16, 32);
        else
            t = mk_tile(64, 16, 32, 4, 1, 1, 16, 16, 32);
    }
    else
        t = mk_tile(128, 128, 32, 2, 2, 1, 32, 32, 16);
    Trait tr;
    tr.pipeline = "compv4";
    tr.scheduler = "intrawave";
    tr.epilogue = "default";
    tr.pad_m = pmod(M, t.tile_m) != 0;
    tr.pad_n = pmod(N, t.tile_n) != 0;
    tr.pad_k = pmod(K, t.tile_k) != 0;
    *tile = t;
    *trait = tr;
}

/* universal_fallback_tile_trait (gfx950) */
void universal_fallback_tile_trait(int64_t M, int64_t N, int64_t K, Tile* tile, Trait* trait)
{
    Tile t = mk_tile(128, 128, 32, 2, 2, 1, 32, 32, 16);
    Trait tr;
    tr.pipeline = "compv4";
    tr.scheduler = "intrawave";
    tr.epilogue = "cshuffle";
    tr.pad_m = pmod(M, t.tile_m) != 0;
    tr.pad_n = pmod(N, t.tile_n) != 0;
    tr.pad_k = pmod(K, t.tile_k) != 0;
    tr.kc_row_guard = true;
    *tile = t;
    *trait = tr;
}

} // namespace

/* ------------------------------------------------------------ g1 adaptation */

UGemm g1_adapt_spec(const Request& req, const UGemm& spec)
{
    const Tile& t = spec.tile;
    const Trait& tr = spec.trait;
    int64_t M = req.M, N = req.N, K = req.K;
    Trait u = tr;
    bool changed = false;
    if(tr.direct_to_lds)
    {
        if(pmod(K, t.tile_k) && pmod(K, G1_DTL_K_TAIL_ALIGN) == 0
           && cdiv(K, t.tile_k) >= G1_DTL_K_TAIL_MIN_TILES && tr.dtl_stages >= 2 && tr.pingpong != 7
           && tr.asm_loop.empty() && !tr.streamk && tr.split_k == 1 && !spec.batched)
        {
            bool iglp = tr.iglp_opt.has_value();
            if(iglp)
            {
                /* G1_K_TAIL_DROP_IGLP */
                u.iglp_opt.reset();
                iglp = false;
            }
            bool pp_or_iglp = tr.pingpong != 0 || iglp;
            int64_t mode = pp_or_iglp ? 1 : G1_DTL_K_TAIL_MODE;
            u.pad_k = true;
            u.dtl_k_tail = mode;
            if(upper(spec.data.layout) != "RCR")
            {
                u.dtl_tight_rsrc = true;
                u.dtl_tight_ragged_only = false;
                u.dtl_soff0 = false;
            }
            u.k_stagger = 0;
            u.k_stagger_tiles = 0;
            u.k_stagger_rc = 0;
            u.stagger_u = 0;
            u.dtl_clamp_k = false;
            changed = true;
        }
    }
    else if(!tr.vgpr_prefetch)
    {
        bool rag = (tr.pad_m && pmod(M, t.tile_m)) || (tr.pad_n && pmod(N, t.tile_n));
        if(rag && !tr.preshuffle_b)
        {
            u.pad_row_clamp = true;
            changed = true;
        }
        if(tr.pad_k && pmod(K, t.tile_k))
        {
            u.pad_k_fast = true;
            changed = true;
        }
    }
    if(!changed)
        return spec;
    UGemm out = spec;
    out.trait = u;
    return out;
}

/* ------------------------------------------------------------- native forms */

namespace
{

UGemm ld_any_form(const UGemm& s)
{
    if(!uses_mn_dtl(s))
        return s;
    UGemm o = s;
    o.trait.mn_ld_any = true;
    return o;
}

/* native_mn_vec (0: None) */
int64_t native_mn_vec(const Request& req, const UGemm& spec)
{
    bool a_mn, b_mn;
    mn_layout_letters(req.layout, &a_mn, &b_mn);
    const Tile& t = spec.tile;
    for(int64_t v : {8, 4, 2, 1})
    {
        bool ok = true;
        bool mns[2] = {a_mn, b_mn};
        int64_t lds[2] = {req.M, req.N};
        int64_t tiles[2] = {t.tile_m, t.tile_n};
        for(int i = 0; i < 2; ++i)
            if(mns[i]
               && (pmod(lds[i], v) || pmod(tiles[i], v)
                   || pmod(fdiv(tiles[i] * t.tile_k, v), spec.block_size)))
                ok = false;
        if(ok)
            return v;
    }
    return 0;
}

bool layout_conflict_free(const UGemm& s, bool* cf)
{
    bool ha, hb;
    MnLayout la, lb;
    std::string w;
    if(!spec_mn_layouts(s, false, &ha, &la, &hb, &lb, &w))
        return false;
    bool ok = true;
    if(ha)
        ok = ok && la.conflict_free();
    if(hb)
        ok = ok && lb.conflict_free();
    *cf = ok;
    return true;
}

} // namespace

bool native_layout_alternatives(const Request& req,
                                const Context& ctx,
                                const UGemm& spec,
                                std::vector<std::pair<std::string, UGemm>>* out,
                                std::string* why)
{
    out->clear();
    std::string lay = upper(req.layout);
    if(lay == "RCR")
    {
        out->push_back({"same", spec});
        return true;
    }
    /* R2_VP_MN: gfx942 only */
    bool a_mn, b_mn;
    if(!mn_layout_letters(lay, &a_mn, &b_mn))
        return set_why(why, "unsupported GEMM layout");
    Data data = spec.data;
    data.layout = lay;
    Trait tr = spec.trait;
    const int64_t al = 8;
    bool ld8 = !((a_mn && pmod(req.M, al)) || (b_mn && pmod(req.N, al)));
    bool ld_any = !ld8 && req.arch == "gfx950";
    if(ld_any)
        ld8 = true;
    if(!uses_mn_dtl(spec) || !ld8 || req.arch != "gfx950")
    {
        UGemm base;
        if(!uses_mn_dtl(spec))
            base = spec;
        else
        {
            /* native_generic_spec */
            Tile t;
            Trait tt;
            universal_fallback_tile_trait(req.M, req.N, req.K, &t, &tt);
            base.name = spec.name;
            base.tile = t;
            base.trait = tt;
            base.data = spec.data;
            base.wave_size = spec.wave_size;
            base.block_size = 0;
            base.post_init();
        }
        int64_t v = native_mn_vec(req, base);
        if(v == 0)
            return set_why(why, "no MN load width fits the tile / leading dimension");
        UGemm g = base;
        g.data = data;
        g.trait.mn_vec = v;
        std::string w;
        if(is_valid_spec_x(g, req.arch, false, false, &w))
            out->push_back({"generic", g});
        else
            return set_why(why, "generic: " + w);
        return true;
    }
    if(tr.lds_row_interleave && tr.sched_strategy.empty())
        tr.sched_strategy = "iterative-minreg";
    std::vector<std::pair<std::string, UGemm>> cands;
    {
        UGemm s = spec;
        s.data = data;
        s.trait = tr;
        cands.push_back({"same", s});
    }
    if(tr.persistent_ctas && b_mn && !tr.lds_row_interleave)
    {
        UGemm s = spec;
        s.data = data;
        s.trait = tr;
        s.trait.persistent_ctas = 0;
        cands.insert(cands.begin(), {"grid", s});
    }
    if(tr.lds_row_interleave && tr.rilv_vec_epilogue && b_mn)
    {
        Trait nr = tr;
        nr.rilv_vec_epilogue = false;
        nr.rve_store_nt = false;
        nr.rve_pad = 0;
        {
            UGemm s = spec;
            s.data = data;
            s.trait = nr;
            cands.push_back({"no_rve", s});
        }
        {
            UGemm s = spec;
            s.data = data;
            s.trait = nr;
            s.trait.epilogue = "cshuffle";
            s.trait.persistent_ctas = 0;
            s.trait.pad_fast_tile = false;
            cands.push_back({"no_rve_csh", s});
        }
        if(!nr.pad_n && !nr.sia_sched)
        {
            UGemm s = spec;
            s.data = data;
            s.trait = nr;
            s.trait.mfma_swap_ab = true;
            cands.push_back({"no_rve_sab", s});
            /* NATIVE_LAYOUT_SAB_VEC16 */
            UGemm s2 = spec;
            s2.data = data;
            s2.trait = nr;
            s2.trait.mfma_swap_ab = true;
            s2.trait.sab_vec16 = true;
            cands.push_back({"no_rve_sv16", s2});
        }
    }
    bool cf = false;
    if(!layout_conflict_free(cands[0].second, &cf))
        cf = false;
    if(tr.lds_row_interleave || !cf)
    {
        /* NATIVE_LAYOUT_B3_FALLBACK */
        Tile tile;
        Trait trait;
        agpr_ilv_big_grid_tile_trait(ctx, &tile, &trait);
        trait.persistent_ctas = 0; /* NATIVE_LAYOUT_B3_PERSISTENT off */
        int64_t M = req.M, N = req.N;
        if(pmod(M, tile.tile_m) || pmod(N, tile.tile_n))
        {
            trait.pad_m = pmod(M, tile.tile_m) != 0;
            trait.pad_n = pmod(N, tile.tile_n) != 0;
            trait.dtl_tight_rsrc = true;
        }
        UGemm s;
        s.name = spec.name;
        s.tile = tile;
        s.trait = trait;
        s.data = data;
        s.wave_size = spec.wave_size;
        s.block_size = 0;
        s.post_init();
        cands.push_back({"b3", s});
    }
    if(!cf && !tr.lds_row_interleave)
    {
        Tile t0 = spec.tile;
        Tile ft = t0;
        ft.tile_m = t0.tile_n;
        ft.tile_n = t0.tile_m;
        ft.warp_m = t0.warp_n;
        ft.warp_n = t0.warp_m;
        bool pm = pmod(req.M, ft.tile_m) != 0, pn = pmod(req.N, ft.tile_n) != 0;
        Trait ftr = tr;
        ftr.pad_m = pm;
        ftr.pad_n = pn;
        if(pm || pn)
        {
            ftr.dtl_tight_rsrc = true;
            ftr.pad_fast_tile = true;
            ftr.dtl_soff0 = false;
            ftr.epilogue = "default";
        }
        UGemm fspec = spec;
        fspec.tile = ft;
        fspec.trait = ftr;
        fspec.data = data;
        bool fcf = false;
        if(layout_conflict_free(fspec, &fcf) && fcf)
            cands.push_back({"flip", fspec});
        UGemm r;
        r.name = spec.name;
        r.tile = mk_tile(128, 128, 64, 2, 4, 1, 16, 16, 32);
        r.trait = ring_pad_trait();
        r.data = data;
        r.wave_size = spec.wave_size;
        r.block_size = 0;
        r.post_init();
        cands.push_back({"r128", r});
    }
    /* NATIVE_LAYOUT_SAB_VEC16_RINGS / NATIVE_LAYOUT_SV16_NT off */
    static const std::vector<std::string> order = {
        "grid", "same", "no_rve", "no_rve_csh", "no_rve_sab", "flip", "b3", "r128", "no_rve_sv16"};
    auto pos = [&](const std::string& n) {
        auto it = std::find(order.begin(), order.end(), n);
        return it == order.end() ? (int)order.size() : (int)(it - order.begin());
    };
    std::stable_sort(cands.begin(), cands.end(), [&](const auto& a, const auto& b) {
        return pos(a.first) < pos(b.first);
    });
    if(b_mn && tr.lds_row_interleave && !tr.persistent_ctas && !tr.pad_n)
        std::stable_sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
            return (a.first != "no_rve_csh") < (b.first != "no_rve_csh");
        });
    if(!cf)
    {
        bool big = cdiv(req.M, 256) * cdiv(req.N, 256) >= current_cus();
        auto rank = [&](const std::string& n) {
            if(n == "flip")
                return 0;
            if(n == "b3")
                return big ? 1 : 2;
            if(n == "r128")
                return big ? 2 : 1;
            return 3;
        };
        std::stable_sort(cands.begin(), cands.end(), [&](const auto& a, const auto& b) {
            return rank(a.first) < rank(b.first);
        });
    }
    std::string whys;
    for(auto& c : cands)
    {
        UGemm s = g1_adapt_spec(req, c.second);
        if(ld_any)
            s = ld_any_form(s);
        std::string w;
        if(is_valid_spec_x(s, req.arch, false, false, &w))
            out->push_back({c.first, s});
        else
            whys += (whys.empty() ? "" : "; ") + c.first + ": " + w;
    }
    if(out->empty())
        set_why(why, whys);
    return true;
}

bool native_layout_spec(
    const Request& req, const Context& ctx, const UGemm& spec, UGemm* out, std::string* why)
{
    std::vector<std::pair<std::string, UGemm>> alts;
    std::string w;
    native_layout_alternatives(req, ctx, spec, &alts, &w);
    if(alts.empty())
        return set_why(why, w);
    if(upper(req.layout) != "RCR")
    {
        for(const auto& a : alts)
            if(launch_shape_error(a.second, req.M, req.N, req.K).empty())
            {
                *out = a.second;
                return true;
            }
    }
    *out = alts[0].second;
    return true;
}

namespace
{

/* apply_split_k (cdna) */
UGemm g1_aligned_split_k(const Request& req, const UGemm& spec)
{
    const Trait& tr = spec.trait;
    const Tile& t = spec.tile;
    if(!tr.pad_k || tr.direct_to_lds || tr.split_k > 1)
        return spec;
    int64_t M = req.M, N = req.N, K = req.K;
    int64_t base = cdiv(M, t.tile_m) * cdiv(N, t.tile_n);
    if(base >= fdiv(current_cus(), 2))
        return spec;
    int64_t want = pround((double)K / 512.0);
    int64_t sk = std::max((int64_t)1, std::min(want, cdiv(K, t.tile_k)));
    if(sk <= 1)
        return spec;
    UGemm o = spec;
    o.trait.split_k = sk;
    o.trait.split_k_aligned = true;
    if(pmod(K, t.tile_k))
        o.trait.pad_k_fast = true;
    return o;
}

UGemm apply_split_k(const Request& req, const UGemm& spec)
{
    const Tile& t = spec.tile;
    int64_t sk = select_split_k(req.M, req.N, req.K, t.tile_m, t.tile_n, t.tile_k);
    if(sk <= 1)
        return g1_aligned_split_k(req, spec);
    if(spec.trait.pad_k && !spec.trait.direct_to_lds)
    {
        UGemm alt = g1_aligned_split_k(req, spec);
        if(alt.trait.split_k >= 2 * sk)
            return alt;
    }
    UGemm o = spec;
    o.trait.split_k = sk;
    return o;
}

/* ------------------------------------------------------------- skinny ring rules */

int64_t ring_any_split(int64_t M, int64_t N, int64_t K, int64_t tm, int64_t tn, int64_t tk)
{
    int64_t tiles = cdiv(M, tm) * cdiv(N, tn);
    if(pmod(K, tk))
        return 1;
    if(!(K >= RING_ANY_SPLIT_MIN_K
         || (tiles <= RING_ANY_SPLIT_FEW_TILES && K >= RING_ANY_SPLIT_FEW_MIN_K)))
        return 1;
    int64_t want
        = std::min({RING_ANY_SPLIT_MAX,
                    pround((double)RING_ANY_SPLIT_CTAS / (double)std::max((int64_t)1, tiles)),
                    fdiv(K, RING_ANY_SPLIT_MIN_SLICE)});
    int64_t kt = fdiv(K, tk);
    for(int64_t d = std::max((int64_t)1, want); d > 0; --d)
        if(pmod(kt, d) == 0)
            return d;
    return 1;
}

UGemm ring_small_any_spec(const Request& req, const UGemm& spec)
{
    const Tile& t = spec.tile;
    int64_t M = req.M, N = req.N, K = req.K;
    Trait tr = spec.trait;
    tr.pad_m = tr.pad_m || pmod(M, t.tile_m) != 0;
    tr.pad_n = tr.pad_n || pmod(N, t.tile_n) != 0;
    int64_t split = ring_any_split(M, N, K, t.tile_m, t.tile_n, t.tile_k);
    if(split > 1)
    {
        tr.split_k = split;
        tr.pad_fast_tile = false;
    }
    else if(tr.pad_fast_tile && !(tr.pad_m || tr.pad_n))
        tr.pad_fast_tile = false;
    UGemm o = spec;
    o.trait = tr;
    return o;
}

bool ring_k_floor(const Request& req, const UGemm& spec)
{
    if(!aligned_to_default(req))
        return false;
    const Tile& t = spec.tile;
    int64_t k_min;
    if(t.tile_k == 64)
        k_min = ring_rule_k_lo(req.dtype);
    else if(spec.trait.acc_agpr)
        k_min = DTL_RING_SMALL_K32_AGPR_MIN_K;
    else
        k_min = DTL_RING_SMALL_K32_MIN_K;
    return req.K < k_min;
}

void r3_ring_split_degree(
    int64_t M, int64_t N, int64_t K, int64_t tm, int64_t tn, int64_t* s_out, int64_t* ser_out)
{
    int64_t tiles = cdiv(M, tm) * cdiv(N, tn);
    int64_t cap = std::max((int64_t)1, fdiv(K, std::max((int64_t)1, R3_RING_SPLIT_MIN_SLICE)));
    int64_t s = pround((double)R3_RING_SPLIT_SERIAL_CTAS / (double)std::max((int64_t)1, tiles));
    if(s < 2)
    {
        *s_out = 1;
        *ser_out = 0;
        return;
    }
    if(s <= R3_RING_SPLIT_SERIAL_MAX)
    {
        s = std::min(s, cap);
        if(s >= 2)
        {
            *s_out = s;
            *ser_out = R3_RING_SPLIT_SERIAL_MODE;
        }
        else
        {
            *s_out = 1;
            *ser_out = 0;
        }
        return;
    }
    s = std::max((int64_t)2,
                 std::min({R3_RING_SPLIT_ATOMIC_MAX,
                           cap,
                           pround((double)R3_RING_SPLIT_ATOMIC_CTAS
                                  / (double)std::max((int64_t)1, tiles))}));
    if(s <= cap)
    {
        *s_out = s;
        *ser_out = 0;
    }
    else
    {
        *s_out = 1;
        *ser_out = 0;
    }
}

UGemm r3_ring_split_spec(const Request& req, const UGemm& spec)
{
    const Tile& t = spec.tile;
    int64_t M = req.M, N = req.N, K = req.K;
    Trait tr = spec.trait;
    tr.pad_m = tr.pad_m || pmod(M, t.tile_m) != 0;
    tr.pad_n = tr.pad_n || pmod(N, t.tile_n) != 0;
    int64_t s, ser;
    r3_ring_split_degree(M, N, K, t.tile_m, t.tile_n, &s, &ser);
    if(s > 1 && pmod(K, t.tile_k) == 0)
    {
        tr.split_k = s;
        tr.split_k_serial = ser;
        tr.split_k_aligned = pmod(fdiv(K, t.tile_k), s) != 0;
        tr.pad_fast_tile = false;
    }
    else if(tr.pad_fast_tile && !(tr.pad_m || tr.pad_n))
        tr.pad_fast_tile = false;
    UGemm o = spec;
    o.trait = tr;
    return o;
}

bool ld_odd(const Request& req)
{
    std::string lay = upper(req.layout);
    bool a_mn = !lay.empty() && lay[0] == 'C', b_mn = lay.size() > 1 && lay[1] == 'R';
    return (a_mn && pmod(req.M, 8)) || (b_mn && pmod(req.N, 8));
}

bool r3_ring_split_admit(const Request& req, const UGemm& spec)
{
    if(dispatch_state().ring_split_bypass)
        return false;
    if(req.arch != "gfx950")
        return false;
    const Tile& t = spec.tile;
    int64_t M = req.M, N = req.N, K = req.K;
    int64_t tiles = cdiv(M, 128) * cdiv(N, 128);
    if(!(R3_RING_SPLIT_MIN_TILES <= tiles && tiles <= R3_RING_SPLIT_MAX_TILES))
        return false;
    if(K < R3_RING_SPLIT_MIN_K)
        return false;
    if(std::min(M, N) < R3_RING_SPLIT_MIN_SMALL)
        return false;
    if(ld_odd(req))
        return false;
    if(pmod(K, t.tile_k))
        return false;
    if(spec.trait.split_k < 2)
        return false;
    if(!is_valid_spec_x(spec, req.arch, false, false, nullptr))
        return false;
    if(!launch_shape_error(spec, M, N, K).empty())
        return false;
    return true;
}

bool ring_small_any_admit(const Request& req, const UGemm& spec)
{
    int64_t s = std::min(req.M, req.N);
    if(s < RING_ANY_MIN_SMALL)
        return false;
    if(ring_k_floor(req, spec))
        return false;
    if(!is_valid_spec_x(spec, req.arch, false, false, nullptr))
        return false;
    if(!launch_shape_error(spec, req.M, req.N, req.K).empty())
        return false;
    return true;
}

/* ------------------------------------------------------------------ sweeps */

std::vector<UGemm> dtl_ring_small_sweep_variants(const UGemm& spec, const std::string& arch)
{
    const Tile& t = spec.tile;
    Trait base = spec.trait;
    base.dtl_stages = 2;
    base.dtl_frag_prefetch = false;
    base.mfma_interleave = false;
    base.ilv_merge = false;
    base.ilv_barrier_after = 0;
    base.acc_agpr = false;
    std::vector<UGemm> cands;
    if(t.tile_k == 64)
    {
        UGemm a = spec;
        a.trait = base;
        cands.push_back(a);
        UGemm b = spec;
        b.trait = s3ilv(base);
        cands.push_back(b);
    }
    else
    {
        Tile c128 = t;
        c128.tile_m = 128;
        c128.tile_n = 128;
        c128.warp_m = 2;
        c128.warp_n = 2;
        Tile c256 = t;
        c256.tile_m = 256;
        c256.tile_n = 128;
        c256.warp_m = 2;
        c256.warp_n = 2;
        UGemm a = spec;
        a.tile = c128;
        a.trait = s3ilv(base);
        cands.push_back(a);
        UGemm b = spec;
        b.tile = c256;
        Trait bb = base;
        bb.acc_agpr = true;
        b.trait = s3ilv(bb);
        cands.push_back(b);
    }
    std::vector<UGemm> out = {spec};
    for(const auto& c : cands)
        if(!(c == spec) && std::find(out.begin(), out.end(), c) == out.end()
           && is_valid_spec_x(c, arch, false, false, nullptr))
            out.push_back(c);
    return out;
}

std::vector<UGemm> dtl_pingpong_sweep_variants(const UGemm& spec, const std::string& arch)
{
    const Trait& tr = spec.trait;
    std::vector<UGemm> out = {spec};
    std::vector<Trait> variants;
    {
        Trait a = tr;
        a.chiplet_chunk_size = 32;
        variants.push_back(a);
        Trait b = tr;
        b.chiplet_swizzle = false;
        variants.push_back(b);
        Trait c = tr;
        c.epilogue = "cshuffle";
        variants.push_back(c);
    }
    int64_t cus = current_cus();
    if(tr.dtl_stages >= 2)
    {
        Trait a = tr;
        a.dtl_stages = 0;
        a.interwave_setprio = false;
        variants.push_back(a);
        Trait b = tr;
        b.persistent_ctas = cus;
        b.chiplet_chunk_size = 32;
        variants.push_back(b);
        Trait c = tr;
        c.dtl_stream_barrier = true;
        c.interwave_setprio = false;
        variants.push_back(c);
        Trait d = tr;
        d.persistent_ctas = cus;
        d.chiplet_chunk_size = 32;
        d.dtl_hoist = true;
        variants.push_back(d);
        Trait e = tr;
        e.iglp_opt = 1;
        variants.push_back(e);
        Trait f = tr;
        f.iglp_opt = 1;
        f.chiplet_wgm = 4;
        variants.push_back(f);
    }
    for(const auto& t2 : variants)
    {
        if(t2 == tr)
            continue;
        UGemm cand = spec;
        cand.trait = t2;
        if(is_valid_spec_x(cand, arch, false, false, nullptr))
            out.push_back(cand);
    }
    if(tr.dtl_stages >= 2 && arch == "gfx950")
    {
        Tile ilv_tile = spec.tile;
        ilv_tile.tile_k = 32;
        ilv_tile.warp_m = 2;
        ilv_tile.warp_n = 2;
        Trait ilv = tr;
        ilv.dtl_stages = 3;
        ilv.dtl_frag_prefetch = true;
        ilv.acc_agpr = true;
        ilv.mfma_interleave = true;
        ilv.ilv_merge = true;
        ilv.ilv_barrier_after = 2;
        ilv.lds_swizzle_r = 1;
        ilv.dtl_hoist = true;
        ilv.chiplet_chunk_size = 64;
        for(int64_t wgm : {8, 4})
        {
            UGemm cand = spec;
            cand.tile = ilv_tile;
            cand.trait = ilv;
            cand.trait.chiplet_wgm = wgm;
            cand.block_size = 4 * spec.wave_size;
            if(is_valid_spec_x(cand, arch, false, false, nullptr))
                out.push_back(cand);
        }
    }
    if(tr.dtl_stages >= 2 && arch == "gfx950")
    {
        struct G
        {
            int64_t wgm, chunk, pc;
        };
        const std::vector<G> grids
            = {{16, 8, 0}, {16, 8, cus}, {8, 16, cus}, {8, 32, cus}, {8, 64, 0}};
        auto with_grid = [](Trait t, const G& g) {
            t.chiplet_wgm = g.wgm;
            t.chiplet_chunk_size = g.chunk;
            t.persistent_ctas = g.pc;
            return t;
        };
        const bool hbs[3][2] = {{false, false}, {true, false}, {true, true}};
        for(const auto& hs : hbs)
        {
            Tile bt;
            Trait btr;
            rilv_tile_trait(hs[0], true, hs[1], &bt, &btr);
            for(const auto& g : grids)
            {
                UGemm cand = spec;
                cand.tile = bt;
                cand.trait = with_grid(btr, g);
                cand.block_size = 4 * spec.wave_size;
                if(std::find(out.begin(), out.end(), cand) == out.end()
                   && is_valid_spec_x(cand, arch, false, false, nullptr))
                    out.push_back(cand);
            }
        }
        Tile at;
        Trait abase;
        rilv_tile_trait(false, true, false, &at, &abase);
        Trait atr = rilv_asm_loop_trait(abase, false);
        std::vector<G> g2 = grids;
        g2.push_back({16, 4, 0});
        for(const auto& g : g2)
        {
            UGemm cand = spec;
            cand.tile = at;
            cand.trait = with_grid(atr, g);
            cand.block_size = 4 * spec.wave_size;
            if(std::find(out.begin(), out.end(), cand) == out.end()
               && is_valid_spec_x(cand, arch, false, false, nullptr))
                out.push_back(cand);
        }
        for(int64_t ku : {RILV_KLOOP_UNROLL_SHORT, RILV_KLOOP_UNROLL_LONG})
        {
            Trait xt = rilv_tuned_trait(abase, ku);
            for(const auto& g : grids)
            {
                UGemm cand = spec;
                cand.tile = at;
                cand.trait = with_grid(xt, g);
                cand.block_size = 4 * spec.wave_size;
                if(std::find(out.begin(), out.end(), cand) == out.end()
                   && is_valid_spec_x(cand, arch, false, false, nullptr))
                    out.push_back(cand);
            }
        }
    }
    return out;
}

/* g1_sweep_filter */
std::vector<UGemm> g1_sweep_filter(const Request& req, const std::vector<UGemm>& specs)
{
    std::vector<UGemm> out;
    for(const auto& s0 : specs)
    {
        UGemm s = g1_adapt_spec(req, s0);
        if(!launch_shape_error(s, req.M, req.N, req.K).empty())
            continue;
        if(!is_valid_spec_x(s, req.arch, false, false, nullptr))
            continue;
        out.push_back(s);
    }
    return out;
}

/* ------------------------------------------------------- K-proxy / qgeo */

int64_t q_waves(int64_t M, int64_t N, int64_t tm, int64_t tn, int64_t per_cu, int64_t cus)
{
    return cdiv(cdiv(M, tm) * cdiv(N, tn), cus * per_cu);
}

/* r3_qgeo_spec (false: None) */
bool r3_qgeo_spec(const Request& req, const Spec& spec, UGemm* out)
{
    if(req.arch != "gfx950" || upper(req.layout) != "RCR")
        return false;
    if(spec.skinny || spec.u.batched)
        return false;
    const Tile& t = spec.u.tile;
    const Trait& tr = spec.u.trait;
    if(!(tr.direct_to_lds && t.tile_m == 128 && t.tile_n == 128 && t.tile_k == 64 && t.warp_m == 2
         && t.warp_n == 4 && t.warp_k == 1 && tr.split_k == 1 && !tr.persistent_ctas
         && !tr.streamk))
        return false;
    int64_t M = req.M, N = req.N, K = req.K;
    if(pmod(K, 64) == 0 || K < R3_QGEO_MIN_K) /* R3_QGEO_KTAIL_ONLY */
        return false;
    int64_t cus = current_cus();
    int64_t c_small = q_waves(M, N, 128, 128, 2, cus) * 128 * 128 * 2;
    bool have = false;
    int64_t bc = 0, btm = 0, btn = 0;
    for(auto p : {std::pair<int64_t, int64_t>{256, 192}, {192, 256}})
    {
        int64_t c = q_waves(M, N, p.first, p.second, 1, cus) * p.first * p.second;
        if((double)c_small >= R3_QGEO_MIN_RATIO * (double)c && (!have || c < bc))
        {
            have = true;
            bc = c;
            btm = p.first;
            btn = p.second;
        }
    }
    if(!have)
        return false;
    UGemm n = spec.u;
    n.tile = mk_tile(btm, btn, 64, 2, 2, 1, 16, 16, 32);
    Trait nt = s3ilv(ring_pad_trait());
    nt.dtl_stages = 2;
    nt.acc_agpr = false; /* R3_QGEO_AGPR */
    n.trait = nt;
    n.block_size = 0;
    n.post_init();
    n = g1_adapt_spec(req, n);
    if(!is_valid_spec_x(n, req.arch, false, false, nullptr)
       || !launch_shape_error(n, M, N, K).empty())
        return false;
    *out = n;
    return true;
}

} // namespace

/* skinny._ring_small_takes: the small DTL ring candidate's geometry, support and admission */
bool ring_small_takes(const Request& req)
{
    if(req.arch != "gfx950")
        return false;
    std::string dt = (req.dtype == "fp16" || req.dtype == "f16") ? "fp16" : "bf16";
    Tile t;
    Trait tr;
    dtl_ring_small_tile_trait(req.M, req.N, req.K, &t, &tr);
    UGemm spec = make_spec("ring_probe", dt, t, tr);
    return universal_spec_supported(spec, req.arch, nullptr)
           && request_shape_supported(req, spec, nullptr) && dtl_ring_small_admit(req, spec);
}

/* ------------------------------------------------------------- registries */

const Candidate* Registry::get(const std::string& name) const
{
    for(const auto& c : candidates)
        if(c.name == name)
            return &c;
    return nullptr;
}

namespace
{

/* rcr_request_errors (op "gemm", trans flags derived from the layout) */
bool rcr_request_ok(const Request& req, const std::string& dtype)
{
    if(req.M <= 0 || req.N <= 0 || req.K <= 0)
        return false;
    ArchFacts f;
    if(!arch_facts(req.arch, &f))
        return false;
    if(req.dtype != dtype)
        return false;
    std::string lay = upper(req.layout);
    return lay == "RCR" || lay == "RRR" || lay == "CRR" || lay == "CCR";
}

bool capability_ok(const Candidate& c, const Request& req)
{
    if(std::find(c.arches.begin(), c.arches.end(), req.arch) == c.arches.end())
        return false;
    std::string lay = upper(req.layout);
    if(!c.layouts.empty() && std::find(c.layouts.begin(), c.layouts.end(), lay) == c.layouts.end())
        return false;
    if(c.max_m >= 0 && req.M > c.max_m)
        return false;
    return true;
}

const std::vector<std::string> ALL_LAYOUTS = {"RCR", "RRR", "CRR", "CCR"};
const std::vector<std::string> RCR_ONLY = {"RCR"};

/* _make_candidate of fp16_rcr / bf16_rcr */
struct Rule
{
    std::string name, spec_id;
    int priority;
    std::vector<std::string> arches, layouts;
    int64_t max_m = -1;
    bool split_k = true;
    std::function<UGemm(const Request&)> spec_fn;
    std::function<bool(const Request&, const UGemm&)> admit; /* null: none */
    int sweep = 0; /* 0: select only, 1: ring_small variants, 2: pingpong variants */
};

Candidate
    make_universal(const Rule& r, const Context& ctx, const std::string& dtype, const Registry* reg)
{
    Candidate c;
    c.name = r.name;
    c.algorithm = "universal_gemm";
    c.spec_id = r.spec_id;
    c.priority = r.priority;
    c.arches = r.arches;
    c.layouts = r.layouts;
    c.max_m = r.max_m;
    Rule rule = r;
    Context cx = ctx;
    std::string dt = dtype;
    auto support = [rule, cx, dt](const Request& req, std::string* why) {
        if(!rcr_request_ok(req, dt))
            return set_why(why, "request errors");
        UGemm spec = g1_adapt_spec(req, rule.spec_fn(req));
        UGemm nspec;
        if(!native_layout_spec(req, cx, spec, &nspec, why))
            return false;
        if(!universal_spec_supported(nspec, req.arch, why))
            return false;
        if(!request_shape_supported(req, nspec, why))
            return false;
        if(rule.admit && !rule.admit(req, spec))
            return set_why(why, "admission rule");
        return true;
    };
    std::shared_ptr<Candidate> self = std::make_shared<Candidate>();
    c.admits = [support, rule](const Request& req, std::string* why) {
        Candidate tmp;
        tmp.arches = rule.arches;
        tmp.layouts = rule.layouts;
        tmp.max_m = rule.max_m;
        if(!capability_ok(tmp, req))
            return set_why(why, "capability");
        return support(req, why);
    };
    auto admits = c.admits;
    auto select = [rule, cx, admits](const Request& req, Spec* out, std::string* why) {
        if(!admits(req, why))
            return false;
        UGemm spec = g1_adapt_spec(req, rule.spec_fn(req));
        if(rule.split_k)
        {
            UGemm sk = apply_split_k(req, spec);
            if(!(sk == spec) && launch_shape_error(sk, req.M, req.N, req.K).empty())
            {
                UGemm nsk;
                std::string w;
                if(native_layout_spec(req, cx, sk, &nsk, &w))
                {
                    out->skinny = false;
                    out->u = nsk;
                    return true;
                }
            }
        }
        UGemm nspec;
        if(!native_layout_spec(req, cx, spec, &nspec, why))
            return false;
        out->skinny = false;
        out->u = nspec;
        return true;
    };
    c.select = select;
    c.sweep = [rule, select, admits](const Request& req) {
        std::vector<Spec> out;
        if(!admits(req, nullptr))
            return out;
        Spec base;
        if(!select(req, &base, nullptr))
            return out;
        if(rule.sweep == 0)
        {
            out.push_back(base);
            return out;
        }
        std::vector<UGemm> v = rule.sweep == 1 ? dtl_ring_small_sweep_variants(base.u, req.arch)
                                               : dtl_pingpong_sweep_variants(base.u, req.arch);
        for(const auto& s : g1_sweep_filter(req, v))
        {
            Spec sp;
            sp.u = s;
            out.push_back(sp);
        }
        return out;
    };
    (void)reg;
    (void)self;
    return c;
}

} // namespace

std::shared_ptr<Registry> make_registry(const std::string& dtype, const Context& ctx)
{
    auto reg = std::make_shared<Registry>();
    reg->dtype = dtype;
    const std::string p = "universal_gemm_" + dtype + "_";
    const std::vector<std::string> G950 = {"gfx950"};
    const std::vector<std::string> CDNA = {"gfx942", "gfx950"};
    std::vector<Rule> rules;
    auto ring_small = [dtype](const Request& req) {
        Tile t;
        Trait tr;
        dtl_ring_small_tile_trait(req.M, req.N, req.K, &t, &tr);
        return make_spec("", dtype, t, tr);
    };
    auto named = [](std::function<UGemm(const Request&)> f, const std::string& name) {
        return [f, name](const Request& req) {
            UGemm s = f(req);
            s.name = name;
            return s;
        };
    };
    Context cx = ctx;
    auto ring_pad = [dtype, cx](const Request& req) {
        Tile t;
        Trait tr;
        dtl_ring_pad_tile_trait(cx, dtl_big_grid_body(req, true) == 1, &t, &tr);
        if(rilv_pad_admits(req))
            rilv_pad_tile_trait(req, &t, &tr);
        return make_spec("", dtype, t, tr);
    };
    Registry* rp = reg.get();
    {
        Rule r;
        r.name = p + "cdna_ring_split";
        r.spec_id = "cdna_ring_split";
        r.priority = 2;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(
            [ring_small](const Request& req) { return r3_ring_split_spec(req, ring_small(req)); },
            r.name);
        r.admit = [](const Request& req, const UGemm& s) { return r3_ring_split_admit(req, s); };
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_ring_small_any";
        r.spec_id = "cdna_ring_small_any";
        r.priority = 9;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(
            [ring_small](const Request& req) { return ring_small_any_spec(req, ring_small(req)); },
            r.name);
        r.admit = [](const Request& req, const UGemm& s) { return ring_small_any_admit(req, s); };
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_dtl_ring_small";
        r.spec_id = "cdna_dtl_ring_small";
        r.priority = 4;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(ring_small, r.name);
        r.admit = [](const Request& req, const UGemm& s) { return dtl_ring_small_admit(req, s); };
        r.sweep = 1;
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_dtl_ring_pad";
        r.spec_id = "cdna_dtl_ring_pad";
        r.priority = 6;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(ring_pad, r.name);
        r.admit = [](const Request& req, const UGemm& s) { return dtl_ring_pad_admit(req, s); };
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_dtl_pingpong";
        r.spec_id = "cdna_dtl_pingpong_256x256";
        r.priority = 5;
        r.arches = CDNA;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(
            [dtype, cx](const Request& req) {
                Tile t;
                Trait tr;
                dtl_pingpong_tile_trait(&t, &tr);
                Tile bt;
                Trait btr;
                if(big_grid_tile_trait(req, cx, &bt, &btr))
                {
                    t = bt;
                    tr = btr;
                }
                if(rilv_admits(req))
                    rilv_spec_tile_trait(req, &t, &tr);
                else if(rilv_shortk_admits(req))
                    rilv_shortk_tile_trait(req, &t, &tr);
                return make_spec("", dtype, t, tr);
            },
            r.name);
        r.admit = [](const Request& req, const UGemm& s) { return dtl_pingpong_admit(req, s); };
        r.sweep = 2;
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_cshuffle";
        r.spec_id = "cdna_cshuffle_default";
        r.priority = 10;
        r.arches = dtype == "bf16" ? std::vector<std::string>{"gfx90a", "gfx942", "gfx950"} : CDNA;
        r.layouts = ALL_LAYOUTS;
        r.split_k = true;
        r.spec_fn = named(
            [dtype](const Request& req) {
                (void)req;
                Trait tr;
                tr.pipeline = "compv4";
                tr.scheduler = "intrawave";
                tr.epilogue = "cshuffle";
                Tile t = dtype == "bf16" ? mk_tile(128, 128, 32, 4, 4, 1, 16, 16, 32)
                                         : mk_tile(128, 128, 32, 2, 2, 1, 32, 32, 16);
                return make_spec("", dtype, t, tr);
            },
            r.name);
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_mem";
        r.spec_id = "cdna_mem_64x128";
        r.priority = 20;
        r.arches = CDNA;
        r.layouts = ALL_LAYOUTS;
        r.split_k = true;
        r.spec_fn = named(
            [dtype](const Request& req) {
                (void)req;
                Trait tr;
                tr.pipeline = "mem";
                tr.scheduler = "intrawave";
                tr.epilogue = "default";
                return make_spec("", dtype, mk_tile(64, 128, 32, 2, 2, 1, 16, 16, 32), tr);
            },
            r.name);
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_universal_fallback";
        r.spec_id = "cdna_universal_fallback";
        r.priority = 100;
        r.arches = CDNA;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(
            [dtype](const Request& req) {
                Tile t;
                Trait tr;
                universal_fallback_tile_trait(req.M, req.N, req.K, &t, &tr);
                return make_spec("", dtype, t, tr);
            },
            r.name);
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_decode";
        r.spec_id = "cdna_decode_16x64";
        r.priority = 30;
        r.arches = dtype == "bf16" ? G950 : CDNA;
        r.layouts = dtype == "bf16" ? ALL_LAYOUTS : RCR_ONLY;
        r.max_m = 32;
        r.split_k = true;
        r.spec_fn = named(
            [dtype](const Request& req) {
                (void)req;
                Trait tr;
                tr.pipeline = "compv4";
                tr.scheduler = "intrawave";
                tr.epilogue = "default";
                tr.pad_m = true;
                tr.pad_n = true;
                tr.pad_k = true;
                return make_spec("", dtype, mk_tile(16, 64, 32, 1, 4, 1, 16, 16, 32), tr);
            },
            r.name);
        rules.push_back(r);
    }
    auto ring_any = [ring_small, ring_pad](const Request& req) {
        if(g1_ring_any_uses_pad_ring(req))
            return ring_pad(req);
        return g1_ring_any_retile(req, ring_small(req));
    };
    {
        Rule r;
        r.name = p + "cdna_dtl_ring_any_first";
        r.spec_id = "cdna_dtl_ring_any_first";
        r.priority = 8;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        std::string small_name = p + "cdna_ring_small_any";
        r.spec_fn = named(ring_any, r.name);
        r.admit = [rp, small_name](const Request& req, const UGemm& s) {
            (void)s;
            if(!g1_ring_any_admit(req))
                return false;
            const Candidate* c = rp->get(small_name);
            if(c == nullptr)
                return false;
            return c->admits(req, nullptr);
        };
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_dtl_ring_any";
        r.spec_id = "cdna_dtl_ring_any";
        r.priority = 35;
        r.arches = G950;
        r.layouts = ALL_LAYOUTS;
        r.split_k = false;
        r.spec_fn = named(ring_any, r.name);
        r.admit = [](const Request& req, const UGemm& s) {
            (void)s;
            return g1_ring_any_admit(req);
        };
        rules.push_back(r);
    }
    {
        Rule r;
        r.name = p + "cdna_pad";
        r.spec_id = "cdna_pad_fallback";
        r.priority = 40;
        r.arches = CDNA;
        r.layouts = RCR_ONLY;
        r.split_k = true;
        r.spec_fn = named(
            [dtype](const Request& req) {
                Tile t;
                Trait tr;
                g1_pad_tile_trait(req, &t, &tr);
                return make_spec("", dtype, t, tr);
            },
            r.name);
        r.admit = [](const Request&, const UGemm&) { return true; };
        rules.push_back(r);
    }
    for(const auto& r : rules)
        reg->candidates.push_back(make_universal(r, ctx, dtype, rp));
    reg->candidates.push_back(make_skinny_candidate(p + "cdna_skinny", dtype, false, ctx));
    reg->candidates.push_back(make_skinny_candidate(p + "cdna_skinny_fallback", dtype, true, ctx));
    /* the opt-in selected candidates (dispatch_override targets; never in select / combos) */
    {
        Candidate c;
        c.name = p + "cdna_selected";
        c.algorithm = "universal_gemm";
        c.spec_id = "cdna_selected";
        c.priority = 1;
        c.opt_in = true;
        c.arches = CDNA;
        c.layouts = ALL_LAYOUTS;
        c.admits = [](const Request&, std::string* why) { return set_why(why, "opt-in"); };
        c.select = [](const Request&, Spec*, std::string* why) { return set_why(why, "opt-in"); };
        c.sweep = [](const Request&) { return std::vector<Spec>{}; };
        reg->candidates.push_back(c);
        c.name = p + "cdna_selected_skinny";
        c.algorithm = "gemm_skinny";
        c.spec_id = "cdna_selected_skinny";
        c.arches = G950;
        c.layouts = RCR_ONLY;
        reg->candidates.push_back(c);
    }
    std::stable_sort(
        reg->candidates.begin(), reg->candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.priority != b.priority ? a.priority < b.priority : a.name < b.name;
        });
    return reg;
}

bool registry_select(const Registry& reg,
                     const Request& req,
                     const Candidate** out,
                     std::string* why)
{
    for(const auto& c : reg.candidates)
    {
        if(c.opt_in)
            continue;
        if(c.admits(req, nullptr))
        {
            *out = &c;
            return true;
        }
    }
    return set_why(why, "no candidate supports request");
}

std::vector<std::pair<const Candidate*, Spec>> registry_combos(const Registry& reg,
                                                               const Request& req)
{
    std::vector<std::pair<const Candidate*, Spec>> out;
    for(const auto& c : reg.candidates)
    {
        if(c.opt_in)
            continue;
        if(std::find(c.arches.begin(), c.arches.end(), req.arch) == c.arches.end())
            continue;
        if(!c.admits(req, nullptr))
            continue;
        std::vector<Spec> specs = c.sweep(req);
        if(specs.empty())
        {
            Spec s;
            if(c.select(req, &s, nullptr))
                out.push_back({&c, s});
            continue;
        }
        for(const auto& s : specs)
            out.push_back({&c, s});
    }
    return out;
}

} // namespace detail

using namespace detail;

/* ------------------------------------------------------------------ dispatch */

namespace
{

bool dispatch_impl(const Request& req, const Context& ctx, Pick* out, std::string* why);

/* r3_kproxy_dispatch (false: None) */
bool kproxy_dispatch(const Request& req, const Context& ctx, Pick* out)
{
    DispatchState& st = dispatch_state();
    if(st.kproxy_active)
        return false;
    st.kproxy_active += 1;
    struct Guard
    {
        DispatchState& s;
        ~Guard()
        {
            s.kproxy_active -= 1;
        }
    } g{st};
    if(req.arch != "gfx950" || upper(req.layout) != "RCR")
        return false;
    int64_t K = req.K;
    if(pmod(K, R3_KPROXY_ALIGN) == 0 || pmod(K, G1_DTL_K_TAIL_ALIGN) || K < R3_KPROXY_MIN_K)
        return false;
    int64_t kp = cdiv(K, R3_KPROXY_ALIGN) * R3_KPROXY_ALIGN;
    Request rk = req;
    rk.K = kp;
    Pick res;
    if(!dispatch_impl(rk, ctx, &res, nullptr))
        return false;
    if(!res.spec.skinny && res.spec_id == "cdna_ring_split" && res.spec.u.trait.split_k > 1)
    {
        st.ring_split_bypass += 1;
        bool ok = dispatch_impl(rk, ctx, &res, nullptr);
        st.ring_split_bypass -= 1;
        if(!ok)
            return false;
    }
    if(res.spec.skinny || !res.spec.u.trait.direct_to_lds)
        return false;
    UGemm adapted = g1_adapt_spec(req, res.spec.u);
    if(!adapted.trait.dtl_k_tail)
        return false;
    if(!is_valid_spec_x(adapted, req.arch, false, false, nullptr)
       || !launch_shape_error(adapted, req.M, req.N, K).empty())
        return false;
    out->candidate = res.candidate;
    out->spec_id = res.spec_id;
    out->spec.skinny = false;
    out->spec.u = adapted;
    return true;
}

bool dispatch_impl(const Request& req, const Context& ctx, Pick* out, std::string* why)
{
    std::shared_ptr<Registry> reg = make_registry(req.dtype, ctx);
    Pick kp;
    bool have_kp = kproxy_dispatch(req, ctx, &kp);
    const Candidate* cand = nullptr;
    Spec spec;
    if(have_kp)
    {
        *out = kp;
        spec = kp.spec;
    }
    else
    {
        if(!registry_select(*reg, req, &cand, why))
            return false;
        if(!cand->select(req, &spec, why))
            return false;
        dispatch_override(req, ctx, *reg, cand, spec);
        out->candidate = cand->name;
        out->spec_id = cand->spec_id;
    }
    UGemm q;
    if(r3_qgeo_spec(req, spec, &q))
    {
        spec.skinny = false;
        spec.u = q;
    }
    out->spec = spec;
    return true;
}

} // namespace

namespace
{
/* dispatch_gemm is a pure function of (request, CU count, models): the planner asks for the
 * same GEMM many times (every body, layout and route of a contraction), so each thread keeps
 * its decisions */
struct MemoKey
{
    int64_t M, N, K, cus;
    std::string arch, dtype, layout;
    const Models* models;
    uint64_t serial;
    bool operator<(const MemoKey& o) const
    {
        return std::tie(M, N, K, cus, arch, dtype, layout, models, serial)
               < std::tie(o.M, o.N, o.K, o.cus, o.arch, o.dtype, o.layout, o.models, o.serial);
    }
};
struct MemoValue
{
    bool ok;
    Pick pick;
    std::string why;
};
constexpr size_t k_memo_limit = 4096;
} // namespace

bool dispatch_gemm(const Request& req, const Context& ctx, Pick* out, std::string* why)
{
    if(req.dtype != "fp16" && req.dtype != "bf16")
        return set_why(why, "unsupported dtype '" + req.dtype + "'");
    thread_local std::map<MemoKey, MemoValue> memo;
    MemoKey key{req.M,
                req.N,
                req.K,
                ctx.cus,
                req.arch,
                req.dtype,
                req.layout,
                ctx.models,
                ctx.models ? ctx.models->serial : 0};
    auto it = memo.find(key);
    if(it == memo.end())
    {
        MemoValue v;
        {
            CusScope scope(ctx.cus);
            v.ok = dispatch_impl(req, ctx, &v.pick, &v.why);
        }
        if(memo.size() >= k_memo_limit)
            memo.clear();
        it = memo.emplace(std::move(key), std::move(v)).first;
    }
    if(!it->second.ok)
        return set_why(why, it->second.why);
    *out = it->second.pick;
    return true;
}

} // namespace gemm_dispatch
} // namespace rocke
