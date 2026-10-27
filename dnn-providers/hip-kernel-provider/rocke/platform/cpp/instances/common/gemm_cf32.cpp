// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gemm_cf32.cpp -- the named universal GEMM bodies of dispatch/gemm/cf32.py
 * (cf32_body: _tile, _ring, _RI, _ILV) and its launch adjustment _adjust_k,
 * for C++ callers that build those kernels without the Python dispatch layer
 * (see rocke/instance_gemm_cf32.h). The gemm_cf32 parity family builds every
 * body through this table and through cf32.py and compares the IR.
 */
#include "rocke/instance_gemm_cf32.h"

#include "rocke/arch_num_cus.h"

#include <string.h>

namespace
{

/* Python _tile(tm, tn, tk, wm, wn) */
rocke_gemm_tile_spec_t tile(int tm, int tn, int tk, int wm, int wn)
{
    rocke_gemm_tile_spec_t t;
    memset(&t, 0, sizeof t);
    t.tile_m = tm;
    t.tile_n = tn;
    t.tile_k = tk;
    t.warp_m = wm;
    t.warp_n = wn;
    t.warp_k = 1;
    t.warp_tile_m = 16;
    t.warp_tile_n = 16;
    t.warp_tile_k = 32;
    return t;
}

/* Python _ring(): the TraitSpec defaults plus the ring body's base levers */
rocke_gemm_trait_spec_t ring()
{
    rocke_gemm_trait_spec_t tr = rocke_gemm_universal_spec_default().trait;
    tr.pipeline = "compv4";
    tr.scheduler = "intrawave";
    tr.epilogue = "default";
    tr.direct_to_lds = true;
    tr.dtl_prefetch = true;
    tr.lds_swizzle = true;
    tr.dtl_cache_a = 0;
    tr.dtl_cache_b = 0;
    tr.chiplet_swizzle = true;
    tr.chiplet_wgm = 8;
    tr.chiplet_chunk_size = 64;
    tr.dtl_stages = 2;
    tr.interwave_setprio = true;
    return tr;
}

/* Python _RI: the row-interleave 2-stage AGPR body */
void apply_ri(rocke_gemm_trait_spec_t* tr)
{
    tr->lds_swizzle = false;
    tr->interwave_setprio = false;
    tr->dtl_stages = 2;
    tr->acc_agpr = true;
    tr->dtl_frag_prefetch = true;
    tr->mfma_interleave = true;
    tr->dtl_hoist = true;
    tr->lds_row_interleave = true;
    tr->rilv_vec_epilogue = true;
    tr->rve_store_nt = true;
    tr->ilv_merge = true;
    tr->ilv_barrier_after = 2;
    tr->chiplet_wgm = 16;
    tr->chiplet_chunk_size = 8;
}

/* Python _ILV: the interleaved ring levers */
void apply_ilv(rocke_gemm_trait_spec_t* tr)
{
    tr->dtl_frag_prefetch = true;
    tr->mfma_interleave = true;
    tr->ilv_merge = true;
    tr->ilv_barrier_after = 2;
    tr->lds_swizzle_r = 1;
    tr->dtl_hoist = true;
}

} // namespace

extern "C" {

bool rocke_gemm_cf32_body(const char* name, const char* arch, rocke_gemm_universal_spec_t* spec)
{
    if(name == NULL || spec == NULL)
        return false;
    int cus = rocke_arch_num_cus(arch != NULL ? arch : "gfx950");
    rocke_gemm_trait_spec_t tr = ring();
    if(strcmp(name, "RIW") == 0)
    {
        spec->tile = tile(256, 256, 64, 2, 2);
        apply_ri(&tr);
        spec->block_size = 256;
    }
    else if(strcmp(name, "RIW_SK") == 0)
    {
        spec->tile = tile(256, 256, 64, 2, 2);
        apply_ri(&tr);
        tr.chiplet_chunk_size = 32;
        tr.persistent_ctas = cus;
        tr.streamk = true;
        spec->block_size = 256;
    }
    else if(strcmp(name, "Q3_SK") == 0)
    {
        spec->tile = tile(128, 128, 64, 2, 2);
        tr.dtl_stages = 3;
        tr.persistent_ctas = cus;
        tr.streamk = true;
        apply_ilv(&tr);
        spec->block_size = 256;
    }
    else if(strcmp(name, "C3") == 0)
    {
        /* 128 x 128 x 64, 8 waves of 64 x 32, 3-stage ring, data parallel */
        spec->tile = tile(128, 128, 64, 2, 4);
        tr.dtl_stages = 3;
        apply_ilv(&tr);
        spec->block_size = 512;
    }
    else
    {
        return false;
    }
    spec->trait = tr;
    return true;
}

bool rocke_gemm_cf32_adjust_k(rocke_gemm_universal_spec_t* spec,
                              long long M,
                              long long Nk,
                              bool wide)
{
    if(spec == NULL)
        return false;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    bool rm = M % t->tile_m != 0, rn = Nk % t->tile_n != 0;
    if(rm || rn)
        return false; /* dtl_tight_ragged_only / rve_pad: Python engine only */
    if(wide)
    {
        rocke_gemm_trait_spec_t* tr = &spec->trait;
        tr->wide_addr = true;
        tr->persistent_ctas = 0;
        tr->streamk = false;
        if(!(tr->dtl_soff0 || tr->dtl_tight_rsrc))
            tr->dtl_soff0 = true;
    }
    return true;
}

} /* extern "C" */
