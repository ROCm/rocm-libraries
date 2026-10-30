// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gemm_cf32.cpp -- the named universal GEMM bodies of dispatch/gemm/cf32.py
 * (cf32_body: _tile, _ring, _RI, _ILV; the gfx942 register-staged bodies
 * _vp_body / _VP_BODIES), its launch adjustment _adjust_k, the constants of its
 * gfx942 rules (CF32_COST, CF32_SPLIT_MODE, CF32_NATIVE_FIRST) and the launch
 * limits of a body (gemm_launch_limits), for C++ callers that build those
 * kernels without the Python dispatch layer (see rocke/instance_gemm_cf32.h).
 * The gemm_cf32 parity family builds every body through this table and
 * through cf32.py and compares the IR.
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

/* Python _VP_BODIES: tile M, N, K, waves M x N and the levers on _vp_body's
 * base trait: AGPR accumulators, the main-loop levers (_LEV) with their release
 * window (0: the default), vp_unroll, lds_swizzle_l (-1: the default) */
struct VpBody
{
    const char* name;
    int tm, tn, tk, wm, wn;
    bool agpr, lev;
    int release, unroll, swl;
};
const VpBody kVpBodies[] = {
    {"G256x256x64L3", 256, 256, 64, 2, 2, true, true, 3, 0, -1},
    {"G256x224x64L3", 256, 224, 64, 2, 2, true, true, 3, 0, -1},
    {"G256x224x64L1", 256, 224, 64, 2, 2, true, true, 0, 0, -1},
    {"G192x128x64L3", 192, 128, 64, 4, 2, false, true, 3, 0, -1},
    {"G192x128x64L1S3", 192, 128, 64, 4, 2, false, true, 0, 0, 3},
    {"G192x128x64US3", 192, 128, 64, 4, 2, false, false, 0, 1, 3},
    {"G128x128x128L3", 128, 128, 128, 2, 4, false, true, 3, 0, -1},
    {"G128x64x64", 128, 64, 64, 4, 2, false, false, 0, 0, -1},
    {"G128x64x64U", 128, 64, 64, 4, 2, false, false, 0, 1, -1},
    {"G64x64x128L3", 64, 64, 128, 2, 2, false, true, 3, 0, -1},
    {"G64x32x128U", 64, 32, 128, 4, 1, false, false, 0, 1, -1},
    {"G32x32x128L3", 32, 32, 128, 2, 2, false, true, 3, 0, -1},
    {"G32x32x128U", 32, 32, 128, 2, 2, false, false, 0, 1, -1},
};

const VpBody* vp_body_of(const char* name)
{
    for(const VpBody& v : kVpBodies)
        if(strcmp(name, v.name) == 0)
            return &v;
    return NULL;
}

/* Python _vp_body(name): vgpr_prefetch 4 (one LDS buffer), swizzle row shift 0,
 * 16 x 16 x 16 atoms, then the body's levers */
void vp_body(const VpBody* v, rocke_gemm_universal_spec_t* spec)
{
    spec->tile = tile(v->tm, v->tn, v->tk, v->wm, v->wn);
    spec->tile.warp_tile_k = 16;
    rocke_gemm_trait_spec_t tr = rocke_gemm_universal_spec_default().trait;
    tr.pipeline = "compv4";
    tr.scheduler = "intrawave";
    tr.epilogue = "default";
    tr.ab_double_buffer_set = true;
    tr.ab_double_buffer = false;
    tr.chiplet_swizzle = true;
    tr.lds_swizzle = true;
    tr.lds_swizzle_r = 0;
    tr.vgpr_prefetch = 4;
    if(v->lev)
    {
        tr.vp_tail_pack = 3;
        tr.vp_pair_reads = 1;
        tr.vp_mfma_first = 1;
    }
    if(v->agpr)
        tr.acc_agpr = true;
    if(v->release > 0)
        tr.vp_release_atoms = v->release;
    if(v->unroll > 0)
        tr.vp_unroll = v->unroll;
    if(v->swl >= 0)
        tr.lds_swizzle_l = v->swl;
    spec->trait = tr;
    spec->block_size = 64 * v->wm * v->wn;
}

/* Python CF32_COST: (a, b, occ) of the "cost" rule per route and body */
struct BodyCost
{
    const char* route;
    const char* body;
    double a, b;
    int occ;
};
const BodyCost kCosts[] = {
    {"h16", "G64x64x128L3", 787.652, 1.0, 1},
    {"h16", "G128x128x128L3", 886.031, 2.41898, 1},
    {"h16", "G64x32x128U", 802.0, 0.686454, 1},
    {"h16", "G192x128x64L3", 1037.05, 3.39667, 1},
    {"h16", "G256x256x64L3", 2169.68, 6.12695, 1},
    {"h16", "G32x32x128U", 883.11, 0.417509, 1},
    {"h16", "G256x224x64L3", 1953.16, 5.59897, 1},
    {"x6", "G64x64x128L3", 163.696, 1.0, 1},
    {"x6", "G32x32x128L3", 215.608, 0.37267, 1},
    {"x6", "G128x128x128L3", 56.4924, 2.9503, 1},
    {"x6", "G192x128x64L1S3", -0.211177, 4.06212, 1},
    {"x6", "G256x224x64L1", 248.237, 7.69944, 1},
    {"x6", "G128x64x64", 94.9866, 1.75501, 1},
    {"x6", "G64x32x128U", 204.381, 0.642756, 1},
    {"cf32", "G64x64x128L3", 168.401, 1.0, 1},
    {"cf32", "G32x32x128L3", 216.253, 0.374793, 1},
    {"cf32", "G128x128x128L3", 38.5741, 2.94739, 1},
    {"cf32", "G192x128x64US3", 16.9452, 4.06542, 1},
    {"cf32", "G256x224x64L1", 233.056, 7.51937, 1},
    {"cf32", "G128x64x64U", 76.2714, 1.76291, 1},
    {"cf32", "G64x32x128U", 207.373, 0.649071, 1},
};

/* Python CF32_ROUTE_ALIAS: the routes on the coefficients of another route */
const char* route_alias(const char* route)
{
    static const char* const alias[][2] = {{"x3", "x6"},
                                           {"x4", "x6"},
                                           {"x9", "x6"},
                                           {"cf32x3", "cf32"},
                                           {"cf32x4", "cf32"},
                                           {"cf32x9", "cf32"}};
    for(const auto& a : alias)
        if(strcmp(route, a[0]) == 0)
            return a[1];
    return route;
}

} // namespace

extern "C" {

bool rocke_gemm_cf32_body(const char* name, const char* arch, rocke_gemm_universal_spec_t* spec)
{
    if(name == NULL || spec == NULL)
        return false;
    if(const VpBody* v = vp_body_of(name))
    {
        vp_body(v, spec);
        return true;
    }
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
    if(spec->trait.vgpr_prefetch)
    {
        /* the register-staged body: buffer ranges end at the operand (pad_m /
         * pad_n bound the ragged tiles); 64-bit batch and tile origins */
        spec->trait.pad_m = spec->trait.pad_m || rm;
        spec->trait.pad_n = spec->trait.pad_n || rn;
        if(wide)
            spec->trait.wide_addr = true;
        return true;
    }
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

bool rocke_gemm_cf32_body_cost(const char* body, const char* route, double* a, double* b, int* occ)
{
    if(body == NULL || route == NULL)
        return false;
    const char* r = route_alias(route);
    for(const BodyCost& c : kCosts)
    {
        if(strcmp(r, c.route) == 0 && strcmp(body, c.body) == 0)
        {
            if(a != NULL)
                *a = c.a;
            if(b != NULL)
                *b = c.b;
            if(occ != NULL)
                *occ = c.occ;
            return true;
        }
    }
    return false;
}

const char* rocke_gemm_cf32_split_mode(const char* arch, const char* route)
{
    if(arch != NULL && route != NULL && strcmp(arch, "gfx942") == 0)
    {
        static const char* const trunc[] = {"x6", "x9", "cf32", "cf32x9"};
        for(const char* r : trunc)
            if(strcmp(route, r) == 0)
                return "trunc";
    }
    return "rne";
}

double rocke_gemm_cf32_native_first(const char* arch, const char* route)
{
    if(arch == NULL || route == NULL || strcmp(arch, "gfx942") != 0)
        return 0.0;
    if(strcmp(route, "x6") == 0)
        return 2.9e10;
    if(strcmp(route, "cf32") == 0)
        return 3.4e7;
    return 0.0;
}

void rocke_gemm_cf32_launch_limits(const rocke_gemm_universal_spec_t* spec,
                                   rocke_gemm_cf32_limits_t* out)
{
    if(out == NULL)
        return;
    const long long range = ROCKE_GEMM_UNIVERSAL_DTL_BUFFER_NUM_RECORDS;
    if(spec != NULL && spec->trait.vgpr_prefetch)
    {
        out->operand_bytes = 1LL << 32;
        out->c_elements = 1LL << 31;
    }
    else
    {
        out->operand_bytes = range;
        out->c_elements = (1LL << 31) - 1;
    }
    out->tile_operand_bytes = range;
    out->c_tile_elements = 1LL << 31;
    out->grid_yz = 65535;
}

} /* extern "C" */
