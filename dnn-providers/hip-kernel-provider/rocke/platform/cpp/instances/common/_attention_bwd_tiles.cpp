// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_tiles.py: the small-tile
 * warp grids and the occupancy bound of the attention backward. Host-side
 * only; see rocke/helper_rocke.instances.common._attention_bwd_tiles.h.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_tiles.h"

#include <algorithm>
#include <new>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/instance_attention_bwd.h"

namespace rocke_attn_bwd
{

/* CDNA: four SIMDs per CU, a 512-entry unified (arch VGPR + AGPR) file per
 * lane with an 8-register allocation granule, at most 8 waves per SIMD. */
static const int SIMDS_PER_CU = 4;
static const int UNIFIED_REGS = 512;
static const int REG_GRANULE = 8;
static const int MAX_WAVES_PER_SIMD = 8;

std::string grid_repr(const Grid& g)
{
    return "(" + std::to_string(g.rows) + ", " + std::to_string(g.cols) + ")";
}

std::string grids_repr(const std::vector<Grid>& gs)
{
    std::string out = "(";
    for(size_t i = 0; i < gs.size(); ++i)
    {
        if(i)
            out += ", ";
        out += grid_repr(gs[i]);
    }
    if(gs.size() == 1)
        out += ",";
    return out + ")";
}

int grid_waves(const Grid& g)
{
    return g.rows * g.cols;
}

/* Python floor division (the operands here are never negative, but waves // 2
 * of a caller's 0 or 1 must match). */
static int floordiv(int a, int b)
{
    if(b == 0) /* Python's ZeroDivisionError; ckc has no twin, so a ValueError */
        throw ckc::ValueError("integer division or modulo by zero");
    int q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

static void push_unique(std::vector<Grid>& out, const Grid& g)
{
    for(const Grid& x : out)
        if(x == g)
            return;
    out.push_back(g);
}

/* _shapes(k): (1, k), (k, 1), (2, k / 2), deduplicated in that order. */
static std::vector<Grid> shapes(int k)
{
    std::vector<Grid> out;
    push_unique(out, Grid{1, k});
    push_unique(out, Grid{k, 1});
    if(k >= 2)
        push_unique(out, Grid{2, floordiv(k, 2)});
    return out;
}

std::vector<Grid> g02_grids(int waves)
{
    std::vector<int> ks;
    for(int k : {1, 2, 4, 8})
        if(k <= waves && waves % k == 0)
            ks.push_back(k);
    std::vector<Grid> out;
    push_unique(out, Grid{1, waves});
    for(auto it = ks.rbegin(); it != ks.rend(); ++it)
        for(const Grid& g : shapes(*it))
            push_unique(out, g);
    return out;
}

std::vector<Grid> g13_grids(int waves)
{
    std::vector<Grid> out;
    push_unique(out, Grid{waves, 1});
    for(const Grid& g : shapes(waves))
        push_unique(out, g);
    return out;
}

static bool contains(const std::vector<Grid>& gs, const Grid& g)
{
    for(const Grid& x : gs)
        if(x == g)
            return true;
    return false;
}

void check_lds_route_grid(const std::string& gname,
                          const Grid& g,
                          const Grid& natural,
                          int waves,
                          int rows,
                          int cols,
                          const std::string& pt_route)
{
    Grid alt{2, floordiv(waves, 2)};
    std::vector<Grid> legal = gname == "warp_grid_g02" ? g02_grids(waves) : g13_grids(waves);
    if(waves < 2 || !contains(legal, g))
    {
        std::vector<Grid> others;
        for(const Grid& x : legal)
            if(!(x == natural) && !(x == alt))
                others.push_back(x);
        std::string more = (waves >= 2 && !others.empty()) ? "; also " + grids_repr(others) : "";
        throw ckc::ValueError(gname + "=" + grid_repr(g) + " must be " + grid_repr(natural)
                              + " (or " + grid_repr(alt) + " with pt_route='lds'" + more + ")");
    }
    if(pt_route != "lds")
        throw ckc::ValueError(gname + "=" + grid_repr(g)
                              + " needs pt_route='lds' (the relabel route fixes it)");
    if(rows % (16 * g.rows) != 0 || cols % (16 * g.cols) != 0)
        throw ckc::ValueError(gname + "=" + grid_repr(g) + " does not divide its "
                              + std::to_string(rows) + " x " + std::to_string(cols) + " tile");
}

int kv_operand_split(const Grid& g, int waves)
{
    if(g == Grid{1, waves} || g == Grid{2, floordiv(waves, 2)})
        return waves;
    return g.cols;
}

std::vector<rocke_attn_bwd_grid_triple_t>
    small_tile_grids(int head_size, int block_m, int block_n, int waves)
{
    int km = block_m, kn = block_n, d = head_size, w = waves;
    std::vector<rocke_attn_bwd_grid_triple_t> out;
    for(const Grid& g02 : g02_grids(w))
    {
        if(km % (16 * g02.rows) || kn % (16 * g02.cols))
            continue;
        for(const Grid& g13 : g13_grids(w))
        {
            if(kn % (16 * g13.rows) || d % (16 * g13.cols))
                continue;
            std::vector<Grid> g4s;
            if(w >= 2)
                g4s = {Grid{1, w}, Grid{2, floordiv(w, 2)}};
            else
                g4s = {Grid{1, 1}};
            for(const Grid& g4 : g4s)
            {
                if(grid_waves(g4) != w || km % (16 * g4.rows) || d % (16 * g4.cols))
                    continue;
                rocke_attn_bwd_grid_triple_t t{g02, g13, g4};
                bool dup = false;
                for(const auto& x : out)
                    if(x.g02 == t.g02 && x.g13 == t.g13 && x.g4 == t.g4)
                        dup = true;
                if(!dup)
                    out.push_back(t);
            }
        }
    }
    return out;
}

/* _waves_per_simd(wgs, waves) */
static int waves_per_simd(int wgs, int waves)
{
    if(wgs <= 0)
        return 0;
    return std::max(1, floordiv(wgs * waves, SIMDS_PER_CU));
}

/* _wgs_by_slots(per_simd, waves) */
static int wgs_by_slots(int per_simd, int waves)
{
    if(waves > SIMDS_PER_CU)
        return floordiv(per_simd, -floordiv(-waves, SIMDS_PER_CU));
    return floordiv(per_simd * SIMDS_PER_CU, waves);
}

rocke_attn_bwd_occupancy_t occupancy(const rocke_attn_bwd_spec_t& spec, const std::string& arch)
{
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    if(std::string(facts.matrix_path) != "mfma")
        throw ckc::ValueError("attn_bwd_occupancy models MFMA targets only, not " + arch);
    int w = spec.waves;
    RegEstimate est = register_estimates(spec, arch);
    int used = std::max(est.arch_vgpr + est.agpr, 1);
    used = -floordiv(-used, REG_GRANULE) * REG_GRANULE;
    int per_simd = std::min(MAX_WAVES_PER_SIMD, floordiv(UNIFIED_REGS, used));
    int by_regs = waves_per_simd(wgs_by_slots(per_simd, w), w);
    int lds = lds_footprint_bytes(spec, arch);
    int wgs_lds = lds > 0 ? floordiv(facts.lds_capacity_bytes, lds) : MAX_WAVES_PER_SIMD * 4;
    wgs_lds = std::min(wgs_lds, wgs_by_slots(MAX_WAVES_PER_SIMD, w));
    int by_lds = waves_per_simd(wgs_lds, w);
    rocke_attn_bwd_occupancy_t o;
    o.by_registers = by_regs;
    o.by_lds = by_lds;
    o.requested = spec.waves_per_eu;
    return o;
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

template <class Fn>
static rocke_status_t tiles_boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        fn();
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc&)
    {
        return store_err(err, ROCKE_ERR_OOM, "out of memory");
    }
}

static void copy_grids(
    const std::vector<Grid>& gs, rocke_attn_bwd_grid_t* out, int cap, int* n_out, const char* what)
{
    if((int)gs.size() > cap)
        throw ckc::ValueError(std::string(what) + ": " + std::to_string(gs.size())
                              + " grids exceed the C port's limit of " + std::to_string(cap));
    for(size_t i = 0; i < gs.size(); ++i)
        out[i] = gs[i];
    if(n_out != NULL)
        *n_out = (int)gs.size();
}

extern "C" {

int rocke_attn_bwd_grid_waves(rocke_attn_bwd_grid_t grid)
{
    return grid_waves(grid);
}

void rocke_attn_bwd_natural_grids(int waves, rocke_attn_bwd_grid_t* g02, rocke_attn_bwd_grid_t* g13)
{
    if(g02 != NULL)
        *g02 = Grid{1, waves};
    if(g13 != NULL)
        *g13 = Grid{waves, 1};
}

rocke_status_t rocke_attn_bwd_g02_grids(int waves,
                                        rocke_attn_bwd_grid_t out[ROCKE_ATTN_BWD_GRIDS_MAX],
                                        int* n_out,
                                        rocke_attn_bwd_err_t* err)
{
    return tiles_boundary(err, [&] {
        copy_grids(g02_grids(waves), out, ROCKE_ATTN_BWD_GRIDS_MAX, n_out, "g02_grids");
    });
}

rocke_status_t rocke_attn_bwd_g13_grids(int waves,
                                        rocke_attn_bwd_grid_t out[ROCKE_ATTN_BWD_GRIDS_MAX],
                                        int* n_out,
                                        rocke_attn_bwd_err_t* err)
{
    return tiles_boundary(err, [&] {
        copy_grids(g13_grids(waves), out, ROCKE_ATTN_BWD_GRIDS_MAX, n_out, "g13_grids");
    });
}

bool rocke_attn_bwd_replicated_g02(rocke_attn_bwd_grid_t grid, int waves)
{
    return grid_waves(grid) < waves;
}

rocke_status_t rocke_attn_bwd_check_lds_route_grid(const char* gname,
                                                   rocke_attn_bwd_grid_t grid,
                                                   rocke_attn_bwd_grid_t natural,
                                                   int waves,
                                                   int rows,
                                                   int cols,
                                                   const char* pt_route,
                                                   rocke_attn_bwd_err_t* err)
{
    return tiles_boundary(err, [&] {
        check_lds_route_grid(
            gname ? gname : "", grid, natural, waves, rows, cols, pt_route ? pt_route : "");
    });
}

int rocke_attn_bwd_kv_operand_split(rocke_attn_bwd_grid_t grid_g02, int waves)
{
    return kv_operand_split(grid_g02, waves);
}

rocke_status_t
    rocke_attn_bwd_small_tile_grids(int head_size,
                                    int block_m,
                                    int block_n,
                                    int waves,
                                    rocke_attn_bwd_grid_triple_t out[ROCKE_ATTN_BWD_TRIPLES_MAX],
                                    int* n_out,
                                    rocke_attn_bwd_err_t* err)
{
    return tiles_boundary(err, [&] {
        std::vector<rocke_attn_bwd_grid_triple_t> t
            = small_tile_grids(head_size, block_m, block_n, waves);
        if((int)t.size() > ROCKE_ATTN_BWD_TRIPLES_MAX)
            throw ckc::ValueError("small_tile_grids: " + std::to_string(t.size())
                                  + " triples exceed the C port's limit");
        for(size_t i = 0; i < t.size(); ++i)
            out[i] = t[i];
        if(n_out != NULL)
            *n_out = (int)t.size();
    });
}

int rocke_attn_bwd_occupancy_waves(const rocke_attn_bwd_occupancy_t* occ)
{
    return std::min(occ->by_registers, occ->by_lds);
}

bool rocke_attn_bwd_occupancy_hint_reachable(const rocke_attn_bwd_occupancy_t* occ)
{
    return occ->requested == ROCKE_ATTN_BWD_NONE || occ->by_lds >= occ->requested;
}

rocke_status_t rocke_attn_bwd_occupancy(const struct rocke_attn_bwd_spec* spec,
                                        const char* arch,
                                        rocke_attn_bwd_occupancy_t* out,
                                        rocke_attn_bwd_err_t* err)
{
    return tiles_boundary(err, [&] {
        rocke_attn_bwd_occupancy_t o = occupancy(*spec, arch ? arch : "");
        if(out != NULL)
            *out = o;
    });
}

} /* extern "C" */
