// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tensor_gett_hooks.cpp -- C++ mirror of the GettHooks class of
 * rocke/instances/common/tensor/gett.py: the `_gett` side channel of the
 * universal GEMM builder (operand addressing, the ring body's K-outer LDS
 * layout and fragment reads, the permuted alpha / beta epilogue). See
 * rocke/instance_tensor_gett.h.
 *
 * Every builder call is made in the order of the Python method it mirrors (an
 * argument that emits ops is computed in its own statement first).
 */
#include "rocke/instance_tensor_gett.h"

#include "rocke/instance_gemm_internal.h"
#include "rocke/ir_internal.h"

#include <string.h>

#include <algorithm>
#include <sstream>

using rocke::gett::Digit;
using rocke::gett::Digits;
using V = rocke_value_t;

namespace
{

int64_t i32_bits(int64_t v)
{
    return v >= ((int64_t)1 << 31) ? v - ((int64_t)1 << 32) : v;
}

V* ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

int ri(char nm)
{
    return nm == 'A' ? 0 : 1;
}

const rocke_type_t* ty_of(const char* dtype)
{
    if(dtype != nullptr && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
        return rocke_f16();
    if(dtype != nullptr && strcmp(dtype, "bf16") == 0)
        return rocke_bf16();
    return nullptr;
}

/* Python repr of trait values (error messages) */
std::string rb(bool v)
{
    return v ? "True" : "False";
}
std::string rs(const char* v)
{
    return v == nullptr ? "None" : std::string("'") + v + "'";
}

/* A compare-able view of every C++ TraitSpec field: (name, repr). */
std::vector<std::pair<std::string, std::string>> trait_fields(const rocke_gemm_trait_spec_t& t)
{
    std::vector<std::pair<std::string, std::string>> f;
    auto I = [](int v) { return std::to_string(v); };
    f.push_back({"pipeline", rs(t.pipeline)});
    f.push_back({"scheduler", rs(t.scheduler)});
    f.push_back({"epilogue", rs(t.epilogue)});
    f.push_back({"pad_m", rb(t.pad_m)});
    f.push_back({"pad_n", rb(t.pad_n)});
    f.push_back({"pad_k", rb(t.pad_k)});
    f.push_back({"persistent", rb(t.persistent)});
    f.push_back({"chiplet_swizzle", rb(t.chiplet_swizzle)});
    f.push_back({"chiplet_wgm", I(t.chiplet_wgm)});
    f.push_back({"chiplet_num_xcds", I(t.chiplet_num_xcds)});
    f.push_back({"chiplet_chunk_size", I(t.chiplet_chunk_size)});
    f.push_back({"waves_per_eu", t.waves_per_eu_set ? I(t.waves_per_eu) : "None"});
    f.push_back({"preshuffle_b", rb(t.preshuffle_b)});
    f.push_back({"direct_to_lds", rb(t.direct_to_lds)});
    f.push_back({"dtl_cache_a", I(t.dtl_cache_a)});
    f.push_back({"dtl_cache_b", I(t.dtl_cache_b)});
    f.push_back({"dtl_prefetch", rb(t.dtl_prefetch)});
    f.push_back({"active_tile_skip", rb(t.active_tile_skip)});
    f.push_back({"lds_k_pad", I(t.lds_k_pad)});
    f.push_back({"lds_swizzle", rb(t.lds_swizzle)});
    f.push_back({"emit_sched_hints", t.emit_sched_hints_set ? rb(t.emit_sched_hints) : "None"});
    f.push_back({"split_k", I(t.split_k)});
    f.push_back({"cshuffle_no_alias", rb(t.cshuffle_no_alias)});
    f.push_back({"ab_double_buffer", t.ab_double_buffer_set ? rb(t.ab_double_buffer) : "None"});
    f.push_back({"interwave_setprio", rb(t.interwave_setprio)});
    f.push_back({"dtl_stages", I(t.dtl_stages)});
    f.push_back({"dtl_issue_spread", rb(t.dtl_issue_spread)});
    f.push_back({"dtl_frag_prefetch", rb(t.dtl_frag_prefetch)});
    f.push_back({"dtl_stream_barrier", rb(t.dtl_stream_barrier)});
    f.push_back({"mfma_swap_ab", rb(t.mfma_swap_ab)});
    f.push_back({"iglp_opt", t.iglp_opt_set ? I(t.iglp_opt) : "None"});
    f.push_back({"persistent_ctas", I(t.persistent_ctas)});
    f.push_back({"acc_agpr", rb(t.acc_agpr)});
    f.push_back({"mfma_interleave", rb(t.mfma_interleave)});
    f.push_back({"acc_agpr_pin", rb(t.acc_agpr_pin)});
    f.push_back({"ilv_dtl_every", I(t.ilv_dtl_every)});
    f.push_back({"ilv_barrier_after", I(t.ilv_barrier_after)});
    f.push_back({"ilv_merge", rb(t.ilv_merge)});
    f.push_back({"lds_swizzle_r", I(t.lds_swizzle_r)});
    f.push_back({"dtl_hoist", rb(t.dtl_hoist)});
    f.push_back({"streamk", rb(t.streamk)});
    f.push_back({"dtl_tight_rsrc", rb(t.dtl_tight_rsrc)});
    f.push_back({"pad_fast_tile", rb(t.pad_fast_tile)});
    f.push_back({"ilv_early_release", I(t.ilv_early_release)});
    f.push_back({"ilv_release_at", I(t.ilv_release_at)});
    f.push_back({"ilv_acquire_at", I(t.ilv_acquire_at)});
    f.push_back({"acc_out_vgpr", rb(t.acc_out_vgpr)});
    f.push_back({"ilv_read_span", I(t.ilv_read_span)});
    f.push_back({"ilv_dtl_span", I(t.ilv_dtl_span)});
    f.push_back({"lds_row_interleave", rb(t.lds_row_interleave)});
    f.push_back({"rilv_pad", I(t.rilv_pad ? t.rilv_pad : 16)});
    f.push_back({"rilv_vec_epilogue", rb(t.rilv_vec_epilogue)});
    f.push_back({"rve_store_nt", rb(t.rve_store_nt)});
    f.push_back({"dtl_soff0", rb(t.dtl_soff0)});
    f.push_back({"wide_addr", rb(t.wide_addr)});
    f.push_back({"mn_pad", I(t.mn_pad_set ? t.mn_pad : 32)});
    f.push_back({"mn_chunk", I(t.mn_chunk)});
    f.push_back({"mn_ld_any", rb(t.mn_ld_any)});
    f.push_back({"hb_sched", rb(t.hb_sched)});
    f.push_back({"hb_rel_slot", I(t.hb_rel_slot)});
    f.push_back({"hb_acq_slot", I(t.hb_acq_slot)});
    f.push_back({"hb_read_every", I(t.hb_read_every)});
    f.push_back({"hb_dtl_every", I(t.hb_dtl_every)});
    f.push_back({"hb_acq_b_slot", I(t.hb_acq_b_slot)});
    f.push_back({"hb_split_rel", rb(t.hb_split_rel)});
    f.push_back({"hb_rel_gap", I(t.hb_rel_gap)});
    f.push_back({"kloop_unroll", I(t.kloop_unroll)});
    f.push_back({"dtl_addr_spread", rb(t.dtl_addr_spread)});
    f.push_back({"hb_dtl_delay", I(t.hb_dtl_delay)});
    f.push_back({"hb_skew", I(t.hb_skew)});
    f.push_back({"hb_skew_dtl", I(t.hb_skew_dtl)});
    f.push_back({"hb_tile_lgkm0", I(t.hb_tile_lgkm0)});
    f.push_back({"das_anchor", rb(t.das_anchor)});
    f.push_back({"das_late", rb(t.das_late)});
    f.push_back({"mfma_serpentine", rb(t.mfma_serpentine)});
    f.push_back({"mfma_order", I(t.mfma_order)});
    f.push_back({"rve_pad", I(t.rve_pad)});
    f.push_back({"dtl_tight_ragged_only", rb(t.dtl_tight_ragged_only)});
    f.push_back({"vgpr_prefetch", I(t.vgpr_prefetch)});
    f.push_back({"vgpr_prefetch_depth", I(t.vgpr_prefetch_depth)});
    f.push_back({"vp_unroll", I(t.vp_unroll)});
    f.push_back({"vp_late_reads", I(t.vp_late_reads)});
    f.push_back({"vp_release_atoms", I(t.vp_release_atoms)});
    f.push_back({"vp_tail_pack", I(t.vp_tail_pack)});
    f.push_back({"vp_pair_reads", I(t.vp_pair_reads)});
    f.push_back({"vp_mfma_first", I(t.vp_mfma_first)});
    f.push_back({"split_k_serial", I(t.split_k_serial)});
    f.push_back({"split_k_aligned", rb(t.split_k_aligned)});
    f.push_back({"lds_swizzle_l", I(t.lds_swizzle_l)});
    return f;
}

std::string field(const std::vector<std::pair<std::string, std::string>>& f, const char* name)
{
    for(const auto& x : f)
        if(x.first == name)
            return x.second;
    return "";
}

} // namespace

/* ======================================================= body rules */

namespace rocke
{
namespace gett
{

/* BODY_TRAIT over the TraitSpec defaults */
rocke_gemm_trait_spec_t body_trait()
{
    rocke_gemm_trait_spec_t t = rocke_gemm_universal_spec_default().trait;
    t.pipeline = "compv4";
    t.scheduler = "intrawave";
    t.epilogue = "default";
    t.chiplet_swizzle = true;
    t.chiplet_wgm = 8;
    t.chiplet_num_xcds = 8;
    t.chiplet_chunk_size = 64;
    t.direct_to_lds = true;
    t.dtl_prefetch = true;
    t.lds_swizzle = false;
    t.acc_agpr = true;
    t.dtl_frag_prefetch = true;
    t.mfma_interleave = true;
    t.ilv_merge = true;
    t.dtl_hoist = true;
    t.dtl_stages = 3;
    return t;
}

std::vector<std::string> body_trait_errors(const rocke_gemm_trait_spec_t& trait)
{
    static const char* const free_fields[] = {"dtl_stages",
                                              "persistent_ctas",
                                              "chiplet_wgm",
                                              "chiplet_chunk_size",
                                              "chiplet_num_xcds",
                                              "dtl_cache_a",
                                              "dtl_cache_b"};
    auto got = trait_fields(trait);
    auto want = trait_fields(body_trait());
    std::vector<std::string> errs;
    for(size_t i = 0; i < got.size(); ++i)
    {
        bool fr = false;
        for(const char* x : free_fields)
            fr = fr || got[i].first == x;
        if(fr)
            continue;
        if(got[i].second != want[i].second)
            errs.push_back(got[i].first + "=" + got[i].second + " (the GETT body needs "
                           + want[i].second + ")");
    }
    if(!(trait.dtl_stages == 2 || trait.dtl_stages == 3 || trait.dtl_stages == 4))
        errs.push_back("dtl_stages=" + std::to_string(trait.dtl_stages) + " (2..4)");
    if(trait.persistent_ctas < 0)
        errs.push_back("persistent_ctas=" + std::to_string(trait.persistent_ctas) + " (>= 0)");
    return errs;
}

std::vector<std::string> mn_body_errors(const rocke_gemm_universal_spec_t& spec)
{
    const rocke_gemm_tile_spec_t& t = spec.tile;
    const rocke_gemm_trait_spec_t& tr = spec.trait;
    std::vector<std::string> errs;
    if(strcmp(spec.data.layout, "CRR") != 0)
        errs.push_back(std::string("layout ") + spec.data.layout + " (the MN body is CRR)");
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
        errs.push_back("atoms must be 16x16x32");
    if(t.warp_k != 1)
        errs.push_back("warp_k must be 1");
    if(!(tr.direct_to_lds && tr.dtl_prefetch && tr.dtl_stages >= 2))
        errs.push_back(
            "needs the alias-scoped DTL ring (direct_to_lds, dtl_prefetch, dtl_stages >= 2)");
    if(strcmp(tr.epilogue, "default") != 0)
        errs.push_back(std::string("epilogue=") + rs(tr.epilogue)
                       + " (the GETT epilogue replaces the direct one)");
    static const char* const fixed[] = {"pad_k",
                                        "persistent",
                                        "preshuffle_b",
                                        "active_tile_skip",
                                        "split_k",
                                        "mfma_swap_ab",
                                        "streamk",
                                        "lds_row_interleave",
                                        "rilv_vec_epilogue",
                                        "rve_store_nt",
                                        "hb_sched",
                                        "dtl_addr_spread"};
    auto got = trait_fields(tr);
    auto ref = trait_fields(rocke_gemm_universal_spec_default().trait);
    for(const char* f : fixed)
        if(field(got, f) != field(ref, f))
            errs.push_back(std::string(f) + "=" + field(got, f) + " (the MN GETT body needs "
                           + field(ref, f) + ")");
    static const std::pair<const char*, const char*> dropped[] = {{"epilogue", "'default'"},
                                                                  {"cshuffle_no_alias", "False"},
                                                                  {"pad_m", "False"},
                                                                  {"pad_n", "False"},
                                                                  {"dtl_tight_rsrc", "False"},
                                                                  {"pad_fast_tile", "False"},
                                                                  {"dtl_soff0", "False"}};
    for(const auto& d : dropped)
        if(field(got, d.first) != d.second)
            errs.push_back(std::string(d.first) + "=" + field(got, d.first)
                           + " (dropped on the MN GETT body)");
    if(tr.persistent_ctas < 0)
        errs.push_back("persistent_ctas=" + std::to_string(tr.persistent_ctas) + " (>= 0)");
    return errs;
}

rocke_gemm_trait_spec_t kc_body_trait(const rocke_gemm_trait_spec_t& trait)
{
    /* KC_BODY_DROPPED, and a cshuffle epilogue becomes the direct one (the hook
     * writes D) */
    rocke_gemm_trait_spec_t t = trait;
    t.rve_store_nt = false;
    t.pad_fast_tile = false;
    t.cshuffle_no_alias = false;
    if(strcmp(t.epilogue, "default") != 0)
        t.epilogue = "default";
    return t;
}

std::vector<std::string> kc_body_errors(const rocke_gemm_universal_spec_t& spec)
{
    const rocke_gemm_tile_spec_t& t = spec.tile;
    const rocke_gemm_trait_spec_t& tr = spec.trait;
    std::vector<std::string> errs;
    if(strcmp(spec.data.layout, "RCR") != 0)
        errs.push_back(std::string("layout ") + spec.data.layout
                       + " (the K-contiguous body is RCR)");
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16))
        errs.push_back("atoms must be 16x16 (the epilogue's lane layout)");
    if(strcmp(tr.epilogue, "default") != 0)
        errs.push_back(std::string("epilogue=") + rs(tr.epilogue)
                       + " (the GETT epilogue replaces the direct one)");
    /* KC_BODY_FIXED_FIELDS (sab_vec16, sab_store_nt and dtl_k_tail are
     * Python-engine-only levers: at their defaults here) */
    static const char* const fixed[]
        = {"persistent", "split_k", "mfma_swap_ab", "streamk", "pad_k", "active_tile_skip"};
    auto got = trait_fields(tr);
    auto ref = trait_fields(rocke_gemm_universal_spec_default().trait);
    for(const char* f : fixed)
        if(field(got, f) != field(ref, f))
            errs.push_back(std::string(f) + "=" + field(got, f)
                           + " (the K-contiguous GETT body needs " + field(ref, f) + ")");
    static const std::pair<const char*, const char*> dropped[]
        = {{"rve_store_nt", "False"}, {"pad_fast_tile", "False"}, {"cshuffle_no_alias", "False"}};
    for(const auto& d : dropped)
        if(field(got, d.first) != d.second)
            errs.push_back(std::string(d.first) + "=" + field(got, d.first)
                           + " (dropped on the K-contiguous GETT body)");
    return errs;
}

} // namespace gett
} // namespace rocke

extern "C" bool rocke_gett_hooks_a_sub_perm(const struct rocke_gett_hooks* h)
{
    return h != nullptr && h->a_sub_perm;
}

/* ======================================================= GettHooks */

namespace
{
std::string join(const std::vector<std::string>& v, const char* sep)
{
    std::string s;
    for(size_t i = 0; i < v.size(); ++i)
        s += (i ? sep : "") + v[i];
    return s;
}
} // namespace

rocke_gett_hooks::rocke_gett_hooks(const rocke::gett::Plan& plan_,
                                   const rocke::gett::Knobs& knobs,
                                   const rocke_gemm_universal_spec_t& spec,
                                   const char* dtype)
    : plan(plan_)
    , k(knobs)
    , dt(ty_of(dtype))
{
    memset(lay, 0, sizeof lay);
    const rocke_gemm_tile_spec_t& t = spec.tile;
    bs = spec.block_size;
    ws = spec.wave_size;
    T = t.tile_k;
    bmn[0] = t.tile_m;
    bmn[1] = t.tile_n;
    wt[0] = t.warp_tile_m;
    wt[1] = t.warp_tile_n;
    mf[0] = t.tile_m / (t.warp_m * t.warp_tile_m);
    mf[1] = t.tile_n / (t.warp_n * t.warp_tile_n);
    wmn[0] = t.tile_m / t.warp_m;
    wmn[1] = t.tile_n / t.warp_n;
    stages = spec.trait.dtl_stages;
    mn = strcmp(spec.data.layout, "CRR") == 0;
    kc = plan.kc;
    addr = !kc;
    a_sub_perm = mn && k.epi_v16;
    if(dt == nullptr)
    {
        err_ = std::string("unknown GETT dtype '") + (dtype ? dtype : "") + "'";
        return;
    }
    if(kc)
    {
        std::vector<std::string> bad = rocke::gett::kc_body_errors(spec);
        if(!bad.empty())
        {
            err_ = "GETT K-contiguous body: " + join(bad, "; ");
            return;
        }
        if(k.epi_v16 || k.store_nt || k.epi_pair)
        {
            err_ = "epi_v16 / store_nt / epi_pair are MN-body knobs (the K-contiguous bodies use "
                   "their own epilogue layout)";
            return;
        }
        if(spec.trait.rilv_vec_epilogue && !plan.vec_dc8)
        {
            err_ = "the row-interleave vector epilogue stores 8 consecutive columns: D needs "
                   "8-element runs on the column side";
            return;
        }
        return;
    }
    if(mn)
    {
        std::vector<std::string> bad = rocke::gett::mn_body_errors(spec);
        if(!bad.empty())
        {
            err_ = "GETT MN body: " + join(bad, "; ");
            return;
        }
        if(T != plan.tile_k)
        {
            err_ = "spec tile_k " + std::to_string(T) + " != plan tile_k "
                   + std::to_string(plan.tile_k);
            return;
        }
        for(int o = 0; o < 2; ++o)
            if(plan.ops[o].vec != 8)
            {
                err_ = std::string(o == 0 ? "A" : "B")
                       + ": the MN body loads 16-byte runs (operand run vec "
                       + std::to_string(plan.ops[o].vec) + "; pack it or use the ring body)";
                return;
            }
        if(k.store_nt && !k.epi_v16)
        {
            err_ = "store_nt needs epi_v16";
            return;
        }
        if(k.epi_v16 && !v16_ok())
        {
            err_ = "epi_v16 needs 8-element D runs (vec_d8) and an even number of M atoms per wave";
            return;
        }
        rocke_gemm_universal_spec_t s2 = spec;
        s2._gett = nullptr;
        char reason[256] = {0};
        if(!rocke_gemm_spec_mn_layouts(
               &s2, &lay[0], &has_lay[0], &lay[1], &has_lay[1], reason, sizeof reason))
        {
            err_ = reason;
            return;
        }
        if(k.epi_v16)
            lay[0].sub_perm = true;
        return;
    }
    if(k.epi_v16 || k.store_nt)
    {
        err_ = "epi_v16 / store_nt are MN-body knobs (TensorGettSpec.mn_tile)";
        return;
    }
    std::vector<std::string> bad = rocke::gett::body_trait_errors(spec.trait);
    if(!bad.empty())
    {
        err_ = "GETT body: " + join(bad, "; ");
        return;
    }
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
    {
        err_ = "GETT fragments are built for the 16x16x32 atom";
        return;
    }
    if(T != plan.tile_k)
    {
        err_
            = "spec tile_k " + std::to_string(T) + " != plan tile_k " + std::to_string(plan.tile_k);
        return;
    }
    for(int o = 0; o < 2; ++o)
    {
        char nm = o == 0 ? 'A' : 'B';
        int64_t tot = (int64_t)bmn[o] * T * 2;
        if(tot % ((int64_t)bs * lane_bytes(nm)))
        {
            err_ = std::string(1, nm) + ": tile bytes " + std::to_string(tot)
                   + " not a multiple of one DTL pass";
            return;
        }
        if(bmn[o] != 256 || bs != 256)
        {
            err_ = "GETT LDS block layout is built for tile_mn 256 and 4 waves";
            return;
        }
    }
}

/* ------------------------------------------------------------ geometry */

bool rocke_gett_hooks::v16_ok() const
{
    return plan.vec_d8 && mf[0] % 2 == 0;
}

int rocke_gett_hooks::lane_bytes(char nm) const
{
    if(mn)
        return 16;
    return plan.op(nm).vec == 8 ? 16 : 4;
}

int rocke_gett_hooks::passes(char nm) const
{
    if(mn)
        return rocke_gemm_mn_n_pass(&lay[ri(nm)]);
    return (bmn[ri(nm)] * T * 2) / (bs * lane_bytes(nm));
}

int rocke_gett_hooks::block_bytes() const
{
    return 1024 + k.pad;
}

void rocke_gett_hooks::lds_shape(char nm, int* shape) const
{
    (void)nm;
    shape[0] = T / 2;
    shape[1] = block_bytes() / 2;
}

int rocke_gett_hooks::wave_bytes(char nm) const
{
    int ib = 64 * lane_bytes(nm);
    return ib == 1024 ? block_bytes() : ib;
}

int rocke_gett_hooks::pass_bytes(char nm) const
{
    if(mn)
        return rocke_gemm_mn_pass_bytes(&lay[ri(nm)]);
    int ib = 64 * lane_bytes(nm);
    return ib == 1024 ? 4 * block_bytes() : block_bytes();
}

std::pair<int64_t, int64_t> rocke_gett_hooks::lane_geom(char nm, int p, int tid) const
{
    int64_t lb = lane_bytes(nm);
    int64_t lin = tid * lb + (int64_t)p * 4 * 64 * lb;
    int64_t blk = lin / 1024, inb = lin % 1024;
    int64_t r = inb / 512;
    int64_t kk = 32 * (blk / 16) + 16 * ((blk / 8) % 2) + 8 * ((blk / 4) % 2) + 4 * r + (blk % 4);
    return {kk, (inb % 512) / 2};
}

/* ------------------------------------------------------------ params */

void rocke_gett_hooks::add_params(rocke_ir_builder_t* b)
{
    p.clear();
    p["alpha"] = rocke_b_param(b, "alpha", rocke_f32(), nullptr);
    if(k.bilinear)
    {
        rocke_param_opts_t o;
        memset(&o, 0, sizeof o);
        o.readonly = true;
        o.readonly_set = true;
        o.align = 8;
        o.align_set = true;
        p["Cin"] = rocke_b_param(b, "Cin", rocke_ptr_type(b, dt, "global"), &o);
        p["beta"] = rocke_b_param(b, "beta", rocke_f32(), nullptr);
    }
}

/* ------------------------------------------------------------ int helpers */

V* rocke_gett_hooks::digits_static(rocke_ir_builder_t* b, V* x, const std::vector<Digit>& dg)
{
    V* off = nullptr;
    V* q = x;
    for(size_t j = 0; j < dg.size(); ++j)
    {
        bool last = j == dg.size() - 1;
        V* d = q;
        if(!last)
        {
            V* ce = ci(b, dg[j].e);
            d = rocke_b_mod(b, q, ce);
        }
        V* term = d;
        if(dg[j].s != 1)
        {
            V* cs = ci(b, dg[j].s);
            term = rocke_b_mul(b, d, cs);
        }
        off = off == nullptr ? term : rocke_b_add(b, off, term);
        if(!last)
        {
            V* ce = ci(b, dg[j].e);
            q = rocke_b_div(b, q, ce);
        }
    }
    return off;
}

V* rocke_gett_hooks::digits(rocke_ir_builder_t* b, V* x, const Digits& dg)
{
    return digits_static(b, x, dg.d);
}

/* ------------------------------------------------------------ bind */

void rocke_gett_hooks::bind(rocke_ir_builder_t* b, V* tid, V* lane, V* warp_m_idx, V* warp_n_idx)
{
    tid_ = tid;
    lane_ = lane;
    widx[0] = warp_m_idx;
    widx[1] = warp_n_idx;
    if(mn)
    {
        bind_mn(b);
        return;
    }
    V* c16 = ci(b, 16);
    V* c4 = ci(b, 4);
    V* kq = rocke_b_div(b, lane, c16);
    V* m16 = rocke_b_mod(b, lane, c16);
    V* g = rocke_b_div(b, m16, c4);
    V* c2 = ci(b, 2);
    V* kq2 = rocke_b_mod(b, kq, c2);
    V* t1 = rocke_b_mul(b, kq2, c4);
    V* c2b = ci(b, 2);
    V* kqd = rocke_b_div(b, kq, c2b);
    V* c8 = ci(b, 8);
    V* t2 = rocke_b_mul(b, kqd, c8);
    V* t12 = rocke_b_add(b, t1, t2);
    V* blk = rocke_b_add(b, g, t12);
    int HB = block_bytes() / 2;
    V* chb = ci(b, HB);
    V* bh = rocke_b_mul(b, blk, chb);
    V* l4 = rocke_b_mod(b, lane, c4);
    V* l4m = rocke_b_mul(b, l4, c4);
    V* lane_part = rocke_b_add(b, bh, l4m);
    /* LDS placement (smem pool: A stages then B stages, back to back) */
    sh_.clear();
    int o = 0;
    for(char nm : {'A', 'B'})
        for(int j = 0; j < stages; ++j)
        {
            sh_[{nm, j}] = o / 2;
            o += (T / 2) * block_bytes();
        }
    fbase_.clear();
    for(char nm : {'A', 'B'})
    {
        V* cw = ci(b, wmn[ri(nm)]);
        V* woff = rocke_b_mul(b, widx[ri(nm)], cw);
        V* lp = rocke_b_add(b, lane_part, woff);
        for(int j = 0; j < stages; ++j)
        {
            int sh = sh_[{nm, j}];
            V* v = lp;
            if(sh)
            {
                V* cs = ci(b, sh);
                v = rocke_b_add(b, lp, cs);
            }
            fbase_[{nm, j}] = rocke_b_optimization_barrier(b, v);
        }
    }
}

void rocke_gett_hooks::bind_mn(rocke_ir_builder_t* b)
{
    for(int o = 0; o < 2; ++o)
    {
        char nm = o == 0 ? 'A' : 'B';
        int np = rocke_gemm_mn_n_pass(&lay[o]);
        std::vector<V*> ks(np), mns(np);
        rocke_gemm_mn_emit_dtl_lane_k_mn(b, &lay[o], tid_, ks.data(), mns.data());
        mn_geo_[o].clear();
        for(int i = 0; i < np; ++i)
        {
            V* kl = klocal_expr(b, nm, ks[i]);
            V* kok = k_valid(b, ks[i]);
            mn_geo_[o].push_back({ks[i], mns[i], kl, kok});
        }
    }
}

V* rocke_gett_hooks::k_valid(rocke_ir_builder_t* b, V* kv)
{
    if(!plan.has_k_pad)
        return nullptr;
    int64_t dv = 1;
    for(const auto& d : plan.k_local)
    {
        if(d.m == plan.k_pad_mode)
        {
            V* x = kv;
            if(dv > 1)
            {
                V* cd = ci(b, dv);
                x = rocke_b_div(b, kv, cd);
            }
            V* cr = ci(b, d.r);
            x = rocke_b_mod(b, x, cr);
            V* creal = ci(b, plan.k_pad_real);
            return rocke_b_cmp_lt(b, x, creal);
        }
        dv *= d.r;
    }
    rocke_i_set_err(
        b, ROCKE_ERR_VALUE, "padded K mode %s is not an in-tile digit", plan.k_pad_mode.c_str());
    return nullptr;
}

std::vector<V*> rocke_gett_hooks::tile_voffs_mn(rocke_ir_builder_t* b, char nm, V* mn0)
{
    const auto& op = plan.op(nm);
    bool rag = op.rows % bmn[ri(nm)] != 0;
    V* c2 = ci(b, 2);
    V* big = ci(b, rocke::gett::OOB);
    std::vector<V*> out;
    for(const auto& g : mn_geo_[ri(nm)])
    {
        V* row = rocke_b_add(b, mn0, g.mn);
        V* ro = digits(b, row, op.row_digits);
        V* s = rocke_b_add(b, ro, g.kl);
        V* v = rocke_b_mul(b, s, c2);
        V* ok = nullptr;
        if(rag)
        {
            V* cr = ci(b, op.rows);
            ok = rocke_b_cmp_lt(b, row, cr);
        }
        if(g.kok != nullptr)
            ok = ok == nullptr ? g.kok : rocke_b_land(b, ok, g.kok);
        if(ok != nullptr)
            v = rocke_b_select(b, ok, v, big);
        out.push_back(v);
    }
    return out;
}

std::vector<V*> rocke_gett_hooks::tile_end(rocke_ir_builder_t* b, const std::vector<V*>& acc)
{
    if(!k.acc_pin)
        return acc;
    std::vector<V*> out;
    for(V* v : acc)
    {
        const rocke_type_t* ty = v->type;
        rocke_inline_asm_opts_t o;
        memset(&o, 0, sizeof o);
        o.sideeffect = k.acc_pin == 2;
        o.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=a,0", &v, 1, &ty, 1, &o);
        out.push_back(op != nullptr ? op->results[0] : nullptr);
    }
    return out;
}

V* rocke_gett_hooks::frag(rocke_ir_builder_t* b, V* smem, char nm, int stage, int idx, int kk)
{
    int sh = sh_[{nm, stage}];
    V* base = fbase_[{nm, stage}];
    int HB = block_bytes() / 2;
    V* parts[2];
    for(int r = 0; r < 2; ++r)
    {
        V* z = ci(b, 0);
        V* cimm = ci(b, 16 * kk * HB + r * 256 + 16 * idx - sh);
        V* addr = rocke_b_add(b, base, cimm);
        V* idx2[2] = {z, addr};
        parts[r] = rocke_b_ds_read_tr16_b64(b, smem, idx2, 2, dt);
    }
    V* fr = rocke_b_vec_concat(b, parts[0], parts[1]);
    if(k.frag_pin)
    {
        const rocke_type_t* ty = fr->type;
        rocke_inline_asm_opts_t o;
        memset(&o, 0, sizeof o);
        o.sideeffect = false;
        o.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &fr, 1, &ty, 1, &o);
        fr = op != nullptr ? op->results[0] : nullptr;
    }
    return fr;
}

/* ------------------------------------------------------------ DTL offsets */

std::pair<V*, V*> rocke_gett_hooks::lane_expr(rocke_ir_builder_t* b, char nm, int pp)
{
    int lb = lane_bytes(nm);
    V* clb = ci(b, lb);
    V* tl = rocke_b_mul(b, tid_, clb);
    V* cp = ci(b, (int64_t)pp * 4 * 64 * lb);
    V* lin = rocke_b_add(b, tl, cp);
    V* c1024 = ci(b, 1024);
    V* blk = rocke_b_div(b, lin, c1024);
    V* c1024b = ci(b, 1024);
    V* inb = rocke_b_mod(b, lin, c1024b);
    V* c512 = ci(b, 512);
    V* r = rocke_b_div(b, inb, c512);
    V* c512b = ci(b, 512);
    V* im = rocke_b_mod(b, inb, c512b);
    V* c2a = ci(b, 2);
    V* col = rocke_b_div(b, im, c2a);
    V* c2 = ci(b, 2);
    /* k = (blk / 16) * 32 + ((blk / 8) % 2) * 16 + ((blk / 4) % 2) * 8 + r * 4 + blk % 4 */
    V* c16 = ci(b, 16);
    V* d16 = rocke_b_div(b, blk, c16);
    V* c32 = ci(b, 32);
    V* t1 = rocke_b_mul(b, d16, c32);
    V* c8 = ci(b, 8);
    V* d8 = rocke_b_div(b, blk, c8);
    V* m8 = rocke_b_mod(b, d8, c2);
    V* t2 = rocke_b_mul(b, m8, c16);
    V* a12 = rocke_b_add(b, t1, t2);
    V* c4 = ci(b, 4);
    V* d4 = rocke_b_div(b, blk, c4);
    V* m4 = rocke_b_mod(b, d4, c2);
    V* c8b = ci(b, 8);
    V* t3 = rocke_b_mul(b, m4, c8b);
    V* c4b = ci(b, 4);
    V* t4 = rocke_b_mul(b, r, c4b);
    V* a34 = rocke_b_add(b, t3, t4);
    V* c4c = ci(b, 4);
    V* t5 = rocke_b_mod(b, blk, c4c);
    V* a345 = rocke_b_add(b, a34, t5);
    V* kv = rocke_b_add(b, a12, a345);
    return {kv, col};
}

bool rocke_gett_hooks::lane_split(char nm)
{
    if(!k.lane_split || mn || plan.has_k_pad)
        return false;
    auto it = split_cache_.find(nm);
    if(it != split_cache_.end())
        return it->second;
    bool ok = true;
    int np = passes(nm);
    std::vector<int64_t> lane_k(bs), col0(bs);
    for(int t = 0; t < bs; ++t)
    {
        auto g = lane_geom(nm, 0, t);
        lane_k[t] = g.first;
        col0[t] = g.second;
    }
    for(int pp = 0; pp < np && ok; ++pp)
    {
        std::vector<int64_t> ks(bs), cs(bs);
        for(int t = 0; t < bs; ++t)
        {
            auto g = lane_geom(nm, pp, t);
            ks[t] = g.first;
            cs[t] = g.second;
        }
        int64_t pk = ks[0] - lane_k[0];
        if(cs != col0)
        {
            ok = false;
            break;
        }
        for(int t = 0; t < bs; ++t)
            if(ks[t] - lane_k[t] != pk)
            {
                ok = false;
                break;
            }
        if(!ok)
            break;
        std::vector<int64_t> uniq = lane_k;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for(int64_t lk : uniq)
            if(plan.klocal(pk + lk, nm) != plan.klocal(pk, nm) + plan.klocal(lk, nm))
            {
                ok = false;
                break;
            }
    }
    split_cache_[nm] = ok;
    return ok;
}

int rocke_gett_hooks::pass_soff_bytes(char nm, int pp)
{
    if(!lane_split(nm))
        return 0;
    int64_t pk = lane_geom(nm, pp, 0).first - lane_geom(nm, 0, 0).first;
    return (int)(2 * plan.klocal(pk, nm));
}

std::vector<V*> rocke_gett_hooks::tile_voffs(rocke_ir_builder_t* b, char nm, V* mn0)
{
    if(mn)
        return tile_voffs_mn(b, nm, mn0);
    const auto& op = plan.op(nm);
    int np = passes(nm);
    bool rag = op.rows % bmn[ri(nm)] != 0;
    if(lane_split(nm))
    {
        std::pair<V*, V*> kr = lane_expr(b, nm, 0);
        V* row = rocke_b_add(b, mn0, kr.second);
        V* ro = digits(b, row, op.row_digits);
        if(rag)
        {
            V* cr = ci(b, op.rows);
            V* lt = rocke_b_cmp_lt(b, row, cr);
            V* neg = ci(b, -((int64_t)1 << 30));
            ro = rocke_b_select(b, lt, ro, neg);
        }
        V* kl = klocal_expr(b, nm, kr.first);
        V* s = rocke_b_add(b, ro, kl);
        V* c2 = ci(b, 2);
        V* v = rocke_b_mul(b, s, c2);
        if(rag)
        {
            V* z = ci(b, 0);
            V* lt = rocke_b_cmp_lt(b, ro, z);
            V* oob = ci(b, rocke::gett::OOB);
            v = rocke_b_select(b, lt, oob, v);
        }
        return std::vector<V*>(np, v);
    }
    /* passes with the same row geometry share their row offset */
    std::vector<std::vector<int64_t>> keys;
    std::vector<int> first_of;
    std::vector<int> rep(np);
    for(int pp = 0; pp < np; ++pp)
    {
        std::vector<int64_t> key(bs);
        for(int t = 0; t < bs; ++t)
            key[t] = lane_geom(nm, pp, t).second;
        int found = -1;
        for(size_t i = 0; i < keys.size(); ++i)
            if(keys[i] == key)
                found = first_of[i];
        if(found < 0)
        {
            keys.push_back(key);
            first_of.push_back(pp);
            found = pp;
        }
        rep[pp] = found;
    }
    std::map<int, V*> rowoff;
    std::vector<V*> out;
    for(int pp = 0; pp < np; ++pp)
    {
        std::pair<V*, V*> kr = lane_expr(b, nm, pp);
        int gp = rep[pp];
        if(rowoff.find(gp) == rowoff.end())
        {
            V* r_v = kr.second;
            if(gp != pp)
                r_v = lane_expr(b, nm, gp).second;
            V* row = rocke_b_add(b, mn0, r_v);
            V* ro = digits(b, row, op.row_digits);
            if(rag)
            {
                V* cr = ci(b, op.rows);
                V* lt = rocke_b_cmp_lt(b, row, cr);
                V* neg = ci(b, -((int64_t)1 << 30));
                ro = rocke_b_select(b, lt, ro, neg);
            }
            rowoff[gp] = ro;
        }
        V* ro = rowoff[gp];
        V* kl = klocal_expr(b, nm, kr.first);
        V* s = rocke_b_add(b, ro, kl);
        V* c2 = ci(b, 2);
        V* v = rocke_b_mul(b, s, c2);
        if(rag)
        {
            V* z = ci(b, 0);
            V* lt = rocke_b_cmp_lt(b, ro, z);
            V* oob = ci(b, rocke::gett::OOB);
            v = rocke_b_select(b, lt, oob, v);
        }
        V* kok = k_valid(b, kr.first);
        if(kok != nullptr)
        {
            /* padded K rows read zeros */
            V* oob = ci(b, rocke::gett::OOB);
            v = rocke_b_select(b, kok, v, oob);
        }
        out.push_back(v);
    }
    return out;
}

V* rocke_gett_hooks::klocal_expr(rocke_ir_builder_t* b, char nm, V* kv)
{
    const auto& st = plan.op(nm).strides;
    std::vector<Digit> dg;
    for(const auto& d : plan.k_local)
        dg.push_back({d.r, d.mult * st.at(d.m)});
    return digits_static(b, kv, dg);
}

V* rocke_gett_hooks::ktile_off_bytes(rocke_ir_builder_t* b, char nm, V* k_ld)
{
    const auto& st = plan.op(nm).strides;
    V* cT = ci(b, T);
    V* kt = rocke_b_div(b, k_ld, cT);
    std::vector<Digit> dg;
    for(const auto& d : plan.k_tile)
        dg.push_back({d.r, d.mult * st.at(d.m) * 2});
    if(dg.empty())
        return ci(b, 0);
    V* x = digits_static(b, kt, dg);
    return rocke_b_to_sgpr_u32(b, x);
}

/* ------------------------------------------------------------ epilogue */

int64_t rocke_gett_hooks::d_bytes() const
{
    int64_t s = 0;
    for(const auto& m : plan.modes_d)
        s += (plan.ext.at(m) - 1) * plan.strides_d.at(m);
    return 2 * (1 + s);
}

bool rocke_gett_hooks::wide_d() const
{
    return d_bytes() > rocke::gett::NARROW_D_BYTES;
}

rocke_gett_hooks::DPart
    rocke_gett_hooks::d_part(rocke_ir_builder_t* b, V* idx, const Digits& dg, int64_t lim, V* big)
{
    DPart out;
    V* e = digits(b, idx, dg);
    if(wide_d())
    {
        out.e = e;
        if(lim)
        {
            V* cl = ci(b, lim);
            out.ok = rocke_b_cmp_lt(b, idx, cl);
        }
        return out;
    }
    V* c2 = ci(b, 2);
    V* off = rocke_b_mul(b, e, c2);
    if(lim)
    {
        V* cl = ci(b, lim);
        V* lt = rocke_b_cmp_lt(b, idx, cl);
        off = rocke_b_select(b, lt, off, big);
    }
    out.off = off;
    return out;
}

V* rocke_gett_hooks::d_off(rocke_ir_builder_t* b, const DPart& r, const DPart& c)
{
    if(!wide_d())
        return rocke_b_add(b, r.off, c.off);
    V* e = rocke_b_add(b, r.e, c.e);
    /* bytes = 2 e < 2^32: an unsigned add */
    V* off = rocke_b_add_nuw(b, e, e);
    V* ok = c.ok == nullptr ? r.ok : (r.ok == nullptr ? c.ok : rocke_b_land(b, r.ok, c.ok));
    if(ok != nullptr)
    {
        V* s = ci(b, i32_bits(rocke::gett::WIDE_OOB));
        off = rocke_b_select(b, ok, off, s);
    }
    return off;
}

void rocke_gett_hooks::epilogue(rocke_ir_builder_t* b,
                                const std::vector<V*>& vals,
                                V* bm,
                                V* bn,
                                V* C,
                                V* warp_m_idx,
                                V* warp_n_idx,
                                V* lane,
                                const std::function<V*(V*)>& acc_out)
{
    const rocke::gett::Plan& pl = plan;
    int vd = pl.vec_d;
    V* c16 = ci(b, 16);
    V* q = rocke_b_div(b, lane, c16);
    V* cn = rocke_b_mod(b, lane, c16);
    int mfA = mf[0], mfB = mf[1];
    V* m0;
    {
        V* cw = ci(b, wmn[0]);
        V* wm = rocke_b_mul(b, warp_m_idx, cw);
        V* a1 = rocke_b_add(b, bm, wm);
        V* c4 = ci(b, 4);
        V* q4 = rocke_b_mul(b, q, c4);
        m0 = rocke_b_add(b, a1, q4);
    }
    V* n0;
    {
        V* cw = ci(b, wmn[1]);
        V* wn = rocke_b_mul(b, warp_n_idx, cw);
        V* a1 = rocke_b_add(b, bn, wn);
        n0 = rocke_b_add(b, a1, cn);
    }
    bool rag_m = pl.M % bmn[0] != 0;
    bool rag_n = pl.N % bmn[1] != 0;
    int64_t db = d_bytes();
    V* cdb = ci(b, i32_bits(db));
    V* rs_d = rocke_b_buffer_rsrc(b, C, cdb);
    V* rs_c = nullptr;
    if(k.bilinear)
    {
        V* cdb2 = ci(b, i32_bits(db));
        rs_c = rocke_b_buffer_rsrc(b, p["Cin"], cdb2);
    }
    V* c0 = ci(b, 0);
    V* big = ci(b, rocke::gett::OOB);
    std::vector<DPart> cols;
    for(int ni = 0; ni < mfB; ++ni)
    {
        V* cc = ci(b, ni * 16);
        V* n = rocke_b_add(b, n0, cc);
        cols.push_back(d_part(b, n, pl.d_col_digits, rag_n ? pl.N : 0, big));
    }
    V* alpha = p["alpha"];
    bool bf = rocke_type_eq(dt, rocke_bf16());
    if(k.epi_v16)
    {
        epilogue_v16(
            b, vals, bm, bn, warp_m_idx, warp_n_idx, lane, acc_out, rs_d, rs_c, big, bf, cols, C);
        return;
    }
    if(k.epi_pair && pl.vec_d8 && mfB % 2 == 0)
    {
        epilogue_pair(b, vals, bm, bn, warp_m_idx, warp_n_idx, lane, acc_out, rs_d, rs_c, big, bf);
        return;
    }
    auto rows_of = [&](int mi) {
        std::vector<DPart> rows;
        for(int v0 = 0; v0 < 4; v0 += vd)
        {
            V* cm = ci(b, mi * 16 + v0);
            V* m = rocke_b_add(b, m0, cm);
            rows.push_back(d_part(b, m, pl.d_row_digits, rag_m ? pl.M : 0, big));
        }
        return rows;
    };
    using CL = std::map<std::pair<int, int>, std::vector<V*>>;
    auto cload = [&](const std::vector<DPart>& rows) {
        CL cl;
        for(int ni = 0; ni < mfB; ++ni)
            for(size_t gi = 0; gi < rows.size(); ++gi)
            {
                V* off = d_off(b, rows[gi], cols[ni]);
                if(vd > 1)
                {
                    V* cv = rocke_b_buffer_load_vN(b, rs_c, off, c0, dt, vd);
                    std::vector<V*> xs;
                    for(int e = 0; e < vd; ++e)
                        xs.push_back(rocke_b_vec_extract(b, cv, e));
                    cl[{ni, (int)gi}] = xs;
                }
                else
                    cl[{ni, (int)gi}] = {rocke_b_buffer_load(b, rs_c, off, c0, dt)};
            }
        return cl;
    };
    int cpf = k.bilinear ? k.c_prefetch : 0;
    std::map<int, std::pair<std::vector<DPart>, CL>> pend;
    for(int mi = 0; mi < std::min(cpf, mfA); ++mi)
    {
        auto r_ = rows_of(mi);
        auto cl_ = cload(r_);
        pend[mi] = {r_, cl_};
    }
    for(int mi = 0; mi < mfA; ++mi)
    {
        std::vector<DPart> rows;
        CL cl;
        if(cpf)
        {
            rows = pend[mi].first;
            cl = pend[mi].second;
            pend.erase(mi);
            if(mi + cpf < mfA)
            {
                auto r_ = rows_of(mi + cpf);
                auto cl_ = cload(r_);
                pend[mi + cpf] = {r_, cl_};
            }
        }
        else
        {
            rows = rows_of(mi);
            if(k.bilinear)
                cl = cload(rows);
        }
        for(int ni = 0; ni < mfB; ++ni)
        {
            V* acc = acc_out(vals[mi * mfB + ni]);
            for(size_t gi = 0; gi < rows.size(); ++gi)
            {
                V* off = d_off(b, rows[gi], cols[ni]);
                std::vector<V*> xs;
                for(int e = 0; e < vd; ++e)
                {
                    V* x = rocke_b_vec_extract(b, acc, (int)gi * vd + e);
                    xs.push_back(rocke_b_fmul(b, alpha, x));
                }
                if(k.bilinear)
                {
                    const auto& cc = cl[{ni, (int)gi}];
                    for(size_t e = 0; e < xs.size(); ++e)
                    {
                        V* cf = rocke_b_cast_to_f32(b, cc[e]);
                        xs[e] = rocke_b_fma(b, p["beta"], cf, xs[e]);
                    }
                }
                if(vd == 1)
                {
                    V* h = rocke_b_cast_f32_to(b, xs[0], dt);
                    if(bf)
                        rocke_b_buffer_store_bf16(b, rs_d, off, c0, h);
                    else
                        rocke_b_buffer_store_f16(b, rs_d, off, c0, h);
                }
                else
                {
                    std::vector<V*> hs;
                    for(V* x : xs)
                        hs.push_back(rocke_b_cast_f32_to(b, x, dt));
                    V* vec = rocke_b_vec_pack(b, hs.data(), (int)hs.size(), dt);
                    if(bf)
                        rocke_b_buffer_store_vN_bf16(b, rs_d, off, c0, vec, vd / 2);
                    else
                        rocke_b_buffer_store_vN_f16(b, rs_d, off, c0, vec, vd / 2);
                }
            }
        }
    }
}

void rocke_gett_hooks::epilogue_rve(rocke_ir_builder_t* b,
                                    const std::vector<V*>& accf,
                                    V* row0,
                                    V* col0,
                                    int mfmas_m,
                                    int mfmas_n,
                                    int c_per_lane,
                                    V* C)
{
    const rocke::gett::Plan& pl = plan;
    if(mfmas_n != 8)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "the GETT row-interleave epilogue needs 8 columns per lane");
        return;
    }
    V* c0 = ci(b, 0);
    V* big = ci(b, rocke::gett::OOB);
    int64_t db = d_bytes();
    V* cdb = ci(b, i32_bits(db));
    V* rs_d = rocke_b_buffer_rsrc(b, C, cdb);
    V* rs_c = nullptr;
    if(k.bilinear)
    {
        V* cdb2 = ci(b, i32_bits(db));
        rs_c = rocke_b_buffer_rsrc(b, p["Cin"], cdb2);
    }
    bool bf = rocke_type_eq(dt, rocke_bf16());
    bool rag_m = pl.M % bmn[0] != 0;
    bool rag_n = pl.N % bmn[1] != 0;
    DPart cpart = d_part(b, col0, pl.d_col_digits, rag_n ? pl.N : 0, big);
    V* alpha = p["alpha"];
    std::vector<std::pair<int, int>> rows;
    for(int mi = 0; mi < mfmas_m; ++mi)
        for(int v = 0; v < c_per_lane; ++v)
            rows.push_back({mi, v});
    auto off_of = [&](int mi, int v) {
        V* cm = ci(b, v * mfmas_m + mi);
        V* m = rocke_b_add(b, row0, cm);
        DPart r = d_part(b, m, pl.d_row_digits, rag_m ? pl.M : 0, big);
        return d_off(b, r, cpart);
    };
    using CL = std::pair<V*, std::vector<V*>>;
    auto cload = [&](size_t i) {
        V* off = off_of(rows[i].first, rows[i].second);
        V* cv = rocke_b_buffer_load_vN(b, rs_c, off, c0, dt, 8);
        std::vector<V*> xs;
        for(int e = 0; e < 8; ++e)
            xs.push_back(rocke_b_vec_extract(b, cv, e));
        return CL{off, xs};
    };
    int cpf = k.bilinear ? k.c_prefetch : 0;
    std::map<size_t, CL> pend;
    for(size_t i = 0; i < std::min((size_t)cpf, rows.size()); ++i)
        pend[i] = cload(i);
    for(size_t i = 0; i < rows.size(); ++i)
    {
        int mi = rows[i].first, v = rows[i].second;
        V* off = nullptr;
        std::vector<V*> cl;
        bool have_c = false;
        if(k.bilinear)
        {
            if(cpf)
            {
                CL x = pend[i];
                pend.erase(i);
                if(i + cpf < rows.size())
                    pend[i + cpf] = cload(i + cpf);
                off = x.first;
                cl = x.second;
            }
            else
            {
                CL x = cload(i);
                off = x.first;
                cl = x.second;
            }
            have_c = true;
        }
        else
            off = off_of(mi, v);
        std::vector<V*> xs;
        for(int ni = 0; ni < 8; ++ni)
        {
            V* x = rocke_b_vec_extract(b, accf[mi * mfmas_n + ni], v);
            xs.push_back(rocke_b_fmul(b, alpha, x));
        }
        if(have_c)
            for(int e = 0; e < 8; ++e)
            {
                V* cf = rocke_b_cast_to_f32(b, cl[e]);
                xs[e] = rocke_b_fma(b, p["beta"], cf, xs[e]);
            }
        std::vector<V*> hs;
        for(V* x : xs)
            hs.push_back(rocke_b_cast_f32_to(b, x, dt));
        V* vec = rocke_b_vec_pack(b, hs.data(), 8, dt);
        if(bf)
            rocke_b_buffer_store_vN_bf16(b, rs_d, off, c0, vec, 4);
        else
            rocke_b_buffer_store_vN_f16(b, rs_d, off, c0, vec, 4);
    }
}

void rocke_gett_hooks::epilogue_v16(rocke_ir_builder_t* b,
                                    const std::vector<V*>& vals,
                                    V* bm,
                                    V* bn,
                                    V* warp_m_idx,
                                    V* warp_n_idx,
                                    V* lane,
                                    const std::function<V*(V*)>& acc_out,
                                    V* rs_d,
                                    V* rs_c,
                                    V* big,
                                    bool bf,
                                    const std::vector<DPart>& cols16,
                                    V* D_ptr)
{
    const rocke::gett::Plan& pl = plan;
    int mfA = mf[0], mfB = mf[1];
    V* c0 = ci(b, 0);
    V* c16q = ci(b, 16);
    V* q = rocke_b_div(b, lane, c16q);
    V* m_lane;
    {
        V* cw = ci(b, wmn[0]);
        V* wm = rocke_b_mul(b, warp_m_idx, cw);
        V* a1 = rocke_b_add(b, bm, wm);
        V* c32 = ci(b, 32);
        V* l32 = rocke_b_div(b, lane, c32);
        V* c16 = ci(b, 16);
        V* t1 = rocke_b_mul(b, l32, c16);
        V* c2 = ci(b, 2);
        V* q2 = rocke_b_mod(b, q, c2);
        V* c8 = ci(b, 8);
        V* t2 = rocke_b_mul(b, q2, c8);
        V* a2 = rocke_b_add(b, t1, t2);
        m_lane = rocke_b_add(b, a1, a2);
    }
    bool rag_m = pl.M % bmn[0] != 0;
    V* alpha = p["alpha"];
    const rocke_type_t* i32x2 = rocke_vector_type(b, rocke_i32(), 2);
    const rocke_type_t* vec8 = rocke_vector_type(b, dt, 8);
    bool nt = k.store_nt;
    struct Elem
    {
        V* e;
        V* ok;
    };
    std::vector<Elem> cols_e;
    if(nt)
    {
        bool rag_n = pl.N % bmn[1] != 0;
        V* c16 = ci(b, 16);
        V* cw = ci(b, wmn[1]);
        V* wn = rocke_b_mul(b, warp_n_idx, cw);
        V* a1 = rocke_b_add(b, bn, wn);
        V* lm = rocke_b_mod(b, lane, c16);
        V* n_lane = rocke_b_add(b, a1, lm);
        for(int ni = 0; ni < mfB; ++ni)
        {
            V* cc = ci(b, 16 * ni);
            V* n = rocke_b_add(b, n_lane, cc);
            V* e = digits(b, n, pl.d_col_digits);
            V* ok = nullptr;
            if(rag_n)
            {
                V* cN = ci(b, pl.N);
                ok = rocke_b_cmp_lt(b, n, cN);
            }
            cols_e.push_back({e, ok});
        }
    }
    struct Row
    {
        DPart d; /* !nt */
        Elem el; /* nt */
    };
    auto row_of = [&](int j) {
        Row r;
        V* cj = ci(b, 32 * j);
        V* m = rocke_b_add(b, m_lane, cj);
        if(nt)
        {
            r.el.e = digits(b, m, pl.d_row_digits);
            r.el.ok = nullptr;
            if(rag_m)
            {
                V* cM = ci(b, pl.M);
                r.el.ok = rocke_b_cmp_lt(b, m, cM);
            }
            return r;
        }
        r.d = d_part(b, m, pl.d_row_digits, rag_m ? pl.M : 0, big);
        return r;
    };
    auto elem = [&](const Row& roff, int ni) {
        Elem out;
        V* rok = roff.el.ok;
        V* cok = cols_e[ni].ok;
        V* ok = cok == nullptr ? rok : (rok == nullptr ? cok : rocke_b_land(b, rok, cok));
        out.e = rocke_b_add(b, roff.el.e, cols_e[ni].e);
        out.ok = ok;
        return out;
    };
    auto boff = [&](const Row& roff, int ni) -> V* {
        if(!nt)
            return d_off(b, roff.d, cols16[ni]);
        Elem el = elem(roff, ni);
        V* off = rocke_b_add_nuw(b, el.e, el.e);
        if(el.ok != nullptr)
        {
            int64_t sent = wide_d() ? rocke::gett::WIDE_OOB : rocke::gett::OOB;
            V* cs = ci(b, i32_bits(sent));
            off = rocke_b_select(b, el.ok, off, cs);
        }
        return off;
    };
    using CL = std::map<int, std::vector<V*>>;
    auto cload = [&](const Row& roff) {
        CL out;
        for(int ni = 0; ni < mfB; ++ni)
        {
            V* off = boff(roff, ni);
            V* cv = rocke_b_buffer_load_vN(b, rs_c, off, c0, dt, 8);
            std::vector<V*> xs;
            for(int e = 0; e < 8; ++e)
                xs.push_back(rocke_b_vec_extract(b, cv, e));
            out[ni] = xs;
        }
        return out;
    };
    int npair = mfA / 2;
    int cpf = k.bilinear ? k.c_prefetch : 0;
    std::map<int, std::pair<Row, CL>> pend;
    for(int j = 0; j < std::min(cpf, npair); ++j)
    {
        Row r_ = row_of(j);
        CL cl_ = cload(r_);
        pend[j] = {r_, cl_};
    }
    for(int j = 0; j < npair; ++j)
    {
        Row roff;
        CL cl;
        if(cpf)
        {
            roff = pend[j].first;
            cl = pend[j].second;
            pend.erase(j);
            if(j + cpf < npair)
            {
                Row r_ = row_of(j + cpf);
                CL cl_ = cload(r_);
                pend[j + cpf] = {r_, cl_};
            }
        }
        else
        {
            roff = row_of(j);
            if(k.bilinear)
                cl = cload(roff);
        }
        for(int ni = 0; ni < mfB; ++ni)
        {
            V* ax = acc_out(vals[(2 * j) * mfB + ni]);
            V* ay = acc_out(vals[(2 * j + 1) * mfB + ni]);
            V* off = nt ? nullptr : d_off(b, roff.d, cols16[ni]);
            V* vec;
            if(k.bilinear)
            {
                std::vector<V*> lo, hi;
                for(int e = 0; e < 4; ++e)
                {
                    V* xe = rocke_b_vec_extract(b, ax, e);
                    V* xi = rocke_b_bitcast(b, xe, rocke_i32());
                    V* ye = rocke_b_vec_extract(b, ay, e);
                    V* yi = rocke_b_bitcast(b, ye, rocke_i32());
                    V* x2 = nullptr;
                    V* y2 = nullptr;
                    rocke_b_permlane32_swap(b, xi, yi, &x2, &y2);
                    lo.push_back(rocke_b_bitcast(b, x2, rocke_f32()));
                    hi.push_back(rocke_b_bitcast(b, y2, rocke_f32()));
                }
                std::vector<V*> all = lo;
                all.insert(all.end(), hi.begin(), hi.end());
                std::vector<V*> hs;
                for(size_t e = 0; e < all.size(); ++e)
                {
                    V* cf = rocke_b_cast_to_f32(b, cl[ni][e]);
                    V* ax_ = rocke_b_fmul(b, alpha, all[e]);
                    V* x = rocke_b_fma(b, p["beta"], cf, ax_);
                    all[e] = x;
                }
                for(V* x : all)
                    hs.push_back(rocke_b_cast_f32_to(b, x, dt));
                vec = rocke_b_vec_pack(b, hs.data(), (int)hs.size(), dt);
            }
            else
            {
                V* dd[2];
                for(int w = 0; w < 2; ++w)
                {
                    V* a = w == 0 ? ax : ay;
                    std::vector<V*> hs;
                    for(int e = 0; e < 4; ++e)
                    {
                        V* x = rocke_b_vec_extract(b, a, e);
                        V* m = rocke_b_fmul(b, alpha, x);
                        hs.push_back(rocke_b_cast_f32_to(b, m, dt));
                    }
                    V* pk = rocke_b_vec_pack(b, hs.data(), 4, dt);
                    dd[w] = rocke_b_vec_bitcast(b, pk, i32x2);
                }
                V* x0 = nullptr;
                V* y0 = nullptr;
                V* x1 = nullptr;
                V* y1 = nullptr;
                {
                    V* a0 = rocke_b_vec_extract(b, dd[0], 0);
                    V* b0 = rocke_b_vec_extract(b, dd[1], 0);
                    rocke_b_permlane32_swap(b, a0, b0, &x0, &y0);
                }
                {
                    V* a1 = rocke_b_vec_extract(b, dd[0], 1);
                    V* b1 = rocke_b_vec_extract(b, dd[1], 1);
                    rocke_b_permlane32_swap(b, a1, b1, &x1, &y1);
                }
                V* four[4] = {x0, x1, y0, y1};
                V* pk = rocke_b_vec_pack(b, four, 4, rocke_i32());
                vec = rocke_b_vec_bitcast(b, pk, vec8);
            }
            if(!nt)
            {
                if(bf)
                    rocke_b_buffer_store_vN_bf16(b, rs_d, off, c0, vec, 4);
                else
                    rocke_b_buffer_store_vN_f16(b, rs_d, off, c0, vec, 4);
                continue;
            }
            Elem el = elem(roff, ni);
            rocke_mem_opts_t mo = ROCKE_MEM_OPTS_INIT;
            mo.temporal_hint = ROCKE_TEMPORAL_STREAMING;
            V* Cp = D_ptr;
            if(el.ok == nullptr)
                rocke_b_global_store_vN_ex(b, Cp, el.e, vec, 8, 16, &mo);
            else
            {
                rocke_if_t f = rocke_b_scf_if(b, el.ok);
                rocke_b_region_enter(b, f.then_region);
                rocke_b_global_store_vN_ex(b, Cp, el.e, vec, 8, 16, &mo);
                rocke_b_region_leave(b);
            }
        }
    }
}

void rocke_gett_hooks::epilogue_pair(rocke_ir_builder_t* b,
                                     const std::vector<V*>& vals,
                                     V* bm,
                                     V* bn,
                                     V* warp_m_idx,
                                     V* warp_n_idx,
                                     V* lane,
                                     const std::function<V*(V*)>& acc_out,
                                     V* rs_d,
                                     V* rs_c,
                                     V* big,
                                     bool bf)
{
    const rocke::gett::Plan& pl = plan;
    int mfA = mf[0], mfB = mf[1];
    V* c16 = ci(b, 16);
    V* q = rocke_b_div(b, lane, c16);
    V* cn = rocke_b_mod(b, lane, c16);
    V* c2 = ci(b, 2);
    V* q2 = rocke_b_mod(b, q, c2);
    V* c1 = ci(b, 1);
    V* odd = rocke_b_cmp_eq(b, q2, c1);
    V* c2b = ci(b, 2);
    V* oddi = rocke_b_mod(b, q, c2b);
    V* c0 = ci(b, 0);
    V* m0;
    {
        V* cw = ci(b, wmn[0]);
        V* wm = rocke_b_mul(b, warp_m_idx, cw);
        V* a1 = rocke_b_add(b, bm, wm);
        V* c2c = ci(b, 2);
        V* qd = rocke_b_div(b, q, c2c);
        V* c8 = ci(b, 8);
        V* t = rocke_b_mul(b, qd, c8);
        m0 = rocke_b_add(b, a1, t);
    }
    V* n0;
    {
        V* cw = ci(b, wmn[1]);
        V* wn = rocke_b_mul(b, warp_n_idx, cw);
        V* a1 = rocke_b_add(b, bn, wn);
        V* om = rocke_b_mul(b, oddi, c16);
        V* a2 = rocke_b_add(b, cn, om);
        n0 = rocke_b_add(b, a1, a2);
    }
    bool rag_m = pl.M % bmn[0] != 0;
    bool rag_n = pl.N % bmn[1] != 0;
    std::vector<DPart> cols;
    for(int j = 0; j < mfB / 2; ++j)
    {
        V* cj = ci(b, 32 * j);
        V* n = rocke_b_add(b, n0, cj);
        cols.push_back(d_part(b, n, pl.d_col_digits, rag_n ? pl.N : 0, big));
    }
    V* alpha = p["alpha"];
    auto row_of = [&](int mi) {
        V* cm = ci(b, mi * 16);
        V* m = rocke_b_add(b, m0, cm);
        return d_part(b, m, pl.d_row_digits, rag_m ? pl.M : 0, big);
    };
    using CL = std::map<int, std::vector<V*>>;
    auto cload = [&](const DPart& roff) {
        CL cl;
        for(int j = 0; j < mfB / 2; ++j)
        {
            V* off = d_off(b, roff, cols[j]);
            V* cv = rocke_b_buffer_load_vN(b, rs_c, off, c0, dt, 8);
            std::vector<V*> xs;
            for(int e = 0; e < 8; ++e)
                xs.push_back(rocke_b_vec_extract(b, cv, e));
            cl[j] = xs;
        }
        return cl;
    };
    int cpf = k.bilinear ? k.c_prefetch : 0;
    std::map<int, std::pair<DPart, CL>> pend;
    for(int mi = 0; mi < std::min(cpf, mfA); ++mi)
    {
        DPart r_ = row_of(mi);
        CL cl_ = cload(r_);
        pend[mi] = {r_, cl_};
    }
    for(int mi = 0; mi < mfA; ++mi)
    {
        DPart roff;
        CL cl;
        if(cpf)
        {
            roff = pend[mi].first;
            cl = pend[mi].second;
            pend.erase(mi);
            if(mi + cpf < mfA)
            {
                DPart r_ = row_of(mi + cpf);
                CL cl_ = cload(r_);
                pend[mi + cpf] = {r_, cl_};
            }
        }
        else
        {
            roff = row_of(mi);
            if(k.bilinear)
                cl = cload(roff);
        }
        for(int j = 0; j < mfB / 2; ++j)
        {
            V* ax = acc_out(vals[mi * mfB + 2 * j]);
            V* ay = acc_out(vals[mi * mfB + 2 * j + 1]);
            std::vector<V*> own, rcv;
            for(int e = 0; e < 4; ++e)
            {
                V* xe = rocke_b_vec_extract(b, ax, e);
                V* ye = rocke_b_vec_extract(b, ay, e);
                V* send = rocke_b_select(b, odd, xe, ye);
                rcv.push_back(rocke_b_warp_shuffle_xor(b, send, 16));
                own.push_back(rocke_b_select(b, odd, ye, xe));
            }
            std::vector<V*> xs8;
            for(int e = 0; e < 4; ++e)
                xs8.push_back(rocke_b_select(b, odd, rcv[e], own[e]));
            for(int e = 0; e < 4; ++e)
                xs8.push_back(rocke_b_select(b, odd, own[e], rcv[e]));
            for(V*& x : xs8)
                x = rocke_b_fmul(b, alpha, x);
            if(k.bilinear)
                for(size_t e = 0; e < xs8.size(); ++e)
                {
                    V* cf = rocke_b_cast_to_f32(b, cl[j][e]);
                    xs8[e] = rocke_b_fma(b, p["beta"], cf, xs8[e]);
                }
            std::vector<V*> hs;
            for(V* x : xs8)
                hs.push_back(rocke_b_cast_f32_to(b, x, dt));
            V* vec = rocke_b_vec_pack(b, hs.data(), (int)hs.size(), dt);
            V* off = d_off(b, roff, cols[j]);
            if(bf)
                rocke_b_buffer_store_vN_bf16(b, rs_d, off, c0, vec, 4);
            else
                rocke_b_buffer_store_vN_f16(b, rs_d, off, c0, vec, 4);
        }
    }
}
