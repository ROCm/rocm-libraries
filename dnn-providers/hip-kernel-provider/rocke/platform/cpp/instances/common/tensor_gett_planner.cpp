/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/tensor_gett_planner.h: the product route of dispatch/families/tensor_gett.py
 * (plan_tensor_gett with _plan_mn / _plan_kc / _spec_mn / _spec_kc / _mn_bodies /
 * _mn_persistent / _plan_cost / _table_plan) and the tensor GETT validity of
 * instances/common/tensor/gett.py (TensorGettSpec.is_valid_spec, GettHooks.__init__, the body
 * trait rules) on the full spec mirror of rocke/gemm_dispatch.h.
 */
#include "rocke/tensor_gett_planner.h"

#include "gemm_dispatch_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace rocke
{
namespace gett
{
namespace planner
{

namespace gd = rocke::gemm_dispatch;

namespace
{

/* the planner's constants of tensor_gett.py / tensor/gett.py (generated) */
#define ROCKE_GETT_CONST(type, name, value) constexpr type GETT_##name = value;
#include "gemm_dispatch_consts.inc"
#undef ROCKE_GETT_CONST

bool set_why(std::string* why, const std::string& m)
{
    if(why != nullptr)
        *why = m;
    return false;
}

std::string gemm_dt(const std::string& d)
{
    return (d == "f16" || d == "fp16") ? "fp16" : d;
}

bool in(const std::vector<std::string>& v, const std::string& m)
{
    return std::find(v.begin(), v.end(), m) != v.end();
}

/* -------------------------------------------------------------- JSON helpers */

std::string jmodemap(const std::map<std::string, int64_t>& m)
{
    ModeMap mm(m.begin(), m.end());
    return json_modemap(mm);
}

std::string knobs_json(const Knobs& k)
{
    std::vector<std::pair<std::string, std::string>> m = {
        {"pad", std::to_string(k.pad)},
        {"bilinear", k.bilinear ? "true" : "false"},
        {"epi_order", json_str(k.epi_order)},
        {"epi_pair", k.epi_pair ? "true" : "false"},
        {"lane_split", k.lane_split ? "true" : "false"},
        {"acc_pin", std::to_string(k.acc_pin)},
        {"frag_pin", k.frag_pin ? "true" : "false"},
        {"llvm_opts", json_str(k.llvm_opts)},
    };
    if(k.epi_v16)
        m.push_back({"epi_v16", "true"});
    if(k.c_prefetch != 0)
        m.push_back({"c_prefetch", std::to_string(k.c_prefetch)});
    if(k.store_nt)
        m.push_back({"store_nt", "true"});
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "{";
    for(size_t i = 0; i < m.size(); ++i)
        o += (i ? ", " : "") + json_str(m[i].first) + ": " + m[i].second;
    return o + "}";
}

/* ------------------------------------------------------------ body traits */

/* gett.BODY_TRAIT over the TraitSpec defaults */
gd::Trait body_trait_ref()
{
    gd::Trait t;
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

gd::Trait mn_body_trait(gd::Trait t)
{
    t.epilogue = "default";
    t.cshuffle_no_alias = false;
    t.pad_m = false;
    t.pad_n = false;
    t.dtl_tight_rsrc = false;
    t.dtl_tight_ragged_only = false;
    t.pad_fast_tile = false;
    t.dtl_soff0 = false;
    return t;
}

gd::Trait kc_body_trait(gd::Trait t)
{
    t.rve_store_nt = false;
    t.pad_fast_tile = false;
    t.cshuffle_no_alias = false;
    if(t.epilogue != "default")
        t.epilogue = "default";
    return t;
}

bool field_default(const gd::Trait& t, const char* name)
{
    gd::Trait d;
    return gd::detail::field_eq(gd::detail::get_trait_field(t, name),
                                gd::detail::get_trait_field(d, name));
}

/* gett.body_trait_errors (true: an error) */
bool body_trait_errors(const gd::Trait& tr)
{
    static const std::set<std::string> free_fields = {"dtl_stages",
                                                      "persistent_ctas",
                                                      "chiplet_wgm",
                                                      "chiplet_chunk_size",
                                                      "chiplet_num_xcds",
                                                      "dtl_cache_a",
                                                      "dtl_cache_b"};
    gd::Trait ref = body_trait_ref();
    std::vector<gd::FieldValue> a = gd::detail::trait_values(tr), b = gd::detail::trait_values(ref);
    const auto& names = gd::detail::trait_field_names();
    for(size_t i = 0; i < names.size(); ++i)
    {
        if(free_fields.count(names[i]))
            continue;
        if(!gd::detail::field_eq(a[i], b[i]))
            return true;
    }
    if(!(tr.dtl_stages == 2 || tr.dtl_stages == 3 || tr.dtl_stages == 4))
        return true;
    if(tr.persistent_ctas < 0)
        return true;
    return false;
}

/* gett.mn_body_errors (true: an error) */
bool mn_body_errors(const gd::UGemm& s)
{
    const gd::Tile& t = s.tile;
    const gd::Trait& tr = s.trait;
    if(s.data.layout != "CRR")
        return true;
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
        return true;
    if(t.warp_k != 1)
        return true;
    if(!(tr.direct_to_lds && tr.dtl_prefetch && tr.dtl_stages >= 2))
        return true;
    if(tr.epilogue != "default")
        return true;
    static const char* const fixed[] = {"pad_k",
                                        "persistent",
                                        "preshuffle_b",
                                        "active_tile_skip",
                                        "split_k",
                                        "mfma_swap_ab",
                                        "sab_vec16",
                                        "sab_store_nt",
                                        "streamk",
                                        "lds_row_interleave",
                                        "rilv_vec_epilogue",
                                        "rve_store_nt",
                                        "hb_sched",
                                        "sia_sched",
                                        "k_stagger",
                                        "k_stagger_rc",
                                        "stagger_u",
                                        "dtl_clamp_k",
                                        "dtl_k_tail",
                                        "dtl_base_fold",
                                        "pingpong",
                                        "asm_loop",
                                        "asm_loop_rawrd",
                                        "vgpr_prefetch",
                                        "vp_mn_kv",
                                        "mn_vec",
                                        "kc_row_guard",
                                        "pad_row_clamp",
                                        "pad_k_fast",
                                        "split_k_aligned",
                                        "split_k_serial",
                                        "dtl_addr_spread",
                                        "dtl_rsrc_word3"};
    for(const char* f : fixed)
        if(!field_default(tr, f))
            return true;
    if(!(tr.epilogue == "default" && !tr.cshuffle_no_alias && !tr.pad_m && !tr.pad_n
         && !tr.dtl_tight_rsrc && !tr.dtl_tight_ragged_only && !tr.pad_fast_tile && !tr.dtl_soff0))
        return true;
    if(tr.persistent_ctas < 0)
        return true;
    return false;
}

/* gett.kc_body_errors (true: an error) */
bool kc_body_errors(const gd::UGemm& s)
{
    const gd::Tile& t = s.tile;
    const gd::Trait& tr = s.trait;
    if(s.data.layout != "RCR")
        return true;
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16))
        return true;
    if(tr.epilogue != "default")
        return true;
    static const char* const fixed[] = {"persistent",
                                        "split_k",
                                        "mfma_swap_ab",
                                        "sab_vec16",
                                        "sab_store_nt",
                                        "streamk",
                                        "pad_k",
                                        "dtl_k_tail",
                                        "active_tile_skip"};
    for(const char* f : fixed)
        if(!field_default(tr, f))
            return true;
    if(tr.rve_store_nt || tr.pad_fast_tile || tr.cshuffle_no_alias)
        return true;
    return false;
}

int64_t span_bytes(const std::map<std::string, int64_t>& ext,
                   const std::map<std::string, int64_t>& st,
                   const std::vector<std::string>& modes)
{
    int64_t s = 0;
    for(const auto& m : modes)
        s += (ext.at(m) - 1) * st.at(m);
    return 2 * (1 + s);
}

} // namespace

/* ------------------------------------------------------------------- GettSpec */

bool GettSpec::plan(Plan* out, std::string* why) const
{
    const std::map<std::string, int64_t>* sa = has_strides_a ? &strides_a : nullptr;
    const std::map<std::string, int64_t>* sb = has_strides_b ? &strides_b : nullptr;
    const std::map<std::string, int64_t>* sd = has_strides_d ? &strides_d : nullptr;
    if(has_kc)
        return plan_gett(modes_a,
                         modes_b,
                         modes_d,
                         extents,
                         sa,
                         sb,
                         sd,
                         (int)kc_tile.tile_k,
                         true,
                         kc_trait.rilv_vec_epilogue ? 'B' : 'A',
                         out,
                         why);
    return plan_gett(modes_a, modes_b, modes_d, extents, sa, sb, sd, tile_k, false, 'A', out, why);
}

gd::UGemm GettSpec::body() const
{
    gd::UGemm s;
    std::string dt = gemm_dt(dtype);
    s.data.dtype_a = dt;
    s.data.dtype_b = dt;
    s.data.dtype_c = dt;
    if(has_kc)
    {
        s.name = kernel_name();
        s.tile = kc_tile;
        s.trait = kc_body_trait(kc_trait);
        s.data.layout = "RCR";
    }
    else if(has_mn)
    {
        s.name = kernel_name();
        s.tile = mn_tile;
        s.trait = mn_body_trait(mn_trait);
        s.data.layout = "CRR";
    }
    else
    {
        s.name = kernel_name();
        s.tile.tile_m = 256;
        s.tile.tile_n = 256;
        s.tile.tile_k = tile_k;
        s.tile.warp_m = 2;
        s.tile.warp_n = 2;
        s.tile.warp_k = 1;
        s.tile.warp_tile_m = 16;
        s.tile.warp_tile_n = 16;
        s.tile.warp_tile_k = 32;
        gd::Trait t = body_trait_ref();
        t.dtl_stages = dtl_stages;
        t.persistent_ctas = persistent_ctas;
        t.chiplet_wgm = chiplet_wgm;
        t.chiplet_chunk_size = chiplet_chunk_size;
        s.trait = t;
        s.data.layout = "RCR";
    }
    s.post_init();
    return s;
}

std::string GettSpec::json_key() const
{
    std::vector<std::pair<std::string, std::string>> m = {
        {"modes_a", json_modes(modes_a)},
        {"modes_b", json_modes(modes_b)},
        {"modes_d", json_modes(modes_d)},
        {"extents", jmodemap(extents)},
        {"strides_a", has_strides_a ? jmodemap(strides_a) : "null"},
        {"strides_b", has_strides_b ? jmodemap(strides_b) : "null"},
        {"strides_d", has_strides_d ? jmodemap(strides_d) : "null"},
        {"dtype", json_str(dtype)},
        {"knobs", knobs_json(knobs)},
        {"tile_k", std::to_string(tile_k)},
        {"dtl_stages", std::to_string(dtl_stages)},
        {"persistent_ctas", std::to_string(persistent_ctas)},
        {"chiplet_wgm", std::to_string(chiplet_wgm)},
        {"chiplet_chunk_size", std::to_string(chiplet_chunk_size)},
    };
    if(has_mn)
    {
        m.push_back({"mn_tile", gd::tile_json(mn_tile)});
        m.push_back({"mn_trait", gd::trait_json(mn_trait)});
    }
    if(has_kc)
    {
        m.push_back({"kc_tile", gd::tile_json(kc_tile)});
        m.push_back({"kc_trait", gd::trait_json(kc_trait)});
    }
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "{";
    for(size_t i = 0; i < m.size(); ++i)
        o += (i ? ", " : "") + json_str(m[i].first) + ": " + m[i].second;
    return o + "}";
}

std::string GettSpec::kernel_name() const
{
    std::string dt = (dtype == "f16" || dtype == "fp16") ? "f16" : dtype;
    return "tensor_gett_" + dt + "_" + sha1_hex(json_key()).substr(0, 12);
}

bool plans_arch(const std::string& arch)
{
    /* TENSOR_GETT_ARCHS (gen_gemm_dispatch_data.py checks the Python value) */
    return arch == "gfx950";
}

bool is_valid(const GettSpec& spec, const std::string& arch, std::string* why)
{
    if(!plans_arch(arch))
        return set_why(why, "arch not in ('gfx950',)");
    if(!(spec.dtype == "fp16" || spec.dtype == "bf16" || spec.dtype == "f16"))
        return set_why(why, "dtype not in ('fp16', 'bf16')");
    if(spec.tile_k != 32 && spec.tile_k != 64)
        return set_why(why, "tile_k not in (32, 64)");
    const Knobs& k = spec.knobs;
    if(k.pad != 0 && k.pad != 32)
        return set_why(why, "knobs.pad not in (0, 32)");
    if(k.epi_order != "mi" && k.epi_order != "ni")
        return set_why(why, "knobs.epi_order");
    if(k.acc_pin < 0 || k.acc_pin > 2)
        return set_why(why, "knobs.acc_pin");
    if(k.c_prefetch < 0 || k.c_prefetch > 8)
        return set_why(why, "knobs.c_prefetch");
    if(spec.has_kc)
    {
        if(spec.has_mn)
            return set_why(why, "kc_tile excludes mn_tile");
        if(!(spec.tile_k == 32 && spec.dtl_stages == 3 && spec.persistent_ctas == 0
             && spec.chiplet_wgm == 8 && spec.chiplet_chunk_size == 64))
            return set_why(why, "a ring-body field is set with a K-contiguous body");
    }
    else if(spec.has_mn)
    {
        if(spec.mn_tile.tile_k != spec.tile_k)
            return set_why(why, "tile_k != the MN body's tile_k");
        if(!(spec.dtl_stages == 3 && spec.persistent_ctas == 0 && spec.chiplet_wgm == 8
             && spec.chiplet_chunk_size == 64))
            return set_why(why, "a ring-body field is set with an MN body");
    }
    Plan plan;
    std::string w;
    if(!spec.plan(&plan, &w))
        return set_why(why, "no GETT plan: " + w);
    for(int r = 0; r < 2; ++r)
        if(plan.ops[r].nbytes > GETT_MAX_TENSOR_BYTES)
            return set_why(why, "operand spans more than 2 GiB");
    gd::UGemm body = spec.body();
    if(plan.kc)
    {
        std::string e = gd::launch_shape_error(body, plan.M, plan.N, plan.K);
        if(!e.empty())
            return set_why(why, "K-contiguous body: " + e);
    }
    if(span_bytes(plan.ext, plan.strides_d, plan.modes_d) > GETT_MAX_D_BYTES)
        return set_why(why, "D spans more than MAX_D_BYTES");
    /* GettHooks(plan, knobs, body) */
    const gd::Tile& t = body.tile;
    int64_t bs = body.block_size;
    int64_t T = t.tile_k;
    int64_t bmn[2] = {t.tile_m, t.tile_n};
    int64_t mfA = (t.warp_m * t.warp_tile_m) != 0
                      ? gd::detail::fdiv(t.tile_m, t.warp_m * t.warp_tile_m)
                      : 0;
    bool mn = body.data.layout == "CRR";
    if(plan.kc)
    {
        if(kc_body_errors(body))
            return set_why(why, "GETT K-contiguous body");
        if(k.epi_v16 || k.store_nt || k.epi_pair)
            return set_why(why, "epi_v16 / store_nt / epi_pair are MN-body knobs");
        if(body.trait.rilv_vec_epilogue && !plan.vec_dc8)
            return set_why(why,
                           "the row-interleave vector epilogue needs 8-element column runs of D");
    }
    else if(mn)
    {
        if(mn_body_errors(body))
            return set_why(why, "GETT MN body");
        if(T != plan.tile_k)
            return set_why(why, "spec tile_k != plan tile_k");
        for(int r = 0; r < 2; ++r)
            if(plan.ops[r].vec != 8)
                return set_why(why, "the MN body loads 16-byte runs");
        if(k.store_nt && !k.epi_v16)
            return set_why(why, "store_nt needs epi_v16");
        if(k.epi_v16 && !(plan.vec_d8 && gd::detail::pmod(mfA, 2) == 0))
            return set_why(why,
                           "epi_v16 needs 8-element D runs and an even number of M atoms per wave");
        bool ha, hb;
        gd::detail::MnLayout la, lb;
        if(!gd::detail::spec_mn_layouts(body, false, &ha, &la, &hb, &lb, &w))
            return set_why(why, w);
    }
    else
    {
        if(k.epi_v16 || k.store_nt)
            return set_why(why, "epi_v16 / store_nt are MN-body knobs");
        if(body_trait_errors(body.trait))
            return set_why(why, "GETT body");
        if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
            return set_why(why, "GETT fragments are built for the 16x16x32 atom");
        if(T != plan.tile_k)
            return set_why(why, "spec tile_k != plan tile_k");
        for(int r = 0; r < 2; ++r)
        {
            /* ops by the hipTensor operand name: GettHooks keys bmn by GEMM role A / B */
            int64_t lane = plan.ops[r].vec == 8 ? 16 : 4;
            int64_t tot = bmn[r] * T * 2;
            if(bs * lane == 0 || tot % (bs * lane))
                return set_why(why, "tile bytes not a multiple of one DTL pass");
            if(bmn[r] != 256 || bs != 256)
                return set_why(why, "GETT LDS block layout is built for tile_mn 256 and 4 waves");
        }
    }
    return gd::detail::is_valid_spec_x(body, arch, false, false, why);
}

/* -------------------------------------------------------- builder spec */

namespace
{

const char* lit(const std::string& s)
{
    static const char* const known[] = {"compv4",
                                        "compv3",
                                        "mem",
                                        "wsp3",
                                        "wmma_v1",
                                        "intrawave",
                                        "interwave",
                                        "default",
                                        "cshuffle"};
    for(const char* k : known)
        if(s == k)
            return k;
    return nullptr;
}

bool to_c_trait(const gd::Trait& t, rocke_gemm_trait_spec_t* out, std::string* why)
{
    std::string py = gd::python_only_levers(t);
    if(!py.empty())
        return set_why(why, "TraitSpec " + py + ": implemented by the Python engine only");
    rocke_gemm_universal_spec_t d = rocke_gemm_universal_spec_default();
    rocke_gemm_trait_spec_t c = d.trait;
    c.pipeline = lit(t.pipeline);
    c.scheduler = lit(t.scheduler);
    c.epilogue = lit(t.epilogue);
    if(c.pipeline == nullptr || c.scheduler == nullptr || c.epilogue == nullptr)
        return set_why(why, "pipeline / scheduler / epilogue value");
    c.pad_m = t.pad_m;
    c.pad_n = t.pad_n;
    c.pad_k = t.pad_k;
    c.persistent = t.persistent;
    c.chiplet_swizzle = t.chiplet_swizzle;
    c.chiplet_wgm = (int)t.chiplet_wgm;
    c.chiplet_num_xcds = (int)t.chiplet_num_xcds;
    c.chiplet_chunk_size = (int)t.chiplet_chunk_size;
    c.waves_per_eu_set = t.waves_per_eu.has_value();
    c.waves_per_eu = t.waves_per_eu.has_value() ? (int)*t.waves_per_eu : 0;
    c.preshuffle_b = t.preshuffle_b;
    c.direct_to_lds = t.direct_to_lds;
    c.dtl_cache_a = (int)t.dtl_cache_a;
    c.dtl_cache_b = (int)t.dtl_cache_b;
    c.dtl_prefetch = t.dtl_prefetch;
    c.active_tile_skip = t.active_tile_skip;
    c.lds_k_pad = (int)t.lds_k_pad;
    c.lds_swizzle = t.lds_swizzle;
    c.emit_sched_hints_set = t.emit_sched_hints.has_value();
    c.emit_sched_hints = t.emit_sched_hints.value_or(false);
    c.split_k = (int)t.split_k;
    c.cshuffle_no_alias = t.cshuffle_no_alias;
    c.ab_double_buffer_set = t.ab_double_buffer.has_value();
    c.ab_double_buffer = t.ab_double_buffer.value_or(false);
    c.interwave_setprio = t.interwave_setprio;
    c.dtl_stages = (int)t.dtl_stages;
    c.dtl_issue_spread = t.dtl_issue_spread;
    c.dtl_frag_prefetch = t.dtl_frag_prefetch;
    c.dtl_stream_barrier = t.dtl_stream_barrier;
    c.mfma_swap_ab = t.mfma_swap_ab;
    c.iglp_opt_set = t.iglp_opt.has_value();
    c.iglp_opt = t.iglp_opt.has_value() ? (int)*t.iglp_opt : 0;
    c.persistent_ctas = (int)t.persistent_ctas;
    c.acc_agpr = t.acc_agpr;
    c.mfma_interleave = t.mfma_interleave;
    c.acc_agpr_pin = t.acc_agpr_pin;
    c.ilv_dtl_every = (int)t.ilv_dtl_every;
    c.ilv_barrier_after = (int)t.ilv_barrier_after;
    c.ilv_merge = t.ilv_merge;
    c.lds_swizzle_r = (int)t.lds_swizzle_r;
    c.dtl_hoist = t.dtl_hoist;
    c.streamk = t.streamk;
    c.dtl_tight_rsrc = t.dtl_tight_rsrc;
    c.pad_fast_tile = t.pad_fast_tile;
    c.ilv_early_release = (int)t.ilv_early_release;
    c.ilv_release_at = (int)t.ilv_release_at;
    c.ilv_acquire_at = (int)t.ilv_acquire_at;
    c.acc_out_vgpr = t.acc_out_vgpr;
    c.ilv_read_span = (int)t.ilv_read_span;
    c.ilv_dtl_span = (int)t.ilv_dtl_span;
    c.lds_row_interleave = t.lds_row_interleave;
    c.rilv_pad = (int)t.rilv_pad;
    c.rilv_vec_epilogue = t.rilv_vec_epilogue;
    c.rve_store_nt = t.rve_store_nt;
    c.dtl_soff0 = t.dtl_soff0;
    c.wide_addr = t.wide_addr;
    c.mn_pad_set = true;
    c.mn_pad = (int)t.mn_pad;
    c.mn_chunk = (int)t.mn_chunk;
    c.mn_ld_any = t.mn_ld_any;
    c.hb_sched = t.hb_sched;
    c.hb_rel_slot = (int)t.hb_rel_slot;
    c.hb_acq_slot = (int)t.hb_acq_slot;
    c.hb_read_every = (int)t.hb_read_every;
    c.hb_dtl_every = (int)t.hb_dtl_every;
    c.hb_acq_b_slot = (int)t.hb_acq_b_slot;
    c.hb_split_rel = t.hb_split_rel;
    c.hb_rel_gap = (int)t.hb_rel_gap;
    c.kloop_unroll = (int)t.kloop_unroll;
    c.dtl_addr_spread = t.dtl_addr_spread;
    c.hb_dtl_delay = (int)t.hb_dtl_delay;
    c.hb_skew = (int)t.hb_skew;
    c.hb_skew_dtl = (int)t.hb_skew_dtl;
    c.hb_tile_lgkm0 = (int)t.hb_tile_lgkm0;
    c.das_anchor = t.das_anchor;
    c.das_late = t.das_late;
    c.mfma_serpentine = t.mfma_serpentine;
    c.mfma_order = (int)t.mfma_order;
    c.rve_pad = (int)t.rve_pad;
    c.dtl_tight_ragged_only = t.dtl_tight_ragged_only;
    *out = c;
    return true;
}

rocke_gemm_tile_spec_t to_c_tile(const gd::Tile& t)
{
    rocke_gemm_tile_spec_t c;
    c.tile_m = (int)t.tile_m;
    c.tile_n = (int)t.tile_n;
    c.tile_k = (int)t.tile_k;
    c.warp_m = (int)t.warp_m;
    c.warp_n = (int)t.warp_n;
    c.warp_k = (int)t.warp_k;
    c.warp_tile_m = (int)t.warp_tile_m;
    c.warp_tile_n = (int)t.warp_tile_n;
    c.warp_tile_k = (int)t.warp_tile_k;
    return c;
}

} // namespace

bool to_builder_spec(const GettSpec& s, TensorGettSpec* out, std::string* why)
{
    TensorGettSpec c = TensorGettSpec::make(s.modes_a, s.modes_b, s.modes_d, s.extents);
    if(s.has_strides_a)
    {
        c.has_strides_a = true;
        c.strides_a = ModeMap(s.strides_a.begin(), s.strides_a.end());
    }
    if(s.has_strides_b)
    {
        c.has_strides_b = true;
        c.strides_b = ModeMap(s.strides_b.begin(), s.strides_b.end());
    }
    if(s.has_strides_d)
    {
        c.has_strides_d = true;
        c.strides_d = ModeMap(s.strides_d.begin(), s.strides_d.end());
    }
    c.dtype = s.dtype;
    c.knobs = s.knobs;
    c.tile_k = s.tile_k;
    c.dtl_stages = s.dtl_stages;
    c.persistent_ctas = s.persistent_ctas;
    c.chiplet_wgm = s.chiplet_wgm;
    c.chiplet_chunk_size = s.chiplet_chunk_size;
    if(s.has_mn)
    {
        c.has_mn_tile = c.has_mn_trait = true;
        c.mn_tile = to_c_tile(s.mn_tile);
        if(!to_c_trait(s.mn_trait, &c.mn_trait, why))
            return false;
    }
    if(s.has_kc)
    {
        c.has_kc_tile = c.has_kc_trait = true;
        c.kc_tile = to_c_tile(s.kc_tile);
        if(!to_c_trait(s.kc_trait, &c.kc_trait, why))
            return false;
    }
    *out = c;
    return true;
}

/* ------------------------------------------------------------------ planner */

namespace
{

struct Req
{
    Problem p; /* the request problem (dtype as given) */
    std::string arch;
    std::string dtype; /* TensorGettRequest.dtype: "fp16" / "bf16" / ... */
};

/* the request problem as a TensorGettSpec (default levers, knobs.bilinear) */
GettSpec problem_spec(const Req& r)
{
    GettSpec s;
    s.modes_a = r.p.modes_a;
    s.modes_b = r.p.modes_b;
    s.modes_d = r.p.modes_d;
    s.extents = r.p.extents;
    s.has_strides_a = r.p.has_strides_a;
    s.has_strides_b = r.p.has_strides_b;
    s.has_strides_d = r.p.has_strides_d;
    s.strides_a = r.p.strides_a;
    s.strides_b = r.p.strides_b;
    s.strides_d = r.p.strides_d;
    s.dtype = r.p.dtype;
    s.knobs.bilinear = r.p.bilinear;
    return s;
}

bool request_ok(const Req& r)
{
    return r.dtype == "fp16" || r.dtype == "bf16";
}

gd::Request gemm_req(const Req& r, int64_t M, int64_t N, int64_t K, const std::string& layout)
{
    gd::Request g;
    g.M = M;
    g.N = N;
    g.K = K;
    g.arch = r.arch;
    g.dtype = r.dtype;
    g.layout = layout;
    return g;
}

/* a dispatched GETT launch (DispatchResult) */
struct Launch
{
    std::string candidate;
    GettSpec spec;
};

/* the ring GETT plan of the problem (replace(p, tile_k=32, mn_tile=None, mn_trait=None).plan()) */
bool ring_plan(const GettSpec& p, Plan* out)
{
    GettSpec r = p;
    r.tile_k = 32;
    r.has_mn = false;
    r.has_kc = false;
    std::string w;
    return r.plan(out, &w);
}

/* _mn_bodies */
std::vector<std::pair<gd::Tile, gd::Trait>> mn_bodies(const Req& r, const gd::Context& ctx)
{
    std::vector<std::pair<gd::Tile, gd::Trait>> out;
    Plan plan;
    if(!ring_plan(problem_spec(r), &plan))
        return out;
    gd::Pick pick;
    if(gd::dispatch_gemm(gemm_req(r, plan.M, plan.N, plan.K, "CRR"), ctx, &pick, nullptr)
       && !pick.spec.skinny)
        out.push_back({pick.spec.u.tile, pick.spec.u.trait});
    gd::Tile t;
    gd::Trait tr;
    {
        gd::detail::CusScope scope(ctx.cus);
        gd::detail::agpr_ilv_big_grid_tile_trait(ctx, &t, &tr);
    }
    tr.persistent_ctas = 0;
    out.push_back({t, tr});
    return out;
}

/* _mn_persistent */
gd::Trait mn_persistent(const Req& r,
                        const gd::Context& ctx,
                        const gd::Tile& tile,
                        const gd::Trait& trait)
{
    if(GETT_MN_PERSISTENT_MIN_WAVES == 0 || trait.persistent_ctas
       || !(tile.tile_m == 256 && tile.tile_n == 256))
        return trait;
    Plan plan;
    if(!ring_plan(problem_spec(r), &plan))
        return trait;
    int64_t cus = ctx.cus;
    int64_t tiles = gd::detail::cdiv(plan.M, 256) * gd::detail::cdiv(plan.N, 256);
    if(tiles < GETT_MN_PERSISTENT_MIN_WAVES * cus)
        return trait;
    gd::Trait o = trait;
    o.persistent_ctas = cus;
    return o;
}

/* _predicted(_gemm_req(req, M, N, K, layout), body) */
bool predicted(const Req& r,
               const gd::Context& ctx,
               int64_t M,
               int64_t N,
               int64_t K,
               const std::string& layout,
               const gd::UGemm& body,
               double* t)
{
    return gd::predicted_us(gemm_req(r, M, N, K, layout), body, ctx, t);
}

/* _spec_mn (explicit: the bodies are used as given) */
bool spec_mn(const Req& r,
             const gd::Context& ctx,
             const std::vector<std::pair<gd::Tile, gd::Trait>>* bodies,
             GettSpec* out)
{
    GettSpec p = problem_spec(r);
    std::vector<std::pair<gd::Tile, gd::Trait>> list = bodies ? *bodies : mn_bodies(r, ctx);
    std::vector<GettSpec> valid;
    for(auto body : list)
    {
        if(bodies == nullptr)
            body.second = mn_persistent(r, ctx, body.first, body.second);
        GettSpec sp = p;
        sp.knobs.bilinear = r.p.bilinear;
        sp.knobs.c_prefetch = r.p.bilinear ? GETT_MN_C_PREFETCH : 0;
        sp.dtype = r.dtype;
        sp.tile_k = (int)body.first.tile_k;
        sp.dtl_stages = 3;
        sp.persistent_ctas = 0;
        sp.chiplet_wgm = 8;
        sp.chiplet_chunk_size = 64;
        sp.has_mn = true;
        sp.mn_tile = body.first;
        sp.mn_trait = body.second;
        sp.has_kc = false;
        GettSpec v16 = sp;
        v16.knobs.epi_v16 = true;
        if(is_valid(v16, r.arch, nullptr))
            valid.push_back(v16);
        else if(is_valid(sp, r.arch, nullptr))
            valid.push_back(sp);
    }
    if(valid.empty())
        return false;
    if(valid.size() > 1)
    {
        bool all = true;
        double best = 0.0;
        size_t bi = 0;
        for(size_t i = 0; i < valid.size(); ++i)
        {
            Plan pl;
            std::string w;
            valid[i].plan(&pl, &w);
            double t;
            if(!predicted(r, ctx, pl.M, pl.N, pl.K, "CRR", valid[i].body(), &t))
            {
                all = false;
                break;
            }
            if(i == 0 || t < best)
            {
                best = t;
                bi = i;
            }
        }
        if(all)
        {
            *out = valid[bi];
            return true;
        }
    }
    for(const auto& s : valid)
    {
        Plan pl;
        std::string w;
        s.plan(&pl, &w);
        if(!pl.has_k_pad)
        {
            *out = s;
            return true;
        }
    }
    *out = valid[0];
    return true;
}

/* _spec_t32s3 */
GettSpec spec_t32s3(const Req& r)
{
    GettSpec s = problem_spec(r);
    s.dtype = r.dtype;
    s.tile_k = 32;
    s.dtl_stages = 3;
    s.persistent_ctas = 0;
    s.knobs.bilinear = r.p.bilinear;
    return s;
}

/* _kc_plan(problem, role) */
bool kc_plan(const GettSpec& p, char role, Plan* out)
{
    std::string w;
    return plan_gett(p.modes_a,
                     p.modes_b,
                     p.modes_d,
                     p.extents,
                     p.has_strides_a ? &p.strides_a : nullptr,
                     p.has_strides_b ? &p.strides_b : nullptr,
                     p.has_strides_d ? &p.strides_d : nullptr,
                     0,
                     true,
                     role,
                     out,
                     &w);
}

/* _spec_kc(req, body) */
bool spec_kc(const Req& r, const gd::Context& ctx, const gd::UGemm* body, GettSpec* out)
{
    GettSpec p = problem_spec(r);
    for(char role : {'B', 'A'})
    {
        Plan plan;
        if(!kc_plan(p, role, &plan))
            continue;
        gd::UGemm rs;
        if(body != nullptr)
            rs = *body;
        else
        {
            gd::Pick pick;
            if(!gd::dispatch_gemm(gemm_req(r, plan.M, plan.N, plan.K, "RCR"), ctx, &pick, nullptr))
                continue;
            if(pick.spec.skinny)
                continue;
            rs = pick.spec.u;
        }
        const gd::Trait& tr = rs.trait;
        if(tr.rilv_vec_epilogue != (role == 'B'))
            continue;
        if(role == 'B' && plan.d_col_digits.d[0].e < GETT_KC_MIN_D_RUN
           && plan.K < GETT_KC_SHORT_RUN_MIN_K)
            continue;
        GettSpec sp = p;
        sp.dtype = r.dtype;
        sp.tile_k = 32;
        sp.dtl_stages = 3;
        sp.persistent_ctas = 0;
        sp.chiplet_wgm = 8;
        sp.chiplet_chunk_size = 64;
        sp.has_mn = false;
        sp.has_kc = true;
        sp.kc_tile = rs.tile;
        sp.kc_trait = tr;
        sp.knobs.epi_v16 = false;
        sp.knobs.store_nt = false;
        sp.knobs.epi_pair = false;
        sp.knobs.c_prefetch = r.p.bilinear ? GETT_KC_C_PREFETCH : 0;
        if(is_valid(sp, r.arch, nullptr))
        {
            *out = sp;
            return true;
        }
    }
    return false;
}

/* _kc_in_place */
bool kc_in_place(const GettSpec& p, const GettSpec& sp)
{
    TensorGettSpec c;
    Plan plan;
    std::string w;
    if(!sp.plan(&plan, &w))
        return false;
    const std::string names[2] = {plan.ops[0].name, plan.ops[1].name};
    std::vector<std::string> k_outer(plan.k_order.rbegin(), plan.k_order.rend());
    for(int i = 0; i < 2; ++i)
    {
        std::vector<std::string> order = plan.ops[i].rows_order;
        order.insert(order.end(), k_outer.begin(), k_outer.end());
        const std::vector<std::string>& modes = names[i] == "A" ? p.modes_a : p.modes_b;
        bool has = names[i] == "A" ? p.has_strides_a : p.has_strides_b;
        std::map<std::string, int64_t> st = has ? (names[i] == "A" ? p.strides_a : p.strides_b)
                                                : packed_strides(modes, p.extents);
        std::map<std::string, int64_t> want = packed_strides(order, p.extents);
        for(const auto& m : modes)
            if(p.extents.at(m) > 1 && st.at(m) != want.at(m))
                return false;
    }
    (void)c;
    return true;
}

/* the tensor GETT registry: tensor_gett_mn (5), tensor_gett_t32s3 (10), tensor_gett_kc (20) */
bool dispatch_tensor_gett(const Req& r, const gd::Context& ctx, Launch* out, std::string* why)
{
    if(r.arch != "gfx950" || !request_ok(r))
        return set_why(why, "no candidate supports request");
    GettSpec sp;
    if(spec_mn(r, ctx, nullptr, &sp) && is_valid(sp, r.arch, nullptr))
    {
        out->candidate = "tensor_gett_mn";
        out->spec = sp;
        return true;
    }
    sp = spec_t32s3(r);
    if(is_valid(sp, r.arch, nullptr))
    {
        out->candidate = "tensor_gett_t32s3";
        out->spec = sp;
        return true;
    }
    if(spec_kc(r, ctx, nullptr, &sp) && kc_in_place(problem_spec(r), sp)
       && is_valid(sp, r.arch, nullptr))
    {
        out->candidate = "tensor_gett_kc";
        out->spec = sp;
        return true;
    }
    return set_why(why, "no candidate supports request");
}

bool mn_admits(const Req& r, const gd::Context& ctx)
{
    if(r.arch != "gfx950" || !request_ok(r))
        return false;
    GettSpec sp;
    return spec_mn(r, ctx, nullptr, &sp) && is_valid(sp, r.arch, nullptr);
}

void gett_grid(const GettSpec& spec, int grid[3], int* block)
{
    Plan plan;
    std::string w;
    spec.plan(&plan, &w);
    gd::UGemm body = spec.body();
    *block = (int)body.block_size;
    if(spec.has_mn || spec.has_kc)
    {
        const gd::Tile& t = body.tile;
        int64_t pc = body.trait.persistent_ctas;
        int64_t tn = gd::detail::cdiv(plan.N, t.tile_n), tm = gd::detail::cdiv(plan.M, t.tile_m);
        if(pc)
        {
            grid[0] = (int)std::max((int64_t)1, std::min(pc, tn * tm));
            grid[1] = grid[2] = 1;
            return;
        }
        grid[0] = (int)tn;
        grid[1] = (int)tm;
        grid[2] = 1;
        return;
    }
    if(spec.persistent_ctas)
    {
        grid[0] = spec.persistent_ctas;
        grid[1] = grid[2] = 1;
        return;
    }
    grid[0] = (int)gd::detail::cdiv(plan.N, 256);
    grid[1] = (int)gd::detail::cdiv(plan.M, 256);
    grid[2] = 1;
}

RoutePlan
    make_plan(const std::string& cand, const GettSpec& spec, const std::vector<PackStep>& packs)
{
    RoutePlan p;
    p.kind = packs.empty() ? "single" : "pack_hybrid";
    p.candidate = cand;
    p.gett = spec;
    p.packs = packs;
    gett_grid(spec, p.grid, &p.block);
    p.workspace_bytes = 0;
    for(const auto& s : packs)
        p.workspace_bytes += s.workspace_bytes;
    return p;
}

bool make_pack(const Req& r,
               const std::string& nm,
               const std::vector<std::string>& order,
               PackStep* out)
{
    const std::vector<std::string>& modes = nm == "A" ? r.p.modes_a : r.p.modes_b;
    bool has = nm == "A" ? r.p.has_strides_a : r.p.has_strides_b;
    const std::map<std::string, int64_t>& st = nm == "A" ? r.p.strides_a : r.p.strides_b;
    std::map<std::string, int64_t> oe;
    for(const auto& m : modes)
        oe[m] = r.p.extents.at(m);
    TensorPackSpec ps = TensorPackSpec::make(modes, order, oe, has ? &st : nullptr, r.dtype);
    std::string w;
    if(!pack_is_valid_spec(ps, r.arch.c_str(), &w))
        return false;
    PackStep s;
    s.operand = nm;
    s.spec = ps;
    tensor_pack_grid(ps, s.grid);
    s.block = tensor_pack_block(ps);
    s.workspace_bytes = 2 * ps.numel();
    *out = s;
    return true;
}

/* _pack_order (local) */
std::vector<std::string> pack_order(const std::vector<std::string>& modes,
                                    const std::vector<std::string>& k_modes,
                                    const std::map<std::string, int64_t>& ext)
{
    std::vector<std::string> full;
    for(const auto& m : modes)
        if(in(k_modes, m))
            full.push_back(m);
    for(const auto& m : modes)
        if(!in(k_modes, m))
            full.push_back(m);
    int64_t prod = 1;
    int first = -1;
    for(int i = (int)modes.size() - 1; i >= 0; --i)
    {
        if(in(k_modes, modes[i]))
            continue;
        prod *= ext.at(modes[i]);
        first = i;
        if(prod % 8 == 0)
            break;
    }
    if(first < 0 || prod % 8 != 0)
        return full;
    std::vector<std::string> out(modes.begin(), modes.begin() + first);
    std::vector<std::string> inner(modes.begin() + first, modes.end());
    for(const auto& m : inner)
        if(in(k_modes, m))
            out.push_back(m);
    for(const auto& m : inner)
        if(!in(k_modes, m))
            out.push_back(m);
    return out;
}

/* _hybrid_roles (false: None) */
bool hybrid_roles(const GettSpec& p, std::vector<std::pair<std::string, bool>>* out)
{
    const auto& ext = p.extents;
    std::map<std::string, int64_t> sa
        = p.has_strides_a && !p.strides_a.empty() ? p.strides_a : packed_strides(p.modes_a, ext);
    std::map<std::string, int64_t> sb
        = p.has_strides_b && !p.strides_b.empty() ? p.strides_b : packed_strides(p.modes_b, ext);
    std::map<std::string, int64_t> sd
        = p.has_strides_d && !p.strides_d.empty() ? p.strides_d : packed_strides(p.modes_d, ext);
    std::vector<std::string> m_modes, n_modes, k_modes;
    for(const auto& x : p.modes_d)
        if(in(p.modes_a, x) && !in(p.modes_b, x))
            m_modes.push_back(x);
    for(const auto& x : p.modes_d)
        if(in(p.modes_b, x) && !in(p.modes_a, x))
            n_modes.push_back(x);
    for(const auto& x : p.modes_a)
        if(in(p.modes_b, x) && !in(p.modes_d, x))
            k_modes.push_back(x);
    std::set<std::string> mn(m_modes.begin(), m_modes.end());
    mn.insert(n_modes.begin(), n_modes.end());
    std::set<std::string> mk(m_modes.begin(), m_modes.end());
    mk.insert(k_modes.begin(), k_modes.end());
    std::set<std::string> nk(n_modes.begin(), n_modes.end());
    nk.insert(k_modes.begin(), k_modes.end());
    std::set<std::string> dset(p.modes_d.begin(), p.modes_d.end());
    std::set<std::string> aset(p.modes_a.begin(), p.modes_a.end());
    std::set<std::string> bset(p.modes_b.begin(), p.modes_b.end());
    if(mn != dset || mk != aset || nk != bset || k_modes.empty() || m_modes.empty()
       || n_modes.empty())
        return false;
    /* d_fast = min(modes_d, key=sd): the first minimum */
    std::string d_fast = p.modes_d[0];
    for(const auto& x : p.modes_d)
        if(sd.at(x) < sd.at(d_fast))
            d_fast = x;
    struct O
    {
        std::string nm;
        const std::map<std::string, int64_t>* st;
        std::vector<std::string> free;
    };
    std::vector<O> order = {{"B", &sb, n_modes}, {"A", &sa, m_modes}};
    if(!in(n_modes, d_fast))
        std::reverse(order.begin(), order.end());
    out->clear();
    for(const auto& o : order)
    {
        std::vector<std::string> rows = o.free;
        std::stable_sort(rows.begin(), rows.end(), [&](const std::string& a, const std::string& b) {
            return -o.st->at(a) < -o.st->at(b);
        });
        std::vector<Digit> dig = merge_digits(rows, ext, *o.st);
        /* gett_plan._vec(dig, [st[m] for m in k_modes]) */
        int64_t v = 1;
        if(dig[0].s == 1)
        {
            v = 8;
            while(v > 1)
            {
                bool ok = dig[0].e % v == 0;
                for(size_t i = 1; i < dig.size() && ok; ++i)
                    ok = dig[i].s % v == 0;
                for(const auto& km : k_modes)
                    if(ok)
                        ok = o.st->at(km) % v == 0;
                if(ok)
                    break;
                v /= 2;
            }
        }
        out->push_back({o.nm, v < 8});
    }
    return true;
}

/* _plan_mn (bodies: explicit, no ring fallback) */
bool plan_mn(const Req& r,
             const gd::Context& ctx,
             const std::vector<std::pair<gd::Tile, gd::Trait>>* bodies,
             RoutePlan* out,
             std::string* why)
{
    GettSpec p = problem_spec(r);
    if(bodies != nullptr)
    {
        GettSpec sp;
        if(request_ok(r) && r.arch == "gfx950" && spec_mn(r, ctx, bodies, &sp)
           && is_valid(sp, r.arch, nullptr))
        {
            *out = make_plan("tensor_gett_mn", sp, {});
            return true;
        }
    }
    else if(mn_admits(r, ctx))
    {
        Launch l;
        if(!dispatch_tensor_gett(r, ctx, &l, why))
            return false;
        *out = make_plan(l.candidate, l.spec, {});
        return true;
    }
    Plan plan;
    bool have_plan = ring_plan(p, &plan);
    std::vector<std::pair<std::string, bool>> roles;
    bool have_roles = false;
    if(have_plan)
    {
        for(int i = 0; i < 2; ++i)
            roles.push_back({plan.ops[i].name, plan.ops[i].vec < 8});
        have_roles = true;
    }
    else
        have_roles = hybrid_roles(p, &roles); /* TENSOR_GETT_PACK_ANY_RUN */
    std::vector<PackStep> packs;
    bool packs_none = false;
    Req pr = r;
    if(have_roles)
    {
        std::vector<std::string> k_modes;
        for(const auto& x : p.modes_a)
            if(in(p.modes_b, x) && !in(p.modes_d, x))
                k_modes.push_back(x);
        for(const auto& role : roles)
        {
            if(!role.second)
                continue;
            const std::string& nm = role.first;
            const std::vector<std::string>& modes = nm == "A" ? p.modes_a : p.modes_b;
            std::vector<std::string> order = pack_order(modes, k_modes, p.extents);
            PackStep st;
            if(!make_pack(r, nm, order, &st))
            {
                packs_none = true;
                packs.clear();
                break;
            }
            packs.push_back(st);
            if(nm == "A")
            {
                pr.p.modes_a = order;
                pr.p.has_strides_a = false;
                pr.p.strides_a.clear();
            }
            else
            {
                pr.p.modes_b = order;
                pr.p.has_strides_b = false;
                pr.p.strides_b.clear();
            }
        }
    }
    if(!packs_none && !packs.empty())
    {
        GettSpec sp;
        bool got = false;
        std::string cand = "tensor_gett_mn";
        if(bodies != nullptr)
            got = request_ok(pr) && pr.arch == "gfx950" && spec_mn(pr, ctx, bodies, &sp)
                  && is_valid(sp, pr.arch, nullptr);
        else if(mn_admits(pr, ctx))
        {
            Launch l;
            got = dispatch_tensor_gett(pr, ctx, &l, nullptr);
            sp = l.spec;
            cand = l.candidate;
        }
        if(got)
        {
            *out = make_plan(cand, sp, packs);
            return true;
        }
    }
    if(bodies != nullptr)
        return set_why(why, "no MN-layout plan on the explicit body");
    Launch l;
    if(!dispatch_tensor_gett(r, ctx, &l, why))
        return false;
    *out = make_plan(l.candidate, l.spec, {});
    return true;
}

/* _plan_kc (body: an explicit RCR body) */
bool plan_kc(const Req& r, const gd::Context& ctx, const gd::UGemm* body, RoutePlan* out)
{
    GettSpec sp;
    if(!spec_kc(r, ctx, body, &sp))
        return false;
    Plan plan;
    std::string w;
    sp.plan(&plan, &w);
    std::vector<std::string> k_outer(plan.k_order.rbegin(), plan.k_order.rend());
    std::vector<PackStep> packs;
    for(int i = 0; i < 2; ++i)
    {
        std::vector<std::string> order = plan.ops[i].rows_order;
        order.insert(order.end(), k_outer.begin(), k_outer.end());
        PackStep st;
        if(!make_pack(r, plan.ops[i].name, order, &st))
            return false;
        packs.push_back(st);
    }
    *out = make_plan("tensor_gett_kc", sp, packs);
    return true;
}

double pack_cost_us(const PackStep& s)
{
    std::string f = s.spec.resolved_form();
    double c0, bw;
    if(f == "pairs")
        c0 = GETT_PACK_PAIRS_C0, bw = GETT_PACK_PAIRS_BW;
    else if(f == "tiled")
        c0 = GETT_PACK_TILED_C0, bw = GETT_PACK_TILED_BW;
    else
        c0 = GETT_PACK_ELEMENTS_C0, bw = GETT_PACK_ELEMENTS_BW;
    return c0 + 4.0 * (double)s.spec.numel() / bw;
}

/* _table_plan of the shipped tuned table (false: no entry / no plan; error: the Python
 * exception path) */
bool tuned_plan(const Req& r, const gd::Context& ctx, RoutePlan* out, bool* error)
{
    *error = false;
    static const std::vector<TunedEntry> entries = [] {
        std::vector<TunedEntry> e;
        std::string w;
        tuned_entries(rocke_tensor_gett_tuned_json(), "gfx950", &e, &w);
        return e;
    }();
    if(r.arch != "gfx950" || entries.empty())
        return false;
    TensorGettSpec key = TensorGettSpec::make(r.p.modes_a, r.p.modes_b, r.p.modes_d, r.p.extents);
    if(r.p.has_strides_a)
    {
        key.has_strides_a = true;
        key.strides_a = ModeMap(r.p.strides_a.begin(), r.p.strides_a.end());
    }
    if(r.p.has_strides_b)
    {
        key.has_strides_b = true;
        key.strides_b = ModeMap(r.p.strides_b.begin(), r.p.strides_b.end());
    }
    if(r.p.has_strides_d)
    {
        key.has_strides_d = true;
        key.strides_d = ModeMap(r.p.strides_d.begin(), r.p.strides_d.end());
    }
    key.dtype = r.dtype;
    key.knobs.bilinear = r.p.bilinear;
    std::string k = route_key(key);
    const TunedEntry* e = nullptr;
    for(const auto& x : entries)
        if(x.key == k)
            e = &x;
    if(e == nullptr)
        return false;
    gd::Tile tile;
    gd::Trait trait;
    if(!gd::parse_config(e->body, &tile, &trait, nullptr))
        return false;
    if(e->route == "mn")
    {
        std::vector<std::pair<gd::Tile, gd::Trait>> b = {{tile, trait}};
        std::string w;
        return plan_mn(r, ctx, &b, out, &w);
    }
    if(e->route == "kc")
    {
        Plan kp;
        if(!kc_plan(problem_spec(r), 'B', &kp))
            return false;
        gd::Pick pick;
        if(!gd::dispatch_gemm(gemm_req(r, kp.M, kp.N, kp.K, "RCR"), ctx, &pick, nullptr))
            return false;
        if(pick.spec.skinny)
        {
            /* replace(SkinnyGemmSpec, tile=..., trait=...) raises TypeError */
            *error = true;
            return false;
        }
        gd::UGemm body = pick.spec.u;
        body.tile = tile;
        body.trait = trait;
        return plan_kc(r, ctx, &body, out);
    }
    return false;
}

/* the request inside the planner's arithmetic domain: operand spans <= 2 GiB and D <= 4 GiB
 * (outside it every route is invalid: GETT operand / D spans, tensor_pack spans) */
bool in_domain(const Problem& p, std::string* why)
{
    auto span = [&](const std::vector<std::string>& modes,
                    bool has,
                    const std::map<std::string, int64_t>& st,
                    double* out) {
        std::map<std::string, int64_t> s
            = has && !st.empty() ? st : std::map<std::string, int64_t>();
        double acc = 1.0;
        if(!(has && !st.empty()))
        {
            for(const auto& m : modes)
                acc *= (double)p.extents.at(m);
            *out = 2.0 * acc;
            return;
        }
        double sum = 0.0;
        for(const auto& m : modes)
            sum += ((double)p.extents.at(m) - 1.0) * (double)s.at(m);
        *out = 2.0 * (1.0 + sum);
    };
    for(const auto& kv : p.extents)
        if(kv.second < 1 || kv.second > (int64_t(1) << 40))
            return set_why(why, "extent outside the planner's domain");
    /* every mode named by the extents and (when given) by the strides */
    const std::vector<std::string>* ml[3] = {&p.modes_a, &p.modes_b, &p.modes_d};
    const bool hs[3] = {p.has_strides_a, p.has_strides_b, p.has_strides_d};
    const std::map<std::string, int64_t>* sl[3] = {&p.strides_a, &p.strides_b, &p.strides_d};
    for(int i = 0; i < 3; ++i)
        for(const auto& m : *ml[i])
        {
            if(p.extents.find(m) == p.extents.end())
                return set_why(why, "mode '" + m + "' has no extent");
            if(hs[i] && sl[i]->find(m) == sl[i]->end())
                return set_why(why, "mode '" + m + "' has no stride");
        }
    for(const auto* st : {&p.strides_a, &p.strides_b, &p.strides_d})
        for(const auto& kv : *st)
            if(kv.second < 0 || kv.second > (int64_t(1) << 40))
                return set_why(why, "stride outside the planner's domain");
    double a, b, d;
    span(p.modes_a, p.has_strides_a, p.strides_a, &a);
    span(p.modes_b, p.has_strides_b, p.strides_b, &b);
    span(p.modes_d, p.has_strides_d, p.strides_d, &d);
    if(a > (double)0x7FFF0000 * 1.01 || b > (double)0x7FFF0000 * 1.01
       || d > (double)0xFFFFFF00 * 1.01)
        return set_why(why, "a tensor spans past the GETT / pack buffer ranges");
    return true;
}

} // namespace

bool plan_cost(const RoutePlan& plan, const std::string& arch, const gd::Context& ctx, double* out)
{
    const GettSpec& sp = plan.gett;
    if(!(sp.has_mn || sp.has_kc))
        return false;
    Plan pl;
    std::string w;
    if(!sp.plan(&pl, &w))
        return false;
    gd::UGemm body = sp.body();
    Req r;
    r.arch = arch;
    r.dtype = gemm_dt(sp.dtype);
    double t;
    if(!predicted(r, ctx, pl.M, pl.N, pl.K, body.data.layout, body, &t))
        return false;
    t *= sp.has_kc ? GETT_KC_TIME_FACTOR : GETT_MN_TIME_FACTOR;
    double s = 0.0;
    for(const auto& p : plan.packs)
        s += pack_cost_us(p);
    *out = t + s;
    return true;
}

bool plan_tensor_gett(const Problem& problem,
                      const std::string& arch,
                      const gd::Context& ctx,
                      RoutePlan* out,
                      std::string* why)
{
    Req r;
    r.p = problem;
    r.arch = arch;
    r.dtype = (problem.dtype == "f16" || problem.dtype == "fp16") ? "fp16" : problem.dtype;
    if(!in_domain(problem, why))
        return false;
    /* the tuned route table (policy auto) */
    {
        bool err = false;
        RoutePlan t;
        if(tuned_plan(r, ctx, &t, &err))
        {
            t.source = "tuned";
            *out = t;
            return true;
        }
        if(err)
            return set_why(why,
                           "tuned route table entry: the RCR pick is a gemm_skinny configuration");
    }
    RoutePlan mn, kc;
    std::string w;
    bool have_mn = plan_mn(r, ctx, nullptr, &mn, &w);
    bool have_kc = plan_kc(r, ctx, nullptr, &kc);
    if(!have_mn)
    {
        if(!have_kc)
            return set_why(why, w);
        kc.source = "kc";
        *out = kc;
        return true;
    }
    if(!have_kc)
    {
        mn.source = "mn";
        *out = mn;
        return true;
    }
    double t_mn = 0.0, t_kc = 0.0;
    bool hm = plan_cost(mn, arch, ctx, &t_mn);
    bool hk = plan_cost(kc, arch, ctx, &t_kc);
    if(!hm || !hk)
    {
        mn.source = "mn";
        mn.has_t_mn = hm;
        mn.t_mn = t_mn;
        mn.has_t_kc = hk;
        mn.t_kc = t_kc;
        *out = mn;
        return true;
    }
    RoutePlan& pick = (std::log(t_mn) - std::log(t_kc) > GETT_KC_MARGIN) ? kc : mn;
    pick.source = "cost";
    pick.has_t_mn = pick.has_t_kc = true;
    pick.t_mn = t_mn;
    pick.t_kc = t_kc;
    *out = pick;
    return true;
}

} // namespace planner
} // namespace gett
} // namespace rocke
