// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/attention_bwd_dq.py (the split dQ kernel
 * of the layout-general attention backward): AttnBwdDqSpec, its default
 * geometry, LDS plan, register model and validation, AttnBwdSpec.dq_spec and
 * the builder (_declare_dq, _lds_views, _emit_dq). The body is
 * _attention_bwd_dqbody.cpp. See rocke/instance_attention_bwd_dq.h.
 *
 * Every Python raise is one ckc::Error throw with the same message; the C
 * entry points catch it. A Python dataclasses.replace(...) re-runs
 * __post_init__, so a replace here is a field update followed by spec_init.
 * Byte identity: every builder call (constants included) is bound to a local
 * in Python's left-to-right, inner-first order; every barrier carries
 * ROCKE_ATTN_BWD_BARRIER_FENCE (the Python builder's fence_barriers).
 */
#include "rocke/instance_attention_bwd_dq.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.attention_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_dqbody.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
namespace
{

typedef std::vector<std::string> Strs;
typedef std::vector<int> Ints;
typedef rocke_ir_builder_t B;
typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;
typedef rocke_attn_bwd_dq_spec_t Spec;

const int NONE = ROCKE_ATTN_BWD_NONE;

/* ====================================================================== *
 * value sets (attention_bwd_dq.py)
 * ====================================================================== */

const Strs DQ_HEAD_LOOPS = {"single", "group"};
const Ints DQ_BLOCK_M_VALUES = {16, 32, 64, 128};
const Ints DQ_BLOCK_N_VALUES = {16, 32, 64, 128, 256};
const Strs DQ_TRANSPOSE_SOURCES = {"tr_read", "lds_plain"};
const Ints DQ_KV_PADS = {0, 8, 16, 32};
const char* const DQ_AGPR_ALLOC_BACKEND = "backend";
const char* const DQ_AGPR_ALLOCS_REPR = "(None, 'backend', (0, 0), (128, 128))";
const Strs DQ_STATIC_SCALES = {"fold_ds", "convert", "runtime"};
const Ints DQ_WAVES = {1, 2, 4, 8};
const Strs DQ_ATOMS = {"16x16x16", "16x16x32"};
const char* const DQ_WAVES_PER_EU_REPR = "(None, 1, 2)";
const Ints NARROW_VECS = {1, 2, 4};
const int REG_ALLOWANCE_FIXED = 24;
const int REGISTER_PRUNE_ALLOWANCE = 32;
const int CDNA_SIMDS_PER_CU = 4;
const int BUFFER_ALIGN = 128;
/* The lowering's flat-work-group-size attribute has a fixed minimum of 64. */
const int MIN_WG_ATTR = 64;

/* ------------------------------------------------------------ helpers */

std::string s_(const char* s)
{
    return s ? std::string(s) : std::string();
}

bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

bool in(const Strs& set, const char* v)
{
    return v != NULL && std::find(set.begin(), set.end(), std::string(v)) != set.end();
}

bool in(const Ints& set, int v)
{
    return std::find(set.begin(), set.end(), v) != set.end();
}

std::string repr_str(const char* v)
{
    return v == NULL ? std::string("None") : py_repr(v);
}

std::string repr_bool(bool v)
{
    return v ? "True" : "False";
}

/* _check_in(name, value, allowed) */
void check_in(const std::string& name, const char* value, const Strs& allowed)
{
    if(!in(allowed, value))
        throw ckc::ValueError(name + "=" + repr_str(value) + " not in " + py_tuple(allowed));
}

void check_in(const std::string& name, int value, const Ints& allowed)
{
    if(!in(allowed, value))
        throw ckc::ValueError(name + "=" + std::to_string(value) + " not in " + py_tuple(allowed));
}

/* _atom_k(atom) */
int atom_k(const char* atom)
{
    std::string a = s_(atom);
    size_t x = a.rfind('x');
    return std::stoi(a.substr(x + 1));
}

bool is_pair(const rocke_attn_bwd_agpr_alloc_t& a, int x, int y)
{
    return a.kind == ROCKE_ATTN_BWD_AGPR_PAIR && a.a == x && a.b == y;
}

rocke_attn_bwd_agpr_alloc_t agpr_pair(int x, int y)
{
    rocke_attn_bwd_agpr_alloc_t a;
    memset(&a, 0, sizeof(a));
    a.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
    a.a = x;
    a.b = y;
    return a;
}

int ceil_neg_div(int a, int b) /* -(-a // b) for a, b > 0 */
{
    return (a + b - 1) / b;
}

/* ====================================================================== *
 * the spec
 * ====================================================================== */

Spec spec_default()
{
    Spec s;
    memset(&s, 0, sizeof(s));
    s.head_size = 0;
    s.dtype = "fp16";
    s.seq_mode = "batched";
    s.stage_vec = 8;
    s.mask_class = "band";
    s.ws_layout = "head_major";
    s.static_scale = "fold_ds";
    s.waves = NONE;
    s.block_m = NONE;
    s.block_n = NONE;
    s.atom_g02 = NULL;
    s.atom_g4 = NULL;
    s.transpose_source = NULL;
    s.head_loop = "single";
    s.stats_prescale = true;
    s.edge_tiles = false;
    s.kv_pad = 0;
    s.kv_prefetch = false;
    s.waves_per_eu = NONE;
    s.agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_NONE;
    s.scheduler_strategy = NULL;
    s.sched = "none";
    s.xcd_remap = false;
    s.narrow_vec = 1;
    s.name = "rocke_attn_bwd";
    return s;
}

/* AttnBwdDqSpec.__post_init__ */
void dq_spec_init(Spec& s)
{
    check_in("head_size", s.head_size, BWD_HEAD_SIZES);
    check_in("dtype", s.dtype, BWD_DTYPES);
    check_in("seq_mode", s.seq_mode, SEQ_MODES);
    check_in("stage_vec", s.stage_vec, STAGE_VECS);
    check_in("mask_class", s.mask_class, MASK_CLASSES);
    check_in("ws_layout", s.ws_layout, WS_LAYOUTS);
    check_in("static_scale", s.static_scale, DQ_STATIC_SCALES);
    check_in("head_loop", s.head_loop, DQ_HEAD_LOOPS);
    check_in("kv_pad", s.kv_pad, DQ_KV_PADS);
    Strs scheds(ROCKE_ATTN_BWD_DQ_SCHEDS, ROCKE_ATTN_BWD_DQ_SCHEDS + ROCKE_ATTN_BWD_N_DQ_SCHEDS);
    check_in("sched", s.sched, scheds);
    check_in("narrow_vec", s.narrow_vec, NARROW_VECS);
    if(s.narrow_vec != 1 && s.stage_vec != 1)
        throw ckc::ValueError("narrow_vec applies to stage_vec = 1 only");
    if(s.waves != NONE)
        check_in("waves", s.waves, DQ_WAVES);
    if(s.block_m != NONE)
        check_in("block_m", s.block_m, DQ_BLOCK_M_VALUES);
    if(s.block_n != NONE)
        check_in("block_n", s.block_n, DQ_BLOCK_N_VALUES);
    if(s.atom_g02 != NULL)
        check_in("atom_g02", s.atom_g02, DQ_ATOMS);
    if(s.atom_g4 != NULL)
        check_in("atom_g4", s.atom_g4, DQ_ATOMS);
    if(s.transpose_source != NULL)
        check_in("transpose_source", s.transpose_source, DQ_TRANSPOSE_SOURCES);
    if(s.waves_per_eu != NONE && s.waves_per_eu != 1 && s.waves_per_eu != 2)
        throw ckc::ValueError("waves_per_eu=" + std::to_string(s.waves_per_eu) + " not in "
                              + DQ_WAVES_PER_EU_REPR);
    const rocke_attn_bwd_agpr_alloc_t& a = s.agpr_alloc;
    if(a.kind != ROCKE_ATTN_BWD_AGPR_NONE)
    {
        bool ok = (a.kind == ROCKE_ATTN_BWD_AGPR_STR && eq(a.str, DQ_AGPR_ALLOC_BACKEND))
                  || is_pair(a, 0, 0) || is_pair(a, 128, 128);
        if(!ok)
            throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(a) + " not in "
                                  + DQ_AGPR_ALLOCS_REPR);
    }
    if(s.scheduler_strategy != NULL && !rocke_scheduler_strategy_is_valid(s.scheduler_strategy))
        throw ckc::ValueError("unsupported scheduler_strategy " + py_repr(s.scheduler_strategy)
                              + "; expected one of: iterative-ilp, iterative-maxocc, "
                                "iterative-minreg, max-ilp, max-memory-clause, or None");
    if(!name_ok(s.name))
        throw ckc::ValueError("name=" + repr_str(s.name) + " must be a non-empty identifier");
}

/* dq_static_scale(main_static_scale) */
const char* dq_static_scale(const char* main)
{
    static const Strs keys = {"fold_ds", "convert", "dq_before_atomic", "runtime"};
    check_in("static_scale", main, keys);
    if(eq(main, "fold_ds"))
        return "fold_ds";
    if(eq(main, "runtime"))
        return "runtime";
    return "convert";
}

/* dq_static_unit_mults(static_scale) */
unsigned dq_static_unit_mults(const char* static_scale)
{
    check_in("static_scale", static_scale, DQ_STATIC_SCALES);
    if(eq(static_scale, "fold_ds"))
        return ROCKE_ATTN_BWD_UNIT_DQ_MULT;
    if(eq(static_scale, "convert"))
        return ROCKE_ATTN_BWD_UNIT_DS_MULT;
    return 0;
}

/* attn_bwd_dq_default_geometry(arch, head_size, dtype, stage_vec) */
Spec default_geometry(const std::string& arch, int head_size, const char* dtype, int stage_vec)
{
    if(std::find(SPLIT_DQ_ARCHS.begin(), SPLIT_DQ_ARCHS.end(), arch) == SPLIT_DQ_ARCHS.end())
        throw ckc::ValueError("the split dQ kernel is built for " + py_tuple(SPLIT_DQ_ARCHS)
                              + " only, not " + py_repr(arch));
    check_in("head_size", head_size, BWD_HEAD_SIZES);
    check_in("dtype", dtype, BWD_DTYPES);
    check_in("stage_vec", stage_vec, STAGE_VECS);
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    /* _DQ_TILES[(key, head_size)]: (kM0, kN0, W) */
    static const std::map<std::pair<std::string, int>, std::vector<int>> tiles = {
        {{"gfx942", 32}, {64, 64, 4}},
        {{"gfx942", 64}, {64, 64, 4}},
        {{"gfx942", 128}, {64, 64, 4}},
        {{"gfx950", 32}, {64, 64, 4}},
        {{"gfx950", 64}, {64, 64, 4}},
        {{"gfx950", 128}, {64, 64, 4}},
    };
    std::string key = facts.has_tr_read ? "gfx950" : "gfx942";
    const std::vector<int>& t = tiles.at({key, head_size});
    bool wide = facts.wide_k_atom && head_size % 32 == 0;
    Spec g = spec_default();
    g.head_size = head_size;
    g.waves = t[2];
    g.block_m = t[0];
    g.block_n = t[1];
    g.atom_g02 = wide ? "16x16x32" : "16x16x16";
    g.atom_g4 = "16x16x16";
    g.transpose_source = facts.has_tr_read ? "tr_read" : "lds_plain";
    return g;
}

/* _require_resolved(spec) */
void require_resolved(const Spec& s)
{
    std::vector<std::string> missing;
    if(s.waves == NONE)
        missing.push_back("waves");
    if(s.block_m == NONE)
        missing.push_back("block_m");
    if(s.block_n == NONE)
        missing.push_back("block_n");
    if(s.atom_g02 == NULL)
        missing.push_back("atom_g02");
    if(s.atom_g4 == NULL)
        missing.push_back("atom_g4");
    if(s.transpose_source == NULL)
        missing.push_back("transpose_source");
    if(!missing.empty())
    {
        std::string l = "[";
        for(size_t i = 0; i < missing.size(); ++i)
            l += (i ? ", " : "") + py_repr(missing[i]);
        throw ckc::ValueError("dQ spec is not resolved (call .resolved(arch)): " + l + "]");
    }
}

/* _raw_estimates(spec, arch, agpr_form=) */
std::pair<int, int> raw_estimates(const Spec& s, const std::string& arch, bool agpr_form)
{
    int wave = arch_facts(arch).wave_size;
    int w = s.waves, d = s.head_size, km = s.block_m, kn = s.block_n;
    int cols = km / w;
    int k02 = atom_k(s.atom_g02), k4 = atom_k(s.atom_g4);
    int vgpr = REG_ALLOWANCE_FIXED;
    vgpr += 2 * cols * d / (2 * wave); /* resident Q^T, dO^T */
    vgpr += kn * cols / (2 * wave); /* dS^T as the G4' A operand */
    vgpr += kn * k02 / (2 * wave); /* one K step of the G0' / G2' A fragments */
    vgpr += d * k4 / (2 * wave); /* one K step of the G4' B fragments */
    vgpr += 6 * std::max(1, cols / 16); /* per-column lse2, Dsum, flags, limits */
    if(s.kv_prefetch)
        vgpr += 2 * kn * d / (2 * wave * w); /* the next K and V tile in flight */
    int acc = 2 * kn * cols / wave + cols * d / wave; /* S^T, dP^T, dQ (fp32) */
    if(agpr_form)
        return {vgpr, acc};
    return {vgpr + acc, 0};
}

/* AttnBwdDqSpec.resolved(arch) */
Spec resolved(const Spec& s, const std::string& arch)
{
    Spec d = default_geometry(arch, s.head_size, s.dtype, s.stage_vec);
    Spec out = s;
    if(out.waves == NONE)
        out.waves = d.waves;
    if(out.block_m == NONE)
        out.block_m = d.block_m;
    if(out.block_n == NONE)
        out.block_n = d.block_n;
    if(out.atom_g02 == NULL)
        out.atom_g02 = d.atom_g02;
    if(out.atom_g4 == NULL)
        out.atom_g4 = d.atom_g4;
    if(out.transpose_source == NULL)
        out.transpose_source = d.transpose_source;
    dq_spec_init(out);
    if(out.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_NONE)
    {
        int vgpr = raw_estimates(out, arch, false).first;
        if(vgpr <= arch_facts(arch).arch_vgprs)
        {
            out.agpr_alloc = agpr_pair(0, 0);
            dq_spec_init(out);
        }
    }
    return out;
}

/* non_default_fields(self, skip=("head_size", "name")) */
std::vector<std::pair<std::string, std::string>> non_default_fields(const Spec& s)
{
    const Spec d = spec_default();
    std::vector<std::pair<std::string, std::string>> out;
    auto str = [&](const char* n, const char* v, const char* dv) {
        if(!eq(v, dv))
            out.emplace_back(n, repr_str(v));
    };
    auto num = [&](const char* n, int v, int dv) {
        if(v != dv)
            out.emplace_back(n, std::to_string(v));
    };
    auto opt_num = [&](const char* n, int v) {
        if(v != NONE)
            out.emplace_back(n, std::to_string(v));
    };
    auto flag = [&](const char* n, bool v, bool dv) {
        if(v != dv)
            out.emplace_back(n, repr_bool(v));
    };
    str("dtype", s.dtype, d.dtype);
    str("seq_mode", s.seq_mode, d.seq_mode);
    num("stage_vec", s.stage_vec, d.stage_vec);
    str("mask_class", s.mask_class, d.mask_class);
    str("ws_layout", s.ws_layout, d.ws_layout);
    str("static_scale", s.static_scale, d.static_scale);
    opt_num("waves", s.waves);
    opt_num("block_m", s.block_m);
    opt_num("block_n", s.block_n);
    str("atom_g02", s.atom_g02, NULL);
    str("atom_g4", s.atom_g4, NULL);
    str("transpose_source", s.transpose_source, NULL);
    str("head_loop", s.head_loop, d.head_loop);
    flag("stats_prescale", s.stats_prescale, d.stats_prescale);
    flag("edge_tiles", s.edge_tiles, d.edge_tiles);
    num("kv_pad", s.kv_pad, d.kv_pad);
    flag("kv_prefetch", s.kv_prefetch, d.kv_prefetch);
    opt_num("waves_per_eu", s.waves_per_eu);
    if(s.agpr_alloc.kind != ROCKE_ATTN_BWD_AGPR_NONE)
        out.emplace_back("agpr_alloc", agpr_alloc_repr(s.agpr_alloc));
    str("scheduler_strategy", s.scheduler_strategy, NULL);
    str("sched", s.sched, d.sched);
    flag("xcd_remap", s.xcd_remap, d.xcd_remap);
    num("narrow_vec", s.narrow_vec, d.narrow_vec);
    return out;
}

/* AttnBwdDqSpec.kernel_name() */
std::string dq_kernel_name(const Spec& s)
{
    std::string base = s_(s.name) + "_dq_" + s_(s.dtype) + "_d" + std::to_string(s.head_size) + "_"
                       + s_(s.seq_mode) + "_" + s_(s.mask_class) + "_sv"
                       + std::to_string(s.stage_vec);
    static const Strs in_base = {"dtype", "seq_mode", "stage_vec", "mask_class"};
    std::vector<std::pair<std::string, std::string>> items;
    for(const auto& kv : non_default_fields(s))
        if(std::find(in_base.begin(), in_base.end(), kv.first) == in_base.end())
            items.push_back(kv);
    std::stable_sort(
        items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    if(items.empty())
        return base;
    std::string r = "[";
    for(size_t i = 0; i < items.size(); ++i)
    {
        if(i)
            r += ", ";
        r += "(" + py_repr(items[i].first) + ", " + items[i].second + ")";
    }
    r += "]";
    return base + "_" + sha1_hex(r).substr(0, 10);
}

/* AttnBwdSpec.dq_spec(**knobs) */
Spec dq_spec_of(const rocke_attn_bwd_spec_t& m, const Spec* knobs)
{
    Spec s = knobs ? *knobs : spec_default();
    s.head_size = m.head_size;
    s.dtype = m.dtype;
    s.seq_mode = m.seq_mode;
    s.stage_vec = m.stage_vec;
    s.mask_class = m.mask_class;
    s.ws_layout = m.ws_layout;
    s.static_scale = dq_static_scale(m.static_scale);
    s.narrow_vec = m.narrow_vec;
    s.name = m.name;
    dq_spec_init(s);
    return s;
}

/* ====================================================================== *
 * LDS plan, registers, legality
 * ====================================================================== */

int align_up(int n, int a = BUFFER_ALIGN)
{
    return (n + a - 1) / a * a;
}

/* dq_lds_plan(spec, arch) */
rocke_attn_bwd_dq_lds_plan_t dq_lds_plan(const Spec& s, const std::string& arch)
{
    require_resolved(s);
    const int eb = 2;
    int d = s.head_size, km = s.block_m, kn = s.block_n;
    int kv_pitch = d + s.kv_pad / eb;
    struct Buf
    {
        const char* name;
        int rows, pitch;
    };
    const std::vector<std::vector<Buf>> phases = {
        {{"q", km, d}, {"do", km, d}},
        {{"k", kn, kv_pitch}, {"v", kn, kv_pitch}},
        {{"out", km, d}},
    };
    rocke_attn_bwd_dq_lds_plan_t p;
    memset(&p, 0, sizeof(p));
    int total = 0, i = 0;
    for(const auto& phase : phases)
    {
        int off = 0;
        for(const Buf& bf : phase)
        {
            int nbytes = bf.rows * bf.pitch * eb;
            p.buffers[i++]
                = rocke_attn_bwd_dq_lds_buffer_t{bf.name, off, nbytes, bf.pitch, bf.rows};
            off = align_up(off + nbytes);
        }
        total = std::max(total, off);
    }
    p.arch = arch_facts(arch).arch;
    p.total_bytes = total;
    p.capacity_bytes = arch_facts(arch).lds_capacity_bytes;
    return p;
}

const rocke_attn_bwd_dq_lds_buffer_t& plan_buffer(const rocke_attn_bwd_dq_lds_plan_t& p,
                                                  const char* name)
{
    for(int i = 0; i < ROCKE_ATTN_BWD_DQ_LDS_BUFFERS; ++i)
        if(eq(p.buffers[i].name, name))
            return p.buffers[i];
    throw ckc::KeyError("no dQ LDS buffer " + py_repr(name));
}

/* dq_register_estimates(spec, arch) */
std::pair<int, int> dq_register_estimates(const Spec& s, const std::string& arch)
{
    require_resolved(s);
    bool agpr_form = !is_pair(s.agpr_alloc, 0, 0);
    return raw_estimates(s, arch, agpr_form);
}

/* validate_attn_bwd_dq_spec(spec, arch) without the refusal tables. */
Spec validate_dq_spec(const Spec& spec, const std::string& arch)
{
    if(std::find(SPLIT_DQ_ARCHS.begin(), SPLIT_DQ_ARCHS.end(), arch) == SPLIT_DQ_ARCHS.end())
        throw ckc::ValueError("the split dQ kernel is built for " + py_tuple(SPLIT_DQ_ARCHS)
                              + " only, not " + py_repr(arch)
                              + " (its dS^T hand-off is the MFMA C-to-A relabel)");
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    Spec s = resolved(spec, arch);
    int w = s.waves, km = s.block_m, kn = s.block_n, d = s.head_size;
    int lw[4];
    int n_lw = rocke_attn_bwd_facts_legal_waves(&facts, lw);
    check_in("waves on " + arch, w, Ints(lw, lw + n_lw));
    const char* la[4];
    int n_la = rocke_attn_bwd_facts_legal_atoms(&facts, la);
    Strs atoms(la, la + n_la);
    check_in("atom_g02 on " + arch, s.atom_g02, atoms);
    check_in("atom_g4 on " + arch, s.atom_g4, atoms);
    const char* lt[4];
    int n_lt = rocke_attn_bwd_facts_legal_transpose_sources(&facts, lt);
    Strs tsrc;
    for(int i = 0; i < n_lt; ++i)
        if(in(DQ_TRANSPOSE_SOURCES, lt[i]))
            tsrc.push_back(lt[i]);
    check_in("transpose_source on " + arch, s.transpose_source, tsrc);
    if(km % (16 * w))
        throw ckc::ValueError("block_m=" + std::to_string(km)
                              + " must be a multiple of 16*waves=" + std::to_string(16 * w));
    if(kn % atom_k(s.atom_g4))
        throw ckc::ValueError("block_n=" + std::to_string(kn)
                              + " must be a multiple of the G4' atom K");
    if(d % atom_k(s.atom_g02))
        throw ckc::ValueError("head_size=" + std::to_string(d)
                              + " must be a multiple of the G0' atom K");
    {
        /* BwdDqGeometry.from_spec(s, arch): the relabel / transpose-read
         * proofs (no IR; a scratch builder carries the error) */
        rocke_ir_builder_t sb;
        if(rocke_ir_builder_init(&sb, "rocke_attn_bwd_dq_check") != ROCKE_OK)
            throw ckc::OOMError("validate_attn_bwd_dq_spec: scratch builder");
        rocke_attn_bwd_dq_geometry_t g;
        bool ok = rocke_attn_bwd_dq_geometry_from_spec(&sb, &s, arch.c_str(), &g);
        rocke_status_t st = sb.status;
        std::string msg = sb.err;
        rocke_ir_builder_free(&sb);
        if(!ok)
            ckc::raise_status(st == ROCKE_OK ? ROCKE_ERR_VALUE : st, msg.c_str());
    }
    rocke_attn_bwd_dq_lds_plan_t plan = dq_lds_plan(s, arch);
    if(plan.total_bytes > plan.capacity_bytes)
        throw ckc::ValueError("LDS plan " + std::to_string(plan.total_bytes) + " B exceeds "
                              + std::to_string(plan.capacity_bytes) + " B on " + arch);
    std::pair<int, int> est = dq_register_estimates(s, arch);
    int vg = est.first, ag = est.second;
    int slack = REGISTER_PRUNE_ALLOWANCE;
    if(vg > facts.arch_vgprs + slack)
        throw ckc::ValueError("arch-VGPR estimate " + std::to_string(vg) + " exceeds "
                              + std::to_string(facts.arch_vgprs) + " (+" + std::to_string(slack)
                              + ")");
    if(is_pair(s.agpr_alloc, 0, 0))
    {
        if(vg > facts.arch_vgprs)
            throw ckc::ValueError("agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate "
                                  + std::to_string(vg) + " <= " + std::to_string(facts.arch_vgprs));
    }
    else if(s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_PAIR)
    {
        int n = s.agpr_alloc.a;
        int wpe = s.waves_per_eu == NONE || s.waves_per_eu == 0 ? 1 : s.waves_per_eu;
        int per_simd = std::max(ceil_neg_div(w, CDNA_SIMDS_PER_CU), wpe);
        int share = (facts.arch_vgprs + facts.agprs) / per_simd;
        if(share - n < REG_ALLOWANCE_FIXED)
            throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(s.agpr_alloc) + " leaves "
                                  + std::to_string(share - n) + " of the " + std::to_string(share)
                                  + " registers per wave for arch VGPRs");
        if(n < ag)
            throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(s.agpr_alloc)
                                  + " below the AGPR estimate " + std::to_string(ag));
    }
    else if(ag > facts.agprs + slack)
        throw ckc::ValueError("AGPR estimate " + std::to_string(ag) + " exceeds "
                              + std::to_string(facts.agprs) + " (+" + std::to_string(slack) + ")");
    if(s.waves_per_eu == 2 && vg + ag > (facts.arch_vgprs + facts.agprs) / 2)
        throw ckc::ValueError("waves_per_eu=2 needs arch-VGPR + AGPR estimate "
                              + std::to_string(vg + ag)
                              + " <= " + std::to_string((facts.arch_vgprs + facts.agprs) / 2));
    int per_wg = ceil_neg_div(w, CDNA_SIMDS_PER_CU);
    if(per_wg > 1 && vg + ag > (facts.arch_vgprs + facts.agprs) / per_wg)
        throw ckc::ValueError("workgroup share: waves=" + std::to_string(w) + " puts "
                              + std::to_string(per_wg) + " waves on each SIMD; arch-VGPR + AGPR "
                              + "estimate " + std::to_string(vg + ag) + " exceeds "
                              + std::to_string((facts.arch_vgprs + facts.agprs) / per_wg)
                              + " per wave");
    /* DQ_SPILLING_COMBOS, DQ_RESOURCE_REFUSED and DQ_SCHED_SPILLING stay
     * Python-only (see the header). */
    return s;
}

/* ====================================================================== *
 * the builder
 * ====================================================================== */

/* ------------------------------------------------------------- plumbing */

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("attention backward dQ kernel: a helper failed");
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

void need_err(rocke_status_t st, const rocke_attn_bwd_err_t& e)
{
    if(st != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
}

IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

V ci(B* b, int64_t x)
{
    return needv(b, rocke_b_const_i32(b, x));
}

/* inline_asm("", constraints, [v], result_type=v.type, sideeffect=, hint) */
V empty_asm(B* b, V v, const char* constraints, bool sideeffect, const char* hint)
{
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", "");
    rocke_attr_set_str(b, &attrs, "constraints", constraints);
    rocke_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    rocke_attr_set_bool(b, &attrs, "convergent", false);
    const rocke_type_t* ty = v->type;
    rocke_op_t* op
        = rocke_b_op(b, ROCKE_OP_TILE_INLINE_ASM, &v, 1, &ty, 1, &attrs, NULL, 0, hint, NULL);
    need(b, op != NULL && op->num_results == 1);
    return op->results[0];
}

/* The kernargs in declaration order; a static unit multiplier is NULL. */
struct Kargs
{
    std::map<std::string, V> m;
    V operator[](const std::string& k) const
    {
        auto it = m.find(k);
        if(it == m.end())
            throw ckc::KeyError("'" + k + "'");
        return it->second;
    }
};

/* _declare_dq(b, io) */
Kargs declare_dq(B* b, const rocke_type_t* io)
{
    static const char* const readonly[]
        = {"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM", "SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV"};
    static const char* const io_ptrs[] = {"Q", "K", "V", "dO", "dQ"};
    const char* names[ROCKE_ATTN_BWD_PARAMS_MAX];
    const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX];
    int n = 0;
    rocke_attn_bwd_err_t e;
    need_err(rocke_attn_bwd_params("dq", names, formats, &n, &e), e);
    Kargs a;
    for(int i = 0; i < n; ++i)
    {
        const char* name = names[i];
        const char* f = formats[i];
        V v;
        if(eq(f, "Q"))
        {
            const rocke_type_t* pointee;
            bool is_io = false;
            for(const char* x : io_ptrs)
                is_io = is_io || eq(name, x);
            if(is_io)
                pointee = io;
            else if(strncmp(name, "WS_", 3) == 0)
                pointee = rocke_f32();
            else
                pointee = rocke_i32();
            bool ro = false;
            for(const char* x : readonly)
                ro = ro || eq(name, x);
            rocke_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            if(ro)
            {
                opts.readonly = true;
                opts.readonly_set = true;
            }
            v = rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), ro ? &opts : NULL);
        }
        else if(eq(f, "q"))
            v = rocke_b_param(b, name, rocke_i64(), NULL);
        else if(eq(f, "f"))
            v = rocke_b_param(b, name, rocke_f32(), NULL);
        else
            v = rocke_b_param(b, name, rocke_i32(), NULL);
        needv(b, v);
        a.m[name] = v;
    }
    return a;
}

/* _decode_side(b, s, a, bz, side) */
rocke_attn_bwd_seq_info_t
    decode_side(B* b, const Spec& s, const Kargs& a, V bz, bool q, int* seq_guards)
{
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = ivv(a[q ? "S_q_max" : "S_kv_max"]);
    args.seq_ptr = a[q ? "SEQ_Q" : "SEQ_KV"];
    args.has_len = ivv(a["has_len"]);
    args.len_stride = ivv(a["len_stride"]);
    rocke_attn_bwd_seq_info_t out;
    if(eq(s.seq_mode, "thd"))
    {
        args.off_ptr = a[q ? "OFF_Q" : "OFF_KV"];
        args.off64 = ivv(a["off64"]);
        args.mult = ivv(a[q ? "q_mult" : "kv_mult"]);
        args.div = ivv(a[q ? "q_div" : "kv_div"]);
        args.ws_seg = ivv(a["ws_seg"]);
        out = rocke_attn_bwd_decode_thd(b, seq_guards, bz, &args);
    }
    else
        out = rocke_attn_bwd_decode_batched(b, seq_guards, bz, &args);
    need(b, true);
    return out;
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

/* Everything _emit_dq keeps alive while it emits (stable addresses). */
struct Dq
{
    B* b;
    Spec s;
    std::string arch;
    int seq_guards = 0;
    rocke_attn_bwd_dq_geometry_t geom;
    rocke_attn_bwd_dq_lds_plan_t plan;
    rocke_attn_bwd_seq_info_t seq_q, seq_kv;
    rocke_attn_bwd_strided_tensor_addr_t q_addr, do_addr, k_addr, v_addr, dq_addr;
    rocke_attn_bwd_workspace_addr_t ws_lse2, ws_dsum;
    rocke_attn_bwd_stats_view_t stats;
    rocke_attn_bwd_mask_policy_t mask;
    rocke_attn_bwd_dq_lds_views_t lds;
    rocke_attn_bwd_dq_step_t step;
    rocke_attn_bwd_packed_rows_t rows;
};

/* pinned(slab, hint): the slab with its pointer through an empty asm. */
rocke_attn_bwd_row_slab_t pinned(B* b, const rocke_attn_bwd_row_slab_t& slab, const char* hint)
{
    V ptr = empty_asm(b, slab.ptr, "=s,0", false, hint);
    rocke_attn_bwd_row_slab_t out = slab;
    out.ptr = ptr;
    return out;
}

/* _emit_dq(spec, arch) after validation, on the initialised builder. */
rocke_kernel_def_t* emit_dq(Dq& m)
{
    B* b = m.b;
    const Spec& s = m.s;
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(m.arch);
    need(b, rocke_attn_bwd_dq_geometry_from_spec(b, &m.s, m.arch.c_str(), &m.geom));
    m.plan = dq_lds_plan(s, m.arch);
    const rocke_type_t* io = rocke_attn_bwd_ir_type(b, s.dtype);
    need(b, io != NULL);
    int km = s.block_m, kn = s.block_n;
    /* the key-loop barriers retire the K / V reads before the next tile is
     * staged over them: fenced as in the main kernel (BWD_BARRIER_FENCE) */
    Kargs a = declare_dq(b, io);
    /* Multipliers that are 1.0 under the compiled placement are not applied. */
    unsigned unit = dq_static_unit_mults(s.static_scale);
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a.m["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a.m["dq_mult"] = NULL;
    int threads = rocke_attn_bwd_dq_geometry_threads(&m.geom);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", std::max(threads, MIN_WG_ATTR));
    if(s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_PAIR)
    {
        int64_t pair[2] = {s.agpr_alloc.a, s.agpr_alloc.b};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", pair, 2);
    }
    if(s.waves_per_eu != NONE)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", s.waves_per_eu);
    if(s.scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", s.scheduler_strategy);
    need(b, true);

    V tid, lane, wave;
    rocke_attn_bwd_lane_and_wave(b, facts.wave_size, NULL, &tid, &lane, &wave);
    need(b, true);
    V zero = ci(b, 0);
    V one = ci(b, 1);
    bool group = eq(s.head_loop, "group");
    V G = group ? a["G"] : NULL;
    /* rows of one head in the CTA (CTA-uniform) */
    V rph;
    if(group)
    {
        V ckm = ci(b, km);
        rph = needv(b, rocke_b_div(b, ckm, G));
    }
    else
        rph = ci(b, km);
    V x = needv(b, rocke_b_block_id_x(b));
    V y = needv(b, rocke_b_block_id_y(b));
    V z = needv(b, rocke_b_block_id_z(b));
    if(s.xcd_remap)
    {
        V r1 = rocke_b_sub(b, rph, one);
        V r2 = rocke_b_add(b, a["S_q_max"], r1);
        V gx = rocke_b_div(b, r2, rph);
        V pos = rocke_b_cmp_gt(b, gx, zero);
        V gx1 = needv(b, rocke_b_select(b, pos, gx, one));
        V gy = group ? needv(b, rocke_b_div(b, a["h_q"], G)) : a["h_q"];
        V my = rocke_b_mul(b, gx1, y);
        V wgid = rocke_b_add(b, x, my);
        V num = needv(b, rocke_b_mul(b, gx1, gy));
        V plane = needv(b, rocke_attn_bwd_xcd_remap_id(b, wgid, num, a["xcd_n"], a["xcd_chunk"]));
        x = needv(b, rocke_b_mod(b, plane, gx1));
        y = needv(b, rocke_b_div(b, plane, gx1));
    }
    m.seq_q = decode_side(b, s, a, z, true, &m.seq_guards);
    m.seq_kv = decode_side(b, s, a, z, false, &m.seq_guards);
    V gk = rocke_b_div(b, a["h_q"], a["h_k"]);
    V gv = rocke_b_div(b, a["h_q"], a["h_v"]);
    V h = group ? needv(b, rocke_b_mul(b, y, G)) : y; /* the (first) query head of the CTA */
    V hk = rocke_b_div(b, h, gk);
    V hv = needv(b, rocke_b_div(b, h, gv));
    V q0 = needv(b, rocke_b_mul(b, x, rph));
    if(group)
    {
        m.rows.head0 = h;
        m.rows.n_heads = G;
        m.rows.rows_per_head = rph;
        m.rows.row0 = q0;
    }
    const rocke_attn_bwd_packed_rows_t* rows = group ? &m.rows : NULL;

    auto tensor = [&](const char* name,
                      const std::string& prefix,
                      const rocke_attn_bwd_seq_info_t* seq,
                      rocke_attn_bwd_strided_tensor_addr_t* out) {
        IV strides[3] = {ivv(a[prefix + "_b"]), ivv(a[prefix + "_h"]), ivv(a[prefix + "_t"])};
        need(b,
             rocke_attn_bwd_strided_tensor_addr_init(
                 out, b, a[name], io, strides, seq, s.stage_vec, s.narrow_vec));
    };
    tensor("Q", "q", &m.seq_q, &m.q_addr);
    tensor("dO", "do", &m.seq_q, &m.do_addr);
    tensor("K", "k", &m.seq_kv, &m.k_addr);
    tensor("V", "v", &m.seq_kv, &m.v_addr);
    tensor("dQ", "dq", &m.seq_q, &m.dq_addr);
    bool token_major = eq(s.ws_layout, "token_major");
    rocke_attn_bwd_ws_layout_t layout
        = token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    IV nh = ivv(a["h_q"]);
    auto ws = [&](const char* name, rocke_attn_bwd_workspace_addr_t* out) {
        need(b,
             rocke_attn_bwd_workspace_addr_init(out,
                                                b,
                                                a[name],
                                                ivv(a["ws_rows_q"]),
                                                1,
                                                m.seq_q.ws_row0,
                                                layout,
                                                token_major ? &nh : NULL));
    };
    ws("WS_LSE2", &m.ws_lse2);
    ws("WS_DSUM", &m.ws_dsum);
    need(b, rocke_attn_bwd_stats_view_init(&m.stats, b, &m.ws_lse2, &m.ws_dsum, NULL, "clamp", 0));
    if(eq(s.mask_class, "band"))
    {
        V ne = rocke_b_cmp_ne(b, a["bottom_right"], zero);
        V diff = rocke_b_sub(b, m.seq_kv.length, m.seq_q.length);
        V off = needv(b, rocke_b_select(b, ne, diff, zero));
        rocke_attn_runtime_bounds_t bounds = rocke_attn_runtime_bounds_make(
            rocke_iov_value(m.seq_q.length), rocke_iov_value(m.seq_kv.length));
        bounds.diag_offset = rocke_iov_value(off);
        bounds.left_bound = rocke_iov_value(a["left"]);
        bounds.right_bound = rocke_iov_value(a["right"]);
        if(rocke_attn_bwd_runtime_band_init(&m.mask, b, &bounds, km, kn) != ROCKE_OK)
            raise_builder(b);
    }
    else if(rocke_attn_bwd_no_mask_ktail_init(&m.mask,
                                              b,
                                              rocke_iov_value(m.seq_q.length),
                                              rocke_iov_value(m.seq_kv.length),
                                              km,
                                              kn)
            != ROCKE_OK)
        raise_builder(b);
    need(b, true);
    /* _lds_views(b, plan, io) */
    int shape[1] = {m.plan.total_bytes / 2};
    V smem = needv(b, rocke_b_smem_alloc(b, io, shape, 1, "bwd_dq_lds"));
    auto view = [&](const char* name, rocke_attn_bwd_lds_view_t* out) {
        const rocke_attn_bwd_dq_lds_buffer_t& bf = plan_buffer(m.plan, name);
        need(b,
             rocke_attn_bwd_lds_view_init(out, b, smem, bf.offset / 2, bf.pitch, bf.rows, 1, NULL));
    };
    view("q", &m.lds.q);
    view("do", &m.lds.do_);
    view("k", &m.lds.k);
    view("v", &m.lds.v);
    view("out", &m.lds.out);
    need(b,
         rocke_attn_bwd_dq_step_init(&m.step,
                                     b,
                                     &m.geom,
                                     &m.mask,
                                     &m.stats,
                                     &m.lds,
                                     tid,
                                     lane,
                                     wave,
                                     a["scale_log2"],
                                     a["ds_mult"],
                                     s.stats_prescale,
                                     true));

    /* Stage 0 (only a CTA with a row < len_q reads Q / dO). */
    V live_ = needv(b, rocke_b_cmp_lt(b, q0, m.seq_q.length)); /* CTA-uniform */
    rocke_attn_bwd_row_slab_t q_slab, do_slab;
    if(!group)
    {
        q_slab = rocke_attn_bwd_strided_tensor_addr_slab(&m.q_addr, ivv(h), ivv(q0), true);
        need(b, true);
        do_slab = rocke_attn_bwd_strided_tensor_addr_slab(&m.do_addr, ivv(h), ivv(q0), true);
        need(b, true);
    }
    need(b,
         rocke_attn_bwd_dq_step_stage_q(&m.step,
                                        b,
                                        live_,
                                        h,
                                        q0,
                                        group ? NULL : &q_slab,
                                        group ? NULL : &do_slab,
                                        &m.q_addr,
                                        &m.do_addr,
                                        rows,
                                        8));

    /* Key tiles of the rows (0 tiles: no live row, an empty band, len_kv = 0). */
    V kt_start, n_kt;
    need(b,
         rocke_attn_bwd_kv_tile_range(b, &m.mask.band, q0, ivv(rph), kn, true, &kt_start, &n_kt));
    V kt_end = needv(b, rocke_b_add(b, kt_start, n_kt));
    struct Range
    {
        V s0, e;
        bool edge;
    };
    std::vector<Range> ranges;
    if(s.edge_tiles)
    {
        V i0, i1;
        need(b,
             rocke_attn_bwd_interior_kv_range(
                 b, &m.mask.band, q0, ivv(rph), kt_start, n_kt, kn, m.mask.masked, true, &i0, &i1));
        ranges = {{kt_start, i0, true}, {i0, i1, false}, {i1, kt_end, true}};
    }
    else
        ranges = {{kt_start, kt_end, true}};
    std::vector<V> trips;
    for(const Range& r : ranges)
    {
        V d = rocke_b_sub(b, r.e, r.s0);
        trips.push_back(needv(b, rocke_b_readfirstlane(b, d)));
    }

    rocke_attn_bwd_row_slab_t k_raw = rocke_attn_bwd_strided_tensor_addr_slab(
        &m.k_addr, ivv(hk), rocke_attn_bwd_iv_int(0), true);
    need(b, true);
    rocke_attn_bwd_row_slab_t k_slab = pinned(b, k_raw, "kptr");
    rocke_attn_bwd_row_slab_t v_raw = rocke_attn_bwd_strided_tensor_addr_slab(
        &m.v_addr, ivv(hv), rocke_attn_bwd_iv_int(0), true);
    need(b, true);
    rocke_attn_bwd_row_slab_t v_slab = pinned(b, v_raw, "vptr");
    /* dQ at head h, row 0 (rows are addressed as given; the store predicates) */
    rocke_attn_bwd_row_slab_t dq_raw = rocke_attn_bwd_strided_tensor_addr_slab(
        &m.dq_addr, ivv(h), rocke_attn_bwd_iv_int(0), false);
    need(b, true);
    rocke_attn_bwd_row_slab_t dq_slab = pinned(b, dq_raw, "dqptr");
    V head_step = NULL;
    if(group)
    {
        /* the head stride (i64 elements) in bytes; fp16 / bf16 elements */
        V two = needv(b, rocke_b_const_i64(b, 2));
        head_step = needv(b, rocke_b_mul(b, a["dq_h"], two));
    }
    if(s.kv_prefetch)
    {
        /* the first tile of the range; every step prefetches the next one */
        V ckn = ci(b, kn);
        V k0 = rocke_b_mul(b, kt_start, ckn);
        V any = needv(b, rocke_b_cmp_lt(b, zero, n_kt));
        need(b, rocke_attn_bwd_dq_step_prologue(&m.step, b, &k_slab, &v_slab, k0, any));
    }
    const rocke_attn_bwd_block_gemm_t* g4t = &m.geom.g4t;
    int fm = rocke_attn_bwd_block_gemm_frags_m(g4t), fn = rocke_attn_bwd_block_gemm_frags_n(g4t);
    std::vector<V> dq((size_t)fm * fn);
    need(b, rocke_attn_bwd_dq_step_zero_dq(&m.step, b, dq.data()));
    static const char* const names[3] = {"k", "ki_", "ke_"};
    for(size_t ri = 0; ri < ranges.size(); ++ri)
    {
        const Range& r = ranges[ri];
        std::string name = names[ri];
        std::vector<std::string> arg_names;
        std::vector<rocke_iter_arg_t> ia;
        for(size_t i = 0; i < dq.size(); ++i)
            arg_names.push_back("dq_" + name + std::to_string(i));
        for(size_t i = 0; i < dq.size(); ++i)
        {
            rocke_iter_arg_t it;
            it.name = arg_names[i].c_str();
            it.init = dq[i];
            ia.push_back(it);
        }
        std::string iv = "dq_" + name + "it";
        rocke_for_t loop = rocke_b_scf_for_iter(
            b, zero, trips[ri], one, ia.data(), (int)ia.size(), iv.c_str(), false, false);
        if(loop.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, loop.body);
        {
            V it = loop.iv;
            V sum = rocke_b_add(b, r.s0, it);
            V ckn = ci(b, kn);
            V k0 = needv(b, rocke_b_mul(b, sum, ckn));
            std::vector<V> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
            std::vector<V> out(cur.size());
            need(b,
                 rocke_attn_bwd_dq_step_run_kv_step(&m.step,
                                                    b,
                                                    cur.data(),
                                                    &k_slab,
                                                    &v_slab,
                                                    k0,
                                                    r.edge,
                                                    s.kv_prefetch,
                                                    8,
                                                    out.data()));
            rocke_b_scf_yield(b, out.data(), (int)out.size());
            need(b, true);
        }
        rocke_b_region_leave(b);
        need(b, true);
        for(int i = 0; i < loop.op->num_results; ++i)
            dq[(size_t)i] = loop.op->results[i];
    }

    need(b,
         rocke_attn_bwd_dq_step_store_dq(&m.step,
                                         b,
                                         dq.data(),
                                         a["dq_mult"],
                                         &dq_slab,
                                         head_step,
                                         h,
                                         q0,
                                         rows,
                                         m.seq_q.length,
                                         a["S_q_max"],
                                         eq(s.seq_mode, "thd") ? ROCKE_ATTN_BWD_SEQ_THD
                                                               : ROCKE_ATTN_BWD_SEQ_BATCHED));
    rocke_b_ret(b);
    need(b, true);
    return rocke_ir_builder_kernel(b);
}

template <class Fn>
rocke_status_t boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    try
    {
        fn();
        return store_err(err, ROCKE_OK, "");
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc& e)
    {
        return store_err(err, ROCKE_ERR_OOM, e.what());
    }
    catch(const std::exception& e)
    {
        return store_err(err, ROCKE_ERR_VALUE, e.what());
    }
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_attn_bwd_dq_spec_t rocke_attn_bwd_dq_spec_default(void)
{
    return spec_default();
}

rocke_status_t rocke_attn_bwd_dq_spec_init(rocke_attn_bwd_dq_spec_t* spec,
                                           rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { dq_spec_init(*spec); });
}

rocke_status_t rocke_attn_bwd_dq_spec_resolved(const rocke_attn_bwd_dq_spec_t* spec,
                                               const char* arch,
                                               rocke_attn_bwd_dq_spec_t* out,
                                               rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = resolved(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_dq_spec_kernel_name(const rocke_attn_bwd_dq_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::string n = dq_kernel_name(*spec);
        if(out == NULL || n.size() >= out_cap)
            throw ckc::ValueError("kernel name does not fit the caller's buffer");
        memcpy(out, n.c_str(), n.size() + 1);
    });
}

rocke_status_t rocke_attn_bwd_spec_dq_spec(const rocke_attn_bwd_spec_t* spec,
                                           const rocke_attn_bwd_dq_spec_t* knobs,
                                           rocke_attn_bwd_dq_spec_t* out,
                                           rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_spec_of(*spec, knobs); });
}

rocke_status_t rocke_attn_bwd_dq_static_scale(const char* main_static_scale,
                                              const char** out,
                                              rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_static_scale(main_static_scale); });
}

rocke_status_t rocke_attn_bwd_dq_static_unit_mults(const char* static_scale,
                                                   unsigned* mask,
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *mask = dq_static_unit_mults(static_scale); });
}

rocke_status_t rocke_attn_bwd_dq_default_geometry(const char* arch,
                                                  int head_size,
                                                  const char* dtype,
                                                  int stage_vec,
                                                  rocke_attn_bwd_dq_spec_t* out,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(
        err, [&] { *out = default_geometry(arch ? arch : "", head_size, dtype, stage_vec); });
}

rocke_status_t rocke_attn_bwd_dq_lds_plan(const rocke_attn_bwd_dq_spec_t* spec,
                                          const char* arch,
                                          rocke_attn_bwd_dq_lds_plan_t* out,
                                          rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_lds_plan(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_dq_register_estimates(const rocke_attn_bwd_dq_spec_t* spec,
                                                    const char* arch,
                                                    int* arch_vgpr,
                                                    int* agpr,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::pair<int, int> r = dq_register_estimates(*spec, arch ? arch : "");
        *arch_vgpr = r.first;
        *agpr = r.second;
    });
}

rocke_status_t rocke_attn_bwd_validate_dq_spec(const rocke_attn_bwd_dq_spec_t* spec,
                                               const char* arch,
                                               rocke_attn_bwd_dq_spec_t* out,
                                               rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = validate_dq_spec(*spec, arch ? arch : ""); });
}

rocke_kernel_def_t* rocke_build_attn_bwd_dq_new(rocke_ir_builder_t* b,
                                                const rocke_attn_bwd_dq_spec_t* spec,
                                                const char* arch)
{
    if(b == NULL)
        return NULL;
    /* validate_attn_bwd_dq_spec raises before the Python builder exists; the
     * C builder must exist to carry the error, so a refused spec gets a
     * placeholder name. */
    std::unique_ptr<Dq> m(new Dq());
    std::string name;
    rocke_status_t code = ROCKE_OK;
    std::string msg;
    try
    {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        m->arch = arch ? arch : "";
        m->s = validate_dq_spec(*spec, m->arch);
        name = dq_kernel_name(m->s);
    }
    catch(const ckc::Error& e)
    {
        code = e.code();
        msg = e.what();
    }
    catch(const std::bad_alloc& e)
    {
        code = ROCKE_ERR_OOM;
        msg = e.what();
    }
    if(rocke_ir_builder_init(b, code == ROCKE_OK ? name.c_str() : "rocke_attn_bwd_dq_invalid")
       != ROCKE_OK)
        return NULL;
    if(code != ROCKE_OK)
    {
        rocke_i_set_err_msg(b, code, msg.c_str());
        return NULL;
    }
    m->b = b;
    m->seq_guards = 0;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return emit_dq(*m); });
}

} /* extern "C" */
