// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/attention_dense_bwd_spec.py: the dense
 * backward spec (__post_init__, kernel_name, band, strides, body_spec),
 * validate_attn_dense_bwd_spec, the not-built reasons and the kernarg ABI
 * rocke.attn_dense_bwd.v1 (rocke/instance_attention_dense_bwd_spec.h).
 *
 * A Python raise is a ckc::Error throw with the same message; the C entry
 * points catch it into a rocke_attn_bwd_err_t.
 */
#include "rocke/instance_attention_dense_bwd_spec.h"

#include <string.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"

namespace rocke_attn_bwd
{

const std::vector<std::string> DENSE_BWD_ARCHS = {"gfx942", "gfx950"};

namespace
{

typedef std::vector<std::string> Strs;
typedef std::vector<std::pair<std::string, std::string>> Items;

const int NONE = ROCKE_ATTN_BWD_NONE;
const std::vector<int> DENSE_BWD_HEAD_SIZES = {64, 128};
const Strs DENSE_LAYOUTS = {"bshd", "bhsd"};
const Strs DENSE_EPILOGUES = {"lds_staged", "lds_staged_pair", "direct"};
const Strs DENSE_OOB_MODES = {"buffer_oob", "clamp"};
const Strs DENSE_DKV_MODES = {"direct", "atomic"};
const Strs DENSE_PERSIST_DECODES = {"kv_tile_major", "hkv_major"};
const Strs DENSE_DQ_MODES = {"atomic", "split"};
const Strs BASE_BODY_VARIANTS = {"general"};
const Strs GFX942_BODY_VARIANTS = {"general", "xt"};
const Strs GFX950_BODY_VARIANTS = {"general", "trload"};
/* Largest in-slab byte offset of a buffer voffset (i32): rows up to one tile
 * past the sequence end, at the tensor's token stride. */
const long long VOFFSET_LIMIT = 1LL << 31;
const long long TILE_SLACK_ROWS = 256;

bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

std::string s_(const char* s)
{
    return s ? std::string(s) : std::string();
}

std::string repr_str(const char* s)
{
    return s == NULL ? std::string("None") : py_repr(s);
}

std::string repr_bool(bool v)
{
    return v ? "True" : "False";
}

std::string repr_opt_int(int v)
{
    return v == NONE ? std::string("None") : std::to_string(v);
}

void check_in_int(const std::string& name, int value, const std::vector<int>& allowed)
{
    if(std::find(allowed.begin(), allowed.end(), value) == allowed.end())
        throw ckc::ValueError(name + "=" + std::to_string(value) + " not in " + py_tuple(allowed));
}

/* The class's BODY_VARIANTS. */
const Strs& body_variants(const rocke_attn_bwd_dense_spec_t& s)
{
    if(eq(s.arch_class, "gfx942"))
        return GFX942_BODY_VARIANTS;
    if(eq(s.arch_class, "gfx950"))
        return GFX950_BODY_VARIANTS;
    return BASE_BODY_VARIANTS;
}

/* type(spec).__name__ */
std::string class_name(const rocke_attn_bwd_dense_spec_t& s)
{
    if(eq(s.arch_class, "gfx942"))
        return "Gfx942AttentionDenseBwdSpec";
    if(eq(s.arch_class, "gfx950"))
        return "Gfx950AttentionDenseBwdSpec";
    return "AttentionDenseBwdSpec";
}

/* spec.variant() */
const char* variant(const rocke_attn_bwd_dense_spec_t& s)
{
    return s.arch_class != NULL ? s.body_variant : "general";
}

/* fields(AttnBwdSpec) by name, in field order (keep in step with
 * rocke_attn_bwd_spec_t). */
const Strs BODY_FIELDS = {"head_size",
                          "dtype",
                          "seq_mode",
                          "dkv_mode",
                          "stage_vec",
                          "mask_class",
                          "dq_mode",
                          "static_scale",
                          "waves",
                          "block_m",
                          "block_n",
                          "block_k4",
                          "warp_grid_g4",
                          "atom_g02",
                          "atom_g13",
                          "atom_g4",
                          "pt_route",
                          "kv_residency",
                          "kt_source",
                          "transpose_source",
                          "ring_depth",
                          "global_path",
                          "lds_swizzle",
                          "sched",
                          "setprio",
                          "waves_per_eu",
                          "edge_tiles",
                          "head_pack",
                          "acc_in_lds",
                          "ws_layout",
                          "xcd_remap",
                          "agpr_alloc",
                          "s_in_agpr",
                          "agpr_route",
                          "asm_tail_nop",
                          "asm_cluster",
                          "asm_load_hoist",
                          "scheduler_strategy",
                          "warp_grid_g02",
                          "warp_grid_g13",
                          "stage_order",
                          "loop_sync",
                          "step_order",
                          "dkv_epilogue",
                          "dkv_store_vec",
                          "dkv_atomic_guard",
                          "dkv_sink",
                          "lds_order",
                          "qdo_stage",
                          "stats_load",
                          "dq_issue",
                          "g4_slicing",
                          "ptr_noalias",
                          "q_rotate",
                          "lb_interleave",
                          "dkv_slab_hoist",
                          "interior_clamp",
                          "barrier_merge",
                          "ring_issue",
                          "counted_issue",
                          "lb_mirror",
                          "bf16_cast",
                          "bwd_exp2",
                          "stats_fma",
                          "stats_vec",
                          "worklist",
                          "ws_bounds",
                          "dkv_cell_vec",
                          "narrow_vec",
                          "slab_carry",
                          "name"};

/* Fields of AttnBwdSpec the dense spec owns itself (_BODY_OWNED). */
const Strs BODY_OWNED = {"head_size",
                         "dtype",
                         "seq_mode",
                         "dkv_mode",
                         "stage_vec",
                         "mask_class",
                         "dq_mode",
                         "ws_layout",
                         "dkv_epilogue",
                         "edge_tiles",
                         "head_pack",
                         "xcd_remap",
                         "name"};

bool owned(const std::string& name)
{
    return std::find(BODY_OWNED.begin(), BODY_OWNED.end(), name) != BODY_OWNED.end();
}

/* name in DENSE_BODY_KNOBS */
bool dense_body_knob(const std::string& name)
{
    return !owned(name)
           && std::find(BODY_FIELDS.begin(), BODY_FIELDS.end(), name) != BODY_FIELDS.end();
}

/* The canonical body_knobs: (name, repr(value)) sorted by name. A dense-owned
 * field away from its default lands in *unknown (head_size and name, which
 * non_default_fields skips, are checked here). */
Items body_knob_items(const rocke_attn_bwd_spec_t& k, Strs* unknown)
{
    const rocke_attn_bwd_spec_t d = rocke_attn_bwd_spec_default();
    Items out;
    if(unknown != NULL)
    {
        if(k.head_size != d.head_size)
            unknown->push_back("head_size");
        if(!eq(k.name, d.name))
            unknown->push_back("name");
    }
    for(const auto& kv : non_default_fields(k))
    {
        if(owned(kv.first))
        {
            if(unknown != NULL)
                unknown->push_back(kv.first);
            continue;
        }
        out.push_back(kv);
    }
    std::stable_sort(
        out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    if(unknown != NULL)
        std::sort(unknown->begin(), unknown->end());
    return out;
}

/* repr() of the body_knobs tuple. */
std::string knobs_repr(const Items& items)
{
    std::string r = "(";
    for(size_t i = 0; i < items.size(); ++i)
    {
        if(i)
            r += ", ";
        r += "(" + py_repr(items[i].first) + ", " + items[i].second + ")";
    }
    if(items.size() == 1)
        r += ",";
    return r + ")";
}

void band_of(const rocke_attn_bwd_dense_spec_t& s, int* left, int* right, bool* br)
{
    *br = s.causal_bottom_right;
    if(s.window_left != NONE || s.window_right != NONE)
    {
        *left = s.window_left == NONE ? -1 : s.window_left;
        *right = s.window_right == NONE ? -1 : s.window_right;
        return;
    }
    if(!s.causal)
    {
        *left = -1;
        *right = -1;
        return;
    }
    if(s.sliding_window)
    {
        *left = s.sliding_window - 1;
        *right = 0;
        return;
    }
    *left = -1;
    *right = 0;
}

bool masked_of(const rocke_attn_bwd_dense_spec_t& s)
{
    int l, r;
    bool br;
    band_of(s, &l, &r, &br);
    return !(l == -1 && r == -1);
}

void strides_of(const rocke_attn_bwd_dense_spec_t& s, const char* tensor, long long out[3])
{
    bool q_side = eq(tensor, "q") || eq(tensor, "o") || eq(tensor, "do") || eq(tensor, "dq");
    long long heads = q_side ? s.num_query_heads : s.num_kv_heads;
    long long seq = q_side ? s.seqlen_q : s.seqlen_kv;
    dense_layout_strides(s.layout, heads, seq, s.head_size, out);
}

/* non_default_fields(self, skip=("name",)): every field with a declared
 * default (the problem shape has none) that differs from it, in field
 * order. */
Items dense_non_default_fields(const rocke_attn_bwd_dense_spec_t& s)
{
    const rocke_attn_bwd_dense_spec_t d = rocke_attn_bwd_dense_spec_default(s.arch_class);
    Items out;
    auto str = [&](const char* n, const char* v, const char* dv) {
        if(!eq(v, dv))
            out.emplace_back(n, repr_str(v));
    };
    auto flag = [&](const char* n, bool v, bool dv) {
        if(v != dv)
            out.emplace_back(n, repr_bool(v));
    };
    auto opt = [&](const char* n, int v) {
        if(v != NONE)
            out.emplace_back(n, repr_opt_int(v));
    };
    opt("batch", s.batch);
    flag("causal", s.causal, d.causal);
    flag("causal_bottom_right", s.causal_bottom_right, d.causal_bottom_right);
    str("dtype", s.dtype, d.dtype);
    str("layout", s.layout, d.layout);
    if(s.sliding_window != d.sliding_window)
        out.emplace_back("sliding_window", std::to_string(s.sliding_window));
    opt("window_left", s.window_left);
    opt("window_right", s.window_right);
    flag("ragged", s.ragged, d.ragged);
    str("dkv_mode", s.dkv_mode, d.dkv_mode);
    str("dq_mode", s.dq_mode, d.dq_mode);
    str("epilogue", s.epilogue, d.epilogue);
    str("oob", s.oob, d.oob);
    flag("edge_tiles", s.edge_tiles, d.edge_tiles);
    str("ws_layout", s.ws_layout, d.ws_layout);
    flag("persistent", s.persistent, d.persistent);
    str("persist_decode", s.persist_decode, d.persist_decode);
    flag("xcd_remap", s.xcd_remap, d.xcd_remap);
    Items knobs = body_knob_items(s.body_knobs, NULL);
    if(!knobs.empty())
        out.emplace_back("body_knobs", knobs_repr(knobs));
    if(s.arch_class != NULL)
        str("body_variant", s.body_variant, d.body_variant);
    return out;
}

std::string dense_kernel_name(const rocke_attn_bwd_dense_spec_t& s)
{
    int left, right;
    bool br;
    band_of(s, &left, &right, &br);
    std::string base = s_(s.name) + "_" + s_(s.dtype) + "_d" + std::to_string(s.head_size) + "_"
                       + s_(s.layout) + "_sq" + std::to_string(s.seqlen_q) + "_sk"
                       + std::to_string(s.seqlen_kv) + "_hq" + std::to_string(s.num_query_heads)
                       + "_hk" + std::to_string(s.num_kv_heads) + "_"
                       + dense_mask_token(left, right, br);
    if(s.ragged)
        base += "_ragged";
    if(s.batch != NONE)
        base += "_b" + std::to_string(s.batch);
    static const Strs keyed = {"batch",
                               "causal",
                               "causal_bottom_right",
                               "dtype",
                               "layout",
                               "sliding_window",
                               "window_left",
                               "window_right",
                               "ragged"};
    Items salted;
    for(const auto& kv : dense_non_default_fields(s))
        if(std::find(keyed.begin(), keyed.end(), kv.first) == keyed.end())
            salted.push_back(kv);
    return salted_kernel_name(base, salted);
}

rocke_attn_bwd_spec_t dense_validate(const rocke_attn_bwd_dense_spec_t& spec,
                                     const std::string& arch)
{
    dense_check_in("arch", arch.c_str(), DENSE_BWD_ARCHS);
    if(spec.arch_class != NULL && arch != spec.arch_class)
        throw ckc::ValueError(class_name(spec) + " is a " + spec.arch_class + " spec, not " + arch);
    /* the general main kernel's recorded resource failures are of another
     * code object: the dense kernel's own gate rows cover it */
    rocke_attn_bwd_spec_t body = validate_spec(dense_body_spec(spec), arch);
    if(!spec.ragged)
    {
        if(spec.seqlen_q % body.block_m || spec.seqlen_kv % body.block_n)
            throw ckc::ValueError("ragged=False needs seqlen_q % " + std::to_string(body.block_m)
                                  + " == 0 and seqlen_kv % " + std::to_string(body.block_n)
                                  + " == 0 (got " + std::to_string(spec.seqlen_q) + ", "
                                  + std::to_string(spec.seqlen_kv) + "); use ragged=True");
    }
    const char* tensors[2] = {"q", "k"};
    for(const char* tensor : tensors)
    {
        long long st[3];
        strides_of(spec, tensor, st);
        long long seq = eq(tensor, "q") ? spec.seqlen_q : spec.seqlen_kv;
        if((seq + TILE_SLACK_ROWS) * st[2] * 2 >= VOFFSET_LIMIT)
            throw ckc::ValueError(std::string(tensor) + " slab of " + std::to_string(seq)
                                  + " rows at token stride " + std::to_string(st[2])
                                  + " passes the 32-bit buffer offset");
    }
    if(spec.batch != NONE)
    {
        long long rows = (long long)spec.batch * std::max(spec.seqlen_q, spec.seqlen_kv);
        if(rows * spec.num_query_heads * spec.head_size >= VOFFSET_LIMIT)
            throw ckc::ValueError("batch * rows * heads * head_size passes 2^31 elements");
    }
    return body;
}

/* _not_built_all(spec, body) */
Strs not_built_all(const rocke_attn_bwd_dense_spec_t& spec, const rocke_attn_bwd_spec_t& body)
{
    Strs out;
    /* dense_bwd_not_built */
    if(!eq(variant(spec), "general"))
        out.push_back("body_variant=" + repr_str(variant(spec)));
    /* _not_built(body): every agpr_route is built */
    if(eq(body.dkv_mode, "atomic") && eq(body.dkv_sink, "last_cta"))
        out.push_back("dkv_sink=" + repr_str(body.dkv_sink));
    /* _shell_not_built(body) */
    if(body.lb_mirror)
        out.push_back("lb_mirror=True (dense shell)");
    if(!eq(body.dkv_sink, "atomic"))
        out.push_back("dkv_sink=" + repr_str(body.dkv_sink) + " (dense shell)");
    if(!eq(body.ws_bounds, "clamp"))
        out.push_back("ws_bounds=" + repr_str(body.ws_bounds) + " (dense shell)");
    if(body.dkv_cell_vec != 1)
        out.push_back("dkv_cell_vec=" + std::to_string(body.dkv_cell_vec) + " (dense shell)");
    if(body.narrow_vec != 1)
        out.push_back("narrow_vec=" + std::to_string(body.narrow_vec) + " (dense shell)");
    if(body.slab_carry)
        out.push_back("slab_carry=True (dense shell)");
    return out;
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
}

void copy_out(const std::string& s, char* out, size_t out_cap)
{
    if(out == NULL || s.size() >= out_cap)
        throw ckc::ValueError("the result does not fit the caller's buffer");
    memcpy(out, s.c_str(), s.size() + 1);
}

} // namespace

void dense_check_in(const std::string& name,
                    const char* value,
                    const std::vector<std::string>& allowed)
{
    for(const std::string& a : allowed)
        if(value != NULL && a == value)
            return;
    throw ckc::ValueError(name + "=" + repr_str(value) + " not in " + py_tuple(allowed));
}

std::string salted_kernel_name(const std::string& base,
                               std::vector<std::pair<std::string, std::string>> salted)
{
    std::stable_sort(salted.begin(), salted.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    if(salted.empty())
        return base;
    std::string r = "[";
    for(size_t i = 0; i < salted.size(); ++i)
    {
        if(i)
            r += ", ";
        r += "(" + py_repr(salted[i].first) + ", " + salted[i].second + ")";
    }
    r += "]";
    return base + "_" + sha1_hex(r).substr(0, 10);
}

std::string dense_mask_token(int left, int right, bool bottom_right)
{
    std::string diag = bottom_right ? "br" : "tl";
    if(left == -1 && right == -1)
        return "none";
    if(left == -1 && right == 0)
        return "causal" + diag;
    return "win" + std::to_string(left) + "x" + std::to_string(right) + diag;
}

void dense_layout_strides(
    const char* layout, long long heads, long long seq, long long d, long long out[3])
{
    if(eq(layout, "bshd"))
    {
        out[0] = seq * heads * d;
        out[1] = d;
        out[2] = heads * d;
        return;
    }
    if(eq(layout, "bhsd"))
    {
        out[0] = heads * seq * d;
        out[1] = seq * d;
        out[2] = d;
        return;
    }
    throw ckc::ValueError("layout=" + repr_str(layout) + " not in " + py_tuple(DENSE_LAYOUTS));
}

rocke_attn_bwd_spec_t dense_body_spec(const rocke_attn_bwd_dense_spec_t& s)
{
    rocke_attn_bwd_spec_t b = s.body_knobs;
    b.head_size = s.head_size;
    b.dtype = s.dtype;
    b.seq_mode = "batched";
    b.dkv_mode = s.dkv_mode;
    b.stage_vec = 8;
    b.mask_class = masked_of(s) ? "band" : "none";
    b.dq_mode = s.dq_mode;
    b.ws_layout = s.ws_layout;
    b.dkv_epilogue = s.epilogue;
    b.edge_tiles = s.edge_tiles ? 1 : 0;
    b.head_pack = 0;
    spec_init(b);
    return b;
}

void dense_spec_init(const rocke_attn_bwd_dense_spec_t& s)
{
    if(s.arch_class != NULL)
        dense_check_in("arch", s.arch_class, DENSE_BWD_ARCHS);
    const std::pair<const char*, int> positive[4] = {{"seqlen_q", s.seqlen_q},
                                                     {"seqlen_kv", s.seqlen_kv},
                                                     {"num_query_heads", s.num_query_heads},
                                                     {"num_kv_heads", s.num_kv_heads}};
    for(const auto& f : positive)
        if(f.second < 1)
            throw ckc::ValueError(std::string(f.first) + "=" + std::to_string(f.second)
                                  + " must be a positive int");
    check_in_int("head_size", s.head_size, DENSE_BWD_HEAD_SIZES);
    if(s.batch != NONE && s.batch < 1)
        throw ckc::ValueError("batch=" + std::to_string(s.batch)
                              + " must be None or a positive int");
    if(s.num_query_heads % s.num_kv_heads)
        throw ckc::ValueError("num_query_heads=" + std::to_string(s.num_query_heads)
                              + " is not a multiple of num_kv_heads="
                              + std::to_string(s.num_kv_heads));
    dense_check_in("dtype", s.dtype, BWD_DTYPES);
    dense_check_in("layout", s.layout, DENSE_LAYOUTS);
    dense_check_in("dkv_mode", s.dkv_mode, DENSE_DKV_MODES);
    dense_check_in("dq_mode", s.dq_mode, DENSE_DQ_MODES);
    dense_check_in("epilogue", s.epilogue, DENSE_EPILOGUES);
    dense_check_in("oob", s.oob, DENSE_OOB_MODES);
    dense_check_in("ws_layout", s.ws_layout, WS_LAYOUTS);
    dense_check_in("persist_decode", s.persist_decode, DENSE_PERSIST_DECODES);
    int sw = s.sliding_window;
    if(sw < 0)
        throw ckc::ValueError("sliding_window=" + std::to_string(sw) + " must be an int >= 0");
    if(sw && !s.causal)
        throw ckc::ValueError("sliding_window > 0 requires causal=True");
    bool explicit_band = s.window_left != NONE || s.window_right != NONE;
    if(explicit_band)
    {
        if(s.causal || sw)
            throw ckc::ValueError("window_left / window_right give the band themselves: "
                                  "causal must be False and sliding_window 0");
        const std::pair<const char*, int> wins[2]
            = {{"window_left", s.window_left}, {"window_right", s.window_right}};
        for(const auto& w : wins)
            if(w.second != NONE && w.second < -1)
                throw ckc::ValueError(std::string(w.first) + "=" + std::to_string(w.second)
                                      + " must be None or an int >= -1");
    }
    if(s.persistent && s.xcd_remap)
        throw ckc::ValueError("xcd_remap applies to the non-persistent grid only");
    if(!eq(s.persist_decode, "kv_tile_major") && !s.persistent)
        throw ckc::ValueError("persist_decode applies to persistent=True only");
    if(!name_ok(s.name))
        throw ckc::ValueError("name=" + repr_str(s.name) + " must be a non-empty identifier");
    /* _canonical_knobs: a name passed twice */
    for(int i = 0; i < s.n_body_knob_names; ++i)
        for(int j = 0; j < i; ++j)
            if(eq(s.body_knob_names[i], s.body_knob_names[j]))
                throw ckc::ValueError("body_knobs sets " + py_repr(s_(s.body_knob_names[i]))
                                      + " twice");
    Strs unknown;
    body_knob_items(s.body_knobs, &unknown);
    /* the names as passed: a dense-owned one is refused at any value */
    for(int i = 0; i < s.n_body_knob_names; ++i)
        if(!dense_body_knob(s_(s.body_knob_names[i])))
            unknown.push_back(s_(s.body_knob_names[i]));
    std::sort(unknown.begin(), unknown.end());
    unknown.erase(std::unique(unknown.begin(), unknown.end()), unknown.end());
    if(!unknown.empty())
    {
        std::string l = "[";
        for(size_t i = 0; i < unknown.size(); ++i)
            l += (i ? ", " : "") + py_repr(unknown[i]);
        throw ckc::ValueError("body_knobs " + l
                              + "] are not tuning knobs of the shared body (dense-owned or "
                                "problem fields are spec fields)");
    }
    dense_check_in("body_variant", variant(s), body_variants(s));
    /* The body spec checks every knob value. */
    dense_body_spec(s);
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_attn_bwd_dense_spec_t rocke_attn_bwd_dense_spec_default(const char* arch_class)
{
    rocke_attn_bwd_dense_spec_t s;
    memset(&s, 0, sizeof(s));
    s.arch_class = arch_class;
    s.batch = NONE;
    s.causal = true;
    s.causal_bottom_right = false;
    s.dtype = "bf16";
    s.layout = "bshd";
    s.sliding_window = 0;
    s.window_left = NONE;
    s.window_right = NONE;
    s.ragged = false;
    s.dkv_mode = "direct";
    s.dq_mode = "atomic";
    s.epilogue = "lds_staged";
    s.oob = "buffer_oob";
    s.edge_tiles = true;
    s.ws_layout = "head_major";
    s.persistent = false;
    s.persist_decode = "kv_tile_major";
    s.xcd_remap = false;
    s.body_knobs = rocke_attn_bwd_spec_default();
    s.name = "rocke_attn_dense_bwd";
    s.body_variant = arch_class != NULL ? "general" : NULL;
    return s;
}

rocke_status_t rocke_attn_bwd_dense_spec_init(const rocke_attn_bwd_dense_spec_t* spec,
                                              rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { dense_spec_init(*spec); });
}

rocke_status_t rocke_attn_bwd_dense_spec_kernel_name(const rocke_attn_bwd_dense_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap,
                                                     rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_out(dense_kernel_name(*spec), out, out_cap); });
}

void rocke_attn_bwd_dense_spec_band(const rocke_attn_bwd_dense_spec_t* spec,
                                    int* left,
                                    int* right,
                                    bool* bottom_right)
{
    band_of(*spec, left, right, bottom_right);
}

bool rocke_attn_bwd_dense_spec_masked(const rocke_attn_bwd_dense_spec_t* spec)
{
    return masked_of(*spec);
}

rocke_status_t rocke_attn_bwd_dense_layout_strides(const char* layout,
                                                   long long heads,
                                                   long long seq,
                                                   long long d,
                                                   long long out[3],
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { dense_layout_strides(layout, heads, seq, d, out); });
}

rocke_status_t rocke_attn_bwd_dense_spec_strides(const rocke_attn_bwd_dense_spec_t* spec,
                                                 const char* tensor,
                                                 long long out[3],
                                                 rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { strides_of(*spec, tensor, out); });
}

rocke_status_t rocke_attn_bwd_dense_spec_body_spec(const rocke_attn_bwd_dense_spec_t* spec,
                                                   rocke_attn_bwd_spec_t* out,
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dense_body_spec(*spec); });
}

rocke_status_t rocke_attn_bwd_dense_validate(const rocke_attn_bwd_dense_spec_t* spec,
                                             const char* arch,
                                             rocke_attn_bwd_spec_t* out,
                                             rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dense_validate(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_dense_not_built(const rocke_attn_bwd_dense_spec_t* spec,
                                              char* out,
                                              size_t out_cap,
                                              rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(spec->arch_class == NULL)
            throw ckc::ValueError("dense_bwd_not_built_reasons needs a per-arch dense spec");
        Strs r = not_built_all(*spec, dense_validate(*spec, spec->arch_class));
        std::string j;
        for(size_t i = 0; i < r.size(); ++i)
            j += (i ? ", " : "") + r[i];
        copy_out(j, out, out_cap);
    });
}

int rocke_attn_bwd_dense_params(const char* names[ROCKE_ATTN_BWD_DENSE_PARAMS],
                                const char* formats[ROCKE_ATTN_BWD_DENSE_PARAMS])
{
    static const char* const ptrs[] = {"Q",
                                       "K",
                                       "V",
                                       "O",
                                       "dO",
                                       "LSE",
                                       "dK",
                                       "dV",
                                       "WS_LSE2",
                                       "WS_DSUM",
                                       "WS_DQ",
                                       "WS_DK",
                                       "WS_DV"};
    static const char* const f32s[] = {"scale_log2", "ds_mult", "dk_mult", "dq_mult"};
    static const char* const i32s[]
        = {"batch", "g_split", "lb_order", "pair", "xcd_n", "xcd_chunk", "num_persistent"};
    int n = 0;
    for(const char* p : ptrs)
    {
        names[n] = p;
        formats[n++] = "Q";
    }
    for(const char* f : f32s)
    {
        names[n] = f;
        formats[n++] = "f";
    }
    for(const char* i : i32s)
    {
        names[n] = i;
        formats[n++] = "i";
    }
    return n;
}

} /* extern "C" */

namespace rocke_attn_bwd
{
/* For the shell (declared in its helper header). */
std::vector<std::string> dense_not_built_all(const rocke_attn_bwd_dense_spec_t& spec,
                                             const rocke_attn_bwd_spec_t& body)
{
    return not_built_all(spec, body);
}

rocke_attn_bwd_spec_t dense_validate_spec(const rocke_attn_bwd_dense_spec_t& spec,
                                          const std::string& arch)
{
    return dense_validate(spec, arch);
}

std::string dense_spec_kernel_name(const rocke_attn_bwd_dense_spec_t& spec)
{
    return dense_kernel_name(spec);
}
} // namespace rocke_attn_bwd
