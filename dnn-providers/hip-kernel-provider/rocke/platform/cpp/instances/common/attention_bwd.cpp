// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the spec, default-geometry policy and static legality of
 * library/kernels/common/attention_bwd.py (the layout-general SDPA backward
 * main kernel). Host-side only; see rocke/instance_attention_bwd.h.
 *
 * Every Python raise is one ckc::Error throw with the same message (the
 * Python spelling of the values: repr for !r, str otherwise); the C entry
 * points catch it. A Python dataclasses.replace(...) re-runs __post_init__,
 * so every replace here is a field update followed by spec_init.
 */
#include "rocke/instance_attention_bwd.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_counted.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_valu.h"
#include "rocke/ir.h"

namespace rocke_attn_bwd
{

static const int NONE = ROCKE_ATTN_BWD_NONE;

/* ====================================================================== *
 * value sets (Python constants; the ones other modules own are mirrored
 * here until their C twins exist: _attention_bwd_body STAGE_ORDERS /
 * G4_SLICINGS / COPY_ORDERS; the sched, counted and valu constants come from
 * those helpers' exports)
 * ====================================================================== */

typedef std::vector<std::string> Strs;
typedef std::vector<int> Ints;

static const Strs BWD_ARCHS = {"gfx942", "gfx950", "gfx1151", "gfx1201"};
static const Strs BWD_DTYPES = {"fp16", "bf16"};
static const Ints BWD_HEAD_SIZES = {32, 64, 128};
static const Strs SEQ_MODES = {"batched", "thd"};
static const Strs DKV_MODES = {"direct", "atomic"};
static const Ints STAGE_VECS = {8, 1};
static const Strs MASK_CLASSES = {"band", "none"};
static const Strs DQ_MODES = {"atomic", "split"};
static const Strs WS_LAYOUTS = {"head_major", "token_major"};
static const Strs STATIC_SCALES = {"fold_ds", "dq_before_atomic", "convert", "runtime"};
static const Ints BLOCK_M_VALUES = {16, 32, 64};
static const Ints BLOCK_N_VALUES = {16, 32, 64, 128, 192, 256};
static const Ints BLOCK_K4_VALUES = {16, 32, 64};
static const Ints WAVES_PER_EU_VALUES = {1, 2, 3, 4};
static const Strs ATOMS = {"16x16x16", "16x16x32", "wmma16x16x16"};
static const Strs PT_ROUTES = {"relabel", "lds"};
static const Strs RESIDENCIES = {"reg", "lds"};
static const Strs TRANSPOSE_SOURCES = {"xt_lds", "tr_read", "lds_plain", "wmma_lds"};
static const Strs GLOBAL_PATHS = {"vgpr", "dma"};
static const Strs SWIZZLES = {"xor", "pad8", "pad16", "pad32", "none"};
static const Ints DKV_STORE_VECS = {8, 4, 2, 1};
static const Strs DKV_ATOMIC_GUARDS = {"branch", "select"};
static const Strs DKV_SINKS = {"atomic", "slices", "last_cta"};
static const Strs WS_BOUNDS = {"clamp", "buffer"};
static const Ints DKV_CELL_VECS = {1, 4};
static const Ints NARROW_VECS = {1, 2, 4};
static const Strs ASM_CLUSTERS = {"k_step", "gemm"};
static const int ASM_TAIL_NOP_MAX = 15;
static const Strs STATS_LOADS = {"at_use", "step_top"};
static const Strs LOOP_SYNCS = {"full", "lds_only", "counted"};
static const Strs STEP_ORDERS = {"qtile_outer", "head_outer"};
static const Strs BARRIER_MERGES = {"off", "on"};
static const Strs RING_ISSUES = {"early", "late"};
static const Strs BARRIER_MERGE_ARCHS = {"gfx942", "gfx950"};
static const Strs BARRIER_MERGE_STAGE_ORDERS = {"g1g3g4", "g3g1g4"};
static const Strs SPLIT_DQ_ARCHS = {"gfx942", "gfx950"};
static const Strs DQ_ISSUES = {"block", "fragment", "fragment_regs"};
static const Strs DKV_EPILOGUES = {"lds_staged", "lds_staged_pair", "direct"};
static const Strs LDS_ORDERS = {"resident_low", "phase_low"};
/* _attention_bwd_body */
static const Strs STAGE_ORDERS = {"g1g3g4", "g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1"};
static const char* const DEFAULT_STAGE_ORDER = "g1g3g4";
static const Strs G4_SLICINGS = {"step", "slice", "slice_db", "slice_db_g3"};
static const char* const DEFAULT_G4_SLICING = "step";
static const Strs G4_AFTER_G3_ORDERS = {"g1g3g4", "g3g4g1"};
static const Strs COPY_ORDERS = {"split", "loads_first", "joint"};
/* _attention_bwd_sched / _counted / _valu: views of the owners' exports */
template <typename T, size_t N>
static std::vector<T> as_vec(const T (&a)[N])
{
    return std::vector<T>(a, a + N);
}
static Strs as_strs(const char* const* a, int n)
{
    return Strs(a, a + n);
}
static const Strs SCHEDS = as_strs(ROCKE_ATTN_BWD_SCHEDS, ROCKE_ATTN_BWD_N_SCHEDS);
static const Strs STAGE_TABLE_SCHEDS
    = as_strs(ROCKE_ATTN_BWD_STAGE_TABLE_SCHEDS, ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS);
static const Strs FENCED_SCHEDS
    = as_strs(ROCKE_ATTN_BWD_FENCED_SCHEDS, ROCKE_ATTN_BWD_N_FENCED_SCHEDS);
static const char* const COUNTED_LATE_DEFERRED = ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED;
static const char* const SPLIT_DEFERRED = ROCKE_ATTN_BWD_SPLIT_DEFERRED;
static const Strs COUNTED_ISSUES
    = as_strs(ROCKE_ATTN_BWD_COUNTED_ISSUES, ROCKE_ATTN_BWD_COUNTED_N_ISSUES);
static const int COUNTED_VMCNT_MAX = ROCKE_ATTN_BWD_COUNTED_VMCNT_MAX;
static const Strs BF16_CASTS = as_strs(ROCKE_ATTN_BWD_BF16_CASTS, ROCKE_ATTN_BWD_BF16_CASTS_N);
static const Strs BWD_EXP2S = as_strs(ROCKE_ATTN_BWD_BWD_EXP2S, ROCKE_ATTN_BWD_BWD_EXP2S_N);
static const Ints STATS_VECS = as_vec(ROCKE_ATTN_BWD_STATS_VECS);

static const char* const AGPR_ALLOCS_REPR
    = "(None, 'backend', (0, 0), (128, 128), (192, 192), (256, 256))";
static const char* const AGPR_ROUTES_REPR = "(None, 'vgpr_form', 'agpr_fence', 'asm_g13')";
static const char* const PTR_NOALIAS_REPR = "(False, 'workspace', 'all')";

static const char* const STAGE_ORDER_SCHED_RULE_A = "stage_order=";
static const char* const XT_SYNC_BODY
    = "transpose_source='xt_lds' needs the synchronous body: global_path='vgpr', "
      "ring_depth=1, barrier_merge='off' and loop_sync != 'counted' (the XT "
      "images are not planned per ring slot and their publishing barrier is the "
      "synchronous step's)";
static const char* const DMA_HEAD_PACK = "global_path='dma' excludes head_pack=True";
static const char* const RING_LOOP_SYNC
    = "ring_depth > 1 requires loop_sync='full' (the ring owns its barriers)";
static const char* const DMA_LOOP_SYNC = "loop_sync='lds_only' requires global_path='vgpr'";

static const int REGISTER_PRUNE_ALLOWANCE = 32;
static const int REG_ALLOWANCE_FIXED = 24;
static const int CDNA_SIMDS_PER_CU = 4;

/* ====================================================================== *
 * small helpers
 * ====================================================================== */

static std::string s_(const char* s)
{
    return s ? std::string(s) : std::string();
}

static bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

static bool in(const Strs& set, const char* v)
{
    if(v == NULL)
        return false;
    return std::find(set.begin(), set.end(), std::string(v)) != set.end();
}

static bool in(const Ints& set, int v)
{
    return std::find(set.begin(), set.end(), v) != set.end();
}

static std::string repr_str(const char* v)
{
    return v == NULL ? std::string("None") : py_repr(v);
}

static std::string repr_opt_int(int v)
{
    return v == NONE ? std::string("None") : std::to_string(v);
}

static std::string repr_bool(bool v)
{
    return v ? "True" : "False";
}

static std::string repr_opt_bool(int v)
{
    return v == NONE ? std::string("None") : repr_bool(v != 0);
}

static int floordiv(int a, int b)
{
    if(b == 0) /* Python's ZeroDivisionError; ckc has no twin, so a ValueError */
        throw ckc::ValueError("integer division or modulo by zero");
    int q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

/* _check_in(name, value, allowed) */
static void check_in(const std::string& name, const char* value, const Strs& allowed)
{
    if(!in(allowed, value))
        throw ckc::ValueError(name + "=" + repr_str(value) + " not in " + py_tuple(allowed));
}

static void check_in(const std::string& name, int value, const Ints& allowed)
{
    if(!in(allowed, value))
        throw ckc::ValueError(name + "=" + std::to_string(value) + " not in " + py_tuple(allowed));
}

/* _atom_k(atom): int(atom.rsplit("x", 1)[1]) */
static int atom_k(const char* atom)
{
    std::string a = s_(atom);
    size_t x = a.rfind('x');
    return std::stoi(a.substr(x + 1));
}

static Grid grid_of(const rocke_attn_bwd_opt_grid_t& g)
{
    return Grid{g.rows, g.cols};
}

static rocke_attn_bwd_opt_grid_t opt_grid(int rows, int cols)
{
    rocke_attn_bwd_opt_grid_t g{true, rows, cols};
    return g;
}

static bool is_pair(const rocke_attn_bwd_agpr_alloc_t& a, int x, int y)
{
    return a.kind == ROCKE_ATTN_BWD_AGPR_PAIR && a.a == x && a.b == y;
}

static bool is_backend(const rocke_attn_bwd_agpr_alloc_t& a)
{
    return a.kind == ROCKE_ATTN_BWD_AGPR_STR && eq(a.str, "backend");
}

static rocke_attn_bwd_agpr_alloc_t agpr_none()
{
    rocke_attn_bwd_agpr_alloc_t a;
    memset(&a, 0, sizeof(a));
    a.kind = ROCKE_ATTN_BWD_AGPR_NONE;
    return a;
}

static rocke_attn_bwd_agpr_alloc_t agpr_pair(int x, int y)
{
    rocke_attn_bwd_agpr_alloc_t a = agpr_none();
    a.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
    a.a = x;
    a.b = y;
    return a;
}

static bool merged_ring(const rocke_attn_bwd_spec_t& s)
{
    return eq(s.barrier_merge, "on");
}

/* Python truthiness of an Optional[bool] field (None is false). */
static bool truthy(int v)
{
    return v != NONE && v != 0;
}

/* ====================================================================== *
 * BwdArchFacts
 * ====================================================================== */

static std::string archs_repr()
{
    return py_tuple(BWD_ARCHS);
}

const rocke_attn_bwd_arch_facts_t& arch_facts(const std::string& arch)
{
    /* @cache: filled lazily under a lock, so concurrent builds are safe */
    static rocke_attn_bwd_arch_facts_t cache[4];
    static bool built[4];
    static std::mutex mu;
    auto it = std::find(BWD_ARCHS.begin(), BWD_ARCHS.end(), arch);
    if(it == BWD_ARCHS.end())
        throw ckc::ValueError("attention backward does not target " + py_repr(arch)
                              + "; known: " + archs_repr());
    int idx = (int)(it - BWD_ARCHS.begin());
    std::lock_guard<std::mutex> lock(mu);
    if(built[idx])
        return cache[idx];
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch.c_str());
    if(t == NULL)
        throw ckc::KeyError(py_repr(arch));
    bool mfma = false;
    for(int i = 0; i < t->mma.num_ops; ++i)
        if(strcmp(t->mma.ops[i].family, "mma") == 0)
            mfma = true;
    int agprs = t->limits.agprs;
    bool wide = false;
    if(mfma)
        for(const std::string& dt : BWD_DTYPES)
        {
            int n = rocke_mma_catalog_enumerate(
                &t->mma, "mma", dt.c_str(), dt.c_str(), "fp32", 16, 16, NULL, 0, NULL);
            std::vector<const rocke_mma_op_t*> ops((size_t)(n > 0 ? n : 0));
            if(n > 0)
                rocke_mma_catalog_enumerate(
                    &t->mma, "mma", dt.c_str(), dt.c_str(), "fp32", 16, 16, ops.data(), n, NULL);
            for(const rocke_mma_op_t* op : ops)
                if(op->k == 32 && op->wave_size == t->wave_size)
                    wide = true;
        }
    rocke_attn_bwd_arch_facts_t f;
    f.arch = t->gfx;
    f.wave_size = t->wave_size;
    f.matrix_path = mfma ? "mfma" : "wmma";
    f.lds_capacity_bytes = t->lds_capacity_bytes;
    f.arch_vgprs = t->limits.vgprs - agprs;
    f.agprs = agprs;
    f.has_tr_read = t->memory.has_ds_read_tr;
    f.has_lds_dma = t->memory.has_async_lds;
    f.wide_k_atom = wide;
    cache[idx] = f;
    built[idx] = true;
    return cache[idx];
}

static bool facts_mfma(const rocke_attn_bwd_arch_facts_t& f)
{
    return strcmp(f.matrix_path, "mfma") == 0;
}

static Ints legal_waves(const rocke_attn_bwd_arch_facts_t& f)
{
    return facts_mfma(f) ? Ints{1, 2, 4, 8} : Ints{1, 2, 4};
}

static Strs legal_atoms(const rocke_attn_bwd_arch_facts_t& f)
{
    if(!facts_mfma(f))
        return {"wmma16x16x16"};
    return f.wide_k_atom ? Strs{"16x16x16", "16x16x32"} : Strs{"16x16x16"};
}

static Strs legal_transpose_sources(const rocke_attn_bwd_arch_facts_t& f)
{
    if(!facts_mfma(f))
        return {"wmma_lds", "xt_lds"};
    return f.has_tr_read ? Strs{"tr_read", "xt_lds", "lds_plain"} : Strs{"xt_lds", "lds_plain"};
}

static Strs legal_global_paths(const rocke_attn_bwd_arch_facts_t& f)
{
    if(facts_mfma(f) && f.has_lds_dma && f.has_tr_read)
        return {"vgpr", "dma"};
    return {"vgpr"};
}

static Ints legal_ring_depths(const rocke_attn_bwd_arch_facts_t& f)
{
    if(!facts_mfma(f))
        return {1};
    return f.has_tr_read ? Ints{1, 2, 3} : Ints{1, 2};
}

/* ====================================================================== *
 * _canonical_swizzle
 * ====================================================================== */

static std::string strip(const std::string& s)
{
    size_t a = 0, b = s.size();
    auto ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
    };
    while(a < b && ws(s[a]))
        ++a;
    while(b > a && ws(s[b - 1]))
        --b;
    return s.substr(a, b - a);
}

/* str.isidentifier() over ASCII text (bytes >= 0x80 are taken as letters, the
 * Unicode identifier rules beyond ASCII are not ported). */
static bool is_identifier(const std::string& s)
{
    if(s.empty())
        return false;
    for(size_t i = 0; i < s.size(); ++i)
    {
        unsigned char c = (unsigned char)s[i];
        bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
        bool digit = c >= '0' && c <= '9';
        if(!(letter || (i > 0 && digit)))
            return false;
    }
    return true;
}

static std::string canonical_swizzle(const std::string& value)
{
    std::vector<std::pair<std::string, std::string>> pairs; /* insertion order */
    size_t start = 0;
    while(true)
    {
        size_t comma = value.find(',', start);
        std::string item = strip(
            value.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if(!item.empty())
        {
            size_t eqp = item.find('=');
            bool sep = eqp != std::string::npos;
            std::string name = strip(sep ? item.substr(0, eqp) : item);
            std::string val = strip(sep ? item.substr(eqp + 1) : std::string());
            if(!sep || !is_identifier(name))
                throw ckc::ValueError("lds_swizzle entry " + py_repr(item)
                                      + " is not 'buffer=value'");
            if(std::find(SWIZZLES.begin(), SWIZZLES.end(), val) == SWIZZLES.end())
                throw ckc::ValueError("lds_swizzle value " + py_repr(val) + " for " + py_repr(name)
                                      + " not in " + py_tuple(SWIZZLES));
            for(const auto& p : pairs)
                if(p.first == name)
                    throw ckc::ValueError("lds_swizzle names buffer " + py_repr(name) + " twice");
            pairs.emplace_back(name, val);
        }
        if(comma == std::string::npos)
            break;
        start = comma + 1;
    }
    std::sort(pairs.begin(), pairs.end());
    std::string out;
    for(size_t i = 0; i < pairs.size(); ++i)
    {
        if(i)
            out += ",";
        out += pairs[i].first + "=" + pairs[i].second;
    }
    return out;
}

static void set_swizzle(rocke_attn_bwd_spec_t& s, const std::string& text)
{
    if(text.size() >= ROCKE_ATTN_BWD_SWIZZLE_CAP)
        throw ckc::ValueError("lds_swizzle of " + std::to_string(text.size())
                              + " characters exceeds the C port's limit of "
                              + std::to_string(ROCKE_ATTN_BWD_SWIZZLE_CAP - 1));
    s.has_lds_swizzle = true;
    memcpy(s.lds_swizzle, text.c_str(), text.size() + 1);
}

/* ====================================================================== *
 * AttnBwdSpec.__post_init__
 * ====================================================================== */

static bool scheduler_strategy_ok(const char* v)
{
    return v == NULL || rocke_scheduler_strategy_is_valid(v);
}

/* str.isalnum() of one code point up to U+00FF: ASCII letters and digits,
 * and the Latin-1 letters / digits / numerics (Python's Unicode tables). */
static bool latin1_isalnum(unsigned c)
{
    if(c < 0x80)
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    return c == 0xaa || c == 0xb2 || c == 0xb3 || c == 0xb5 || c == 0xb9 || c == 0xba
           || (c >= 0xbc && c <= 0xbe) || (c >= 0xc0 && c <= 0xff && c != 0xd7 && c != 0xf7);
}

/* `name and name.replace("_", "").isalnum()`. A code point beyond U+00FF
 * needs the Unicode tables: refused (NotImplError) unless another character
 * already decides False. */
static bool name_ok(const char* name)
{
    if(name == NULL || !*name)
        return false;
    std::vector<unsigned> r;
    for(unsigned c : py_codepoints(name))
        if(c != '_')
            r.push_back(c);
    if(r.empty())
        return false;
    bool wide = false;
    for(unsigned c : r)
    {
        if(c > 0xff)
            wide = true;
        else if(!latin1_isalnum(c))
            return false;
    }
    if(wide)
        throw ckc::NotImplError("name: str.isalnum() beyond U+00FF is not ported");
    return true;
}

static std::string opt_grid_repr(const rocke_attn_bwd_opt_grid_t& g)
{
    if(!g.has)
        return "None";
    return grid_repr(grid_of(g));
}

void spec_init(rocke_attn_bwd_spec_t& s)
{
    check_in("head_size", s.head_size, BWD_HEAD_SIZES);
    check_in("dtype", s.dtype, BWD_DTYPES);
    check_in("seq_mode", s.seq_mode, SEQ_MODES);
    check_in("dkv_mode", s.dkv_mode, DKV_MODES);
    check_in("stage_vec", s.stage_vec, STAGE_VECS);
    check_in("mask_class", s.mask_class, MASK_CLASSES);
    check_in("dq_mode", s.dq_mode, DQ_MODES);
    check_in("ws_layout", s.ws_layout, WS_LAYOUTS);
    check_in("asm_cluster", s.asm_cluster, ASM_CLUSTERS);
    int nop = s.asm_tail_nop;
    if(nop != NONE && !(0 <= nop && nop <= ASM_TAIL_NOP_MAX))
        throw ckc::ValueError("asm_tail_nop=" + std::to_string(nop)
                              + " must be None or an int in [0, " + std::to_string(ASM_TAIL_NOP_MAX)
                              + "]");
    check_in("stage_order", s.stage_order, STAGE_ORDERS);
    check_in("loop_sync", s.loop_sync, LOOP_SYNCS);
    check_in("step_order", s.step_order, STEP_ORDERS);
    check_in("dkv_epilogue", s.dkv_epilogue, DKV_EPILOGUES);
    check_in("dkv_store_vec", s.dkv_store_vec, DKV_STORE_VECS);
    check_in("dkv_atomic_guard", s.dkv_atomic_guard, DKV_ATOMIC_GUARDS);
    check_in("dkv_sink", s.dkv_sink, DKV_SINKS);
    check_in("ws_bounds", s.ws_bounds, WS_BOUNDS);
    check_in("dkv_cell_vec", s.dkv_cell_vec, DKV_CELL_VECS);
    check_in("narrow_vec", s.narrow_vec, NARROW_VECS);
    if(s.narrow_vec != 1 && s.stage_vec != 1)
        throw ckc::ValueError("narrow_vec applies to stage_vec = 1 instances only");
    if(!eq(s.ws_bounds, "clamp") && !eq(s.ws_layout, "head_major"))
        throw ckc::ValueError("ws_bounds needs the head-major workspace (token-major rows of one "
                              "head are H floats apart)");
    check_in("lds_order", s.lds_order, LDS_ORDERS);
    if(eq(s.dkv_mode, "direct") && eq(s.dkv_epilogue, "direct") && s.dkv_store_vec != 8)
        throw ckc::ValueError("dkv_store_vec applies to the LDS-staged epilogue; "
                              "dkv_epilogue='direct' stores one element per accumulator slot");
    check_in("qdo_stage", s.qdo_stage, COPY_ORDERS);
    check_in("stats_load", s.stats_load, STATS_LOADS);
    check_in("static_scale", s.static_scale, STATIC_SCALES);
    /* SPLIT_DQ_AVAILABLE is True: no refusal of dq_mode='split' here. */
    /* the optional knobs, in Python's dict order */
    if(s.waves != NONE)
        check_in("waves", s.waves, Ints{1, 2, 4, 8});
    if(s.block_m != NONE)
        check_in("block_m", s.block_m, BLOCK_M_VALUES);
    if(s.block_n != NONE)
        check_in("block_n", s.block_n, BLOCK_N_VALUES);
    if(s.block_k4 != NONE)
        check_in("block_k4", s.block_k4, BLOCK_K4_VALUES);
    if(s.atom_g02 != NULL)
        check_in("atom_g02", s.atom_g02, ATOMS);
    if(s.atom_g13 != NULL)
        check_in("atom_g13", s.atom_g13, ATOMS);
    if(s.atom_g4 != NULL)
        check_in("atom_g4", s.atom_g4, ATOMS);
    if(s.pt_route != NULL)
        check_in("pt_route", s.pt_route, PT_ROUTES);
    if(s.kv_residency != NULL)
        check_in("kv_residency", s.kv_residency, RESIDENCIES);
    if(s.kt_source != NULL)
        check_in("kt_source", s.kt_source, RESIDENCIES);
    if(s.transpose_source != NULL)
        check_in("transpose_source", s.transpose_source, TRANSPOSE_SOURCES);
    if(s.ring_depth != NONE)
        check_in("ring_depth", s.ring_depth, Ints{1, 2, 3});
    if(s.global_path != NULL)
        check_in("global_path", s.global_path, GLOBAL_PATHS);
    if(s.sched != NULL)
        check_in("sched", s.sched, SCHEDS);
    if(s.waves_per_eu != NONE)
        check_in("waves_per_eu", s.waves_per_eu, WAVES_PER_EU_VALUES);
    /* setprio / edge_tiles / head_pack / acc_in_lds / s_in_agpr: a C int is
     * None, 0 or 1; another int is Python's non-bool */
    const std::pair<const char*, int> bools[] = {{"setprio", s.setprio},
                                                 {"edge_tiles", s.edge_tiles},
                                                 {"head_pack", s.head_pack},
                                                 {"acc_in_lds", s.acc_in_lds},
                                                 {"s_in_agpr", s.s_in_agpr}};
    for(const auto& b : bools)
        if(b.second != NONE && b.second != 0 && b.second != 1)
            throw ckc::ValueError(std::string(b.first) + " must be a bool or None, got "
                                  + std::to_string(b.second));
    check_in("dq_issue", s.dq_issue, DQ_ISSUES);
    check_in("g4_slicing", s.g4_slicing, G4_SLICINGS);
    if(!eq(s.g4_slicing, DEFAULT_G4_SLICING))
    {
        if(!eq(s.dq_issue, "block"))
            throw ckc::ValueError("g4_slicing=" + py_repr(s.g4_slicing)
                                  + " needs dq_issue='block' (the per-fragment issue reads dS "
                                    "its own way)");
        if(truthy(s.acc_in_lds))
            throw ckc::ValueError("g4_slicing=" + py_repr(s.g4_slicing)
                                  + " needs register accumulators (acc_in_lds feeds the dQ sink "
                                    "one fragment at a time)");
        if(eq(s.g4_slicing, "slice_db_g3") && !in(G4_AFTER_G3_ORDERS, s.stage_order))
            throw ckc::ValueError("g4_slicing='slice_db_g3' reads G4's first dS slice under G3: "
                                  "stage_order must be one of "
                                  + py_tuple(G4_AFTER_G3_ORDERS));
    }
    if(!(s.ptr_noalias == NULL || eq(s.ptr_noalias, "workspace") || eq(s.ptr_noalias, "all")))
        throw ckc::ValueError("ptr_noalias=" + py_repr(s.ptr_noalias) + " not in "
                              + PTR_NOALIAS_REPR);
    check_in("counted_issue", s.counted_issue, COUNTED_ISSUES);
    if(!eq(s.counted_issue, COUNTED_ISSUES[0].c_str()) && !eq(s.loop_sync, "counted"))
        throw ckc::ValueError("counted_issue applies to loop_sync='counted' only");
    check_in("bf16_cast", s.bf16_cast, BF16_CASTS);
    check_in("bwd_exp2", s.bwd_exp2, BWD_EXP2S);
    check_in("stats_vec", s.stats_vec, STATS_VECS);
    if(!eq(s.bf16_cast, "generic") && !eq(s.dtype, "bf16"))
        throw ckc::ValueError("bf16_cast=" + py_repr(s.bf16_cast)
                              + " rounds to bf16: it needs dtype='bf16'");
    if(s.stats_vec != 1 && !eq(s.ws_layout, "head_major"))
        throw ckc::ValueError("stats_vec=" + std::to_string(s.stats_vec)
                              + " needs ws_layout='head_major' (token-major stats rows are h_q "
                                "floats apart)");
    check_in("barrier_merge", s.barrier_merge, BARRIER_MERGES);
    check_in("ring_issue", s.ring_issue, RING_ISSUES);
    if(eq(s.barrier_merge, "off") && !eq(s.ring_issue, RING_ISSUES[0].c_str()))
        throw ckc::ValueError("ring_issue applies to barrier_merge='on' only");
    if(eq(s.barrier_merge, "on") && !(s.ring_depth == NONE || s.ring_depth == 2))
        throw ckc::ValueError("barrier_merge='on' needs ring_depth=2 (or None)");
    if(s.worklist && !eq(s.seq_mode, "thd"))
        throw ckc::ValueError("worklist=True needs seq_mode='thd': a batched CTA past len_kv "
                              "still stores the zero rows of its tile, so every grid CTA runs");
    if(s.worklist && s.lb_mirror)
        throw ckc::ValueError("worklist=True excludes lb_mirror: the work list fixes the CTA "
                              "order (natural), mirror pairing reorders it");
    const std::pair<const char*, rocke_attn_bwd_opt_grid_t*> grids[]
        = {{"warp_grid_g4", &s.warp_grid_g4},
           {"warp_grid_g02", &s.warp_grid_g02},
           {"warp_grid_g13", &s.warp_grid_g13}};
    for(const auto& g : grids)
        if(g.second->has && (g.second->rows < 1 || g.second->cols < 1))
            throw ckc::ValueError(std::string(g.first) + "=" + opt_grid_repr(*g.second)
                                  + " must be (rows, cols)");
    if(s.agpr_alloc.kind != ROCKE_ATTN_BWD_AGPR_NONE)
    {
        const rocke_attn_bwd_agpr_alloc_t& a = s.agpr_alloc;
        bool ok = is_backend(a) || is_pair(a, 0, 0) || is_pair(a, 128, 128) || is_pair(a, 192, 192)
                  || is_pair(a, 256, 256);
        if(!ok)
            throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(a) + " not in "
                                  + AGPR_ALLOCS_REPR);
    }
    if(s.has_lds_swizzle)
        set_swizzle(s, canonical_swizzle(s.lds_swizzle));
    if(!scheduler_strategy_ok(s.scheduler_strategy))
        throw ckc::ValueError("unsupported scheduler_strategy " + py_repr(s.scheduler_strategy)
                              + "; expected one of: iterative-ilp, iterative-maxocc, "
                                "iterative-minreg, max-ilp, max-memory-clause, or None");
    if(!name_ok(s.name))
        throw ckc::ValueError("name=" + repr_str(s.name) + " must be a non-empty identifier");
    /* couplings that need no arch */
    if(eq(s.kt_source, "reg") && eq(s.kv_residency, "lds"))
        throw ckc::ValueError("kt_source='reg' requires kv_residency='reg'");
    if(eq(s.global_path, "dma") && s.stage_vec != 8)
        throw ckc::ValueError("global_path='dma' requires stage_vec=8");
    if(eq(s.global_path, "dma") && eq(s.transpose_source, "xt_lds"))
        throw ckc::ValueError("global_path='dma' excludes transpose_source='xt_lds'");
    if(eq(s.global_path, "dma") && !eq(s.loop_sync, "full"))
        throw ckc::ValueError(DMA_LOOP_SYNC);
    if(eq(s.global_path, "dma") && truthy(s.head_pack))
        throw ckc::ValueError(DMA_HEAD_PACK);
    if(s.ring_depth != NONE && s.ring_depth > 1)
        if(!eq(s.loop_sync, "full") && !merged_ring(s))
            throw ckc::ValueError(RING_LOOP_SYNC);
    if(truthy(s.setprio) && s.sched != NULL && !in(STAGE_TABLE_SCHEDS, s.sched))
        throw ckc::ValueError("setprio=True is legal only with a stage-table sched");
    if(!eq(s.stage_order, DEFAULT_STAGE_ORDER) && in(STAGE_TABLE_SCHEDS, s.sched))
        throw ckc::ValueError(std::string(STAGE_ORDER_SCHED_RULE_A) + py_repr(s.stage_order)
                              + " is not legal with sched=" + py_repr(s.sched)
                              + ": the stage-table count model assumes the default G1, G3, G4 "
                                "order");
    if(truthy(s.s_in_agpr) && is_pair(s.agpr_alloc, 0, 0))
        throw ckc::ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)");
    if(truthy(s.acc_in_lds)
       && !(s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_NONE || is_pair(s.agpr_alloc, 0, 0)))
        throw ckc::ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)");
    if(eq(s.agpr_route, "attr"))
        s.agpr_route = NULL;
    if(s.agpr_route != NULL && !in(Strs{"vgpr_form", "agpr_fence", "asm_g13"}, s.agpr_route))
        throw ckc::ValueError("agpr_route=" + py_repr(s.agpr_route) + " not in "
                              + AGPR_ROUTES_REPR);
    if(s.agpr_route != NULL)
    {
        const rocke_attn_bwd_agpr_alloc_t& a = s.agpr_alloc;
        if(!(a.kind == ROCKE_ATTN_BWD_AGPR_PAIR && a.a > 0))
            throw ckc::ValueError("agpr_route=" + py_repr(s.agpr_route)
                                  + " needs a reservation agpr_alloc=(n, n) with n > 0, got "
                                  + agpr_alloc_repr(a));
        if(truthy(s.s_in_agpr) && !eq(s.agpr_route, "asm_g13"))
            throw ckc::ValueError("s_in_agpr=True is legal with agpr_route None or 'asm_g13', not "
                                  + py_repr(s.agpr_route) + " (it keeps S / dP in arch VGPRs)");
        if(eq(s.agpr_route, "asm_g13") && s.sched != NULL
           && (in(STAGE_TABLE_SCHEDS, s.sched) || s_(s.sched).rfind("iglp", 0) == 0))
            throw ckc::ValueError("agpr_route='asm_g13' excludes stage tables and iglp (the "
                                  "scheduler cannot see the MFMAs inside an asm block)");
    }
    if(!eq(s.agpr_route, "asm_g13"))
    {
        if(s.asm_tail_nop != NONE)
            throw ckc::ValueError("asm_tail_nop is legal with agpr_route='asm_g13' only");
        if(!eq(s.asm_cluster, "k_step"))
            throw ckc::ValueError("asm_cluster=" + py_repr(s.asm_cluster)
                                  + " is legal with agpr_route='asm_g13' only");
        if(s.asm_load_hoist)
            throw ckc::ValueError("asm_load_hoist is legal with agpr_route='asm_g13' only");
    }
    if(eq(s.atom_g13, "16x16x32") && s.block_m != NONE && s.block_m < 32)
        throw ckc::ValueError("atom_g13='16x16x32' requires block_m >= 32");
}

/* ====================================================================== *
 * default geometry policy
 * ====================================================================== */

struct Tile
{
    int km, kn, kk, w, g4r, g4c;
    const char* kv;
    const char* kt;
};

static Tile cdna_tile(const std::string& key, int d)
{
    if(key == "gfx942")
    {
        if(d == 32)
            return Tile{32, 128, 64, 4, 2, 2, "reg", "reg"};
        if(d == 64)
            return Tile{32, 128, 32, 8, 2, 4, "reg", "reg"};
        return Tile{16, 128, 32, 8, 1, 8, "reg", "reg"};
    }
    if(d == 32)
        return Tile{32, 128, 64, 4, 2, 2, "reg", "reg"};
    if(d == 64)
        return Tile{32, 256, 32, 8, 2, 4, "reg", "lds"};
    return Tile{16, 128, 32, 8, 1, 8, "reg", "lds"};
}

static const Tile NARROW_TILE = {16, 64, 32, 4, 1, 4, "reg", "reg"};
static const Tile NARROW_TILE_D32 = {32, 64, 32, 4, 2, 2, "reg", "reg"};
static const Tile DEBUG_TILE = {16, 16, 16, 1, 1, 1, "lds", "lds"};
static const char* const RDNA_D128_SCHED = "iterative-minreg";
static const char* const D128_LOW_WAVE_SCHED = RDNA_D128_SCHED;
static const char* const NARROW_D128_SCHED = RDNA_D128_SCHED;

/* The dict of attn_bwd_default_geometry, as the values of a spec's geometry
 * fields (every key of the dict is a spec field). */
struct Geometry
{
    int waves, block_m, block_n, block_k4;
    rocke_attn_bwd_opt_grid_t warp_grid_g4;
    const char *atom_g02, *atom_g13, *atom_g4, *pt_route, *kv_residency, *kt_source,
        *transpose_source;
    int ring_depth;
    const char *global_path, *sched;
    int setprio, waves_per_eu, edge_tiles, head_pack, acc_in_lds, s_in_agpr;
    const char* scheduler_strategy;
};

static Geometry default_geometry(
    const std::string& arch, int head_size, const char* dtype, int stage_vec, const char* seq_mode)
{
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    check_in("head_size", head_size, BWD_HEAD_SIZES);
    check_in("dtype", dtype, BWD_DTYPES);
    check_in("stage_vec", stage_vec, STAGE_VECS);
    check_in("seq_mode", seq_mode, SEQ_MODES);
    Geometry g;
    if(!facts_mfma(facts))
    {
        const Tile& t = DEBUG_TILE;
        g.waves = t.w;
        g.block_m = t.km;
        g.block_n = t.kn;
        g.block_k4 = t.kk;
        g.warp_grid_g4 = opt_grid(t.g4r, t.g4c);
        g.atom_g02 = g.atom_g13 = g.atom_g4 = "wmma16x16x16";
        g.pt_route = "lds";
        g.kv_residency = t.kv;
        g.kt_source = t.kt;
        g.transpose_source = "wmma_lds";
        g.ring_depth = 1;
        g.global_path = "vgpr";
        g.sched = "none";
        g.setprio = 0;
        g.waves_per_eu = NONE;
        g.edge_tiles = 0;
        g.head_pack = 0;
        g.acc_in_lds = 1;
        g.s_in_agpr = 0;
        g.scheduler_strategy = head_size == 128 ? RDNA_D128_SCHED : NULL;
        return g;
    }
    std::string key = facts.has_tr_read ? "gfx950" : "gfx942";
    Tile t = stage_vec == 8 ? cdna_tile(key, head_size)
                            : (head_size == 32 ? NARROW_TILE_D32 : NARROW_TILE);
    const char* transpose = (facts.has_tr_read && stage_vec == 8) ? "tr_read" : "lds_plain";
    bool wide = facts.wide_k_atom && t.kk % 32 == 0 && head_size % 32 == 0;
    g.waves = t.w;
    g.block_m = t.km;
    g.block_n = t.kn;
    g.block_k4 = t.kk;
    g.warp_grid_g4 = opt_grid(t.g4r, t.g4c);
    g.atom_g02 = wide ? "16x16x32" : "16x16x16";
    g.atom_g13 = "16x16x16";
    g.atom_g4 = wide ? "16x16x32" : "16x16x16";
    g.pt_route = "relabel";
    g.kv_residency = t.kv;
    g.kt_source = t.kt;
    g.transpose_source = transpose;
    g.ring_depth = 1;
    g.global_path = "vgpr";
    g.sched = "none";
    g.setprio = 0;
    g.waves_per_eu = 1;
    g.edge_tiles = 0;
    g.head_pack = 0;
    g.acc_in_lds = 0;
    g.s_in_agpr = 0;
    g.scheduler_strategy = (stage_vec == 1 && head_size == 128) ? NARROW_D128_SCHED : NULL;
    return g;
}

/* ====================================================================== *
 * register model
 * ====================================================================== */

static void require_resolved(const rocke_attn_bwd_spec_t& s)
{
    std::vector<std::string> missing;
    auto opt_i = [&](const char* n, int v) {
        if(v == NONE)
            missing.push_back(n);
    };
    auto opt_s = [&](const char* n, const char* v) {
        if(v == NULL)
            missing.push_back(n);
    };
    auto opt_g = [&](const char* n, const rocke_attn_bwd_opt_grid_t& g) {
        if(!g.has)
            missing.push_back(n);
    };
    /* field order; lds_swizzle, agpr_alloc, agpr_route, waves_per_eu,
     * scheduler_strategy and asm_tail_nop may stay None */
    opt_i("waves", s.waves);
    opt_i("block_m", s.block_m);
    opt_i("block_n", s.block_n);
    opt_i("block_k4", s.block_k4);
    opt_g("warp_grid_g4", s.warp_grid_g4);
    opt_s("atom_g02", s.atom_g02);
    opt_s("atom_g13", s.atom_g13);
    opt_s("atom_g4", s.atom_g4);
    opt_s("pt_route", s.pt_route);
    opt_s("kv_residency", s.kv_residency);
    opt_s("kt_source", s.kt_source);
    opt_s("transpose_source", s.transpose_source);
    opt_i("ring_depth", s.ring_depth);
    opt_s("global_path", s.global_path);
    opt_s("sched", s.sched);
    opt_i("setprio", s.setprio);
    opt_i("edge_tiles", s.edge_tiles);
    opt_i("head_pack", s.head_pack);
    opt_i("acc_in_lds", s.acc_in_lds);
    opt_i("s_in_agpr", s.s_in_agpr);
    opt_g("warp_grid_g02", s.warp_grid_g02);
    opt_g("warp_grid_g13", s.warp_grid_g13);
    if(!missing.empty())
    {
        std::string l = "[";
        for(size_t i = 0; i < missing.size(); ++i)
            l += (i ? ", " : "") + py_repr(missing[i]);
        throw ckc::ValueError("spec is not resolved (call .resolved(arch)): " + l + "]");
    }
}

static int ds_slice_regs(const rocke_attn_bwd_spec_t& s, int wave)
{
    int km = s.block_m;
    if(eq(s.g4_slicing, DEFAULT_G4_SLICING))
        return floordiv(2 * km * atom_k(s.atom_g4), 2 * wave);
    int kk = s.block_k4;
    int regs = floordiv(2 * km * kk, 2 * wave);
    if(eq(s.kt_source, "lds"))
        regs += floordiv(2 * s.head_size * kk, 2 * wave * s.warp_grid_g4.cols);
    return regs;
}

struct AccTerms
{
    int acc, s_dp, dq;
};

static AccTerms accumulator_terms(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    int wave = arch_facts(arch).wave_size, w = s.waves;
    int d = s.head_size, km = s.block_m, kn = s.block_n;
    int acc = truthy(s.acc_in_lds) ? 0 : 2 * floordiv(kn * d, wave * w);
    int w02 = grid_waves(grid_of(s.warp_grid_g02));
    int dq = eq(s.dq_mode, "atomic") ? floordiv(km * d, wave * w) : 0;
    return AccTerms{acc, floordiv(2 * km * kn, wave * w02), dq};
}

enum
{
    PLACE_ACC = 1,
    PLACE_S_DP = 2,
    PLACE_DQ = 4
};

static unsigned
    agpr_placement(const rocke_attn_bwd_spec_t& s, const std::string& arch, bool agpr_form)
{
    if(!facts_mfma(arch_facts(arch)) || !agpr_form)
        return 0;
    if(s.agpr_route == NULL)
        return PLACE_ACC | PLACE_S_DP | PLACE_DQ;
    if(eq(s.agpr_route, "asm_g13") && truthy(s.s_in_agpr))
        return PLACE_ACC | PLACE_S_DP;
    return PLACE_ACC;
}

static RegEstimate
    raw_estimates(const rocke_attn_bwd_spec_t& s, const std::string& arch, bool agpr_form)
{
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    int wave = facts.wave_size, w = s.waves;
    int d = s.head_size, km = s.block_m, kn = s.block_n;
    int half = 2 * wave * w;
    bool g4 = eq(s.dq_mode, "atomic");
    int w_kv = kv_operand_split(grid_of(s.warp_grid_g02), w);
    int w02 = grid_waves(grid_of(s.warp_grid_g02));
    int vgpr = REG_ALLOWANCE_FIXED;
    if(eq(s.kv_residency, "reg"))
        vgpr += 2 * floordiv(kn * d, 2 * wave * w_kv);
    if(eq(s.kt_source, "reg") && g4)
        vgpr += floordiv(kn * d, half);
    vgpr += floordiv(2 * km * d, 2 * wave);
    bool prefetch = eq(s.global_path, "vgpr") && s.ring_depth >= 2;
    if(prefetch)
        vgpr += std::max(s.ring_depth - 1, 1) * floordiv(km * d, wave * w);
    if(eq(s.loop_sync, "counted"))
        vgpr += floordiv(km * d, wave * w) + floordiv(2 * km * 16, wave);
    vgpr += floordiv(km * kn, wave * w02);
    if(g4)
        vgpr += ds_slice_regs(s, wave);
    AccTerms t = accumulator_terms(s, arch);
    unsigned placed = agpr_placement(s, arch, agpr_form);
    int agpr = 0;
    const std::pair<int, unsigned> terms[]
        = {{t.acc, PLACE_ACC}, {t.s_dp, PLACE_S_DP}, {t.dq, PLACE_DQ}};
    for(const auto& tm : terms)
    {
        if(placed & tm.second)
            agpr += tm.first;
        else
            vgpr += tm.first;
    }
    return RegEstimate{vgpr, agpr};
}

static rocke_attn_bwd_agpr_alloc_t policy_agpr_alloc(const rocke_attn_bwd_spec_t& s,
                                                     const std::string& arch)
{
    if(truthy(s.acc_in_lds))
        return agpr_pair(0, 0);
    RegEstimate e = raw_estimates(s, arch, false);
    return e.arch_vgpr <= arch_facts(arch).arch_vgprs ? agpr_pair(0, 0) : agpr_none();
}

RegEstimate register_estimates(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    require_resolved(s);
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    bool agpr_form = facts_mfma(facts) && !is_pair(s.agpr_alloc, 0, 0);
    return raw_estimates(s, arch, agpr_form);
}

struct CopyBudget
{
    int reads, writes, moves;
};

static CopyBudget agpr_copy_budget(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    require_resolved(s);
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    if(!facts_mfma(facts) || is_pair(s.agpr_alloc, 0, 0) || !truthy(s.s_in_agpr))
        return CopyBudget{0, 0, 0};
    AccTerms t = accumulator_terms(s, arch);
    bool agpr_form = !is_pair(s.agpr_alloc, 0, 0);
    unsigned placed = agpr_placement(s, arch, agpr_form);
    int reads = ((placed & PLACE_S_DP) ? t.s_dp : 0) + ((placed & PLACE_DQ) ? t.dq : 0);
    return CopyBudget{reads, 0, 0};
}

/* ====================================================================== *
 * resolved / canonical forms
 * ====================================================================== */

/* dataclasses.replace(spec, ...): the updates are applied by the caller,
 * then __post_init__ runs on the new instance. */
static rocke_attn_bwd_spec_t replaced(rocke_attn_bwd_spec_t s)
{
    spec_init(s);
    return s;
}

static rocke_attn_bwd_spec_t canonical_asm_knobs(const rocke_attn_bwd_spec_t& spec)
{
    if(!eq(spec.agpr_route, "asm_g13") || spec.atom_g13 == NULL)
        return spec;
    std::vector<std::pair<int, int>> asm_ks = {{atom_k(spec.atom_g13), spec.block_m}};
    if(truthy(spec.s_in_agpr) && spec.atom_g02 != NULL)
        asm_ks.emplace_back(atom_k(spec.atom_g02), spec.head_size);
    bool upd_nop = false, upd_cluster = false;
    /* shipped = {AGPR_ASM_TAIL_NOP.get(k)}; equal to {asm_tail_nop} */
    if(spec.asm_tail_nop != NONE)
    {
        bool all_same = true;
        for(const auto& kk : asm_ks)
            if(rocke_attn_bwd_agpr_asm_tail_nop(kk.first) != spec.asm_tail_nop)
                all_same = false;
        upd_nop = all_same;
    }
    if(eq(spec.asm_cluster, "gemm"))
    {
        bool all_one = true;
        for(const auto& kk : asm_ks)
            if(kk.second != kk.first)
                all_one = false;
        upd_cluster = all_one;
    }
    if(!upd_nop && !upd_cluster)
        return spec;
    rocke_attn_bwd_spec_t out = spec;
    if(upd_nop)
        out.asm_tail_nop = NONE;
    if(upd_cluster)
        out.asm_cluster = "k_step";
    return replaced(out);
}

static rocke_attn_bwd_spec_t canonical_g4_slicing(const rocke_attn_bwd_spec_t& spec)
{
    std::string v = s_(spec.g4_slicing);
    if(v == DEFAULT_G4_SLICING || spec.block_k4 == NONE || spec.atom_g4 == NULL)
        return spec;
    if(v == "slice_db" && spec.block_k4 == spec.block_n)
        v = "slice";
    if(v == "slice" && spec.block_k4 == atom_k(spec.atom_g4))
        v = DEFAULT_G4_SLICING;
    if(v == s_(spec.g4_slicing))
        return spec;
    rocke_attn_bwd_spec_t out = spec;
    for(const std::string& x : G4_SLICINGS)
        if(x == v)
            out.g4_slicing = G4_SLICINGS[&x - &G4_SLICINGS[0]].c_str();
    return replaced(out);
}

rocke_attn_bwd_spec_t resolved(const rocke_attn_bwd_spec_t& self, const std::string& arch)
{
    Geometry g = default_geometry(arch, self.head_size, self.dtype, self.stage_vec, self.seq_mode);
    rocke_attn_bwd_spec_t out = self;
    if(out.waves == NONE)
        out.waves = g.waves;
    if(out.block_m == NONE)
        out.block_m = g.block_m;
    if(out.block_n == NONE)
        out.block_n = g.block_n;
    if(out.block_k4 == NONE)
        out.block_k4 = g.block_k4;
    if(!out.warp_grid_g4.has)
        out.warp_grid_g4 = g.warp_grid_g4;
    if(out.atom_g02 == NULL)
        out.atom_g02 = g.atom_g02;
    if(out.atom_g13 == NULL)
        out.atom_g13 = g.atom_g13;
    if(out.atom_g4 == NULL)
        out.atom_g4 = g.atom_g4;
    if(out.pt_route == NULL)
        out.pt_route = g.pt_route;
    if(out.kv_residency == NULL)
        out.kv_residency = g.kv_residency;
    if(out.kt_source == NULL)
        out.kt_source = g.kt_source;
    if(out.transpose_source == NULL)
        out.transpose_source = g.transpose_source;
    if(out.ring_depth == NONE)
        out.ring_depth = g.ring_depth;
    if(out.global_path == NULL)
        out.global_path = g.global_path;
    if(out.sched == NULL)
        out.sched = g.sched;
    if(out.setprio == NONE)
        out.setprio = g.setprio;
    if(out.waves_per_eu == NONE)
        out.waves_per_eu = g.waves_per_eu;
    if(out.edge_tiles == NONE)
        out.edge_tiles = g.edge_tiles;
    if(out.head_pack == NONE)
        out.head_pack = g.head_pack;
    if(out.acc_in_lds == NONE)
        out.acc_in_lds = g.acc_in_lds;
    if(out.s_in_agpr == NONE)
        out.s_in_agpr = g.s_in_agpr;
    if(out.scheduler_strategy == NULL)
        out.scheduler_strategy = g.scheduler_strategy;
    if(eq(self.barrier_merge, "on") && self.ring_depth == NONE)
        out.ring_depth = 2; /* the merge runs on the 2-slot ring */
    out = replaced(out);
    bool grids = false;
    if(!out.warp_grid_g02.has)
    {
        out.warp_grid_g02 = opt_grid(1, out.waves);
        grids = true;
    }
    if(!out.warp_grid_g13.has)
    {
        out.warp_grid_g13 = opt_grid(out.waves, 1);
        grids = true;
    }
    if(grids)
        out = replaced(out);
    if(out.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_NONE && facts_mfma(arch_facts(arch)))
    {
        out.agpr_alloc = policy_agpr_alloc(out, arch);
        out = replaced(out);
    }
    if(out.scheduler_strategy == NULL && out.waves == 1 && out.head_size == 128)
    {
        out.scheduler_strategy = D128_LOW_WAVE_SCHED;
        out = replaced(out);
    }
    return canonical_g4_slicing(canonical_asm_knobs(out));
}

/* ====================================================================== *
 * kernel_name: non_default_fields + salted_kernel_name (SHA-1)
 * ====================================================================== */

/* SHA-1 (FIPS 180-4) of a byte string, lower-case hex. */
static std::string sha1_hex(const std::string& msg)
{
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::string m = msg;
    uint64_t bits = (uint64_t)msg.size() * 8u;
    m += (char)0x80;
    while(m.size() % 64 != 56)
        m += (char)0;
    for(int i = 7; i >= 0; --i)
        m += (char)((bits >> (8 * i)) & 0xff);
    auto rol = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
    for(size_t off = 0; off < m.size(); off += 64)
    {
        uint32_t w[80];
        for(int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)(unsigned char)m[off + 4 * i] << 24)
                   | ((uint32_t)(unsigned char)m[off + 4 * i + 1] << 16)
                   | ((uint32_t)(unsigned char)m[off + 4 * i + 2] << 8)
                   | (uint32_t)(unsigned char)m[off + 4 * i + 3];
        for(int i = 16; i < 80; ++i)
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for(int i = 0; i < 80; ++i)
        {
            uint32_t f, k;
            if(i < 20)
            {
                f = (b & c) | (~b & d);
                k = 0x5A827999u;
            }
            else if(i < 40)
            {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1u;
            }
            else if(i < 60)
            {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCu;
            }
            else
            {
                f = b ^ c ^ d;
                k = 0xCA62C1D6u;
            }
            uint32_t tmp = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = tmp;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    char out[41];
    for(int i = 0; i < 5; ++i)
        snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 40);
}

/* non_default_fields(spec, skip=("head_size", "name")): (name, repr(value))
 * of every field that differs from its declared default, in field order. */
static std::vector<std::pair<std::string, std::string>>
    non_default_fields(const rocke_attn_bwd_spec_t& s)
{
    const rocke_attn_bwd_spec_t d = rocke_attn_bwd_spec_default();
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
            out.emplace_back(n, repr_opt_int(v));
    };
    auto opt_bool = [&](const char* n, int v) {
        if(v != NONE)
            out.emplace_back(n, repr_opt_bool(v));
    };
    auto flag = [&](const char* n, bool v, bool dv) {
        if(v != dv)
            out.emplace_back(n, repr_bool(v));
    };
    auto grid = [&](const char* n, const rocke_attn_bwd_opt_grid_t& g) {
        if(g.has)
            out.emplace_back(n, opt_grid_repr(g));
    };
    str("dtype", s.dtype, d.dtype);
    str("seq_mode", s.seq_mode, d.seq_mode);
    str("dkv_mode", s.dkv_mode, d.dkv_mode);
    num("stage_vec", s.stage_vec, d.stage_vec);
    str("mask_class", s.mask_class, d.mask_class);
    str("dq_mode", s.dq_mode, d.dq_mode);
    str("static_scale", s.static_scale, d.static_scale);
    opt_num("waves", s.waves);
    opt_num("block_m", s.block_m);
    opt_num("block_n", s.block_n);
    opt_num("block_k4", s.block_k4);
    grid("warp_grid_g4", s.warp_grid_g4);
    str("atom_g02", s.atom_g02, NULL);
    str("atom_g13", s.atom_g13, NULL);
    str("atom_g4", s.atom_g4, NULL);
    str("pt_route", s.pt_route, NULL);
    str("kv_residency", s.kv_residency, NULL);
    str("kt_source", s.kt_source, NULL);
    str("transpose_source", s.transpose_source, NULL);
    opt_num("ring_depth", s.ring_depth);
    str("global_path", s.global_path, NULL);
    if(s.has_lds_swizzle)
        out.emplace_back("lds_swizzle", py_repr(s.lds_swizzle));
    str("sched", s.sched, NULL);
    opt_bool("setprio", s.setprio);
    opt_num("waves_per_eu", s.waves_per_eu);
    opt_bool("edge_tiles", s.edge_tiles);
    opt_bool("head_pack", s.head_pack);
    opt_bool("acc_in_lds", s.acc_in_lds);
    str("ws_layout", s.ws_layout, d.ws_layout);
    flag("xcd_remap", s.xcd_remap, d.xcd_remap);
    if(s.agpr_alloc.kind != ROCKE_ATTN_BWD_AGPR_NONE)
        out.emplace_back("agpr_alloc", agpr_alloc_repr(s.agpr_alloc));
    opt_bool("s_in_agpr", s.s_in_agpr);
    str("agpr_route", s.agpr_route, NULL);
    opt_num("asm_tail_nop", s.asm_tail_nop);
    str("asm_cluster", s.asm_cluster, d.asm_cluster);
    flag("asm_load_hoist", s.asm_load_hoist, d.asm_load_hoist);
    str("scheduler_strategy", s.scheduler_strategy, NULL);
    grid("warp_grid_g02", s.warp_grid_g02);
    grid("warp_grid_g13", s.warp_grid_g13);
    str("stage_order", s.stage_order, d.stage_order);
    str("loop_sync", s.loop_sync, d.loop_sync);
    str("step_order", s.step_order, d.step_order);
    str("dkv_epilogue", s.dkv_epilogue, d.dkv_epilogue);
    num("dkv_store_vec", s.dkv_store_vec, d.dkv_store_vec);
    str("dkv_atomic_guard", s.dkv_atomic_guard, d.dkv_atomic_guard);
    str("dkv_sink", s.dkv_sink, d.dkv_sink);
    str("lds_order", s.lds_order, d.lds_order);
    str("qdo_stage", s.qdo_stage, d.qdo_stage);
    str("stats_load", s.stats_load, d.stats_load);
    str("dq_issue", s.dq_issue, d.dq_issue);
    str("g4_slicing", s.g4_slicing, d.g4_slicing);
    if(s.ptr_noalias != NULL)
        out.emplace_back("ptr_noalias", py_repr(s.ptr_noalias));
    flag("q_rotate", s.q_rotate, d.q_rotate);
    flag("lb_interleave", s.lb_interleave, d.lb_interleave);
    flag("dkv_slab_hoist", s.dkv_slab_hoist, d.dkv_slab_hoist);
    flag("interior_clamp", s.interior_clamp, d.interior_clamp);
    str("barrier_merge", s.barrier_merge, d.barrier_merge);
    str("ring_issue", s.ring_issue, d.ring_issue);
    str("counted_issue", s.counted_issue, d.counted_issue);
    flag("lb_mirror", s.lb_mirror, d.lb_mirror);
    str("bf16_cast", s.bf16_cast, d.bf16_cast);
    str("bwd_exp2", s.bwd_exp2, d.bwd_exp2);
    flag("stats_fma", s.stats_fma, d.stats_fma);
    num("stats_vec", s.stats_vec, d.stats_vec);
    flag("worklist", s.worklist, d.worklist);
    str("ws_bounds", s.ws_bounds, d.ws_bounds);
    num("dkv_cell_vec", s.dkv_cell_vec, d.dkv_cell_vec);
    num("narrow_vec", s.narrow_vec, d.narrow_vec);
    flag("slab_carry", s.slab_carry, d.slab_carry);
    return out;
}

std::string kernel_name(const rocke_attn_bwd_spec_t& s, const std::string& stage)
{
    if(stage != "main" && stage != "dq")
        throw ckc::ValueError("stage=" + py_repr(stage) + " not in ('main', 'dq')");
    std::string base = s_(s.name) + "_" + stage + "_" + s_(s.dtype) + "_d"
                       + std::to_string(s.head_size) + "_" + s_(s.seq_mode) + "_" + s_(s.dkv_mode)
                       + "_" + s_(s.mask_class) + "_sv" + std::to_string(s.stage_vec);
    static const Strs in_base = {"dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class"};
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

/* ====================================================================== *
 * LDS
 * ====================================================================== */

static const char* planned_epilogue(const rocke_attn_bwd_spec_t& s)
{
    if(eq(s.dkv_mode, "atomic") && !eq(s.dkv_sink, "slices")
       && eq(s.dkv_epilogue, "lds_staged_pair"))
        return "lds_staged";
    return s.dkv_epilogue;
}

rocke_attn_bwd_lds_request_t lds_request(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    require_resolved(s);
    rocke_attn_bwd_lds_request_t r = rocke_attn_bwd_lds_request_default();
    r.head_size = s.head_size;
    r.k_m0 = s.block_m;
    r.k_n0 = s.block_n;
    r.waves = s.waves;
    r.wave_size = arch_facts(arch).wave_size;
    r.dtype = s.dtype;
    r.kv_residency = s.kv_residency;
    r.kt_source = s.kt_source;
    r.transpose_source = s.transpose_source;
    r.ring_depth = s.ring_depth;
    r.global_path = s.global_path;
    r.pt_route = s.pt_route;
    r.acc_in_lds = truthy(s.acc_in_lds);
    r.dq_mode = s.dq_mode;
    r.lds_swizzle = s.has_lds_swizzle ? s.lds_swizzle : NULL;
    r.dkv_epilogue = planned_epilogue(s);
    r.lds_order = s.lds_order;
    r.ds_slots = eq(s.barrier_merge, "on") ? 2 : 1;
    r.atom_k_g13 = atom_k(s.atom_g13);
    r.atom_k_g4 = atom_k(s.atom_g4);
    lds_request_check(r); /* BwdLdsRequest.__post_init__ */
    return r;
}

rocke_attn_bwd_lds_plan_t lds_plan(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    rocke_attn_bwd_lds_request_t r = lds_request(s, arch);
    return plan_bwd_lds(r, arch, false);
}

int lds_footprint_bytes(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    return lds_plan(s, arch).total_bytes;
}

/* ====================================================================== *
 * BwdTileGeometry.from_spec(...).check(), without the lane-map proofs
 * ====================================================================== */

/* ====================================================================== *
 * _attention_bwd_frag.eval_layout: the catalog lane maps evaluated on ints
 * ====================================================================== */

/* IntLaneBuilder: the C lane maps emit index math into a builder, so each
 * coordinate is built into a scratch builder and folded here. Python's
 * IntLaneBuilder offers const_i32 / add / sub / mul / div / mod / xor / shl /
 * lshr / land; any other op means the map is not plain integer math. */
static long long fold_lane_int(const rocke_value_t* v)
{
    const rocke_op_t* op = v != NULL ? v->op : NULL;
    if(op == NULL)
        throw ckc::NotImplError("eval_layout: a lane coordinate is not integer index math");
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        int64_t x = 0;
        if(!rocke_attr_get_int(&op->attrs, "value", &x))
            throw ckc::NotImplError("eval_layout: a non-integer constant in a lane map");
        return (long long)x;
    }
    if(op->num_operands != 2)
        throw ckc::NotImplError(std::string("eval_layout: ") + op->name
                                + " is not integer index math");
    long long a = fold_lane_int(op->operands[0]);
    long long c = fold_lane_int(op->operands[1]);
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        return a + c;
    case ROCKE_OP_ARITH_SUB:
        return a - c;
    case ROCKE_OP_ARITH_MUL:
        return a * c;
    case ROCKE_OP_ARITH_DIV:
    case ROCKE_OP_ARITH_MOD:
    {
        if(c == 0)
            throw ckc::ValueError("integer division or modulo by zero");
        long long q = a / c;
        if((a % c != 0) && ((a < 0) != (c < 0)))
            --q;
        return op->opcode == ROCKE_OP_ARITH_DIV ? q : a - q * c;
    }
    case ROCKE_OP_ARITH_XOR:
        return a ^ c;
    case ROCKE_OP_ARITH_SHL:
        return a << c;
    case ROCKE_OP_ARITH_LSHR:
        return a >> c;
    case ROCKE_OP_ARITH_AND:
        return a & c;
    default:
        throw ckc::NotImplError(std::string("eval_layout: ") + op->name
                                + " is not integer index math");
    }
}

typedef std::pair<long long, long long> Coord;
typedef std::vector<std::vector<Coord>> LaneCoords; /* [lane][slot] */

struct ScratchBuilder
{
    rocke_ir_builder_t b;
    bool live = false;
    ScratchBuilder()
    {
        if(rocke_ir_builder_init(&b, "rocke_attn_bwd_eval_layout") != ROCKE_OK)
            throw ckc::OOMError("eval_layout: scratch builder");
        live = true;
    }
    ~ScratchBuilder()
    {
        if(live)
            rocke_ir_builder_free(&b);
    }
    ScratchBuilder(const ScratchBuilder&) = delete;
    ScratchBuilder& operator=(const ScratchBuilder&) = delete;
};

/* _eval(op, "c"): op.c_layout() for every lane and slot, cached per op like
 * Python's _eval_cached (the catalog ops live in static storage). A deque, so
 * the references handed out stay valid when another thread appends. */
static const LaneCoords& eval_c_layout(const rocke_mma_op_t* op)
{
    static std::mutex mu;
    static std::deque<std::pair<const rocke_mma_op_t*, LaneCoords>> cache;
    std::lock_guard<std::mutex> lock(mu);
    for(const auto& e : cache)
        if(e.first == op)
            return e.second;
    const rocke_layout_map_t* m = op->c_layout;
    if(m == NULL || m->fn == NULL)
        throw ckc::NotImplError(std::string("no verified 'c' layout map for MMA op_id '")
                                + op->op_id + "' (" + std::to_string(op->m) + "x"
                                + std::to_string(op->n) + "x" + std::to_string(op->k)
                                + "); add one to _MMA_FRAGMENT_INFO before consuming it");
    LaneCoords lanes((size_t)m->wave_size);
    for(int lane = 0; lane < m->wave_size; ++lane)
    {
        /* one scratch builder per lane keeps the arena small */
        ScratchBuilder sb;
        for(int s = 0; s < m->frag_len; ++s)
        {
            rocke_value_t* lv = rocke_b_const_i32(&sb.b, lane);
            rocke_value_t* c0 = NULL;
            rocke_value_t* c1 = NULL;
            if(!rocke_layout_map_coord(m, &sb.b, lv, s, &c0, &c1) || !rocke_ir_builder_ok(&sb.b))
                throw ckc::NotImplError(std::string("eval_layout: ")
                                        + rocke_ir_builder_error(&sb.b));
            lanes[(size_t)lane].push_back(Coord{fold_lane_int(c0), fold_lane_int(c1)});
        }
    }
    cache.emplace_back(op, std::move(lanes));
    return cache.back().second;
}

/* repr of one lane's coordinate tuple */
static std::string coords_repr(const std::vector<Coord>& coords)
{
    std::string out = "(";
    for(size_t i = 0; i < coords.size(); ++i)
    {
        if(i)
            out += ", ";
        out += "(" + std::to_string(coords[i].first) + ", " + std::to_string(coords[i].second)
               + ")";
    }
    if(coords.size() == 1)
        out += ",";
    return out + ")";
}

/* check_stats_vec_layout(op, width) */
void check_stats_vec_layout(const rocke_mma_op_t* op, int width)
{
    if(!in(STATS_VECS, width))
        throw ckc::ValueError("stats_vec=" + std::to_string(width) + " not in "
                              + py_tuple(STATS_VECS));
    const LaneCoords& lanes = eval_c_layout(op);
    size_t n = lanes.empty() ? 0 : lanes[0].size();
    if(n % (size_t)width)
        throw ckc::ValueError("stats_vec=" + std::to_string(width) + ": the C fragment of "
                              + op->op_id + " has " + std::to_string(n) + " slots");
    for(const std::vector<Coord>& coords : lanes)
        for(size_t g = 0; g < n; g += (size_t)width)
        {
            Coord c0 = coords[g];
            for(int j = 0; j < width; ++j)
                if(coords[g + (size_t)j] != Coord{c0.first + j, c0.second})
                    throw ckc::ValueError("stats_vec=" + std::to_string(width) + ": the C rows of "
                                          + op->op_id + " are not runs of " + std::to_string(width)
                                          + " consecutive rows (a lane holds " + coords_repr(coords)
                                          + ")");
        }
}

bool stats_vec_layout_ok(const rocke_mma_op_t* op, int width)
{
    try
    {
        check_stats_vec_layout(op, width);
        return true;
    }
    catch(const ckc::ValueError&)
    {
        return false;
    }
}

/* c_row_runs(op, run) */
static bool c_row_runs(const rocke_mma_op_t* op, int run)
{
    if(op->c_frag_len % run)
        return false;
    for(const std::vector<Coord>& coords : eval_c_layout(op))
    {
        Coord c0 = coords[0];
        for(size_t s = 0; s < coords.size(); ++s)
            if(coords[s] != Coord{c0.first + (long long)s, c0.second})
                return false;
    }
    return true;
}

/* c_quad_transposable(op) */
bool c_quad_transposable(const rocke_mma_op_t* op)
{
    if(op->c_frag_len != 4 || !c_row_runs(op, 4) || op->wave_size % 4)
        return false;
    const LaneCoords& ev = eval_c_layout(op);
    for(int q = 0; q < op->wave_size; q += 4)
    {
        Coord c0 = ev[(size_t)q][0];
        if(c0.second % 4)
            return false;
        for(int j = 0; j < 4; ++j)
            if(ev[(size_t)(q + j)][0] != Coord{c0.first, c0.second + j})
                return false;
    }
    return true;
}

struct BlockGemm
{
    int m, n, k;
    Grid grid;
    int atom_k;
    const rocke_mma_op_t* op;
};

/* BlockGemm(arch, dtype, m, n, k, grid, atom_k).__post_init__ */
static BlockGemm
    block_gemm(const std::string& arch, const char* dtype, int m, int n, int k, Grid grid, int ak)
{
    int rows = grid.rows, cols = grid.cols;
    int waves = rows * cols;
    check_waves(arch, waves);
    if(!(grid == Grid{1, waves} || grid == Grid{waves, 1} || grid == Grid{2, floordiv(waves, 2)}))
        throw ckc::ValueError("warp grid " + grid_repr(grid)
                              + " is not one of (1, W), (W, 1), (2, W/2)");
    const rocke_mma_op_t* op = select_atom(arch, s_(dtype), ak);
    if(m % (op->m * rows) || n % (op->n * cols))
        throw ckc::ValueError("block " + std::to_string(m) + "x" + std::to_string(n)
                              + " does not split into " + std::to_string(op->m) + "x"
                              + std::to_string(op->n) + " atoms over the warp grid "
                              + grid_repr(grid));
    if(k <= 0 || k % ak)
        throw ckc::ValueError("K=" + std::to_string(k) + " is not a multiple of the atom K "
                              + std::to_string(ak));
    return BlockGemm{m, n, k, grid, ak, op};
}

struct Geom
{
    BlockGemm g0, g1, g4;
};

static Geom geometry_check(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    if(s.waves == NONE || s.block_m == NONE)
        throw ckc::ValueError("BwdTileGeometry needs a resolved spec");
    const char* epilogue = s.dkv_epilogue;
    if(eq(s.dkv_mode, "atomic") && !eq(epilogue, "direct"))
        epilogue = "lds_staged";
    int w = s.waves;
    Grid g0g = grid_of(s.warp_grid_g02), g1g = grid_of(s.warp_grid_g13),
         g4g = grid_of(s.warp_grid_g4);
    /* check() */
    static const Strs readers = {"lds_plain", "wmma_lds", "tr_read", "xt_lds"};
    if(!in(readers, s.transpose_source))
        throw ckc::ValueError("transpose_source=" + repr_str(s.transpose_source)
                              + " has no operand reader in this body (built: " + py_tuple(readers)
                              + ")");
    if(!(g0g == Grid{1, w}) || !(g1g == Grid{w, 1}))
        if(eq(s.pt_route, "relabel"))
            throw ckc::ValueError("the relabel route needs G0 grid (1, W), G1 (W, 1)");
    int w0 = g0g.rows * g0g.cols;
    if(w0 > w || w % w0)
        throw ckc::ValueError("G0 grid " + grid_repr(g0g) + " must hold W=" + std::to_string(w)
                              + " waves or a divisor of W (a replicated grid)");
    const std::pair<const char*, Grid> gs[] = {{"G1", g1g}, {"G4", g4g}};
    for(const auto& g : gs)
        if(g.second.rows * g.second.cols != w)
            throw ckc::ValueError(std::string(g.first) + " grid " + grid_repr(g.second)
                                  + " must hold W=" + std::to_string(w) + " waves");
    /* BlockGemm constructions (atom_facts holds for every backward atom) */
    BlockGemm g0
        = block_gemm(arch, s.dtype, s.block_m, s.block_n, s.head_size, g0g, atom_k(s.atom_g02));
    BlockGemm g1
        = block_gemm(arch, s.dtype, s.block_n, s.head_size, s.block_m, g1g, atom_k(s.atom_g13));
    BlockGemm g4
        = block_gemm(arch, s.dtype, s.block_m, s.head_size, s.block_n, g4g, atom_k(s.atom_g4));
    /* relabel plan / K permutation: proved for the natural grids (not ported) */
    std::string reader = eq(s.transpose_source, "tr_read")  ? "tr16"
                         : eq(s.transpose_source, "xt_lds") ? "xt"
                                                            : "plain";
    if(reader == "tr16")
        check_transpose_read(arch, 64); /* the tr16 read plans: not ported */
    if(!eq(s.dq_mode, "atomic") && !eq(s.dq_mode, "split"))
        throw ckc::ValueError("dq_mode=" + repr_str(s.dq_mode) + " not in ('atomic', 'split')");
    if(!in(STAGE_ORDERS, s.stage_order))
        throw ckc::ValueError("stage_order=" + repr_str(s.stage_order) + " not in "
                              + py_tuple(STAGE_ORDERS));
    if(!eq(s.stage_order, DEFAULT_STAGE_ORDER) && in(STAGE_TABLE_SCHEDS, s.sched))
        throw ckc::ValueError("stage_order=" + py_repr(s.stage_order)
                              + " with sched=" + py_repr(s.sched)
                              + ": the stage-table count model assumes the default G1, G3, G4 "
                                "order");
    if(!in(DKV_EPILOGUES, epilogue))
        throw ckc::ValueError("dkv_epilogue must be one of " + py_tuple(DKV_EPILOGUES));
    if(!in(DKV_STORE_VECS, s.dkv_store_vec))
        throw ckc::ValueError("dkv_store_vec must be 8, 4, 2 or 1");
    if(truthy(s.acc_in_lds) && eq(epilogue, "lds_staged_pair"))
        throw ckc::ValueError("acc_in_lds has no dK / dV staging buffer to pair");
    if(!in(DQ_ISSUES, s.dq_issue))
        throw ckc::ValueError("dq_issue=" + repr_str(s.dq_issue) + " is not built in this body");
    if(s.asm_load_hoist && (!eq(s.agpr_route, "asm_g13") || truthy(s.acc_in_lds)))
        throw ckc::ValueError("asm_load_hoist needs the asm_g13 route with register accumulators");
    if(!eq(s.bf16_cast, "generic") && !eq(s.dtype, "bf16"))
        throw ckc::ValueError("bf16_cast=" + py_repr(s.bf16_cast) + " needs dtype='bf16'");
    if(s.stats_vec != 1)
        check_stats_vec_layout(g0.op, s.stats_vec);
    if(!eq(s.barrier_merge, "off") && !eq(s.barrier_merge, "on"))
        throw ckc::ValueError("barrier_merge=" + repr_str(s.barrier_merge) + " is not built");
    if(!eq(s.ring_issue, "early") && !eq(s.ring_issue, "late"))
        throw ckc::ValueError("ring_issue=" + repr_str(s.ring_issue) + " is not built");
    if(eq(s.barrier_merge, "on")
       && (s.ring_depth != 2 || !eq(s.global_path, "vgpr") || !eq(s.pt_route, "relabel")
           || truthy(s.acc_in_lds) || s_(s.stage_order).substr(4, 2) != "g4"))
        throw ckc::ValueError("barrier_merge='on' needs ring_depth=2, global_path='vgpr', the "
                              "relabel route, register dK / dV and G4 last in stage_order");
    /* _check_counted */
    if(!in(COUNTED_ISSUES, s.counted_issue))
        throw ckc::ValueError("counted_issue must be one of " + py_tuple(COUNTED_ISSUES));
    if(!eq(s.loop_sync, "counted"))
    {
        if(!eq(s.counted_issue, "early"))
            throw ckc::ValueError("counted_issue applies to loop_sync='counted' only");
    }
    else
    {
        if(strcmp(arch_caps(arch).matrix_path, "mfma") != 0)
            throw ckc::ValueError("loop_sync='counted' is CDNA-only (RDNA counts no-return "
                                  "atomics in vscnt, not vmcnt)");
        if(!(eq(s.global_path, "vgpr") && s.ring_depth == 1))
            throw ckc::ValueError("loop_sync='counted' owns the next-step loads: it needs "
                                  "global_path='vgpr' and ring_depth=1");
        if(eq(s.sched, "stage_table_atomics") && eq(s.counted_issue, "late"))
            throw ckc::ValueError(COUNTED_LATE_DEFERRED);
        if(!(eq(s.qdo_stage, "split") && eq(s.stats_load, "at_use")))
            throw ckc::ValueError("loop_sync='counted' owns the Q / dO and stats issue order: it "
                                  "needs qdo_stage='split' and stats_load='at_use'");
        int frags_m = (g4.m / g4.grid.rows) / g4.op->m;
        int frags_n = (g4.n / g4.grid.cols) / g4.op->n;
        int n_atomics = frags_m * frags_n * g4.op->c_frag_len;
        if(!(0 <= n_atomics && n_atomics <= COUNTED_VMCNT_MAX))
            throw ckc::ValueError(std::to_string(n_atomics)
                                  + " dQ atomics per wave and step exceed the vmcnt field ("
                                  + std::to_string(COUNTED_VMCNT_MAX) + ")");
    }
    /* _check_sched */
    if(!(eq(s.sched, "none") && !truthy(s.setprio)))
        if(!eq(s.dq_mode, "atomic") && eq(s.sched, "stage_table_atomics"))
            throw ckc::ValueError(SPLIT_DEFERRED);
    /* _check_g4_slicing */
    if(!in(G4_SLICINGS, s.g4_slicing))
        throw ckc::ValueError("g4_slicing=" + repr_str(s.g4_slicing) + " not in "
                              + py_tuple(G4_SLICINGS));
    if(!eq(s.g4_slicing, DEFAULT_G4_SLICING))
    {
        std::string why;
        int k4 = atom_k(s.atom_g4);
        if(truthy(s.acc_in_lds) || !arch_caps(arch).has_agprs)
            why = "MFMA register accumulators (acc_in_lds=False)";
        else if(!eq(s.dq_issue, "block"))
            why = "dq_issue='block' (the per-fragment issue reads dS its own way)";
        else if(s.block_k4 % k4 || s.block_n % s.block_k4)
            why = "kK4 a multiple of the G4 atom K that divides kN0";
        else if(eq(s.g4_slicing, "slice_db_g3") && !in(G4_AFTER_G3_ORDERS, s.stage_order))
            why = "a stage_order with G3 right before G4 " + py_tuple(G4_AFTER_G3_ORDERS);
        else if(!eq(s.dq_mode, "atomic"))
            why = "dq_mode='atomic' (the split-dQ main kernel has no G4)";
        else if(!eq(s.barrier_merge, "off") || s.ring_depth != 1 || !eq(s.global_path, "vgpr")
                || eq(s.loop_sync, "counted"))
            why = "the synchronous body: barrier_merge='off', ring_depth=1, "
                  "global_path='vgpr' and loop_sync != 'counted' (the slice reads "
                  "and the moved dS barrier are built for its three barriers)";
        else if(!eq(s.sched, "none") || truthy(s.setprio))
            why = "sched='none' (the stage table does not model the dS key slices)";
        else if(reader == "xt" && eq(s.kt_source, "lds"))
            why = "K^T from registers under xt_lds (no K-inner K^T image slices)";
        if(!why.empty())
            throw ckc::ValueError("g4_slicing=" + py_repr(s.g4_slicing) + " needs " + why);
    }
    return Geom{g0, g1, g4};
}

/* ====================================================================== *
 * check_counted_sync / validate_attn_bwd_spec
 * ====================================================================== */

static void check_counted_sync(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    if(!eq(s.loop_sync, "counted"))
        return;
    if(!(eq(s.barrier_merge, "off") || s.barrier_merge == NULL))
        throw ckc::ValueError("loop_sync='counted' excludes barrier_merge (the merged ring owns "
                              "the next-step loads and its one barrier precedes the dQ atomics)");
    if(!facts_mfma(arch_facts(arch)))
        throw ckc::ValueError("loop_sync='counted' is CDNA-only; " + arch
                              + " counts no-return atomics in vscnt, not vmcnt");
    if(eq(s.step_order, "head_outer") && truthy(s.edge_tiles))
        throw ckc::ValueError("loop_sync='counted' excludes the grouped head-outer form "
                              "(step_order='head_outer' with edge_tiles): its three range loops "
                              "nested in the head-group loop, each with a carried next-step tile, "
                              "exceed the register file");
    if(!eq(s.dq_mode, "atomic"))
        throw ckc::ValueError("loop_sync='counted' needs dq_mode='atomic': its waits keep the "
                              "step's dQ atomics in flight, and the split-dQ main kernel issues "
                              "none");
    geometry_check(s, arch);
}

/* LAYOUT_SPILLING_COMBOS: (arch, head_size, dtype, block_m, block_n, waves,
 * seq_mode, mask_class, edge_tiles, head_pack, dkv_mode, stage_vec,
 * transpose_source, lds_swizzle) */
struct SpillCombo
{
    const char* arch;
    int d;
    const char* dtype;
    int km, kn, w;
    const char *seq_mode, *mask_class;
    bool edge_tiles, head_pack;
    const char* dkv_mode;
    int stage_vec;
    const char* transpose_source;
    const char* lds_swizzle; /* NULL: None */
};

static const char* const ROW_XOR_ALL
    = "dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,"
      "v_stage=xor";

static const SpillCombo LAYOUT_SPILLING_COMBOS[] = {
    {"gfx1151",
     64,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     false,
     false,
     "direct",
     8,
     "xt_lds",
     NULL},
    {"gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", true, false, "direct", 8, "xt_lds", NULL},
    {"gfx1151",
     64,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     false,
     false,
     "direct",
     1,
     "xt_lds",
     NULL},
    {"gfx1151", 64, "fp16", 16, 16, 1, "thd", "band", false, false, "atomic", 8, "xt_lds", NULL},
    {"gfx1151",
     64,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     true,
     false,
     "direct",
     8,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx1151",
     64,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     false,
     true,
     "direct",
     8,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx1201",
     64,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     false,
     true,
     "direct",
     8,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx1201",
     128,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     true,
     false,
     "direct",
     8,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx1201",
     128,
     "fp16",
     16,
     16,
     1,
     "batched",
     "band",
     false,
     true,
     "direct",
     8,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx1201",
     128,
     "fp16",
     16,
     16,
     1,
     "thd",
     "none",
     false,
     true,
     "atomic",
     1,
     "wmma_lds",
     ROW_XOR_ALL},
    {"gfx942",
     128,
     "fp16",
     16,
     128,
     8,
     "batched",
     "band",
     true,
     false,
     "direct",
     8,
     "lds_plain",
     ROW_XOR_ALL},
    {"gfx950",
     64,
     "fp16",
     32,
     256,
     8,
     "batched",
     "band",
     false,
     true,
     "direct",
     8,
     "tr_read",
     ROW_XOR_ALL},
};

static bool layout_spills(const rocke_attn_bwd_spec_t& s, const std::string& arch)
{
    for(const SpillCombo& c : LAYOUT_SPILLING_COMBOS)
    {
        bool sw = c.lds_swizzle == NULL
                      ? !s.has_lds_swizzle
                      : (s.has_lds_swizzle && strcmp(c.lds_swizzle, s.lds_swizzle) == 0);
        /* Python compares bools with ints: edge_tiles / head_pack are 0 / 1 */
        if(arch == c.arch && s.head_size == c.d && eq(s.dtype, c.dtype) && s.block_m == c.km
           && s.block_n == c.kn && s.waves == c.w && eq(s.seq_mode, c.seq_mode)
           && eq(s.mask_class, c.mask_class) && s.edge_tiles == (int)c.edge_tiles
           && s.head_pack == (int)c.head_pack && eq(s.dkv_mode, c.dkv_mode)
           && s.stage_vec == c.stage_vec && eq(s.transpose_source, c.transpose_source) && sw)
            return true;
    }
    return false;
}

rocke_attn_bwd_spec_t validate_spec(const rocke_attn_bwd_spec_t& spec, const std::string& arch)
{
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(arch);
    rocke_attn_bwd_spec_t s = resolved(spec, arch);
    int w = s.waves, km = s.block_m, kn = s.block_n, kk = s.block_k4, d = s.head_size;
    bool mfma = facts_mfma(facts);
    if(eq(s.barrier_merge, "on") && !in(BARRIER_MERGE_ARCHS, arch.c_str()))
        throw ckc::ValueError("barrier_merge='on' is built for " + py_tuple(BARRIER_MERGE_ARCHS));
    /* catalog */
    check_in("waves on " + arch, w, legal_waves(facts));
    check_in("atom_g02 on " + arch, s.atom_g02, legal_atoms(facts));
    check_in("atom_g13 on " + arch, s.atom_g13, legal_atoms(facts));
    check_in("atom_g4 on " + arch, s.atom_g4, legal_atoms(facts));
    check_in("transpose_source on " + arch, s.transpose_source, legal_transpose_sources(facts));
    check_in("global_path on " + arch, s.global_path, legal_global_paths(facts));
    check_in("ring_depth on " + arch, s.ring_depth, legal_ring_depths(facts));
    if(eq(s.dq_mode, "split") && !in(SPLIT_DQ_ARCHS, arch.c_str()))
        throw ckc::ValueError("dq_mode='split' on " + arch + ": the split dQ kernel is built for "
                              + py_tuple(SPLIT_DQ_ARCHS)
                              + " only (its dS^T hand-off is the MFMA C-to-A relabel)");
    /* geometry */
    Grid g02 = grid_of(s.warp_grid_g02), g13 = grid_of(s.warp_grid_g13);
    bool natural = g02 == Grid{1, w} && g13 == Grid{w, 1};
    if((natural || in(STAGE_TABLE_SCHEDS, s.sched)) && kn % (16 * w) != 0)
        throw ckc::ValueError("block_n=" + std::to_string(kn)
                              + " must be a multiple of 16*waves=" + std::to_string(16 * w));
    if(in(STAGE_TABLE_SCHEDS, s.sched) && grid_waves(g02) != w)
        throw ckc::ValueError("sched=" + py_repr(s.sched)
                              + ": the stage-table count model needs a G0/G2 grid of every wave, "
                                "not the replicated "
                              + grid_repr(g02));
    if(w == 1 && !(km == 16 && kn == 16))
        throw ckc::ValueError("waves=1 is legal only for the 16 x 16 debug / fallback tile");
    int k4a = atom_k(s.atom_g4);
    if(kk % k4a != 0)
        throw ckc::ValueError("block_k4=" + std::to_string(kk)
                              + " must be a multiple of the G4 atom K (" + std::to_string(k4a)
                              + ")");
    if(kn % kk != 0)
        throw ckc::ValueError("block_n=" + std::to_string(kn)
                              + " must be a multiple of block_k4=" + std::to_string(kk));
    if(d % atom_k(s.atom_g02) != 0)
        throw ckc::ValueError("head_size=" + std::to_string(d)
                              + " must be a multiple of the G0/G2 atom K");
    if(eq(s.atom_g13, "16x16x32") && km < 32)
        throw ckc::ValueError("atom_g13='16x16x32' requires block_m >= 32");
    int g4r = s.warp_grid_g4.rows, g4c = s.warp_grid_g4.cols;
    Grid g4 = grid_of(s.warp_grid_g4);
    if(!(g4 == Grid{1, w} || g4 == Grid{2, floordiv(w, 2)}) || g4r * g4c != w)
        throw ckc::ValueError("warp_grid_g4=" + grid_repr(g4) + " must be (1, " + std::to_string(w)
                              + ") or (2, " + std::to_string(floordiv(w, 2)) + ")");
    if(d % (16 * g4c) != 0 || km % (16 * g4r) != 0)
        throw ckc::ValueError("warp_grid_g4=" + grid_repr(g4) + " does not divide D="
                              + std::to_string(d) + " / block_m=" + std::to_string(km));
    if(km % 16 != 0)
        throw ckc::ValueError("block_m=" + std::to_string(km) + " must be a multiple of 16");
    struct GridRule
    {
        const char* gname;
        Grid g, nat;
        int rows, cols;
    };
    const GridRule rules[]
        = {{"warp_grid_g02", g02, Grid{1, w}, km, kn}, {"warp_grid_g13", g13, Grid{w, 1}, kn, d}};
    for(const GridRule& r : rules)
    {
        if(r.g == r.nat)
        {
            if(!natural && (r.rows % (16 * r.g.rows) || r.cols % (16 * r.g.cols)))
                throw ckc::ValueError(std::string(r.gname) + "=" + grid_repr(r.g)
                                      + " does not divide its " + std::to_string(r.rows) + " x "
                                      + std::to_string(r.cols) + " tile");
            continue;
        }
        check_lds_route_grid(r.gname, r.g, r.nat, w, r.rows, r.cols, s_(s.pt_route));
    }
    /* couplings */
    if(eq(s.ptr_noalias, "all") && !eq(s.dkv_mode, "atomic"))
        throw ckc::ValueError("ptr_noalias='all' needs dkv_mode='atomic' (direct mode stores dK / "
                              "dV, which the request cannot prove disjoint from the inputs)");
    if(eq(s.kt_source, "reg") && !eq(s.kv_residency, "reg"))
        throw ckc::ValueError("kt_source='reg' requires kv_residency='reg'");
    if(!mfma && eq(s.kv_residency, "reg"))
        throw ckc::ValueError("kv_residency='reg' is not legal on WMMA targets");
    if(!mfma && eq(s.pt_route, "relabel"))
        throw ckc::ValueError("pt_route='relabel' is not legal on WMMA targets (C is not A^T)");
    if(eq(s.global_path, "dma") && (s.stage_vec != 8 || eq(s.transpose_source, "xt_lds")))
        throw ckc::ValueError("global_path='dma' needs stage_vec=8 and excludes xt_lds");
    if(!eq(s.loop_sync, "full") && !eq(s.global_path, "vgpr"))
        throw ckc::ValueError(DMA_LOOP_SYNC);
    if(!eq(s.loop_sync, "full") && s.ring_depth > 1 && !merged_ring(s))
        throw ckc::ValueError(RING_LOOP_SYNC);
    if(eq(s.global_path, "dma") && truthy(s.head_pack))
        throw ckc::ValueError(DMA_HEAD_PACK);
    if(eq(s.global_path, "dma") && (km * s.head_size * 2) % 1024)
        throw ckc::ValueError("global_path='dma' copies 1 KiB pieces: block_m * head_size * 2 = "
                              + std::to_string(km * s.head_size * 2)
                              + " is not a multiple of 1024");
    check_counted_sync(s, arch);
    if(eq(s.global_path, "dma") && eq(s.transpose_source, "tr_read") && s.has_lds_swizzle
       && s.lds_swizzle[0])
    {
        std::string text(s.lds_swizzle);
        size_t start = 0;
        while(true)
        {
            size_t comma = text.find(',', start);
            std::string item = text.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            size_t eqp = item.find('=');
            std::string buf = eqp == std::string::npos ? item : item.substr(0, eqp);
            std::string val = eqp == std::string::npos ? std::string() : item.substr(eqp + 1);
            if((buf == "q" || buf == "do") && val != "xor")
                throw ckc::ValueError("lds_swizzle " + buf + "=" + val
                                      + ": DMA buffers read by tr_read need 'xor'");
            if(comma == std::string::npos)
                break;
            start = comma + 1;
        }
    }
    if(s.dkv_cell_vec != 1)
    {
        Geom g = geometry_check(s, arch);
        if(!c_quad_transposable(g.g1.op))
            throw ckc::ValueError("dkv_cell_vec=" + std::to_string(s.dkv_cell_vec)
                                  + " needs a G1 / G3 C layout whose lane quads hold 4 "
                                    "consecutive columns of 4 rows (not on "
                                  + arch + ")");
    }
    if(s.slab_carry)
    {
        if(s.q_rotate)
            throw ckc::ValueError("slab_carry excludes q_rotate (a rotated range wraps to its "
                                  "start: no constant per-step delta)");
        if(eq(s.global_path, "dma") || s.ring_depth > 1)
            throw ckc::ValueError("slab_carry is built for the synchronous step: the prefetch ring "
                                  "forms the slabs of the steps ahead from their (q tile, head) "
                                  "index (not built with a ring)");
        if(eq(s.loop_sync, "counted"))
            throw ckc::ValueError("slab_carry is built for the synchronous step: the counted waits "
                                  "load the next step from its index (not built with "
                                  "loop_sync='counted')");
    }
    if(truthy(s.acc_in_lds) && eq(s.dkv_mode, "direct") && !eq(s.dkv_epilogue, "lds_staged"))
        throw ckc::ValueError("dkv_epilogue=" + py_repr(s.dkv_epilogue)
                              + " needs register accumulators; with acc_in_lds the epilogue "
                                "reads the fp32 LDS images ('lds_staged')");
    bool staged = !(eq(s.global_path, "vgpr") && s.ring_depth == 1);
    if(staged && (!eq(s.qdo_stage, "split") || !eq(s.stats_load, "at_use")))
        throw ckc::ValueError("qdo_stage / stats_load reorder the synchronous VGPR staging; with "
                              "global_path='dma' or ring_depth > 1 the ring owns the issue order");
    if(eq(s.barrier_merge, "on"))
    {
        if(s.ring_depth != 2 || !eq(s.global_path, "vgpr"))
            throw ckc::ValueError("barrier_merge='on' needs ring_depth=2, global_path='vgpr'");
        if(!eq(s.pt_route, "relabel") || truthy(s.acc_in_lds))
            throw ckc::ValueError(
                "barrier_merge='on' needs pt_route='relabel' and register dK / dV");
        if(!in(BARRIER_MERGE_STAGE_ORDERS, s.stage_order))
            throw ckc::ValueError("barrier_merge='on' needs G4 last (stage_order in "
                                  + py_tuple(BARRIER_MERGE_STAGE_ORDERS)
                                  + "): G1 / G3 read this step's Q / dO slot, which the next step "
                                    "overwrites after the merged barrier");
        if(!eq(s.dq_mode, "atomic"))
            throw ckc::ValueError("barrier_merge='on' needs dq_mode='atomic': the merged barrier "
                                  "and the next step's slot stores sit before G4, which the split "
                                  "dQ main kernel does not issue (its steps would have no "
                                  "barrier)");
    }
    if(eq(s.step_order, "head_outer") && truthy(s.edge_tiles) && s.ring_depth > 1)
        throw ckc::ValueError("ring_depth=" + std::to_string(s.ring_depth)
                              + " (the prefetch / DMA ring or the merged barrier) excludes the "
                                "grouped head-outer form (step_order='head_outer' with "
                                "edge_tiles): its three range loops nested in the head-group loop, "
                                "each carrying the ring's next-step tile, spill SGPRs on the "
                                "production backend on both CDNA targets (the flat head-outer loop "
                                "and edge tiles alone take the ring)");
    if(eq(s.qdo_stage, "joint"))
    {
        int chunks = floordiv(km * d, 8);
        int threads = w * facts.wave_size;
        if(chunks >= threads || chunks % facts.wave_size)
            throw ckc::ValueError("qdo_stage='joint' needs block_m * D / 8 = "
                                  + std::to_string(chunks) + " below the " + std::to_string(threads)
                                  + " threads and a multiple of the wave size (otherwise it is the "
                                    "'split' distribution)");
    }
    if(s_(s.sched).rfind("iglp", 0) == 0 && truthy(s.setprio))
        throw ckc::ValueError("setprio is legal only inside a stage table");
    if(truthy(s.setprio) && !in(STAGE_TABLE_SCHEDS, s.sched))
        throw ckc::ValueError("setprio is legal only inside a stage table");
    if(!eq(s.stage_order, DEFAULT_STAGE_ORDER) && in(STAGE_TABLE_SCHEDS, s.sched))
        throw ckc::ValueError(std::string(STAGE_ORDER_SCHED_RULE_A) + py_repr(s.stage_order)
                              + " is not legal with sched=" + py_repr(s.sched)
                              + ": the stage-table count model assumes the default G1, G3, G4 "
                                "order");
    if(eq(s.sched, "stage_table_atomics") && !eq(s.dq_mode, "atomic"))
        throw ckc::ValueError(SPLIT_DEFERRED);
    if(eq(s.transpose_source, "xt_lds"))
    {
        if(staged || !eq(s.barrier_merge, "off") || eq(s.loop_sync, "counted"))
            throw ckc::ValueError(XT_SYNC_BODY);
        if(in(FENCED_SCHEDS, s.sched))
            throw ckc::ValueError("transpose_source='xt_lds' with sched=" + py_repr(s.sched)
                                  + ": the fenced read-ahead of dO^T would read the XT image "
                                    "before the barrier that publishes it");
    }
    if(!mfma)
    {
        if(!truthy(s.acc_in_lds))
            throw ckc::ValueError("acc_in_lds=False is not legal on WMMA targets");
        if(s.agpr_alloc.kind != ROCKE_ATTN_BWD_AGPR_NONE || truthy(s.s_in_agpr)
           || s.agpr_route != NULL)
            throw ckc::ValueError("agpr_alloc / s_in_agpr / agpr_route apply to MFMA targets only");
        if(!(s.waves_per_eu == NONE || s.waves_per_eu == 1 || s.waves_per_eu == 2))
            throw ckc::ValueError("waves_per_eu on WMMA targets must be None, 1 or 2");
    }
    else
    {
        if(truthy(s.acc_in_lds) && !is_pair(s.agpr_alloc, 0, 0))
            throw ckc::ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)");
        if(truthy(s.s_in_agpr) && is_pair(s.agpr_alloc, 0, 0))
            throw ckc::ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)");
        if(eq(s.agpr_route, "asm_g13") && s.asm_tail_nop != NONE)
        {
            check_asm_tail_nop(arch, s_(s.dtype), atom_k(s.atom_g13), s.asm_tail_nop);
            if(truthy(s.s_in_agpr))
                check_asm_tail_nop(arch, s_(s.dtype), atom_k(s.atom_g02), s.asm_tail_nop);
        }
    }
    if(!eq(spec.g4_slicing, DEFAULT_G4_SLICING) || !eq(s.g4_slicing, DEFAULT_G4_SLICING))
    {
        if(!mfma || truthy(s.acc_in_lds))
            throw ckc::ValueError("g4_slicing=" + py_repr(spec.g4_slicing)
                                  + " needs MFMA register accumulators (acc_in_lds=False); WMMA / "
                                    "LDS-accumulator G4 feeds the dQ sink one fragment at a time");
        if(!eq(s.dq_issue, "block"))
            throw ckc::ValueError("g4_slicing=" + py_repr(spec.g4_slicing)
                                  + " needs dq_issue='block'");
        if(staged || !eq(s.barrier_merge, "off") || eq(s.loop_sync, "counted")
           || !eq(s.dq_mode, "atomic") || !eq(s.sched, "none") || truthy(s.setprio)
           || (eq(s.transpose_source, "xt_lds") && eq(s.kt_source, "lds")))
            throw ckc::ValueError("g4_slicing=" + py_repr(spec.g4_slicing)
                                  + " needs the synchronous body (global_path='vgpr', "
                                    "ring_depth=1, barrier_merge='off', loop_sync != 'counted'), "
                                    "dq_mode='atomic', sched='none' without setprio, and "
                                    "kt_source='reg' under xt_lds");
    }
    /* LDS: the phase planner decides */
    rocke_attn_bwd_lds_plan_t plan = lds_plan(s, arch);
    if(plan.total_bytes > plan.capacity_bytes)
        throw ckc::ValueError("LDS plan " + std::to_string(plan.total_bytes) + " B exceeds "
                              + std::to_string(plan.capacity_bytes) + " B on " + arch);
    std::vector<std::string> slotted_xor;
    for(int i = 0; i < plan.n_buffers; ++i)
    {
        const rocke_attn_bwd_lds_buffer_t& b = plan.buffers[i];
        if(b.slot && strcmp(b.layout, "rowmajor:xor") == 0
           && std::find(slotted_xor.begin(), slotted_xor.end(), b.name) == slotted_xor.end())
            slotted_xor.push_back(b.name);
    }
    std::sort(slotted_xor.begin(), slotted_xor.end());
    if(!slotted_xor.empty())
    {
        std::string names;
        for(size_t i = 0; i < slotted_xor.size(); ++i)
            names += (i ? "," : "") + slotted_xor[i];
        throw ckc::ValueError("lds_swizzle " + names
                              + "=xor on a ring-slotted row-major buffer: the XOR view has no "
                                "run-time slot shift (pad / none, or the DMA slab layout, "
                                "instead)");
    }
    if(layout_spills(s, arch))
        throw ckc::ValueError("transpose_source=" + py_repr(s.transpose_source) + " / lds_swizzle="
                              + (s.has_lds_swizzle ? py_repr(s.lds_swizzle) : std::string("None"))
                              + " on this " + std::to_string(s.block_m) + " x "
                              + std::to_string(s.block_n) + " x " + std::to_string(s.waves)
                              + "-wave d" + std::to_string(s.head_size)
                              + " tile spills arch VGPRs on " + arch + " (LAYOUT_SPILLING_COMBOS)");
    /* registers */
    RegEstimate est = register_estimates(s, arch);
    int vg = est.arch_vgpr, ag = est.agpr;
    int slack = REGISTER_PRUNE_ALLOWANCE;
    if(mfma)
    {
        if(vg > facts.arch_vgprs + slack)
            throw ckc::ValueError("arch-VGPR estimate " + std::to_string(vg) + " exceeds "
                                  + std::to_string(facts.arch_vgprs) + " (+" + std::to_string(slack)
                                  + ")");
        if(is_pair(s.agpr_alloc, 0, 0))
        {
            if(vg > facts.arch_vgprs)
                throw ckc::ValueError("agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate "
                                      + std::to_string(vg) + " <= 256");
        }
        else if(!(s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_NONE || is_backend(s.agpr_alloc)))
        {
            int n = s.agpr_alloc.a;
            int wpe = s.waves_per_eu == NONE ? 1 : s.waves_per_eu; /* `or 1` */
            if(wpe == 0)
                wpe = 1;
            int per_simd = std::max(-floordiv(-w, CDNA_SIMDS_PER_CU), wpe);
            int share = floordiv(facts.arch_vgprs + facts.agprs, per_simd);
            if(share - n < REG_ALLOWANCE_FIXED)
                throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(s.agpr_alloc) + " leaves "
                                      + std::to_string(share - n) + " of the "
                                      + std::to_string(share) + " registers per wave ("
                                      + std::to_string(per_simd) + " waves per SIMD at waves="
                                      + std::to_string(w) + ") for arch VGPRs; at least "
                                      + std::to_string(REG_ALLOWANCE_FIXED) + " are needed");
            if(n < ag)
                throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(s.agpr_alloc)
                                      + " below the AGPR estimate " + std::to_string(ag));
            if(ag > std::min(facts.agprs, n) + slack)
                throw ckc::ValueError("AGPR estimate " + std::to_string(ag) + " exceeds "
                                      + std::to_string(std::min(facts.agprs, n)) + " (+"
                                      + std::to_string(slack) + ")");
        }
        else if(ag > facts.agprs + slack)
            throw ckc::ValueError("AGPR estimate " + std::to_string(ag) + " exceeds "
                                  + std::to_string(facts.agprs) + " (+" + std::to_string(slack)
                                  + ")");
        int wpe = s.waves_per_eu;
        if(wpe != NONE && wpe > 1 && vg + ag > floordiv(facts.arch_vgprs + facts.agprs, wpe))
            throw ckc::ValueError("waves_per_eu=" + std::to_string(wpe)
                                  + " needs arch-VGPR + AGPR estimate " + std::to_string(vg + ag)
                                  + " <= "
                                  + std::to_string(floordiv(facts.arch_vgprs + facts.agprs, wpe)));
        int per_wg = -floordiv(-w, CDNA_SIMDS_PER_CU);
        int wg_share = floordiv(facts.arch_vgprs + facts.agprs, per_wg);
        if(per_wg > 1 && vg + ag > wg_share)
            throw ckc::ValueError(
                "workgroup share: waves=" + std::to_string(w) + " puts " + std::to_string(per_wg)
                + " waves on each SIMD; arch-VGPR + AGPR estimate " + std::to_string(vg + ag)
                + " exceeds " + std::to_string(wg_share) + " per wave");
    }
    else
    {
        if(vg > facts.arch_vgprs + slack)
            throw ckc::ValueError("register estimate " + std::to_string(vg) + " exceeds "
                                  + std::to_string(facts.arch_vgprs) + " (+" + std::to_string(slack)
                                  + ")");
        if(s.waves_per_eu == 2 && vg > floordiv(facts.arch_vgprs, 2))
            throw ckc::ValueError("waves_per_eu=2 needs the register estimate <= half the file");
    }
    if(s.stats_vec != 1)
    {
        if(!mfma)
            throw ckc::ValueError("stats_vec=" + std::to_string(s.stats_vec)
                                  + " is built for MFMA targets only");
        Geom g = geometry_check(s, arch);
        check_stats_vec_layout(g.g0.op, s.stats_vec);
    }
    /* resource_refusal(s, arch): Python-only (main_kernel_refusals=False) */
    return s;
}

/* ====================================================================== *
 * ABI rocke.attn_bwd.v3
 * ====================================================================== */

static Strs split_words(const char* text)
{
    Strs out;
    std::string cur;
    for(const char* p = text;; ++p)
    {
        if(*p == ' ' || *p == '\0')
        {
            if(!cur.empty())
                out.push_back(cur);
            cur.clear();
            if(*p == '\0')
                break;
        }
        else
            cur += *p;
    }
    return out;
}

struct AbiLists
{
    const char *ptrs, *i64s, *f32s, *i32s;
};

static bool abi_lists(const std::string& stage, AbiLists* out)
{
    static const char* const main_ptrs
        = "Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST";
    static const char* const main_i64 = "q_b q_h k_b k_h v_b v_h do_b do_h dk_b dk_h dv_b dv_h";
    static const char* const main_f32 = "scale_log2 ds_mult dk_mult dq_mult";
    static const char* const main_i32
        = "q_t k_t v_t do_t dk_t dv_t "
          "h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult "
          "kv_div left right bottom_right ws_rows_q ws_rows_kv ws_seg "
          "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist";
    static const char* const prep_ptrs
        = "O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST";
    static const char* const prep_i64 = "o_b o_h do_b do_h l_b l_h";
    static const char* const prep_i32
        = "o_t do_t l_t h_q h_k h_v S_q_max S_kv_max "
          "has_len len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv ws_seg "
          "zero_kv zero_dq use_worklist n_batch wl_kn0";
    static const char* const conv_ptrs = "SRC DST SEQ OFF";
    static const char* const conv_i64 = "d_b d_h";
    static const char* const conv_f32 = "mult";
    static const char* const conv_i32
        = "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg";
    static const char* const red_i64 = "d_b d_h s_stride";
    static const char* const red_i32
        = "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg n_slices";
    static const char* const dq_ptrs = "Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV";
    static const char* const dq_i64 = "q_b q_h k_b k_h v_b v_h do_b do_h dq_b dq_h";
    static const char* const dq_f32 = "scale_log2 ds_mult dq_mult";
    static const char* const dq_i32
        = "q_t k_t v_t do_t dq_t "
          "h_q h_k h_v G S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
          "left right bottom_right ws_rows_q ws_seg xcd_n xcd_chunk";
    if(stage == "main")
        *out = AbiLists{main_ptrs, main_i64, main_f32, main_i32};
    else if(stage == "prep")
        *out = AbiLists{prep_ptrs, prep_i64, "", prep_i32};
    else if(stage == "convert")
        *out = AbiLists{conv_ptrs, conv_i64, conv_f32, conv_i32};
    else if(stage == "reduce")
        *out = AbiLists{conv_ptrs, red_i64, conv_f32, red_i32};
    else if(stage == "dq")
        *out = AbiLists{dq_ptrs, dq_i64, dq_f32, dq_i32};
    else
        return false;
    return true;
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

/* ====================================================================== *
 * C entry points
 * ====================================================================== */

template <class Fn>
static rocke_status_t spec_boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
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

extern "C" {

rocke_attn_bwd_spec_t rocke_attn_bwd_spec_default(void)
{
    rocke_attn_bwd_spec_t s;
    memset(&s, 0, sizeof(s));
    s.head_size = 0;
    s.dtype = "fp16";
    s.seq_mode = "batched";
    s.dkv_mode = "direct";
    s.stage_vec = 8;
    s.mask_class = "band";
    s.dq_mode = "atomic";
    s.static_scale = "fold_ds";
    s.waves = s.block_m = s.block_n = s.block_k4 = ROCKE_ATTN_BWD_NONE;
    s.warp_grid_g4.has = false;
    s.ring_depth = ROCKE_ATTN_BWD_NONE;
    s.has_lds_swizzle = false;
    s.setprio = s.waves_per_eu = s.edge_tiles = s.head_pack = s.acc_in_lds = ROCKE_ATTN_BWD_NONE;
    s.ws_layout = "head_major";
    s.xcd_remap = false;
    s.agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_NONE;
    s.s_in_agpr = ROCKE_ATTN_BWD_NONE;
    s.agpr_route = NULL;
    s.asm_tail_nop = ROCKE_ATTN_BWD_NONE;
    s.asm_cluster = "k_step";
    s.asm_load_hoist = false;
    s.scheduler_strategy = NULL;
    s.warp_grid_g02.has = false;
    s.warp_grid_g13.has = false;
    s.stage_order = "g1g3g4";
    s.loop_sync = "full";
    s.step_order = "qtile_outer";
    s.dkv_epilogue = "lds_staged";
    s.dkv_store_vec = 8;
    s.dkv_atomic_guard = "branch";
    s.dkv_sink = "atomic";
    s.lds_order = "resident_low";
    s.qdo_stage = "split";
    s.stats_load = "at_use";
    s.dq_issue = "block";
    s.g4_slicing = "step";
    s.ptr_noalias = NULL;
    s.barrier_merge = "off";
    s.ring_issue = "early";
    s.counted_issue = "early";
    s.bf16_cast = "generic";
    s.bwd_exp2 = "exact";
    s.stats_fma = false;
    s.stats_vec = 1;
    s.worklist = false;
    s.ws_bounds = "clamp";
    s.dkv_cell_vec = 1;
    s.narrow_vec = 1;
    s.slab_carry = false;
    s.name = "rocke_attn_bwd";
    return s;
}

rocke_status_t rocke_attn_bwd_spec_init(rocke_attn_bwd_spec_t* spec, rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { spec_init(*spec); });
}

rocke_status_t rocke_attn_bwd_spec_set_lds_swizzle(rocke_attn_bwd_spec_t* spec,
                                                   const char* text,
                                                   rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        if(text == NULL)
        {
            spec->has_lds_swizzle = false;
            spec->lds_swizzle[0] = '\0';
            return;
        }
        set_swizzle(*spec, text);
    });
}

rocke_status_t rocke_attn_bwd_spec_resolved(const rocke_attn_bwd_spec_t* spec,
                                            const char* arch,
                                            rocke_attn_bwd_spec_t* out,
                                            rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        rocke_attn_bwd_spec_t r = resolved(*spec, arch ? arch : "");
        *out = r;
    });
}

rocke_status_t rocke_attn_bwd_spec_kernel_name(const rocke_attn_bwd_spec_t* spec,
                                               const char* stage,
                                               char* out,
                                               size_t out_cap,
                                               rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        std::string n = kernel_name(*spec, stage ? stage : "");
        if(out == NULL || n.size() >= out_cap)
            throw ckc::ValueError("kernel name does not fit the caller's buffer");
        memcpy(out, n.c_str(), n.size() + 1);
    });
}

rocke_status_t rocke_attn_bwd_spec_aux_spec(const rocke_attn_bwd_spec_t* spec,
                                            const char* stage,
                                            int rows_per_cta,
                                            int convert_vec,
                                            rocke_attn_bwd_aux_spec_t* out,
                                            rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        std::string st = stage ? stage : "";
        if(st != "prep" && st != "convert")
            throw ckc::ValueError("stage=" + repr_str(stage) + " not in ('prep', 'convert')");
        if(st == "prep" && convert_vec != 8)
            throw ckc::ValueError("convert_vec applies to the convert stage only");
        rocke_attn_bwd_aux_spec_t a = rocke_attn_bwd_aux_spec_default();
        a.head_size = spec->head_size;
        a.dtype = spec->dtype;
        a.seq_mode = spec->seq_mode;
        a.stage_vec = spec->stage_vec;
        a.ws_layout = spec->ws_layout;
        a.has_rows_per_cta = rows_per_cta != ROCKE_ATTN_BWD_NONE;
        a.rows_per_cta = a.has_rows_per_cta ? rows_per_cta : 0;
        a.convert_vec = convert_vec;
        a.narrow_vec = spec->narrow_vec;
        a.name = spec->name;
        a.worklist = st == "prep" && spec->worklist;
        char reason[512];
        if(!rocke_attn_bwd_aux_spec_validate(&a, reason, sizeof(reason)))
            throw ckc::ValueError(reason);
        *out = a;
    });
}

rocke_status_t rocke_attn_bwd_arch_facts(const char* arch,
                                         rocke_attn_bwd_arch_facts_t* out,
                                         rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { *out = arch_facts(arch ? arch : ""); });
}

static int copy_ints(const Ints& v, int out[4])
{
    for(size_t i = 0; i < v.size() && i < 4; ++i)
        out[i] = v[i];
    return (int)v.size();
}

/* The legal_* string tuples, as static spellings. */
static int copy_strs(const Strs& v, const char* out[4])
{
    static const char* const spellings[] = {"16x16x16",
                                            "16x16x32",
                                            "wmma16x16x16",
                                            "tr_read",
                                            "xt_lds",
                                            "lds_plain",
                                            "wmma_lds",
                                            "vgpr",
                                            "dma"};
    int n = 0;
    for(const std::string& x : v)
        for(const char* sp : spellings)
            if(x == sp && n < 4)
                out[n++] = sp;
    return n;
}

int rocke_attn_bwd_facts_legal_waves(const rocke_attn_bwd_arch_facts_t* f, int out[4])
{
    return copy_ints(legal_waves(*f), out);
}

int rocke_attn_bwd_facts_legal_atoms(const rocke_attn_bwd_arch_facts_t* f, const char* out[4])
{
    return copy_strs(legal_atoms(*f), out);
}

int rocke_attn_bwd_facts_legal_transpose_sources(const rocke_attn_bwd_arch_facts_t* f,
                                                 const char* out[4])
{
    return copy_strs(legal_transpose_sources(*f), out);
}

int rocke_attn_bwd_facts_legal_global_paths(const rocke_attn_bwd_arch_facts_t* f,
                                            const char* out[4])
{
    return copy_strs(legal_global_paths(*f), out);
}

int rocke_attn_bwd_facts_legal_ring_depths(const rocke_attn_bwd_arch_facts_t* f, int out[4])
{
    return copy_ints(legal_ring_depths(*f), out);
}

rocke_status_t rocke_attn_bwd_default_geometry(const char* arch,
                                               int head_size,
                                               const char* dtype,
                                               int stage_vec,
                                               const char* seq_mode,
                                               rocke_attn_bwd_spec_t* out,
                                               rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        Geometry g = default_geometry(arch ? arch : "", head_size, dtype, stage_vec, seq_mode);
        rocke_attn_bwd_spec_t s = rocke_attn_bwd_spec_default();
        s.head_size = head_size;
        s.waves = g.waves;
        s.block_m = g.block_m;
        s.block_n = g.block_n;
        s.block_k4 = g.block_k4;
        s.warp_grid_g4 = g.warp_grid_g4;
        s.atom_g02 = g.atom_g02;
        s.atom_g13 = g.atom_g13;
        s.atom_g4 = g.atom_g4;
        s.pt_route = g.pt_route;
        s.kv_residency = g.kv_residency;
        s.kt_source = g.kt_source;
        s.transpose_source = g.transpose_source;
        s.ring_depth = g.ring_depth;
        s.global_path = g.global_path;
        s.sched = g.sched;
        s.setprio = g.setprio;
        s.waves_per_eu = g.waves_per_eu;
        s.edge_tiles = g.edge_tiles;
        s.head_pack = g.head_pack;
        s.acc_in_lds = g.acc_in_lds;
        s.s_in_agpr = g.s_in_agpr;
        s.scheduler_strategy = g.scheduler_strategy;
        *out = s;
    });
}

rocke_status_t rocke_attn_bwd_policy_agpr_alloc(const rocke_attn_bwd_spec_t* spec,
                                                const char* arch,
                                                rocke_attn_bwd_agpr_alloc_t* out,
                                                rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { *out = policy_agpr_alloc(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_lds_request(const rocke_attn_bwd_spec_t* spec,
                                          const char* arch,
                                          rocke_attn_bwd_lds_request_t* out,
                                          rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { *out = lds_request(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_lds_plan(const rocke_attn_bwd_spec_t* spec,
                                       const char* arch,
                                       rocke_attn_bwd_lds_plan_t* out,
                                       rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { *out = lds_plan(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_lds_footprint_bytes(const rocke_attn_bwd_spec_t* spec,
                                                  const char* arch,
                                                  int* out,
                                                  rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { *out = lds_footprint_bytes(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_register_estimates(const rocke_attn_bwd_spec_t* spec,
                                                 const char* arch,
                                                 int* arch_vgpr,
                                                 int* agpr,
                                                 rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        RegEstimate e = register_estimates(*spec, arch ? arch : "");
        if(arch_vgpr != NULL)
            *arch_vgpr = e.arch_vgpr;
        if(agpr != NULL)
            *agpr = e.agpr;
    });
}

rocke_status_t rocke_attn_bwd_agpr_copy_budget(const rocke_attn_bwd_spec_t* spec,
                                               const char* arch,
                                               int* reads,
                                               int* writes,
                                               int* moves,
                                               rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        CopyBudget c = agpr_copy_budget(*spec, arch ? arch : "");
        if(reads != NULL)
            *reads = c.reads;
        if(writes != NULL)
            *writes = c.writes;
        if(moves != NULL)
            *moves = c.moves;
    });
}

rocke_status_t rocke_attn_bwd_check_counted_sync(const rocke_attn_bwd_spec_t* spec,
                                                 const char* arch,
                                                 rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { check_counted_sync(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_validate_spec(const rocke_attn_bwd_spec_t* spec,
                                            const char* arch,
                                            rocke_attn_bwd_spec_t* out,
                                            rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        rocke_attn_bwd_spec_t r = validate_spec(*spec, arch ? arch : "");
        *out = r;
    });
}

rocke_status_t rocke_attn_bwd_tile_geometry_check(const rocke_attn_bwd_spec_t* spec,
                                                  const char* arch,
                                                  rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] { geometry_check(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_static_unit_mults(const char* static_scale,
                                                const char* dkv_mode,
                                                unsigned* mask,
                                                rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        check_in("static_scale", static_scale, STATIC_SCALES);
        unsigned unit = 0;
        if(!eq(static_scale, "runtime"))
        {
            unit = ROCKE_ATTN_BWD_UNIT_DS_MULT | ROCKE_ATTN_BWD_UNIT_DK_MULT
                   | ROCKE_ATTN_BWD_UNIT_DQ_MULT;
            if(eq(static_scale, "fold_ds"))
                unit &= ~(unsigned)ROCKE_ATTN_BWD_UNIT_DS_MULT;
            else if(eq(dkv_mode, "direct"))
                unit &= ~(unsigned)ROCKE_ATTN_BWD_UNIT_DK_MULT;
            if(eq(static_scale, "dq_before_atomic"))
                unit &= ~(unsigned)ROCKE_ATTN_BWD_UNIT_DQ_MULT;
        }
        *mask = unit;
    });
}

rocke_status_t rocke_attn_bwd_params(const char* stage,
                                     const char* names[ROCKE_ATTN_BWD_PARAMS_MAX],
                                     const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX],
                                     int* n_out,
                                     rocke_attn_bwd_err_t* err)
{
    return spec_boundary(err, [&] {
        AbiLists l;
        if(!abi_lists(stage ? stage : "", &l))
            throw ckc::ValueError("stage=" + repr_str(stage)
                                  + " not in ('main', 'prep', 'convert', 'reduce', 'dq')");
        /* static storage of the split names, per stage */
        static std::vector<std::string> store[5][4];
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        static const char* const stages[5] = {"main", "prep", "convert", "reduce", "dq"};
        int si = 0;
        while(strcmp(stages[si], stage) != 0)
            ++si;
        const char* lists[4] = {l.ptrs, l.i64s, l.f32s, l.i32s};
        static const char* const fmts[4] = {"Q", "q", "f", "i"};
        int n = 0;
        for(int k = 0; k < 4; ++k)
        {
            if(store[si][k].empty())
                store[si][k] = split_words(lists[k]);
            for(const std::string& nm : store[si][k])
            {
                if(n >= ROCKE_ATTN_BWD_PARAMS_MAX)
                    throw ckc::ValueError("kernarg list exceeds the C port's limit");
                names[n] = nm.c_str();
                formats[n] = fmts[k];
                ++n;
            }
        }
        *n_out = n;
    });
}

} /* extern "C" */
