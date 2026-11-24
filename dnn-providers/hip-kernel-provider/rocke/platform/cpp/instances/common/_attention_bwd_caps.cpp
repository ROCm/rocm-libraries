// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_caps.py: the per-arch
 * capability facts and checks of the attention backward kernels, plus the
 * Python-repr helpers the backward spec ports share for their messages.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_caps.h for the
 * mapping. The checks throw ckc::Error internally (one throw per Python
 * raise, same message); the C entry points catch and report it.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/dtypes.h"
#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp" /* ckc::guard_status */
#include "rocke/helper_rocke.helpers.attention_arch_caps.h"
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_floor_div, rocke_ceil_div */
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

const char* const rocke_attn_bwd_arches[ROCKE_ATTN_BWD_N_ARCHES]
    = {"gfx942", "gfx950", "gfx1151", "gfx1201"};

namespace rocke_attn_bwd
{

/* ====================================================================== *
 * Python repr helpers
 * ====================================================================== */

std::vector<unsigned> py_codepoints(const std::string& s)
{
    std::vector<unsigned> out;
    size_t i = 0;
    while(i < s.size())
    {
        unsigned char c = (unsigned char)s[i];
        int extra = c < 0x80             ? 0
                    : (c & 0xe0) == 0xc0 ? 1
                    : (c & 0xf0) == 0xe0 ? 2
                    : (c & 0xf8) == 0xf0 ? 3
                                         : -1;
        if(extra < 0 || (extra > 0 && i + (size_t)extra >= s.size()))
            throw ckc::NotImplError("text that is not UTF-8 has no Python spelling");
        unsigned cp = extra == 0 ? c : (unsigned)(c & (0x3f >> extra));
        for(int k = 1; k <= extra; ++k)
        {
            unsigned char t = (unsigned char)s[i + (size_t)k];
            if((t & 0xc0) != 0x80)
                throw ckc::NotImplError("text that is not UTF-8 has no Python spelling");
            cp = (cp << 6) | (t & 0x3f);
        }
        out.push_back(cp);
        i += (size_t)extra + 1;
    }
    return out;
}

std::string py_repr(const char* s)
{
    /* rocke_py_str_repr (ir.h) is the engine's one port of repr(str). */
    std::string out(rocke_py_str_repr(s, NULL, 0), '\0');
    rocke_py_str_repr(s, &out[0], out.size() + 1);
    return out;
}

static std::string join_tuple(const std::vector<std::string>& reprs)
{
    std::string out = "(";
    for(size_t i = 0; i < reprs.size(); ++i)
    {
        if(i)
            out += ", ";
        out += reprs[i];
    }
    if(reprs.size() == 1)
        out += ",";
    out += ")";
    return out;
}

std::string py_tuple(const std::vector<std::string>& items)
{
    std::vector<std::string> r;
    for(const auto& s : items)
        r.push_back(py_repr(s));
    return join_tuple(r);
}

static long long py_divisor(long long b)
{
    if(b == 0) /* Python's ZeroDivisionError; ckc has no twin, so a ValueError */
        throw ckc::ValueError("integer division or modulo by zero");
    return b;
}

long long py_floordiv(long long a, long long b)
{
    return rocke_floor_div(a, py_divisor(b));
}

long long py_mod(long long a, long long b)
{
    return a - py_floordiv(a, b) * b;
}

long long py_ceil_div(long long a, long long b)
{
    return rocke_ceil_div(a, py_divisor(b));
}

std::string py_tuple(const std::vector<int>& items)
{
    std::vector<std::string> r;
    for(int v : items)
        r.push_back(std::to_string(v));
    return join_tuple(r);
}

std::string py_list(const std::vector<int>& items)
{
    std::string out = "[";
    for(size_t i = 0; i < items.size(); ++i)
    {
        if(i)
            out += ", ";
        out += std::to_string(items[i]);
    }
    return out + "]";
}

rocke_status_t store_err(rocke_attn_bwd_err_t* err, rocke_status_t code, const char* msg)
{
    if(err != NULL)
    {
        err->code = code;
        snprintf(err->msg, sizeof(err->msg), "%s", msg ? msg : "");
    }
    return code;
}

static void clear_err(rocke_attn_bwd_err_t* err)
{
    if(err != NULL)
    {
        err->code = ROCKE_OK;
        err->msg[0] = '\0';
    }
}

/* Run fn at a C boundary: ROCKE_OK, or the caught Python exception. */
template <class Fn>
static rocke_status_t boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    clear_err(err);
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

/* ====================================================================== *
 * BwdArchCaps
 * ====================================================================== */

static const char* arch_static(const std::string& arch)
{
    for(int i = 0; i < ROCKE_ATTN_BWD_N_ARCHES; ++i)
        if(arch == rocke_attn_bwd_arches[i])
            return rocke_attn_bwd_arches[i];
    return NULL;
}

static std::string arches_repr()
{
    std::vector<std::string> v(rocke_attn_bwd_arches,
                               rocke_attn_bwd_arches + ROCKE_ATTN_BWD_N_ARCHES);
    return py_tuple(v);
}

/* A rocke.helpers.attention_arch_caps call (the shared table the Python
 * backward reads): its false return raises the Python ValueError message it
 * wrote. */
template <class Fn>
static void shared_caps(Fn&& fn)
{
    char err[ROCKE_ATTN_BWD_ERR_CAP];
    err[0] = '\0';
    if(!fn(err, sizeof(err)))
        throw ckc::ValueError(err);
}

const rocke_attn_bwd_caps_t& arch_caps(const std::string& arch)
{
    /* @cache: filled lazily under a lock, so concurrent builds are safe */
    static rocke_attn_bwd_caps_t cache[ROCKE_ATTN_BWD_N_ARCHES];
    static bool built[ROCKE_ATTN_BWD_N_ARCHES];
    static std::mutex mu;
    const char* sarch = arch_static(arch);
    if(sarch == NULL)
        throw ckc::ValueError("attention backward does not target " + py_repr(arch)
                              + "; known: " + arches_repr());
    int idx = 0;
    while(rocke_attn_bwd_arches[idx] != sarch)
        ++idx;
    std::lock_guard<std::mutex> lock(mu);
    if(built[idx])
        return cache[idx];
    /* bwd_arch_caps: the shared attention table's facts plus the backward's
     * own policy (no 128-bit transpose read, the legal wave counts) */
    rocke_attn_arch_caps_t a;
    shared_caps([&](char* err, size_t cap) { return rocke_attn_arch_caps(sarch, &a, err, cap); });
    bool mfma = strcmp(a.matrix_path, "mfma") == 0;
    rocke_attn_bwd_caps_t c;
    memset(&c, 0, sizeof(c));
    c.arch = sarch;
    c.wave_size = a.wave_size;
    c.matrix_path = mfma ? "mfma" : "wmma";
    c.lds_capacity_bytes = a.lds_capacity_bytes;
    c.n_lds_dma_dwords = a.n_lds_dma_dwords;
    for(int i = 0; i < a.n_lds_dma_dwords; ++i)
        c.lds_dma_dwords[i] = a.lds_dma_dwords[i];
    c.has_tr_read_b64 = rocke_attn_caps_has_tr_read_b64(&a);
    c.has_tr_read_b128 = false;
    c.has_agprs = a.has_agprs;
    c.arch_vgprs = a.arch_vgprs;
    c.agprs = a.agprs;
    c.native_fp32_atomic_add = a.native_fp32_atomic_add;
    static const int mfma_waves[] = {1, 2, 4, 8};
    static const int wmma_waves[] = {1, 2, 4};
    const int* w = mfma ? mfma_waves : wmma_waves;
    c.n_legal_waves = mfma ? 4 : 3;
    for(int i = 0; i < c.n_legal_waves; ++i)
        c.legal_waves[i] = w[i];
    cache[idx] = c;
    built[idx] = true;
    return cache[idx];
}

static std::vector<int> legal_waves(const rocke_attn_bwd_caps_t& c)
{
    return std::vector<int>(c.legal_waves, c.legal_waves + c.n_legal_waves);
}

std::string norm_dtype(const std::string& dtype)
{
    if(dtype == "fp16" || dtype == "f16")
        return "fp16";
    if(dtype == "bf16")
        return "bf16";
    throw ckc::ValueError("backward operand dtype must be fp16 or bf16, got " + py_repr(dtype));
}

int elem_bytes(const char* dtype)
{
    const rocke_dtype_info_t* info = rocke_dtype_info(dtype);
    if(info == NULL || info->encoded_bits % 8 != 0)
        throw ckc::ValueError("no whole-byte element size for dtype " + py_repr(dtype));
    return info->encoded_bits / 8;
}

std::vector<const rocke_mma_op_t*> atoms(const std::string& arch, const std::string& dtype)
{
    const rocke_attn_bwd_caps_t& caps = arch_caps(arch);
    std::string dt = norm_dtype(dtype);
    const char* family = strcmp(caps.matrix_path, "mfma") == 0 ? "mma" : "wmma";
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(caps.arch);
    rocke_mma_scale_filter_t* any_scale = NULL;
    int n = rocke_mma_catalog_enumerate(
        &t->mma, family, dt.c_str(), dt.c_str(), "fp32", 16, 16, NULL, 0, any_scale);
    std::vector<const rocke_mma_op_t*> ops((size_t)(n > 0 ? n : 0));
    if(n > 0)
        rocke_mma_catalog_enumerate(
            &t->mma, family, dt.c_str(), dt.c_str(), "fp32", 16, 16, ops.data(), n, any_scale);
    std::vector<const rocke_mma_op_t*> out;
    for(const rocke_mma_op_t* op : ops)
        if(op->wave_size == caps.wave_size && op->srcs[0].frag_len > 0)
            out.push_back(op);
    /* sorted(..., key=k): stable */
    std::stable_sort(out.begin(), out.end(), [](const rocke_mma_op_t* a, const rocke_mma_op_t* b) {
        return a->k < b->k;
    });
    if(out.empty())
        throw ckc::ValueError("no 16x16 " + dt + " MMA atom on " + arch);
    return out;
}

const rocke_mma_op_t* select_atom(const std::string& arch, const std::string& dtype, int k)
{
    std::vector<const rocke_mma_op_t*> ops = atoms(arch, dtype);
    for(const rocke_mma_op_t* op : ops)
        if(op->k == k)
            return op;
    std::string dt = norm_dtype(dtype);
    std::vector<int> ks;
    for(const rocke_mma_op_t* op : atoms(arch, dtype))
        ks.push_back(op->k);
    throw ckc::ValueError("no native 16x16x" + std::to_string(k) + " " + dt + " atom on " + arch
                          + " (available K: " + py_list(ks) + ")");
}

/* A knob and its catalog shape, owning its spelling. */
struct KnobShape
{
    std::string spelling;
    bool wmma;
    int m, n, k;
};

/* atom_knob(op): "<m>x<n>x<k>", prefixed "wmma" for a WMMA atom. */
static KnobShape knob_of(const rocke_mma_op_t* op)
{
    bool wmma = strcmp(op->family, "wmma") == 0;
    std::string s = std::string(wmma ? "wmma" : "") + std::to_string(op->m) + "x"
                    + std::to_string(op->n) + "x" + std::to_string(op->k);
    return KnobShape{s, wmma, op->m, op->n, op->k};
}

static void add_knob(std::vector<KnobShape>& out, const KnobShape& k)
{
    for(const KnobShape& have : out)
        if(have.spelling == k.spelling)
            return;
    out.push_back(k);
}

/* _knob_ks(arch): the knobs of arch's backward atoms (fp16 and bf16
 * operands), by increasing K. */
static std::vector<KnobShape> arch_knobs(const std::string& arch)
{
    std::vector<KnobShape> out;
    for(const char* dtype : {"fp16", "bf16"})
        for(const rocke_mma_op_t* op : atoms(arch, dtype))
            add_knob(out, knob_of(op));
    std::stable_sort(
        out.begin(), out.end(), [](const KnobShape& a, const KnobShape& b) { return a.k < b.k; });
    return out;
}

std::vector<std::string> bwd_atom_knobs(const std::string& arch)
{
    std::vector<std::string> out;
    for(const KnobShape& k : arch_knobs(arch))
        out.push_back(k.spelling);
    return out;
}

/* _all_knob_ks(): the union over the backward arches, MFMA by K, then WMMA.
 * Built once; atom_knobs() points into these strings, which never move. */
static const std::vector<KnobShape>& all_knobs()
{
    static const std::vector<KnobShape> all = [] {
        std::vector<KnobShape> out;
        for(int i = 0; i < ROCKE_ATTN_BWD_N_ARCHES; ++i)
            for(const KnobShape& k : arch_knobs(rocke_attn_bwd_arches[i]))
                add_knob(out, k);
        std::stable_sort(out.begin(), out.end(), [](const KnobShape& a, const KnobShape& b) {
            return a.wmma != b.wmma ? !a.wmma : a.k < b.k;
        });
        return out;
    }();
    return all;
}

const std::vector<AtomKnob>& atom_knobs()
{
    static const std::vector<AtomKnob> knobs = [] {
        std::vector<AtomKnob> out;
        for(const KnobShape& k : all_knobs())
            out.push_back(AtomKnob{k.spelling.c_str(), k.wmma, k.m, k.n, k.k});
        return out;
    }();
    return knobs;
}

const AtomKnob* find_atom_knob(const char* spelling)
{
    if(spelling == NULL)
        return NULL;
    for(const AtomKnob& a : atom_knobs())
        if(strcmp(a.spelling, spelling) == 0)
            return &a;
    return NULL;
}

int atom_knob_k(const char* spelling)
{
    const AtomKnob* a = find_atom_knob(spelling);
    if(a == NULL)
        throw ckc::ValueError("unknown atom " + py_repr(spelling)
                              + "; known atoms: " + py_tuple(atom_knob_spellings(false)));
    return a->k;
}

const char* atom_knob_spelling(bool wmma, int k)
{
    for(const AtomKnob& a : atom_knobs())
        if(a.wmma == wmma && a.k == k)
            return a.spelling;
    throw ckc::ValueError(std::string(wmma ? "no WMMA" : "no MFMA") + " atom knob with K "
                          + std::to_string(k));
}

std::vector<std::string> atom_knob_spellings(bool mfma_only)
{
    std::vector<std::string> out;
    for(const AtomKnob& a : atom_knobs())
        if(!(mfma_only && a.wmma))
            out.push_back(a.spelling);
    return out;
}

bool wide_k_atom_available(const std::string& arch, const std::string& dtype)
{
    arch_caps(arch); /* _gate */
    return rocke_attn_wide_k_atom_available(arch.c_str(), norm_dtype(dtype).c_str());
}

void check_waves(const std::string& arch, int waves)
{
    const rocke_attn_bwd_caps_t& caps = arch_caps(arch);
    std::vector<int> legal = legal_waves(caps);
    if(std::find(legal.begin(), legal.end(), waves) == legal.end())
        throw ckc::ValueError("waves=" + std::to_string(waves) + " is not legal on " + arch
                              + "; legal: " + py_tuple(legal));
}

void check_lds_bytes(const std::string& arch, long long nbytes)
{
    arch_caps(arch); /* _gate */
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_check_lds_bytes(arch.c_str(), (long)nbytes, err, cap);
    });
}

void check_lds_dma_width(const std::string& arch, int dwords)
{
    arch_caps(arch); /* _gate */
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_check_lds_dma_width(arch.c_str(), dwords, err, cap);
    });
}

void check_transpose_read(const std::string& arch, int bits)
{
    arch_caps(arch); /* _gate */
    if(bits == 128)
        throw ckc::ValueError("ds_read_tr16_b128 is not used by the attention backward (only the "
                              "64-bit transpose read is legal)");
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_check_transpose_read(arch.c_str(), bits, err, cap);
    });
}

/* ====================================================================== *
 * asm MFMA route
 * ====================================================================== */

const char* mfma_asm_mnemonic(const std::string& arch, const std::string& dtype, int atom_k)
{
    arch_caps(arch); /* _gate */
    const char* m = NULL;
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_mfma_asm_mnemonic(arch.c_str(), dtype.c_str(), atom_k, &m, err, cap);
    });
    return m;
}

int mfma_result_wait_states(const std::string& arch, int atom_k)
{
    arch_caps(arch); /* _gate */
    int n = 0;
    char err[ROCKE_ATTN_BWD_ERR_CAP];
    err[0] = '\0';
    if(!rocke_attn_mfma_result_wait_states(arch.c_str(), atom_k, &n, err, sizeof(err)))
    {
        /* The shared C entry reports Python's KeyError(atom_k), an atom K
         * with no XDL pass count, as the bare key; every other refusal is a
         * ValueError sentence. */
        if(std::string(err) == std::to_string(atom_k))
            throw ckc::KeyError(err);
        throw ckc::ValueError(err);
    }
    return n;
}

int asm_tail_nop_min(const std::string& arch, const std::string& dtype, int atom_k)
{
    arch_caps(arch); /* _gate */
    int n = 0;
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_asm_tail_nop_min(arch.c_str(), dtype.c_str(), atom_k, &n, err, cap);
    });
    return n;
}

void check_asm_tail_nop(const std::string& arch, const std::string& dtype, int atom_k, int nop)
{
    arch_caps(arch); /* _gate */
    shared_caps([&](char* err, size_t cap) {
        return rocke_attn_check_asm_tail_nop(arch.c_str(), dtype.c_str(), atom_k, nop, err, cap);
    });
}

/* ====================================================================== *
 * agpr_alloc
 * ====================================================================== */

std::string agpr_alloc_repr(const rocke_attn_bwd_agpr_alloc_t& a)
{
    if(a.kind == ROCKE_ATTN_BWD_AGPR_NONE)
        return "None";
    if(a.kind == ROCKE_ATTN_BWD_AGPR_STR)
        return py_repr(a.str ? a.str : "");
    return "(" + std::to_string(a.a) + ", " + std::to_string(a.b) + ")";
}

/* _AGPR_ALLOC */
static const char* const AGPR_ALLOC_REPR
    = "(None, 'backend', (0, 0), (128, 128), (192, 192), (256, 256))";

static bool agpr_alloc_legal(const rocke_attn_bwd_agpr_alloc_t& a)
{
    if(a.kind == ROCKE_ATTN_BWD_AGPR_NONE)
        return true;
    if(a.kind == ROCKE_ATTN_BWD_AGPR_STR)
        return a.str != NULL && strcmp(a.str, "backend") == 0;
    return a.a == a.b && (a.a == 0 || a.a == 128 || a.a == 192 || a.a == 256);
}

void check_agpr_alloc(const std::string& arch, const rocke_attn_bwd_agpr_alloc_t& a)
{
    const rocke_attn_bwd_caps_t& caps = arch_caps(arch);
    if(a.kind == ROCKE_ATTN_BWD_AGPR_NONE)
        return;
    if(a.kind == ROCKE_ATTN_BWD_AGPR_STR && a.str != NULL && strcmp(a.str, "backend") == 0)
    {
        if(!caps.has_agprs)
            throw ckc::ValueError(arch + " has no AGPRs; agpr_alloc must be None");
        return;
    }
    if(!caps.has_agprs)
        throw ckc::ValueError(arch + " has no AGPRs; agpr_alloc must be None");
    if(!agpr_alloc_legal(a))
        throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(a)
                              + " is not a legal value; legal: " + AGPR_ALLOC_REPR);
    if(a.b > caps.agprs)
        throw ckc::ValueError("agpr_alloc=" + agpr_alloc_repr(a) + " exceeds "
                              + std::to_string(caps.agprs) + " AGPRs");
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

/* ====================================================================== *
 * C entry points
 * ====================================================================== */

static std::string s_or_empty(const char* s)
{
    return s ? std::string(s) : std::string();
}

extern "C" {

rocke_status_t rocke_attn_bwd_arch_caps(const char* arch,
                                        rocke_attn_bwd_caps_t* out,
                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        rocke_attn_bwd_caps_t c = arch_caps(s_or_empty(arch));
        if(out != NULL)
            *out = c;
    });
}

int rocke_attn_bwd_caps_max_lds_dma_dwords(const rocke_attn_bwd_caps_t* caps)
{
    int m = 0;
    for(int i = 0; caps != NULL && i < caps->n_lds_dma_dwords; ++i)
        m = std::max(m, caps->lds_dma_dwords[i]);
    return m;
}

rocke_status_t rocke_attn_bwd_atoms(const char* arch,
                                    const char* dtype,
                                    const rocke_mma_op_t** out,
                                    int cap,
                                    int* n_out,
                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::vector<const rocke_mma_op_t*> ops = atoms(s_or_empty(arch), s_or_empty(dtype));
        if((int)ops.size() > cap)
            throw ckc::ValueError("bwd_atoms: " + std::to_string(ops.size())
                                  + " atoms exceed the caller's buffer of " + std::to_string(cap));
        for(size_t i = 0; i < ops.size(); ++i)
            out[i] = ops[i];
        if(n_out != NULL)
            *n_out = (int)ops.size();
    });
}

const rocke_mma_op_t* rocke_attn_bwd_select_atom(const char* arch,
                                                 const char* dtype,
                                                 int k,
                                                 rocke_attn_bwd_err_t* err)
{
    const rocke_mma_op_t* op = NULL;
    boundary(err, [&] { op = select_atom(s_or_empty(arch), s_or_empty(dtype), k); });
    return op;
}

rocke_status_t rocke_attn_bwd_wide_k_atom_available(const char* arch,
                                                    const char* dtype,
                                                    bool* out,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        bool v = wide_k_atom_available(s_or_empty(arch), s_or_empty(dtype));
        if(out != NULL)
            *out = v;
    });
}

rocke_status_t rocke_attn_bwd_check_waves(const char* arch, int waves, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { check_waves(s_or_empty(arch), waves); });
}

rocke_status_t
    rocke_attn_bwd_check_lds_bytes(const char* arch, long long nbytes, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { check_lds_bytes(s_or_empty(arch), nbytes); });
}

rocke_status_t
    rocke_attn_bwd_check_lds_dma_width(const char* arch, int dwords, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { check_lds_dma_width(s_or_empty(arch), dwords); });
}

rocke_status_t
    rocke_attn_bwd_check_transpose_read(const char* arch, int bits, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { check_transpose_read(s_or_empty(arch), bits); });
}

int rocke_attn_bwd_agpr_asm_tail_nop(int atom_k)
{
    return (atom_k == 16 || atom_k == 32) ? 15 : -1;
}

const char* rocke_attn_bwd_mfma_asm_mnemonic(const char* arch,
                                             const char* dtype,
                                             int atom_k,
                                             rocke_attn_bwd_err_t* err)
{
    const char* m = NULL;
    boundary(err, [&] { m = mfma_asm_mnemonic(s_or_empty(arch), s_or_empty(dtype), atom_k); });
    return m;
}

int rocke_attn_bwd_mfma_result_wait_states(const char* arch, int atom_k, rocke_attn_bwd_err_t* err)
{
    int v = -1;
    boundary(err, [&] { v = mfma_result_wait_states(s_or_empty(arch), atom_k); });
    return v;
}

int rocke_attn_bwd_asm_tail_nop_min(const char* arch,
                                    const char* dtype,
                                    int atom_k,
                                    rocke_attn_bwd_err_t* err)
{
    int v = -1;
    boundary(err, [&] { v = asm_tail_nop_min(s_or_empty(arch), s_or_empty(dtype), atom_k); });
    return v;
}

rocke_status_t rocke_attn_bwd_check_asm_tail_nop(
    const char* arch, const char* dtype, int atom_k, int nop, rocke_attn_bwd_err_t* err)
{
    return boundary(err,
                    [&] { check_asm_tail_nop(s_or_empty(arch), s_or_empty(dtype), atom_k, nop); });
}

rocke_status_t rocke_attn_bwd_check_agpr_alloc(const char* arch,
                                               rocke_attn_bwd_agpr_alloc_t agpr_alloc,
                                               rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { check_agpr_alloc(s_or_empty(arch), agpr_alloc); });
}

rocke_status_t rocke_attn_bwd_emit_lds_dma(rocke_ir_builder_t* b,
                                           const char* arch,
                                           rocke_value_t* rsrc,
                                           rocke_value_t* lds_ptr,
                                           rocke_value_t* voffset,
                                           rocke_value_t* soffset,
                                           int dwords,
                                           int coherency)
{
    if(b == NULL)
        return ROCKE_ERR_VALUE;
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(rocke_ir_builder_status(b) != ROCKE_OK)
            return rocke_ir_builder_status(b);
        /* py: check_lds_dma_width(arch, dwords) -- raises before any op */
        check_lds_dma_width(s_or_empty(arch), dwords);
        rocke_b_async_buffer_load_lds(b, rsrc, lds_ptr, voffset, soffset, dwords, coherency);
        return rocke_ir_builder_status(b);
    });
}

} /* extern "C" */
