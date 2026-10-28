// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_attention.cpp -- attention-family bindings for the
 * rocke_engine module. Kept in its own translation unit because the attention
 * instance headers share a set of struct tags (rocke_fmha_*, the tiled 2d/3d spec
 * tags) that are re-used across several headers; isolating them here keeps the
 * GEMM/conv translation unit's include surface clean.
 *
 * register_attention(m) is called from the main module file. Each family mirrors
 * its tests/parity/<fam>_emit.c make_spec/make_cfg + build + lower exactly, so
 * for the same spec dict the binding output is byte-identical to the standalone
 * emitter.
 *
 * Families that emit two kernels per config (the tiled-3d segment+reduce pair
 * and splitkv-decode segment+reduce) concatenate the two .ll texts in the same
 * order the standalone emitter prints them, controlled by the spec dict "phase"
 * key ("both" default | "segment" | "reduce").
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "builder_lifetime.hpp"
#include "llvm_flavor.hpp"

#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "family_glue.hpp"
#include "rocke/error.hpp"

extern "C" {
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors_3d.h"
#include "rocke/instance_attention_unified.h"
#include "rocke/instance_fmha_appendkv.h"
#include "rocke/instance_fmha_bwd.h"
#include "rocke/instance_fmha_fwd_fp8.h"
#include "rocke/instance_fmha_head_grouping.h"
#include "rocke/instance_fmha_mfma.h"
#include "rocke/instance_fmha_paged_prefill.h"
#include "rocke/instance_fmha_splitkv_decode.h"
#include "rocke/instance_fmha_varlen.h"
#include "rocke/instance_gfx942_attention_4warp_gqa.h"
#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#include "rocke/instance_sage_attention.h"
#include "rocke/instance_sparse_attention.h"
#include "rocke/instance_sparse_attention_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

/* ----------------------------- dict helpers ----------------------------- */
int a_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
    return dflt;
}
bool a_bool(const py::dict& d, const char* key, bool dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<bool>();
    return dflt;
}
double a_double(const py::dict& d, const char* key, double dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<double>();
    return dflt;
}
bool a_str(const py::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

[[noreturn]] void raise_not_implemented(const std::string& msg)
{
    PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
    throw py::error_already_set();
}

/* A Python spec/problem field the C struct has no slot for. Its default is
 * accepted (so asdict() of a default spec still lowers); any other value would
 * be silently dropped, so it is rejected instead. dflt_none: the Python default
 * is None (dflt is then ignored). */
struct unported_field
{
    const char* key;
    bool dflt_none;
    double dflt;
};

void reject_unported(const py::dict& d, const char* binding, const unported_field* fields, size_t n)
{
    for(size_t i = 0; i < n; ++i)
    {
        const unported_field& f = fields[i];
        if(!d.contains(f.key) || d[f.key].is_none())
            continue;
        if(!f.dflt_none && d[f.key].cast<double>() == f.dflt)
            continue;
        raise_not_implemented(std::string(binding) + ": " + f.key
                              + " is not yet ported to the C++ engine");
    }
}

/* Lifetime helper for any const char* fields pulled from the dict. */
struct Store
{
    std::deque<std::string> v;
    const char* keep(const std::string& s)
    {
        v.push_back(s);
        return v.back().c_str();
    }
};

/* `err` is taken by reference, not as a pointer, so that no call site can pass
 * nullptr: an arch-domain refusal lowered through the err-less
 * rocke_lower_kernel_to_llvm used to reach Python as an ArchDomainError with an
 * empty message, losing the intrinsic, target and flavor that explain it. Lower
 * through the _ex entry. */
std::string
    take_ll(rocke_status_t st, char* ll, const char (&err)[ROCKE_ERR_MSG_CAP], const char* fn)
{
    if(st != ROCKE_OK || !ll)
    {
        if(ll)
            free(ll);
        throw_if_arch_domain(st, err);
        std::string msg = std::string(fn) + " failed (status=" + std::to_string((int)st)
                          + "): " + (err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(ll);
    free(ll);
    return out;
}

std::string ser_kernel(rocke_kernel_def_t* k, const char* fn)
{
    char* t = nullptr;
    rocke_status_t st = rocke_ir_serialize(k, &t);
    if(st != ROCKE_OK || !t)
    {
        if(t)
            free(t);
        throw std::runtime_error(std::string(fn) + " serialize failed");
    }
    std::string out(t);
    free(t);
    return out;
}

std::vector<std::string> ver_kernel(rocke_kernel_def_t* k)
{
    rocke_diag_t* diags = nullptr;
    size_t n = 0;
    rocke_verify(k, &diags, &n);
    std::vector<std::string> out;
    out.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        char* s = rocke_diag_to_string(&diags[i]);
        if(s)
        {
            out.emplace_back(s);
            free(s);
        }
    }
    rocke_diags_free(diags, n);
    return out;
}

rocke_fmha_mask_mode_t mask_of(const py::dict& d, const char* key)
{
    std::string s;
    if(!a_str(d, key, s))
        return ROCKE_FMHA_MASK_NONE;
    if(s == "causal")
        return ROCKE_FMHA_MASK_CAUSAL;
    if(s == "sliding_window")
        return ROCKE_FMHA_MASK_SLIDING_WINDOW;
    if(s == "alibi")
        return ROCKE_FMHA_MASK_ALIBI;
    if(s == "custom")
        return ROCKE_FMHA_MASK_CUSTOM;
    return ROCKE_FMHA_MASK_NONE;
}

/* Build a rocke_fmha_common_spec_t from common sub-keys of the dict. The shape is
 * taken from head_size/num_query_heads/num_kv_heads (+ optional
 * block_size_q/block_size_k for the shape_make form). */
rocke_fmha_common_spec_t common_of(const py::dict& d, Store& st)
{
    rocke_fmha_shape_t shape;
    if(d.contains("block_size_q") || d.contains("block_size_k"))
        shape = rocke_fmha_shape_make(a_int(d, "head_size", 0),
                                      a_int(d, "num_query_heads", 0),
                                      a_int(d, "num_kv_heads", 0),
                                      a_int(d, "block_size_q", 16),
                                      a_int(d, "block_size_k", 64));
    else
        shape = rocke_fmha_shape_default(
            a_int(d, "head_size", 0), a_int(d, "num_query_heads", 0), a_int(d, "num_kv_heads", 0));
    rocke_fmha_common_spec_t c = rocke_fmha_common_spec_default(shape);
    std::string v;
    if(a_str(d, "dtype", v))
        c.dtype = st.keep(v);
    c.mask_mode = mask_of(d, "mask_mode");
    c.sliding_window = a_int(d, "sliding_window", c.sliding_window);
    c.scale_log2 = a_double(d, "scale_log2", c.scale_log2);
    c.use_softcap = a_bool(d, "use_softcap", c.use_softcap);
    c.use_rotary = a_bool(d, "use_rotary", c.use_rotary);
    c.use_dropout = a_bool(d, "use_dropout", c.use_dropout);
    c.use_sinks = a_bool(d, "use_sinks", c.use_sinks);
    return c;
}

/* ===================== attention_unified (scalar 2d) ==================== */

/* use_fp8 has a C slot, but the scalar builder does not emit the fp8 K/V path
 * the Python builder does, so it is treated as unported here. */
const unported_field k_au_unported[] = {
    {"use_additive_bias", false, 0},
    {"use_fp8", false, 0},
    {"fp8_fnuz", false, 0},
    {"target_ctas", false, 0},
    {"causal_top_left", false, 0},
    {"right_bound", false, 0},
};

rocke_unified_attention_problem_t au_build(const py::dict& d, Store& st)
{
    reject_unported(
        d, "attention_unified", k_au_unported, sizeof(k_au_unported) / sizeof(k_au_unported[0]));
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.head_size = a_int(d, "head_size", p.head_size);
    p.block_size = a_int(d, "block_size", p.block_size);
    p.num_query_heads = a_int(d, "num_query_heads", p.num_query_heads);
    p.num_kv_heads = a_int(d, "num_kv_heads", p.num_kv_heads);
    p.total_q = a_int(d, "total_q", p.total_q);
    p.max_seqlen_q = a_int(d, "max_seqlen_q", p.max_seqlen_q);
    p.max_seqlen_k = a_int(d, "max_seqlen_k", p.max_seqlen_k);
    p.num_seqs = a_int(d, "num_seqs", p.num_seqs);
    p.sliding_window = a_int(d, "sliding_window", p.sliding_window);
    p.softcap = a_double(d, "softcap", p.softcap);
    p.use_sinks = a_bool(d, "use_sinks", p.use_sinks);
    p.use_alibi = a_bool(d, "use_alibi", p.use_alibi);
    p.use_qq_bias = a_bool(d, "use_qq_bias", p.use_qq_bias);
    p.num_cus = a_int(d, "num_cus", p.num_cus);
    p.num_kv_blocks = a_int(d, "num_kv_blocks", p.num_kv_blocks);
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        p.waves_per_eu_set = true;
        p.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
    std::string v;
    if(a_str(d, "dtype", v))
        p.dtype = st.keep(v);
    if(a_str(d, "q_dtype", v))
        p.q_dtype = st.keep(v);
    if(a_str(d, "compile_backend", v))
        p.compile_backend = st.keep(v);
    return p;
}

rocke_kernel_def_t* au_make(const py::dict& d, Store& st, rocke_ir_builder_t* b)
{
    rocke_unified_attention_problem_t p = au_build(d, st);
    char kname[512];
    kname[0] = '\0';
    rocke_unified_attention_2d_scalar_kernel_name(&p, nullptr, kname, sizeof kname);
    rocke_ir_builder_init(b, kname);
    return rocke_build_unified_attention_2d_scalar(b, &p, nullptr);
}

std::string au_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = au_make(d, st, &b);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("attention_unified build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.attention_unified_lower_llvm");
}
std::string au_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch); /* validated, not consumed: the kernel is arch-neutral */
    Store st;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = au_make(d, st, &b);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("attention_unified build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.attention_unified_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> au_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch); /* validated, not consumed: the kernel is arch-neutral */
    Store st;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = au_make(d, st, &b);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("attention_unified build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ============================== fmha_mfma ============================== */

rocke_fmha_mfma_spec_t mfma_build(const py::dict& d, Store& st)
{
    rocke_fmha_mfma_spec_t s = rocke_fmha_mfma_spec_default();
    s.head_size = a_int(d, "head_size", s.head_size);
    s.num_query_heads = a_int(d, "num_query_heads", s.num_query_heads);
    s.num_kv_heads = a_int(d, "num_kv_heads", s.num_kv_heads);
    s.seqlen_q = a_int(d, "seqlen_q", s.seqlen_q);
    s.seqlen_k = a_int(d, "seqlen_k", s.seqlen_k);
    s.mask_mode = mask_of(d, "mask_mode");
    s.sliding_window = a_int(d, "sliding_window", s.sliding_window);
    s.scale_log2 = a_double(d, "scale_log2", s.scale_log2);
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    return s;
}

std::string fmma_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    const auto flavor = resolve_python_llvm_flavor();
    const char* a = arch.c_str();
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_mfma(nullptr, &s, a);
    if(!k)
        throw std::runtime_error("rocke_engine.fmha_mfma_lower_llvm build failed");
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(k, flavor, a, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.fmha_mfma_lower_llvm");
}
std::string fmma_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_mfma(nullptr, &s, arch.c_str());
    if(!k)
        throw std::runtime_error("rocke_engine.fmha_mfma_serialize_ir build failed");
    return ser_kernel(k, "rocke_engine.fmha_mfma_serialize_ir");
}
std::vector<std::string> fmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_mfma(nullptr, &s, arch.c_str());
    if(!k)
        throw std::runtime_error("rocke_engine.fmha_mfma_verify build failed");
    return ver_kernel(k);
}

/* ===================== fmha_fwd_fp8 (kb + convenience lower) ============ */

rocke_fmha_fwd_fp8_spec_t fp8_build(const py::dict& d, Store& st, bool* persistent_unused)
{
    (void)persistent_unused;
    rocke_fmha_fwd_fp8_spec_t s = rocke_fmha_fwd_fp8_spec_default();
    s.common = common_of(d, st);
    s.seqlen_q = a_int(d, "seqlen_q", s.seqlen_q);
    s.seqlen_k = a_int(d, "seqlen_k", s.seqlen_k);
    {
        std::string kv;
        if(a_str(d, "kv_dtype", kv))
            s.kv_dtype = (kv == "bf8e5m2" || kv == "bf8") ? ROCKE_KV_BF8_E5M2 : ROCKE_KV_FP8_E4M3;
    }
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
    return s;
}

std::string fp8_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_fmha_fwd_fp8_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.fmha_fwd_fp8_lower_llvm");
}
std::string fp8_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_fp8_new(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_fwd_fp8_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_fwd_fp8_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> fp8_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_fp8_new(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_fwd_fp8_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ===================== fmha_bwd (kb, generic _ex lower) ================= */

rocke_fmha_bwd_spec_t bwd_build(const py::dict& d, Store& st)
{
    rocke_fmha_common_spec_t c = common_of(d, st);
    rocke_fmha_bwd_spec_t s
        = rocke_fmha_bwd_spec_default(c, a_int(d, "seqlen_q", 0), a_int(d, "seqlen_k", 0));
    return s;
}
std::string bwd_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_bwd_lower_llvm build failed");
    }
    auto guard = rocke_bindings::own_builder(&kb, rocke_fmha_kernel_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.fmha_bwd_lower_llvm");
}
std::string bwd_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_bwd_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_bwd_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> bwd_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_bwd_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ===================== fmha_head_grouping (kb) ========================== */

rocke_fmha_head_grouping_spec_t hg_build(const py::dict& d, Store& st)
{
    rocke_fmha_common_spec_t c = common_of(d, st);
    return rocke_fmha_head_grouping_spec_make(c, a_int(d, "seqlen_q", 0), a_int(d, "seqlen_k", 0));
}
std::string hg_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_head_grouping_lower_llvm build failed");
    }
    auto guard = rocke_bindings::own_builder(&kb, rocke_fmha_kernel_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.fmha_head_grouping_lower_llvm");
}
std::string hg_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_head_grouping_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_head_grouping_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> hg_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_head_grouping_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ===================== sparse_attention (jenga / vsa) ================== */

/* Build the jenga / vsa sparse-attention kernel into a caller-owned context so
 * the binding holds the FmhaKernelBuilder and can apply the documented teardown
 * (rocke_fmha_kernel_builder_free) on every exit path. The public
 * rocke_build_{jenga,vsa}_sparse_attention entries own an internal builder that the
 * binding has no handle to; replicating the engine's own prologue/stage/emit
 * sequence here (identical order, byte-identical IR) keeps that builder reachable
 * so it is reclaimed instead of leaked. Exactly one of jenga_ctx / vsa_ctx is
 * populated; *is_vsa selects which one's kb the caller must free. */
rocke_kernel_def_t* sparse_build(const py::dict& d,
                                 Store& st,
                                 const char* arch,
                                 rocke_jenga_sparse_ctx_t* jenga_ctx,
                                 rocke_vsa_sparse_ctx_t* vsa_ctx,
                                 bool* is_vsa)
{
    std::string kind = "jenga";
    a_str(d, "kind", kind);
    rocke_fmha_common_spec_t c = common_of(d, st);
    if(kind == "vsa")
    {
        *is_vsa = true;
        rocke_vsa_sparse_spec_t s
            = rocke_vsa_sparse_spec_default(c, a_int(d, "seqlen_q", 0), a_int(d, "seqlen_k", 0));
        s.block_q = a_int(d, "block_q", s.block_q);
        s.block_k = a_int(d, "block_k", s.block_k);
        s.max_blocks_per_q = a_int(d, "max_blocks_per_q", s.max_blocks_per_q);
        s.use_wave_ballot_scatter = a_bool(d, "use_wave_ballot_scatter", s.use_wave_ballot_scatter);

        std::memset(vsa_ctx, 0, sizeof *vsa_ctx);
        vsa_ctx->spec = &s;
        vsa_ctx->arch = arch;
        vsa_ctx->s = s.common;
        if(!rocke_vsa_prologue(vsa_ctx))
            return nullptr;
        rocke_vsa_stage_bitmap(vsa_ctx);
        rocke_kernel_def_t* k = rocke_vsa_emit_body(vsa_ctx);
        if(!k || rocke_ir_builder_status(vsa_ctx->b) != ROCKE_OK)
            return nullptr;
        return k;
    }
    *is_vsa = false;
    rocke_jenga_sparse_spec_t s
        = rocke_jenga_sparse_spec_default(c, a_int(d, "seqlen_q", 0), a_int(d, "seqlen_k", 0));
    s.block_q = a_int(d, "block_q", s.block_q);
    s.block_k = a_int(d, "block_k", s.block_k);

    std::memset(jenga_ctx, 0, sizeof *jenga_ctx);
    jenga_ctx->spec = &s;
    jenga_ctx->arch = arch;
    jenga_ctx->s = s.common;
    if(!rocke_jenga_prologue(jenga_ctx))
        return nullptr;
    rocke_jenga_stage_mask(jenga_ctx);
    rocke_kernel_def_t* k = rocke_jenga_emit_body(jenga_ctx);
    if(!k || rocke_ir_builder_status(jenga_ctx->b) != ROCKE_OK)
        return nullptr;
    return k;
}
/* Free whichever context's FmhaKernelBuilder was populated by sparse_build. Safe
 * to call on the failure path: rocke_fmha_kernel_builder_free tolerates a
 * partially-initialised (zeroed) builder. */
void sparse_free(rocke_jenga_sparse_ctx_t* jenga_ctx, rocke_vsa_sparse_ctx_t* vsa_ctx, bool is_vsa)
{
    if(is_vsa)
        rocke_fmha_kernel_builder_free(&vsa_ctx->kb);
    else
        rocke_fmha_kernel_builder_free(&jenga_ctx->kb);
}
std::string sparse_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    const char* arch_c = arch.c_str();
    rocke_kernel_def_t* k = sparse_build(d, st, arch_c, &jctx, &vctx, &is_vsa);
    if(!k)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw std::runtime_error("rocke_engine.sparse_attention_lower_llvm build failed");
    }
    auto guard
        = rocke_bindings::own_builder(is_vsa ? &vctx.kb : &jctx.kb, rocke_fmha_kernel_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch_c, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.sparse_attention_lower_llvm");
}
std::string sparse_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    rocke_kernel_def_t* k = sparse_build(d, st, arch.c_str(), &jctx, &vctx, &is_vsa);
    if(!k)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw std::runtime_error("rocke_engine.sparse_attention_serialize_ir build failed");
    }
    std::string out;
    try
    {
        out = ser_kernel(k, "rocke_engine.sparse_attention_serialize_ir");
    }
    catch(...)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw;
    }
    sparse_free(&jctx, &vctx, is_vsa);
    return out;
}
std::vector<std::string> sparse_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    rocke_kernel_def_t* k = sparse_build(d, st, arch.c_str(), &jctx, &vctx, &is_vsa);
    if(!k)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw std::runtime_error("rocke_engine.sparse_attention_verify build failed");
    }
    std::vector<std::string> out;
    try
    {
        out = ver_kernel(k);
    }
    catch(...)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw;
    }
    sparse_free(&jctx, &vctx, is_vsa);
    return out;
}

/* ===================== sage_attention (kb + convenience lower) ========== */

rocke_qk_scale_layout_t qkl_of(const std::string& s)
{
    return (s == "per_block") ? ROCKE_QK_SCALE_PER_BLOCK : ROCKE_QK_SCALE_PER_HEAD;
}
void fill_qk_scale(rocke_qk_scale_spec_t* q, const py::dict& d)
{
    std::string lay;
    if(a_str(d, "layout", lay))
        q->layout = qkl_of(lay);
    q->scale_block = a_int(d, "scale_block", q->scale_block);
    q->stride_batch = a_int(d, "stride_batch", q->stride_batch);
    q->stride_head = a_int(d, "stride_head", q->stride_head);
    q->stride_block = a_int(d, "stride_block", q->stride_block);
}
rocke_sage_quant_mode_t sage_mode_of(const std::string& s)
{
    if(s == "fp8_bf16")
        return ROCKE_SAGE_QUANT_FP8_BF16;
    if(s == "i8_fp8_bf16")
        return ROCKE_SAGE_QUANT_I8_FP8_BF16;
    if(s == "i4_fp8_bf16")
        return ROCKE_SAGE_QUANT_I4_FP8_BF16;
    return ROCKE_SAGE_QUANT_FP16_BF16;
}
rocke_sage_attention_spec_t sage_build(const py::dict& d, Store& st)
{
    rocke_sage_attention_spec_t s = rocke_sage_attention_spec_default();
    s.common = common_of(d, st);
    s.seqlen_q = a_int(d, "seqlen_q", s.seqlen_q);
    s.seqlen_k = a_int(d, "seqlen_k", s.seqlen_k);
    {
        std::string qm;
        if(a_str(d, "quant_mode", qm))
            s.quant_mode = sage_mode_of(qm);
    }
    if(d.contains("q_scale") && py::isinstance<py::dict>(d["q_scale"]))
        fill_qk_scale(&s.q_scale, d["q_scale"].cast<py::dict>());
    if(d.contains("k_scale") && py::isinstance<py::dict>(d["k_scale"]))
        fill_qk_scale(&s.k_scale, d["k_scale"].cast<py::dict>());
    return s;
}
std::string sage_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_sage_attention_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.sage_attention_lower_llvm");
}
std::string sage_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_sage_attention_new(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.sage_attention_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.sage_attention_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> sage_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_sage_attention_new(&kb, &s, arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.sage_attention_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ===================== gfx942 / gfx950 tiled-2d ======================== */

/* Python fields of the gfx942 and gfx950 UnifiedAttention2DTiledSpec that the
 * shared C spec struct has no slot for. */
const unported_field k_t2d_unported[] = {
    {"use_additive_bias", false, 0},
    {"causal_top_left", false, 0},
    {"right_bound", false, 0},
};

/* Overlay every spec field present in `d` onto `s` (absent or None keeps the
 * current value), so a dict built with asdict(spec) reaches the builder whole. */
void t2d_read(const py::dict& d, Store& st, rocke_attention_tiled_2d_spec_t& s)
{
    reject_unported(d,
                    "attention_tiled_2d",
                    k_t2d_unported,
                    sizeof(k_t2d_unported) / sizeof(k_t2d_unported[0]));
    s.head_size = a_int(d, "head_size", s.head_size);
    s.block_size = a_int(d, "block_size", s.block_size);
    s.num_query_heads = a_int(d, "num_query_heads", s.num_query_heads);
    s.num_kv_heads = a_int(d, "num_kv_heads", s.num_kv_heads);
    s.use_sinks = a_bool(d, "use_sinks", s.use_sinks);
    s.sliding_window = a_int(d, "sliding_window", s.sliding_window);
    s.has_softcap = a_bool(d, "has_softcap", s.has_softcap);
    s.use_qq_bias = a_bool(d, "use_qq_bias", s.use_qq_bias);
    s.use_alibi = a_bool(d, "use_alibi", s.use_alibi);
    s.num_warps = a_int(d, "num_warps", s.num_warps);
    s.num_seqs = a_int(d, "num_seqs", s.num_seqs);
    if(d.contains("tile_size") && !d["tile_size"].is_none())
    {
        s.has_tile_size = true;
        s.tile_size = d["tile_size"].cast<int>();
    }
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
    s.use_fp8_mfma_qk = a_bool(d, "use_fp8_mfma_qk", s.use_fp8_mfma_qk);
    s.use_fp8_mfma_pv = a_bool(d, "use_fp8_mfma_pv", s.use_fp8_mfma_pv);
    s.use_register_pv = a_bool(d, "use_register_pv", s.use_register_pv);
    s.use_v_double_buffer = a_bool(d, "use_v_double_buffer", s.use_v_double_buffer);
    s.kv_ring_depth = a_int(d, "kv_ring_depth", s.kv_ring_depth);
    s.use_staggered_iter_wait = a_bool(d, "use_staggered_iter_wait", s.use_staggered_iter_wait);
    s.use_q_reread = a_bool(d, "use_q_reread", s.use_q_reread);
    s.use_q_direct_reg = a_bool(d, "use_q_direct_reg", s.use_q_direct_reg);
    s.use_sched_barrier = a_bool(d, "use_sched_barrier", s.use_sched_barrier);
    s.sched_barrier_mask = a_int(d, "sched_barrier_mask", s.sched_barrier_mask);
    s.use_softmax_mfma_interleave
        = a_bool(d, "use_softmax_mfma_interleave", s.use_softmax_mfma_interleave);
    s.softmax_interleave_mode = a_int(d, "softmax_interleave_mode", s.softmax_interleave_mode);
    s.softmax_interleave_groups
        = a_int(d, "softmax_interleave_groups", s.softmax_interleave_groups);
    s.block_m_per_warp = a_int(d, "block_m_per_warp", s.block_m_per_warp);
    s.use_mfma_32x32 = a_bool(d, "use_mfma_32x32", s.use_mfma_32x32);
    s.use_mfma_32x32x8 = a_bool(d, "use_mfma_32x32x8", s.use_mfma_32x32x8);
    s.use_transposed_qk_32x32 = a_bool(d, "use_transposed_qk_32x32", s.use_transposed_qk_32x32);
    s.use_transposed_scalar_state
        = a_bool(d, "use_transposed_scalar_state", s.use_transposed_scalar_state);
    s.use_transposed_invariant_hoist
        = a_bool(d, "use_transposed_invariant_hoist", s.use_transposed_invariant_hoist);
    s.use_transposed_mask_once = a_bool(d, "use_transposed_mask_once", s.use_transposed_mask_once);
    s.use_transposed_half_local_pv
        = a_bool(d, "use_transposed_half_local_pv", s.use_transposed_half_local_pv);
    s.use_mfma32_skip_legacy_qreg
        = a_bool(d, "use_mfma32_skip_legacy_qreg", s.use_mfma32_skip_legacy_qreg);
    s.use_transposed_mask_limit
        = a_bool(d, "use_transposed_mask_limit", s.use_transposed_mask_limit);
    s.use_grouped_kv2_softmax = a_bool(d, "use_grouped_kv2_softmax", s.use_grouped_kv2_softmax);
    s.use_fast_paged_kv_desc = a_bool(d, "use_fast_paged_kv_desc", s.use_fast_paged_kv_desc);
    s.use_i64_kv_addr = a_bool(d, "use_i64_kv_addr", s.use_i64_kv_addr);
    s.use_early_v_schedule = a_bool(d, "use_early_v_schedule", s.use_early_v_schedule);
    s.use_agpr_alloc_zero = a_bool(d, "use_agpr_alloc_zero", s.use_agpr_alloc_zero);
    s.use_conflict_free_v = a_bool(d, "use_conflict_free_v", s.use_conflict_free_v);
    s.use_conflict_free_v_store
        = a_bool(d, "use_conflict_free_v_store", s.use_conflict_free_v_store);
    s.use_conflict_free_v_store_split
        = a_bool(d, "use_conflict_free_v_store_split", s.use_conflict_free_v_store_split);
    s.use_conflict_free_v_ck_vlds
        = a_bool(d, "use_conflict_free_v_ck_vlds", s.use_conflict_free_v_ck_vlds);
    s.use_k_single_buffer = a_bool(d, "use_k_single_buffer", s.use_k_single_buffer);
    s.use_k_sliced_ring = a_bool(d, "use_k_sliced_ring", s.use_k_sliced_ring);
    s.ring_depth = a_int(d, "ring_depth", s.ring_depth);
    s.k_slice_hd = a_int(d, "k_slice_hd", s.k_slice_hd);
    s.use_k_sliced_ldsseq = a_bool(d, "use_k_sliced_ldsseq", s.use_k_sliced_ldsseq);
    s.use_iglp_opt = a_bool(d, "use_iglp_opt", s.use_iglp_opt);
    s.use_qk_pv_sched_group_barrier
        = a_bool(d, "use_qk_pv_sched_group_barrier", s.use_qk_pv_sched_group_barrier);
    s.use_q_direct_global = a_bool(d, "use_q_direct_global", s.use_q_direct_global);
    s.use_global_load_lds_k = a_bool(d, "use_global_load_lds_k", s.use_global_load_lds_k);
    s.use_q_major_grid = a_bool(d, "use_q_major_grid", s.use_q_major_grid);
    s.use_kq_lds_pad = a_bool(d, "use_kq_lds_pad", s.use_kq_lds_pad);
    s.kq_lds_pad_halves = a_int(d, "kq_lds_pad_halves", s.kq_lds_pad_halves);
    s.use_mask_phase_split = a_bool(d, "use_mask_phase_split", s.use_mask_phase_split);
    s.use_v_hbm_direct = a_bool(d, "use_v_hbm_direct", s.use_v_hbm_direct);
    s.use_k_hbm_direct = a_bool(d, "use_k_hbm_direct", s.use_k_hbm_direct);
    s.use_causal_mask_phase_split
        = a_bool(d, "use_causal_mask_phase_split", s.use_causal_mask_phase_split);
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    if(a_str(d, "kv_storage_dtype", v))
        s.kv_storage_dtype = st.keep(v);
    if(a_str(d, "kv_cache_policy", v))
        s.kv_cache_policy = st.keep(v);
}

rocke_attention_tiled_2d_spec_t t2d_build(const py::dict& d, Store& st)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    t2d_read(d, st, s);
    return s;
}

std::string t942_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.c_str();
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, resolve_python_llvm_flavor(), a, &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.gfx942_attention_tiled_2d_lower_llvm");
}
std::string t942_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.gfx942_attention_tiled_2d_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> t942_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* build_gfx942_4warp_gqa over the shared tiled-2D spec. Throws on a build
 * failure; on success the caller owns (and frees) b. */
rocke_kernel_def_t* t942_4w_build(const py::dict& d, const std::string& arch, rocke_ir_builder_t* b)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_kernel_def_t* k = rocke_build_gfx942_4warp_gqa(b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(b))
    {
        std::string m
            = std::string("gfx942_attention_4warp_gqa build failed: ") + rocke_ir_builder_error(b);
        rocke_ir_builder_free(b);
        throw std::runtime_error(m);
    }
    return k;
}
std::string t942_4w_lower(const py::dict& d, const std::string& arch)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = t942_4w_build(d, arch, &b);
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.gfx942_attention_4warp_gqa_lower_llvm");
}
std::string t942_4w_serialize(const py::dict& d, const std::string& arch)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = t942_4w_build(d, arch, &b);
    std::string out = ser_kernel(k, "rocke_engine.gfx942_attention_4warp_gqa_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> t942_4w_verify(const py::dict& d, const std::string& arch)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = t942_4w_build(d, arch, &b);
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

std::string t950_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.c_str();
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, resolve_python_llvm_flavor(), a, &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.gfx950_attention_tiled_2d_lower_llvm");
}
std::string t950_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.gfx950_attention_tiled_2d_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> t950_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ===================== gfx950 tiled-2d fastkv_regp ===================== */

rocke_attention_tiled_2d_spec_t fkv_build(const py::dict& d, Store& st)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    /* make_base() defaults from the standalone emitter */
    s.head_size = 64;
    s.block_size = 32;
    s.num_query_heads = 64;
    s.num_kv_heads = 8;
    s.dtype = "bf16";
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    s.num_warps = 4;
    s.has_waves_per_eu = true;
    s.waves_per_eu = 2;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.block_m_per_warp = 32;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    s.use_transposed_scalar_state = true;
    s.use_transposed_mask_once = true;
    s.use_fast_paged_kv_desc = true;
    t2d_read(d, st, s);
    return s;
}
std::string fkv_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.c_str();
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_fastkv_register_p(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("gfx950_attention_tiled_2d_fastkv_regp build failed: ")
                        + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, resolve_python_llvm_flavor(), a, &ll, err, sizeof err);
    guard.reset();
    return take_ll(s2, ll, err, "rocke_engine.gfx950_attention_tiled_2d_fastkv_regp_lower_llvm");
}
std::string fkv_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k
        = rocke_build_unified_attention_2d_fastkv_register_p(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("gfx950_attention_tiled_2d_fastkv_regp build failed: ")
                        + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out
        = ser_kernel(k, "rocke_engine.gfx950_attention_tiled_2d_fastkv_regp_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> fkv_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k
        = rocke_build_unified_attention_2d_fastkv_register_p(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("gfx950_attention_tiled_2d_fastkv_regp build failed: ")
                        + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ===================== gfx942 / gfx950 tiled-3d ======================== */

/* gfx950 UnifiedAttention3DTiledSpec fields the shared C struct has no slot
 * for (kv_layout is read by t3d_strided). */
const unported_field k_t3d_unported[] = {
    {"use_additive_bias", false, 0},
    {"causal_top_left", false, 0},
    {"right_bound", false, 0},
};

rocke_unified_attention_3d_tiled_spec_t t3d_build(const py::dict& d, Store& st)
{
    reject_unported(d,
                    "attention_tiled_3d",
                    k_t3d_unported,
                    sizeof(k_t3d_unported) / sizeof(k_t3d_unported[0]));
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = a_int(d, "head_size", s.head_size);
    s.block_size = a_int(d, "block_size", s.block_size);
    s.num_query_heads = a_int(d, "num_query_heads", s.num_query_heads);
    s.num_kv_heads = a_int(d, "num_kv_heads", s.num_kv_heads);
    s.num_segments = a_int(d, "num_segments", s.num_segments);
    s.sliding_window = a_int(d, "sliding_window", s.sliding_window);
    s.use_sinks = a_bool(d, "use_sinks", s.use_sinks);
    s.has_softcap = a_bool(d, "has_softcap", s.has_softcap);
    s.use_alibi = a_bool(d, "use_alibi", s.use_alibi);
    s.use_qq_bias = a_bool(d, "use_qq_bias", s.use_qq_bias);
    s.num_seqs = a_int(d, "num_seqs", s.num_seqs);
    s.use_invariant_hoist = a_bool(d, "use_invariant_hoist", s.use_invariant_hoist);
    s.use_wide_kv_load = a_bool(d, "use_wide_kv_load", s.use_wide_kv_load);
    s.use_i64_kv_addr = a_bool(d, "use_i64_kv_addr", s.use_i64_kv_addr);
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = a_int(d, "waves_per_eu", 0);
    }
    if(d.contains("tile_size_override") && !d["tile_size_override"].is_none())
    {
        s.has_tile_size_override = true;
        s.tile_size_override = a_int(d, "tile_size_override", 0);
    }
    s.use_decode_grid = a_bool(d, "use_decode_grid", s.use_decode_grid);
    s.use_kv_lds_swizzle = a_bool(d, "use_kv_lds_swizzle", s.use_kv_lds_swizzle);
    s.use_wave_page_id_load = a_bool(d, "use_wave_page_id_load", s.use_wave_page_id_load);
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    if(a_str(d, "kv_storage_dtype", v))
        s.kv_storage_dtype = st.keep(v);
    if(a_str(d, "kv_cache_policy", v))
        s.kv_cache_policy = st.keep(v);
    return s;
}

bool t3d_strided(const py::dict& d)
{
    std::string layout = "paged";
    a_str(d, "kv_layout", layout);
    if(layout != "paged" && layout != "strided")
        throw py::value_error("kv_layout must be 'paged' or 'strided'");
    return layout == "strided";
}

/* gfx942 3d: emitter lowers only the segment via the family convenience. */
std::string t3d942_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const char* target = arch.c_str();
    if(!t3d_strided(d))
    {
        rocke_status_t status = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
            &s, target, resolve_python_llvm_flavor(), &ll, err, sizeof err);
        return take_ll(status, ll, err, "rocke_engine.gfx942_attention_tiled_3d_lower_llvm");
    }
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
    rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_strided_gfx942(&b, &s, target);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string message = rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(message);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    rocke_status_t status = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), target, &ll, err, sizeof err);
    guard.reset();
    return take_ll(status, ll, err, "rocke_engine.gfx942_attention_tiled_3d_lower_llvm");
}
std::string t3d942_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    rocke_ir_builder_t b;
    char kname[512];
    kname[0] = '\0';
    rocke_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k
        = (t3d_strided(d) ? rocke_build_unified_attention_3d_tiled_strided_gfx942
                          : rocke_build_unified_attention_3d_tiled_gfx942)(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_3d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.gfx942_attention_tiled_3d_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> t3d942_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    rocke_ir_builder_t b;
    char kname[512];
    kname[0] = '\0';
    rocke_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k
        = (t3d_strided(d) ? rocke_build_unified_attention_3d_tiled_strided_gfx942
                          : rocke_build_unified_attention_3d_tiled_gfx942)(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_3d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* gfx950 3d: emitter lowers segment then reduce, concatenated. */
std::string t3d950_lower(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.c_str();
    std::string out;
    /* segment */
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
        rocke_kernel_def_t* k
            = (t3d_strided(d) ? rocke_build_unified_attention_3d_tiled_strided_gfx950
                              : rocke_build_unified_attention_3d_tiled_gfx950)(&b, &s, a);
        if(!k || !rocke_ir_builder_ok(&b))
        {
            std::string m = std::string("gfx950_attention_tiled_3d segment build failed: ")
                            + rocke_ir_builder_error(&b);
            rocke_ir_builder_free(&b);
            throw std::runtime_error(m);
        }
        auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
            k, resolve_python_llvm_flavor(), a, &ll, err, sizeof err);
        out += take_ll(s2, ll, err, "rocke_engine.gfx950_attention_tiled_3d_lower_llvm");
        guard.reset();
    }
    /* reduce: derived from the segment spec, exactly as the standalone emitter */
    {
        rocke_unified_attention_reduce_tiled_spec_t r
            = rocke_unified_attention_reduce_tiled_spec_default();
        r.head_size = s.head_size;
        r.num_query_heads = s.num_query_heads;
        r.num_kv_heads = s.num_kv_heads;
        r.dtype = s.dtype;
        r.num_segments = s.num_segments;
        r.has_waves_per_eu = s.has_waves_per_eu;
        r.waves_per_eu = s.waves_per_eu;
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "attention_tiled_3d_reduce");
        rocke_kernel_def_t* k = rocke_build_unified_attention_reduce_tiled_gfx950(&b, &r, a);
        if(!k || !rocke_ir_builder_ok(&b))
        {
            std::string m = std::string("gfx950_attention_tiled_3d reduce build failed: ")
                            + rocke_ir_builder_error(&b);
            rocke_ir_builder_free(&b);
            throw std::runtime_error(m);
        }
        auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
            k, resolve_python_llvm_flavor(), a, &ll, err, sizeof err);
        out += take_ll(s2, ll, err, "rocke_engine.gfx950_attention_tiled_3d_lower_llvm");
        guard.reset();
    }
    return out;
}
std::string t3d950_serialize(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.c_str();
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
    rocke_kernel_def_t* k
        = (t3d_strided(d) ? rocke_build_unified_attention_3d_tiled_strided_gfx950
                          : rocke_build_unified_attention_3d_tiled_gfx950)(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_3d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.gfx950_attention_tiled_3d_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> t3d950_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.c_str();
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
    rocke_kernel_def_t* k
        = (t3d_strided(d) ? rocke_build_unified_attention_3d_tiled_strided_gfx950
                          : rocke_build_unified_attention_3d_tiled_gfx950)(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_3d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

void reg3(py::module_& m,
          const char* prefix,
          std::string (*lower)(const py::dict&, const std::string&),
          std::string (*ser)(const py::dict&, const std::string&),
          std::vector<std::string> (*ver)(const py::dict&, const std::string&))
{
    m.def((std::string(prefix) + "_lower_llvm").c_str(), lower, py::arg("spec"), py::arg("arch"));
    m.def((std::string(prefix) + "_serialize_ir").c_str(), ser, py::arg("spec"), py::arg("arch"));
    m.def((std::string(prefix) + "_verify").c_str(), ver, py::arg("spec"), py::arg("arch"));
}

} // namespace

/* ===================== spec_from_problem ===================== */

/* UnifiedAttentionProblem fields the C++ selectors do not read yet; each one
 * steers selection in Python, so accepting them would silently diverge. */
static const unported_field k_problem_unported[] = {
    {"use_additive_bias", false, 0},
    {"fp8_fnuz", false, 0},
    {"target_ctas", false, 0},
    {"compile_backend", true, 0},
    {"causal_top_left", false, 0},
    {"right_bound", false, 0},
};

/* The C problem struct has no slot for these: dropping one would answer for a
 * different problem, so every problem binding refuses it. emit_lse steers the
 * support gates and the split-KV reduce spec (known_gaps.md section 7). */
static const unported_field k_problem_no_c_field[] = {
    {"emit_lse", false, 0},
};

/* Convert every field of a Python UnifiedAttentionProblem dict to the C
 * struct. */
static rocke_unified_attention_problem_t problem_fields_from_dict(const py::dict& d, Store& st)
{
    reject_unported(d,
                    "unified_attention_problem",
                    k_problem_no_c_field,
                    sizeof(k_problem_no_c_field) / sizeof(k_problem_no_c_field[0]));
    rocke_unified_attention_problem_t p{};
    p.total_q = a_int(d, "total_q", 0);
    p.num_seqs = a_int(d, "num_seqs", 0);
    p.num_query_heads = a_int(d, "num_query_heads", 0);
    p.num_kv_heads = a_int(d, "num_kv_heads", 0);
    p.head_size = a_int(d, "head_size", 0);
    p.block_size = a_int(d, "block_size", 0);
    p.max_seqlen_q = a_int(d, "max_seqlen_q", 0);
    p.max_seqlen_k = a_int(d, "max_seqlen_k", 0);
    p.sliding_window = a_int(d, "sliding_window", 0);
    p.softcap = a_double(d, "softcap", 0.0);
    p.use_sinks = a_bool(d, "use_sinks", false);
    p.use_alibi = a_bool(d, "use_alibi", false);
    p.use_qq_bias = a_bool(d, "use_qq_bias", false);
    p.use_additive_bias = a_bool(d, "use_additive_bias", false);
    p.use_fp8 = a_bool(d, "use_fp8", false);
    p.fp8_fnuz = a_bool(d, "fp8_fnuz", false);
    p.num_cus = a_int(d, "num_cus", 120);
    p.target_ctas = a_int(d, "target_ctas", 0);
    p.waves_per_eu_set = d.contains("waves_per_eu") && !d["waves_per_eu"].is_none();
    p.waves_per_eu = a_int(d, "waves_per_eu", 0);
    p.num_kv_blocks = a_int(d, "num_kv_blocks", 0);
    p.causal_top_left = a_bool(d, "causal_top_left", false);
    p.right_bound = a_int(d, "right_bound", 0);
    std::string v;
    if(a_str(d, "dtype", v))
        p.dtype = st.keep(v);
    else
        p.dtype = "bf16";
    if(a_str(d, "q_dtype", v))
        p.q_dtype = st.keep(v);
    if(a_str(d, "compile_backend", v))
        p.compile_backend = st.keep(v);
    return p;
}

/* As problem_fields_from_dict, for the tiled spec selector: rejects the
 * fields it does not read yet. */
static rocke_unified_attention_problem_t problem_dict_to_c(const py::dict& d, Store& st)
{
    reject_unported(d,
                    "attn_tiled_spec_from_problem",
                    k_problem_unported,
                    sizeof(k_problem_unported) / sizeof(k_problem_unported[0]));
    return problem_fields_from_dict(d, st);
}

/* Convert a rocke_attention_tiled_2d_spec_t back to a Python dict so the
 * parity harness can compare specs field-by-field. Every field of the C struct
 * is returned, under its Python UnifiedAttention2DTiledSpec name. */
static py::dict spec_to_dict(const rocke_attention_tiled_2d_spec_t& s)
{
    py::dict d;
    d["head_size"] = s.head_size;
    d["block_size"] = s.block_size;
    d["num_query_heads"] = s.num_query_heads;
    d["num_kv_heads"] = s.num_kv_heads;
    d["dtype"] = s.dtype ? std::string(s.dtype) : std::string("");
    d["use_sinks"] = s.use_sinks;
    d["sliding_window"] = s.sliding_window;
    d["has_softcap"] = s.has_softcap;
    d["use_alibi"] = s.use_alibi;
    d["use_qq_bias"] = s.use_qq_bias;
    d["num_seqs"] = s.num_seqs;
    d["num_warps"] = s.num_warps;
    d["waves_per_eu"] = s.has_waves_per_eu ? py::cast(s.waves_per_eu) : py::none();
    d["kv_storage_dtype"]
        = s.kv_storage_dtype ? py::cast(std::string(s.kv_storage_dtype)) : py::none();
    d["use_fp8_mfma_qk"] = s.use_fp8_mfma_qk;
    d["use_fp8_mfma_pv"] = s.use_fp8_mfma_pv;
    d["use_register_pv"] = s.use_register_pv;
    d["use_v_double_buffer"] = s.use_v_double_buffer;
    d["kv_ring_depth"] = s.kv_ring_depth;
    d["use_staggered_iter_wait"] = s.use_staggered_iter_wait;
    d["use_q_reread"] = s.use_q_reread;
    d["use_q_direct_reg"] = s.use_q_direct_reg;
    d["use_sched_barrier"] = s.use_sched_barrier;
    d["sched_barrier_mask"] = s.sched_barrier_mask;
    d["use_softmax_mfma_interleave"] = s.use_softmax_mfma_interleave;
    d["softmax_interleave_mode"] = s.softmax_interleave_mode;
    d["softmax_interleave_groups"] = s.softmax_interleave_groups;
    d["tile_size"] = s.has_tile_size ? py::cast(s.tile_size) : py::none();
    d["block_m_per_warp"] = s.block_m_per_warp;
    d["use_mfma_32x32"] = s.use_mfma_32x32;
    d["use_mfma_32x32x8"] = s.use_mfma_32x32x8;
    d["use_transposed_qk_32x32"] = s.use_transposed_qk_32x32;
    d["use_transposed_scalar_state"] = s.use_transposed_scalar_state;
    d["use_transposed_invariant_hoist"] = s.use_transposed_invariant_hoist;
    d["use_transposed_mask_once"] = s.use_transposed_mask_once;
    d["use_transposed_half_local_pv"] = s.use_transposed_half_local_pv;
    d["use_mfma32_skip_legacy_qreg"] = s.use_mfma32_skip_legacy_qreg;
    d["use_transposed_mask_limit"] = s.use_transposed_mask_limit;
    d["use_grouped_kv2_softmax"] = s.use_grouped_kv2_softmax;
    d["use_fast_paged_kv_desc"] = s.use_fast_paged_kv_desc;
    d["use_i64_kv_addr"] = s.use_i64_kv_addr;
    d["use_early_v_schedule"] = s.use_early_v_schedule;
    d["use_agpr_alloc_zero"] = s.use_agpr_alloc_zero;
    d["use_conflict_free_v"] = s.use_conflict_free_v;
    d["use_conflict_free_v_store"] = s.use_conflict_free_v_store;
    d["use_conflict_free_v_store_split"] = s.use_conflict_free_v_store_split;
    d["use_conflict_free_v_ck_vlds"] = s.use_conflict_free_v_ck_vlds;
    d["use_k_single_buffer"] = s.use_k_single_buffer;
    d["use_k_sliced_ring"] = s.use_k_sliced_ring;
    d["ring_depth"] = s.ring_depth;
    d["k_slice_hd"] = s.k_slice_hd;
    d["use_k_sliced_ldsseq"] = s.use_k_sliced_ldsseq;
    d["use_iglp_opt"] = s.use_iglp_opt;
    d["use_qk_pv_sched_group_barrier"] = s.use_qk_pv_sched_group_barrier;
    d["use_q_direct_global"] = s.use_q_direct_global;
    d["kv_cache_policy"]
        = s.kv_cache_policy ? py::cast(std::string(s.kv_cache_policy)) : py::none();
    d["use_global_load_lds_k"] = s.use_global_load_lds_k;
    d["use_q_major_grid"] = s.use_q_major_grid;
    d["use_kq_lds_pad"] = s.use_kq_lds_pad;
    d["kq_lds_pad_halves"] = s.kq_lds_pad_halves;
    d["use_mask_phase_split"] = s.use_mask_phase_split;
    d["use_v_hbm_direct"] = s.use_v_hbm_direct;
    d["use_k_hbm_direct"] = s.use_k_hbm_direct;
    d["use_causal_mask_phase_split"] = s.use_causal_mask_phase_split;
    return d;
}

/* Run a spec validator (the port of a spec class __post_init__) and raise what
 * Python would: NotImplementedError for ROCKE_ERR_NOTIMPL, else ValueError. */
template <class Fn>
static void raise_if_spec_invalid(const char* who, Fn&& validate)
{
    rocke_ir_builder_t b;
    rocke_status_t code = ROCKE_OK;
    std::string err;
    if(rocke_ir_builder_init(&b, who) != ROCKE_OK)
        throw std::runtime_error(std::string(who) + ": builder init failed");
    try
    {
        if(!validate(&b))
        {
            const char* m = rocke_ir_builder_error(&b);
            code = rocke_ir_builder_status(&b);
            err = (m != nullptr && *m != '\0') ? m : "spec validation failed";
        }
    }
    catch(const ckc::Error& e)
    {
        code = e.code();
        err = e.what();
    }
    rocke_ir_builder_free(&b);
    if(err.empty())
        return;
    if(code == ROCKE_ERR_NOTIMPL)
        raise_not_implemented(err);
    throw py::value_error(err);
}

/* Python attention_unified._tiled_spec_from_problem: build the spec (Python
 * validates it in the spec class __post_init__), then resolve its LDS budget.
 * ``hook_arch`` is the one arch the calling hook serves. */
py::dict attn_tiled_spec_from_problem(const py::dict& d,
                                      const std::string& arch,
                                      const std::string& hook_arch)
{
    require_arch(arch);
    if(arch != hook_arch)
        raise_not_implemented("attn_tiled_spec_from_problem: this hook builds only the " + hook_arch
                              + " spec (got " + arch + ")");
    Store st;
    rocke_unified_attention_problem_t p = problem_dict_to_c(d, st);
    const char* a = arch.c_str();
    rocke_attention_tiled_2d_spec_t spec = rocke_unified_attn_tiled_spec_from_problem(&p, a);
    if(arch == "gfx942")
        raise_if_spec_invalid("attn_tiled_spec_from_problem", [&](rocke_ir_builder_t* b) {
            return rocke_attention_tiled_2d_spec_validate(b, &spec);
        });
    char reason[512] = {};
    if(!rocke_unified_attn_resolve_lds_budget(&spec, a, reason, sizeof(reason)))
        throw std::runtime_error(reason);
    return spec_to_dict(spec);
}

/* Every public tiled-2D selector on one problem, keyed by the Python selector's
 * name without its leading underscore, so a parity test can compare each one
 * with Python on any arch, not only the arches C++ assembles a spec for. */
static py::dict tiled_2d_selectors(const py::dict& d, const std::string& arch)
{
    Store st;
    const rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
    const rocke_unified_attention_problem_t* pp = &p;
    const char* a = arch.c_str();
    py::dict r;
    r["select_2d_tile_size"] = rocke_unified_attn_select_2d_tile_size(pp, a);
    r["select_2d_num_warps"] = rocke_unified_attn_select_2d_num_warps(pp, a);
    r["select_2d_block_m_per_warp"] = rocke_unified_attn_select_2d_block_m_per_warp(pp, a);
    int wpe = 0;
    r["select_2d_waves_per_eu"]
        = rocke_unified_attn_select_2d_waves_per_eu(pp, a, &wpe) ? py::cast(wpe) : py::none();
    const char* kv = rocke_unified_attn_kv_storage_dtype(pp);
    r["kv_storage_dtype"] = kv ? py::cast(std::string(kv)) : py::none();

    r["enable_combo_2d"] = rocke_unified_attn_enable_combo_2d(pp, a);
    r["enable_transposed_qk_32x32"] = rocke_unified_attn_enable_transposed_qk_32x32(pp, a);
    r["enable_mfma_32x32"] = rocke_unified_attn_enable_mfma_32x32(pp, a);
    r["enable_transposed_half_local_pv"]
        = rocke_unified_attn_enable_transposed_half_local_pv(pp, a);
    r["enable_register_pv"] = rocke_unified_attn_enable_register_pv(pp, a);
    r["enable_single_batch_combo"] = rocke_unified_attn_enable_single_batch_combo(pp, a);
    r["enable_transposed_subflags"] = rocke_unified_attn_enable_transposed_subflags(pp, a);
    r["enable_d128_small_tile"] = rocke_unified_attn_enable_d128_small_tile(pp, a);
    r["enable_softmax_mfma_interleave"] = rocke_unified_attn_enable_softmax_mfma_interleave(pp, a);
    r["enable_v_double_buffer"] = rocke_unified_attn_enable_v_double_buffer(pp, a);
    r["enable_early_v_schedule"] = rocke_unified_attn_enable_early_v_schedule(pp, a);

    r["gfx942_4warp_fast"] = rocke_unified_attn_gfx942_4warp_fast(pp, a);
    r["enable_gfx942_fp16_flash"] = rocke_unified_attn_enable_gfx942_fp16_flash(pp, a);
    r["enable_gfx942_bf16_flash"] = rocke_unified_attn_enable_gfx942_bf16_flash(pp, a);
    r["enable_gfx942_d128_smalltile_doublek"]
        = rocke_unified_attn_enable_gfx942_d128_smalltile_doublek(pp, a);
    r["gfx942_bf16_wide_tile_size"] = rocke_unified_attn_gfx942_bf16_wide_tile_size(pp, a);
    int nw = 0;
    bool single_k = false;
    rocke_unified_attn_gfx942_bf16_wide_geometry(pp, a, &nw, &single_k);
    r["gfx942_bf16_wide_geometry"] = py::make_tuple(nw, single_k);
    r["gfx942_bf16_wide_use_cfvst"] = rocke_unified_attn_gfx942_bf16_wide_use_cfvst(pp);
    r["gfx942_flash_use_cfvst"] = rocke_unified_attn_gfx942_flash_use_cfvst(pp);
    r["gfx942_flash_use_single_buffer"] = rocke_unified_attn_gfx942_flash_use_single_buffer(pp);
    r["enable_gfx942_flash_mask_limit"] = rocke_unified_attn_enable_gfx942_flash_mask_limit(pp, a);
    r["enable_gfx942_flash_k_sliced_ring"]
        = rocke_unified_attn_enable_gfx942_flash_k_sliced_ring(pp, a);
    r["select_gfx942_flash_ring_depth"] = rocke_unified_attn_select_gfx942_flash_ring_depth(pp, a);
    r["select_gfx942_flash_k_slice_hd"] = rocke_unified_attn_select_gfx942_flash_k_slice_hd(pp);
    r["select_gfx942_flash_num_warps"] = rocke_unified_attn_select_gfx942_flash_num_warps(pp, a);
    return r;
}

/* UnifiedAttentionProblem fields the C 3D spec has no slot for (they set the
 * gfx950-only spec fields of the same name). */
static const unported_field k_problem_3d_unported[] = {
    {"use_additive_bias", false, 0},
    {"causal_top_left", false, 0},
    {"right_bound", false, 0},
};

static py::dict spec_3d_to_dict(const rocke_unified_attention_3d_tiled_spec_t& s)
{
    py::dict d;
    d["head_size"] = s.head_size;
    d["block_size"] = s.block_size;
    d["num_query_heads"] = s.num_query_heads;
    d["num_kv_heads"] = s.num_kv_heads;
    d["dtype"] = s.dtype ? std::string(s.dtype) : std::string("");
    d["use_sinks"] = s.use_sinks;
    d["sliding_window"] = s.sliding_window;
    d["has_softcap"] = s.has_softcap;
    d["num_segments"] = s.num_segments;
    d["use_alibi"] = s.use_alibi;
    d["use_qq_bias"] = s.use_qq_bias;
    d["num_seqs"] = s.num_seqs;
    d["waves_per_eu"] = s.has_waves_per_eu ? py::cast(s.waves_per_eu) : py::none();
    d["kv_storage_dtype"]
        = s.kv_storage_dtype ? py::cast(std::string(s.kv_storage_dtype)) : py::none();
    d["tile_size_override"]
        = s.has_tile_size_override ? py::cast(s.tile_size_override) : py::none();
    d["use_invariant_hoist"] = s.use_invariant_hoist;
    d["use_wide_kv_load"] = s.use_wide_kv_load;
    d["use_i64_kv_addr"] = s.use_i64_kv_addr;
    d["use_decode_grid"] = s.use_decode_grid;
    d["kv_cache_policy"]
        = s.kv_cache_policy ? py::cast(std::string(s.kv_cache_policy)) : py::none();
    d["use_kv_lds_swizzle"] = s.use_kv_lds_swizzle;
    d["use_wave_page_id_load"] = s.use_wave_page_id_load;
    return d;
}

/* Python attention_unified._tiled_3d_spec_from_problem: build the 3D spec (Python
 * validates it in the spec class __post_init__). ``hook_arch`` is the one arch
 * the calling hook serves. */
static py::dict attn_tiled_3d_spec_from_problem(const py::dict& d,
                                                const std::string& arch,
                                                const std::string& hook_arch)
{
    require_arch(arch);
    if(arch != hook_arch)
        raise_not_implemented("attn_tiled_3d_spec_from_problem: this hook builds only the "
                              + hook_arch + " spec (got " + arch + ")");
    reject_unported(d,
                    "attn_tiled_3d_spec_from_problem",
                    k_problem_3d_unported,
                    sizeof(k_problem_3d_unported) / sizeof(k_problem_3d_unported[0]));
    Store st;
    rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
    if(p.num_kv_heads <= 0 || p.num_query_heads % p.num_kv_heads != 0)
        throw py::value_error("num_query_heads must be divisible by num_kv_heads");
    rocke_unified_attention_3d_tiled_spec_t spec
        = rocke_unified_attn_tiled_3d_spec_from_problem(&p, arch.c_str());
    raise_if_spec_invalid("attn_tiled_3d_spec_from_problem", [&](rocke_ir_builder_t* b) {
        return arch == "gfx950" ? rocke_gfx950_unified_attention_3d_tiled_spec_validate(b, &spec)
                                : rocke_unified_attention_3d_tiled_spec_validate(b, &spec);
    });
    return spec_3d_to_dict(spec);
}

static py::dict config_2d_to_dict(const rocke_attention_2d_config_t& c)
{
    py::dict d;
    d["BLOCK_M"] = c.block_m;
    d["BLOCK_Q"] = c.block_q;
    d["TILE_SIZE"] = c.tile_size;
    d["num_warps"] = c.num_warps;
    d["num_stages"] = c.num_stages;
    d["waves_per_eu"] = c.waves_per_eu;
    return d;
}

static py::dict config_3d_to_dict(const rocke_attention_3d_config_t& c)
{
    py::dict d;
    d["TILE_SIZE"] = c.tile_size;
    d["NUM_SEGMENTS_PER_SEQ"] = c.num_segments_per_seq;
    d["num_warps"] = c.num_warps;
    d["num_stages"] = c.num_stages;
    d["waves_per_eu"] = c.waves_per_eu;
    return d;
}

[[noreturn]] static void raise_bad_problem(const char* fn)
{
    throw py::value_error(std::string(fn)
                          + ": invalid problem (num_query_heads must be divisible by "
                            "num_kv_heads, and the 2D program count must be positive)");
}

static py::tuple supports_result(bool ok, const char* reason)
{
    return py::make_tuple(ok, std::string(reason));
}

void register_attention(py::module_& m)
{
    reg3(m, "attention_unified", &au_lower, &au_serialize, &au_verify);
    reg3(m, "fmha_mfma", &fmma_lower, &fmma_serialize, &fmma_verify);
    reg3(m, "fmha_fwd_fp8", &fp8_lower, &fp8_serialize, &fp8_verify);
    reg3(m, "fmha_bwd", &bwd_lower, &bwd_serialize, &bwd_verify);
    reg3(m, "fmha_head_grouping", &hg_lower, &hg_serialize, &hg_verify);
    reg3(m, "sparse_attention", &sparse_lower, &sparse_serialize, &sparse_verify);
    reg3(m, "sage_attention", &sage_lower, &sage_serialize, &sage_verify);
    reg3(m, "gfx942_attention_tiled_2d", &t942_lower, &t942_serialize, &t942_verify);
    reg3(m, "gfx942_attention_4warp_gqa", &t942_4w_lower, &t942_4w_serialize, &t942_4w_verify);
    reg3(m, "gfx950_attention_tiled_2d", &t950_lower, &t950_serialize, &t950_verify);
    reg3(m, "gfx950_attention_tiled_2d_fastkv_regp", &fkv_lower, &fkv_serialize, &fkv_verify);
    reg3(m, "gfx942_attention_tiled_3d", &t3d942_lower, &t3d942_serialize, &t3d942_verify);
    reg3(m, "gfx950_attention_tiled_3d", &t3d950_lower, &t3d950_serialize, &t3d950_verify);

    m.def(
        "gfx950_attn_tiled_spec_from_problem",
        [](const py::dict& d, const std::string& arch) {
            return attn_tiled_spec_from_problem(d, arch, "gfx950");
        },
        py::arg("problem"),
        py::arg("arch"),
        "Run the gfx950 tiled-2D selector on a UnifiedAttentionProblem dict "
        "and return the assembled spec as a dict (mirrors "
        "attention_unified._tiled_spec_from_problem).");
    m.def(
        "gfx942_attn_tiled_spec_from_problem",
        [](const py::dict& d, const std::string& arch) {
            return attn_tiled_spec_from_problem(d, arch, "gfx942");
        },
        py::arg("problem"),
        py::arg("arch"),
        "Run the gfx942 tiled-2D selector on a UnifiedAttentionProblem dict "
        "and return the assembled spec as a dict (mirrors "
        "attention_unified._tiled_spec_from_problem: raises ValueError where the "
        "spec fails validation, RuntimeError where its LDS budget cannot be met).");
    m.def("unified_attention_tiled_2d_selectors",
          &tiled_2d_selectors,
          py::arg("problem"),
          py::arg("arch"),
          "Every public C++ tiled-2D selector on a UnifiedAttentionProblem dict, "
          "keyed by the Python selector name without its leading underscore.");

    for(const char* a : {"gfx942", "gfx950"})
    {
        std::string hook_arch(a);
        m.def((hook_arch + "_attn_tiled_3d_spec_from_problem").c_str(),
              [hook_arch](const py::dict& d, const std::string& arch) {
                  return attn_tiled_3d_spec_from_problem(d, arch, hook_arch);
              },
              py::arg("problem"),
              py::arg("arch"),
              "Run the tiled-3D selectors on a UnifiedAttentionProblem dict and return "
              "the assembled spec as a dict (mirrors "
              "attention_unified._tiled_3d_spec_from_problem).");
    }
    m.def(
        "unified_attention_select_path",
        [](const py::dict& d) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            const char* path = rocke_unified_attention_problem_select_path(&p);
            if(path == NULL)
                raise_bad_problem("unified_attention_select_path");
            return std::string(path);
        },
        py::arg("problem"),
        "UnifiedAttentionProblem.select_path() on a problem dict: '2d' or '3d'.");
    m.def(
        "unified_attention_select_2d",
        [](const py::dict& d) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            rocke_attention_2d_config_t c{};
            if(!rocke_unified_attention_problem_select_2d(&p, &c))
                raise_bad_problem("unified_attention_select_2d");
            return config_2d_to_dict(c);
        },
        py::arg("problem"),
        "UnifiedAttentionProblem.select_2d() as an Attention2DConfig field dict.");
    m.def(
        "unified_attention_select_3d",
        [](const py::dict& d) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            rocke_attention_3d_config_t attn{};
            rocke_attention_3d_config_t reduce{};
            if(!rocke_unified_attention_problem_select_3d(&p, &attn, &reduce))
                raise_bad_problem("unified_attention_select_3d");
            return py::make_tuple(config_3d_to_dict(attn), config_3d_to_dict(reduce));
        },
        py::arg("problem"),
        "UnifiedAttentionProblem.select_3d() as two Attention3DConfig field dicts.");
    m.def(
        "unified_attention_supports_scalar",
        [](const py::dict& d, const std::string& arch) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            char reason[1024] = {};
            bool ok
                = rocke_unified_attention_supports_scalar(&p, arch.c_str(), reason, sizeof(reason));
            return supports_result(ok, reason);
        },
        py::arg("problem"),
        py::arg("arch"),
        "supports_native_unified_attention(problem, arch) -> (ok, reason).");
    m.def(
        "unified_attention_supports_tiled_2d",
        [](const py::dict& d, const std::string& arch) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            char reason[1024] = {};
            bool ok = rocke_unified_attention_supports_tiled_2d(
                &p, arch.c_str(), reason, sizeof(reason));
            return supports_result(ok, reason);
        },
        py::arg("problem"),
        py::arg("arch"),
        "supports_native_unified_attention_tiled(problem, arch) -> (ok, reason).");
    m.def(
        "gfx942_attention_tiled_2d_supports_spec",
        [](const py::dict& d, const std::string& arch) {
            Store st;
            rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
            char reason[1024] = {};
            bool ok = rocke_gfx942_attention_tiled_2d_supports_spec(
                &s, arch.c_str(), reason, sizeof(reason));
            return supports_result(ok, reason);
        },
        py::arg("spec"),
        py::arg("arch"),
        "gfx942 supports_tiled_2d_spec(spec, arch) -> (ok, reason).");
    m.def(
        "gfx942_attention_4warp_gqa_supports_spec",
        [](const py::dict& d, const std::string& arch) {
            Store st;
            rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
            char reason[1024] = {};
            bool ok = rocke_gfx942_attention_4warp_gqa_supports_spec(
                &s, arch.c_str(), reason, sizeof(reason));
            return supports_result(ok, reason);
        },
        py::arg("spec"),
        py::arg("arch"),
        "gfx942 supports_gfx942_4warp_gqa_spec(spec, arch) -> (ok, reason).");
    m.def(
        "gfx942_gqa_fold_eligible",
        [](int head_size,
           int num_queries_per_kv,
           int sliding_window,
           const std::string& dtype,
           int block_size) {
            return rocke_gfx942_gqa_fold_eligible(
                head_size, num_queries_per_kv, sliding_window, dtype.c_str(), block_size);
        },
        py::arg("head_size"),
        py::arg("num_queries_per_kv"),
        py::arg("sliding_window"),
        py::arg("dtype"),
        py::arg("block_size"),
        "gfx942_gqa_fold_eligible(...) -> bool.");
    m.def(
        "gfx942_4warp_launch_grid",
        [](const py::dict& d) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            int grid[3] = {0, 0, 0};
            if(!rocke_gfx942_4warp_launch_grid(&p, grid))
                raise_bad_problem("gfx942_4warp_launch_grid");
            return py::make_tuple(grid[0], grid[1], grid[2]);
        },
        py::arg("problem"),
        "gfx942_4warp_launch_grid(problem) -> (x, y, z).");
    m.def(
        "unified_attention_supports_tiled_3d",
        [](const py::dict& d, const std::string& arch) {
            Store st;
            rocke_unified_attention_problem_t p = problem_fields_from_dict(d, st);
            char reason[1024] = {};
            bool ok = rocke_unified_attention_supports_tiled_3d(
                &p, arch.c_str(), reason, sizeof(reason));
            return supports_result(ok, reason);
        },
        py::arg("problem"),
        py::arg("arch"),
        "supports_native_unified_attention_3d_tiled(problem, arch) -> (ok, reason).");
}
