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

#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "rocke/instance_attention_unified.h"
#include "rocke/instance_fmha_appendkv.h"
#include "rocke/instance_fmha_bwd.h"
#include "rocke/instance_fmha_fwd_fp8.h"
#include "rocke/instance_fmha_head_grouping.h"
#include "rocke/instance_fmha_mfma.h"
#include "rocke/instance_fmha_paged_prefill.h"
#include "rocke/instance_fmha_splitkv_decode.h"
#include "rocke/instance_fmha_varlen.h"
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

std::string take_ll(rocke_status_t st, char* ll, const char* err, const char* fn)
{
    if(st != ROCKE_OK || !ll)
    {
        if(ll)
            free(ll);
        std::string msg = std::string(fn) + " failed (status=" + std::to_string((int)st)
                          + "): " + (err && err[0] ? err : "unknown error");
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

rocke_unified_attention_problem_t au_build(const py::dict& d, Store& st)
{
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
    std::string v;
    if(a_str(d, "dtype", v))
        p.dtype = st.keep(v);
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
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch.empty() ? "gfx950" : arch.c_str(), &ll, err, sizeof err);
    rocke_ir_builder_free(&b);
    return take_ll(s2, ll, err, "rocke_engine.attention_unified_lower_llvm");
}
std::string au_serialize(const py::dict& d, const std::string& arch)
{
    (void)arch;
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
    (void)arch;
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
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_mfma(nullptr, &s, a);
    if(!k)
        throw std::runtime_error("rocke_engine.fmha_mfma_lower_llvm build failed");
    char* ll = nullptr;
    rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    return take_ll(s2, ll, nullptr, "rocke_engine.fmha_mfma_lower_llvm");
}
std::string fmma_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_mfma(nullptr, &s, arch.empty() ? "gfx950" : arch.c_str());
    if(!k)
        throw std::runtime_error("rocke_engine.fmha_mfma_serialize_ir build failed");
    return ser_kernel(k, "rocke_engine.fmha_mfma_serialize_ir");
}
std::vector<std::string> fmma_verify(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_fmha_mfma_spec_t s = mfma_build(d, st);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_mfma(nullptr, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_fmha_fwd_fp8_lower_to_llvm(
        &s, arch.empty() ? "gfx950" : arch.c_str(), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.fmha_fwd_fp8_lower_llvm");
}
std::string fp8_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_fp8_new(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_fwd_fp8_spec_t s = fp8_build(d, st, nullptr);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_fp8_new(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_bwd_lower_llvm build failed");
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch.empty() ? "gfx950" : arch.c_str(), &ll, err, sizeof err);
    rocke_fmha_kernel_builder_free(&kb);
    return take_ll(s2, ll, err, "rocke_engine.fmha_bwd_lower_llvm");
}
std::string bwd_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_bwd_spec_t s = bwd_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = rocke_build_fmha_bwd(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_head_grouping_lower_llvm build failed");
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch.empty() ? "gfx950" : arch.c_str(), &ll, err, sizeof err);
    rocke_fmha_kernel_builder_free(&kb);
    return take_ll(s2, ll, err, "rocke_engine.fmha_head_grouping_lower_llvm");
}
std::string hg_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_fmha_head_grouping_spec_t s = hg_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_fmha_fwd_head_grouping(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    const char* arch_c = arch.empty() ? "gfx950" : arch.c_str();
    rocke_kernel_def_t* k = sparse_build(d, st, arch_c, &jctx, &vctx, &is_vsa);
    if(!k)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw std::runtime_error("rocke_engine.sparse_attention_lower_llvm build failed");
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, arch_c, &ll, err, sizeof err);
    std::string out;
    try
    {
        out = take_ll(s2, ll, err, "rocke_engine.sparse_attention_lower_llvm");
    }
    catch(...)
    {
        sparse_free(&jctx, &vctx, is_vsa);
        throw;
    }
    sparse_free(&jctx, &vctx, is_vsa);
    return out;
}
std::string sparse_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    rocke_kernel_def_t* k
        = sparse_build(d, st, arch.empty() ? "gfx950" : arch.c_str(), &jctx, &vctx, &is_vsa);
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
    Store st;
    rocke_jenga_sparse_ctx_t jctx;
    rocke_vsa_sparse_ctx_t vctx;
    bool is_vsa = false;
    rocke_kernel_def_t* k
        = sparse_build(d, st, arch.empty() ? "gfx950" : arch.c_str(), &jctx, &vctx, &is_vsa);
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
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_sage_attention_lower_to_llvm(
        &s, arch.empty() ? "gfx950" : arch.c_str(), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.sage_attention_lower_llvm");
}
std::string sage_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_sage_attention_new(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_sage_attention_spec_t s = sage_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k
        = rocke_build_sage_attention_new(&kb, &s, arch.empty() ? "gfx950" : arch.c_str());
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

rocke_attention_tiled_2d_spec_t t2d_build(const py::dict& d, Store& st)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.head_size = a_int(d, "head_size", s.head_size);
    s.block_size = a_int(d, "block_size", s.block_size);
    s.num_query_heads = a_int(d, "num_query_heads", s.num_query_heads);
    s.num_kv_heads = a_int(d, "num_kv_heads", s.num_kv_heads);
    s.use_sinks = a_bool(d, "use_sinks", s.use_sinks);
    s.sliding_window = a_int(d, "sliding_window", s.sliding_window);
    s.has_softcap = a_bool(d, "has_softcap", s.has_softcap);
    /* extended gfx950 flag stack (harmless defaults on gfx942) */
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
    s.block_m_per_warp = a_int(d, "block_m_per_warp", s.block_m_per_warp);
    s.use_fp8_mfma_pv = a_bool(d, "use_fp8_mfma_pv", s.use_fp8_mfma_pv);
    s.use_fp8_mfma_qk = a_bool(d, "use_fp8_mfma_qk", s.use_fp8_mfma_qk);
    s.use_i64_kv_addr = a_bool(d, "use_i64_kv_addr", s.use_i64_kv_addr);
    s.use_register_pv = a_bool(d, "use_register_pv", s.use_register_pv);
    s.use_mfma_32x32 = a_bool(d, "use_mfma_32x32", s.use_mfma_32x32);
    s.use_transposed_qk_32x32 = a_bool(d, "use_transposed_qk_32x32", s.use_transposed_qk_32x32);
    s.use_transposed_scalar_state
        = a_bool(d, "use_transposed_scalar_state", s.use_transposed_scalar_state);
    s.use_transposed_invariant_hoist
        = a_bool(d, "use_transposed_invariant_hoist", s.use_transposed_invariant_hoist);
    s.use_transposed_mask_once = a_bool(d, "use_transposed_mask_once", s.use_transposed_mask_once);
    s.use_grouped_kv2_softmax = a_bool(d, "use_grouped_kv2_softmax", s.use_grouped_kv2_softmax);
    s.use_early_v_schedule = a_bool(d, "use_early_v_schedule", s.use_early_v_schedule);
    s.use_fast_paged_kv_desc = a_bool(d, "use_fast_paged_kv_desc", s.use_fast_paged_kv_desc);
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    if(a_str(d, "kv_storage_dtype", v))
        s.kv_storage_dtype = st.keep(v);
    return s;
}

std::string t942_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx942" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx942_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    char* ll = nullptr;
    rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_ll(s2, ll, nullptr, "rocke_engine.gfx942_attention_tiled_2d_lower_llvm");
}
std::string t942_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(
        &b, &s, arch.empty() ? "gfx942" : arch.c_str());
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
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_tiled_scalar_new(
        &b, &s, arch.empty() ? "gfx942" : arch.c_str());
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

std::string t950_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m
            = std::string("gfx950_attention_tiled_2d build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    char* ll = nullptr;
    rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_ll(s2, ll, nullptr, "rocke_engine.gfx950_attention_tiled_2d_lower_llvm");
}
std::string t950_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(
        &b, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_attention_tiled_2d_spec_t s = t2d_build(d, st);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_gfx950_build_unified_attention_2d_tiled_new(
        &b, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    s.head_size = a_int(d, "head_size", 64);
    s.block_size = a_int(d, "block_size", 32);
    s.num_query_heads = a_int(d, "num_query_heads", 64);
    s.num_kv_heads = a_int(d, "num_kv_heads", 8);
    s.use_sinks = a_bool(d, "use_sinks", false);
    s.sliding_window = a_int(d, "sliding_window", 0);
    s.has_softcap = a_bool(d, "has_softcap", false);
    s.num_warps = a_int(d, "num_warps", 4);
    s.has_waves_per_eu = true;
    s.waves_per_eu = a_int(d, "waves_per_eu", 2);
    s.has_tile_size = true;
    s.tile_size = a_int(d, "tile_size", 64);
    s.block_m_per_warp = a_int(d, "block_m_per_warp", 32);
    s.use_mfma_32x32 = a_bool(d, "use_mfma_32x32", true);
    s.use_transposed_qk_32x32 = a_bool(d, "use_transposed_qk_32x32", true);
    s.use_transposed_scalar_state = a_bool(d, "use_transposed_scalar_state", true);
    s.use_transposed_mask_once = a_bool(d, "use_transposed_mask_once", true);
    s.use_fast_paged_kv_desc = a_bool(d, "use_fast_paged_kv_desc", true);
    /* additive per-config flags */
    s.use_transposed_half_local_pv
        = a_bool(d, "use_transposed_half_local_pv", s.use_transposed_half_local_pv);
    s.use_mfma32_skip_legacy_qreg
        = a_bool(d, "use_mfma32_skip_legacy_qreg", s.use_mfma32_skip_legacy_qreg);
    s.use_agpr_alloc_zero = a_bool(d, "use_agpr_alloc_zero", s.use_agpr_alloc_zero);
    s.use_grouped_kv2_softmax = a_bool(d, "use_grouped_kv2_softmax", s.use_grouped_kv2_softmax);
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    else
        s.dtype = "bf16";
    return s;
}
std::string fkv_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_fastkv_register_p(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("gfx950_attention_tiled_2d_fastkv_regp build failed: ")
                        + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    char* ll = nullptr;
    rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_ll(
        s2, ll, nullptr, "rocke_engine.gfx950_attention_tiled_2d_fastkv_regp_lower_llvm");
}
std::string fkv_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_fastkv_register_p(
        &b, &s, arch.empty() ? "gfx950" : arch.c_str());
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
    Store st;
    rocke_attention_tiled_2d_spec_t s = fkv_build(d, st);
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp");
    rocke_kernel_def_t* k = rocke_build_unified_attention_2d_fastkv_register_p(
        &b, &s, arch.empty() ? "gfx950" : arch.c_str());
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

rocke_unified_attention_3d_tiled_spec_t t3d_build(const py::dict& d, Store& st)
{
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
    std::string v;
    if(a_str(d, "dtype", v))
        s.dtype = st.keep(v);
    if(a_str(d, "kv_storage_dtype", v))
        s.kv_storage_dtype = st.keep(v);
    return s;
}

/* gfx942 3d: emitter lowers only the segment via the family convenience. */
std::string t3d942_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
        &s, arch.empty() ? "gfx942" : arch.c_str(), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.gfx942_attention_tiled_3d_lower_llvm");
}
std::string t3d942_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    rocke_ir_builder_t b;
    char kname[512];
    kname[0] = '\0';
    rocke_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_gfx942(
        &b, &s, arch.empty() ? "gfx942" : arch.c_str());
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
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    rocke_ir_builder_t b;
    char kname[512];
    kname[0] = '\0';
    rocke_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_gfx942(
        &b, &s, arch.empty() ? "gfx942" : arch.c_str());
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
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    std::string out;
    /* segment */
    {
        rocke_ir_builder_t b;
        rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
        rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_gfx950(&b, &s, a);
        if(!k || !rocke_ir_builder_ok(&b))
        {
            std::string m = std::string("gfx950_attention_tiled_3d segment build failed: ")
                            + rocke_ir_builder_error(&b);
            rocke_ir_builder_free(&b);
            throw std::runtime_error(m);
        }
        char* ll = nullptr;
        rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
        out += take_ll(s2, ll, nullptr, "rocke_engine.gfx950_attention_tiled_3d_lower_llvm");
        rocke_ir_builder_free(&b);
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
        char* ll = nullptr;
        rocke_status_t s2 = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
        out += take_ll(s2, ll, nullptr, "rocke_engine.gfx950_attention_tiled_3d_lower_llvm");
        rocke_ir_builder_free(&b);
    }
    return out;
}
std::string t3d950_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
    rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_gfx950(&b, &s, a);
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
    Store st;
    rocke_unified_attention_3d_tiled_spec_t s = t3d_build(d, st);
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_t b;
    rocke_ir_builder_init(&b, "attention_tiled_3d_segment");
    rocke_kernel_def_t* k = rocke_build_unified_attention_3d_tiled_gfx950(&b, &s, a);
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
          std::vector<std::string> (*ver)(const py::dict&, const std::string&),
          const char* dflt_arch)
{
    m.def((std::string(prefix) + "_lower_llvm").c_str(),
          lower,
          py::arg("spec"),
          py::arg("arch") = dflt_arch);
    m.def((std::string(prefix) + "_serialize_ir").c_str(),
          ser,
          py::arg("spec"),
          py::arg("arch") = dflt_arch);
    m.def((std::string(prefix) + "_verify").c_str(),
          ver,
          py::arg("spec"),
          py::arg("arch") = dflt_arch);
}

} // namespace

void register_attention(py::module_& m)
{
    reg3(m, "attention_unified", &au_lower, &au_serialize, &au_verify, "gfx950");
    reg3(m, "fmha_mfma", &fmma_lower, &fmma_serialize, &fmma_verify, "gfx950");
    reg3(m, "fmha_fwd_fp8", &fp8_lower, &fp8_serialize, &fp8_verify, "gfx950");
    reg3(m, "fmha_bwd", &bwd_lower, &bwd_serialize, &bwd_verify, "gfx950");
    reg3(m, "fmha_head_grouping", &hg_lower, &hg_serialize, &hg_verify, "gfx950");
    reg3(m, "sparse_attention", &sparse_lower, &sparse_serialize, &sparse_verify, "gfx950");
    reg3(m, "sage_attention", &sage_lower, &sage_serialize, &sage_verify, "gfx950");
    reg3(m, "gfx942_attention_tiled_2d", &t942_lower, &t942_serialize, &t942_verify, "gfx942");
    reg3(m, "gfx950_attention_tiled_2d", &t950_lower, &t950_serialize, &t950_verify, "gfx950");
    reg3(m,
         "gfx950_attention_tiled_2d_fastkv_regp",
         &fkv_lower,
         &fkv_serialize,
         &fkv_verify,
         "gfx950");
    reg3(
        m, "gfx942_attention_tiled_3d", &t3d942_lower, &t3d942_serialize, &t3d942_verify, "gfx942");
    reg3(
        m, "gfx950_attention_tiled_3d", &t3d950_lower, &t3d950_serialize, &t3d950_verify, "gfx950");
}
