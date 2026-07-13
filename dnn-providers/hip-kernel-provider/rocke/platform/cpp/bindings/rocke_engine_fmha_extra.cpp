// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_fmha_extra.cpp -- bindings for the remaining FMHA-fwd
 * variant families and the fused-MoE end-to-end orchestrator, kept in their own
 * translation unit for the same reason as rocke_engine_attention.cpp: the FMHA
 * instance headers re-use a set of struct tags (rocke_fmha_*) across several
 * headers, so isolating these families keeps the GEMM/conv translation unit's
 * include surface clean.
 *
 * register_fmha_extra(m) is called from the main module file. Each family mirrors
 * its tests/parity/<fam>_emit.c make_spec + build + lower exactly, so for the
 * same spec dict the binding output is byte-identical to the standalone emitter.
 *
 *   fmha_appendkv         single KernelDef -> one .ll (vector KV scatter)
 *   fmha_paged_prefill    single KernelDef -> one .ll
 *   fmha_varlen           single KernelDef -> one .ll
 *   fmha_splitkv_decode   two kernels (segment + reduce); the lower binding emits
 *                         the segment phase by default to match the parity
 *                         harness (which runs the emitter as `<exe> <idx> ll`,
 *                         i.e. mode=ll / phase=seg), with a spec "phase" key
 *                         ("seg" default | "reduce" | "both") selecting the
 *                         segment-then-reduce concatenation, exactly as the
 *                         standalone emitter prints them.
 *
 * fused_moe_e2e is handled in a separate translation unit
 * (rocke_engine_fused_moe_e2e.cpp): its internal header drags in the
 * moe_gemm_fused / tensor_view headers, whose rocke_tensor_descriptor tag clashes
 * with the transforms.h definition pulled in by _fmha_common.h here.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "rocke/helper_rocke.helpers.rotary.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/instance_fmha_appendkv.h"
#include "rocke/instance_fmha_paged_prefill.h"
#include "rocke/instance_fmha_splitkv_decode.h"
#include "rocke/instance_fmha_varlen.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

/* ----------------------------- dict helpers ----------------------------- */
int x_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
    return dflt;
}
bool x_bool(const py::dict& d, const char* key, bool dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<bool>();
    return dflt;
}
bool x_str(const py::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

/* Lifetime helper for any const char* fields pulled from the dict. std::deque,
 * not std::vector: a vector realloc would dangle previously-kept c_str()
 * pointers held by the spec struct. */
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
    if(!x_str(d, key, s))
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

/* Build a rocke_fmha_common_spec_t from the common sub-keys of the dict. Mirrors
 * the make_spec helpers in the standalone emitters: when block_size_q /
 * block_size_k are present the shape is built via rocke_fmha_shape_make, else
 * rocke_fmha_shape_default. */
rocke_fmha_common_spec_t common_of(const py::dict& d, Store& st)
{
    rocke_fmha_shape_t shape;
    if(d.contains("block_size_q") || d.contains("block_size_k"))
        shape = rocke_fmha_shape_make(x_int(d, "head_size", 0),
                                      x_int(d, "num_query_heads", 0),
                                      x_int(d, "num_kv_heads", 0),
                                      x_int(d, "block_size_q", 16),
                                      x_int(d, "block_size_k", 64));
    else
        shape = rocke_fmha_shape_default(
            x_int(d, "head_size", 0), x_int(d, "num_query_heads", 0), x_int(d, "num_kv_heads", 0));
    rocke_fmha_common_spec_t c = rocke_fmha_common_spec_default(shape);
    std::string v;
    if(x_str(d, "dtype", v))
        c.dtype = st.keep(v);
    c.mask_mode = mask_of(d, "mask_mode");
    c.sliding_window = x_int(d, "sliding_window", c.sliding_window);
    return c;
}

/* ============================ fmha_appendkv ============================= */
/* Single KernelDef -> one .ll. The standalone emitter inits the builder with
 * spec.kernel_name(), then calls rocke_build_fmha_fwd_appendkv(&b, &spec, arch),
 * then lowers via rocke_lower_kernel_to_llvm_ex. */

rocke_fmha_appendkv_spec_t appendkv_build(const py::dict& d, Store& st, bool* ok, std::string* why)
{
    *ok = true;
    rocke_fmha_common_spec_t c = common_of(d, st);
    rocke_fmha_appendkv_spec_t s = rocke_fmha_appendkv_spec_default(c, x_int(d, "batch", 1));
    s.block_size = x_int(d, "block_size", s.block_size);
    if(x_bool(d, "has_rotary", false))
    {
        std::string lay = "half";
        x_str(d, "rotary_layout", lay);
        rocke_rotary_layout_t l
            = (lay == "interleaved") ? ROCKE_ROTARY_INTERLEAVED : ROCKE_ROTARY_HALF;
        rocke_rotary_spec_t rot;
        rocke_status_t rs = rocke_rotary_spec_init(&rot,
                                                   x_int(d, "rotary_head_size", c.shape.head_size),
                                                   l,
                                                   x_int(d, "rotary_table_stride_pos", 0));
        if(rs != ROCKE_OK)
        {
            *ok = false;
            *why = "invalid rotary spec for fmha_appendkv";
            return s;
        }
        s.has_rotary = true;
        s.rotary = rot;
    }
    return s;
}

rocke_kernel_def_t*
    appendkv_make(const py::dict& d, Store& st, rocke_ir_builder_t* b, const char* arch)
{
    bool ok = true;
    std::string why;
    rocke_fmha_appendkv_spec_t s = appendkv_build(d, st, &ok, &why);
    if(!ok)
        throw std::runtime_error("rocke_engine.fmha_appendkv: " + why);
    char namebuf[256];
    namebuf[0] = '\0';
    if(rocke_fmha_appendkv_kernel_name(&s, namebuf, sizeof namebuf) != ROCKE_OK)
        throw std::runtime_error("rocke_engine.fmha_appendkv kernel_name failed");
    rocke_ir_builder_init(b, namebuf);
    return rocke_build_fmha_fwd_appendkv(b, &s, arch);
}

std::string appendkv_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = appendkv_make(d, st, &b, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("fmha_appendkv build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll, err, sizeof err);
    rocke_ir_builder_free(&b);
    return take_ll(s2, ll, err, "rocke_engine.fmha_appendkv_lower_llvm");
}
std::string appendkv_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = appendkv_make(d, st, &b, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("fmha_appendkv build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_appendkv_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}
std::vector<std::string> appendkv_verify(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = appendkv_make(d, st, &b, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string m = std::string("fmha_appendkv build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(m);
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ========================== fmha_paged_prefill ========================== */
/* Single KernelDef -> one .ll. kb-based build + rocke_lower_kernel_to_llvm_ex. */

rocke_fmha_fwd_paged_prefill_spec_t paged_build(const py::dict& d, Store& st)
{
    rocke_fmha_common_spec_t c = common_of(d, st);
    rocke_fmha_fwd_paged_prefill_spec_t s
        = rocke_fmha_fwd_paged_prefill_spec_default(c,
                                                    x_int(d, "page_block_size", 16),
                                                    x_int(d, "max_blocks_per_seq", 32),
                                                    x_int(d, "batch", 1));
    s.use_mfma_body = x_bool(d, "use_mfma_body", s.use_mfma_body);
    return s;
}
std::string paged_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_fwd_paged_prefill_spec_t s = paged_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_paged_prefill(&kb, &s, a);
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_paged_prefill_lower_llvm build failed");
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll, err, sizeof err);
    rocke_fmha_kernel_builder_free(&kb);
    return take_ll(s2, ll, err, "rocke_engine.fmha_paged_prefill_lower_llvm");
}
std::string paged_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_fwd_paged_prefill_spec_t s = paged_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_paged_prefill(&kb, &s, a);
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_paged_prefill_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_paged_prefill_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> paged_verify(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_fwd_paged_prefill_spec_t s = paged_build(d, st);
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_paged_prefill(&kb, &s, a);
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_paged_prefill_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ============================== fmha_varlen ============================= */
/* Single KernelDef -> one .ll. rocke_build_fmha_fwd_varlen takes an err buffer. */

rocke_fmha_fwd_varlen_spec_t varlen_build(const py::dict& d, Store& st)
{
    rocke_fmha_common_spec_t c = common_of(d, st);
    return rocke_fmha_fwd_varlen_spec_default(
        c, x_int(d, "max_seqlen_q", 0), x_int(d, "max_seqlen_k", 0), x_int(d, "batch", 1));
}
rocke_kernel_def_t*
    varlen_make(const py::dict& d, Store& st, rocke_fmha_kernel_builder_t* kb, const char* a)
{
    rocke_fmha_fwd_varlen_spec_t s = varlen_build(d, st);
    std::memset(kb, 0, sizeof *kb);
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_kernel_def_t* k = rocke_build_fmha_fwd_varlen(kb, &s, a, err, sizeof err);
    if(!k)
    {
        std::string m
            = std::string("fmha_varlen build failed: ") + (err[0] ? err : "unknown error");
        rocke_fmha_kernel_builder_free(kb);
        throw std::runtime_error("rocke_engine." + m);
    }
    return k;
}
std::string varlen_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = varlen_make(d, st, &kb, a);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2
        = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll, err, sizeof err);
    rocke_fmha_kernel_builder_free(&kb);
    return take_ll(s2, ll, err, "rocke_engine.fmha_varlen_lower_llvm");
}
std::string varlen_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = varlen_make(d, st, &kb, a);
    std::string out;
    try
    {
        out = ser_kernel(k, "rocke_engine.fmha_varlen_serialize_ir");
    }
    catch(...)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw;
    }
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> varlen_verify(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* k = varlen_make(d, st, &kb, a);
    std::vector<std::string> out;
    try
    {
        out = ver_kernel(k);
    }
    catch(...)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw;
    }
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}

/* ========================== fmha_splitkv_decode ========================= */
/* Two kernels: segment + reduce. The lower binding defaults to the segment
 * phase (matching the parity harness, which runs the emitter `<exe> <idx> ll`,
 * i.e. mode=ll / phase=seg). A "phase" spec key selects "seg" (default),
 * "reduce", or "both" (segment-then-reduce concatenation). */

rocke_fmha_splitkv_decode_spec_t splitkv_build(const py::dict& d, Store& st)
{
    rocke_fmha_common_spec_t c = common_of(d, st);
    rocke_fmha_splitkv_decode_spec_t s = rocke_fmha_splitkv_decode_spec_default(
        c, x_int(d, "batch", 1), x_int(d, "num_segments", 4));
    s.use_mfma_body = x_bool(d, "use_mfma_body", s.use_mfma_body);
    s.prune_sliding_window = x_bool(d, "prune_sliding_window", s.prune_sliding_window);
    return s;
}
std::string splitkv_lower_one(const rocke_fmha_splitkv_decode_spec_t* s, const char* a, bool reduce)
{
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t s2 = reduce ? rocke_fmha_splitkv_decode_reduce_lower_to_llvm(
                                     s, a, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err)
                               : rocke_fmha_splitkv_decode_segment_lower_to_llvm(
                                     s, a, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_ll(s2, ll, err, "rocke_engine.fmha_splitkv_decode_lower_llvm");
}
std::string splitkv_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_splitkv_decode_spec_t s = splitkv_build(d, st);
    std::string phase = "seg";
    x_str(d, "phase", phase);
    if(phase == "reduce")
        return splitkv_lower_one(&s, a, true);
    if(phase == "both")
        return splitkv_lower_one(&s, a, false) + splitkv_lower_one(&s, a, true);
    return splitkv_lower_one(&s, a, false); /* "seg" default */
}
std::string splitkv_serialize(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_splitkv_decode_spec_t s = splitkv_build(d, st);
    std::string phase = "seg";
    x_str(d, "phase", phase);
    bool reduce = (phase == "reduce");
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = reduce ? rocke_build_fmha_fwd_splitkv_decode_reduce(&kb, &s, a)
                                   : rocke_build_fmha_fwd_splitkv_decode_segment(&kb, &s, a);
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_splitkv_decode_serialize_ir build failed");
    }
    std::string out = ser_kernel(k, "rocke_engine.fmha_splitkv_decode_serialize_ir");
    rocke_fmha_kernel_builder_free(&kb);
    return out;
}
std::vector<std::string> splitkv_verify(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmha_splitkv_decode_spec_t s = splitkv_build(d, st);
    std::string phase = "seg";
    x_str(d, "phase", phase);
    bool reduce = (phase == "reduce");
    rocke_fmha_kernel_builder_t kb;
    std::memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* k = reduce ? rocke_build_fmha_fwd_splitkv_decode_reduce(&kb, &s, a)
                                   : rocke_build_fmha_fwd_splitkv_decode_segment(&kb, &s, a);
    if(!k)
    {
        rocke_fmha_kernel_builder_free(&kb);
        throw std::runtime_error("rocke_engine.fmha_splitkv_decode_verify build failed");
    }
    std::vector<std::string> out = ver_kernel(k);
    rocke_fmha_kernel_builder_free(&kb);
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

void register_fmha_extra(py::module_& m)
{
    reg3(m, "fmha_appendkv", &appendkv_lower, &appendkv_serialize, &appendkv_verify, "gfx950");
    reg3(m, "fmha_paged_prefill", &paged_lower, &paged_serialize, &paged_verify, "gfx950");
    reg3(m, "fmha_varlen", &varlen_lower, &varlen_serialize, &varlen_verify, "gfx950");
    reg3(m, "fmha_splitkv_decode", &splitkv_lower, &splitkv_serialize, &splitkv_verify, "gfx950");
}
