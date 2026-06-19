// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/ckc_engine.cpp -- pybind11 module `ckc_engine` exposing the C++
 * ck_dsl_c engine (libckc_core.a) to Python. This is the foundation of the
 * CK_DSL_BACKEND=cpp dual-backend path (WS4).
 *
 * It binds the universal-GEMM family as the first template:
 *
 *   ckc_engine.gemm_lower_llvm(spec_dict, arch="gfx950") -> str   (.ll text)
 *   ckc_engine.gemm_serialize_ir(spec_dict, arch="gfx950") -> str (ck.dsl.ir/v1)
 *   ckc_engine.gemm_verify(spec_dict, arch="gfx950") -> list[str] (diagnostics)
 *
 * The spec_dict carries the UniversalGemmSpec fields. The binding drives the
 * EXACT same C++ engine that the differential parity harness already validates,
 * so for the same spec the outputs are byte-identical to the Python engine's
 * lower_kernel_to_llvm(build_universal_gemm(spec)) / ir_serialize.serialize.
 *
 * Error model: the engine uses a sticky-error IRBuilder; on failure we raise a
 * RuntimeError carrying the ckc_ir_builder_error() / status text.
 *
 * ISOLATION: this file is additive and lives entirely under bindings/. It does
 * not touch the engine src/include or the Python ck_dsl package; it only
 * #includes the public C API headers and links the prebuilt archive.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gemm_universal.h"
}

namespace py = pybind11;

namespace {

/* --------------------------------------------------------------------------
 * String-lifetime helper.
 *
 * ckc_gemm_universal_spec_t stores const char* (string literals in the C
 * emitters). When we build a spec from a Python dict we must keep the backing
 * std::string alive for as long as the spec is used. SpecHolder owns both the
 * struct and every string it points into.
 * ------------------------------------------------------------------------ */
struct SpecHolder
{
    ckc_gemm_universal_spec_t spec;
    std::vector<std::string> storage; /* keeps pointed-to strings alive */

    const char* keep(const std::string& s)
    {
        storage.push_back(s);
        return storage.back().c_str();
    }
};

/* Pull an int out of the dict (with default). */
int dict_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key))
        return d[key].cast<int>();
    return dflt;
}

bool dict_bool(const py::dict& d, const char* key, bool dflt)
{
    if(d.contains(key))
        return d[key].cast<bool>();
    return dflt;
}

/* Optional string: returns true + sets `out` if present (non-None). */
bool dict_str(const py::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

/* Build a ckc_gemm_universal_spec_t from a Python spec dict.
 *
 * The dict mirrors UniversalGemmSpec. Recognised layouts (both accepted):
 *   (a) flat:   {name, tile_m, tile_n, ..., pipeline, epilogue, dtype_a, ...}
 *   (b) nested: {name, tile:{...}, trait:{...}, data:{...}, wave_size, ...}
 * Nested sub-dicts take precedence; any flat key is also honoured. Every field
 * defaults to the C engine default (ckc_gemm_universal_spec_default), so a
 * minimal dict {name, tile_m, tile_n, tile_k, warp_m, warp_n} works. */
SpecHolder build_spec(const py::dict& root)
{
    SpecHolder h;
    h.spec = ckc_gemm_universal_spec_default();
    h.storage.reserve(16);

    /* Merge nested sub-dicts down into a flat view for convenience. */
    auto sub = [&](const char* key) -> py::dict {
        if(root.contains(key) && py::isinstance<py::dict>(root[key]))
            return root[key].cast<py::dict>();
        return py::dict();
    };
    py::dict tile  = sub("tile");
    py::dict trait = sub("trait");
    py::dict data  = sub("data");

    /* A getter that checks the nested sub-dict first, then the flat root. */
    auto pick = [&](const py::dict& nested, const char* key) -> py::dict {
        py::dict r;
        if(nested.contains(key))
            r[key] = nested[key];
        else if(root.contains(key))
            r[key] = root[key];
        return r;
    };

    /* name */
    {
        std::string s;
        if(dict_str(root, "name", s))
            h.spec.name = h.keep(s);
    }

    /* ---- tile geometry ---- */
    {
        py::dict t;
        const char* keys[] = {"tile_m",
                              "tile_n",
                              "tile_k",
                              "warp_m",
                              "warp_n",
                              "warp_k",
                              "warp_tile_m",
                              "warp_tile_n",
                              "warp_tile_k"};
        for(const char* k : keys)
        {
            py::dict got = pick(tile, k);
            if(got.contains(k))
                t[k] = got[k];
        }
        h.spec.tile.tile_m      = dict_int(t, "tile_m", h.spec.tile.tile_m);
        h.spec.tile.tile_n      = dict_int(t, "tile_n", h.spec.tile.tile_n);
        h.spec.tile.tile_k      = dict_int(t, "tile_k", h.spec.tile.tile_k);
        h.spec.tile.warp_m      = dict_int(t, "warp_m", h.spec.tile.warp_m);
        h.spec.tile.warp_n      = dict_int(t, "warp_n", h.spec.tile.warp_n);
        h.spec.tile.warp_k      = dict_int(t, "warp_k", h.spec.tile.warp_k);
        h.spec.tile.warp_tile_m = dict_int(t, "warp_tile_m", h.spec.tile.warp_tile_m);
        h.spec.tile.warp_tile_n = dict_int(t, "warp_tile_n", h.spec.tile.warp_tile_n);
        h.spec.tile.warp_tile_k = dict_int(t, "warp_tile_k", h.spec.tile.warp_tile_k);
    }

    /* ---- trait ---- */
    {
        py::dict tr;
        const char* keys[] = {"pipeline",
                              "scheduler",
                              "epilogue",
                              "pad_m",
                              "pad_n",
                              "pad_k",
                              "persistent",
                              "chiplet_swizzle",
                              "chiplet_wgm",
                              "chiplet_num_xcds",
                              "chiplet_chunk_size",
                              "waves_per_eu",
                              "preshuffle_b",
                              "direct_to_lds",
                              "dtl_cache_a",
                              "dtl_cache_b",
                              "dtl_prefetch",
                              "active_tile_skip",
                              "lds_k_pad",
                              "lds_swizzle"};
        for(const char* k : keys)
        {
            py::dict got = pick(trait, k);
            if(got.contains(k))
                tr[k] = got[k];
        }
        std::string s;
        if(dict_str(tr, "pipeline", s))
            h.spec.trait.pipeline = h.keep(s);
        if(dict_str(tr, "scheduler", s))
            h.spec.trait.scheduler = h.keep(s);
        if(dict_str(tr, "epilogue", s))
            h.spec.trait.epilogue = h.keep(s);
        h.spec.trait.pad_m      = dict_bool(tr, "pad_m", h.spec.trait.pad_m);
        h.spec.trait.pad_n      = dict_bool(tr, "pad_n", h.spec.trait.pad_n);
        h.spec.trait.pad_k      = dict_bool(tr, "pad_k", h.spec.trait.pad_k);
        h.spec.trait.persistent = dict_bool(tr, "persistent", h.spec.trait.persistent);
        h.spec.trait.chiplet_swizzle =
            dict_bool(tr, "chiplet_swizzle", h.spec.trait.chiplet_swizzle);
        h.spec.trait.chiplet_wgm = dict_int(tr, "chiplet_wgm", h.spec.trait.chiplet_wgm);
        h.spec.trait.chiplet_num_xcds =
            dict_int(tr, "chiplet_num_xcds", h.spec.trait.chiplet_num_xcds);
        h.spec.trait.chiplet_chunk_size =
            dict_int(tr, "chiplet_chunk_size", h.spec.trait.chiplet_chunk_size);
        if(tr.contains("waves_per_eu") && !tr["waves_per_eu"].is_none())
        {
            h.spec.trait.waves_per_eu_set = true;
            h.spec.trait.waves_per_eu     = tr["waves_per_eu"].cast<int>();
        }
        h.spec.trait.preshuffle_b  = dict_bool(tr, "preshuffle_b", h.spec.trait.preshuffle_b);
        h.spec.trait.direct_to_lds = dict_bool(tr, "direct_to_lds", h.spec.trait.direct_to_lds);
        h.spec.trait.dtl_cache_a   = dict_int(tr, "dtl_cache_a", h.spec.trait.dtl_cache_a);
        h.spec.trait.dtl_cache_b   = dict_int(tr, "dtl_cache_b", h.spec.trait.dtl_cache_b);
        h.spec.trait.dtl_prefetch  = dict_bool(tr, "dtl_prefetch", h.spec.trait.dtl_prefetch);
        h.spec.trait.active_tile_skip =
            dict_bool(tr, "active_tile_skip", h.spec.trait.active_tile_skip);
        h.spec.trait.lds_k_pad   = dict_int(tr, "lds_k_pad", h.spec.trait.lds_k_pad);
        h.spec.trait.lds_swizzle = dict_bool(tr, "lds_swizzle", h.spec.trait.lds_swizzle);
    }

    /* ---- data ---- */
    {
        py::dict da;
        const char* keys[] = {"dtype_a", "dtype_b", "dtype_c", "dtype_acc", "layout"};
        for(const char* k : keys)
        {
            py::dict got = pick(data, k);
            if(got.contains(k))
                da[k] = got[k];
        }
        std::string s;
        if(dict_str(da, "dtype_a", s))
            h.spec.data.dtype_a = h.keep(s);
        if(dict_str(da, "dtype_b", s))
            h.spec.data.dtype_b = h.keep(s);
        if(dict_str(da, "dtype_c", s))
            h.spec.data.dtype_c = h.keep(s);
        if(dict_str(da, "dtype_acc", s))
            h.spec.data.dtype_acc = h.keep(s);
        if(dict_str(da, "layout", s))
            h.spec.data.layout = h.keep(s);
    }

    /* ---- top-level scalars ---- */
    h.spec.wave_size  = dict_int(root, "wave_size", h.spec.wave_size);
    h.spec.block_size = dict_int(root, "block_size", h.spec.block_size);
    h.spec.batched    = dict_bool(root, "batched", h.spec.batched);

    ckc_gemm_universal_spec_finalize(&h.spec);
    return h;
}

const char* arch_or_default(const std::string& arch)
{
    return arch.empty() ? "gfx950" : arch.c_str();
}

/* ---------------------------------------------------------------- bindings */

std::string gemm_lower_llvm(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h    = build_spec(spec_dict);
    char* llvm_text = nullptr;
    char err[CKC_ERR_MSG_CAP];
    err[0]          = '\0';
    ckc_status_t st = ckc_gemm_universal_lower_to_llvm(
        &h.spec, arch_or_default(arch), CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if(st != CKC_OK || !llvm_text)
    {
        if(llvm_text)
            free(llvm_text);
        std::string msg = "ckc_engine.gemm_lower_llvm failed (status=" + std::to_string((int)st) +
                          "): " + (err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(llvm_text);
    free(llvm_text);
    return out;
}

std::string gemm_serialize_ir(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = ckc_build_universal_gemm_new(&b, &h.spec, arch_or_default(arch));
    if(!kernel || !ckc_ir_builder_ok(&b))
    {
        std::string msg =
            std::string("ckc_engine.gemm_serialize_ir build failed: ") + ckc_ir_builder_error(&b);
        ckc_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* text      = nullptr;
    ckc_status_t st = ckc_ir_serialize(kernel, &text);
    if(st != CKC_OK || !text)
    {
        if(text)
            free(text);
        ckc_ir_builder_free(&b);
        throw std::runtime_error("ckc_engine.gemm_serialize_ir serialize failed (status=" +
                                 std::to_string((int)st) + ")");
    }
    std::string out(text);
    free(text);
    ckc_ir_builder_free(&b);
    return out;
}

std::vector<std::string> gemm_verify(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = ckc_build_universal_gemm_new(&b, &h.spec, arch_or_default(arch));
    if(!kernel || !ckc_ir_builder_ok(&b))
    {
        std::string msg =
            std::string("ckc_engine.gemm_verify build failed: ") + ckc_ir_builder_error(&b);
        ckc_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    ckc_diag_t* diags = nullptr;
    size_t n          = 0;
    ckc_verify(kernel, &diags, &n);
    std::vector<std::string> out;
    out.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        char* s = ckc_diag_to_string(&diags[i]);
        if(s)
        {
            out.emplace_back(s);
            free(s);
        }
    }
    ckc_diags_free(diags, n);
    ckc_ir_builder_free(&b);
    return out;
}

/* is_valid_spec(spec, arch) -> (bool, reason) tuple. Convenience extra. */
py::tuple gemm_is_valid(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    char reason[CKC_ERR_MSG_CAP];
    reason[0] = '\0';
    bool ok =
        ckc_gemm_universal_is_valid_spec(&h.spec, arch_or_default(arch), reason, sizeof reason);
    return py::make_tuple(ok, std::string(reason));
}

/* kernel_name(spec) -> str. Convenience extra. */
std::string gemm_kernel_name(const py::dict& spec_dict)
{
    SpecHolder h = build_spec(spec_dict);
    char name[512];
    name[0]         = '\0';
    ckc_status_t st = ckc_gemm_universal_kernel_name(&h.spec, name, sizeof name);
    if(st != CKC_OK)
        throw std::runtime_error(
            "ckc_engine.gemm_kernel_name failed (status=" + std::to_string((int)st) + ")");
    return std::string(name);
}

} // namespace

PYBIND11_MODULE(ckc_engine, m)
{
    m.doc() = "pybind11 binding for the C++ ck_dsl_c engine (universal GEMM "
              "family). Foundation of the CK_DSL_BACKEND=cpp dual-backend path.";

    m.def("gemm_lower_llvm",
          &gemm_lower_llvm,
          py::arg("spec"),
          py::arg("arch") = "gfx950",
          "Build a universal GEMM from spec dict and lower to AMDGPU LLVM IR (.ll) "
          "text. Byte-identical to the Python engine for the same spec.");
    m.def("gemm_serialize_ir",
          &gemm_serialize_ir,
          py::arg("spec"),
          py::arg("arch") = "gfx950",
          "Build a universal GEMM and serialize its IR (ck.dsl.ir/v1 text). "
          "Byte-identical to ck_dsl.core.ir_serialize.serialize for the same spec.");
    m.def("gemm_verify",
          &gemm_verify,
          py::arg("spec"),
          py::arg("arch") = "gfx950",
          "Build a universal GEMM and run the IR verifier; returns a list of "
          "diagnostic strings (empty == well-formed).");
    m.def("gemm_is_valid",
          &gemm_is_valid,
          py::arg("spec"),
          py::arg("arch") = "gfx950",
          "is_valid_spec(spec, arch) -> (ok: bool, reason: str).");
    m.def("gemm_kernel_name",
          &gemm_kernel_name,
          py::arg("spec"),
          "UniversalGemmSpec.kernel_name(spec) -> str.");
}
