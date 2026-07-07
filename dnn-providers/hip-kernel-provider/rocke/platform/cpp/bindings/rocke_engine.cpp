// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine.cpp -- pybind11 module `rocke_engine` exposing the C++
 * rocke engine (librocke_core.a) to Python. This is the foundation of the
 * ROCKE_BACKEND=cpp dual-backend path.
 *
 * It binds the universal-GEMM family as the first template:
 *
 *   rocke_engine.gemm_lower_llvm(spec_dict, arch="gfx950") -> str   (.ll text)
 *   rocke_engine.gemm_serialize_ir(spec_dict, arch="gfx950") -> str (ck.dsl.ir/v1)
 *   rocke_engine.gemm_verify(spec_dict, arch="gfx950") -> list[str] (diagnostics)
 *
 * The spec_dict carries the UniversalGemmSpec fields. The binding drives the
 * EXACT same C++ engine that the differential parity harness already validates,
 * so for the same spec the outputs are byte-identical to the Python engine's
 * lower_kernel_to_llvm(build_universal_gemm(spec)) / ir_serialize.serialize.
 *
 * Error model: the engine uses a sticky-error IRBuilder; on failure we raise a
 * RuntimeError carrying the rocke_ir_builder_error() / status text.
 *
 * ISOLATION: this file is additive and lives entirely under bindings/. It does
 * not touch the engine src/include or the Python rocke package; it only
 * #includes the public C API headers and links the prebuilt archive.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "family_glue.hpp"

extern "C" {
#include "rocke/arena.h"
#include "rocke/instance_batched_gemm.h"
#include "rocke/instance_block_scale_gemm.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_deep_fused_conv_pool.h"
#include "rocke/instance_flatmm.h"
#include "rocke/instance_gemm_multi_abd.h"
#include "rocke/instance_gemm_multi_d.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_grouped_gemm.h"
#include "rocke/instance_matmul_nbits.h"
#include "rocke/instance_mfma_gemm.h"
#include "rocke/instance_mx_gemm.h"
#include "rocke/instance_streamk_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/rocke_build_id.h"
#include "rocke/verify.h"
/* ---- norm / elementwise / tensor-op families ---- */
#include "rocke/instance_add_rmsnorm2d_bf16.h"
#include "rocke/instance_add_rmsnorm2d_rdquant.h"
#include "rocke/instance_elementwise.h"
#include "rocke/instance_layernorm2d.h"
#include "rocke/instance_permute_nd.h"
#include "rocke/instance_pooling.h"
#include "rocke/instance_reduce.h"
#include "rocke/instance_rmsnorm2d.h"
#include "rocke/instance_smoothquant.h"
#include "rocke/instance_transpose.h"
/* ---- MoE families (standalone-TU-safe) ---- */
#include "rocke/instance_fused_moe.h"
#include "rocke/instance_moe_fused_mega.h"
#include "rocke/instance_moe_fused_mega_fp8.h"
#include "rocke/instance_moe_smoothquant.h"
#include "rocke/instance_moe_sorting.h"
#include "rocke/instance_topk_softmax.h"
/* NOTE: instance_moe_gemm_fused.h is intentionally NOT included here. It pulls
 * in helper_rocke.helpers.tensor_view.h, whose struct rocke_tensor_descriptor
 * collides (same tag, different definition) with the copy in
 * helper_rocke.helpers.transforms.h that the WMMA-FMHA / fmha attention headers
 * pull in. The moe_gemm_fused family is bound in its own translation unit
 * (rocke_engine_moe_gemm_fused.cpp) and registered via register_moe_gemm_fused(). */
/* ---- RDNA WMMA families ---- */
#include "rocke/instance_gfx1151_wmma_fmha_fwd.h"
#include "rocke/instance_gfx1151_wmma_gemm.h"
#include "rocke/instance_gfx1151_wmma_gemm_int8.h"
#include "rocke/instance_gfx1151_wmma_gemm_iu8.h"
#include "rocke/instance_gfx1151_wmma_gemm_iu8_dequant.h"
#include "rocke/instance_gfx1201_wmma_gemm.h"
/* NOTE: instance_img2col.h is intentionally NOT included here. Its helper
 * header defines its own struct rocke_conv_problem (a peer copy of the conv
 * implicit-GEMM problem with the same tag), which clashes with
 * instance_conv_implicit_gemm.h in this translation unit. The img2col family is
 * bound in a separate translation unit (rocke_engine_img2col.cpp) and registered
 * via register_img2col(). */
}

namespace py = pybind11;

/* Defined in rocke_engine_img2col.cpp (separate TU to dodge the rocke_conv_problem
 * tag clash described above). Adds the img2col_* entry points to the module. */
void register_img2col(py::module_& m);

/* The attention families share a set of struct tags (rocke_fmha_*,
 * rocke_attention_tiled_2d_spec_t, the 3d spec tags) across many instance headers;
 * pulling them into this translation unit alongside the GEMM/conv headers is
 * fine on its own, but to keep the per-family include surface isolated they are
 * registered from a dedicated translation unit. */
void register_attention(py::module_& m);

/* The remaining FMHA-fwd variant families (appendkv / paged_prefill / varlen /
 * splitkv_decode) and the fused-MoE end-to-end orchestrator share the same
 * rocke_fmha_* struct tags as the attention TU; they are registered from their own
 * dedicated translation unit (rocke_engine_fmha_extra.cpp). */
void register_fmha_extra(py::module_& m);

/* The fused-MoE end-to-end orchestrator (fused_moe_e2e). A separate TU: its
 * internal header drags in the moe_gemm_fused / tensor_view headers whose
 * rocke_tensor_descriptor tag clashes with the FMHA common header's transforms.h.
 * It is a host-launch multi-kernel orchestrator, bound as a concatenated
 * multi-stage lower (see rocke_engine_fused_moe_e2e.cpp). */
void register_fused_moe_e2e(py::module_& m);

/* Defined in rocke_engine_moe_gemm_fused.cpp (separate TU; see the tensor_view /
 * transforms rocke_tensor_descriptor tag clash note above). */
void register_moe_gemm_fused(py::module_& m);

namespace
{

/* --------------------------------------------------------------------------
 * String-lifetime helper.
 *
 * rocke_gemm_universal_spec_t stores const char* (string literals in the C
 * emitters). When we build a spec from a Python dict we must keep the backing
 * std::string alive for as long as the spec is used. SpecHolder owns both the
 * struct and every string it points into.
 * ------------------------------------------------------------------------ */
struct SpecHolder
{
    rocke_gemm_universal_spec_t spec;
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

/* Build a rocke_gemm_universal_spec_t from a Python spec dict.
 *
 * The dict mirrors UniversalGemmSpec. Recognised layouts (both accepted):
 *   (a) flat:   {name, tile_m, tile_n, ..., pipeline, epilogue, dtype_a, ...}
 *   (b) nested: {name, tile:{...}, trait:{...}, data:{...}, wave_size, ...}
 * Nested sub-dicts take precedence; any flat key is also honoured. Every field
 * defaults to the C engine default (rocke_gemm_universal_spec_default), so a
 * minimal dict {name, tile_m, tile_n, tile_k, warp_m, warp_n} works. */
/* Fill a rocke_gemm_universal_spec_t (already default-initialised) from a spec
 * dict. `store` keeps any string fields alive for the spec's lifetime. The
 * GEMM-variant families that wrap a UniversalGemmSpec (gemm_multi_d /
 * gemm_multi_abd) reuse this to populate their nested `base` field. The dict
 * accepts both a flat key view and nested tile/trait/data sub-dicts. */
void fill_universal_spec(rocke_gemm_universal_spec_t* spec,
                         std::deque<std::string>& store,
                         const py::dict& root)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };

    auto sub = [&](const char* key) -> py::dict {
        if(root.contains(key) && py::isinstance<py::dict>(root[key]))
            return root[key].cast<py::dict>();
        return py::dict();
    };
    py::dict tile = sub("tile");
    py::dict trait = sub("trait");
    py::dict data = sub("data");

    auto pick = [&](const py::dict& nested, const char* key) -> py::dict {
        py::dict r;
        if(nested.contains(key))
            r[key] = nested[key];
        else if(root.contains(key))
            r[key] = root[key];
        return r;
    };

    {
        std::string s;
        if(dict_str(root, "name", s))
            spec->name = keep(s);
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
        spec->tile.tile_m = dict_int(t, "tile_m", spec->tile.tile_m);
        spec->tile.tile_n = dict_int(t, "tile_n", spec->tile.tile_n);
        spec->tile.tile_k = dict_int(t, "tile_k", spec->tile.tile_k);
        spec->tile.warp_m = dict_int(t, "warp_m", spec->tile.warp_m);
        spec->tile.warp_n = dict_int(t, "warp_n", spec->tile.warp_n);
        spec->tile.warp_k = dict_int(t, "warp_k", spec->tile.warp_k);
        spec->tile.warp_tile_m = dict_int(t, "warp_tile_m", spec->tile.warp_tile_m);
        spec->tile.warp_tile_n = dict_int(t, "warp_tile_n", spec->tile.warp_tile_n);
        spec->tile.warp_tile_k = dict_int(t, "warp_tile_k", spec->tile.warp_tile_k);
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
            spec->trait.pipeline = keep(s);
        if(dict_str(tr, "scheduler", s))
            spec->trait.scheduler = keep(s);
        if(dict_str(tr, "epilogue", s))
            spec->trait.epilogue = keep(s);
        spec->trait.pad_m = dict_bool(tr, "pad_m", spec->trait.pad_m);
        spec->trait.pad_n = dict_bool(tr, "pad_n", spec->trait.pad_n);
        spec->trait.pad_k = dict_bool(tr, "pad_k", spec->trait.pad_k);
        spec->trait.persistent = dict_bool(tr, "persistent", spec->trait.persistent);
        spec->trait.chiplet_swizzle = dict_bool(tr, "chiplet_swizzle", spec->trait.chiplet_swizzle);
        spec->trait.chiplet_wgm = dict_int(tr, "chiplet_wgm", spec->trait.chiplet_wgm);
        spec->trait.chiplet_num_xcds
            = dict_int(tr, "chiplet_num_xcds", spec->trait.chiplet_num_xcds);
        spec->trait.chiplet_chunk_size
            = dict_int(tr, "chiplet_chunk_size", spec->trait.chiplet_chunk_size);
        if(tr.contains("waves_per_eu") && !tr["waves_per_eu"].is_none())
        {
            spec->trait.waves_per_eu_set = true;
            spec->trait.waves_per_eu = tr["waves_per_eu"].cast<int>();
        }
        spec->trait.preshuffle_b = dict_bool(tr, "preshuffle_b", spec->trait.preshuffle_b);
        spec->trait.direct_to_lds = dict_bool(tr, "direct_to_lds", spec->trait.direct_to_lds);
        spec->trait.dtl_cache_a = dict_int(tr, "dtl_cache_a", spec->trait.dtl_cache_a);
        spec->trait.dtl_cache_b = dict_int(tr, "dtl_cache_b", spec->trait.dtl_cache_b);
        spec->trait.dtl_prefetch = dict_bool(tr, "dtl_prefetch", spec->trait.dtl_prefetch);
        spec->trait.active_tile_skip
            = dict_bool(tr, "active_tile_skip", spec->trait.active_tile_skip);
        spec->trait.lds_k_pad = dict_int(tr, "lds_k_pad", spec->trait.lds_k_pad);
        spec->trait.lds_swizzle = dict_bool(tr, "lds_swizzle", spec->trait.lds_swizzle);
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
            spec->data.dtype_a = keep(s);
        if(dict_str(da, "dtype_b", s))
            spec->data.dtype_b = keep(s);
        if(dict_str(da, "dtype_c", s))
            spec->data.dtype_c = keep(s);
        if(dict_str(da, "dtype_acc", s))
            spec->data.dtype_acc = keep(s);
        if(dict_str(da, "layout", s))
            spec->data.layout = keep(s);
    }

    /* ---- top-level scalars ---- */
    spec->wave_size = dict_int(root, "wave_size", spec->wave_size);
    spec->block_size = dict_int(root, "block_size", spec->block_size);
    spec->batched = dict_bool(root, "batched", spec->batched);
}

SpecHolder build_spec(const py::dict& root)
{
    SpecHolder h;
    h.spec = rocke_gemm_universal_spec_default();
    h.storage.reserve(16);

    /* Merge nested sub-dicts down into a flat view for convenience. */
    auto sub = [&](const char* key) -> py::dict {
        if(root.contains(key) && py::isinstance<py::dict>(root[key]))
            return root[key].cast<py::dict>();
        return py::dict();
    };
    py::dict tile = sub("tile");
    py::dict trait = sub("trait");
    py::dict data = sub("data");

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
        h.spec.tile.tile_m = dict_int(t, "tile_m", h.spec.tile.tile_m);
        h.spec.tile.tile_n = dict_int(t, "tile_n", h.spec.tile.tile_n);
        h.spec.tile.tile_k = dict_int(t, "tile_k", h.spec.tile.tile_k);
        h.spec.tile.warp_m = dict_int(t, "warp_m", h.spec.tile.warp_m);
        h.spec.tile.warp_n = dict_int(t, "warp_n", h.spec.tile.warp_n);
        h.spec.tile.warp_k = dict_int(t, "warp_k", h.spec.tile.warp_k);
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
        h.spec.trait.pad_m = dict_bool(tr, "pad_m", h.spec.trait.pad_m);
        h.spec.trait.pad_n = dict_bool(tr, "pad_n", h.spec.trait.pad_n);
        h.spec.trait.pad_k = dict_bool(tr, "pad_k", h.spec.trait.pad_k);
        h.spec.trait.persistent = dict_bool(tr, "persistent", h.spec.trait.persistent);
        h.spec.trait.chiplet_swizzle
            = dict_bool(tr, "chiplet_swizzle", h.spec.trait.chiplet_swizzle);
        h.spec.trait.chiplet_wgm = dict_int(tr, "chiplet_wgm", h.spec.trait.chiplet_wgm);
        h.spec.trait.chiplet_num_xcds
            = dict_int(tr, "chiplet_num_xcds", h.spec.trait.chiplet_num_xcds);
        h.spec.trait.chiplet_chunk_size
            = dict_int(tr, "chiplet_chunk_size", h.spec.trait.chiplet_chunk_size);
        if(tr.contains("waves_per_eu") && !tr["waves_per_eu"].is_none())
        {
            h.spec.trait.waves_per_eu_set = true;
            h.spec.trait.waves_per_eu = tr["waves_per_eu"].cast<int>();
        }
        h.spec.trait.preshuffle_b = dict_bool(tr, "preshuffle_b", h.spec.trait.preshuffle_b);
        h.spec.trait.direct_to_lds = dict_bool(tr, "direct_to_lds", h.spec.trait.direct_to_lds);
        h.spec.trait.dtl_cache_a = dict_int(tr, "dtl_cache_a", h.spec.trait.dtl_cache_a);
        h.spec.trait.dtl_cache_b = dict_int(tr, "dtl_cache_b", h.spec.trait.dtl_cache_b);
        h.spec.trait.dtl_prefetch = dict_bool(tr, "dtl_prefetch", h.spec.trait.dtl_prefetch);
        h.spec.trait.active_tile_skip
            = dict_bool(tr, "active_tile_skip", h.spec.trait.active_tile_skip);
        h.spec.trait.lds_k_pad = dict_int(tr, "lds_k_pad", h.spec.trait.lds_k_pad);
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
    h.spec.wave_size = dict_int(root, "wave_size", h.spec.wave_size);
    h.spec.block_size = dict_int(root, "block_size", h.spec.block_size);
    h.spec.batched = dict_bool(root, "batched", h.spec.batched);

    rocke_gemm_universal_spec_finalize(&h.spec);
    return h;
}

const char* arch_or_default(const std::string& arch)
{
    return arch.empty() ? "gfx950" : arch.c_str();
}

/* ---------------------------------------------------------------- bindings */

std::string gemm_lower_llvm(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    char* llvm_text = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_gemm_universal_lower_to_llvm(
        &h.spec, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if(st != ROCKE_OK || !llvm_text)
    {
        if(llvm_text)
            free(llvm_text);
        std::string msg = "rocke_engine.gemm_lower_llvm failed (status=" + std::to_string((int)st)
                          + "): " + (err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(llvm_text);
    free(llvm_text);
    return out;
}

std::string gemm_serialize_ir(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &h.spec, arch_or_default(arch));
    if(!kernel || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gemm_serialize_ir build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* text = nullptr;
    rocke_status_t st = rocke_ir_serialize(kernel, &text);
    if(st != ROCKE_OK || !text)
    {
        if(text)
            free(text);
        rocke_ir_builder_free(&b);
        throw std::runtime_error("rocke_engine.gemm_serialize_ir serialize failed (status="
                                 + std::to_string((int)st) + ")");
    }
    std::string out(text);
    free(text);
    rocke_ir_builder_free(&b);
    return out;
}

std::vector<std::string> gemm_verify(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &h.spec, arch_or_default(arch));
    if(!kernel || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gemm_verify build failed: ") + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    rocke_diag_t* diags = nullptr;
    size_t n = 0;
    rocke_verify(kernel, &diags, &n);
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
    rocke_ir_builder_free(&b);
    return out;
}

/* is_valid_spec(spec, arch) -> (bool, reason) tuple. Convenience extra. */
py::tuple gemm_is_valid(const py::dict& spec_dict, const std::string& arch)
{
    SpecHolder h = build_spec(spec_dict);
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = '\0';
    bool ok
        = rocke_gemm_universal_is_valid_spec(&h.spec, arch_or_default(arch), reason, sizeof reason);
    return py::make_tuple(ok, std::string(reason));
}

/* kernel_name(spec) -> str. Convenience extra. */
std::string gemm_kernel_name(const py::dict& spec_dict)
{
    SpecHolder h = build_spec(spec_dict);
    char name[512];
    name[0] = '\0';
    rocke_status_t st = rocke_gemm_universal_kernel_name(&h.spec, name, sizeof name);
    if(st != ROCKE_OK)
        throw std::runtime_error(
            "rocke_engine.gemm_kernel_name failed (status=" + std::to_string((int)st) + ")");
    return std::string(name);
}

/* --------------------------------------------------------------------------
 * Generic helpers shared by every additional family.
 *
 * Each family exposes a `*_lower_to_llvm(spec, arch, flavor, out_ll, err, cap)`
 * convenience plus a kernel-producing build entry; IR serialization and the
 * verifier run generically on the returned rocke_kernel_def_t*.
 * ------------------------------------------------------------------------ */

/* Wrap a `*_lower_to_llvm` result: raise on failure, else return the .ll. */
std::string take_lowered(rocke_status_t st, char* llvm_text, const char* err, const char* fn)
{
    if(st != ROCKE_OK || !llvm_text)
    {
        if(llvm_text)
            free(llvm_text);
        std::string msg = std::string(fn) + " failed (status=" + std::to_string((int)st)
                          + "): " + (err && err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(llvm_text);
    free(llvm_text);
    return out;
}

/* Serialize a built kernel to ck.dsl.ir/v1 text (raises on failure). */
std::string serialize_kernel(rocke_kernel_def_t* kernel, const char* fn)
{
    char* text = nullptr;
    rocke_status_t st = rocke_ir_serialize(kernel, &text);
    if(st != ROCKE_OK || !text)
    {
        if(text)
            free(text);
        throw std::runtime_error(std::string(fn)
                                 + " serialize failed (status=" + std::to_string((int)st) + ")");
    }
    std::string out(text);
    free(text);
    return out;
}

/* Run the verifier on a built kernel; return the diagnostic strings. */
std::vector<std::string> verify_kernel(rocke_kernel_def_t* kernel)
{
    rocke_diag_t* diags = nullptr;
    size_t n = 0;
    rocke_verify(kernel, &diags, &n);
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

/* ============================ batched GEMM ============================== */

rocke_batched_gemm_spec_t bg_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_batched_gemm_spec_t s = rocke_batched_gemm_spec_default();
    /* batched_gemm shares the universal tile/trait pair; fill a throwaway
     * universal spec to reuse the converter, then copy tile/trait across. */
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
    fill_universal_spec(&u, store, d);
    s.tile = u.tile;
    s.trait = u.trait;
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
    }
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.batch_size = dict_int(d, "batch_size", s.batch_size);
    rocke_batched_gemm_spec_finalize(&s);
    return s;
}

std::string batched_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_batched_gemm_spec_t s = bg_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_batched_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.batched_gemm_lower_llvm");
}

std::string batched_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.batched_gemm_serialize_ir",
                                rocke_batched_gemm_spec_t,
                                bg_build_spec,
                                rocke_build_batched_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> batched_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.batched_gemm_verify",
                             rocke_batched_gemm_spec_t,
                             bg_build_spec,
                             rocke_build_batched_gemm_new(&b, &s, arch_or_default(arch)));
}

py::tuple batched_gemm_is_valid(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_batched_gemm_spec_t s = bg_build_spec(d, store);
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = '\0';
    bool ok = rocke_batched_gemm_is_valid_spec(&s, arch_or_default(arch), reason, sizeof reason);
    return py::make_tuple(ok, std::string(reason));
}

/* ============================ grouped GEMM ============================== */

rocke_grouped_gemm_spec_t gg_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_grouped_gemm_spec_t s = rocke_grouped_gemm_spec_default();
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
    fill_universal_spec(&u, store, d);
    s.tile = u.tile;
    s.trait = u.trait;
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
    }
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.block_size = dict_int(d, "block_size", s.block_size);
    rocke_grouped_gemm_spec_finalize(&s);
    return s;
}

std::string grouped_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_grouped_gemm_spec_t s = gg_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_grouped_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.grouped_gemm_lower_llvm");
}

std::string grouped_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.grouped_gemm_serialize_ir",
                                rocke_grouped_gemm_spec_t,
                                gg_build_spec,
                                rocke_build_grouped_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> grouped_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.grouped_gemm_verify",
                             rocke_grouped_gemm_spec_t,
                             gg_build_spec,
                             rocke_build_grouped_gemm_new(&b, &s, arch_or_default(arch)));
}

py::tuple grouped_gemm_is_valid(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_grouped_gemm_spec_t s = gg_build_spec(d, store);
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = '\0';
    bool ok = rocke_grouped_gemm_is_valid_spec(&s, arch_or_default(arch), reason, sizeof reason);
    return py::make_tuple(ok, std::string(reason));
}

/* ================================ flatmm =============================== */

rocke_flatmm_spec_t fm_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_flatmm_spec_t s = rocke_flatmm_spec_default();
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
    fill_universal_spec(&u, store, d);
    s.tile = u.tile;
    s.trait = u.trait;
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.batch_size = dict_int(d, "batch_size", s.batch_size);
    s.preshuffle_b = dict_bool(d, "preshuffle_b", s.preshuffle_b);
    rocke_flatmm_spec_finalize(&s);
    return s;
}

std::string flatmm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_flatmm_spec_t s = fm_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_flatmm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.flatmm_lower_llvm");
}

std::string flatmm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.flatmm_serialize_ir",
                                rocke_flatmm_spec_t,
                                fm_build_spec,
                                rocke_build_flatmm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> flatmm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.flatmm_verify",
                             rocke_flatmm_spec_t,
                             fm_build_spec,
                             rocke_build_flatmm_new(&b, &s, arch_or_default(arch)));
}

/* ============================ stream-K GEMM =========================== */

rocke_streamk_gemm_spec_t sk_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_streamk_gemm_spec_t s = rocke_streamk_gemm_spec_default();
    s.M = dict_int(d, "M", s.M);
    s.N = dict_int(d, "N", s.N);
    s.K = dict_int(d, "K", s.K);
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.tile_k = dict_int(d, "tile_k", s.tile_k);
    s.num_cus = dict_int(d, "num_cus", s.num_cus);
    s.blocks_per_cu = dict_int(d, "blocks_per_cu", s.blocks_per_cu);
    s.persistent = dict_bool(d, "persistent", s.persistent);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string streamk_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_streamk_gemm_spec_t s = sk_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_streamk_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.streamk_gemm_lower_llvm");
}

std::string streamk_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.streamk_gemm_serialize_ir",
                                rocke_streamk_gemm_spec_t,
                                sk_build_spec,
                                rocke_build_streamk_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> streamk_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.streamk_gemm_verify",
                             rocke_streamk_gemm_spec_t,
                             sk_build_spec,
                             rocke_build_streamk_gemm_new(&b, &s, arch_or_default(arch)));
}

/* ========================== block-scale GEMM ========================== */

rocke_block_scale_gemm_spec_t bs_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_block_scale_gemm_spec_t s = rocke_block_scale_gemm_spec_default();
    s.M = dict_int(d, "M", s.M);
    s.N = dict_int(d, "N", s.N);
    s.K = dict_int(d, "K", s.K);
    s.preshuffle_b = dict_bool(d, "preshuffle_b", s.preshuffle_b);
    s.group_m = dict_int(d, "group_m", s.group_m);
    s.group_n = dict_int(d, "group_n", s.group_n);
    s.group_k = dict_int(d, "group_k", s.group_k);
    s.block_tile_m = dict_int(d, "block_tile_m", s.block_tile_m);
    s.block_tile_n = dict_int(d, "block_tile_n", s.block_tile_n);
    s.per_input_row = dict_bool(d, "per_input_row", s.per_input_row);
    {
        std::string v;
        if(dict_str(d, "quant_mode", v))
            s.quant_mode = keep(v);
        if(dict_str(d, "mantissa_dtype", v))
            s.mantissa_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string block_scale_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_block_scale_gemm_spec_t s = bs_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_block_scale_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.block_scale_gemm_lower_llvm");
}

std::string block_scale_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.block_scale_gemm_serialize_ir",
                                rocke_block_scale_gemm_spec_t,
                                bs_build_spec,
                                rocke_build_block_scale_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> block_scale_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.block_scale_gemm_verify",
                             rocke_block_scale_gemm_spec_t,
                             bs_build_spec,
                             rocke_build_block_scale_gemm_new(&b, &s, arch_or_default(arch)));
}

/* =============================== mx GEMM ============================== */

rocke_mx_gemm_spec_t mx_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_mx_gemm_spec_t s = rocke_mx_gemm_spec_default();
    s.M = dict_int(d, "M", s.M);
    s.N = dict_int(d, "N", s.N);
    s.K = dict_int(d, "K", s.K);
    s.group_k = dict_int(d, "group_k", s.group_k);
    s.block_tile_m = dict_int(d, "block_tile_m", s.block_tile_m);
    s.block_tile_n = dict_int(d, "block_tile_n", s.block_tile_n);
    s.per_input_row = dict_bool(d, "per_input_row", s.per_input_row);
    {
        std::string v;
        if(dict_str(d, "mantissa_dtype", v))
            s.mantissa_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string mx_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_mx_gemm_spec_t s = mx_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_mx_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.mx_gemm_lower_llvm");
}

std::string mx_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.mx_gemm_serialize_ir",
                                rocke_mx_gemm_spec_t,
                                mx_build_spec,
                                rocke_build_mx_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> mx_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.mx_gemm_verify",
                             rocke_mx_gemm_spec_t,
                             mx_build_spec,
                             rocke_build_mx_gemm_new(&b, &s, arch_or_default(arch)));
}

/* ============================== mfma GEMM ============================= */

rocke_mfma_gemm_spec_t mfma_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_mfma_gemm_spec_t s = rocke_mfma_gemm_spec_default();
    s.M = dict_int(d, "M", s.M);
    s.N = dict_int(d, "N", s.N);
    s.K = dict_int(d, "K", s.K);
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.kpack = dict_bool(d, "kpack", s.kpack);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string mfma_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_mfma_gemm_spec_t s = mfma_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_mfma_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.mfma_gemm_lower_llvm");
}

std::string mfma_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.mfma_gemm_serialize_ir",
                                rocke_mfma_gemm_spec_t,
                                mfma_build_spec,
                                rocke_build_mfma_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> mfma_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.mfma_gemm_verify",
                             rocke_mfma_gemm_spec_t,
                             mfma_build_spec,
                             rocke_build_mfma_gemm_new(&b, &s, arch_or_default(arch)));
}

py::tuple mfma_gemm_is_valid(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_mfma_gemm_spec_t s = mfma_build_spec(d, store);
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = '\0';
    bool ok = rocke_mfma_gemm_is_valid_spec(&s, arch_or_default(arch), reason, sizeof reason);
    return py::make_tuple(ok, std::string(reason));
}

/* ============================ matmul_nbits =========================== */

rocke_matmul_nbits_spec_t mn_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_matmul_nbits_spec_t s = rocke_matmul_nbits_spec_default();
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "scale_dtype", v))
            s.scale_dtype = keep(v);
        if(dict_str(d, "packing", v))
            s.packing = keep(v);
        if(dict_str(d, "family", v))
            s.family = keep(v);
    }
    s.N = dict_int(d, "N", s.N);
    s.K = dict_int(d, "K", s.K);
    /* tile geometry (flat or nested under "tile") */
    {
        py::dict t = (d.contains("tile") && py::isinstance<py::dict>(d["tile"]))
                         ? d["tile"].cast<py::dict>()
                         : d;
        s.tile.tile_m = dict_int(t, "tile_m", s.tile.tile_m);
        s.tile.tile_n = dict_int(t, "tile_n", s.tile.tile_n);
        s.tile.tile_k = dict_int(t, "tile_k", s.tile.tile_k);
        s.tile.warp_m = dict_int(t, "warp_m", s.tile.warp_m);
        s.tile.warp_n = dict_int(t, "warp_n", s.tile.warp_n);
        s.tile.warp_k = dict_int(t, "warp_k", s.tile.warp_k);
        s.tile.warp_tile_m = dict_int(t, "warp_tile_m", s.tile.warp_tile_m);
        s.tile.warp_tile_n = dict_int(t, "warp_tile_n", s.tile.warp_tile_n);
        s.tile.warp_tile_k = dict_int(t, "warp_tile_k", s.tile.warp_tile_k);
    }
    s.group_size = dict_int(d, "group_size", s.group_size);
    s.seq_len_tile = dict_int(d, "seq_len_tile", s.seq_len_tile);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.zero_points = dict_bool(d, "zero_points", s.zero_points);
    s.optimized = dict_bool(d, "optimized", s.optimized);
    rocke_matmul_nbits_spec_finalize(&s);
    return s;
}

std::string matmul_nbits_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_matmul_nbits_spec_t s = mn_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_matmul_nbits_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.matmul_nbits_lower_llvm");
}

std::string matmul_nbits_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.matmul_nbits_serialize_ir",
                                rocke_matmul_nbits_spec_t,
                                mn_build_spec,
                                rocke_build_matmul_nbits_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> matmul_nbits_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.matmul_nbits_verify",
                             rocke_matmul_nbits_spec_t,
                             mn_build_spec,
                             rocke_build_matmul_nbits_new(&b, &s, arch_or_default(arch)));
}

/* ============================ gemm_multi_d =========================== */

/* Fill the operand/load-kind side of a multi-D / multi-ABD spec from the dict.
 * d_operands is a list of (param_name, "add"|"mul") pairs. */
rocke_d_load_kind_t parse_d_load_kind(const std::string& s)
{
    if(s == "stock")
        return ROCKE_D_LOAD_STOCK;
    if(s == "tiled")
        return ROCKE_D_LOAD_TILED;
    return ROCKE_D_LOAD_VECTOR; /* "vector" / default */
}

rocke_gemm_multi_d_spec_t md_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_gemm_multi_d_spec_t s = rocke_gemm_multi_d_spec_default();

    /* nested base universal spec */
    if(d.contains("base") && py::isinstance<py::dict>(d["base"]))
    {
        py::dict base = d["base"].cast<py::dict>();
        fill_universal_spec(&s.base, store, base);
    }
    rocke_gemm_universal_spec_finalize(&s.base);

    {
        std::string v;
        if(dict_str(d, "d_dtype", v))
            s.d_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "d_load_kind", v))
            s.d_load_kind = parse_d_load_kind(v);
    }

    /* d_operands: list of (name, "add"|"mul") */
    s.num_d_operands = 0;
    if(d.contains("d_operands") && !d["d_operands"].is_none())
    {
        py::list ops = d["d_operands"].cast<py::list>();
        for(auto item : ops)
        {
            if(s.num_d_operands >= ROCKE_GEMM_MULTI_D_MAX_D)
                break;
            py::tuple pr = item.cast<py::tuple>();
            std::string nm = pr[0].cast<std::string>();
            std::string op = pr[1].cast<std::string>();
            s.d_operands[s.num_d_operands].param_name = keep(nm);
            s.d_operands[s.num_d_operands].op_is_mul = (op == "mul");
            s.num_d_operands++;
        }
    }
    return s;
}

std::string gemm_multi_d_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_gemm_multi_d_spec_t s = md_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_gemm_multi_d_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.gemm_multi_d_lower_llvm");
}

std::string gemm_multi_d_serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_gemm_multi_d_spec_t s = md_build_spec(d, store);
    rocke_kernel_def_t* k = rocke_build_gemm_multi_d(&s, arch_or_default(arch));
    if(!k)
        throw std::runtime_error("rocke_engine.gemm_multi_d_serialize_ir build failed");
    std::string out;
    try
    {
        out = serialize_kernel(k, "rocke_engine.gemm_multi_d_serialize_ir");
    }
    catch(...)
    {
        rocke_gemm_multi_d_kernel_free(k);
        throw;
    }
    rocke_gemm_multi_d_kernel_free(k);
    return out;
}

std::vector<std::string> gemm_multi_d_verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_gemm_multi_d_spec_t s = md_build_spec(d, store);
    rocke_kernel_def_t* k = rocke_build_gemm_multi_d(&s, arch_or_default(arch));
    if(!k)
        throw std::runtime_error("rocke_engine.gemm_multi_d_verify build failed");
    std::vector<std::string> out;
    try
    {
        out = verify_kernel(k);
    }
    catch(...)
    {
        rocke_gemm_multi_d_kernel_free(k);
        throw;
    }
    rocke_gemm_multi_d_kernel_free(k);
    return out;
}

/* =========================== gemm_multi_abd ========================== */

rocke_gemm_multi_abd_spec_t abd_build_spec(const py::dict& d,
                                           std::deque<std::string>& store,
                                           std::vector<rocke_gemm_abd_a_operand_t>& a_ops,
                                           std::vector<rocke_gemm_abd_b_operand_t>& b_ops)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_gemm_multi_abd_spec_t s = rocke_gemm_multi_abd_spec_default();

    if(d.contains("base") && py::isinstance<py::dict>(d["base"]))
    {
        py::dict base = d["base"].cast<py::dict>();
        fill_universal_spec(&s.base, store, base);
    }
    rocke_gemm_universal_spec_finalize(&s.base);

    {
        std::string v;
        if(dict_str(d, "d_dtype", v))
            s.d_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "d_load_kind", v))
            s.d_load_kind = parse_d_load_kind(v);
    }

    /* a_operands / b_operands: list of (name, dtype). Stored in caller-owned
     * vectors so the pointers stay valid for the spec's lifetime. */
    if(d.contains("a_operands") && !d["a_operands"].is_none())
    {
        for(auto item : d["a_operands"].cast<py::list>())
        {
            if(a_ops.size() >= ROCKE_GEMM_ABD_MAX_A)
                break;
            py::tuple pr = item.cast<py::tuple>();
            rocke_gemm_abd_a_operand_t op;
            op.name = keep(pr[0].cast<std::string>());
            op.dtype = keep(pr[1].cast<std::string>());
            a_ops.push_back(op);
        }
    }
    if(d.contains("b_operands") && !d["b_operands"].is_none())
    {
        for(auto item : d["b_operands"].cast<py::list>())
        {
            if(b_ops.size() >= ROCKE_GEMM_ABD_MAX_B)
                break;
            py::tuple pr = item.cast<py::tuple>();
            rocke_gemm_abd_b_operand_t op;
            op.name = keep(pr[0].cast<std::string>());
            op.dtype = keep(pr[1].cast<std::string>());
            b_ops.push_back(op);
        }
    }
    s.a_operands = a_ops.empty() ? nullptr : a_ops.data();
    s.num_a_operands = a_ops.size();
    s.b_operands = b_ops.empty() ? nullptr : b_ops.data();
    s.num_b_operands = b_ops.size();

    s.num_d_operands = 0;
    if(d.contains("d_operands") && !d["d_operands"].is_none())
    {
        for(auto item : d["d_operands"].cast<py::list>())
        {
            if(s.num_d_operands >= ROCKE_GEMM_MULTI_D_MAX_D)
                break;
            py::tuple pr = item.cast<py::tuple>();
            s.d_operands[s.num_d_operands].param_name = keep(pr[0].cast<std::string>());
            s.d_operands[s.num_d_operands].op_is_mul = (pr[1].cast<std::string>() == "mul");
            s.num_d_operands++;
        }
    }
    return s;
}

std::string gemm_multi_abd_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    std::vector<rocke_gemm_abd_a_operand_t> a_ops;
    std::vector<rocke_gemm_abd_b_operand_t> b_ops;
    rocke_gemm_multi_abd_spec_t s = abd_build_spec(d, store, a_ops, b_ops);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_gemm_multi_abd_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.gemm_multi_abd_lower_llvm");
}

std::string gemm_multi_abd_serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    std::vector<rocke_gemm_abd_a_operand_t> a_ops;
    std::vector<rocke_gemm_abd_b_operand_t> b_ops;
    rocke_gemm_multi_abd_spec_t s = abd_build_spec(d, store, a_ops, b_ops);
    rocke_ir_builder_t b;
    rocke_arena_t arena;
    rocke_arena_init(&arena, 0);
    rocke_kernel_def_t* k = rocke_build_gemm_multi_abd_new(&b, &arena, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gemm_multi_abd_serialize_ir build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        throw std::runtime_error(msg);
    }
    std::string out;
    try
    {
        out = serialize_kernel(k, "rocke_engine.gemm_multi_abd_serialize_ir");
    }
    catch(...)
    {
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        throw;
    }
    rocke_ir_builder_free(&b);
    rocke_arena_destroy(&arena);
    return out;
}

std::vector<std::string> gemm_multi_abd_verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    std::vector<rocke_gemm_abd_a_operand_t> a_ops;
    std::vector<rocke_gemm_abd_b_operand_t> b_ops;
    rocke_gemm_multi_abd_spec_t s = abd_build_spec(d, store, a_ops, b_ops);
    rocke_ir_builder_t b;
    rocke_arena_t arena;
    rocke_arena_init(&arena, 0);
    rocke_kernel_def_t* k = rocke_build_gemm_multi_abd_new(&b, &arena, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gemm_multi_abd_verify build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        throw std::runtime_error(msg);
    }
    std::vector<std::string> out;
    try
    {
        out = verify_kernel(k);
    }
    catch(...)
    {
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        throw;
    }
    rocke_ir_builder_free(&b);
    rocke_arena_destroy(&arena);
    return out;
}

/* ======================= conv: shared problem ======================= */

/* Fill a rocke_conv_problem_t from a "problem" sub-dict (used by
 * conv_implicit_gemm and img2col, which share the same field set). */
void fill_conv_problem(rocke_conv_problem_t* p, const py::dict& d)
{
    p->N = dict_int(d, "N", p->N);
    p->Hi = dict_int(d, "Hi", p->Hi);
    p->Wi = dict_int(d, "Wi", p->Wi);
    p->C = dict_int(d, "C", p->C);
    p->K = dict_int(d, "K", p->K);
    p->Y = dict_int(d, "Y", p->Y);
    p->X = dict_int(d, "X", p->X);
    p->sH = dict_int(d, "sH", p->sH);
    p->sW = dict_int(d, "sW", p->sW);
    p->pH = dict_int(d, "pH", p->pH);
    p->pW = dict_int(d, "pW", p->pW);
    p->dH = dict_int(d, "dH", p->dH);
    p->dW = dict_int(d, "dW", p->dW);
    p->is_3d = dict_bool(d, "is_3d", p->is_3d);
    p->Di = dict_int(d, "Di", p->Di);
    p->Z = dict_int(d, "Z", p->Z);
    p->sD = dict_int(d, "sD", p->sD);
    p->pD = dict_int(d, "pD", p->pD);
    p->dD = dict_int(d, "dD", p->dD);
}

/* ========================= conv_implicit_gemm ======================= */

rocke_implicit_gemm_conv_spec_t conv_igemm_build_spec(const py::dict& d,
                                                      std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_implicit_gemm_conv_spec_t s = rocke_implicit_gemm_conv_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_conv_problem(&s.problem, d["problem"].cast<py::dict>());

    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "pipeline", v))
            s.pipeline = keep(v);
        if(dict_str(d, "epilogue", v))
            s.epilogue = keep(v);
        if(dict_str(d, "dtype_a", v))
            s.dtype_a = keep(v);
        if(dict_str(d, "dtype_b", v))
            s.dtype_b = keep(v);
        if(dict_str(d, "dtype_d", v))
            s.dtype_d = keep(v);
        if(dict_str(d, "dtype_acc", v))
            s.dtype_acc = keep(v);
    }
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.tile_k = dict_int(d, "tile_k", s.tile_k);
    s.warp_m = dict_int(d, "warp_m", s.warp_m);
    s.warp_n = dict_int(d, "warp_n", s.warp_n);
    s.warp_tile_m = dict_int(d, "warp_tile_m", s.warp_tile_m);
    s.warp_tile_n = dict_int(d, "warp_tile_n", s.warp_tile_n);
    s.warp_tile_k = dict_int(d, "warp_tile_k", s.warp_tile_k);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.async_dma = dict_bool(d, "async_dma", s.async_dma);
    s.unroll_k = dict_bool(d, "unroll_k", s.unroll_k);
    if(d.contains("lds_k_pad") && !d["lds_k_pad"].is_none())
    {
        s.has_lds_k_pad = true;
        s.lds_k_pad = d["lds_k_pad"].cast<int>();
    }
    s.chiplet_swizzle = dict_bool(d, "chiplet_swizzle", s.chiplet_swizzle);
    s.chiplet_wgm = dict_int(d, "chiplet_wgm", s.chiplet_wgm);
    s.chiplet_num_xcds = dict_int(d, "chiplet_num_xcds", s.chiplet_num_xcds);
    s.chiplet_chunk_size = dict_int(d, "chiplet_chunk_size", s.chiplet_chunk_size);
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
    s.k0_k1_split = dict_bool(d, "k0_k1_split", s.k0_k1_split);
    s.groups = dict_int(d, "groups", s.groups);
    if(d.contains("vector_size_a") && !d["vector_size_a"].is_none())
    {
        s.has_vector_size_a = true;
        s.vector_size_a = d["vector_size_a"].cast<int>();
    }
    if(d.contains("vector_size_b") && !d["vector_size_b"].is_none())
    {
        s.has_vector_size_b = true;
        s.vector_size_b = d["vector_size_b"].cast<int>();
    }
    if(d.contains("vector_size_c") && !d["vector_size_c"].is_none())
    {
        s.has_vector_size_c = true;
        s.vector_size_c = d["vector_size_c"].cast<int>();
    }
    return s;
}

std::string conv_implicit_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_implicit_gemm_conv_spec_t s = conv_igemm_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_conv_implicit_gemm_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.conv_implicit_gemm_lower_llvm");
}

std::string conv_implicit_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY(
        "rocke_engine.conv_implicit_gemm_serialize_ir",
        rocke_implicit_gemm_conv_spec_t,
        conv_igemm_build_spec,
        rocke_build_implicit_gemm_conv_new(&b, &s, arch_or_default(arch), nullptr));
}

std::vector<std::string> conv_implicit_gemm_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY(
        "rocke_engine.conv_implicit_gemm_verify",
        rocke_implicit_gemm_conv_spec_t,
        conv_igemm_build_spec,
        rocke_build_implicit_gemm_conv_new(&b, &s, arch_or_default(arch), nullptr));
}

/* ======================= conv_direct_grouped ======================== */

/* The direct-grouped family has two distinct spec structs (16-channel and
 * 4-channel). The dict carries "kind" ("16c"|"4c") to select the path. */
void fill_direct_conv_problem(rocke_direct_conv_problem_t* p, const py::dict& d)
{
    p->N = dict_int(d, "N", p->N);
    p->H = dict_int(d, "H", p->H);
    p->W = dict_int(d, "W", p->W);
    p->groups = dict_int(d, "groups", p->groups);
    p->cpg = dict_int(d, "cpg", p->cpg);
    p->kpg = dict_int(d, "kpg", p->kpg);
    p->KH = dict_int(d, "KH", p->KH);
    p->KW = dict_int(d, "KW", p->KW);
    p->PAD = dict_int(d, "PAD", p->PAD);
    p->stride = dict_int(d, "stride", p->stride);
}

std::string conv_direct_grouped_kind(const py::dict& d)
{
    std::string kind = "16c";
    dict_str(d, "kind", kind);
    return kind;
}

rocke_direct_conv_16c_spec_t dg16_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_conv_16c_spec_t s = rocke_direct_conv_16c_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_direct_conv_problem(&s.problem, d["problem"].cast<py::dict>());
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_q = dict_int(d, "block_q", s.block_q);
    s.block_groups = dict_int(d, "block_groups", s.block_groups);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.double_buffer = dict_bool(d, "double_buffer", s.double_buffer);
    s.fold_k32 = dict_bool(d, "fold_k32", s.fold_k32);
    return s;
}

rocke_direct_conv_4c_spec_t dg4_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_conv_4c_spec_t s = rocke_direct_conv_4c_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_direct_conv_problem(&s.problem, d["problem"].cast<py::dict>());
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_q = dict_int(d, "block_q", s.block_q);
    s.block_groups = dict_int(d, "block_groups", s.block_groups);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_grouped_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        st = rocke_direct_conv_4c_lower_to_llvm(
            &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        st = rocke_direct_conv_16c_lower_to_llvm(
            &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    }
    return take_lowered(st, ll, err, "rocke_engine.conv_direct_grouped_lower_llvm");
}

std::string conv_direct_grouped_serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        k = rocke_build_direct_conv_4c_new(&b, &s, arch_or_default(arch));
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        k = rocke_build_direct_conv_16c_new(&b, &s, arch_or_default(arch));
    }
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.conv_direct_grouped_serialize_ir build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::string out = serialize_kernel(k, "rocke_engine.conv_direct_grouped_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}

std::vector<std::string> conv_direct_grouped_verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        k = rocke_build_direct_conv_4c_new(&b, &s, arch_or_default(arch));
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        k = rocke_build_direct_conv_16c_new(&b, &s, arch_or_default(arch));
    }
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.conv_direct_grouped_verify build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::vector<std::string> out = verify_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ======================= deep_fused_conv_pool ======================= */

/* This family is constructed through a factory in both reference emitters; the
 * binding mirrors the factory call exactly (tile_m is auto-derived inside). */
rocke_deep_fused_conv_pool_spec_t dfcp_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    const char* name = nullptr;
    const char* pipeline = nullptr;
    {
        std::string v;
        if(dict_str(d, "name", v))
            name = keep(v);
        if(dict_str(d, "pipeline", v))
            pipeline = keep(v);
    }
    return rocke_make_deep_fused_conv_pool_spec(
        dict_int(d, "n", 1),
        dict_int(d, "h", 0),
        dict_int(d, "w", 0),
        dict_int(d, "c", 0),
        dict_int(d, "k0", 0),
        dict_int(d, "k1", 0),
        dict_int(d, "r", 3),
        dict_int(d, "s", 3),
        dict_int(d, "pool_tile_h", 4),
        dict_int(d, "pool_tile_w", 8),
        dict_int(d, "tile_n", 32),
        dict_int(d, "tile_k", 16),
        dict_int(d, "conv1_tile_k", 0),
        dict_int(d, "warp_m", 2),
        dict_int(d, "warp_n", 1),
        dict_int(d, "warp_tile_m", 32),
        dict_int(d, "warp_tile_n", 32),
        dict_int(d, "warp_tile_k", 16),
        dict_int(d, "wave_size", 64),
        name,
        pipeline,
        dict_bool(d, "unroll_k", false),
        dict_bool(d, "async_dma", false),
        dict_bool(d, "cache_input_footprint", false),
        dict_bool(d, "direct_conv0_from_input_cache", false));
}

std::string deep_fused_conv_pool_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_deep_fused_conv_pool_spec_t s = dfcp_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_deep_fused_conv_pool_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.deep_fused_conv_pool_lower_llvm");
}

std::string deep_fused_conv_pool_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY(
        "rocke_engine.deep_fused_conv_pool_serialize_ir",
        rocke_deep_fused_conv_pool_spec_t,
        dfcp_build_spec,
        rocke_build_deep_fused_conv_pool_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> deep_fused_conv_pool_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.deep_fused_conv_pool_verify",
                             rocke_deep_fused_conv_pool_spec_t,
                             dfcp_build_spec,
                             rocke_build_deep_fused_conv_pool_new(&b, &s, arch_or_default(arch)));
}

/* ==================================================================== *
 *  Norm / elementwise / tensor-op families.
 *
 *  These specs are flat scalar/string bags; the build/lower symbols differ
 *  only in whether the build takes an arch and which lower convenience the
 *  emitter uses. Each block mirrors its tests/parity/<fam>_emit.c make_spec
 *  exactly so the binding output is byte-identical to the standalone emitter.
 * ==================================================================== */

/* ---- layernorm2d (build takes no arch) ---- */
rocke_layernorm2d_spec_t ln_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_layernorm2d_spec_t s = rocke_layernorm2d_spec_default();
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    s.save_mean_invstd = dict_bool(d, "save_mean_invstd", s.save_mean_invstd);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string layernorm2d_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_layernorm2d_spec_t s = ln_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_layernorm2d_new(&b, &s);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.layernorm2d_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.layernorm2d_lower_llvm");
}

std::string layernorm2d_serialize_ir(const py::dict& d, const std::string& arch)
{
    (void)arch;
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.layernorm2d_serialize_ir",
                                rocke_layernorm2d_spec_t,
                                ln_build_spec,
                                rocke_build_layernorm2d_new(&b, &s));
}

std::vector<std::string> layernorm2d_verify(const py::dict& d, const std::string& arch)
{
    (void)arch;
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.layernorm2d_verify",
                             rocke_layernorm2d_spec_t,
                             ln_build_spec,
                             rocke_build_layernorm2d_new(&b, &s));
}

/* ---- rmsnorm2d (build takes arch) ---- */
rocke_rmsnorm2d_spec_t rms_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_rmsnorm2d_spec_t s = rocke_rmsnorm2d_spec_default();
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    s.save_inv_rms = dict_bool(d, "save_inv_rms", s.save_inv_rms);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string rmsnorm2d_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_rmsnorm2d_spec_t s = rms_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_rmsnorm2d_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.rmsnorm2d_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll, err, sizeof err);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, err, "rocke_engine.rmsnorm2d_lower_llvm");
}

std::string rmsnorm2d_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.rmsnorm2d_serialize_ir",
                                rocke_rmsnorm2d_spec_t,
                                rms_build_spec,
                                rocke_build_rmsnorm2d_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> rmsnorm2d_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.rmsnorm2d_verify",
                             rocke_rmsnorm2d_spec_t,
                             rms_build_spec,
                             rocke_build_rmsnorm2d_new(&b, &s, arch_or_default(arch)));
}

/* ---- add_rmsnorm2d_bf16 (family lower convenience, build takes arch) ---- */
rocke_add_rmsnorm2d_bf16_spec_t arb_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_add_rmsnorm2d_bf16_spec_t s = rocke_add_rmsnorm2d_bf16_spec_default();
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    s.save_residual = dict_bool(d, "save_residual", s.save_residual);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string add_rmsnorm2d_bf16_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_add_rmsnorm2d_bf16_spec_t s = arb_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_add_rmsnorm2d_bf16_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.add_rmsnorm2d_bf16_lower_llvm");
}

std::string add_rmsnorm2d_bf16_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.add_rmsnorm2d_bf16_serialize_ir",
                                rocke_add_rmsnorm2d_bf16_spec_t,
                                arb_build_spec,
                                rocke_build_add_rmsnorm2d_bf16_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> add_rmsnorm2d_bf16_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.add_rmsnorm2d_bf16_verify",
                             rocke_add_rmsnorm2d_bf16_spec_t,
                             arb_build_spec,
                             rocke_build_add_rmsnorm2d_bf16_new(&b, &s, arch_or_default(arch)));
}

/* ---- add_rmsnorm2d_rdquant (generic lower, build takes arch) ---- */
rocke_add_rmsnorm2d_rdquant_spec_t ard_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_add_rmsnorm2d_rdquant_spec_t s = rocke_add_rmsnorm2d_rdquant_spec_default();
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    s.save_residual = dict_bool(d, "save_residual", s.save_residual);
    s.save_yscale = dict_bool(d, "save_yscale", s.save_yscale);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "out_dtype", v))
            s.out_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string add_rmsnorm2d_rdquant_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_add_rmsnorm2d_rdquant_spec_t s = ard_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_add_rmsnorm2d_rdquant_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.add_rmsnorm2d_rdquant_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.add_rmsnorm2d_rdquant_lower_llvm");
}

std::string add_rmsnorm2d_rdquant_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY(
        "rocke_engine.add_rmsnorm2d_rdquant_serialize_ir",
        rocke_add_rmsnorm2d_rdquant_spec_t,
        ard_build_spec,
        rocke_build_add_rmsnorm2d_rdquant_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> add_rmsnorm2d_rdquant_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.add_rmsnorm2d_rdquant_verify",
                             rocke_add_rmsnorm2d_rdquant_spec_t,
                             ard_build_spec,
                             rocke_build_add_rmsnorm2d_rdquant_new(&b, &s, arch_or_default(arch)));
}

/* ---- elementwise (family lower convenience, build takes no arch) ---- */
rocke_elementwise_spec_t ew_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_elementwise_spec_t s = rocke_elementwise_spec_default();
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    {
        std::string v;
        if(dict_str(d, "op", v))
            s.op = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string elementwise_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_elementwise_spec_t s = ew_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_elementwise_lower_to_llvm(
        &s, arch_or_default(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.elementwise_lower_llvm");
}

std::string elementwise_serialize_ir(const py::dict& d, const std::string& arch)
{
    (void)arch;
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.elementwise_serialize_ir",
                                rocke_elementwise_spec_t,
                                ew_build_spec,
                                rocke_build_elementwise_new(&b, &s));
}

std::vector<std::string> elementwise_verify(const py::dict& d, const std::string& arch)
{
    (void)arch;
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.elementwise_verify",
                             rocke_elementwise_spec_t,
                             ew_build_spec,
                             rocke_build_elementwise_new(&b, &s));
}

/* ---- reduce (generic lower, build takes arch) ---- */
rocke_reduce2d_spec_t rd_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_reduce2d_spec_t s = rocke_reduce2d_spec_default();
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    {
        std::string v;
        if(dict_str(d, "op", v))
            s.op = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string reduce_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_reduce2d_spec_t s = rd_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_reduce2d_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.reduce_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.reduce_lower_llvm");
}

std::string reduce_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.reduce_serialize_ir",
                                rocke_reduce2d_spec_t,
                                rd_build_spec,
                                rocke_build_reduce2d_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> reduce_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.reduce_verify",
                             rocke_reduce2d_spec_t,
                             rd_build_spec,
                             rocke_build_reduce2d_new(&b, &s, arch_or_default(arch)));
}

/* ---- pooling (nested problem, generic lower, build takes arch) ---- */
rocke_pooling2d_spec_t pool_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_pooling2d_spec_t s = rocke_pooling2d_spec_default();
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    {
        std::string v;
        if(dict_str(d, "op", v))
            s.op = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
    {
        py::dict p = d["problem"].cast<py::dict>();
        s.problem.N = dict_int(p, "N", s.problem.N);
        s.problem.H = dict_int(p, "H", s.problem.H);
        s.problem.W = dict_int(p, "W", s.problem.W);
        s.problem.C = dict_int(p, "C", s.problem.C);
        s.problem.Y = dict_int(p, "Y", s.problem.Y);
        s.problem.X = dict_int(p, "X", s.problem.X);
        s.problem.sH = dict_int(p, "sH", s.problem.sH);
        s.problem.sW = dict_int(p, "sW", s.problem.sW);
        s.problem.pH = dict_int(p, "pH", s.problem.pH);
        s.problem.pW = dict_int(p, "pW", s.problem.pW);
        s.problem.dH = dict_int(p, "dH", s.problem.dH);
        s.problem.dW = dict_int(p, "dW", s.problem.dW);
    }
    return s;
}

std::string pooling_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_pooling2d_spec_t s = pool_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_pooling2d_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.pooling_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.pooling_lower_llvm");
}

std::string pooling_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.pooling_serialize_ir",
                                rocke_pooling2d_spec_t,
                                pool_build_spec,
                                rocke_build_pooling2d_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> pooling_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.pooling_verify",
                             rocke_pooling2d_spec_t,
                             pool_build_spec,
                             rocke_build_pooling2d_new(&b, &s, arch_or_default(arch)));
}

/* ---- transpose (generic lower, build takes arch) ---- */
rocke_transpose2d_spec_t tr_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_transpose2d_spec_t s = rocke_transpose2d_spec_default();
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.vec = dict_int(d, "vec", s.vec);
    s.lds_pad = dict_int(d, "lds_pad", s.lds_pad);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "grid_order", v))
            s.grid_order = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string transpose_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_transpose2d_spec_t s = tr_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_transpose2d_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.transpose_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.transpose_lower_llvm");
}

std::string transpose_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.transpose_serialize_ir",
                                rocke_transpose2d_spec_t,
                                tr_build_spec,
                                rocke_build_transpose2d_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> transpose_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.transpose_verify",
                             rocke_transpose2d_spec_t,
                             tr_build_spec,
                             rocke_build_transpose2d_new(&b, &s, arch_or_default(arch)));
}

/* ---- permute_nd (fixed-rank arrays, generic lower, build takes arch) ---- */
rocke_permute_spec_t pm_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_permute_spec_t s = rocke_permute_spec_default();
    s.rank = dict_int(d, "rank", s.rank);
    s.block_size = dict_int(d, "block_size", s.block_size);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    if(d.contains("x_shape") && !d["x_shape"].is_none())
    {
        int i = 0;
        for(auto item : d["x_shape"].cast<py::list>())
        {
            if(i >= ROCKE_PERMUTE_MAX_RANK)
                break;
            s.x_shape[i++] = item.cast<int>();
        }
    }
    if(d.contains("perm") && !d["perm"].is_none())
    {
        int i = 0;
        for(auto item : d["perm"].cast<py::list>())
        {
            if(i >= ROCKE_PERMUTE_MAX_RANK)
                break;
            s.perm[i++] = item.cast<int>();
        }
    }
    return s;
}

std::string permute_nd_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_permute_spec_t s = pm_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_permute_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.permute_nd_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.permute_nd_lower_llvm");
}

std::string permute_nd_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.permute_nd_serialize_ir",
                                rocke_permute_spec_t,
                                pm_build_spec,
                                rocke_build_permute_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> permute_nd_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.permute_nd_verify",
                             rocke_permute_spec_t,
                             pm_build_spec,
                             rocke_build_permute_new(&b, &s, arch_or_default(arch)));
}

/* ---- smoothquant (init-by-ptr ctor, generic _ex lower, build takes arch) -- */
rocke_smoothquant_spec_t sq_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_smoothquant_spec_t s;
    rocke_smoothquant_spec_init(&s, 0);
    s.n_per_block = dict_int(d, "n_per_block", s.n_per_block);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "out_dtype", v))
            s.out_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string smoothquant_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_smoothquant_spec_t s = sq_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_smoothquant_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.smoothquant_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll, err, sizeof err);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, err, "rocke_engine.smoothquant_lower_llvm");
}

std::string smoothquant_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.smoothquant_serialize_ir",
                                rocke_smoothquant_spec_t,
                                sq_build_spec,
                                rocke_build_smoothquant_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> smoothquant_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.smoothquant_verify",
                             rocke_smoothquant_spec_t,
                             sq_build_spec,
                             rocke_build_smoothquant_new(&b, &s, arch_or_default(arch)));
}

/* ==================================================================== *
 *  MoE families.
 * ==================================================================== */

/* ---- topk_softmax ---- */
rocke_topk_softmax_spec_t tk_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_topk_softmax_spec_t s = rocke_topk_softmax_spec_default();
    s.n_per_row = dict_int(d, "n_per_row", s.n_per_row);
    s.k = dict_int(d, "k", s.k);
    s.block_size = dict_int(d, "block_size", s.block_size);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "out_dtype", v))
            s.out_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string topk_softmax_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_topk_softmax_spec_t s = tk_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_topk_softmax_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.topk_softmax_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.topk_softmax_lower_llvm");
}

std::string topk_softmax_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.topk_softmax_serialize_ir",
                                rocke_topk_softmax_spec_t,
                                tk_build_spec,
                                rocke_build_topk_softmax_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> topk_softmax_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.topk_softmax_verify",
                             rocke_topk_softmax_spec_t,
                             tk_build_spec,
                             rocke_build_topk_softmax_new(&b, &s, arch_or_default(arch)));
}

/* ---- moe_smoothquant ---- */
rocke_moe_smoothquant_spec_t msq_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_moe_smoothquant_spec_t s;
    rocke_moe_smoothquant_spec_init(
        &s, dict_int(d, "n_per_block", 0), dict_int(d, "topk", 1), dict_int(d, "experts", 1));
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    if(d.contains("tokens") && !d["tokens"].is_none())
    {
        s.tokens_set = true;
        s.tokens = d["tokens"].cast<int>();
    }
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "out_dtype", v))
            s.out_dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string moe_smoothquant_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_moe_smoothquant_spec_t s = msq_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_moe_smoothquant_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_smoothquant_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll, err, sizeof err);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, err, "rocke_engine.moe_smoothquant_lower_llvm");
}

std::string moe_smoothquant_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.moe_smoothquant_serialize_ir",
                                rocke_moe_smoothquant_spec_t,
                                msq_build_spec,
                                rocke_build_moe_smoothquant_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> moe_smoothquant_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.moe_smoothquant_verify",
                             rocke_moe_smoothquant_spec_t,
                             msq_build_spec,
                             rocke_build_moe_smoothquant_new(&b, &s, arch_or_default(arch)));
}

/* ---- moe_fused_mega ---- */
rocke_moe_fused_mega_kernel_spec_t mfm_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_moe_fused_mega_kernel_spec_t s = rocke_moe_fused_mega_kernel_spec_default();
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n_inter = dict_int(d, "tile_n_inter", s.tile_n_inter);
    s.tile_k_gu = dict_int(d, "tile_k_gu", s.tile_k_gu);
    s.tile_n_down = dict_int(d, "tile_n_down", s.tile_n_down);
    s.tile_k_down = dict_int(d, "tile_k_down", s.tile_k_down);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    rocke_moe_fused_mega_kernel_spec_finalize(&s);
    return s;
}

std::string moe_fused_mega_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_moe_fused_mega_kernel_spec_t s = mfm_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_moe_fused_mega_gemm_new(&b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_fused_mega_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.moe_fused_mega_lower_llvm");
}

std::string moe_fused_mega_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.moe_fused_mega_serialize_ir",
                                rocke_moe_fused_mega_kernel_spec_t,
                                mfm_build_spec,
                                rocke_build_moe_fused_mega_gemm_new(&b, &s, arch_or_default(arch)));
}

std::vector<std::string> moe_fused_mega_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.moe_fused_mega_verify",
                             rocke_moe_fused_mega_kernel_spec_t,
                             mfm_build_spec,
                             rocke_build_moe_fused_mega_gemm_new(&b, &s, arch_or_default(arch)));
}

/* ---- moe_fused_mega_fp8 (build takes persistent + levers; levers=NULL) ---- */
rocke_fused_mega_kernel_spec_fp8_t
    mfp_build_spec(const py::dict& d, std::deque<std::string>& store, bool* persistent)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_fused_mega_kernel_spec_fp8_t s = rocke_fused_mega_kernel_spec_fp8_default();
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n_inter = dict_int(d, "tile_n_inter", s.tile_n_inter);
    s.gate_up_k = dict_int(d, "gate_up_k", s.gate_up_k);
    s.down_k = dict_int(d, "down_k", s.down_k);
    s.use_dtla = dict_bool(d, "use_dtla", s.use_dtla);
    {
        std::string v;
        if(dict_str(d, "sched_cadence", v))
        {
            s.has_sched_cadence = true;
            s.sched_cadence = keep(v);
        }
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
    }
    if(persistent)
        *persistent = dict_bool(d, "persistent", false);
    rocke_fused_mega_kernel_spec_fp8_post_init(&s);
    return s;
}

std::string moe_fused_mega_fp8_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    bool persistent = false;
    rocke_fused_mega_kernel_spec_fp8_t s = mfp_build_spec(d, store, &persistent);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_moe_fused_mega_gemm_fp8_new(
        &b, &s, arch_or_default(arch), persistent, nullptr);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_fused_mega_fp8_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.moe_fused_mega_fp8_lower_llvm");
}

std::string moe_fused_mega_fp8_serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    bool persistent = false;
    rocke_fused_mega_kernel_spec_fp8_t s = mfp_build_spec(d, store, &persistent);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_moe_fused_mega_gemm_fp8_new(
        &b, &s, arch_or_default(arch), persistent, nullptr);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_fused_mega_fp8_serialize_ir build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::string out = serialize_kernel(k, "rocke_engine.moe_fused_mega_fp8_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}

std::vector<std::string> moe_fused_mega_fp8_verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    bool persistent = false;
    rocke_fused_mega_kernel_spec_fp8_t s = mfp_build_spec(d, store, &persistent);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_moe_fused_mega_gemm_fp8_new(
        &b, &s, arch_or_default(arch), persistent, nullptr);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_fused_mega_fp8_verify build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::vector<std::string> out = verify_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ---- fused_moe (phase-selected via "phase") ---- */
rocke_fused_moe_spec_t fmoe_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_fused_moe_spec_t s = rocke_fused_moe_spec_default();
    s.tokens = dict_int(d, "tokens", s.tokens);
    s.experts = dict_int(d, "experts", s.experts);
    s.topk = dict_int(d, "topk", s.topk);
    s.hidden = dict_int(d, "hidden", s.hidden);
    s.intermediate = dict_int(d, "intermediate", s.intermediate);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.vec = dict_int(d, "vec", s.vec);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

/* Resolve the per-phase build entry. "phase": gather (default) | silu_mul |
 * silu_mul_packed | static_scatter_gather | topk_weighted_reduce. */
rocke_kernel_def_t* fmoe_build_phase(const py::dict& d,
                                     rocke_ir_builder_t* b,
                                     const rocke_fused_moe_spec_t* s,
                                     const char* arch)
{
    std::string phase = "gather";
    dict_str(d, "phase", phase);
    if(phase == "silu_mul")
        return rocke_build_moe_silu_mul_new(b, s, arch);
    if(phase == "silu_mul_packed")
        return rocke_build_moe_silu_mul_packed_new(b, s, arch);
    if(phase == "static_scatter_gather")
        return rocke_build_moe_static_scatter_gather_new(b, s, arch);
    if(phase == "topk_weighted_reduce")
        return rocke_build_moe_topk_weighted_reduce_new(b, s, arch);
    return rocke_build_moe_gather_new(b, s, arch);
}

std::string fused_moe_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_fused_moe_spec_t s = fmoe_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = fmoe_build_phase(d, &b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.fused_moe_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.fused_moe_lower_llvm");
}

std::string fused_moe_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.fused_moe_serialize_ir",
                                rocke_fused_moe_spec_t,
                                fmoe_build_spec,
                                fmoe_build_phase(d, &b, &s, arch_or_default(arch)));
}

std::vector<std::string> fused_moe_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.fused_moe_verify",
                             rocke_fused_moe_spec_t,
                             fmoe_build_spec,
                             fmoe_build_phase(d, &b, &s, arch_or_default(arch)));
}

/* ---- moe_sorting (phase-selected via "phase") ---- */
rocke_moe_sorting_spec_t msort_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_moe_sorting_spec_t s = rocke_moe_sorting_spec_default();
    s.tokens = dict_int(d, "tokens", s.tokens);
    s.topk = dict_int(d, "topk", s.topk);
    s.experts = dict_int(d, "experts", s.experts);
    s.block_size = dict_int(d, "block_size", s.block_size);
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

rocke_kernel_def_t* msort_build_phase(const py::dict& d,
                                      rocke_ir_builder_t* b,
                                      const rocke_moe_sorting_spec_t* s,
                                      const char* arch)
{
    std::string phase = "histogram";
    dict_str(d, "phase", phase);
    if(phase == "scan")
        return rocke_build_moe_sort_scan_new(b, s, arch);
    if(phase == "scatter")
        return rocke_build_moe_sort_scatter_new(b, s, arch);
    if(phase == "persistent")
        return rocke_build_moe_sort_persistent_new(b, s, arch);
    return rocke_build_moe_sort_histogram_new(b, s, arch);
}

std::string moe_sorting_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_moe_sorting_spec_t s = msort_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = msort_build_phase(d, &b, &s, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_sorting_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.moe_sorting_lower_llvm");
}

std::string moe_sorting_serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.moe_sorting_serialize_ir",
                                rocke_moe_sorting_spec_t,
                                msort_build_spec,
                                msort_build_phase(d, &b, &s, arch_or_default(arch)));
}

std::vector<std::string> moe_sorting_verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.moe_sorting_verify",
                             rocke_moe_sorting_spec_t,
                             msort_build_spec,
                             msort_build_phase(d, &b, &s, arch_or_default(arch)));
}

/* ==================================================================== *
 *  RDNA WMMA families. The C LLVM lowerer ports the gfx1151/gfx1201
 *  (RDNA) backends, so these emit real non-empty .ll at their arch.
 * ==================================================================== */

/* ---- gfx1151_wmma_gemm ---- */
rocke_wmma_gemm_gfx1151_spec_t w1151_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_gemm_gfx1151_spec_t s = rocke_wmma_gemm_gfx1151_spec_default();
    s.block_x_is_m = dict_bool(d, "block_x_is_m", s.block_x_is_m);
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1151_wmma_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_gemm_gfx1151_spec_t s = w1151_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_wmma_gemm_gfx1151_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gfx1151_wmma_gemm_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1151_wmma_gemm_lower_llvm");
}

std::string gfx1151_wmma_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.gfx1151_wmma_gemm_serialize_ir",
                                rocke_wmma_gemm_gfx1151_spec_t,
                                w1151_build_spec,
                                rocke_build_wmma_gemm_gfx1151_new(&b, &s, a));
}

std::vector<std::string> gfx1151_wmma_gemm_verify(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.gfx1151_wmma_gemm_verify",
                             rocke_wmma_gemm_gfx1151_spec_t,
                             w1151_build_spec,
                             rocke_build_wmma_gemm_gfx1151_new(&b, &s, a));
}

/* ---- gfx1151_wmma_gemm_int8 ---- */
rocke_wmma_gemm_int8_spec_t w1151i8_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_gemm_int8_spec_t s = rocke_wmma_gemm_int8_spec_default();
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1151_wmma_gemm_int8_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_gemm_int8_spec_t s = w1151i8_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_wmma_gemm_int8_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gfx1151_wmma_gemm_int8_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1151_wmma_gemm_int8_lower_llvm");
}

std::string gfx1151_wmma_gemm_int8_serialize_ir(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.gfx1151_wmma_gemm_int8_serialize_ir",
                                rocke_wmma_gemm_int8_spec_t,
                                w1151i8_build_spec,
                                rocke_build_wmma_gemm_int8_new(&b, &s, a));
}

std::vector<std::string> gfx1151_wmma_gemm_int8_verify(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.gfx1151_wmma_gemm_int8_verify",
                             rocke_wmma_gemm_int8_spec_t,
                             w1151i8_build_spec,
                             rocke_build_wmma_gemm_int8_new(&b, &s, a));
}

/* ---- gfx1151_wmma_gemm_iu8 ---- */
rocke_wmma_gemm_iu8_spec_t w1151iu8_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_gemm_iu8_spec_t s = rocke_wmma_gemm_iu8_spec_default();
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1151_wmma_gemm_iu8_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_gemm_iu8_spec_t s = w1151iu8_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_wmma_gemm_iu8_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gfx1151_wmma_gemm_iu8_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1151_wmma_gemm_iu8_lower_llvm");
}

std::string gfx1151_wmma_gemm_iu8_serialize_ir(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.gfx1151_wmma_gemm_iu8_serialize_ir",
                                rocke_wmma_gemm_iu8_spec_t,
                                w1151iu8_build_spec,
                                rocke_build_wmma_gemm_iu8_new(&b, &s, a));
}

std::vector<std::string> gfx1151_wmma_gemm_iu8_verify(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.gfx1151_wmma_gemm_iu8_verify",
                             rocke_wmma_gemm_iu8_spec_t,
                             w1151iu8_build_spec,
                             rocke_build_wmma_gemm_iu8_new(&b, &s, a));
}

/* ---- gfx1151_wmma_gemm_iu8_dequant ---- */
rocke_wmma_gemm_iu8_dequant_spec_t w1151iu8dq_build_spec(const py::dict& d,
                                                         std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_gemm_iu8_dequant_spec_t s = rocke_wmma_gemm_iu8_dequant_spec_default();
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1151_wmma_gemm_iu8_dequant_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_gemm_iu8_dequant_spec_t s = w1151iu8dq_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_wmma_gemm_iu8_dequant_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gfx1151_wmma_gemm_iu8_dequant_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1151_wmma_gemm_iu8_dequant_lower_llvm");
}

std::string gfx1151_wmma_gemm_iu8_dequant_serialize_ir(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.gfx1151_wmma_gemm_iu8_dequant_serialize_ir",
                                rocke_wmma_gemm_iu8_dequant_spec_t,
                                w1151iu8dq_build_spec,
                                rocke_build_wmma_gemm_iu8_dequant_new(&b, &s, a));
}

std::vector<std::string> gfx1151_wmma_gemm_iu8_dequant_verify(const py::dict& d,
                                                              const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.gfx1151_wmma_gemm_iu8_dequant_verify",
                             rocke_wmma_gemm_iu8_dequant_spec_t,
                             w1151iu8dq_build_spec,
                             rocke_build_wmma_gemm_iu8_dequant_new(&b, &s, a));
}

/* ---- gfx1151_wmma_fmha_fwd ---- */
rocke_fmha_mask_mode_t parse_fmha_mask(const std::string& s)
{
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

rocke_wmma_fmha_fwd_spec_t w1151fmha_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_fmha_fwd_spec_t s = rocke_wmma_fmha_fwd_spec_default();
    s.head_size = dict_int(d, "head_size", s.head_size);
    s.num_query_heads = dict_int(d, "num_query_heads", s.num_query_heads);
    s.num_kv_heads = dict_int(d, "num_kv_heads", s.num_kv_heads);
    s.v_lds_stage = dict_bool(d, "v_lds_stage", s.v_lds_stage);
    s.sliding_window = dict_int(d, "sliding_window", s.sliding_window);
    {
        std::string v;
        if(dict_str(d, "mask_mode", v))
            s.mask_mode = parse_fmha_mask(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1151_wmma_fmha_fwd_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_fmha_fwd_spec_t s = w1151fmha_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    char kname[512];
    kname[0] = '\0';
    rocke_wmma_fmha_fwd_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k = rocke_build_wmma_fmha_fwd(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gfx1151_wmma_fmha_fwd_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1151_wmma_fmha_fwd_lower_llvm");
}

std::string gfx1151_wmma_fmha_fwd_serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_fmha_fwd_spec_t s = w1151fmha_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    char kname[512];
    kname[0] = '\0';
    rocke_wmma_fmha_fwd_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k = rocke_build_wmma_fmha_fwd(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_engine.gfx1151_wmma_fmha_fwd_serialize_ir build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::string out = serialize_kernel(k, "rocke_engine.gfx1151_wmma_fmha_fwd_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}

std::vector<std::string> gfx1151_wmma_fmha_fwd_verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_fmha_fwd_spec_t s = w1151fmha_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1151" : arch.c_str();
    char kname[512];
    kname[0] = '\0';
    rocke_wmma_fmha_fwd_kernel_name(&s, kname, sizeof kname);
    rocke_ir_builder_init(&b, kname);
    rocke_kernel_def_t* k = rocke_build_wmma_fmha_fwd(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gfx1151_wmma_fmha_fwd_verify build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::vector<std::string> out = verify_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ---- gfx1201_wmma_gemm ---- */
rocke_wmma_gemm_gfx1201_spec_t w1201_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wmma_gemm_gfx1201_spec_t s = rocke_wmma_gemm_gfx1201_spec_default();
    {
        std::string v;
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string gfx1201_wmma_gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_wmma_gemm_gfx1201_spec_t s = w1201_build_spec(d, store);
    rocke_ir_builder_t b;
    const char* a = arch.empty() ? "gfx1201" : arch.c_str();
    rocke_kernel_def_t* k = rocke_build_wmma_gemm_gfx1201_new(&b, &s, a);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.gfx1201_wmma_gemm_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, a, &ll);
    rocke_ir_builder_free(&b);
    return take_lowered(st, ll, nullptr, "rocke_engine.gfx1201_wmma_gemm_lower_llvm");
}

std::string gfx1201_wmma_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1201" : arch.c_str();
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.gfx1201_wmma_gemm_serialize_ir",
                                rocke_wmma_gemm_gfx1201_spec_t,
                                w1201_build_spec,
                                rocke_build_wmma_gemm_gfx1201_new(&b, &s, a));
}

std::vector<std::string> gfx1201_wmma_gemm_verify(const py::dict& d, const std::string& arch)
{
    const char* a = arch.empty() ? "gfx1201" : arch.c_str();
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.gfx1201_wmma_gemm_verify",
                             rocke_wmma_gemm_gfx1201_spec_t,
                             w1201_build_spec,
                             rocke_build_wmma_gemm_gfx1201_new(&b, &s, a));
}

/* --------------------------------------------------------------------------
 * Family-agnostic lower-from-serialized-IR.
 *
 * lower_serialized_ir(ir_text, arch) parses serialized ck.dsl.ir/v1 text back
 * into a kernel (rocke_ir_parse) and lowers it to AMDGPU LLVM IR text
 * (rocke_lower_kernel_to_llvm_ex), exactly as tests/ir_lower_cli.cpp does. This
 * is the engine-side endpoint the ROCKE_BACKEND=cpp default uses: a Python
 * front end serializes whatever KernelDef it built (any family) and this
 * reproduces the .ll the C++ engine would emit, with no per-family C builder
 * involved. Every failure (parse or lower) is converted to a Python exception;
 * the engine's extern "C" boundary never aborts/terminates.
 * ------------------------------------------------------------------------ */
std::string lower_serialized_ir(const std::string& ir_text,
                                const std::string& arch,
                                const std::string& flavor)
{
    if(ir_text.empty())
    {
        throw std::runtime_error("rocke_engine.lower_serialized_ir: empty IR input");
    }
    const char* a = arch.empty() ? "gfx950" : arch.c_str();

    /* flavor: "" => AUTO (resolve from env / ROCm version); "llvm20"/"llvm22"
     * pin the intrinsic declaration shape. An unrecognised non-empty flavor is
     * rejected so callers get the same hard error the Python lowerer raises. */
    rocke_llvm_flavor_t fl = ROCKE_LLVM_FLAVOR_AUTO;
    if(!flavor.empty())
    {
        fl = rocke_llvm_flavor_from_name(flavor.c_str());
        if(fl == ROCKE_LLVM_FLAVOR_AUTO)
        {
            throw std::runtime_error(
                std::string("rocke_engine.lower_serialized_ir: unknown LLVM flavor '") + flavor
                + "' (expected 'llvm20' or 'llvm22')");
        }
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "lower_serialized_ir") != ROCKE_OK)
    {
        throw std::runtime_error("rocke_engine.lower_serialized_ir: builder init failed (OOM)");
    }

    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t st = rocke_ir_parse(ir_text.c_str(), &b, &kernel);
    if(st != ROCKE_OK || !kernel)
    {
        const char* m = rocke_ir_builder_error(&b);
        std::string msg = std::string("rocke_engine.lower_serialized_ir: parse failed (status ")
                          + std::to_string((int)st)
                          + "): " + ((m && *m) ? m : "unknown parse error");
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }

    char* out_ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    st = rocke_lower_kernel_to_llvm_ex(kernel, fl, a, &out_ll, err, sizeof(err));
    if(st != ROCKE_OK || !out_ll)
    {
        std::string msg = std::string("rocke_engine.lower_serialized_ir: lower failed for arch '")
                          + a + "' (status " + std::to_string((int)st)
                          + "): " + (err[0] ? err : "unknown lowering error");
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }

    std::string result(out_ll);
    std::free(out_ll);
    rocke_ir_builder_free(&b);
    return result;
}

} // namespace

PYBIND11_MODULE(rocke_engine, m)
{
    m.doc() = "pybind11 binding for the C++ rocke engine (universal GEMM "
              "family). Foundation of the ROCKE_BACKEND=cpp dual-backend path.";

    /* ---- engine freshness / provenance stamp ----
     * build_id() is a content hash of the engine sources this .so was built
     * from; engine_version() is the human-readable version. Harnesses and the
     * provider compare this against the build-id recorded with a set of
     * prebuilt artifacts (emitters, manifests) to fail loud on a stale/mixed
     * build instead of producing spurious mismatches. These are artifact
     * stamps only; they never touch the emitted IR. */
    m.def(
        "build_id",
        []() { return std::string(rocke_build_id()); },
        "Content hash of the engine sources this module was built from. "
        "Compare against the build-id recorded with prebuilt artifacts to "
        "detect a stale/mixed build.");
    m.def(
        "engine_version",
        []() { return std::string(rocke_engine_version()); },
        "Human-readable engine version of this module.");

    /* ---- family-agnostic lower-from-serialized-IR ----
     * The keystone of the ROCKE_BACKEND=cpp default for Python-authored
     * kernels: a Python front end serializes any KernelDef to ck.dsl.ir/v1
     * text, and this reproduces the exact AMDGPU .ll the C++ engine emits --
     * no per-family C builder involved. Mirrors tests/ir_lower_cli.cpp. */
    m.def("lower_serialized_ir",
          &lower_serialized_ir,
          py::arg("ir_text"),
          py::arg("arch") = "gfx950",
          py::arg("flavor") = "",
          "Parse serialized ck.dsl.ir/v1 text and lower it to AMDGPU LLVM IR "
          "(.ll) text via the C++ engine. Family-agnostic; byte-identical to "
          "the Python lowerer for the same serialized IR. flavor='' resolves "
          "the LLVM flavor automatically; 'llvm20'/'llvm22' pin it.");

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
          "Byte-identical to rocke.core.ir_serialize.serialize for the same spec.");
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

    /* ---- GEMM variant families ---- */
    auto reg3 = [&](const char* prefix,
                    std::string (*lower)(const py::dict&, const std::string&),
                    std::string (*ser)(const py::dict&, const std::string&),
                    std::vector<std::string> (*ver)(const py::dict&, const std::string&)) {
        m.def((std::string(prefix) + "_lower_llvm").c_str(),
              lower,
              py::arg("spec"),
              py::arg("arch") = "gfx950");
        m.def((std::string(prefix) + "_serialize_ir").c_str(),
              ser,
              py::arg("spec"),
              py::arg("arch") = "gfx950");
        m.def((std::string(prefix) + "_verify").c_str(),
              ver,
              py::arg("spec"),
              py::arg("arch") = "gfx950");
    };

    reg3(
        "batched_gemm", &batched_gemm_lower_llvm, &batched_gemm_serialize_ir, &batched_gemm_verify);
    m.def("batched_gemm_is_valid",
          &batched_gemm_is_valid,
          py::arg("spec"),
          py::arg("arch") = "gfx950");

    reg3(
        "grouped_gemm", &grouped_gemm_lower_llvm, &grouped_gemm_serialize_ir, &grouped_gemm_verify);
    m.def("grouped_gemm_is_valid",
          &grouped_gemm_is_valid,
          py::arg("spec"),
          py::arg("arch") = "gfx950");

    reg3("flatmm", &flatmm_lower_llvm, &flatmm_serialize_ir, &flatmm_verify);
    reg3(
        "streamk_gemm", &streamk_gemm_lower_llvm, &streamk_gemm_serialize_ir, &streamk_gemm_verify);
    reg3("block_scale_gemm",
         &block_scale_gemm_lower_llvm,
         &block_scale_gemm_serialize_ir,
         &block_scale_gemm_verify);
    reg3("mx_gemm", &mx_gemm_lower_llvm, &mx_gemm_serialize_ir, &mx_gemm_verify);
    reg3("mfma_gemm", &mfma_gemm_lower_llvm, &mfma_gemm_serialize_ir, &mfma_gemm_verify);
    m.def("mfma_gemm_is_valid", &mfma_gemm_is_valid, py::arg("spec"), py::arg("arch") = "gfx950");

    reg3(
        "matmul_nbits", &matmul_nbits_lower_llvm, &matmul_nbits_serialize_ir, &matmul_nbits_verify);
    reg3(
        "gemm_multi_d", &gemm_multi_d_lower_llvm, &gemm_multi_d_serialize_ir, &gemm_multi_d_verify);
    reg3("gemm_multi_abd",
         &gemm_multi_abd_lower_llvm,
         &gemm_multi_abd_serialize_ir,
         &gemm_multi_abd_verify);

    /* ---- convolution families ---- */
    reg3("conv_implicit_gemm",
         &conv_implicit_gemm_lower_llvm,
         &conv_implicit_gemm_serialize_ir,
         &conv_implicit_gemm_verify);
    reg3("conv_direct_grouped",
         &conv_direct_grouped_lower_llvm,
         &conv_direct_grouped_serialize_ir,
         &conv_direct_grouped_verify);
    register_img2col(m); /* separate TU; see note at top of file */
    reg3("deep_fused_conv_pool",
         &deep_fused_conv_pool_lower_llvm,
         &deep_fused_conv_pool_serialize_ir,
         &deep_fused_conv_pool_verify);

    /* ---- norm / elementwise / tensor-op families ---- */
    reg3("layernorm2d", &layernorm2d_lower_llvm, &layernorm2d_serialize_ir, &layernorm2d_verify);
    reg3("rmsnorm2d", &rmsnorm2d_lower_llvm, &rmsnorm2d_serialize_ir, &rmsnorm2d_verify);
    reg3("add_rmsnorm2d_bf16",
         &add_rmsnorm2d_bf16_lower_llvm,
         &add_rmsnorm2d_bf16_serialize_ir,
         &add_rmsnorm2d_bf16_verify);
    reg3("add_rmsnorm2d_rdquant",
         &add_rmsnorm2d_rdquant_lower_llvm,
         &add_rmsnorm2d_rdquant_serialize_ir,
         &add_rmsnorm2d_rdquant_verify);
    reg3("elementwise", &elementwise_lower_llvm, &elementwise_serialize_ir, &elementwise_verify);
    reg3("reduce", &reduce_lower_llvm, &reduce_serialize_ir, &reduce_verify);
    reg3("pooling", &pooling_lower_llvm, &pooling_serialize_ir, &pooling_verify);
    reg3("transpose", &transpose_lower_llvm, &transpose_serialize_ir, &transpose_verify);
    reg3("permute_nd", &permute_nd_lower_llvm, &permute_nd_serialize_ir, &permute_nd_verify);
    reg3("smoothquant", &smoothquant_lower_llvm, &smoothquant_serialize_ir, &smoothquant_verify);

    /* ---- MoE families ---- */
    reg3(
        "topk_softmax", &topk_softmax_lower_llvm, &topk_softmax_serialize_ir, &topk_softmax_verify);
    reg3("moe_smoothquant",
         &moe_smoothquant_lower_llvm,
         &moe_smoothquant_serialize_ir,
         &moe_smoothquant_verify);
    reg3("moe_fused_mega",
         &moe_fused_mega_lower_llvm,
         &moe_fused_mega_serialize_ir,
         &moe_fused_mega_verify);
    reg3("moe_fused_mega_fp8",
         &moe_fused_mega_fp8_lower_llvm,
         &moe_fused_mega_fp8_serialize_ir,
         &moe_fused_mega_fp8_verify);
    /* fused_moe / moe_sorting / moe_gemm_fused expose multiple phase/sub-kernels;
     * the phase is selected via the spec dict's "phase" / "kind" key. */
    m.def(
        "fused_moe_lower_llvm", &fused_moe_lower_llvm, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("fused_moe_serialize_ir",
          &fused_moe_serialize_ir,
          py::arg("spec"),
          py::arg("arch") = "gfx950");
    m.def("fused_moe_verify", &fused_moe_verify, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("moe_sorting_lower_llvm",
          &moe_sorting_lower_llvm,
          py::arg("spec"),
          py::arg("arch") = "gfx950");
    m.def("moe_sorting_serialize_ir",
          &moe_sorting_serialize_ir,
          py::arg("spec"),
          py::arg("arch") = "gfx950");
    m.def("moe_sorting_verify", &moe_sorting_verify, py::arg("spec"), py::arg("arch") = "gfx950");
    register_moe_gemm_fused(m); /* separate TU; see tensor_view/transforms note */

    /* ---- RDNA WMMA families ---- */
    reg3("gfx1151_wmma_gemm",
         &gfx1151_wmma_gemm_lower_llvm,
         &gfx1151_wmma_gemm_serialize_ir,
         &gfx1151_wmma_gemm_verify);
    reg3("gfx1151_wmma_gemm_int8",
         &gfx1151_wmma_gemm_int8_lower_llvm,
         &gfx1151_wmma_gemm_int8_serialize_ir,
         &gfx1151_wmma_gemm_int8_verify);
    reg3("gfx1151_wmma_gemm_iu8",
         &gfx1151_wmma_gemm_iu8_lower_llvm,
         &gfx1151_wmma_gemm_iu8_serialize_ir,
         &gfx1151_wmma_gemm_iu8_verify);
    reg3("gfx1151_wmma_gemm_iu8_dequant",
         &gfx1151_wmma_gemm_iu8_dequant_lower_llvm,
         &gfx1151_wmma_gemm_iu8_dequant_serialize_ir,
         &gfx1151_wmma_gemm_iu8_dequant_verify);
    reg3("gfx1151_wmma_fmha_fwd",
         &gfx1151_wmma_fmha_fwd_lower_llvm,
         &gfx1151_wmma_fmha_fwd_serialize_ir,
         &gfx1151_wmma_fmha_fwd_verify);
    reg3("gfx1201_wmma_gemm",
         &gfx1201_wmma_gemm_lower_llvm,
         &gfx1201_wmma_gemm_serialize_ir,
         &gfx1201_wmma_gemm_verify);

    /* ---- attention families (separate TU; shared fmha/tiled struct tags) ---- */
    register_attention(m);

    /* ---- remaining FMHA-fwd variants (separate TU) ---- */
    register_fmha_extra(m);

    /* ---- fused-MoE end-to-end orchestrator (separate TU) ---- */
    register_fused_moe_e2e(m);
}
