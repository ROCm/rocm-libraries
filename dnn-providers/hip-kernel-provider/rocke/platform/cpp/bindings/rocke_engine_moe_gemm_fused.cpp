// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_moe_gemm_fused.cpp -- moe_gemm_fused family bindings.
 * Kept in its own translation unit because instance_moe_gemm_fused.h pulls in
 * helper_rocke.helpers.tensor_view.h, whose struct rocke_tensor_descriptor has
 * the same tag as (but a different definition from) the copy in
 * helper_rocke.helpers.transforms.h that the fmha/WMMA-FMHA attention headers
 * pull in. Compiling both in one translation unit is a C++ redefinition; here
 * only the moe_gemm_fused header chain is present, so the tag is unique.
 *
 * register_moe_gemm_fused(m) is called from the main module file. The family has
 * three sub-kernels (gate_up_silu / interleaved / down_reduce) selected by the
 * spec dict "kind" key; the three spec structs share a common field layout.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "rocke/instance_moe_gemm_fused.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

int g_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
    return dflt;
}
bool g_bool(const py::dict& d, const char* key, bool dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<bool>();
    return dflt;
}
bool g_str(const py::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

/* The three spec types share {name, tile, trait, wave_size, block_size, dtype,
 * grouped}; one templated filler covers them all. */
template <typename SpecT>
void fill(SpecT* s, const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& v) -> const char* {
        store.push_back(v);
        return store.back().c_str();
    };
    if(d.contains("tile") && py::isinstance<py::dict>(d["tile"]))
    {
        py::dict t = d["tile"].cast<py::dict>();
        s->tile.tile_m = g_int(t, "tile_m", s->tile.tile_m);
        s->tile.tile_n = g_int(t, "tile_n", s->tile.tile_n);
        s->tile.tile_k = g_int(t, "tile_k", s->tile.tile_k);
        s->tile.warp_m = g_int(t, "warp_m", s->tile.warp_m);
        s->tile.warp_n = g_int(t, "warp_n", s->tile.warp_n);
        s->tile.warp_k = g_int(t, "warp_k", s->tile.warp_k);
        s->tile.warp_tile_m = g_int(t, "warp_tile_m", s->tile.warp_tile_m);
        s->tile.warp_tile_n = g_int(t, "warp_tile_n", s->tile.warp_tile_n);
        s->tile.warp_tile_k = g_int(t, "warp_tile_k", s->tile.warp_tile_k);
    }
    s->grouped = g_bool(d, "grouped", s->grouped);
    if(d.contains("trait") && py::isinstance<py::dict>(d["trait"]))
    {
        py::dict tr = d["trait"].cast<py::dict>();
        std::string v;
        if(g_str(tr, "epilogue", v))
            s->trait.epilogue = keep(v);
        s->trait.pad_m = g_bool(tr, "pad_m", s->trait.pad_m);
        s->trait.pad_n = g_bool(tr, "pad_n", s->trait.pad_n);
    }
    std::string v;
    if(g_str(d, "dtype", v))
        s->dtype = keep(v);
    if(g_str(d, "name", v))
        s->name = keep(v);
}

std::string kind_of(const py::dict& d)
{
    std::string kind = "gate_up_silu";
    g_str(d, "kind", kind);
    return kind;
}

rocke_kernel_def_t* build_kind(const py::dict& d,
                               rocke_ir_builder_t* b,
                               std::deque<std::string>& store,
                               const char* arch)
{
    std::string kind = kind_of(d);
    if(kind == "interleaved")
    {
        rocke_moe_interleaved_gate_up_silu_gemm_spec_t s
            = rocke_moe_interleaved_gate_up_silu_gemm_spec_default();
        fill(&s, d, store);
        rocke_moe_interleaved_gate_up_silu_gemm_spec_finalize(&s);
        return rocke_build_moe_interleaved_gate_up_silu_gemm_new(b, &s, arch);
    }
    if(kind == "down_reduce")
    {
        rocke_moe_down_reduce_gemm_spec_t s = rocke_moe_down_reduce_gemm_spec_default();
        fill(&s, d, store);
        rocke_moe_down_reduce_gemm_spec_finalize(&s);
        return rocke_build_moe_down_reduce_gemm_new(b, &s, arch);
    }
    rocke_moe_gate_up_silu_gemm_spec_t s = rocke_moe_gate_up_silu_gemm_spec_default();
    fill(&s, d, store);
    rocke_moe_gate_up_silu_gemm_spec_finalize(&s);
    return rocke_build_moe_gate_up_silu_gemm_new(b, &s, arch);
}

const char* arch_or_default(const std::string& arch)
{
    return arch.empty() ? "gfx950" : arch.c_str();
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_gemm_fused_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    rocke_ir_builder_free(&b);
    if(st != ROCKE_OK || !ll)
    {
        if(ll)
            free(ll);
        throw std::runtime_error("rocke_engine.moe_gemm_fused_lower_llvm lower failed (status="
                                 + std::to_string((int)st) + ")");
    }
    std::string out(ll);
    free(ll);
    return out;
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_gemm_fused_serialize_ir build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* t = nullptr;
    rocke_status_t st = rocke_ir_serialize(k, &t);
    rocke_ir_builder_free(&b);
    if(st != ROCKE_OK || !t)
    {
        if(t)
            free(t);
        throw std::runtime_error("rocke_engine.moe_gemm_fused_serialize_ir serialize failed");
    }
    std::string out(t);
    free(t);
    return out;
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_engine.moe_gemm_fused_verify build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
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
    rocke_ir_builder_free(&b);
    return out;
}

} // namespace

void register_moe_gemm_fused(py::module_& m)
{
    m.def("moe_gemm_fused_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def(
        "moe_gemm_fused_serialize_ir", &serialize_ir, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("moe_gemm_fused_verify", &verify, py::arg("spec"), py::arg("arch") = "gfx950");
}
