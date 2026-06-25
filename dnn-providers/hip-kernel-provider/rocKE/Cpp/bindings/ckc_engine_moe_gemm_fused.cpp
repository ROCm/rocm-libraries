// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/ckc_engine_moe_gemm_fused.cpp -- moe_gemm_fused family bindings.
 * Kept in its own translation unit because instance_moe_gemm_fused.h pulls in
 * helper_ck_dsl.helpers.tensor_view.h, whose struct ckc_tensor_descriptor has
 * the same tag as (but a different definition from) the copy in
 * helper_ck_dsl.helpers.transforms.h that the fmha/WMMA-FMHA attention headers
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
#include "ckc/instance_moe_gemm_fused.h"
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
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

ckc_kernel_def_t* build_kind(const py::dict& d,
                             ckc_ir_builder_t* b,
                             std::deque<std::string>& store,
                             const char* arch)
{
    std::string kind = kind_of(d);
    if(kind == "interleaved")
    {
        ckc_moe_interleaved_gate_up_silu_gemm_spec_t s
            = ckc_moe_interleaved_gate_up_silu_gemm_spec_default();
        fill(&s, d, store);
        ckc_moe_interleaved_gate_up_silu_gemm_spec_finalize(&s);
        return ckc_build_moe_interleaved_gate_up_silu_gemm_new(b, &s, arch);
    }
    if(kind == "down_reduce")
    {
        ckc_moe_down_reduce_gemm_spec_t s = ckc_moe_down_reduce_gemm_spec_default();
        fill(&s, d, store);
        ckc_moe_down_reduce_gemm_spec_finalize(&s);
        return ckc_build_moe_down_reduce_gemm_new(b, &s, arch);
    }
    ckc_moe_gate_up_silu_gemm_spec_t s = ckc_moe_gate_up_silu_gemm_spec_default();
    fill(&s, d, store);
    ckc_moe_gate_up_silu_gemm_spec_finalize(&s);
    return ckc_build_moe_gate_up_silu_gemm_new(b, &s, arch);
}

const char* arch_or_default(const std::string& arch)
{
    return arch.empty() ? "gfx950" : arch.c_str();
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !ckc_ir_builder_ok(&b))
    {
        std::string msg = std::string("ckc_engine.moe_gemm_fused_lower_llvm build failed: ")
                          + ckc_ir_builder_error(&b);
        ckc_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    ckc_status_t st = ckc_lower_kernel_to_llvm(k, CKC_LLVM_FLAVOR_AUTO, arch_or_default(arch), &ll);
    ckc_ir_builder_free(&b);
    if(st != CKC_OK || !ll)
    {
        if(ll)
            free(ll);
        throw std::runtime_error("ckc_engine.moe_gemm_fused_lower_llvm lower failed (status="
                                 + std::to_string((int)st) + ")");
    }
    std::string out(ll);
    free(ll);
    return out;
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !ckc_ir_builder_ok(&b))
    {
        std::string msg = std::string("ckc_engine.moe_gemm_fused_serialize_ir build failed: ")
                          + ckc_ir_builder_error(&b);
        ckc_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    char* t = nullptr;
    ckc_status_t st = ckc_ir_serialize(k, &t);
    ckc_ir_builder_free(&b);
    if(st != CKC_OK || !t)
    {
        if(t)
            free(t);
        throw std::runtime_error("ckc_engine.moe_gemm_fused_serialize_ir serialize failed");
    }
    std::string out(t);
    free(t);
    return out;
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    ckc_ir_builder_t b;
    ckc_kernel_def_t* k = build_kind(d, &b, store, arch_or_default(arch));
    if(!k || !ckc_ir_builder_ok(&b))
    {
        std::string msg = std::string("ckc_engine.moe_gemm_fused_verify build failed: ")
                          + ckc_ir_builder_error(&b);
        ckc_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    ckc_diag_t* diags = nullptr;
    size_t n = 0;
    ckc_verify(k, &diags, &n);
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

} // namespace

void register_moe_gemm_fused(py::module_& m)
{
    m.def("moe_gemm_fused_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def(
        "moe_gemm_fused_serialize_ir", &serialize_ir, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("moe_gemm_fused_verify", &verify, py::arg("spec"), py::arg("arch") = "gfx950");
}
