// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/rocke_library_engine_conv.cpp -- convolution family
 * bindings of the rocke_library_engine module: implicit-GEMM conv (fwd and the
 * two-stage wgrad with its Stage-2 workspace reduce), direct grouped and
 * non-grouped conv, and the deep fused conv+pool family. Each takes the spec
 * dict the matching *_spec_to_dict in kernels/conv/engine_bridge.py produces
 * and drives the library C++ mirror, byte-identical to the Python builder.
 *
 * Moved out of the engine's rocke_engine.cpp with the conv C++ mirrors; the
 * small dict/lowering helpers below are this translation unit's own copies of
 * the engine module's, as rocke_library_engine_img2col.cpp keeps its own.
 *
 * register_conv(m) is called from rocke_library_engine.cpp.
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

extern "C" {
#include "rocke/arena.h"
#include "rocke/instance_conv_direct_dgrad_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_nongrouped.h"
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm_wgrad.h"
#include "rocke/instance_conv_wgrad_workspace_reduce.h"
#include "rocke/instance_deep_fused_conv_pool.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

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
    p->groups = dict_int(d, "groups", p->groups);
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
    s.group_merge = dict_int(d, "group_merge", s.group_merge);
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
    require_arch(arch);
    std::deque<std::string> store;
    rocke_implicit_gemm_conv_spec_t s = conv_igemm_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_conv_implicit_gemm_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_implicit_gemm_lower_llvm");
}

std::string conv_implicit_gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_implicit_gemm_serialize_ir",
                                rocke_implicit_gemm_conv_spec_t,
                                conv_igemm_build_spec,
                                rocke_build_implicit_gemm_conv_new(&b, &s, arch.c_str(), nullptr));
}

std::vector<std::string> conv_implicit_gemm_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_implicit_gemm_verify",
                             rocke_implicit_gemm_conv_spec_t,
                             conv_igemm_build_spec,
                             rocke_build_implicit_gemm_conv_new(&b, &s, arch.c_str(), nullptr));
}

/* ======================= conv_direct_grouped ======================== */

/* The direct-grouped family has three spec structs (16-channel, 4-channel and
 * the generic DirectConvSpec). The dict carries "kind" ("16c"|"4c"|"generic")
 * to select the path. */
void fill_direct_conv_problem(rocke_direct_conv_problem_t* p,
                              const py::dict& d,
                              std::deque<std::string>* store = nullptr)
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
    p->dil_h = dict_int(d, "dil_h", p->dil_h);
    p->dil_w = dict_int(d, "dil_w", p->dil_w);
    if(store != nullptr)
    {
        /* dtype is a borrowed C string; park it in `store` so it outlives the
         * spec (same pattern as the spec name). */
        std::string v;
        if(dict_str(d, "dtype", v))
        {
            store->push_back(v);
            p->dtype = store->back().c_str();
        }
    }
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
        fill_direct_conv_problem(&s.problem, d["problem"].cast<py::dict>(), &store);
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_q = dict_int(d, "block_q", s.block_q);
    s.block_groups = dict_int(d, "block_groups", s.block_groups);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.dgrad_fused_weights = dict_bool(d, "dgrad_fused_weights", s.dgrad_fused_weights);
    s.dgrad_weights_lds = dict_bool(d, "dgrad_weights_lds", s.dgrad_weights_lds);
    s.stage_rows = dict_bool(d, "stage_rows", s.stage_rows);
    s.waves_q = dict_int(d, "waves_q", s.waves_q);
    return s;
}

/* The generic DirectConvSpec ("generic"): every dataclass field but the
 * nested problem, so a knob the dict carries always reaches the C builder. */
rocke_direct_conv_spec_t dgg_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_conv_spec_t s = rocke_direct_conv_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_direct_conv_problem(&s.problem, d["problem"].cast<py::dict>(), &store);
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_q = dict_int(d, "block_q", s.block_q);
    s.block_groups = dict_int(d, "block_groups", s.block_groups);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.double_buffer = dict_bool(d, "double_buffer", s.double_buffer);
    s.block_h = dict_int(d, "block_h", s.block_h);
    s.waves_q = dict_int(d, "waves_q", s.waves_q);
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.runtime_k_loop = dict_bool(d, "runtime_k_loop", s.runtime_k_loop);
    s.persistent_grid = dict_bool(d, "persistent_grid", s.persistent_grid);
    s.persistent_rounds = dict_int(d, "persistent_rounds", s.persistent_rounds);
    s.fold_k32 = dict_bool(d, "fold_k32", s.fold_k32);
    s.runtime_channels = dict_bool(d, "runtime_channels", s.runtime_channels);
    s.k_atoms = dict_int(d, "k_atoms", s.k_atoms);
    s.m_tiles = dict_int(d, "m_tiles", s.m_tiles);
    s.preload_weights = dict_bool(d, "preload_weights", s.preload_weights);
    s.dgrad_fused_weights = dict_bool(d, "dgrad_fused_weights", s.dgrad_fused_weights);
    s.dgrad_weights_lds = dict_bool(d, "dgrad_weights_lds", s.dgrad_weights_lds);
    s.waves_per_eu = dict_int(d, "waves_per_eu", s.waves_per_eu);
    s.row_guard = dict_bool(d, "row_guard", s.row_guard);
    s.prefetch_rows = dict_int(d, "prefetch_rows", s.prefetch_rows);
    s.lds_only_sync = dict_bool(d, "lds_only_sync", s.lds_only_sync);
    s.waves_m = dict_int(d, "waves_m", s.waves_m);
    s.xcd_tiles = dict_bool(d, "xcd_tiles", s.xcd_tiles);
    s.lds_pad = dict_int(d, "lds_pad", s.lds_pad);
    s.stage_out = dict_bool(d, "stage_out", s.stage_out);
    s.row_exit = dict_bool(d, "row_exit", s.row_exit);
    return s;
}

std::string conv_direct_grouped_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        st = rocke_direct_conv_4c_lower_to_llvm(
            &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    }
    else if(conv_direct_grouped_kind(d) == "generic")
    {
        rocke_direct_conv_spec_t s = dgg_build_spec(d, store);
        st = rocke_direct_conv_lower_to_llvm(
            &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        st = rocke_direct_conv_16c_lower_to_llvm(
            &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    }
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_grouped_lower_llvm");
}

std::string conv_direct_grouped_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        k = rocke_build_direct_conv_4c_new(&b, &s, arch.c_str());
    }
    else if(conv_direct_grouped_kind(d) == "generic")
    {
        rocke_direct_conv_spec_t s = dgg_build_spec(d, store);
        k = rocke_build_direct_conv_new(&b, &s, arch.c_str());
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        k = rocke_build_direct_conv_16c_new(&b, &s, arch.c_str());
    }
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_library_engine.conv_direct_grouped_serialize_ir build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::string out = serialize_kernel(k, "rocke_library_engine.conv_direct_grouped_serialize_ir");
    rocke_ir_builder_free(&b);
    return out;
}

std::vector<std::string> conv_direct_grouped_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    if(conv_direct_grouped_kind(d) == "4c")
    {
        rocke_direct_conv_4c_spec_t s = dg4_build_spec(d, store);
        k = rocke_build_direct_conv_4c_new(&b, &s, arch.c_str());
    }
    else if(conv_direct_grouped_kind(d) == "generic")
    {
        rocke_direct_conv_spec_t s = dgg_build_spec(d, store);
        k = rocke_build_direct_conv_new(&b, &s, arch.c_str());
    }
    else
    {
        rocke_direct_conv_16c_spec_t s = dg16_build_spec(d, store);
        k = rocke_build_direct_conv_16c_new(&b, &s, arch.c_str());
    }
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_library_engine.conv_direct_grouped_verify build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    std::vector<std::string> out = verify_kernel(k);
    rocke_ir_builder_free(&b);
    return out;
}

/* ======================== conv_direct_dgrad_wmma ========================== */

/* Wave32 WMMA direct dgrad (gfx1250). The dict mirrors DirectDgradWmmaSpec
 * with the problem nested; `sched` may be None. */
rocke_direct_dgrad_wmma_spec_t ddgw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_dgrad_wmma_spec_t s = rocke_direct_dgrad_wmma_spec_default();
    /* The problem's shape fields are required, as in DirectConvProblem: a
     * missing one would silently read as the zero default. */
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("DirectDgradWmmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("DirectDgradWmmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        if(dict_str(pd, "dtype", v))
            s.problem.dtype = keep(v);
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.waves_c = dict_int(d, "waves_c", s.waves_c);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_c = dict_int(d, "tiles_c", s.tiles_c);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.tile_w = dict_int(d, "tile_w", s.tile_w);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    if(d.contains("sched") && !d["sched"].is_none())
        s.sched = keep(d["sched"].cast<std::string>());
    return s;
}

std::string conv_direct_dgrad_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_dgrad_wmma_spec_t s = ddgw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_direct_dgrad_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_dgrad_wmma_lower_llvm");
}

std::string conv_direct_dgrad_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_dgrad_wmma_serialize_ir",
                                rocke_direct_dgrad_wmma_spec_t,
                                ddgw_build_spec,
                                rocke_build_direct_dgrad_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_dgrad_wmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_dgrad_wmma_verify",
                             rocke_direct_dgrad_wmma_spec_t,
                             ddgw_build_spec,
                             rocke_build_direct_dgrad_wmma_new(&b, &s, arch.c_str()));
}

/* is_valid_dgrad_wmma_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_dgrad_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_dgrad_wmma_spec_t s = ddgw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_direct_dgrad_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_direct_dgrad_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_direct_dgrad_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_direct_dgrad_wmma_lds_bytes(&s));
}

/* ======================== conv_direct_nongrouped ========================== */

/* Non-grouped (groups == 1, or grouped=True) direct conv. The dict mirrors
 * DirectNongroupedConvSpec with the problem nested; `iglp` / `waves_per_eu` may
 * be None (absent knob). */
rocke_direct_conv_nongrouped_spec_t dnongrouped_build_spec(const py::dict& d,
                                                           std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_conv_nongrouped_spec_t s = rocke_direct_conv_nongrouped_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
    {
        py::dict pd = d["problem"].cast<py::dict>();
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        if(dict_str(pd, "dtype", v))
            s.problem.dtype = keep(v);
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "atom", v))
            s.atom = keep(v);
    }
    s.tile_h = dict_int(d, "tile_h", s.tile_h);
    s.tile_w = dict_int(d, "tile_w", s.tile_w);
    s.tile_k = dict_int(d, "tile_k", s.tile_k);
    s.ck = dict_int(d, "ck", s.ck);
    s.waves_m = dict_int(d, "waves_m", s.waves_m);
    s.waves_n = dict_int(d, "waves_n", s.waves_n);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.lds_pad = dict_int(d, "lds_pad", s.lds_pad);
    s.chiplet_swizzle = dict_bool(d, "chiplet_swizzle", s.chiplet_swizzle);
    s.swizzle_wgm = dict_int(d, "swizzle_wgm", s.swizzle_wgm);
    s.chiplet_chunk = dict_int(d, "chiplet_chunk", s.chiplet_chunk);
    s.num_xcds = dict_int(d, "num_xcds", s.num_xcds);
    s.double_buffer = dict_bool(d, "double_buffer", s.double_buffer);
    s.split_c = dict_int(d, "split_c", s.split_c);
    s.grouped = dict_bool(d, "grouped", s.grouped);
    s.group_merge = dict_int(d, "group_merge", s.group_merge);
    /* The C spec encodes None as a sentinel (-1 / 0). An explicit value in the
     * sentinel range would silently read as None here, so reject it with the
     * message Python's validate() raises for the same spec. */
    if(d.contains("iglp") && !d["iglp"].is_none())
    {
        s.iglp = d["iglp"].cast<int>();
        if(s.iglp < 0)
            throw std::runtime_error("iglp must be None or >= 0 (got " + std::to_string(s.iglp)
                                     + ")");
    }
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
        if(s.waves_per_eu < 1)
            throw std::runtime_error("waves_per_eu must be None or >= 1 (got "
                                     + std::to_string(s.waves_per_eu) + ")");
    }
    return s;
}

std::string conv_direct_nongrouped_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_conv_nongrouped_spec_t s = dnongrouped_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_direct_conv_nongrouped_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_nongrouped_lower_llvm");
}

std::string conv_direct_nongrouped_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_nongrouped_serialize_ir",
                                rocke_direct_conv_nongrouped_spec_t,
                                dnongrouped_build_spec,
                                rocke_build_direct_conv_nongrouped_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_nongrouped_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_nongrouped_verify",
                             rocke_direct_conv_nongrouped_spec_t,
                             dnongrouped_build_spec,
                             rocke_build_direct_conv_nongrouped_new(&b, &s, arch.c_str()));
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
    require_arch(arch);
    std::deque<std::string> store;
    rocke_deep_fused_conv_pool_spec_t s = dfcp_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_deep_fused_conv_pool_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.deep_fused_conv_pool_lower_llvm");
}

std::string deep_fused_conv_pool_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.deep_fused_conv_pool_serialize_ir",
                                rocke_deep_fused_conv_pool_spec_t,
                                dfcp_build_spec,
                                rocke_build_deep_fused_conv_pool_new(&b, &s, arch.c_str()));
}

std::vector<std::string> deep_fused_conv_pool_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.deep_fused_conv_pool_verify",
                             rocke_deep_fused_conv_pool_spec_t,
                             dfcp_build_spec,
                             rocke_build_deep_fused_conv_pool_new(&b, &s, arch.c_str()));
}

/* =================== conv_implicit_gemm_wgrad (two-stage) ================== */

rocke_implicit_gemm_conv_wgrad_spec_t conv_wgrad_build_spec(const py::dict& d,
                                                            std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_implicit_gemm_conv_wgrad_spec_t s = rocke_implicit_gemm_conv_wgrad_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_conv_problem(&s.problem, d["problem"].cast<py::dict>());

    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.tile_k = dict_int(d, "tile_k", s.tile_k);
    s.warp_m = dict_int(d, "warp_m", s.warp_m);
    s.warp_n = dict_int(d, "warp_n", s.warp_n);
    s.warp_tile_m = dict_int(d, "warp_tile_m", s.warp_tile_m);
    s.warp_tile_n = dict_int(d, "warp_tile_n", s.warp_tile_n);
    s.warp_tile_k = dict_int(d, "warp_tile_k", s.warp_tile_k);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.split_k = dict_int(d, "split_k", s.split_k);
    s.two_stage = dict_bool(d, "two_stage", s.two_stage);
    s.ws_replicas = dict_int(d, "ws_replicas", s.ws_replicas);
    s.prefetch_addr = dict_bool(d, "prefetch_addr", s.prefetch_addr);
    s.streamk_ctas = dict_int(d, "streamk_ctas", s.streamk_ctas);
    s.group_merge = dict_int(d, "group_merge", s.group_merge);
    s.async_dma = dict_bool(d, "async_dma", s.async_dma);
    s.unroll_k = dict_bool(d, "unroll_k", s.unroll_k);
    s.lds_k_outer = dict_bool(d, "lds_k_outer", s.lds_k_outer);
    s.chiplet_swizzle = dict_bool(d, "chiplet_swizzle", s.chiplet_swizzle);
    s.chiplet_wgm = dict_int(d, "chiplet_wgm", s.chiplet_wgm);
    s.chiplet_num_xcds = dict_int(d, "chiplet_num_xcds", s.chiplet_num_xcds);
    s.chiplet_chunk_size = dict_int(d, "chiplet_chunk_size", s.chiplet_chunk_size);
    /* Optional ints: absent or None keeps Python's None (has_* = false).
     * Only the fields the C builder or validator honours are read: the C wgrad
     * builder does not apply lds_k_pad or vector_size_a/b, so reading them
     * would tag the kernel name (e.g. "pad<N>") without changing its body. */
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
    if(d.contains("vector_size_c") && !d["vector_size_c"].is_none())
    {
        s.has_vector_size_c = true;
        s.vector_size_c = d["vector_size_c"].cast<int>();
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "streamk", v))
            s.streamk = keep(v);
        if(dict_str(d, "streamk_reduction", v))
            s.streamk_reduction = keep(v);
        if(dict_str(d, "dtype_a", v))
            s.dtype_a = keep(v);
        if(dict_str(d, "dtype_b", v))
            s.dtype_b = keep(v);
        if(dict_str(d, "dtype_d", v))
            s.dtype_d = keep(v);
        if(dict_str(d, "pipeline", v))
            s.pipeline = keep(v);
        if(dict_str(d, "epilogue", v))
            s.epilogue = keep(v);
    }
    return s;
}

std::string conv_wgrad_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_implicit_gemm_conv_wgrad_spec_t s = conv_wgrad_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_implicit_gemm_conv_wgrad_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_library_engine.conv_wgrad_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, resolve_python_llvm_flavor(), arch.c_str(), &ll);
    guard.reset();
    return take_lowered(st, ll, nullptr, "rocke_library_engine.conv_wgrad_lower_llvm");
}

std::string conv_wgrad_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_wgrad_serialize_ir",
                                rocke_implicit_gemm_conv_wgrad_spec_t,
                                conv_wgrad_build_spec,
                                rocke_build_implicit_gemm_conv_wgrad_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_wgrad_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_wgrad_verify",
                             rocke_implicit_gemm_conv_wgrad_spec_t,
                             conv_wgrad_build_spec,
                             rocke_build_implicit_gemm_conv_wgrad_new(&b, &s, arch.c_str()));
}

/* =================== conv_wgrad_workspace_reduce (Stage 2) ================ */

rocke_wgrad_reduce_spec_t conv_wgrad_reduce_build_spec(const py::dict& d,
                                                       std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_wgrad_reduce_spec_t s = rocke_wgrad_reduce_spec_default();
    s.tile_m = dict_int(d, "tile_m", s.tile_m);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.wg_M = dict_int(d, "wg_M", s.wg_M);
    s.wg_N = dict_int(d, "wg_N", s.wg_N);
    s.groups = dict_int(d, "groups", s.groups);
    /* Must track the Stage 1 spec's ws_replicas: folding fewer slabs than
     * Stage 1 wrote drops part of the sum, folding more reads past the
     * scratch. Defaulting here instead of parsing would silently do one or the
     * other for every non-default Stage 1 count. */
    s.ws_replicas = dict_int(d, "ws_replicas", s.ws_replicas);
    {
        std::string v;
        if(dict_str(d, "dtype_d", v))
            s.dtype_d = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
        if(dict_str(d, "problem_short", v))
            s.problem_short = keep(v);
    }
    return s;
}

std::string conv_wgrad_reduce_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_wgrad_reduce_spec_t s = conv_wgrad_reduce_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_wgrad_workspace_reduce_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_library_engine.conv_wgrad_reduce_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    rocke_status_t st
        = rocke_lower_kernel_to_llvm(k, resolve_python_llvm_flavor(), arch.c_str(), &ll);
    guard.reset();
    return take_lowered(st, ll, nullptr, "rocke_library_engine.conv_wgrad_reduce_lower_llvm");
}

std::string conv_wgrad_reduce_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_wgrad_reduce_serialize_ir",
                                rocke_wgrad_reduce_spec_t,
                                conv_wgrad_reduce_build_spec,
                                rocke_build_wgrad_workspace_reduce_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_wgrad_reduce_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_wgrad_reduce_verify",
                             rocke_wgrad_reduce_spec_t,
                             conv_wgrad_reduce_build_spec,
                             rocke_build_wgrad_workspace_reduce_new(&b, &s, arch.c_str()));
}

} // namespace

void register_conv(py::module_& m)
{
    auto reg3 = [&](const char* prefix,
                    std::string (*lower)(const py::dict&, const std::string&),
                    std::string (*ser)(const py::dict&, const std::string&),
                    std::vector<std::string> (*ver)(const py::dict&, const std::string&)) {
        m.def(
            (std::string(prefix) + "_lower_llvm").c_str(), lower, py::arg("spec"), py::arg("arch"));
        m.def(
            (std::string(prefix) + "_serialize_ir").c_str(), ser, py::arg("spec"), py::arg("arch"));
        m.def((std::string(prefix) + "_verify").c_str(), ver, py::arg("spec"), py::arg("arch"));
    };

    /* ---- convolution families ---- */
    reg3("conv_implicit_gemm",
         &conv_implicit_gemm_lower_llvm,
         &conv_implicit_gemm_serialize_ir,
         &conv_implicit_gemm_verify);
    reg3("conv_direct_grouped",
         &conv_direct_grouped_lower_llvm,
         &conv_direct_grouped_serialize_ir,
         &conv_direct_grouped_verify);
    reg3("conv_direct_nongrouped",
         &conv_direct_nongrouped_lower_llvm,
         &conv_direct_nongrouped_serialize_ir,
         &conv_direct_nongrouped_verify);
    reg3("conv_direct_dgrad_wmma",
         &conv_direct_dgrad_wmma_lower_llvm,
         &conv_direct_dgrad_wmma_serialize_ir,
         &conv_direct_dgrad_wmma_verify);
    m.def("conv_direct_dgrad_wmma_describe",
          &conv_direct_dgrad_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("deep_fused_conv_pool",
         &deep_fused_conv_pool_lower_llvm,
         &deep_fused_conv_pool_serialize_ir,
         &deep_fused_conv_pool_verify);

    /* ---- wgrad two-stage family ---- */
    reg3("conv_wgrad", conv_wgrad_lower_llvm, conv_wgrad_serialize_ir, conv_wgrad_verify);
    /* conv_wgrad_reduce is used by the byte-identity gate
     * (platform/tools/check_byte_identity.py) and by external callers driving
     * Stage 2 via the Python engine API directly.  The benchmark drives Stage 2
     * via the Python instance layer (rocke.instances.common.conv_wgrad_workspace_reduce),
     * not through this binding, so the registration may appear unused in that context. */
    reg3("conv_wgrad_reduce",
         conv_wgrad_reduce_lower_llvm,
         conv_wgrad_reduce_serialize_ir,
         conv_wgrad_reduce_verify);
    m.def(
        "conv_wgrad_workspace_bytes",
        [](const py::dict& d) -> size_t {
            std::deque<std::string> store;
            rocke_implicit_gemm_conv_wgrad_spec_t s = conv_wgrad_build_spec(d, store);
            return rocke_wgrad_conv_workspace_bytes(&s);
        },
        py::arg("spec"),
        "Return workspace bytes for the two-stage wgrad path.\n"
        "Formula: groups * ws_replicas * wg_M * wg_N * 4 (always f32).\n"
        "Returns 0 when two_stage=false, or split_k <= 1.");
    m.def(
        "conv_wgrad_is_valid_spec",
        [](const py::dict& d, const std::string& arch) -> py::tuple {
            const char* a = require_arch(arch);
            std::deque<std::string> store;
            rocke_implicit_gemm_conv_wgrad_spec_t s = conv_wgrad_build_spec(d, store);
            char reason[1024] = {0};
            const bool ok
                = rocke_implicit_gemm_conv_wgrad_is_valid_spec(&s, a, reason, sizeof(reason));
            return py::make_tuple(ok, ok ? std::string("ok") : std::string(reason));
        },
        py::arg("spec"),
        py::arg("arch"),
        "C port of is_valid_wgrad_spec: (ok, reason), reason 'ok' when accepted.");
    m.def(
        "conv_wgrad_streamk_default_pool",
        [](const std::string& reduction, const std::string& arch) -> int {
            return rocke_wgrad_conv_streamk_default_pool(reduction.c_str(), require_arch(arch));
        },
        py::arg("reduction"),
        py::arg("arch"),
        "C port of wgrad_streamk_default_pool: the CTA pool of a streamk_ctas=-1 launch.");
    m.def(
        "conv_wgrad_streamk_host",
        [](const py::dict& d, const std::string& arch) -> py::dict {
            /* Every helper resolves an auto pool (-1) for `arch` itself, as the
             * Python wgrad_streamk_*(spec, arch=arch) twins do. Errors are
             * ValueError, matching Python. */
            const char* a = require_arch(arch);
            std::deque<std::string> store;
            rocke_implicit_gemm_conv_wgrad_spec_t s = conv_wgrad_build_spec(d, store);
            int grid[3] = {0, 0, 0};
            if(rocke_wgrad_conv_streamk_grid(&s, a, grid) != ROCKE_OK)
                throw py::value_error(
                    "rocke_library_engine.conv_wgrad_streamk_host: needs streamk != 'off' and "
                    "streamk_ctas == -1 or > 0");
            size_t flags_bytes = 0;
            size_t partials_bytes = 0;
            if(rocke_wgrad_conv_streamk_workspace_layout(&s, a, &flags_bytes, &partials_bytes)
               != ROCKE_OK)
                throw py::value_error("rocke_library_engine.conv_wgrad_streamk_host: invalid "
                                      "stream-K workspace layout");
            py::dict out;
            out["grid"] = py::make_tuple(grid[0], grid[1], grid[2]);
            out["flags_bytes"] = flags_bytes;
            out["partials_bytes"] = partials_bytes;
            out["workspace_bytes"] = rocke_wgrad_conv_streamk_workspace_bytes(&s, a);
            out["is_deterministic"] = rocke_wgrad_conv_spec_is_deterministic(&s);
            return out;
        },
        py::arg("spec"),
        py::arg("arch"),
        "Stream-K wgrad host plan of the C port: launch grid, workspace layout\n"
        "and determinism, for cross-checking the Python wgrad_streamk_* helpers.");
}
