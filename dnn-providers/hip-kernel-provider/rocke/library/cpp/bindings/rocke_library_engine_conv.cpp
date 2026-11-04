// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/rocke_library_engine_conv.cpp -- convolution family
 * bindings of the rocke_library_engine module: implicit-GEMM conv (fwd with its
 * split-K Stage-2 reduce, and the two-stage wgrad with its Stage-2 workspace
 * reduce), direct grouped and non-grouped conv, the gfx1250 WMMA direct fwd /
 * dgrad / small-channel / patch-stem kernels, and the deep fused conv+pool
 * family. Img2col and Winograd bind from their own translation units. Each takes the spec
 * dict the matching *_spec_to_dict in kernels/conv/engine_bridge.py produces
 * and drives the library C++ mirror, byte-identical to the Python builder.
 *
 * Moved out of the engine's rocke_engine.cpp with the conv C++ mirrors. The
 * dict/lowering helpers come from the engine's family_glue.hpp (namespace
 * rocke_bind), shared with rocke_engine.cpp rather than copied.
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

#include "conv_knob_fields.hpp"
#include "family_glue.hpp"

extern "C" {
#include "rocke/arch_domain.h"
#include "rocke/arena.h"
#include "rocke/instance_conv_dgrad_interior.h"
#include "rocke/instance_conv_dgrad_patch_wmma.h"
#include "rocke/instance_conv_dgrad_smallc_wmma.h"
#include "rocke/instance_conv_direct_dgrad_pp_mfma.h"
#include "rocke/instance_conv_direct_dgrad_wmma.h"
#include "rocke/instance_conv_direct_dw_fwd_wmma.h"
#include "rocke/instance_conv_direct_fwd_flex_mfma.h"
#include "rocke/instance_conv_direct_fwd_gstream_wmma.h"
#include "rocke/instance_conv_direct_fwd_pp_mfma.h"
#include "rocke/instance_conv_direct_fwd_tapouter_wmma.h"
#include "rocke/instance_conv_direct_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_nongrouped.h"
#include "rocke/instance_conv_fwd_small_mfma.h"
#include "rocke/instance_conv_fwd_small_wmma.h"
#include "rocke/instance_conv_fwd_split_k_reduce.h"
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm_dgrad.h"
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

using rocke_bind::dict_bool;
using rocke_bind::dict_int;
using rocke_bind::dict_str;
using rocke_bind::serialize_kernel;
using rocke_bind::take_lowered;
using rocke_bind::verify_kernel;

/* ======================= conv: shared problem ======================= */

/* A conv-spec dict key the C++ engine has no field for: a non-default value
 * would be dropped and the engine would build a different kernel under the
 * spec's name, so refuse it (ValueError) instead. */
void refuse_unported_conv_field(const py::dict& d, const char* family, const char* key)
{
    if(!d.contains(key))
        return;
    py::object v = d[key];
    if(v.is_none() || (py::isinstance<py::bool_>(v) && !v.cast<bool>())
       || (py::isinstance<py::int_>(v) && v.cast<long long>() == 0))
        return;
    throw py::value_error(std::string("rocke_library_engine: ") + family + " field '" + key
                          + "' has no C++ engine twin (non-default value)");
}

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
    /* The depth fields: None (a 2-D dataclasses.asdict(ConvProblem)) keeps
     * the 2-D default; a set depth makes the problem 3-D unless the dict says
     * otherwise (asdict has no is_3d key). */
    bool has_depth = false;
    auto depth = [&](const char* key, int dflt) -> int {
        if(!d.contains(key) || d[key].is_none())
            return dflt;
        has_depth = true;
        return d[key].cast<int>();
    };
    p->Di = depth("Di", p->Di);
    p->Z = depth("Z", p->Z);
    p->sD = depth("sD", p->sD);
    p->pD = depth("pD", p->pD);
    p->dD = depth("dD", p->dD);
    p->is_3d = dict_bool(d, "is_3d", p->is_3d || has_depth);
}

/* ========================= conv_implicit_gemm ======================= */

/* ConvAccumulatorEpilogue dict {bias, scale, relu, clamp_min, clamp_max}
 * (clamp_* None => unset). Absent keys keep the identity default. */
void fill_conv_acc_epilogue(rocke_conv_acc_epilogue_t* e, const py::dict& d)
{
    if(d.contains("bias") && !d["bias"].is_none())
        e->bias = d["bias"].cast<double>();
    if(d.contains("scale") && !d["scale"].is_none())
        e->scale = d["scale"].cast<double>();
    e->relu = dict_bool(d, "relu", e->relu);
    if(d.contains("clamp_min") && !d["clamp_min"].is_none())
    {
        e->has_clamp_min = true;
        e->clamp_min = d["clamp_min"].cast<double>();
    }
    if(d.contains("clamp_max") && !d["clamp_max"].is_none())
    {
        e->has_clamp_max = true;
        e->clamp_max = d["clamp_max"].cast<double>();
    }
}

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
    /* An explicit pipeline=None / epilogue=None is a malformed spec (Python
     * rejects it), not a request for the "mem" / "default" defaults: pass
     * NULL so the engine rejects it. Only an absent key keeps the default. */
    if(d.contains("pipeline") && d["pipeline"].is_none())
        s.pipeline = NULL;
    if(d.contains("epilogue") && d["epilogue"].is_none())
        s.epilogue = NULL;

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
        /* ConvDataSpec.mma_dtype: absent or None keeps the NULL default (the
         * MMA consumes the storage dtype); "tf32" builds the TF32 kernel. */
        if(dict_str(d, "mma_dtype", v))
            s.mma_dtype = keep(v);
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
    /* Split-K forward: a different ABI and epilogue (and the cluster form a
     * cluster launch shape); dropped, the engine would build the plain kernel. */
    s.split_k = dict_int(d, "split_k", s.split_k);
    s.split_k_cluster = dict_int(d, "split_k_cluster", s.split_k_cluster);
    /* The wavelet load-wave count sizes the launch block and the load-wave
     * loaders; dropped, a non-default count builds a different kernel. */
    s.num_load_waves = dict_int(d, "num_load_waves", s.num_load_waves);
    /* The fused accumulator epilogue: dropped, the engine would build the
     * identity epilogue (and admit split-K specs Python rejects). */
    if(d.contains("acc_epilogue") && py::isinstance<py::dict>(d["acc_epilogue"]))
        fill_conv_acc_epilogue(&s.acc_epilogue, d["acc_epilogue"].cast<py::dict>());
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
    /* The K-loop / epilogue knobs (absent keys keep the defaults). */
#define ROCKE_KNOB_INT(f) s.f = dict_int(d, #f, s.f);
#define ROCKE_KNOB_BOOL(f) s.f = dict_bool(d, #f, s.f);
    ROCKE_CONV_FWD_KNOB_FIELDS(ROCKE_KNOB_INT, ROCKE_KNOB_BOOL)
    s.cshuffle_no_alias = dict_bool(d, "cshuffle_no_alias", s.cshuffle_no_alias);
#undef ROCKE_KNOB_INT
#undef ROCKE_KNOB_BOOL
    for(const char* key : {"prefetch_regs", "xh_dpp", "a_direct", "xh_subtile", "pp_fetch_deep"})
        refuse_unported_conv_field(d, "conv_implicit_gemm", key);
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
    p->pad_w = dict_int(d, "pad_w", p->pad_w);
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

/* ===================== conv_direct_fwd_small{c,k}_wmma ===================== */

/* The nested problem dict and the optional "name" of a direct-conv spec. */
void direct_spec_fill_problem(rocke_direct_conv_problem_t* problem,
                              const char** name,
                              const py::dict& d,
                              std::deque<std::string>& store,
                              const char* who)
{
    /* The problem's shape fields are required, as in DirectConvProblem: a
     * missing one would silently read as the zero default. */
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error(std::string(who) + " requires a problem dictionary");
    py::dict pd = d["problem"].cast<py::dict>();
    const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
    for(const char* key : required)
    {
        if(!pd.contains(key))
            throw py::value_error(std::string(who) + ".problem requires " + key);
    }
    /* An explicit dtype=None is invalid (Python rejects it); only an omitted
     * key keeps the fp16 default. */
    if(pd.contains("dtype") && pd["dtype"].is_none())
        throw py::value_error(std::string(who)
                              + ".problem.dtype must be 'fp16' or 'bf16', not None");
    std::string v;
    fill_direct_conv_problem(problem, pd);
    if(dict_str(pd, "dtype", v))
    {
        store.push_back(v);
        problem->dtype = store.back().c_str();
    }
    if(dict_str(d, "name", v))
    {
        store.push_back(v);
        *name = store.back().c_str();
    }
}

/* direct_spec_fill_problem plus the optional "sched" (None: no strategy) of
 * the WMMA forward specs that carry one. */
template <typename Spec>
void direct_spec_fill_problem_sched(Spec* s,
                                    const py::dict& d,
                                    std::deque<std::string>& store,
                                    const char* who)
{
    direct_spec_fill_problem(&s->problem, &s->name, d, store, who);
    if(d.contains("sched") && !d["sched"].is_none())
    {
        store.push_back(d["sched"].cast<std::string>());
        s->sched = store.back().c_str();
    }
}

/* Wave32 WMMA forward of tiny channel counts (gfx1250). The dicts mirror
 * FwdSmallcWmmaSpec / FwdSmallkWmmaSpec with the problem nested; `sched` may
 * be None. */
rocke_fwd_smallc_wmma_spec_t fscw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_smallc_wmma_spec_t s = rocke_fwd_smallc_wmma_spec_default();
    direct_spec_fill_problem_sched(&s, d, store, "FwdSmallcWmmaSpec");
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_k = dict_int(d, "tiles_k", s.tiles_k);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.out_h = dict_int(d, "out_h", s.out_h);
    s.out_w = dict_int(d, "out_w", s.out_w);
    s.tiles_per_block = dict_int(d, "tiles_per_block", s.tiles_per_block);
    s.tdm_store = dict_bool(d, "tdm_store", s.tdm_store);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

rocke_fwd_smallk_wmma_spec_t fskw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_smallk_wmma_spec_t s = rocke_fwd_smallk_wmma_spec_default();
    direct_spec_fill_problem_sched(&s, d, store, "FwdSmallkWmmaSpec");
    s.waves_c = dict_int(d, "waves_c", s.waves_c);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_c = dict_int(d, "tiles_c", s.tiles_c);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.out_h = dict_int(d, "out_h", s.out_h);
    s.out_w = dict_int(d, "out_w", s.out_w);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_fwd_smallc_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallc_wmma_spec_t s = fscw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_smallc_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_smallc_wmma_lower_llvm");
}

std::string conv_direct_fwd_smallc_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_smallc_wmma_serialize_ir",
                                rocke_fwd_smallc_wmma_spec_t,
                                fscw_build_spec,
                                rocke_build_fwd_smallc_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_smallc_wmma_verify(const py::dict& d,
                                                            const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_smallc_wmma_verify",
                             rocke_fwd_smallc_wmma_spec_t,
                             fscw_build_spec,
                             rocke_build_fwd_smallc_wmma_new(&b, &s, arch.c_str()));
}

/* is_valid_fwd_smallc_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_smallc_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallc_wmma_spec_t s = fscw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_smallc_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_smallc_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_fwd_smallc_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_smallc_wmma_lds_bytes(&s));
}

std::string conv_direct_fwd_smallk_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallk_wmma_spec_t s = fskw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_smallk_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_smallk_wmma_lower_llvm");
}

std::string conv_direct_fwd_smallk_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_smallk_wmma_serialize_ir",
                                rocke_fwd_smallk_wmma_spec_t,
                                fskw_build_spec,
                                rocke_build_fwd_smallk_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_smallk_wmma_verify(const py::dict& d,
                                                            const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_smallk_wmma_verify",
                             rocke_fwd_smallk_wmma_spec_t,
                             fskw_build_spec,
                             rocke_build_fwd_smallk_wmma_new(&b, &s, arch.c_str()));
}

py::tuple conv_direct_fwd_smallk_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallk_wmma_spec_t s = fskw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_smallk_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_smallk_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_fwd_smallk_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_smallk_wmma_lds_bytes(&s));
}

/* ===================== conv_direct_fwd_small{c,k}_mfma ===================== */

/* Wave64 MFMA forward of tiny channel counts (gfx950). The dicts mirror
 * FwdSmallcMfmaSpec / FwdSmallkMfmaSpec with the problem nested. */
rocke_fwd_smallc_mfma_spec_t fscm_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_smallc_mfma_spec_t s = rocke_fwd_smallc_mfma_spec_default();
    direct_spec_fill_problem(&s.problem, &s.name, d, store, "FwdSmallcMfmaSpec");
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_k = dict_int(d, "tiles_k", s.tiles_k);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.out_h = dict_int(d, "out_h", s.out_h);
    s.out_w = dict_int(d, "out_w", s.out_w);
    s.tiles_per_block = dict_int(d, "tiles_per_block", s.tiles_per_block);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

rocke_fwd_smallk_mfma_spec_t fskm_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_smallk_mfma_spec_t s = rocke_fwd_smallk_mfma_spec_default();
    direct_spec_fill_problem(&s.problem, &s.name, d, store, "FwdSmallkMfmaSpec");
    s.waves = dict_int(d, "waves", s.waves);
    s.out_h = dict_int(d, "out_h", s.out_h);
    s.out_w = dict_int(d, "out_w", s.out_w);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_fwd_smallc_mfma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallc_mfma_spec_t s = fscm_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_smallc_mfma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_smallc_mfma_lower_llvm");
}

std::string conv_direct_fwd_smallc_mfma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_smallc_mfma_serialize_ir",
                                rocke_fwd_smallc_mfma_spec_t,
                                fscm_build_spec,
                                rocke_build_fwd_smallc_mfma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_smallc_mfma_verify(const py::dict& d,
                                                            const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_smallc_mfma_verify",
                             rocke_fwd_smallc_mfma_spec_t,
                             fscm_build_spec,
                             rocke_build_fwd_smallc_mfma_new(&b, &s, arch.c_str()));
}

/* is_valid_fwd_smallc_mfma_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_smallc_mfma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallc_mfma_spec_t s = fscm_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_smallc_mfma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_smallc_mfma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_fwd_smallc_mfma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_smallc_mfma_lds_bytes(&s));
}

std::string conv_direct_fwd_smallk_mfma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallk_mfma_spec_t s = fskm_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_smallk_mfma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_smallk_mfma_lower_llvm");
}

std::string conv_direct_fwd_smallk_mfma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_smallk_mfma_serialize_ir",
                                rocke_fwd_smallk_mfma_spec_t,
                                fskm_build_spec,
                                rocke_build_fwd_smallk_mfma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_smallk_mfma_verify(const py::dict& d,
                                                            const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_smallk_mfma_verify",
                             rocke_fwd_smallk_mfma_spec_t,
                             fskm_build_spec,
                             rocke_build_fwd_smallk_mfma_new(&b, &s, arch.c_str()));
}

py::tuple conv_direct_fwd_smallk_mfma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_smallk_mfma_spec_t s = fskm_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_smallk_mfma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_smallk_mfma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_fwd_smallk_mfma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_smallk_mfma_lds_bytes(&s));
}

/* ======================== conv_direct_fwd_pp_mfma ======================= */

/* LDS-DMA ping-pong MFMA direct forward (gfx950). The dict mirrors
 * FwdPpMfmaSpec with the problem nested. */
rocke_fwd_pp_mfma_spec_t fpp_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_fwd_pp_mfma_spec_t s = rocke_fwd_pp_mfma_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("FwdPpMfmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("FwdPpMfmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        /* dict_str reads None as absent. An explicit None dtype is rejected
         * (Python's validator refuses it) rather than read as the fp16
         * default; only an omitted dtype takes the default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error("FwdPpMfmaSpec.problem.dtype must be a string, not None");
        if(dict_str(pd, "dtype", v))
            s.problem.dtype = keep(v);
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_k = dict_int(d, "block_k", s.block_k);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_fwd_pp_mfma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_pp_mfma_spec_t s = fpp_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_pp_mfma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_pp_mfma_lower_llvm");
}

std::string conv_direct_fwd_pp_mfma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_pp_mfma_serialize_ir",
                                rocke_fwd_pp_mfma_spec_t,
                                fpp_build_spec,
                                rocke_build_fwd_pp_mfma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_pp_mfma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_pp_mfma_verify",
                             rocke_fwd_pp_mfma_spec_t,
                             fpp_build_spec,
                             rocke_build_fwd_pp_mfma_new(&b, &s, arch.c_str()));
}

/* is_valid_fwd_pp_mfma_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_pp_mfma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_pp_mfma_spec_t s = fpp_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_pp_mfma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_pp_mfma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_fwd_pp_mfma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_pp_mfma_lds_bytes(&s));
}

/* ======================= conv_direct_fwd_flex_mfma ====================== */

/* Flattened-pixel implicit-GEMM MFMA direct forward (gfx950). The dict
 * mirrors FwdFlexMfmaSpec with the problem nested. */
rocke_fwd_flex_mfma_spec_t ffx_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_fwd_flex_mfma_spec_t s = rocke_fwd_flex_mfma_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("FwdFlexMfmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("FwdFlexMfmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        /* dict_str reads None as absent. An explicit None dtype is rejected
         * (Python's validator refuses it) rather than read as the fp16
         * default; only an omitted dtype takes the default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error("FwdFlexMfmaSpec.problem.dtype must be a string, not None");
        if(dict_str(pd, "dtype", v))
            s.problem.dtype = keep(v);
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.bk = dict_int(d, "bk", s.bk);
    s.split_c = dict_int(d, "split_c", s.split_c);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_fwd_flex_mfma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_flex_mfma_spec_t s = ffx_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_flex_mfma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_flex_mfma_lower_llvm");
}

std::string conv_direct_fwd_flex_mfma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_flex_mfma_serialize_ir",
                                rocke_fwd_flex_mfma_spec_t,
                                ffx_build_spec,
                                rocke_build_fwd_flex_mfma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_flex_mfma_verify(const py::dict& d,
                                                          const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_flex_mfma_verify",
                             rocke_fwd_flex_mfma_spec_t,
                             ffx_build_spec,
                             rocke_build_fwd_flex_mfma_new(&b, &s, arch.c_str()));
}

/* fwd_flex_mfma_reason + kernel_name / grid / workspace_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), workspace_bytes). */
py::tuple conv_direct_fwd_flex_mfma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_flex_mfma_spec_t s = ffx_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    long long ws = 0;
    bool ok = rocke_fwd_flex_mfma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_flex_mfma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
    {
        rocke_fwd_flex_mfma_grid(&s, grid);
        ws = rocke_fwd_flex_mfma_workspace_bytes(&s);
    }
    return py::make_tuple(
        ok, std::string(reason), std::string(name), py::make_tuple(grid[0], grid[1], grid[2]), ws);
}

/* ======================== conv_direct_dw_fwd_wmma ======================== */

/* Toeplitz WMMA depthwise forward (gfx1250). The dict mirrors DwFwdWmmaSpec
 * with the problem nested. */
rocke_direct_dw_fwd_wmma_spec_t dfw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_direct_dw_fwd_wmma_spec_t s = rocke_direct_dw_fwd_wmma_spec_default();
    direct_spec_fill_problem(&s.problem, &s.name, d, store, "DwFwdWmmaSpec");
    s.block_waves = dict_int(d, "block_waves", s.block_waves);
    s.block_h = dict_int(d, "block_h", s.block_h);
    s.w_fold = dict_int(d, "w_fold", s.w_fold);
    s.prefetch_rows = dict_int(d, "prefetch_rows", s.prefetch_rows);
    return s;
}

std::string conv_direct_dw_fwd_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_dw_fwd_wmma_spec_t s = dfw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_direct_dw_fwd_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_dw_fwd_wmma_lower_llvm");
}

std::string conv_direct_dw_fwd_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_dw_fwd_wmma_serialize_ir",
                                rocke_direct_dw_fwd_wmma_spec_t,
                                dfw_build_spec,
                                rocke_build_direct_dw_fwd_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_dw_fwd_wmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_dw_fwd_wmma_verify",
                             rocke_direct_dw_fwd_wmma_spec_t,
                             dfw_build_spec,
                             rocke_build_direct_dw_fwd_wmma_new(&b, &s, arch.c_str()));
}

/* -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_dw_fwd_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_dw_fwd_wmma_spec_t s = dfw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_direct_dw_fwd_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_direct_dw_fwd_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_direct_dw_fwd_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_direct_dw_fwd_wmma_lds_bytes(&s));
}

/* ===================== conv_direct_fwd_gstream_wmma ====================== */

/* Small-group forward walking row bands (gfx1250). The dict mirrors
 * FwdGstreamWmmaSpec with the problem nested; `sched` may be None. */
rocke_fwd_gstream_wmma_spec_t fgs_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_gstream_wmma_spec_t s = rocke_fwd_gstream_wmma_spec_default();
    direct_spec_fill_problem_sched(&s, d, store, "FwdGstreamWmmaSpec");
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_k = dict_int(d, "tiles_k", s.tiles_k);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.tile_w = dict_int(d, "tile_w", s.tile_w);
    s.walk = dict_int(d, "walk", s.walk);
    s.stages = dict_int(d, "stages", s.stages);
    s.phase_split = dict_bool(d, "phase_split", s.phase_split);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.pf_depth = dict_int(d, "pf_depth", s.pf_depth);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_fwd_gstream_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_gstream_wmma_spec_t s = fgs_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_gstream_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(
        st, ll, err, "rocke_library_engine.conv_direct_fwd_gstream_wmma_lower_llvm");
}

std::string conv_direct_fwd_gstream_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_gstream_wmma_serialize_ir",
                                rocke_fwd_gstream_wmma_spec_t,
                                fgs_build_spec,
                                rocke_build_fwd_gstream_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_gstream_wmma_verify(const py::dict& d,
                                                             const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_gstream_wmma_verify",
                             rocke_fwd_gstream_wmma_spec_t,
                             fgs_build_spec,
                             rocke_build_fwd_gstream_wmma_new(&b, &s, arch.c_str()));
}

/* -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_gstream_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_gstream_wmma_spec_t s = fgs_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_gstream_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_gstream_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_fwd_gstream_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_gstream_wmma_lds_bytes(&s));
}

/* ===================== conv_direct_fwd_tapouter_wmma ===================== */

/* Tap-outer forward of large dilations (gfx1250). The dict mirrors
 * DirectFwdTapOuterSpec with the problem nested; `sched` may be None. */
rocke_fwd_tapouter_wmma_spec_t fto_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_fwd_tapouter_wmma_spec_t s = rocke_fwd_tapouter_wmma_spec_default();
    direct_spec_fill_problem_sched(&s, d, store, "DirectFwdTapOuterSpec");
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_k = dict_int(d, "tiles_k", s.tiles_k);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.tile_w = dict_int(d, "tile_w", s.tile_w);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.tdm_split = dict_int(d, "tdm_split", s.tdm_split);
    return s;
}

std::string conv_direct_fwd_tapouter_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_tapouter_wmma_spec_t s = fto_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_tapouter_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(
        st, ll, err, "rocke_library_engine.conv_direct_fwd_tapouter_wmma_lower_llvm");
}

std::string conv_direct_fwd_tapouter_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_tapouter_wmma_serialize_ir",
                                rocke_fwd_tapouter_wmma_spec_t,
                                fto_build_spec,
                                rocke_build_fwd_tapouter_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_tapouter_wmma_verify(const py::dict& d,
                                                              const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_tapouter_wmma_verify",
                             rocke_fwd_tapouter_wmma_spec_t,
                             fto_build_spec,
                             rocke_build_fwd_tapouter_wmma_new(&b, &s, arch.c_str()));
}

/* -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_tapouter_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_tapouter_wmma_spec_t s = fto_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_fwd_tapouter_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_fwd_tapouter_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_fwd_tapouter_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_fwd_tapouter_wmma_lds_bytes(&s));
}

/* ======================== conv_dgrad_patch_wmma ========================== */

/* Wave32 WMMA patch-embedding dgrad (gfx1250). The dict mirrors DgradPatchWmmaSpec
 * with the problem nested; `sched` may be None. */
rocke_dgrad_patch_wmma_spec_t dpw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_dgrad_patch_wmma_spec_t s = rocke_dgrad_patch_wmma_spec_default();
    /* The problem's shape fields are required, as in DirectConvProblem: a
     * missing one would silently read as the zero default. */
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("DgradPatchWmmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("DgradPatchWmmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        /* dict_str reads None as absent. An explicit None dtype is rejected
         * (Python's validator refuses it) rather than read as the fp16
         * default; only an omitted dtype takes the default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error("DgradPatchWmmaSpec.problem.dtype must be a string, not None");
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
    s.r_blk = dict_int(d, "r_blk", s.r_blk);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    if(d.contains("sched") && !d["sched"].is_none())
        s.sched = keep(d["sched"].cast<std::string>());
    return s;
}

std::string conv_dgrad_patch_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_patch_wmma_spec_t s = dpw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_dgrad_patch_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_dgrad_patch_wmma_lower_llvm");
}

std::string conv_dgrad_patch_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_dgrad_patch_wmma_serialize_ir",
                                rocke_dgrad_patch_wmma_spec_t,
                                dpw_build_spec,
                                rocke_build_dgrad_patch_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_dgrad_patch_wmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_dgrad_patch_wmma_verify",
                             rocke_dgrad_patch_wmma_spec_t,
                             dpw_build_spec,
                             rocke_build_dgrad_patch_wmma_new(&b, &s, arch.c_str()));
}

/* is_valid + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_dgrad_patch_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_patch_wmma_spec_t s = dpw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_dgrad_patch_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_dgrad_patch_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_dgrad_patch_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_dgrad_patch_wmma_lds_bytes(&s));
}

/* ======================== conv_dgrad_smallc_wmma ========================== */

/* Wave32 WMMA small-C stem dgrad (gfx1250). The dict mirrors DgradSmallcWmmaSpec
 * with the problem nested; `sched` may be None. */
rocke_dgrad_smallc_wmma_spec_t dsw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_dgrad_smallc_wmma_spec_t s = rocke_dgrad_smallc_wmma_spec_default();
    /* The problem's shape fields are required, as in DirectConvProblem: a
     * missing one would silently read as the zero default. */
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("DgradSmallcWmmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("DgradSmallcWmmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        /* dict_str reads None as absent. An explicit None dtype is rejected
         * (Python's validator refuses it) rather than read as the fp16
         * default; only an omitted dtype takes the default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error("DgradSmallcWmmaSpec.problem.dtype must be a string, not None");
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
    s.out_h = dict_int(d, "out_h", s.out_h);
    s.out_w = dict_int(d, "out_w", s.out_w);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    if(d.contains("sched") && !d["sched"].is_none())
        s.sched = keep(d["sched"].cast<std::string>());
    return s;
}

std::string conv_dgrad_smallc_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_smallc_wmma_spec_t s = dsw_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_dgrad_smallc_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_dgrad_smallc_wmma_lower_llvm");
}

std::string conv_dgrad_smallc_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_dgrad_smallc_wmma_serialize_ir",
                                rocke_dgrad_smallc_wmma_spec_t,
                                dsw_build_spec,
                                rocke_build_dgrad_smallc_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_dgrad_smallc_wmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_dgrad_smallc_wmma_verify",
                             rocke_dgrad_smallc_wmma_spec_t,
                             dsw_build_spec,
                             rocke_build_dgrad_smallc_wmma_new(&b, &s, arch.c_str()));
}

/* is_valid + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_dgrad_smallc_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_smallc_wmma_spec_t s = dsw_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_dgrad_smallc_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_dgrad_smallc_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_dgrad_smallc_wmma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_dgrad_smallc_wmma_lds_bytes(&s));
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
    s.pack_groups = dict_int(d, "pack_groups", s.pack_groups);
    s.split_barrier = dict_bool(d, "split_barrier", s.split_barrier);
    s.wave_epilogue = dict_bool(d, "wave_epilogue", s.wave_epilogue);
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

/* ======================== conv_direct_dgrad_pp_mfma ======================= */

/* LDS-DMA ping-pong MFMA direct dgrad (gfx950). The dict mirrors
 * DgradPpMfmaSpec with the (transposed, forward-pass) problem nested. */
rocke_dgrad_pp_mfma_spec_t ddpp_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_dgrad_pp_mfma_spec_t s = rocke_dgrad_pp_mfma_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("DgradPpMfmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("DgradPpMfmaSpec.problem requires ") + key);
        }
        std::string v;
        fill_direct_conv_problem(&s.problem, pd);
        /* dict_str reads None as absent. An explicit None dtype is rejected
         * (Python's validator refuses it) rather than read as the fp16
         * default; only an omitted dtype takes the default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error("DgradPpMfmaSpec.problem.dtype must be a string, not None");
        if(dict_str(pd, "dtype", v))
            s.problem.dtype = keep(v);
    }
    {
        std::string v;
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_k = dict_int(d, "block_k", s.block_k);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

std::string conv_direct_dgrad_pp_mfma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_pp_mfma_spec_t s = ddpp_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_dgrad_pp_mfma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_dgrad_pp_mfma_lower_llvm");
}

std::string conv_direct_dgrad_pp_mfma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_dgrad_pp_mfma_serialize_ir",
                                rocke_dgrad_pp_mfma_spec_t,
                                ddpp_build_spec,
                                rocke_build_dgrad_pp_mfma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_dgrad_pp_mfma_verify(const py::dict& d,
                                                          const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_dgrad_pp_mfma_verify",
                             rocke_dgrad_pp_mfma_spec_t,
                             ddpp_build_spec,
                             rocke_build_dgrad_pp_mfma_new(&b, &s, arch.c_str()));
}

/* is_valid_dgrad_pp_mfma_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_dgrad_pp_mfma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_pp_mfma_spec_t s = ddpp_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_dgrad_pp_mfma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_dgrad_pp_mfma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok)
        rocke_dgrad_pp_mfma_grid(&s, grid);
    return py::make_tuple(ok,
                          std::string(reason),
                          std::string(name),
                          py::make_tuple(grid[0], grid[1], grid[2]),
                          rocke_dgrad_pp_mfma_lds_bytes(&s));
}

/* ======================== conv_direct_fwd_wmma ============================ */

/* Wave32 WMMA direct forward (gfx1250). The dict mirrors DirectFwdWmmaSpec
 * with the problem nested; `sched` may be None. */
rocke_direct_fwd_wmma_spec_t dfww_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_direct_fwd_wmma_spec_t s = rocke_direct_fwd_wmma_spec_default();
    /* The problem's shape fields are required, as in DirectConvProblem: a
     * missing one would silently read as the zero default. */
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("DirectFwdWmmaSpec requires a problem dictionary");
    {
        py::dict pd = d["problem"].cast<py::dict>();
        const char* required[] = {"N", "H", "W", "groups", "cpg", "kpg"};
        for(const char* key : required)
        {
            if(!pd.contains(key))
                throw py::value_error(std::string("DirectFwdWmmaSpec.problem requires ") + key);
        }
        /* An explicit dtype=None is invalid (Python rejects it); only an
         * omitted key keeps the fp16 default. */
        if(pd.contains("dtype") && pd["dtype"].is_none())
            throw py::value_error(
                "DirectFwdWmmaSpec.problem.dtype must be 'fp16' or 'bf16', not None");
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
    s.waves_k = dict_int(d, "waves_k", s.waves_k);
    s.waves_p = dict_int(d, "waves_p", s.waves_p);
    s.tiles_k = dict_int(d, "tiles_k", s.tiles_k);
    s.tiles_p = dict_int(d, "tiles_p", s.tiles_p);
    s.tile_w = dict_int(d, "tile_w", s.tile_w);
    s.tile_n = dict_int(d, "tile_n", s.tile_n);
    s.ck = dict_int(d, "ck", s.ck);
    s.stages = dict_int(d, "stages", s.stages);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    s.pack_groups = dict_int(d, "pack_groups", s.pack_groups);
    s.split_barrier = dict_bool(d, "split_barrier", s.split_barrier);
    s.wave_epilogue = dict_bool(d, "wave_epilogue", s.wave_epilogue);
    s.phase_split = dict_bool(d, "phase_split", s.phase_split);
    s.row_ring = dict_bool(d, "row_ring", s.row_ring);
    s.tap_ring = dict_bool(d, "tap_ring", s.tap_ring);
    s.pingpong = dict_bool(d, "pingpong", s.pingpong);
    s.pp_group = dict_int(d, "pp_group", s.pp_group);
    s.tap_reuse = dict_bool(d, "tap_reuse", s.tap_reuse);
    s.tap_shift = dict_bool(d, "tap_shift", s.tap_shift);
    s.strided_view = dict_bool(d, "strided_view", s.strided_view);
    s.fold_n = dict_bool(d, "fold_n", s.fold_n);
    s.poly_view = dict_bool(d, "poly_view", s.poly_view);
    if(d.contains("sched") && !d["sched"].is_none())
        s.sched = keep(d["sched"].cast<std::string>());
    return s;
}

std::string conv_direct_fwd_wmma_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_fwd_wmma_spec_t s = dfww_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_direct_fwd_wmma_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_direct_fwd_wmma_lower_llvm");
}

std::string conv_direct_fwd_wmma_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_direct_fwd_wmma_serialize_ir",
                                rocke_direct_fwd_wmma_spec_t,
                                dfww_build_spec,
                                rocke_build_direct_fwd_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_direct_fwd_wmma_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_direct_fwd_wmma_verify",
                             rocke_direct_fwd_wmma_spec_t,
                             dfww_build_spec,
                             rocke_build_direct_fwd_wmma_new(&b, &s, arch.c_str()));
}

/* is_valid_fwd_wmma_spec + kernel_name / grid / lds_bytes in one call:
 * -> (ok, reason, kernel_name, (gx, gy, gz), lds_bytes). */
py::tuple conv_direct_fwd_wmma_describe(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_direct_fwd_wmma_spec_t s = dfww_build_spec(d, store);
    char reason[512];
    char name[512];
    int grid[3] = {0, 0, 0};
    bool ok = rocke_direct_fwd_wmma_is_valid_spec(&s, arch.c_str(), reason, sizeof reason);
    if(rocke_direct_fwd_wmma_kernel_name(&s, name, sizeof name) != ROCKE_OK)
        name[0] = '\0';
    if(ok) /* a rejected spec may have zero tile extents */
        rocke_direct_fwd_wmma_grid(&s, grid);
    /* The phase-split halo divides by the stride; a zero stride (rejected
     * above) reports no LDS rather than dividing by zero. */
    long lds = (s.phase_split && s.problem.stride == 0) ? 0 : rocke_direct_fwd_wmma_lds_bytes(&s);
    return py::make_tuple(
        ok, std::string(reason), std::string(name), py::make_tuple(grid[0], grid[1], grid[2]), lds);
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
    s.register_prefetch = dict_bool(d, "register_prefetch", s.register_prefetch);
    s.prefetch_global_regs = dict_bool(d, "prefetch_global_regs", s.prefetch_global_regs);
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
     * lds_k_pad sets the M-outer LDS row stride (and the "pad<N>" name tag).
     * vector_size_a/b cap the sync free-axis load widths (and are rejected on
     * the async K loop); dropped, the engine would build the uncapped kernel
     * and skip those checks. */
    if(d.contains("lds_k_pad") && !d["lds_k_pad"].is_none())
    {
        s.has_lds_k_pad = true;
        s.lds_k_pad = d["lds_k_pad"].cast<int>();
    }
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
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
        /* ConvDataSpec.mma_dtype: absent or None keeps the NULL default. */
        if(dict_str(d, "mma_dtype", v))
            s.mma_dtype = keep(v);
        if(dict_str(d, "pipeline", v))
            s.pipeline = keep(v);
        if(dict_str(d, "epilogue", v))
            s.epilogue = keep(v);
        if(dict_str(d, "dtype_acc", v))
            s.dtype_acc = keep(v);
    }
#define ROCKE_KNOB_INT(f) s.f = dict_int(d, #f, s.f);
#define ROCKE_KNOB_BOOL(f) s.f = dict_bool(d, #f, s.f);
    ROCKE_CONV_WGRAD_KNOB_FIELDS(ROCKE_KNOB_INT, ROCKE_KNOB_BOOL)
#undef ROCKE_KNOB_INT
#undef ROCKE_KNOB_BOOL
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
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_lowered(st, ll, err, "rocke_library_engine.conv_wgrad_lower_llvm");
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

/* =================== conv_implicit_gemm_dgrad =================== */

rocke_dgrad_conv_spec_t conv_dgrad_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_dgrad_conv_spec_t s = rocke_dgrad_conv_spec_default();
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
        /* ConvDataSpec.mma_dtype: absent or None keeps the NULL default. */
        if(dict_str(d, "mma_dtype", v))
            s.mma_dtype = keep(v);
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
    s.lds_k_outer = dict_bool(d, "lds_k_outer", s.lds_k_outer);
    s.chiplet_swizzle = dict_bool(d, "chiplet_swizzle", s.chiplet_swizzle);
    s.chiplet_wgm = dict_int(d, "chiplet_wgm", s.chiplet_wgm);
    s.chiplet_num_xcds = dict_int(d, "chiplet_num_xcds", s.chiplet_num_xcds);
    s.chiplet_chunk_size = dict_int(d, "chiplet_chunk_size", s.chiplet_chunk_size);
    s.split_k = dict_int(d, "split_k", s.split_k);
    s.num_load_waves = dict_int(d, "num_load_waves", s.num_load_waves);
    s.max_sub_gemms = dict_int(d, "max_sub_gemms", s.max_sub_gemms);
    s.uniform_k_decode = dict_bool(d, "uniform_k_decode", s.uniform_k_decode);
    s.patch = dict_bool(d, "patch", s.patch);
    s.tap_outer_k = dict_bool(d, "tap_outer_k", s.tap_outer_k);
    s.xcd_tile_order = dict_bool(d, "xcd_tile_order", s.xcd_tile_order);
    s.dy_halo = dict_int(d, "dy_halo", s.dy_halo);
    s.dy_halo_setprio = dict_int(d, "dy_halo_setprio", s.dy_halo_setprio);
    s.dy_halo_kouter_pad = dict_int(d, "dy_halo_kouter_pad", s.dy_halo_kouter_pad);
    s.halo_max_wo = dict_int(d, "halo_max_wo", s.halo_max_wo);
    if(d.contains("lds_k_pad") && !d["lds_k_pad"].is_none())
    {
        s.has_lds_k_pad = true;
        s.lds_k_pad = d["lds_k_pad"].cast<int>();
    }
    if(d.contains("waves_per_eu") && !d["waves_per_eu"].is_none())
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = d["waves_per_eu"].cast<int>();
    }
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
#define ROCKE_KNOB_INT(f) s.f = dict_int(d, #f, s.f);
#define ROCKE_KNOB_BOOL(f) s.f = dict_bool(d, #f, s.f);
    ROCKE_CONV_DGRAD_KNOB_FIELDS(ROCKE_KNOB_INT, ROCKE_KNOB_BOOL)
#undef ROCKE_KNOB_INT
#undef ROCKE_KNOB_BOOL
    refuse_unported_conv_field(d, "conv_dgrad", "cshuffle_no_alias");
    return s;
}

std::string conv_dgrad_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_conv_spec_t s = conv_dgrad_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_implicit_gemm_conv_dgrad_new(&b, &s, arch.c_str());
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg = std::string("rocke_library_engine.conv_dgrad_lower_llvm build failed: ")
                          + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_lowered(st, ll, err, "rocke_library_engine.conv_dgrad_lower_llvm");
}

std::string conv_dgrad_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_dgrad_serialize_ir",
                                rocke_dgrad_conv_spec_t,
                                conv_dgrad_build_spec,
                                rocke_build_implicit_gemm_conv_dgrad_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_dgrad_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_dgrad_verify",
                             rocke_dgrad_conv_spec_t,
                             conv_dgrad_build_spec,
                             rocke_build_implicit_gemm_conv_dgrad_new(&b, &s, arch.c_str()));
}

/* ============ dgrad interior-only dX (dx_interior) + shell zero-fill ============ */

py::dict conv_problem_to_py(const rocke_conv_problem_t& p)
{
    py::dict d;
    d["N"] = p.N;
    d["Hi"] = p.Hi;
    d["Wi"] = p.Wi;
    d["C"] = p.C;
    d["K"] = p.K;
    d["Y"] = p.Y;
    d["X"] = p.X;
    d["sH"] = p.sH;
    d["sW"] = p.sW;
    d["pH"] = p.pH;
    d["pW"] = p.pW;
    d["dH"] = p.dH;
    d["dW"] = p.dW;
    d["groups"] = p.groups;
    if(p.is_3d)
    {
        d["Di"] = p.Di;
        d["Z"] = p.Z;
        d["sD"] = p.sD;
        d["pD"] = p.pD;
        d["dD"] = p.dD;
    }
    return d;
}

rocke_conv_problem_t conv_problem_from_py(const py::dict& d)
{
    rocke_conv_problem_t p = rocke_conv_problem_default(1, 1, 1, 1, 1, 1, 1);
    fill_conv_problem(&p, d);
    return p;
}

/* The dX kernarg dict of a view launch (Python interior_view_dx / wmask_view):
 * p_dX_stride_di only for a 3-D problem. */
py::dict dx_view_to_py(const rocke_dgrad_dx_view_t& v, bool is_3d)
{
    py::dict s;
    s["p_dX_stride_n"] = v.stride_n;
    s["p_dX_stride_hi"] = v.stride_hi;
    s["p_dX_stride_wi"] = v.stride_wi;
    if(is_3d)
        s["p_dX_stride_di"] = v.stride_di;
    return s;
}

rocke_dgrad_halo_zero_spec_t halo_zero_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_dgrad_halo_zero_spec_t s = rocke_dgrad_halo_zero_spec_default();
    s.block = dict_int(d, "block", s.block);
    std::string v;
    if(dict_str(d, "name", v))
    {
        store.push_back(v);
        s.name = store.back().c_str();
    }
    return s;
}

std::string conv_dgrad_halo_zero_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_dgrad_halo_zero_spec_t s = halo_zero_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_dgrad_halo_zero_new(&b, &s);
    if(!k || !rocke_ir_builder_ok(&b))
    {
        std::string msg
            = std::string("rocke_library_engine.conv_dgrad_halo_zero_lower_llvm build failed: ")
              + rocke_ir_builder_error(&b);
        rocke_ir_builder_free(&b);
        throw std::runtime_error(msg);
    }
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_lowered(st, ll, err, "rocke_library_engine.conv_dgrad_halo_zero_lower_llvm");
}

std::string conv_dgrad_halo_zero_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_dgrad_halo_zero_serialize_ir",
                                rocke_dgrad_halo_zero_spec_t,
                                halo_zero_build_spec,
                                rocke_build_dgrad_halo_zero_new(&b, &s));
}

std::vector<std::string> conv_dgrad_halo_zero_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_dgrad_halo_zero_verify",
                             rocke_dgrad_halo_zero_spec_t,
                             halo_zero_build_spec,
                             rocke_build_dgrad_halo_zero_new(&b, &s));
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
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    guard.reset();
    return take_lowered(st, ll, err, "rocke_library_engine.conv_wgrad_reduce_lower_llvm");
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

/* ================= conv_fwd_split_k_reduce (fwd split-K Stage 2) =============== */

rocke_fwd_split_k_reduce_spec_t conv_fwd_split_k_reduce_build_spec(const py::dict& d,
                                                                   std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_fwd_split_k_reduce_spec_t s = rocke_fwd_split_k_reduce_spec_default();
    /* An explicit dtype_d=None is a malformed spec (Python validate() rejects
     * it), not a request for the bf16 default: pass NULL so the builder
     * rejects it. Only an absent key keeps the default. */
    if(d.contains("dtype_d") && d["dtype_d"].is_none())
        s.dtype_d = NULL;
    s.vec = dict_int(d, "vec", s.vec);
    s.block_size = dict_int(d, "block_size", s.block_size);
    /* The unrolled fold bakes the slab count: it must match the launch's
     * ks_count, so it is parsed rather than left at the runtime-loop default. */
    s.degree = dict_int(d, "degree", s.degree);
    {
        std::string v;
        if(dict_str(d, "dtype_d", v))
            s.dtype_d = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    return s;
}

std::string conv_fwd_split_k_reduce_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_fwd_split_k_reduce_spec_t s = conv_fwd_split_k_reduce_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_fwd_split_k_reduce_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_library_engine.conv_fwd_split_k_reduce_lower_llvm");
}

std::string conv_fwd_split_k_reduce_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.conv_fwd_split_k_reduce_serialize_ir",
                                rocke_fwd_split_k_reduce_spec_t,
                                conv_fwd_split_k_reduce_build_spec,
                                rocke_build_conv_fwd_split_k_reduce_new(&b, &s, arch.c_str()));
}

std::vector<std::string> conv_fwd_split_k_reduce_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.conv_fwd_split_k_reduce_verify",
                             rocke_fwd_split_k_reduce_spec_t,
                             conv_fwd_split_k_reduce_build_spec,
                             rocke_build_conv_fwd_split_k_reduce_new(&b, &s, arch.c_str()));
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
    reg3("conv_dgrad_patch_wmma",
         &conv_dgrad_patch_wmma_lower_llvm,
         &conv_dgrad_patch_wmma_serialize_ir,
         &conv_dgrad_patch_wmma_verify);
    m.def("conv_dgrad_patch_wmma_describe",
          &conv_dgrad_patch_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_dgrad_smallc_wmma",
         &conv_dgrad_smallc_wmma_lower_llvm,
         &conv_dgrad_smallc_wmma_serialize_ir,
         &conv_dgrad_smallc_wmma_verify);
    m.def("conv_dgrad_smallc_wmma_describe",
          &conv_dgrad_smallc_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_smallc_wmma",
         &conv_direct_fwd_smallc_wmma_lower_llvm,
         &conv_direct_fwd_smallc_wmma_serialize_ir,
         &conv_direct_fwd_smallc_wmma_verify);
    m.def("conv_direct_fwd_smallc_wmma_describe",
          &conv_direct_fwd_smallc_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_smallk_wmma",
         &conv_direct_fwd_smallk_wmma_lower_llvm,
         &conv_direct_fwd_smallk_wmma_serialize_ir,
         &conv_direct_fwd_smallk_wmma_verify);
    m.def("conv_direct_fwd_smallk_wmma_describe",
          &conv_direct_fwd_smallk_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_smallc_mfma",
         &conv_direct_fwd_smallc_mfma_lower_llvm,
         &conv_direct_fwd_smallc_mfma_serialize_ir,
         &conv_direct_fwd_smallc_mfma_verify);
    m.def("conv_direct_fwd_smallc_mfma_describe",
          &conv_direct_fwd_smallc_mfma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_smallk_mfma",
         &conv_direct_fwd_smallk_mfma_lower_llvm,
         &conv_direct_fwd_smallk_mfma_serialize_ir,
         &conv_direct_fwd_smallk_mfma_verify);
    m.def("conv_direct_fwd_smallk_mfma_describe",
          &conv_direct_fwd_smallk_mfma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_pp_mfma",
         &conv_direct_fwd_pp_mfma_lower_llvm,
         &conv_direct_fwd_pp_mfma_serialize_ir,
         &conv_direct_fwd_pp_mfma_verify);
    m.def("conv_direct_fwd_pp_mfma_describe",
          &conv_direct_fwd_pp_mfma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_flex_mfma",
         &conv_direct_fwd_flex_mfma_lower_llvm,
         &conv_direct_fwd_flex_mfma_serialize_ir,
         &conv_direct_fwd_flex_mfma_verify);
    m.def("conv_direct_fwd_flex_mfma_describe",
          &conv_direct_fwd_flex_mfma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_dw_fwd_wmma",
         &conv_direct_dw_fwd_wmma_lower_llvm,
         &conv_direct_dw_fwd_wmma_serialize_ir,
         &conv_direct_dw_fwd_wmma_verify);
    m.def("conv_direct_dw_fwd_wmma_describe",
          &conv_direct_dw_fwd_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_gstream_wmma",
         &conv_direct_fwd_gstream_wmma_lower_llvm,
         &conv_direct_fwd_gstream_wmma_serialize_ir,
         &conv_direct_fwd_gstream_wmma_verify);
    m.def("conv_direct_fwd_gstream_wmma_describe",
          &conv_direct_fwd_gstream_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_tapouter_wmma",
         &conv_direct_fwd_tapouter_wmma_lower_llvm,
         &conv_direct_fwd_tapouter_wmma_serialize_ir,
         &conv_direct_fwd_tapouter_wmma_verify);
    m.def("conv_direct_fwd_tapouter_wmma_describe",
          &conv_direct_fwd_tapouter_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_dgrad_wmma",
         &conv_direct_dgrad_wmma_lower_llvm,
         &conv_direct_dgrad_wmma_serialize_ir,
         &conv_direct_dgrad_wmma_verify);
    m.def("conv_direct_dgrad_wmma_describe",
          &conv_direct_dgrad_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_dgrad_pp_mfma",
         &conv_direct_dgrad_pp_mfma_lower_llvm,
         &conv_direct_dgrad_pp_mfma_serialize_ir,
         &conv_direct_dgrad_pp_mfma_verify);
    m.def("conv_direct_dgrad_pp_mfma_describe",
          &conv_direct_dgrad_pp_mfma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("conv_direct_fwd_wmma",
         &conv_direct_fwd_wmma_lower_llvm,
         &conv_direct_fwd_wmma_serialize_ir,
         &conv_direct_fwd_wmma_verify);
    m.def("conv_direct_fwd_wmma_describe",
          &conv_direct_fwd_wmma_describe,
          py::arg("spec"),
          py::arg("arch"));
    reg3("deep_fused_conv_pool",
         &deep_fused_conv_pool_lower_llvm,
         &deep_fused_conv_pool_serialize_ir,
         &deep_fused_conv_pool_verify);

    /* ---- wgrad two-stage family ---- */
    reg3("conv_wgrad", conv_wgrad_lower_llvm, conv_wgrad_serialize_ir, conv_wgrad_verify);
    m.def(
        "conv_implicit_gemm_is_valid_spec",
        [](const py::dict& d, const std::string& arch) -> py::tuple {
            const char* a = require_arch(arch);
            std::deque<std::string> store;
            rocke_implicit_gemm_conv_spec_t s = conv_igemm_build_spec(d, store);
            char reason[1024] = {0};
            const bool ok = rocke_implicit_gemm_conv_is_valid_spec(&s, a, reason, sizeof(reason));
            return py::make_tuple(ok, ok ? std::string("ok") : std::string(reason));
        },
        py::arg("spec"),
        py::arg("arch"),
        "C port of the fwd is_valid_spec: (ok, reason), reason 'ok' when accepted.");
    /* ---- dgrad interior-only dX (launch-time dx_interior) ---- */
    reg3("conv_dgrad_halo_zero",
         conv_dgrad_halo_zero_lower_llvm,
         conv_dgrad_halo_zero_serialize_ir,
         conv_dgrad_halo_zero_verify);
    m.def(
        "conv_dgrad_interior_view",
        [](const py::dict& problem, bool keep_w, int axes) -> py::dict {
            rocke_conv_problem_t p = conv_problem_from_py(problem);
            rocke_conv_problem_t v = rocke_dgrad_interior_view_problem(&p, keep_w, axes);
            rocke_dgrad_dx_view_t dx;
            rocke_dgrad_interior_view_dx(&p, keep_w, axes, &dx);
            py::dict out;
            out["problem"] = conv_problem_to_py(v);
            out["dx_offset"] = dx.offset;
            out["dx_strides"] = dx_view_to_py(dx, p.is_3d);
            return out;
        },
        py::arg("problem"),
        py::arg("keep_w"),
        py::arg("axes"),
        "C port of interior_view_problem + interior_view_dx: {problem, dx_offset, dx_strides}.");
    m.def(
        "conv_dgrad_wmask_view",
        [](const py::dict& problem) -> py::dict {
            rocke_conv_problem_t p = conv_problem_from_py(problem);
            rocke_conv_problem_t v;
            rocke_dgrad_dx_view_t dx;
            rocke_dgrad_wmask_view(&p, &v, &dx);
            py::dict out;
            out["problem"] = conv_problem_to_py(v);
            out["dx_offset"] = dx.offset;
            out["dx_strides"] = dx_view_to_py(dx, p.is_3d);
            return out;
        },
        py::arg("problem"),
        "C port of wmask_view: {problem, dx_offset, dx_strides}.");
    m.def(
        "conv_dgrad_dx_interior_problem_ok",
        [](const py::dict& problem, int mask) -> py::tuple {
            rocke_conv_problem_t p = conv_problem_from_py(problem);
            char reason[256] = {0};
            const bool ok = rocke_dgrad_dx_interior_problem_ok(&p, mask, reason, sizeof(reason));
            return py::make_tuple(ok, std::string(reason));
        },
        py::arg("problem"),
        py::arg("mask"),
        "C port of dx_interior_problem_ok: (ok, reason).");
    m.def(
        "conv_dgrad_dx_interior_launch_ok",
        [](const py::dict& spec, int dx_interior, int axes, const py::dict& problem) -> py::tuple {
            std::deque<std::string> store;
            rocke_dgrad_conv_spec_t s = conv_dgrad_build_spec(spec, store);
            rocke_conv_problem_t p = conv_problem_from_py(problem);
            char reason[256] = {0};
            const bool ok = rocke_dgrad_dx_interior_launch_ok(
                &s, dx_interior, axes, &p, reason, sizeof(reason));
            return py::make_tuple(ok, std::string(reason));
        },
        py::arg("spec"),
        py::arg("dx_interior"),
        py::arg("axes"),
        py::arg("problem"),
        "C port of dx_interior_launch_conflict: (ok, reason), reason 'ok' when accepted.");
    m.def(
        "conv_dgrad_halo_zero_launch",
        [](const py::dict& problem, int elem_bytes, int axes, const py::dict& spec) -> py::tuple {
            rocke_conv_problem_t p = conv_problem_from_py(problem);
            std::deque<std::string> store;
            rocke_dgrad_halo_zero_spec_t s = halo_zero_build_spec(spec, store);
            int64_t vals[ROCKE_DGRAD_HALO_ZERO_SCALARS];
            int grid[3], block[3];
            if(rocke_dgrad_halo_zero_launch(&p, elem_bytes, axes, &s, vals, grid, block)
               != ROCKE_OK)
                throw py::value_error("dx_interior zero-fill needs C * elem_bytes % 16 == 0");
            static const char* const names[ROCKE_DGRAD_HALO_ZERO_SCALARS] = {"Di",
                                                                             "Do",
                                                                             "Hi",
                                                                             "Ho",
                                                                             "ru",
                                                                             "hd",
                                                                             "hh",
                                                                             "e",
                                                                             "L1",
                                                                             "L2",
                                                                             "L3",
                                                                             "n1",
                                                                             "n2",
                                                                             "u1",
                                                                             "u2",
                                                                             "u3"};
            py::dict v;
            for(int i = 0; i < ROCKE_DGRAD_HALO_ZERO_SCALARS; ++i)
                v[names[i]] = vals[i];
            return py::make_tuple(v,
                                  py::make_tuple(grid[0], grid[1], grid[2]),
                                  py::make_tuple(block[0], block[1], block[2]));
        },
        py::arg("problem"),
        py::arg("elem_bytes"),
        py::arg("axes"),
        py::arg("spec"),
        "C port of halo_zero_launch without the pointer kernargs: (values, grid, block).");
    /* ---- implicit-GEMM dgrad ---- */
    reg3("conv_dgrad", conv_dgrad_lower_llvm, conv_dgrad_serialize_ir, conv_dgrad_verify);
    m.def(
        "conv_dgrad_is_valid_spec",
        [](const py::dict& d, const std::string& arch) -> py::tuple {
            const char* a = require_arch(arch);
            std::deque<std::string> store;
            rocke_dgrad_conv_spec_t s = conv_dgrad_build_spec(d, store);
            char reason[1024] = {0};
            const bool ok = rocke_dgrad_conv_is_valid_spec(&s, a, reason, sizeof(reason));
            return py::make_tuple(ok, ok ? std::string("ok") : std::string(reason));
        },
        py::arg("spec"),
        py::arg("arch"),
        "C port of is_valid_dgrad_spec: (ok, reason), reason 'ok' when accepted.");
    /* conv_wgrad_reduce is used by the byte-identity gate
     * (platform/tools/check_byte_identity.py) and by external callers driving
     * Stage 2 via the Python engine API directly.  The benchmark drives Stage 2
     * via the Python instance layer (rocke.instances.common.conv_wgrad_workspace_reduce),
     * not through this binding, so the registration may appear unused in that context. */
    reg3("conv_wgrad_reduce",
         conv_wgrad_reduce_lower_llvm,
         conv_wgrad_reduce_serialize_ir,
         conv_wgrad_reduce_verify);
    /* ---- fwd split-K Stage 2 (folds the [ks_count][M][K] fp32 slabs) ---- */
    reg3("conv_fwd_split_k_reduce",
         conv_fwd_split_k_reduce_lower_llvm,
         conv_fwd_split_k_reduce_serialize_ir,
         conv_fwd_split_k_reduce_verify);
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
