// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_fused_moe_e2e.cpp -- binding for the fused-MoE end-to-end
 * forward orchestrator (fused_moe_e2e). Kept in its own translation unit because
 * the orchestrator's internal header (instance_fused_moe_e2e_internal.h) pulls
 * in the moe_gemm_fused / tensor_view headers, whose rocke_tensor_descriptor tag
 * clashes with the transforms.h definition pulled in by the FMHA common header
 * in rocke_engine_fmha_extra.cpp -- the same isolation reason as
 * rocke_engine_moe_gemm_fused.cpp.
 *
 * fused_moe_e2e is NOT a single bound lowering unit. It is a host-launch
 * orchestrator: rocke_build_fused_moe_forward returns a kernel whose forward_fn
 * drives a multi-kernel pipeline, and the instance header is explicit that it
 * "emits NO single monolithic kernel". So it cannot be a single KernelDef ->
 * one .ll the way the four FMHA-fwd variants are.
 *
 * Its parity emitter (tests/parity/fused_moe_e2e_emit.c / .py) does not lower
 * the orchestrator; it lowers three representative *device* sub-kernels --
 * ROUTER (topk_softmax), GATE_UP_GEMM and DOWN_GEMM (both the batched-GEMM
 * builder shape) -- via rocke_fused_moe_forward_lower_to_llvm and prints them
 * concatenated, each behind a "; === fused_moe_e2e stage: <STAGE> ===" banner.
 * (The sort / gather / silu_mul / topk_reduce stages return NOTIMPL and are
 * excluded.) This binding reproduces that concatenated multi-stage output
 * byte-for-byte: the _lower_llvm entry is a *concatenated multi-stage* lower,
 * not a single-kernel lower; there is no genuine single-kernel serialize/verify
 * for the orchestrator, so _serialize_ir / _verify replay the same three stage
 * kernels in order, matching the emitter's ir / verify modes.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "rocke/instance_batched_gemm.h"
#include "rocke/instance_fused_moe_e2e.h"
#include "rocke/instance_fused_moe_e2e_internal.h"
#include "rocke/instance_topk_softmax.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

int x_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
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

/* Lifetime helper for const char* spec fields (deque, not vector: a realloc
 * would dangle kept c_str() pointers). */
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

/* Materialise the FusedMoeForwardSpec from the dict: every non-shape field stays
 * at the dataclass default (rocke_fmoe_forward_spec_default), so the tile-swap
 * policy and the static gate see exactly the Python defaults; only the
 * enumerated shape + dtype differ -- exactly as the standalone emitter. */
rocke_fmoe_forward_spec_t fmoe_build(const py::dict& d, Store& st)
{
    rocke_fmoe_forward_spec_t s = rocke_fmoe_forward_spec_default();
    s.arch = st.keep("gfx950");
    s.tokens = x_int(d, "tokens", s.tokens);
    s.experts = x_int(d, "experts", s.experts);
    s.topk = x_int(d, "topk", s.topk);
    s.hidden = x_int(d, "hidden", s.hidden);
    s.intermediate = x_int(d, "intermediate", s.intermediate);
    std::string v;
    if(x_str(d, "dtype", v))
        s.dtype = st.keep(v);
    return s;
}

struct FmoeStage
{
    const char* banner;
    rocke_fmoe_stage_t stage;
};
const FmoeStage FMOE_STAGES[] = {
    {"ROUTER", ROCKE_FMOE_STAGE_ROUTER},
    {"GATE_UP_GEMM", ROCKE_FMOE_STAGE_GATE_UP_GEMM},
    {"DOWN_GEMM", ROCKE_FMOE_STAGE_DOWN_GEMM},
};
constexpr size_t FMOE_NSTAGES = sizeof(FMOE_STAGES) / sizeof(FMOE_STAGES[0]);

std::string fmoe_lower(const py::dict& d, const std::string& arch)
{
    Store st;
    const char* a = arch.empty() ? "gfx950" : arch.c_str();
    rocke_fmoe_forward_spec_t s = fmoe_build(d, st);
    std::string out;
    for(size_t i = 0; i < FMOE_NSTAGES; ++i)
    {
        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t s2 = rocke_fused_moe_forward_lower_to_llvm(
            &s, a, FMOE_STAGES[i].stage, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
        std::string text = take_ll(s2, ll, err, "rocke_engine.fused_moe_e2e_lower_llvm");
        out += "; === fused_moe_e2e stage: ";
        out += FMOE_STAGES[i].banner;
        out += " ===\n";
        out += text;
        if(text.empty() || text.back() != '\n')
            out += '\n';
    }
    return out;
}

/* Build one stage's device sub-kernel into the supplied builder, applying the
 * same tile-swap policy the lower path runs (rocke_fmoe_build_ctx_init). Returns
 * NULL on failure; the caller owns *b. Mirrors the standalone emitter's
 * build_stage_kernel(). */
rocke_kernel_def_t* fmoe_build_stage_kernel(rocke_ir_builder_t* b,
                                            const rocke_fmoe_forward_spec_t* spec,
                                            rocke_fmoe_stage_t stage)
{
    rocke_fmoe_build_ctx_t* ctx = (rocke_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(!ctx)
        return nullptr;
    if(rocke_fmoe_build_ctx_init(ctx, spec, spec->arch ? spec->arch : "gfx950") != ROCKE_OK)
    {
        rocke_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return nullptr;
    }
    rocke_fmoe_forward_spec_t adj = ctx->spec;
    const char* arch = ctx->arch;
    rocke_kernel_def_t* k = nullptr;
    if(stage == ROCKE_FMOE_STAGE_ROUTER)
    {
        rocke_topk_softmax_spec_t ts = rocke_fmoe_forward_spec_to_topk_softmax_spec(&adj);
        k = rocke_build_topk_softmax_new(b, &ts, arch);
    }
    else /* GATE_UP_GEMM / DOWN_GEMM -> batched-GEMM builder shape */
    {
        char name_buf[256];
        rocke_batched_gemm_spec_t gs;
        if(rocke_fmoe_forward_spec_to_batched_gemm_spec(&adj, name_buf, sizeof name_buf, &gs)
           == ROCKE_OK)
            k = rocke_build_batched_gemm_new(b, &gs, arch);
    }
    rocke_fmoe_build_ctx_destroy(ctx);
    free(ctx);
    return k;
}

std::string fmoe_serialize(const py::dict& d, const std::string& arch)
{
    (void)arch;
    Store st;
    rocke_fmoe_forward_spec_t s = fmoe_build(d, st);
    std::string out;
    for(size_t i = 0; i < FMOE_NSTAGES; ++i)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* k = fmoe_build_stage_kernel(&b, &s, FMOE_STAGES[i].stage);
        if(!k)
        {
            rocke_ir_builder_free(&b);
            throw std::runtime_error(
                std::string("rocke_engine.fused_moe_e2e_serialize_ir build failed (stage ")
                + FMOE_STAGES[i].banner + ")");
        }
        out += "; === fused_moe_e2e stage: ";
        out += FMOE_STAGES[i].banner;
        out += " ===\n";
        std::string text;
        try
        {
            text = ser_kernel(k, "rocke_engine.fused_moe_e2e_serialize_ir");
        }
        catch(...)
        {
            rocke_ir_builder_free(&b);
            throw;
        }
        out += text;
        if(text.empty() || text.back() != '\n')
            out += '\n';
        rocke_ir_builder_free(&b);
    }
    return out;
}

std::vector<std::string> fmoe_verify(const py::dict& d, const std::string& arch)
{
    (void)arch;
    Store st;
    rocke_fmoe_forward_spec_t s = fmoe_build(d, st);
    std::vector<std::string> out;
    for(size_t i = 0; i < FMOE_NSTAGES; ++i)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* k = fmoe_build_stage_kernel(&b, &s, FMOE_STAGES[i].stage);
        if(!k)
        {
            rocke_ir_builder_free(&b);
            throw std::runtime_error(
                std::string("rocke_engine.fused_moe_e2e_verify build failed (stage ")
                + FMOE_STAGES[i].banner + ")");
        }
        out.emplace_back(std::string("; === fused_moe_e2e stage: ") + FMOE_STAGES[i].banner
                         + " ===");
        std::vector<std::string> diags = ver_kernel(k);
        for(auto& ds : diags)
            out.emplace_back(std::move(ds));
        rocke_ir_builder_free(&b);
    }
    return out;
}

} // namespace

void register_fused_moe_e2e(py::module_& m)
{
    m.def("fused_moe_e2e_lower_llvm", &fmoe_lower, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def(
        "fused_moe_e2e_serialize_ir", &fmoe_serialize, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("fused_moe_e2e_verify", &fmoe_verify, py::arg("spec"), py::arg("arch") = "gfx950");
}
