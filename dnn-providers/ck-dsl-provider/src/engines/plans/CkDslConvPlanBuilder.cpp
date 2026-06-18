// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslConvPlanBuilder.hpp"

#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>

#include "CkDslContext.hpp"
#include "CkDslConvPlan.hpp"
#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/timing.hpp"
#include "engines/CkDslConvParamParser.hpp"

namespace ck_dsl_plugin {

namespace {
// C-JIT path gate (env CK_DSL_C_JIT=1); see CkDslGemmPlanBuilder.
bool c_jit_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CK_DSL_C_JIT");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
    }();
    return on;
}

// Overlay the heuristic's chosen knobs (from the winning candidate's manifest)
// onto a conv problem, replacing the implicit-GEMM tile defaults. Only knobs the
// manifest carries are applied (zero fields keep the POD default). Conv has no
// scheduler/warp_k knob in the C-engine spec; those manifest fields are ignored.
void apply_conv_knobs(ck_dsl::CEngine::ConvProblem& prob, const ck_dsl::MlKernelConfig& k) {
    if (k.tile_m > 0) prob.tile_m = k.tile_m;
    if (k.tile_n > 0) prob.tile_n = k.tile_n;
    if (k.tile_k > 0) prob.tile_k = k.tile_k;
    if (k.warp_m > 0) prob.warp_m = k.warp_m;
    if (k.warp_n > 0) prob.warp_n = k.warp_n;
    if (k.warp_tile_m > 0) prob.warp_tile_m = k.warp_tile_m;
    if (k.warp_tile_n > 0) prob.warp_tile_n = k.warp_tile_n;
    if (k.warp_tile_k > 0) prob.warp_tile_k = k.warp_tile_k;
    prob.pipeline = ck_dsl::MlKernelConfig::dec_pipeline(k.pipeline);
    prob.epilogue = ck_dsl::MlKernelConfig::dec_epilogue(k.epilogue);
}

// Build a Kernel for a conv (implicit-GEMM) problem directly from the C engine.
// When the dispatcher (LGBM heuristic if CK_DSL_ML_MODEL_DIR is set, FirstFit
// otherwise) selects a candidate from the registry, the JIT'd kernel uses that
// candidate's knobs; otherwise it falls back to the recon implicit-GEMM defaults.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_conv_kernel(
    const CkDslHandle& handle, const CkDslConvParamParser::ParsedConvParams& params,
    const std::string& arch, const std::string& isa) {
    ck_dsl::CEngine::ConvProblem prob;
    prob.N = params.N;
    prob.Hi = params.Hi;
    prob.Wi = params.Wi;
    prob.C = params.C;
    prob.K = params.K;
    prob.R = params.R;
    prob.S = params.S;
    prob.sH = params.sH;
    prob.sW = params.sW;
    prob.pH = params.pH;
    prob.pW = params.pW;
    prob.dH = params.dH;
    prob.dW = params.dW;
    prob.arch = arch.c_str();

    // Wire the heuristic into the C-JIT selection path.
    try {
        auto problem = CkDslConvParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (choice.valid() && handle.store().has(choice.cache_key)) {
            const auto& manifest = handle.store().at(choice.cache_key).manifest;
            apply_conv_knobs(prob, ck_dsl::MlKernelConfig::from_manifest(manifest));
        }
    } catch (...) {
        // No registry / heuristic failure: keep the recon defaults.
    }

    auto r = ck_dsl::CEngine::build_conv(prob);
    return std::make_unique<ck_dsl::Kernel>(
        ck_dsl::Kernel::from_llvm_ir(std::move(r.llvm_ir), std::move(r.manifest), isa));
}
}  // namespace

bool CkDslConvPlanBuilder::isApplicable(
    const CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    if (!CkDslConvParamParser::isConvGraph(opGraph)) return false;
    try {
        auto params = CkDslConvParamParser::parseConvGraph(opGraph);
        if (c_jit_enabled()) {
            // C-JIT path: the kernel is generated on demand from the pure-C
            // engine, so applicability does NOT depend on the shipped
            // ArtifactStore/dispatcher catalog (which is empty unless
            // CK_DSL_KERNEL_LIB_PATH is set). Accept any well-formed conv with a
            // dtype the C engine supports.
            if (params.N <= 0 || params.C <= 0 || params.K <= 0 || params.Hi <= 0 ||
                params.Wi <= 0 || params.R <= 0 || params.S <= 0 ||
                params.Ho() <= 0 || params.Wo() <= 0)
                return false;
            return params.dtype == "fp16" || params.dtype == "bf16";
        }
        // The conv ML model was trained on dilation=1 data only. For dilated
        // convolutions the heuristic still runs but may produce suboptimal
        // rankings; FirstFit is an equivalent fallback when no model is loaded.
        auto problem = CkDslConvParamParser::buildProblem(params, handle.gfxArch());
        return handle.dispatcher().select(problem).valid();
    } catch (...) {
        return false;
    }
}

size_t CkDslConvPlanBuilder::getMaxWorkspaceSize(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const CkDslSettings&) const {
    return 0;  // dense GEMM needs no scratch
}

void CkDslConvPlanBuilder::initializeExecutionSettings(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslSettings&) const {}

void CkDslConvPlanBuilder::buildPlan(
    const CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslContext& ctx) const {
    auto params = CkDslConvParamParser::parseConvGraph(opGraph);

    std::unique_ptr<ck_dsl::Kernel> kernel;
    if (c_jit_enabled()) {
        // C-JIT path: build the conv kernel .ll + manifest directly from the
        // pure-C engine (no ArtifactStore, no Python, no shipped HSACO).
        kernel = make_c_jit_conv_kernel(handle, params, handle.gfxArch(), handle.isa());
    } else {
        // Default path: dispatcher select + ArtifactStore materialize.
        auto problem = CkDslConvParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (!choice.valid())
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "CkDsl: no conv kernel for problem");
        kernel = std::make_unique<ck_dsl::Kernel>(
            handle.store().make_kernel(choice.cache_key, handle.isa()));
    }

    // Stage 4 (comgr compile if .ll-only) + module load, AOT. Timed (compileMs).
    {
        ck_dsl::ScopedTimer t("conv", ck_dsl::ScopedTimer::Unit::Ms);
        kernel->ensure_compiled();
    }

    ctx.setPlan(std::make_unique<CkDslConvPlan>(std::move(params), std::move(kernel)));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> CkDslConvPlanBuilder::getCustomKnobs(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&) const {
    return {};
}

}  // namespace ck_dsl_plugin
