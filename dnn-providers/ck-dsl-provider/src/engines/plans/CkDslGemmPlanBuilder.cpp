// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslGemmPlanBuilder.hpp"

#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>

#include "CkDslContext.hpp"
#include "CkDslGemmPlan.hpp"
#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/timing.hpp"
#include "engines/CkDslParamParser.hpp"

namespace ck_dsl_plugin {

namespace {
// C-JIT path gate (env CK_DSL_C_JIT=1). When set, kernels are JIT-built from the
// pure-C ck_dsl engine at plan-build time (no shipped artifact, no Python);
// otherwise the default ArtifactStore lookup is used.
bool c_jit_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CK_DSL_C_JIT");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
    }();
    return on;
}

// Populate a C-engine GEMM problem from the parsed graph params + the recon
// default knobs (compv3/default tile combo). Strings (pipeline/epilogue/...)
// keep their POD defaults except for the demo-proven overrides below; dtype is
// derived from the graph. Pointers into `arch`/`dt` stay valid for the caller's
// build_gemm() call (their backing storage outlives this populate step).
ck_dsl::CEngine::GemmProblem make_gemm_problem(const CkDslParamParser::ParsedGemmParams& params,
                                               const std::string& arch, const char* dt) {
    ck_dsl::CEngine::GemmProblem prob;
    prob.M = static_cast<int>(params.M);
    prob.N = static_cast<int>(params.N);
    prob.K = static_cast<int>(params.K);
    prob.dtype_a = dt;
    prob.dtype_b = dt;
    prob.dtype_c = dt;
    // recon-proven default combo for the C engine demo path.
    prob.pipeline = "compv3";
    prob.epilogue = "default";
    prob.arch = arch.c_str();
    return prob;
}

// Overlay the heuristic's chosen knobs (extracted from the winning candidate's
// manifest) onto a GEMM problem, replacing the hardcoded defaults. Only knobs
// the manifest actually carries are applied; zero/blank fields keep the POD
// default so a sparse manifest never zeroes out a valid tile dim. The decoded
// pipeline/scheduler/epilogue strings are owned by the MlKernelConfig encoding
// tables (static string literals), so they stay valid for build_gemm().
void apply_gemm_knobs(ck_dsl::CEngine::GemmProblem& prob, const ck_dsl::MlKernelConfig& k) {
    if (k.tile_m > 0) prob.tile_m = k.tile_m;
    if (k.tile_n > 0) prob.tile_n = k.tile_n;
    if (k.tile_k > 0) prob.tile_k = k.tile_k;
    if (k.warp_m > 0) prob.warp_m = k.warp_m;
    if (k.warp_n > 0) prob.warp_n = k.warp_n;
    if (k.warp_k > 0) prob.warp_k = k.warp_k;
    if (k.warp_tile_m > 0) prob.warp_tile_m = k.warp_tile_m;
    if (k.warp_tile_n > 0) prob.warp_tile_n = k.warp_tile_n;
    if (k.warp_tile_k > 0) prob.warp_tile_k = k.warp_tile_k;
    prob.pipeline = ck_dsl::MlKernelConfig::dec_pipeline(k.pipeline);
    prob.scheduler = ck_dsl::MlKernelConfig::dec_scheduler(k.scheduler);
    prob.epilogue = ck_dsl::MlKernelConfig::dec_epilogue(k.epilogue);
}

// Build a Kernel for a GEMM problem directly from the C engine (.ll + manifest),
// set up for comgr-from-.ll compilation against `isa`. When the dispatcher (with
// the LGBM heuristic if CK_DSL_ML_MODEL_DIR is set, FirstFit otherwise) can
// select a candidate from the registry, the JIT'd kernel uses THAT candidate's
// knobs; otherwise (empty registry / no model) it falls back to recon defaults.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_gemm_kernel(
    const CkDslHandle& handle, const CkDslParamParser::ParsedGemmParams& params,
    const std::string& arch, const std::string& isa) {
    const char* dt = params.dtype == "bf16" ? "bf16" : "fp16";
    ck_dsl::CEngine::GemmProblem prob = make_gemm_problem(params, arch, dt);

    // Wire the heuristic into the C-JIT selection path: rank registered
    // candidates for this problem and, if one wins, JIT it with its knobs.
    try {
        auto problem = CkDslParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (choice.valid() && handle.store().has(choice.cache_key)) {
            const auto& manifest = handle.store().at(choice.cache_key).manifest;
            apply_gemm_knobs(prob, ck_dsl::MlKernelConfig::from_manifest(manifest));
        }
    } catch (...) {
        // No registry / heuristic failure: keep the recon defaults.
    }

    auto r = ck_dsl::CEngine::build_gemm(prob);
    return std::make_unique<ck_dsl::Kernel>(
        ck_dsl::Kernel::from_llvm_ir(std::move(r.llvm_ir), std::move(r.manifest), isa));
}
}  // namespace

bool CkDslGemmPlanBuilder::isApplicable(
    const CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    if (!CkDslParamParser::isGemmGraph(opGraph)) return false;
    try {
        auto params = CkDslParamParser::parseGemmGraph(opGraph);
        // Reject B layouts the shipped RCR GEMM can't execute (e.g. row-major
        // [K,N] / NN, or an unrecognized stride pattern) so the engine manager
        // can fall back to another provider instead of us silently miscomputing.
        if (!CkDslParamParser::isSupportedBLayout(params.b_layout)) return false;
        if (c_jit_enabled()) {
            // C-JIT path: the kernel is generated on demand from the pure-C
            // engine, so applicability does NOT depend on the shipped
            // ArtifactStore/dispatcher catalog (which is empty unless
            // CK_DSL_KERNEL_LIB_PATH is set). Accept any well-formed GEMM with a
            // dtype the C engine supports.
            if (params.M <= 0 || params.N <= 0 || params.K <= 0) return false;
            return params.dtype == "fp16" || params.dtype == "bf16";
        }
        auto problem = CkDslParamParser::buildProblem(params, handle.gfxArch());
        return handle.dispatcher().select(problem).valid();
    } catch (...) {
        return false;
    }
}

size_t CkDslGemmPlanBuilder::getMaxWorkspaceSize(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const CkDslSettings&) const {
    return 0;  // dense GEMM needs no scratch
}

void CkDslGemmPlanBuilder::initializeExecutionSettings(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslSettings&) const {}

void CkDslGemmPlanBuilder::buildPlan(
    const CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslContext& ctx) const {
    auto params = CkDslParamParser::parseGemmGraph(opGraph);

    // Guard the execute path: the shipped GEMM is RCR-only. Reject any other
    // detected B layout with a clear BAD_PARAM status rather than launching the
    // RCR kernel against a row-major [K,N] buffer and returning wrong results.
    if (!CkDslParamParser::isSupportedBLayout(params.b_layout)) {
        std::string msg = std::string("CkDsl: unsupported GEMM B layout '") +
                          CkDslParamParser::bLayoutName(params.b_layout) +
                          "' (only RCR / B stored [N,K] is supported)";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, msg);
    }

    std::unique_ptr<ck_dsl::Kernel> kernel;
    if (c_jit_enabled()) {
        // C-JIT path: build the kernel .ll + manifest directly from the pure-C
        // engine (no ArtifactStore, no Python, no shipped HSACO).
        kernel = make_c_jit_gemm_kernel(handle, params, handle.gfxArch(), handle.isa());
    } else {
        // Default path: dispatcher selects a shipped candidate, ArtifactStore
        // materializes it (prebuilt HSACO, else comgr-from-.ll).
        auto problem = CkDslParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (!choice.valid())
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "CkDsl: no GEMM kernel for problem");
        kernel = std::make_unique<ck_dsl::Kernel>(
            handle.store().make_kernel(choice.cache_key, handle.isa()));
    }

    // Stage 4 (comgr compile if .ll-only) + module load, AOT at plan-build time.
    // Timed under CK_DSL_TIME=1 (compileMs).
    {
        ck_dsl::ScopedTimer t("gemm", ck_dsl::ScopedTimer::Unit::Ms);
        kernel->ensure_compiled();
    }

    ctx.setPlan(std::make_unique<CkDslGemmPlan>(std::move(params), std::move(kernel)));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> CkDslGemmPlanBuilder::getCustomKnobs(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&) const {
    return {};
}

}  // namespace ck_dsl_plugin
