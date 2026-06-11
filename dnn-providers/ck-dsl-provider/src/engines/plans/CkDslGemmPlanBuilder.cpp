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

// Build a Kernel for a GEMM problem directly from the C engine (.ll + manifest),
// set up for comgr-from-.ll compilation against `isa`.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_gemm_kernel(
    const CkDslParamParser::ParsedGemmParams& params, const std::string& arch,
    const std::string& isa) {
    ck_dsl::CEngine::GemmProblem prob;
    prob.M = static_cast<int>(params.M);
    prob.N = static_cast<int>(params.N);
    prob.K = static_cast<int>(params.K);
    const char* dt = params.dtype == "bf16" ? "bf16" : "fp16";
    prob.dtype_a = dt;
    prob.dtype_b = dt;
    prob.dtype_c = dt;
    // recon-proven default combo for the C engine demo path.
    prob.pipeline = "compv3";
    prob.epilogue = "default";
    prob.arch = arch.c_str();

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

    std::unique_ptr<ck_dsl::Kernel> kernel;
    if (c_jit_enabled()) {
        // C-JIT path: build the kernel .ll + manifest directly from the pure-C
        // engine (no ArtifactStore, no Python, no shipped HSACO).
        kernel = make_c_jit_gemm_kernel(params, handle.gfxArch(), handle.isa());
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
