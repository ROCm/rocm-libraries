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

// Build a Kernel for a conv (implicit-GEMM) problem directly from the C engine.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_conv_kernel(
    const CkDslConvParamParser::ParsedConvParams& params, const std::string& arch,
    const std::string& isa) {
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
        kernel = make_c_jit_conv_kernel(params, handle.gfxArch(), handle.isa());
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
