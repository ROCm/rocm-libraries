// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslAttnPlanBuilder.hpp"

#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>

#include "CkDslAttnPlan.hpp"
#include "CkDslContext.hpp"
#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/json.hpp"
#include "ck_dsl_runtime/timing.hpp"
#include "engines/CkDslAttnParamParser.hpp"

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

// Build a Kernel for an SDPA (unified-attention 2D scalar) problem directly from
// the C engine. The C-engine manifest does not carry the `attention_config` raw
// block that CkDslAttnPlan's constructor reads (block_size / block_q); synthesize
// it here so the existing Plan logic works unchanged on the C-JIT path.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_attn_kernel(
    const CkDslAttnParamParser::ParsedAttnParams& params, const std::string& arch,
    const std::string& isa) {
    ck_dsl::CEngine::SdpaProblem prob;
    prob.total_q = static_cast<int>(params.batch * params.seqlen_q);
    prob.num_seqs = static_cast<int>(params.batch);
    prob.num_query_heads = static_cast<int>(params.nhead_q);
    prob.num_kv_heads = static_cast<int>(params.nhead_k);
    prob.head_size = static_cast<int>(params.hdim_q);
    prob.block_size = 16;
    prob.max_seqlen_q = static_cast<int>(params.seqlen_q);
    prob.max_seqlen_k = static_cast<int>(params.seqlen_k);
    prob.dtype = params.dtype == "bf16" ? "bf16" : "fp16";
    prob.sliding_window = 0;
    prob.softcap = 0.0;
    prob.arch = arch.c_str();

    auto r = ck_dsl::CEngine::build_sdpa(prob);

    // Inject attention_config so CkDslAttnPlan (which reads block_size/block_q
    // from manifest.raw) is satisfied identically to the shipped-artifact path.
    ck_dsl::json::Object cfg;
    cfg["block_size"] = ck_dsl::json::Value(static_cast<double>(prob.block_size));
    cfg["block_q"] = ck_dsl::json::Value(static_cast<double>(prob.block_size));
    ck_dsl::json::Object root;
    root["attention_config"] = ck_dsl::json::Value(std::move(cfg));
    r.manifest.raw = ck_dsl::json::Value(std::move(root));

    return std::make_unique<ck_dsl::Kernel>(
        ck_dsl::Kernel::from_llvm_ir(std::move(r.llvm_ir), std::move(r.manifest), isa));
}
}  // namespace

bool CkDslAttnPlanBuilder::isApplicable(
    const CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    if (!CkDslAttnParamParser::isSdpaGraph(opGraph)) return false;
    try {
        auto params = CkDslAttnParamParser::parseSdpaGraph(opGraph);
        auto problem = CkDslAttnParamParser::buildProblem(params, handle.gfxArch());
        return handle.dispatcher().select(problem).valid();
    } catch (...) {
        return false;
    }
}

size_t CkDslAttnPlanBuilder::getMaxWorkspaceSize(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const CkDslSettings&) const {
    return 0;  // dense->paged metadata workspace TODO (see CkDslAttnPlan note)
}

void CkDslAttnPlanBuilder::initializeExecutionSettings(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslSettings&) const {}

void CkDslAttnPlanBuilder::buildPlan(
    const CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&, CkDslContext& ctx) const {
    auto params = CkDslAttnParamParser::parseSdpaGraph(opGraph);

    std::unique_ptr<ck_dsl::Kernel> kernel;
    if (c_jit_enabled()) {
        // C-JIT path: build the attention kernel .ll + manifest directly from
        // the pure-C engine (no ArtifactStore, no Python, no shipped HSACO).
        kernel = make_c_jit_attn_kernel(params, handle.gfxArch(), handle.isa());
    } else {
        // Default path: dispatcher select + ArtifactStore materialize.
        auto problem = CkDslAttnParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (!choice.valid())
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "CkDslAttn: no attention kernel for problem");
        kernel = std::make_unique<ck_dsl::Kernel>(
            handle.store().make_kernel(choice.cache_key, handle.isa()));
    }

    // Stage 4 (comgr compile if .ll-only) + module load, AOT. Timed (compileMs).
    {
        ck_dsl::ScopedTimer t("attn", ck_dsl::ScopedTimer::Unit::Ms);
        kernel->ensure_compiled();
    }

    ctx.setPlan(std::make_unique<CkDslAttnPlan>(std::move(params), std::move(kernel)));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> CkDslAttnPlanBuilder::getCustomKnobs(
    const CkDslHandle&, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&) const {
    return {};
}

}  // namespace ck_dsl_plugin
