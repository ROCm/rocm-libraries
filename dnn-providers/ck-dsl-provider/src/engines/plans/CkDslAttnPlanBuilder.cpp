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
#include "ck_dsl_runtime/manifest.hpp"
#include "ck_dsl_runtime/ml_heuristic.hpp"
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

// Overlay the heuristic's chosen attention knobs (from the winning candidate's
// manifest attention_config) onto an SDPA problem, replacing the scalar-path
// defaults. Mirrors apply_gemm_knobs / apply_conv_knobs: only knobs the manifest
// actually carries are applied (zero/blank fields keep the POD default), so a
// sparse manifest never zeroes out a valid block size. The knob the scalar 2D
// reference kernel consumes is the KV block size, carried by the manifest as
// block_q (== block_size in attention_config); tile_size / num_warps are carried
// for parity with the full FMHA knob space (consumed by the tiled kernel, not
// the scalar reference) so the overlay faithfully mirrors the GEMM path.
//
// Knobs are read via the same FmhaKernelConfig::from_manifest the LGBM feature
// extractor uses, so the config the heuristic RANKED on is exactly the config
// applied to the JIT build (no second, divergent parse of the manifest).
void apply_attn_knobs(ck_dsl::CEngine::SdpaProblem& prob, const ck_dsl::Manifest& m,
                      const ck_dsl::Problem& problem) {
    const auto& cfg = m.raw.has("attention_config") ? m.raw.at("attention_config") : m.raw;
    // block_q / block_size: the scalar path's KV block-size tile knob.
    int bq = static_cast<int>(cfg.get_int("block_q", 0));
    if (bq <= 0) bq = static_cast<int>(cfg.get_int("block_size", 0));
    if (bq > 0) prob.block_q = bq;
    // tile_size / num_warps: full FMHA knob space (parity with apply_gemm_knobs;
    // not consumed by the scalar lower, see SdpaProblem doc).
    int ts = static_cast<int>(cfg.get_int("tile_size", 0));
    if (ts > 0) prob.tile_size = ts;
    int nw = static_cast<int>(cfg.get_int("num_warps", 0));
    if (nw > 0) prob.num_warps = nw;
    // Cross-check the FmhaKernelConfig the heuristic ranked on resolves the same
    // block_q (tm0); keep prob.block_q authoritative if the manifest carried it.
    auto k = ck_dsl::FmhaKernelConfig::from_manifest(m, problem);
    if (prob.block_q <= 0 && k.tm0 > 0) prob.block_q = static_cast<int>(k.tm0);
}

// Build a Kernel for an SDPA (unified-attention 2D scalar) problem directly from
// the C engine. The C-engine manifest does not carry the `attention_config` raw
// block that CkDslAttnPlan's constructor reads (block_size / block_q); synthesize
// it here so the existing Plan logic works unchanged on the C-JIT path.
std::unique_ptr<ck_dsl::Kernel> make_c_jit_attn_kernel(
    const CkDslHandle& handle, const CkDslAttnParamParser::ParsedAttnParams& params,
    const std::string& arch, const std::string& isa) {
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

    // Wire the FMHA heuristic into the C-JIT selection path: rank registered
    // attention candidates (LGBM if CK_DSL_ML_MODEL_DIR is set, FirstFit
    // otherwise) and, if one wins, JIT the scalar SDPA kernel with that
    // candidate's attention knobs (apply_attn_knobs, mirroring the GEMM/conv
    // builders). No registry -> keep the scalar defaults.
    try {
        auto problem = CkDslAttnParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        if (choice.valid() && handle.store().has(choice.cache_key)) {
            const auto& m = handle.store().at(choice.cache_key).manifest;
            apply_attn_knobs(prob, m, problem);
        }
    } catch (...) {
        // No registry / heuristic failure: keep the default knobs.
    }

    auto r = ck_dsl::CEngine::build_sdpa(prob);

    // The block size build_sdpa actually used (block_q overlay folded in).
    const int used_block = prob.block_q > 0 ? prob.block_q : prob.block_size;

    // Inject attention_config so CkDslAttnPlan (which reads block_size/block_q
    // from manifest.raw) is satisfied identically to the shipped-artifact path.
    ck_dsl::json::Object cfg;
    cfg["block_size"] = ck_dsl::json::Value(static_cast<double>(used_block));
    cfg["block_q"] = ck_dsl::json::Value(static_cast<double>(used_block));
    if (prob.tile_size > 0)
        cfg["tile_size"] = ck_dsl::json::Value(static_cast<double>(prob.tile_size));
    if (prob.num_warps > 0)
        cfg["num_warps"] = ck_dsl::json::Value(static_cast<double>(prob.num_warps));
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
        if (c_jit_enabled()) {
            // C-JIT path: the kernel is generated on demand from the pure-C
            // engine, so applicability does NOT depend on the shipped
            // ArtifactStore/dispatcher catalog (which is empty unless
            // CK_DSL_KERNEL_LIB_PATH is set). Accept any well-formed SDPA in the
            // BSHD layout (BHSD is rejected at plan build) with a dtype the C
            // engine supports.
            if (params.is_bhsd) return false;
            if (params.batch <= 0 || params.seqlen_q <= 0 || params.seqlen_k <= 0 ||
                params.nhead_q <= 0 || params.nhead_k <= 0 || params.hdim_q <= 0)
                return false;
            return params.dtype == "fp16" || params.dtype == "bf16";
        }
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
        kernel = make_c_jit_attn_kernel(handle, params, handle.gfxArch(), handle.isa());
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
