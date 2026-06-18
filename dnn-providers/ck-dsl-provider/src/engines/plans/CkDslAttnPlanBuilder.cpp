// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslAttnPlanBuilder.hpp"

#include <cstdio>
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

bool debug_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CK_DSL_DEBUG");
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
    bool sel_valid = false, store_has = false;
    const bool dbg = debug_enabled();
    try {
        auto problem = CkDslAttnParamParser::buildProblem(params, handle.gfxArch());
        auto choice = handle.dispatcher().select(problem);
        sel_valid = choice.valid();
        store_has = sel_valid && handle.store().has(choice.cache_key);
        if (store_has) {
            const auto& m = handle.store().at(choice.cache_key).manifest;
            apply_attn_knobs(prob, m, problem);
        }
    } catch (const std::exception& e) {
        if (dbg) fprintf(stderr, "[ckdsl-cfg] select/knobs exception: %s\n", e.what());
    } catch (...) {
        // No registry / heuristic failure: keep the default knobs.
    }
    // Prefer the tiled MFMA attention kernel (the fast path); fall back to the
    // scalar 2D reference when the tiled gate declines the problem (arch not
    // gfx942/gfx950, head_size/block_size/GQA outside the tiled admission, ...).
    // build_sdpa_tiled throws TiledUnsupported on a decline (expected fallback)
    // and the base runtime_error on a genuine build failure (also falls back so
    // the engine still serves the graph via the reference kernel).
    ck_dsl::CEngineResult r;
    const char* path = "scalar";
    try {
        r = ck_dsl::CEngine::build_sdpa_tiled(prob);
        path = "tiled";
    } catch (const ck_dsl::TiledUnsupported& e) {
        if (dbg)
            fprintf(stderr, "[ckdsl-cfg] tiled declined (%s); falling back to scalar\n", e.what());
        r = ck_dsl::CEngine::build_sdpa(prob);
    } catch (const std::exception& e) {
        if (dbg)
            fprintf(stderr, "[ckdsl-cfg] tiled build failed (%s); falling back to scalar\n",
                    e.what());
        r = ck_dsl::CEngine::build_sdpa(prob);
    }

    if (dbg) {
        fprintf(stderr,
                "[ckdsl-cfg] B=%d Hq=%d Hkv=%d S=%d D=%d dtype=%s sel_valid=%d store_has=%d "
                "-> block_size=%d block_q=%d tile_size=%d num_warps=%d path=%s kernel=%s\n",
                prob.num_seqs, prob.num_query_heads, prob.num_kv_heads, prob.max_seqlen_q,
                prob.head_size, prob.dtype, (int)sel_valid, (int)store_has, prob.block_size,
                prob.block_q, prob.tile_size, prob.num_warps, path, r.manifest.kernel_name.c_str());
    }

    // The block size the build actually used (block_q overlay folded in).
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
    const bool dbg = debug_enabled();
    if (!CkDslAttnParamParser::isSdpaGraph(opGraph)) {
        if (dbg)
            fprintf(stderr, "[ckdsl-attn] DECLINE: not isSdpaGraph (nodeCount=%d)\n",
                    (int)opGraph.nodeCount());
        return false;
    }
    try {
        auto params = CkDslAttnParamParser::parseSdpaGraph(opGraph);
        if (params.is_bhsd) {
            if (dbg) fprintf(stderr, "[ckdsl-attn] DECLINE: is_bhsd (physical BHSD)\n");
            return false;
        }
        if (params.batch <= 0 || params.seqlen_q <= 0 || params.seqlen_k <= 0 ||
            params.nhead_q <= 0 || params.nhead_k <= 0 || params.hdim_q <= 0) {
            if (dbg) fprintf(stderr, "[ckdsl-attn] DECLINE: non-positive dim\n");
            return false;
        }
        if (c_jit_enabled()) {
            // C-JIT path: the kernel is generated on demand from the pure-C
            // engine, so applicability does NOT depend on the shipped
            // ArtifactStore/dispatcher catalog. The gate must still MATCH what
            // the C-JIT build can actually deliver, or buildPlan() will throw
            // after we already reported applicable.
            //
            // buildPlan() prefers the tiled MFMA kernel and falls back to the
            // scalar 2D reference (build_sdpa). The scalar gate is the broadest
            // capability, and on a scalar reject build_sdpa throws -- which is
            // NOT caught on the C-JIT path -- so admit exactly what the scalar
            // backend (ckc_unified_attention_supports_scalar) accepts:
            //   - dtype in {fp16, bf16}
            //   - head_size in {64, 128, 256}
            // (block_size is fixed to the valid 16 on this build path.)
            // Grouped-query attention additionally requires num_query_heads to
            // be a positive multiple of num_kv_heads.
            const bool dtype_ok = params.dtype == "fp16" || params.dtype == "bf16";
            const bool hdim_ok =
                params.hdim_q == 64 || params.hdim_q == 128 || params.hdim_q == 256;
            const bool gqa_ok = params.nhead_k > 0 && (params.nhead_q % params.nhead_k == 0);
            const bool ok = dtype_ok && hdim_ok && gqa_ok;
            if (dbg)
                fprintf(stderr,
                        "[ckdsl-attn] %s (C-JIT) dtype=%s hdim_q=%ld nhead_q=%ld nhead_k=%ld\n",
                        ok ? "ACCEPT" : "DECLINE", params.dtype.c_str(), params.hdim_q,
                        params.nhead_q, params.nhead_k);
            return ok;
        }
        auto problem = CkDslAttnParamParser::buildProblem(params, handle.gfxArch());
        bool ok = handle.dispatcher().select(problem).valid();
        if (dbg && !ok)
            fprintf(stderr, "[ckdsl-attn] DECLINE: dispatcher.select invalid (no C-JIT)\n");
        return ok;
    } catch (const std::exception& e) {
        if (dbg) fprintf(stderr, "[ckdsl-attn] DECLINE (exception): %s\n", e.what());
        return false;
    } catch (...) {
        if (dbg) fprintf(stderr, "[ckdsl-attn] DECLINE (unknown exception)\n");
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
