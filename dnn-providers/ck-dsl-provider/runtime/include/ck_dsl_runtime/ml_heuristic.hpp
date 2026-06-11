// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Trained-model kernel selector for the ck_dsl runtime, mirroring the CK Tile
// dispatcher's ML heuristic (`dispatcher/include/ck_tile/dispatcher/ml_heuristic.hpp`):
// a LightGBM regressor predicts TFLOPS for each candidate from a 72-feature
// vector (problem + kernel config + hardware), and candidates are ranked
// best-first. The feature order is byte-identical to CK Tile's feature_spec.json,
// so the CK Tile `gemm_universal_fp16_gfx950` model can be reused directly for
// ck_dsl GEMM candidates (same knob space).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ck_dsl_runtime/dispatcher.hpp"
#include "ck_dsl_runtime/manifest.hpp"

namespace ck_dsl {

// LightGBM C API (libLightGBM / lib_lightgbm.so), declared like CK Tile does.
extern "C" {
int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
int LGBM_BoosterPredictForMat(void*, const void*, int, int, int, int, int, int, int, const char*,
                              int64_t*, double*);
int LGBM_BoosterFree(void*);
}

struct HardwareProfile {
    int num_cus = 256, simds_per_cu = 4, shader_engines = 32, max_clock_mhz = 2400,
        max_waves_per_cu = 32, wavefront_size = 64, lds_capacity = 65536, l1_cache_kb = 32,
        l2_cache_kb = 4096, l3_cache_kb = 262144, num_xcd = 8;
    int total_simds() const {
        return num_cus * simds_per_cu;
    }
};

// Kernel knobs the feature extractor needs, read from a ck_dsl manifest.
struct MlKernelConfig {
    int tile_m = 0, tile_n = 0, tile_k = 0;
    int warp_m = 1, warp_n = 1, warp_k = 1;
    int warp_tile_m = 0, warp_tile_n = 0, warp_tile_k = 0;
    int pipeline = 0, scheduler = 0, epilogue = 0;  // encoded like CK Tile
    bool pad_m = false, pad_n = false, pad_k = false, persistent = false;
    double dtype_bytes = 2.0;

    static int enc_pipeline(const std::string& p) {
        if (p == "compv3") return 0;
        if (p == "compv4") return 1;
        if (p == "compv5") return 2;
        if (p == "mem") return 3;
        if (p == "preshufflev2") return 4;
        return 0;
    }
    static int enc_scheduler(const std::string& s) {
        return s == "interwave" ? 1 : 0;
    }
    static int enc_epilogue(const std::string& e) {
        return e == "cshuffle" ? 1 : 0;
    }

    // Inverse decoders: turn the int-encoded knobs back into the C-engine's
    // string knob vocabulary (CEngine::*Problem.pipeline/scheduler/epilogue).
    // Used by the C-JIT path to feed the heuristic's pick into the C engine.
    static const char* dec_pipeline(int p) {
        switch (p) {
            case 1:
                return "compv4";
            case 2:
                return "compv5";
            case 3:
                return "mem";
            case 4:
                return "preshufflev2";
            case 0:
            default:
                return "compv3";
        }
    }
    static const char* dec_scheduler(int s) {
        return s == 1 ? "interwave" : "intrawave";
    }
    static const char* dec_epilogue(int e) {
        return e == 1 ? "cshuffle" : "default";
    }

    static MlKernelConfig from_manifest(const Manifest& m) {
        MlKernelConfig c;
        c.tile_m = m.block_m;
        c.tile_n = m.block_n;
        c.tile_k = m.block_k;
        const auto& r = m.raw;
        c.pipeline = enc_pipeline(r.get_str("pipeline", "compv4"));
        c.scheduler = enc_scheduler(r.get_str("scheduler", "intrawave"));
        c.epilogue = enc_epilogue(r.get_str("epilogue", "default"));
        // Optional richer config block (warp/warp_tile/pad/persistent).
        if (r.has("kernel_config") && r.at("kernel_config").is_object()) {
            const auto& kc = r.at("kernel_config");
            c.warp_m = (int)kc.get_int("warp_m", 1);
            c.warp_n = (int)kc.get_int("warp_n", 1);
            c.warp_k = (int)kc.get_int("warp_k", 1);
            c.warp_tile_m = (int)kc.get_int("warp_tile_m", 0);
            c.warp_tile_n = (int)kc.get_int("warp_tile_n", 0);
            c.warp_tile_k = (int)kc.get_int("warp_tile_k", 0);
            c.pad_m = kc.get_int("pad_m", 0) != 0;
            c.pad_n = kc.get_int("pad_n", 0) != 0;
            c.pad_k = kc.get_int("pad_k", 0) != 0;
            c.persistent = kc.get_int("persistent", 0) != 0;
        }
        return c;
    }
};

static constexpr int CKDSL_NUM_FEATURES = 72;

inline std::array<double, CKDSL_NUM_FEATURES> ml_extract_features(const Problem& prob,
                                                                  const MlKernelConfig& k,
                                                                  const HardwareProfile& hw) {
    double M = prob.M, N = prob.N, K = prob.K, sk = 1.0, bpe = k.dtype_bytes;
    double l2M = std::log2(std::max(M, 1.0)), l2N = std::log2(std::max(N, 1.0)),
           l2K = std::log2(std::max(K, 1.0)), l2MNK = std::log2(std::max(M * N * K, 1.0));
    double mem = (M * K + K * N + M * N) * bpe, ai = 2.0 * M * N * K / std::max(mem, 1.0);
    double ar_mn = M / std::max(N, 1.0), ar_mk = M / std::max(K, 1.0), ar_nk = N / std::max(K, 1.0);
    double layout = 0;  // RCR
    double tm = k.tile_m, tn = k.tile_n, tk = k.tile_k;
    double wm = k.warp_m, wn = k.warp_n, wk = k.warp_k;
    double wtm = k.warp_tile_m, wtn = k.warp_tile_n, wtk = k.warp_tile_k;
    double pipeline = k.pipeline, scheduler = k.scheduler, epilogue = k.epilogue;
    double pad_m = k.pad_m, pad_n = k.pad_n, pad_k = k.pad_k, persistent = k.persistent;
    double num_warps = wm * wn * wk, tile_volume = tm * tn * tk, tile_mn = tm * tn;
    double lest = (tm * tk + tn * tk) * bpe;
    double lcap = (k.pipeline == 1 /*CompV4*/) ? 32768.0 : (double)hw.lds_capacity;
    double lds_ratio = lest / std::max(lcap, 1.0);
    double ntm = std::ceil(M / std::max(tm, 1.0)), ntn = std::ceil(N / std::max(tn, 1.0)),
           ntk = std::ceil(K / std::max(tk, 1.0)), tot = ntm * ntn;
    auto ef = [](double d, double t) {
        if (t <= 0) return 1.0;
        double r = std::fmod(d, t);
        return r > 0 ? r / t : 1.0;
    };
    double em = ef(M, tm), en = ef(N, tn), ek = ef(K, tk), oeff = em * en * ek;
    double cu = tot / std::max((double)hw.num_cus, 1.0);
    double rm = M / std::max(tm, 1.0), rn = N / std::max(tn, 1.0), rk = K / std::max(tk, 1.0);
    double sm = M < tm, sn = N < tn, skk = K < tk, any_small = (M < tm || N < tn || K < tk);
    double nm = (tm > 0 && std::fmod(M, tm) != 0), nn = (tn > 0 && std::fmod(N, tn) != 0),
           nk = (tk > 0 && std::fmod(K, tk) != 0);
    double hpm = (nm && pad_m), hpn = (nn && pad_n), hpk = (nk && pad_k);
    double mm = (nm && !pad_m), mn = (nn && !pad_n), mk = (nk && !pad_k);
    double many = (mm || mn || mk);
    return {{M,
             N,
             K,
             sk,
             l2M,
             l2N,
             l2K,
             l2MNK,
             ai,
             ar_mn,
             ar_mk,
             ar_nk,
             layout,
             tm,
             tn,
             tk,
             wm,
             wn,
             wk,
             wtm,
             wtn,
             wtk,
             pipeline,
             scheduler,
             epilogue,
             pad_m,
             pad_n,
             pad_k,
             persistent,
             num_warps,
             tile_volume,
             tile_mn,
             lest,
             lds_ratio,
             ntm,
             ntn,
             ntk,
             tot,
             em,
             en,
             ek,
             oeff,
             cu,
             rm,
             rn,
             rk,
             sm,
             sn,
             skk,
             any_small,
             nm,
             nn,
             nk,
             hpm,
             hpn,
             hpk,
             mm,
             mn,
             mk,
             many,
             (double)hw.num_cus,
             (double)hw.simds_per_cu,
             (double)hw.total_simds(),
             (double)hw.shader_engines,
             (double)hw.max_clock_mhz,
             (double)hw.max_waves_per_cu,
             (double)hw.wavefront_size,
             (double)hw.lds_capacity,
             (double)hw.l1_cache_kb,
             (double)hw.l2_cache_kb,
             (double)hw.l3_cache_kb,
             (double)hw.num_xcd}};
}

// ---- FMHA (attention) ML features, mirroring CK Tile fmha_ml_heuristic.hpp ----

struct FmhaHardwareProfile {
    int num_cus = 304, simds_per_cu = 4, shader_engines = 32, max_clock_mhz = 2400,
        wavefront_size = 64, lds_capacity = 65536, num_xcd = 8;
    int total_simds() const {
        return num_cus * simds_per_cu;
    }
};

struct FmhaKernelConfig {
    double tm0 = 0, tn0 = 0, tk0 = 0, tn1 = 0, tk1 = 0, tk0max = 0;
    int pipeline = 1;  // qr_async
    int mask = 0, bias = 0;
    bool lse = false, sink = false, paged = true;
    static FmhaKernelConfig from_manifest(const Manifest& m, const Problem& p) {
        FmhaKernelConfig c;
        const auto& cfg = m.raw.has("attention_config") ? m.raw.at("attention_config") : m.raw;
        double hd = (double)cfg.get_int("head_size", p.hdim_q);
        double T = (double)cfg.get_int("tile_size", cfg.get_int("block_size", 16) * 2);
        double bq = (double)cfg.get_int("block_q", 16);
        c.tm0 = bq;
        c.tn0 = T;
        c.tk0 = hd;
        c.tn1 = (double)p.hdim_v;
        c.tk1 = T;
        c.tk0max = hd;
        c.mask = p.mask_type;
        c.sink = p.use_sinks;
        return c;
    }
};

inline double fmha_dtype_bytes(const std::string& dt) {
    if (dt == "fp32") return 4.0;
    if (dt == "fp8" || dt == "bf8") return 1.0;
    return 2.0;
}
inline int encode_fmha_dtype(const std::string& dt) {
    if (dt == "bf16") return 1;
    return 0;  // fp16=0
}

static constexpr int CKDSL_FMHA_NUM_FEATURES = 68;

inline std::array<double, CKDSL_FMHA_NUM_FEATURES> ml_extract_fmha_features(
    const Problem& p, const FmhaKernelConfig& k, const FmhaHardwareProfile& hw) {
    double batch = p.batch, sq = p.seqlen_q, sk = p.seqlen_k;
    double hq = p.nhead_q, hk = std::max((double)p.nhead_k, 1.0), dq = p.hdim_q, dv = p.hdim_v;
    double bpe = fmha_dtype_bytes(p.dtype), dt_enc = encode_fmha_dtype(p.dtype);
    auto l2 = [](double x) { return std::log2(std::max(x, 1.0)); };
    double gqa = hq / hk, asp = sq / std::max(sk, 1.0);
    double ops = 2.0 * batch * hq * sq * sk * (dq + dv);
    double mem = (batch * hq * sq * dq + batch * hk * sk * dq + batch * hk * sk * dv +
                  batch * hq * sq * dv) *
                 bpe;
    double ai = ops / std::max(mem, 1.0), decode = (sq <= 1) ? 1.0 : 0.0;
    double pip = k.pipeline, tm0 = k.tm0, tn0 = k.tn0, tk0 = k.tk0, tn1 = k.tn1, tk1 = k.tk1,
           tk0max = k.tk0max;
    double ps = 0, psk = 0, pd = 0, pdv = 0, mask = k.mask, bias = k.bias, lse = k.lse, dropout = 0,
           logits = 0, sink = k.sink, skip = 0, qscale = 0, paged = k.paged ? 1.0 : 0.0;
    double ntm = std::ceil(sq / std::max(tm0, 1.0)), ntk = std::ceil(sk / std::max(tn0, 1.0));
    double tot = batch * hq * ntm * ntk;
    auto eff = [](double d, double t) {
        if (t <= 0) return 1.0;
        double r = std::fmod(d, t);
        return r > 0 ? r / t : 1.0;
    };
    double esq = eff(sq, tm0), esk = eff(sk, tn0), oeff = esq * esk;
    double cu = tot / std::max((double)hw.num_cus, 1.0);
    double tvol = tm0 * tn0 * tk0, tarea = tm0 * tn0, lds = (tm0 * tk0 + tn0 * tk0) * bpe;
    double ldsr = lds / std::max((double)hw.lds_capacity, 1.0);
    double rdk0 = dq / std::max(tk0, 1.0), rdn1 = tn1 > 0 ? dv / tn1 : 0.0;
    double sq1 = sq <= tm0, sk1 = sk <= tn0, deq = (dq == dv), gqa_f = (hq != hk);
    double totq = batch * hq * sq * dq, totkv = batch * hk * sk * (dq + dv);
    double fc = lse + dropout + logits + sink + skip + paged + (mask > 0 ? 1.0 : 0.0) +
                (bias > 0 ? 1.0 : 0.0);
    return {{batch,
             sq,
             sk,
             hq,
             hk,
             dq,
             dv,
             dt_enc,
             l2(batch),
             l2(sq),
             l2(sk),
             l2(hq),
             l2(hk),
             l2(dq),
             l2(dv),
             gqa,
             asp,
             l2(ops),
             ai,
             decode,
             pip,
             tm0,
             tn0,
             tk0,
             tn1,
             tk1,
             tk0max,
             ps,
             psk,
             pd,
             pdv,
             mask,
             bias,
             lse,
             dropout,
             logits,
             sink,
             skip,
             qscale,
             paged,
             ntm,
             ntk,
             tot,
             esq,
             esk,
             oeff,
             cu,
             tvol,
             tarea,
             lds,
             ldsr,
             rdk0,
             rdn1,
             sq1,
             sk1,
             deq,
             gqa_f,
             totq,
             totkv,
             fc,
             (double)hw.num_cus,
             (double)hw.simds_per_cu,
             (double)hw.total_simds(),
             (double)hw.shader_engines,
             (double)hw.max_clock_mhz,
             (double)hw.wavefront_size,
             (double)hw.lds_capacity,
             (double)hw.num_xcd}};
}

// Trained-model heuristic: a Dispatcher::HeuristicFn that reorders the supported
// candidates by predicted TFLOPS (best first). Loads per-op LightGBM models from
// a model directory (<dir>/gemm/model_tflops.lgbm, <dir>/fmha/model_tflops.lgbm)
// and dispatches feature extraction by op. Reuses CK Tile's trained models
// directly (identical feature layouts). Conv reuses the GEMM model on the
// implicit-GEMM (M=N*Ho*Wo, N=K, K=R*S*C) problem.
class DslMlHeuristic {
   public:
    // model_dir holds per-op subdirs: <dir>/gemm/model_tflops.lgbm,
    // <dir>/fmha/model_tflops.lgbm. Missing models leave that op on FirstFit.
    DslMlHeuristic(const std::string& model_dir, const ArtifactStore* store) : store_(store) {
        gemm_booster_ = load(model_dir + "/gemm/model_tflops.lgbm");
        fmha_booster_ = load(model_dir + "/fmha/model_tflops.lgbm");
    }
    ~DslMlHeuristic() {
        if (gemm_booster_) LGBM_BoosterFree(gemm_booster_);
        if (fmha_booster_) LGBM_BoosterFree(fmha_booster_);
    }
    DslMlHeuristic(const DslMlHeuristic&) = delete;
    DslMlHeuristic& operator=(const DslMlHeuristic&) = delete;
    bool is_loaded() const {
        return gemm_booster_ || fmha_booster_;
    }
    bool has_gemm() const {
        return gemm_booster_ != nullptr;
    }
    bool has_fmha() const {
        return fmha_booster_ != nullptr;
    }

    // Predict TFLOPS for one candidate, dispatching feature extraction + model
    // by op. GEMM and Conv use the GEMM model (Conv via implicit-GEMM M/N/K);
    // attention uses the FMHA model.
    double predict_tflops(const Problem& prob, const Manifest& m) const {
        if (prob.op == "attention") {
            if (!fmha_booster_) return 0;
            auto f = ml_extract_fmha_features(prob, FmhaKernelConfig::from_manifest(m, prob), fhw_);
            return predict(fmha_booster_, f.data(), CKDSL_FMHA_NUM_FEATURES);
        }
        if (!gemm_booster_) return 0;
        auto f = ml_extract_features(prob, MlKernelConfig::from_manifest(m), ghw_);
        return predict(gemm_booster_, f.data(), CKDSL_NUM_FEATURES);
    }

    // Dispatcher::HeuristicFn: rank supported candidates by predicted TFLOPS.
    std::vector<Dispatcher::Choice> operator()(const Problem& prob,
                                               std::vector<Dispatcher::Choice> cands) const {
        if (!store_) return cands;
        bool have = (prob.op == "attention") ? has_fmha() : has_gemm();
        if (!have) return cands;  // no model for this op -> keep FirstFit order
        std::stable_sort(cands.begin(), cands.end(),
                         [&](const Dispatcher::Choice& a, const Dispatcher::Choice& b) {
                             return predict_tflops(prob, store_->at(a.cache_key).manifest) >
                                    predict_tflops(prob, store_->at(b.cache_key).manifest);
                         });
        return cands;
    }

   private:
    static void* load(const std::string& path) {
        void* b = nullptr;
        int iters = 0;
        if (LGBM_BoosterCreateFromModelfile(path.c_str(), &iters, &b) != 0) return nullptr;
        return b;
    }
    static double predict(void* booster, const double* feats, int n) {
        int64_t ol = 0;
        double pred = 0;
        // data_type=1 == C_API_DTYPE_FLOAT64 (features are double); row-major.
        if (LGBM_BoosterPredictForMat(booster, feats, 1, 1, n, 1, 0, 0, 0, "", &ol, &pred) != 0)
            return 0;
        return std::expm1(pred);  // tflops is a log target in both feature_spec.json
    }

    const ArtifactStore* store_ = nullptr;
    void* gemm_booster_ = nullptr;
    void* fmha_booster_ = nullptr;
    HardwareProfile ghw_;
    FmhaHardwareProfile fhw_;
};

}  // namespace ck_dsl
