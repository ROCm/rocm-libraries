// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_dispatcher.hpp"
#include "ck_tile/dispatcher/fmha_kernel_instance.hpp"
#include "ck_tile/dispatcher/fmha_problem.hpp"
#include "ck_tile/dispatcher/fmha_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {

// LightGBM C API (linked at runtime, no header dependency)
extern "C" {
int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
int LGBM_BoosterPredictForMat(
    void*, const void*, int, int, int, int, int, int, int, const char*, int64_t*, double*);
int LGBM_BoosterFree(void*);
}

struct FmhaHardwareProfile
{
    int num_cus        = 304;
    int simds_per_cu   = 4;
    int shader_engines = 32;
    int max_clock_mhz  = 2400;
    int wavefront_size = 64;
    int lds_capacity   = 65536;
    int num_xcd        = 8;
    int total_simds() const { return num_cus * simds_per_cu; }
};

inline int encode_fmha_pipeline(const std::string& p)
{
    if(p == "qr")
        return 0;
    if(p == "qr_async")
        return 1;
    if(p == "qr_async_trload")
        return 2;
    if(p == "qr_async_trload_v3")
        return 3;
    if(p == "qr_pagedkv")
        return 4;
    return 1;
}

inline double fmha_dtype_bytes(const std::string& dt)
{
    if(dt == "fp32")
        return 4.0;
    if(dt == "fp16" || dt == "bf16")
        return 2.0;
    if(dt == "fp8" || dt == "bf8" || dt == "fp8bf16" || dt == "fp8fp32")
        return 1.0;
    return 2.0;
}

inline int encode_fmha_dtype(const std::string& dt)
{
    if(dt == "fp16")
        return 0;
    if(dt == "bf16")
        return 1;
    if(dt == "fp8bf16")
        return 2;
    if(dt == "fp8fp32")
        return 3;
    return 0;
}

static constexpr int FMHA_NUM_FEATURES = 68;

inline std::array<double, FMHA_NUM_FEATURES> extract_fmha_features(const FmhaProblem& prob,
                                                                   const FmhaKernelKey& key,
                                                                   const FmhaHardwareProfile& hw)
{
    double batch = prob.batch, sq = prob.seqlen_q, sk = prob.seqlen_k;
    double hq = prob.nhead_q, hk = std::max(prob.nhead_k, (int64_t)1);
    double dq = prob.hdim_q, dv = prob.hdim_v;
    double bpe    = fmha_dtype_bytes(prob.data_type);
    double dt_enc = encode_fmha_dtype(prob.data_type);

    auto l2    = [](double x) { return std::log2(std::max(x, 1.0)); };
    double gqa = hq / hk;
    double asp = sq / std::max(sk, 1.0);
    double ops = 2.0 * batch * hq * sq * sk * (dq + dv);
    double mem = (batch * hq * sq * dq + batch * hk * sk * dq + batch * hk * sk * dv +
                  batch * hq * sq * dv) *
                 bpe;
    double ai     = ops / std::max(mem, 1.0);
    double decode = (sq <= 1) ? 1.0 : 0.0;

    double pip     = encode_fmha_pipeline(key.algorithm.pipeline_name);
    double tm0     = key.algorithm.tile_shape.m0;
    double tn0     = key.algorithm.tile_shape.n0;
    double tk0     = key.algorithm.tile_shape.k0;
    double tn1     = key.algorithm.tile_shape.n1;
    double tk1     = key.algorithm.tile_shape.k1;
    double tk0max  = key.algorithm.tile_shape.k0max;
    double ps      = key.signature.pad_s;
    double psk     = key.signature.pad_sk;
    double pd      = key.signature.pad_d;
    double pdv     = key.signature.pad_dv;
    double mask    = key.signature.mask_type;
    double bias    = key.signature.bias_type;
    double lse     = key.signature.has_lse ? 1.0 : 0.0;
    double dropout = key.signature.has_dropout ? 1.0 : 0.0;
    double logits  = key.signature.has_logits_soft_cap ? 1.0 : 0.0;
    double sink    = key.signature.has_sink ? 1.0 : 0.0;
    double skip    = key.signature.skip_min_seqlen_q ? 1.0 : 0.0;
    double qscale  = 0.0;
    double paged   = key.signature.use_paged_kv ? 1.0 : 0.0;

    double ntm = std::ceil(sq / std::max(tm0, 1.0));
    double ntk = std::ceil(sk / std::max(tn0, 1.0));
    double tot = batch * hq * ntm * ntk;
    auto eff   = [](double d, double t) -> double {
        if(t <= 0)
            return 1.0;
        double r = std::fmod(d, t);
        return r > 0 ? r / t : 1.0;
    };
    double esq   = eff(sq, tm0);
    double esk   = eff(sk, tn0);
    double oeff  = esq * esk;
    double cu    = tot / std::max((double)hw.num_cus, 1.0);
    double tvol  = tm0 * tn0 * tk0;
    double tarea = tm0 * tn0;
    double lds   = (tm0 * tk0 + tn0 * tk0) * bpe;
    double ldsr  = lds / std::max((double)hw.lds_capacity, 1.0);
    double rdk0  = dq / std::max(tk0, 1.0);
    double rdn1  = tn1 > 0 ? dv / tn1 : 0.0;
    double sq1   = sq <= tm0 ? 1.0 : 0.0;
    double sk1   = sk <= tn0 ? 1.0 : 0.0;
    double deq   = (dq == dv) ? 1.0 : 0.0;
    double gqa_f = (hq != hk) ? 1.0 : 0.0;
    double totq  = batch * hq * sq * dq;
    double totkv = batch * hk * sk * (dq + dv);
    double fc    = lse + dropout + logits + sink + skip + paged + (mask > 0 ? 1.0 : 0.0) +
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

class FmhaMLHeuristic
{
    public:
    FmhaMLHeuristic(const std::string& model_path,
                    const FmhaRegistry* registry,
                    FmhaHardwareProfile hw = {},
                    bool log_transform     = true)
        : registry_(registry), hw_(hw), log_transform_(log_transform)
    {
        int iters = 0;
        if(LGBM_BoosterCreateFromModelfile(model_path.c_str(), &iters, &booster_) != 0 || !booster_)
        {
            std::cerr << "FmhaMLHeuristic: Failed to load " << model_path << std::endl;
            booster_ = nullptr;
        }
        else
        {
            std::cout << "FmhaMLHeuristic: Loaded (" << iters << " trees)" << std::endl;
        }
    }

    ~FmhaMLHeuristic()
    {
        if(booster_)
            LGBM_BoosterFree(booster_);
    }

    FmhaMLHeuristic(const FmhaMLHeuristic&)            = delete;
    FmhaMLHeuristic& operator=(const FmhaMLHeuristic&) = delete;

    bool is_loaded() const { return booster_ != nullptr; }

    double predict_tflops(const FmhaProblem& prob, const FmhaKernelKey& key) const
    {
        if(!booster_)
            return 0.0;
        auto features   = extract_fmha_features(prob, key, hw_);
        int64_t out_len = 0;
        double pred     = 0.0;
        // data_type=1 (C_API_DTYPE_FLOAT64): features is std::array<double>.
        if(LGBM_BoosterPredictForMat(booster_,
                                     features.data(),
                                     1,
                                     1,
                                     FMHA_NUM_FEATURES,
                                     1,
                                     0,
                                     0,
                                     0,
                                     "",
                                     &out_len,
                                     &pred) != 0)
            return 0.0;
        return log_transform_ ? std::expm1(pred) : pred;
    }

    /// FmhaHeuristicFunction-compatible: returns kernel IDs ranked by predicted TFLOPS.
    std::vector<std::string> operator()(const FmhaProblem& prob) const
    {
        if(!booster_ || !registry_)
            return {};

        auto kernels = registry_->get_all();
        struct Candidate
        {
            std::string id;
            double tflops;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(kernels.size());

        for(const auto& k : kernels)
        {
            const auto& key = k->get_key();
            double t        = predict_tflops(prob, key);
            candidates.push_back({key.encode_identifier(), t});
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.tflops > b.tflops;
        });

        std::vector<std::string> result;
        result.reserve(candidates.size());
        for(auto& c : candidates)
            result.push_back(std::move(c.id));
        return result;
    }

    private:
    void* booster_                = nullptr;
    const FmhaRegistry* registry_ = nullptr;
    FmhaHardwareProfile hw_;
    bool log_transform_ = true;
};

} // namespace dispatcher
} // namespace ck_tile
