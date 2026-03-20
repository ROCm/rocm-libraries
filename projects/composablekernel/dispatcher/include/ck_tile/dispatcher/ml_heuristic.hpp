// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
namespace ck_tile {
namespace dispatcher {
extern "C" {
int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
int LGBM_BoosterPredictForMat(
    void*, const void*, int, int, int, int, int, int, int, const char*, int64_t*, double*);
int LGBM_BoosterFree(void*);
}
inline int encode_pipeline(Pipeline p)
{
    switch(p)
    {
    case Pipeline::CompV3: return 0;
    case Pipeline::CompV4: return 1;
    case Pipeline::CompV5: return 2;
    case Pipeline::Mem: return 3;
    case Pipeline::PreShuffleV2: return 4;
    default: return 0;
    }
}
inline int encode_scheduler(Scheduler s)
{
    switch(s)
    {
    case Scheduler::Intrawave: return 0;
    case Scheduler::Interwave: return 1;
    default: return 0;
    }
}
inline int encode_epilogue(Epilogue e)
{
    switch(e)
    {
    case Epilogue::Default: return 0;
    case Epilogue::CShuffle: return 1;
    default: return 0;
    }
}
inline int encode_layout(LayoutTag a, LayoutTag b, LayoutTag c)
{
    bool ra = (a == LayoutTag::RowMajor), rb = (b == LayoutTag::RowMajor);
    if(ra && !rb)
        return 0;
    if(ra && rb)
        return 1;
    if(!ra && rb)
        return 2;
    return 3;
}
inline double dtype_bytes_ml(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return 4;
    case DataType::FP16:
    case DataType::BF16: return 2;
    case DataType::FP8:
    case DataType::BF8:
    case DataType::INT8: return 1;
    case DataType::INT4: return 0.5;
    default: return 2;
    }
}
struct HardwareProfile
{
    int num_cus = 256, simds_per_cu = 4, shader_engines = 32, max_clock_mhz = 2400,
        max_waves_per_cu = 32, wavefront_size = 64, lds_capacity = 65536, l1_cache_kb = 32,
        l2_cache_kb = 4096, l3_cache_kb = 262144, num_xcd = 8;
    int total_simds() const { return num_cus * simds_per_cu; }
};
static constexpr int NUM_FEATURES = 55;
inline std::array<double, NUM_FEATURES>
extract_features(const Problem& prob, const KernelKey& key, const HardwareProfile& hw)
{
    double M = prob.M, N = prob.N, K = prob.K, sk = (prob.k_batch > 0 ? prob.k_batch : 1),
           bpe = dtype_bytes_ml(key.signature.dtype_a);
    double l2M = std::log2(std::max(M, 1.0)), l2N = std::log2(std::max(N, 1.0)),
           l2K = std::log2(std::max(K, 1.0)), l2MNK = std::log2(std::max(M * N * K, 1.0));
    double mem = (M * K + K * N + M * N) * bpe, ai = 2.0 * M * N * K / std::max(mem, 1.0);
    double tm = key.algorithm.tile_shape.m, tn = key.algorithm.tile_shape.n,
           tk = key.algorithm.tile_shape.k;
    double wm = key.algorithm.wave_shape.m, wn = key.algorithm.wave_shape.n,
           wk  = key.algorithm.wave_shape.k;
    double wtm = key.algorithm.warp_tile_shape.m, wtn = key.algorithm.warp_tile_shape.n,
           wtk  = key.algorithm.warp_tile_shape.k;
    double lest = (tm * tk + tn * tk) * bpe,
           lcap = (key.algorithm.pipeline == Pipeline::CompV4) ? 32768.0 : (double)hw.lds_capacity;
    double ntm = std::ceil(M / std::max(tm, 1.0)), ntn = std::ceil(N / std::max(tn, 1.0)),
           ntk = std::ceil(K / std::max(tk, 1.0));
    auto ef    = [](double d, double t) -> double {
        if(t <= 0)
            return 1.0;
        double r = std::fmod(d, t);
        return r > 0 ? r / t : 1.0;
    };
    return {{M,
             N,
             K,
             sk,
             l2M,
             l2N,
             l2K,
             l2MNK,
             ai,
             M / std::max(N, 1.0),
             M / std::max(K, 1.0),
             N / std::max(K, 1.0),
             (double)encode_layout(
                 key.signature.layout_a, key.signature.layout_b, key.signature.layout_c),
             tm,
             tn,
             tk,
             wm,
             wn,
             wk,
             wtm,
             wtn,
             wtk,
             (double)encode_pipeline(key.algorithm.pipeline),
             (double)encode_scheduler(key.algorithm.scheduler),
             (double)encode_epilogue(key.algorithm.epilogue),
             0.0,
             0.0,
             0.0,
             key.algorithm.persistent ? 1.0 : 0.0,
             wm * wn * wk,
             tm * tn * tk,
             tm * tn,
             lest,
             lest / std::max(lcap, 1.0),
             ntm,
             ntn,
             ntk,
             ntm * ntn,
             ef(M, tm),
             ef(N, tn),
             ef(K, tk),
             ef(M, tm) * ef(N, tn) * ef(K, tk),
             ntm * ntn / std::max((double)hw.num_cus, 1.0),
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
class MLHeuristic
{
    public:
    MLHeuristic(const std::string& path,
                const Registry* reg,
                HardwareProfile hw = {},
                bool log_t         = false)
        : registry_(reg), hw_(hw), log_t_(log_t)
    {
        int iters = 0;
        if(LGBM_BoosterCreateFromModelfile(path.c_str(), &iters, &b_) != 0 || !b_)
        {
            std::cerr << "MLHeuristic: Failed to load " << path << std::endl;
            b_ = nullptr;
        }
        else
            std::cout << "MLHeuristic: Loaded (" << iters << " iters)" << std::endl;
    }
    ~MLHeuristic()
    {
        if(b_)
            LGBM_BoosterFree(b_);
    }
    MLHeuristic(const MLHeuristic&)            = delete;
    MLHeuristic& operator=(const MLHeuristic&) = delete;
    bool is_loaded() const { return b_ != nullptr; }
    double predict_tflops(const Problem& prob, const KernelKey& key) const
    {
        if(!b_)
            return 0;
        auto f      = extract_features(prob, key, hw_);
        int64_t ol  = 0;
        double pred = 0;
        if(LGBM_BoosterPredictForMat(
               b_, f.data(), 0, 1, NUM_FEATURES, 1, 0, 0, 0, "", &ol, &pred) != 0)
            return 0;
        return log_t_ ? std::expm1(pred) : pred;
    }
    std::vector<std::string> operator()(const Problem& prob) const
    {
        if(!b_ || !registry_)
            return {};
        auto insts = registry_->get_all_instances();
        struct C
        {
            std::string id;
            double t;
        };
        std::vector<C> cs;
        cs.reserve(insts.size());
        for(auto& i : insts)
        {
            auto& k = i->get_key();
            cs.push_back({k.encode_identifier(), predict_tflops(prob, k)});
        }
        std::sort(cs.begin(), cs.end(), [](auto& a, auto& b) { return a.t > b.t; });
        std::vector<std::string> r;
        r.reserve(cs.size());
        for(auto& c : cs)
            r.push_back(std::move(c.id));
        return r;
    }

    private:
    void* b_                  = nullptr;
    const Registry* registry_ = nullptr;
    HardwareProfile hw_;
    bool log_t_ = false;
};
} // namespace dispatcher
} // namespace ck_tile
