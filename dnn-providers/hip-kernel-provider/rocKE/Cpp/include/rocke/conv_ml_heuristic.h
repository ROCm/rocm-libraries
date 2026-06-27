/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/conv_ml_heuristic.h -- LightGBM-based ML heuristic for rocKE implicit-
 * GEMM convolution kernel selection.
 *
 * Mirrors the Python GroupedConvFeatureEngine.extract() superset (109 features).
 * feature_spec.json written by train.py supplies feature_indices to project the
 * full superset down to the arch-specific subset the model was trained on.
 *
 * Usage:
 *   ConvMLHeuristic h("path/to/models/grouped_conv_forward_fp16_gfx950",
 *                     "gfx950", "fp16");
 *   if (h.is_loaded()) {
 *       double score = h.predict_tflops(prob, spec);
 *   }
 *
 * Requires LightGBM shared library at link time (lib_lightgbm / LightGBM).
 * The CMake option ROCKE_CONV_ML_HEURISTIC enables the linkage dependency.
 */
#pragma once

#ifdef __cplusplus

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

/* LightGBM C API — weak symbols so the plugin loads even when liblgbm.so is
 * absent at runtime (e.g. containers without LightGBM installed).  Callers
 * must check is_loaded() before invoking predict_tflops(). */
extern "C" {
__attribute__((weak)) int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
__attribute__((weak)) int LGBM_BoosterPredictForMat(
    void*, const void*, int, int, int, int, int, int, int, const char*, int64_t*, double*);
__attribute__((weak)) int LGBM_BoosterFree(void*);
}

namespace rocke {

/* -------------------------------------------------------------------------
 * Hardware profile — mirrors GroupedConvFeatureEngine hw_* fields.
 * -------------------------------------------------------------------------*/
struct ConvHwProfile
{
    int num_cus         = 304;
    int simds_per_cu    = 4;
    int shader_engines  = 38;
    int max_clock_mhz   = 2400;
    int max_waves_per_cu = 32;
    int wavefront_size  = 64;
    int lds_capacity    = 65536;
    int l1_cache_kb     = 32;
    int l2_cache_kb     = 4096;
    int l3_cache_kb     = 262144;
    int num_xcd         = 8;

    int total_simds() const { return num_cus * simds_per_cu; }

    static ConvHwProfile for_arch(const std::string& arch)
    {
        if(arch == "gfx950")
            return ConvHwProfile{}; /* MI350X: default values */
        if(arch == "gfx942")
        {
            ConvHwProfile p;
            p.num_cus        = 228;
            p.shader_engines = 28;
            p.max_clock_mhz  = 2100;
            p.l3_cache_kb    = 262144;
            p.num_xcd        = 8;
            return p;
        }
        if(arch == "gfx90a")
        {
            ConvHwProfile p;
            p.num_cus        = 104;
            p.shader_engines = 4;
            p.max_clock_mhz  = 1700;
            p.l1_cache_kb    = 16;
            p.l2_cache_kb    = 8192;
            p.l3_cache_kb    = 0;
            p.num_xcd        = 1;
            return p;
        }
        return ConvHwProfile{};
    }
};

/* -------------------------------------------------------------------------
 * Pipeline encoding — mirrors conv_encode_pipeline in feature engine.
 * -------------------------------------------------------------------------*/
inline int conv_encode_pipeline(const std::string& p)
{
    if(p == "compv3") return 0;
    if(p == "compv4") return 1;
    if(p == "compv5") return 2;
    if(p == "mem")    return 3;
    if(p == "preshufflev2") return 4;
    if(p == "basic_v1")     return 5;
    if(p == "compv6") return 6;
    return 0;
}

inline double conv_dtype_bytes(const std::string& dt)
{
    if(dt == "fp32" || dt == "f32")           return 4.0;
    if(dt == "fp8" || dt == "bf8" || dt == "int8") return 1.0;
    return 2.0; /* fp16 / bf16 */
}

/* -------------------------------------------------------------------------
 * Feature extraction — C++ mirror of GroupedConvFeatureEngine.extract().
 * Returns the full 109-feature superset in the same order as the Python
 * engine.  ConvMLHeuristic projects it down to the model subset via
 * feature_spec.json indices.
 *
 * prob : rocke_conv_problem_t  (N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, ...)
 * spec : rocke_implicit_gemm_conv_spec_t  (tile_m, tile_n, tile_k, pipeline,
 *                                          warp_m, warp_n, wave_size)
 * hw   : ConvHwProfile
 * dtype: "fp16" | "fp32" | "fp8" ...
 * -------------------------------------------------------------------------*/
inline std::vector<double>
conv_extract_features(const rocke_conv_problem_t&             prob,
                      const rocke_implicit_gemm_conv_spec_t&  spec,
                      const ConvHwProfile&                    hw,
                      const std::string&                      dtype)
{
    const int N  = prob.N;
    const int C  = prob.C;
    const int K  = prob.K;
    const int G  = 1; /* rocKE implicit-gemm conv is always G=1 */
    const int Hi = prob.Hi;
    const int Wi = prob.Wi;
    const int Y  = prob.Y;
    const int X  = prob.X;

    const int stride_h   = prob.sH > 0 ? prob.sH : 1;
    const int stride_w   = prob.sW > 0 ? prob.sW : 1;
    const int pad_h      = prob.pH;
    const int pad_w      = prob.pW;
    const int dilation_h = prob.dH > 0 ? prob.dH : 1;
    const int dilation_w = prob.dW > 0 ? prob.dW : 1;

    const double is_3d = 0.0; /* rocKE implicit-gemm currently 2-D only */

    const int effY = (Y - 1) * dilation_h + 1;
    const int effX = (X - 1) * dilation_w + 1;
    const int Ho   = (Hi + 2 * pad_h - effY) / stride_h + 1;
    const int Wo   = (Wi + 2 * pad_w - effX) / stride_w + 1;

    auto l2 = [](double v) { return std::log2(std::max(v, 1.0)); };

    const double log2_N  = l2(N),  log2_C = l2(C), log2_K = l2(K), log2_G = l2(G);
    const double log2_Hi = l2(Hi), log2_Wi = l2(Wi);
    const double spatial_volume = (double)Hi * Wi;
    const double filter_volume  = (double)Y * X;
    const double output_volume  = (double)Ho * Wo;
    const double log2_spatial   = l2(spatial_volume);
    const double log2_filter    = l2(filter_volume);
    const double log2_output    = l2(output_volume);

    const double bpe   = conv_dtype_bytes(dtype);
    const double cpg   = (double)C / G;
    const double flops = (double)N * K * output_volume * cpg * filter_volume * 2.0;
    const double bytes_io =
        ((double)N * C * spatial_volume + (double)K * cpg * filter_volume +
         (double)N * K * output_volume) * bpe;
    const double ai          = flops / std::max(bytes_io, 1.0);
    const double filter_area = filter_volume;
    const double is_1x1      = (Y == 1 && X == 1) ? 1.0 : 0.0;
    const double is_3x3      = (Y == 3 && X == 3) ? 1.0 : 0.0;
    const double aspect_hw   = (double)Hi / std::max(Wi, 1);
    const double aspect_filt = (double)Y  / std::max(X,  1);

    const double ocpg      = (double)K / G;
    const double log2_cpg  = l2(cpg), log2_ocpg = l2(ocpg);
    const double is_depthwise          = (G == C && G == K) ? 1.0 : 0.0;
    const double group_density         = (double)G / std::max(C, 1);
    const double is_small_group        = (cpg < 16.0 || ocpg < 16.0) ? 1.0 : 0.0;
    const double cprod                 = cpg * ocpg;
    const double batch_group           = (double)N * G;
    const double is_small_batch_grouped = (N < 8 && G > 1) ? 1.0 : 0.0;
    const double k_per_c               = (double)K / std::max(C, 1);

    const std::string pipeline_str = spec.pipeline ? spec.pipeline : "mem";
    const int pipeline_code        = conv_encode_pipeline(pipeline_str);

    /* block_size = warp_m * warp_n * wave_size */
    const int block_size = rocke_implicit_gemm_conv_spec_block_size(&spec);
    const int tile_m     = spec.tile_m;
    const int tile_n     = spec.tile_n;
    const int tile_k     = spec.tile_k > 0 ? spec.tile_k : 64;

    const double num_warps = (double)block_size / 4.0;
    const double tile_vol  = (double)tile_m * tile_n * tile_k;
    const double tile_mn   = (double)tile_m * tile_n;

    const double lds_est = ((double)tile_m * tile_k + (double)tile_n * tile_k) * bpe;
    double lds_cap       = (double)hw.lds_capacity;
    if(pipeline_str.rfind("compv4", 0) == 0) lds_cap = 32768.0;
    const double lds_ratio = lds_est / std::max(lds_cap, 1.0);

    const double btr_m  = (double)tile_m / std::max(block_size, 1);
    const double btr_n  = (double)tile_n / std::max(block_size, 1);
    const int    gmin   = std::min(tile_m, tile_n);
    const int    gmax   = std::max({tile_m, tile_n, 1});
    const double block_eff = (double)gmin / gmax;

    const double is_compv3    = (pipeline_str == "compv3") ? 1.0 : 0.0;
    const double is_compv4    = (pipeline_str == "compv4") ? 1.0 : 0.0;
    const double is_compv5    = (pipeline_str == "compv5") ? 1.0 : 0.0;
    const double is_intrawave = 1.0; /* all rocKE conv pipelines are intrawave */
    const double has_dsb      = 0.0;
    const double has_si       = 0.0;
    const double is_basic  = (pipeline_str.rfind("basic_v", 0) == 0) ? 1.0 : 0.0;
    const double is_compv6 = (pipeline_str == "compv6") ? 1.0 : 0.0;
    const double is_mem    = (pipeline_str == "mem")    ? 1.0 : 0.0;

    const double gemm_m  = (double)N * output_volume;
    const double gemm_n  = K;
    const double gemm_k  = std::floor(cpg * filter_volume);
    const double ntm     = std::ceil(gemm_m / std::max(tile_m, 1));
    const double ntn     = std::ceil(gemm_n / std::max(tile_n, 1));
    const double ntk     = std::ceil(gemm_k / std::max(tile_k, 1));
    const double tot_tiles = ntm * ntn;

    auto tile_eff = [](double d, int t) -> double {
        if(t <= 0) return 1.0;
        double r = std::fmod(d, (double)t);
        return r > 0.0 ? r / t : 1.0;
    };
    const double te_m        = tile_eff(gemm_m, tile_m);
    const double te_n        = tile_eff(gemm_n, tile_n);
    const double te_k        = tile_eff(gemm_k, tile_k);
    const double overall_eff = te_m * te_n * te_k;
    const double cu_util     = tot_tiles / std::max((double)hw.num_cus, 1.0);
    const double rm          = gemm_m / std::max(tile_m, 1);
    const double rn          = gemm_n / std::max(tile_n, 1);
    const double rk          = gemm_k / std::max(tile_k, 1);
    const double psm         = (gemm_m < tile_m) ? 1.0 : 0.0;
    const double psn         = (gemm_n < tile_n) ? 1.0 : 0.0;
    const double psk         = (gemm_k < tile_k) ? 1.0 : 0.0;

    const double log_gemm_m_n_ratio     = std::log(std::max(gemm_m, 1.0) / std::max(gemm_n, 1.0));
    const double log_total_output_tiles = std::log(std::max(tot_tiles, 1.0));
    const double log_num_tiles_m        = std::log(std::max(ntm, 1.0));
    const double log_gemm_m_raw         = std::log(std::max(gemm_m, 1.0));
    const double log_gemm_m_over_num_cus =
        std::log(std::max(gemm_m, 1.0) / std::max((double)hw.num_cus, 1.0));
    const double log_cu_fill = std::log(std::max(tot_tiles / std::max((double)hw.num_cus, 1.0), 1e-6));
    const double k_tiles_over_mn_tiles = ntk / std::max(tot_tiles, 1.0);
    const double num_waves             = std::ceil(tot_tiles / std::max((double)hw.num_cus, 1.0));
    const double wave_quant_efficiency =
        tot_tiles / std::max(num_waves * (double)hw.num_cus, 1.0);
    const double active_cus          = std::min(tot_tiles, (double)hw.num_cus);
    const double log_k_per_active_cu = std::log(std::max(ntk / std::max(active_cus, 1.0), 1e-6));
    const double is_subwave          = (tot_tiles < hw.num_cus) ? 1.0 : 0.0;

    return {
        /* Problem (30) */
        (double)N, (double)C, (double)K, (double)G, (double)Hi, (double)Wi, (double)Y, (double)X,
        (double)stride_h, (double)stride_w, (double)pad_h, (double)pad_w,
        (double)Ho, (double)Wo,
        log2_N, log2_C, log2_K, log2_G, log2_Hi, log2_Wi,
        log2_spatial, log2_filter, log2_output,
        ai, filter_area, is_1x1, is_3x3, cpg, aspect_hw, aspect_filt,
        /* 3-D-pinned (8) */
        is_3d, 1.0, 1.0, 1.0, 1.0, 0.0,
        (double)dilation_h, (double)dilation_w,
        /* Group (9) */
        log2_cpg, log2_ocpg, is_depthwise, group_density, is_small_group,
        cprod, batch_group, is_small_batch_grouped, k_per_c,
        /* Kernel (16) */
        (double)block_size, (double)tile_m, (double)tile_n, (double)tile_k,
        (double)pipeline_code, num_warps, tile_vol, tile_mn,
        lds_est, lds_ratio, btr_m, btr_n, block_eff,
        is_compv3, is_compv4, is_compv5,
        /* Suffix (6) */
        is_intrawave, has_dsb, has_si, is_basic, is_compv6, is_mem,
        /* Interaction (20) */
        gemm_m, gemm_n, gemm_k,
        ntm, ntn, ntk, tot_tiles,
        te_m, te_n, te_k, overall_eff,
        cu_util, rm, rn, rk,
        psm, psn, psk,
        log_gemm_m_n_ratio, log_total_output_tiles,
        /* Hardware (12) */
        (double)hw.num_cus, (double)hw.simds_per_cu, (double)hw.total_simds(),
        (double)hw.shader_engines, (double)hw.max_clock_mhz, (double)hw.max_waves_per_cu,
        (double)hw.wavefront_size, (double)hw.lds_capacity,
        (double)hw.l1_cache_kb, (double)hw.l2_cache_kb, (double)hw.l3_cache_kb,
        (double)hw.num_xcd,
        /* Extended interaction (8) */
        log_num_tiles_m, log_gemm_m_raw, log_gemm_m_over_num_cus,
        log_cu_fill, k_tiles_over_mn_tiles,
        wave_quant_efficiency, log_k_per_active_cu, is_subwave,
    };
}

/* -------------------------------------------------------------------------
 * Feature spec — parsed from feature_spec.json written by train.py.
 * -------------------------------------------------------------------------*/
struct ConvFeatureSpec
{
    std::vector<int> indices;
    bool log_transform = true;
};

/* Parse feature_indices and tflops_log_transform from feature_spec.json.
 * Returns defaults (empty indices, log_transform=true) when file is absent. */
inline ConvFeatureSpec load_conv_feature_spec(const std::string& spec_path)
{
    ConvFeatureSpec out;
    std::ifstream f(spec_path);
    if(!f.is_open()) return out;
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    auto fi_pos = src.find("\"feature_indices\"");
    if(fi_pos != std::string::npos)
    {
        auto arr_start = src.find('[', fi_pos);
        auto arr_end   = src.find(']', arr_start);
        if(arr_start != std::string::npos && arr_end != std::string::npos)
        {
            std::istringstream arr(src.substr(arr_start + 1, arr_end - arr_start - 1));
            std::string tok;
            while(std::getline(arr, tok, ','))
            {
                tok.erase(0, tok.find_first_not_of(" \t\r\n"));
                tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
                if(!tok.empty())
                    out.indices.push_back(std::stoi(tok));
            }
        }
    }

    auto lt_pos = src.find("\"tflops_log_transform\"");
    if(lt_pos != std::string::npos)
    {
        auto colon = src.find(':', lt_pos);
        if(colon != std::string::npos)
        {
            auto val = src.find_first_not_of(" \t\r\n", colon + 1);
            if(val != std::string::npos)
                out.log_transform = (src.substr(val, 5) != "false");
        }
    }

    return out;
}

/* -------------------------------------------------------------------------
 * ConvMLHeuristic — loads a per-arch LightGBM model and scores
 * (rocke_conv_problem_t, rocke_implicit_gemm_conv_spec_t) pairs.
 *
 * Constructed once at provider init time; predict_tflops() is thread-safe.
 * -------------------------------------------------------------------------*/
class ConvMLHeuristic
{
public:
    ConvMLHeuristic(const std::string& model_dir,
                    const std::string& arch  = "gfx950",
                    const std::string& dtype = "fp16")
        : hw_(ConvHwProfile::for_arch(arch)), dtype_(dtype)
    {
        const std::string model_path = model_dir + "/model_tflops.lgbm";
        int iters = 0;
        if(!LGBM_BoosterCreateFromModelfile ||
           LGBM_BoosterCreateFromModelfile(model_path.c_str(), &iters, &b_) != 0 || !b_)
        {
            if(!LGBM_BoosterCreateFromModelfile)
                std::cerr << "[rocKE] ConvMLHeuristic: LightGBM not available (weak symbol null)\n";
            else
                std::cerr << "[rocKE] ConvMLHeuristic: failed to load " << model_path << "\n";
            std::string gz = model_path + ".gz";
            if(std::ifstream(gz).good())
                std::cerr << "[rocKE] ConvMLHeuristic: decompress with: gunzip " << gz << "\n";
            b_ = nullptr;
            return;
        }

        const auto spec = load_conv_feature_spec(model_dir + "/feature_spec.json");
        indices_       = spec.indices;
        log_transform_ = spec.log_transform;

        if(indices_.empty())
        {
            std::cerr << "[rocKE] ConvMLHeuristic: feature_spec.json missing or has no "
                         "feature_indices — scoring disabled.\n";
            if(LGBM_BoosterFree) LGBM_BoosterFree(b_);
            b_ = nullptr;
            return;
        }

        std::cout << "[rocKE] ConvMLHeuristic: loaded " << model_path
                  << " (" << iters << " iters, " << indices_.size() << " features)\n";
    }

    ~ConvMLHeuristic() { if(b_ && LGBM_BoosterFree) LGBM_BoosterFree(b_); }

    ConvMLHeuristic(const ConvMLHeuristic&)            = delete;
    ConvMLHeuristic& operator=(const ConvMLHeuristic&) = delete;

    bool is_loaded() const { return b_ != nullptr; }

    double predict_tflops(const rocke_conv_problem_t&            prob,
                          const rocke_implicit_gemm_conv_spec_t& spec) const
    {
        if(!b_) return 0.0;
        auto all = conv_extract_features(prob, spec, hw_, dtype_);
        std::vector<double> f(indices_.size());
        for(size_t i = 0; i < indices_.size(); ++i)
        {
            const size_t idx = static_cast<size_t>(indices_[i]);
            f[i] = idx < all.size() ? all[idx] : 0.0;
        }
        int64_t ol  = 0;
        double  pred = 0.0;
        std::lock_guard<std::mutex> lock(mu_);
        if(!LGBM_BoosterPredictForMat ||
           LGBM_BoosterPredictForMat(
               b_, f.data(), 1, 1, static_cast<int>(f.size()),
               1, 0, 0, 0, "", &ol, &pred) != 0)
            return 0.0;
        return log_transform_ ? std::expm1(pred) : pred;
    }

private:
    void*         b_             = nullptr;
    ConvHwProfile hw_;
    std::string   dtype_;
    std::vector<int> indices_;
    bool          log_transform_ = true;
    mutable std::mutex mu_;
};

} // namespace rocke

#endif /* __cplusplus */
