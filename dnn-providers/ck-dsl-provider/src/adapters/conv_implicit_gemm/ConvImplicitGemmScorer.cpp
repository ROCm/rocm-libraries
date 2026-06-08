// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Plain CXX TU (no HIP includes). The LightGBM C API is declared
// extern "C" locally so we do not pull in any CK-dispatcher header.

#include "ConvImplicitGemmScorer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "ConvImplicitGemmCandidateSelector.hpp"
#include "ConvImplicitGemmPerfKnobs.hpp"
#include "ckdsl_provider_paths.h"

extern "C" {
int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
int LGBM_BoosterPredictForMat(
    void*, const void*, int, int, int, int, int, int, int, const char*, std::int64_t*, double*);
int LGBM_BoosterFree(void*);
}

namespace ck_dsl_provider {

namespace {

// Feature count MUST match feature_spec.json for
// grouped_conv_forward_2d3d_suffix_bf16_gfx950 (97 features).
constexpr int kNumConvFeatures = 97;

// gfx950 / MI300 hardware defaults -- mirror of GroupedConvFeatureEngine
// __init__ in feature_engine_grouped_conv.py. The model was trained with
// these constants in the hw_* columns, so we ship the same values.
struct ConvHardwareProfile {
    int num_cus = 256;
    int simds_per_cu = 4;
    int shader_engines = 32;
    int max_clock_mhz = 2400;
    int max_waves_per_cu = 32;
    int wavefront_size = 64;
    int lds_capacity = 65536;
    int l1_cache_kb = 32;
    int l2_cache_kb = 4096;
    int l3_cache_kb = 262144;
    int num_xcd = 8;
    int total_simds() const {
        return num_cus * simds_per_cu;
    }
};

// PIPELINE_MAP from feature_engine.py:30. The model treats `pipeline` as
// a categorical feature, so the encoded integer maps onto the trained
// vocabulary. Unknown names default to 0 (== compv3) -- the same
// default the Python side uses; comp_async / basic_async_v1 fall here.
int encodePipeline(const std::string& p) {
    if (p == "compv3") return 0;
    if (p == "compv4") return 1;
    if (p == "compv5") return 2;
    if (p == "mem") return 3;
    if (p == "preshufflev2") return 4;
    if (p == "basic_v1") return 5;
    if (p == "compv6") return 6;
    return 0;
}

double dtypeBytes(const std::string& dtype) {
    // DTYPE_BYTES from feature_engine.py; bf16 == 2, fp16 == 2, fp32 == 4.
    if (dtype == "fp32" || dtype == "f32") return 4.0;
    if (dtype == "fp8" || dtype == "bf8" || dtype == "int8") return 1.0;
    return 2.0;  // bf16 / fp16 default
}

// Pin-version sentinel for the 2D-only contract. ConvSelectionProblem
// carries no depth fields today; this constant is the single place that
// declares the extractor is built for ``kConvSelectionDim == 2``. When
// a future change widens ConvSelectionProblem with D/Z/Do/stride_d/pad_d
// fields, bumping this constant to 3 is the same change that grep'd
// guardrails in ConvSelectionProblem doc, the extractor body, and any
// integration tests will surface. Keeping it as a named compile-time
// constant (rather than a magic ``1`` in the body) means an extender
// hits the symbol immediately when extending the struct.
constexpr int kConvSelectionDim = 2;

// Direct C++ mirror of GroupedConvFeatureEngine.extract() for the 2D
// conv case (ck_dsl conv-fwd is 2D-only today, so is_3d / Di / Z / Do /
// stride_d / pad_d are pinned to their 2D values per
// ``kConvSelectionDim``). The feature ORDER matches feature_spec.json
// exactly; any reordering breaks the trained booster.
std::array<double, kNumConvFeatures> extractConvFeatures(const ConvSelectionProblem& p,
                                                          const ConvImplicitGemmPerfKnobs& k,
                                                          const ConvHardwareProfile& hw) {
    // 2D-only contract guard. The trained feature schema's 3D slots
    // (is_3d / Di / Z / Do / stride_d / pad_d) are pinned to their 2D
    // values below. Tying the function to kConvSelectionDim makes any
    // future 3D widening a load-bearing change: this static_assert
    // fires until the extractor body, ConvSelectionProblem, and the
    // model retraining all agree on the new dim.
    static_assert(kConvSelectionDim == 2,
                  "extractConvFeatures pins the 3D feature slots and the "
                  "depth-axis kernel features; widening ConvSelectionProblem "
                  "to carry depth fields requires updating those pins (and "
                  "retraining the booster against a 3D-aware feature spec) "
                  "before bumping kConvSelectionDim");

    // --- 2D problem (3D dims pinned per kConvSelectionDim) ---
    const int N = p.N;
    const int C = p.C;
    const int K = p.K;
    const int G = p.G > 0 ? p.G : 1;
    const int Hi = p.Hi;
    const int Wi = p.Wi;
    const int Y = p.R;
    const int X = p.S;
    const int stride_h = p.sH;
    const int stride_w = p.sW;
    const int pad_h = p.pH;
    const int pad_w = p.pW;
    const int dilation_h = p.dH;
    const int dilation_w = p.dW;
    // dilation_d unused (no feature exports it).

    // 2D-pinned 3D-feature inputs. Kept as named constants (rather than
    // inlined ``1`` literals) so a 3D widening explicitly removes the
    // pin and the diff shows it. The Python feature engine reads these
    // as the depth axis of the conv; bf16/gfx950 training data has them
    // pinned to these values for 2D conv records.
    constexpr int Di = 1;
    constexpr int Z = 1;
    constexpr int stride_d = 1;
    constexpr int pad_d = 0;

    const double is_3d = (Di > 1 || Z > 1 || pad_d > 0) ? 1.0 : 0.0;

    const int effY = (Y - 1) * dilation_h + 1;
    const int effX = (X - 1) * dilation_w + 1;
    const int Ho = (Hi + 2 * pad_h - effY) / stride_h + 1;
    const int Wo = (Wi + 2 * pad_w - effX) / stride_w + 1;
    constexpr int Do = 1;  // 2D pin (see kConvSelectionDim)

    auto log2_max1 = [](double v) { return std::log2(std::max(v, 1.0)); };
    const double log2_N = log2_max1(N);
    const double log2_C = log2_max1(C);
    const double log2_K = log2_max1(K);
    const double log2_G = log2_max1(G);
    const double log2_Hi = log2_max1(Hi);
    const double log2_Wi = log2_max1(Wi);

    const double spatial_volume = static_cast<double>(Hi) * Wi;  // 2D path
    const double filter_volume = static_cast<double>(Y) * X;     // 2D path
    const double output_volume = static_cast<double>(Ho) * Wo;   // 2D path
    const double log2_spatial = log2_max1(spatial_volume);
    const double log2_filter = log2_max1(filter_volume);
    const double log2_output = log2_max1(output_volume);

    const double bpe = dtypeBytes(p.dtype);

    // FLOPs and AI -- mirror of the Python expression. (C/G) is float
    // there too.
    const double flops = static_cast<double>(N) * K * output_volume * (static_cast<double>(C) / G) *
                         filter_volume * 2.0;
    const double input_bytes = static_cast<double>(N) * C * spatial_volume * bpe;
    const double filter_bytes = static_cast<double>(K) * (static_cast<double>(C) / G) *
                                filter_volume * bpe;
    const double output_bytes = static_cast<double>(N) * K * output_volume * bpe;
    const double bytes_transferred = input_bytes + filter_bytes + output_bytes;
    const double ai = flops / std::max(bytes_transferred, 1.0);

    const double filter_area = filter_volume;
    const double is_1x1_conv = (Y == 1 && X == 1 && Z == 1) ? 1.0 : 0.0;
    const double is_3x3_conv = (Y == 3 && X == 3) ? 1.0 : 0.0;  // 2D form
    const double channels_per_group = static_cast<double>(C) / G;
    const double aspect_ratio_hw = static_cast<double>(Hi) / std::max(Wi, 1);
    const double aspect_ratio_filter = static_cast<double>(Y) / std::max(X, 1);

    // --- Group-specific features ---
    const double output_channels_per_group = static_cast<double>(K) / G;
    const double log2_channels_per_group = log2_max1(channels_per_group);
    const double log2_output_channels_per_group = log2_max1(output_channels_per_group);
    const double is_depthwise = (G == C && G == K) ? 1.0 : 0.0;
    const double group_density = static_cast<double>(G) / std::max(C, 1);
    const double is_small_group =
        (channels_per_group < 16.0 || output_channels_per_group < 16.0) ? 1.0 : 0.0;
    const double channels_product_per_group = channels_per_group * output_channels_per_group;
    const double batch_group_product = static_cast<double>(N) * G;
    const double is_small_batch_grouped = (N < 8 && G > 1) ? 1.0 : 0.0;

    // --- Kernel features ---
    const int block_size = k.block_size();           // warp_m * warp_n * wave_size
    const int gemm_m_per_block = k.tile_m;
    const int gemm_n_per_block = k.tile_n;
    const std::string& pipeline_str = k.pipeline;
    const int pipeline_code = encodePipeline(pipeline_str);
    const double num_warps = block_size / 4.0;       // feature-engine convention

    const double tile_volume = static_cast<double>(gemm_m_per_block) * gemm_n_per_block * block_size;
    const double tile_mn = static_cast<double>(gemm_m_per_block) * gemm_n_per_block;

    const double lds_est =
        (static_cast<double>(gemm_m_per_block) * block_size + static_cast<double>(gemm_n_per_block) *
                                                                  block_size) *
        bpe;
    double lds_cap = static_cast<double>(hw.lds_capacity);
    if (pipeline_str.rfind("compv4", 0) == 0) {
        // Python: pipeline_str.startswith("compv4")
        lds_cap = 32768.0;
    }
    const double lds_ratio = lds_est / std::max(lds_cap, 1.0);

    const double block_tile_ratio_m = static_cast<double>(gemm_m_per_block) / std::max(block_size, 1);
    const double block_tile_ratio_n = static_cast<double>(gemm_n_per_block) / std::max(block_size, 1);
    const int gm = std::min(gemm_m_per_block, gemm_n_per_block);
    const int gM = std::max({gemm_m_per_block, gemm_n_per_block, 1});
    const double block_efficiency = static_cast<double>(gm) / gM;

    const double is_compv3 = (pipeline_str == "compv3") ? 1.0 : 0.0;
    const double is_compv4 = (pipeline_str == "compv4") ? 1.0 : 0.0;
    const double is_compv5 = (pipeline_str == "compv5") ? 1.0 : 0.0;

    // --- Suffix-aware features (Phase 1 pins these) ---
    // ck_dsl has no wave_mode / has_dsb / has_si knobs. Pin to the
    // Python default ("intrawave", 0, 0) -- matches the dominant cell of
    // the trained distribution.
    const double is_intrawave = 1.0;
    const double has_dsb = 0.0;
    const double has_si = 0.0;
    const double is_basic = (pipeline_str.rfind("basic_v", 0) == 0) ? 1.0 : 0.0;
    const double is_compv6 = (pipeline_str == "compv6") ? 1.0 : 0.0;
    const double is_mem = (pipeline_str == "mem") ? 1.0 : 0.0;

    // --- Interaction features ---
    const double gemm_m = static_cast<double>(N) * output_volume;
    const double gemm_n = K;
    const double gemm_k = std::floor(channels_per_group * filter_volume);

    const double num_tiles_m = std::ceil(gemm_m / std::max(gemm_m_per_block, 1));
    const double num_tiles_n = std::ceil(gemm_n / std::max(gemm_n_per_block, 1));
    const double num_tiles_k = std::ceil(gemm_k / std::max(block_size, 1));
    const double total_output_tiles = num_tiles_m * num_tiles_n;

    auto tile_eff = [](double dim, int tile) {
        if (tile <= 0) return 1.0;
        const double rem = std::fmod(dim, static_cast<double>(tile));
        return rem > 0.0 ? rem / tile : 1.0;
    };
    const double tile_eff_m = tile_eff(gemm_m, gemm_m_per_block);
    const double tile_eff_n = tile_eff(gemm_n, gemm_n_per_block);
    const double tile_eff_k = tile_eff(gemm_k, block_size);
    const double overall_eff = tile_eff_m * tile_eff_n * tile_eff_k;

    const double cu_util = total_output_tiles / std::max(hw.num_cus, 1);

    const double ratio_gemm_m_to_tile_m = gemm_m / std::max(gemm_m_per_block, 1);
    const double ratio_gemm_n_to_tile_n = gemm_n / std::max(gemm_n_per_block, 1);
    const double ratio_gemm_k_to_tile_k = gemm_k / std::max(block_size, 1);

    const double problem_smaller_than_tile_m = (gemm_m < gemm_m_per_block) ? 1.0 : 0.0;
    const double problem_smaller_than_tile_n = (gemm_n < gemm_n_per_block) ? 1.0 : 0.0;
    const double problem_smaller_than_tile_k = (gemm_k < block_size) ? 1.0 : 0.0;

    return {{
        // --- Problem features (30) ---
        static_cast<double>(N),
        static_cast<double>(C),
        static_cast<double>(K),
        static_cast<double>(G),
        static_cast<double>(Hi),
        static_cast<double>(Wi),
        static_cast<double>(Y),
        static_cast<double>(X),
        static_cast<double>(stride_h),
        static_cast<double>(stride_w),
        static_cast<double>(pad_h),
        static_cast<double>(pad_w),
        static_cast<double>(Ho),
        static_cast<double>(Wo),
        log2_N,
        log2_C,
        log2_K,
        log2_G,
        log2_Hi,
        log2_Wi,
        log2_spatial,
        log2_filter,
        log2_output,
        ai,
        filter_area,
        is_1x1_conv,
        is_3x3_conv,
        channels_per_group,
        aspect_ratio_hw,
        aspect_ratio_filter,
        // --- 3D-specific (8) -- pinned for 2D path ---
        is_3d,
        static_cast<double>(Di),
        static_cast<double>(Z),
        static_cast<double>(Do),
        static_cast<double>(stride_d),
        static_cast<double>(pad_d),
        static_cast<double>(dilation_h),
        static_cast<double>(dilation_w),
        // --- Group-specific (8) ---
        log2_channels_per_group,
        log2_output_channels_per_group,
        is_depthwise,
        group_density,
        is_small_group,
        channels_product_per_group,
        batch_group_product,
        is_small_batch_grouped,
        // --- Kernel features (15) ---
        static_cast<double>(block_size),
        static_cast<double>(gemm_m_per_block),
        static_cast<double>(gemm_n_per_block),
        static_cast<double>(pipeline_code),
        num_warps,
        tile_volume,
        tile_mn,
        lds_est,
        lds_ratio,
        block_tile_ratio_m,
        block_tile_ratio_n,
        block_efficiency,
        is_compv3,
        is_compv4,
        is_compv5,
        // --- Suffix-aware (6) ---
        is_intrawave,
        has_dsb,
        has_si,
        is_basic,
        is_compv6,
        is_mem,
        // --- Interaction (18) ---
        gemm_m,
        gemm_n,
        gemm_k,
        num_tiles_m,
        num_tiles_n,
        num_tiles_k,
        total_output_tiles,
        tile_eff_m,
        tile_eff_n,
        tile_eff_k,
        overall_eff,
        cu_util,
        ratio_gemm_m_to_tile_m,
        ratio_gemm_n_to_tile_n,
        ratio_gemm_k_to_tile_k,
        problem_smaller_than_tile_m,
        problem_smaller_than_tile_n,
        problem_smaller_than_tile_k,
        // --- Hardware (12) ---
        static_cast<double>(hw.num_cus),
        static_cast<double>(hw.simds_per_cu),
        static_cast<double>(hw.total_simds()),
        static_cast<double>(hw.shader_engines),
        static_cast<double>(hw.max_clock_mhz),
        static_cast<double>(hw.max_waves_per_cu),
        static_cast<double>(hw.wavefront_size),
        static_cast<double>(hw.lds_capacity),
        static_cast<double>(hw.l1_cache_kb),
        static_cast<double>(hw.l2_cache_kb),
        static_cast<double>(hw.l3_cache_kb),
        static_cast<double>(hw.num_xcd),
    }};
}

}  // namespace

/// Opaque body: owns the LightGBM booster + the hardware profile and a
/// flag for whether the model declares log-transformed targets (the
/// grouped-conv forward 2D/3D suffix model does: log_targets = ["tflops"]
/// in feature_spec.json).
struct ConvImplicitGemmScorer::Impl {
    explicit Impl(const std::string& modelPath) {
        int iters = 0;
        if (LGBM_BoosterCreateFromModelfile(modelPath.c_str(), &iters, &booster) != 0 || !booster) {
            std::cerr << "ConvImplicitGemmScorer: failed to load " << modelPath << std::endl;
            // Mirror MLHeuristic's hint about the .gz fallback -- the
            // conv models ship gzipped in-tree; configure-time
            // decompression is expected.
            const std::string gz = modelPath + ".gz";
            std::ifstream check(gz);
            if (check.good()) {
                std::cerr << "ConvImplicitGemmScorer: found compressed model at " << gz
                          << "; decompress with `gunzip " << gz << "` for the scorer to load"
                          << std::endl;
            }
            booster = nullptr;
        }
    }

    ~Impl() {
        if (booster) {
            LGBM_BoosterFree(booster);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void* booster{nullptr};
    ConvHardwareProfile hw{};
    // grouped_conv_forward_2d3d_suffix_bf16_gfx950/feature_spec.json
    // declares log_targets = ["tflops"], so the raw booster output is
    // log1p(tflops). expm1 inverts that.
    bool log_transform{true};
};

ConvImplicitGemmScorer::ConvImplicitGemmScorer()
    : ConvImplicitGemmScorer(std::string(kCkDslGroupedConvFwdModelPath)) {}

ConvImplicitGemmScorer::ConvImplicitGemmScorer(const std::string& modelPath)
    : impl_(std::make_unique<Impl>(modelPath)) {}

ConvImplicitGemmScorer::~ConvImplicitGemmScorer() = default;

ConvImplicitGemmScorer::ConvImplicitGemmScorer(ConvImplicitGemmScorer&&) noexcept = default;
ConvImplicitGemmScorer& ConvImplicitGemmScorer::operator=(ConvImplicitGemmScorer&&) noexcept =
    default;

bool ConvImplicitGemmScorer::isLoaded() const {
    return impl_->booster != nullptr;
}

std::vector<double> ConvImplicitGemmScorer::extractFeaturesForTest(
    const ConvSelectionProblem& problem, const ConvImplicitGemmPerfKnobs& knobs) const {
    const auto features = extractConvFeatures(problem, knobs, impl_->hw);
    return std::vector<double>(features.begin(), features.end());
}

double ConvImplicitGemmScorer::predict(const ConvSelectionProblem& problem,
                                       const ConvImplicitGemmPerfKnobs& knobs) const {
    if (!impl_->booster) {
        return 0.0;
    }
    const auto features = extractConvFeatures(problem, knobs, impl_->hw);
    std::int64_t outLen = 0;
    double raw = 0.0;
    // data_type=1 == C_API_DTYPE_FLOAT64 (matches the std::array<double>
    // backing buffer). Passing 0 (FLOAT32) reinterprets the double bytes
    // as floats and returns near-constant garbage; this fix is mirrored
    // from the SDPA scorer.
    if (LGBM_BoosterPredictForMat(impl_->booster, features.data(), /*data_type=*/1,
                                  /*nrow=*/1, /*ncol=*/kNumConvFeatures,
                                  /*is_row_major=*/1, /*predict_type=*/0, /*start_iteration=*/0,
                                  /*num_iteration=*/0, /*parameter=*/"", &outLen, &raw) != 0) {
        return 0.0;
    }
    return impl_->log_transform ? std::expm1(raw) : raw;
}

}  // namespace ck_dsl_provider
