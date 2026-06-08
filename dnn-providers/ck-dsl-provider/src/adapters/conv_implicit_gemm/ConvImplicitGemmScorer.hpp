// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ck_dsl_provider {

struct ConvSelectionProblem;
struct ConvImplicitGemmPerfKnobs;

/// Thin RAII wrapper around the LightGBM C API (booster +
/// PredictForMat), bound to the in-tree grouped-conv-forward 2D/3D
/// suffix model for bf16 / gfx950.
///
/// **Why an opaque Impl**: unlike ``SdpaScorer``, the conv side has no
/// CK-dispatcher ``conv_ml_heuristic.hpp`` to wrap (the SDPA scorer's
/// pimpl shields callers from ``fmha_ml_heuristic.hpp`` -> HIP). We
/// still hide the LightGBM handle behind an ``Impl`` so the header only
/// needs ``<memory>`` / ``<string>`` and so a future move to a shared
/// ``conv_ml_heuristic.hpp`` (if/when CK adds one) is a one-file change.
///
/// **Lifetime / cost**: constructing a scorer loads the conv LightGBM
/// model from the path baked in at configure time
/// (``CK_DSL_GROUPED_CONV_FWD_MODEL_PATH``). Construct it once and reuse
/// it; the plan builder will hold a function-local ``static const``
/// instance, mirroring the SDPA path.
///
/// **Load failure is non-fatal**: a missing / unreadable model leaves
/// ``isLoaded()`` false. ``selectPerfKnobs`` falls back to the analytic
/// policy in that case (and also for any non-bf16 dtype, since there is
/// no oracle there).
///
/// **Feature extraction**: the model expects 97 features in the exact
/// order of ``feature_engine_grouped_conv.py::get_feature_names()``.
/// The extractor is implemented inside ``ConvImplicitGemmScorer.cpp``
/// as a direct mirror of that Python file; the categorical ``pipeline``
/// feature is encoded via the same ``PIPELINE_MAP`` (compv3=0, compv4=1,
/// compv5=2, mem=3, preshufflev2=4, basic_v1=5, compv6=6; anything else
/// defaults to 0). ``predict_tflops`` calls ``LGBM_BoosterPredictForMat``
/// with ``data_type=1`` (FLOAT64 -- the matching dtype for the
/// ``std::array<double>`` feature buffer; data_type=0 would reinterpret
/// the bytes as floats and return garbage).
class ConvImplicitGemmScorer {
   public:
    /// Loads the in-tree grouped-conv-forward gfx950 / bf16 model from
    /// the path baked in at configure time
    /// (``CK_DSL_GROUPED_CONV_FWD_MODEL_PATH``, generated into
    /// ``ckdsl_provider_paths.h``).
    ConvImplicitGemmScorer();

    /// Test seam: load from an explicit path. A bogus path yields a
    /// scorer with ``isLoaded() == false`` (no throw).
    explicit ConvImplicitGemmScorer(const std::string& modelPath);

    ~ConvImplicitGemmScorer();

    ConvImplicitGemmScorer(const ConvImplicitGemmScorer&) = delete;
    ConvImplicitGemmScorer& operator=(const ConvImplicitGemmScorer&) = delete;
    ConvImplicitGemmScorer(ConvImplicitGemmScorer&&) noexcept;
    ConvImplicitGemmScorer& operator=(ConvImplicitGemmScorer&&) noexcept;

    /// True iff the LightGBM booster loaded successfully.
    [[nodiscard]] bool isLoaded() const;

    /// Predicted TFLOPS for the (problem, knobs) pair. Returns 0.0 when
    /// the model is not loaded or the booster call fails.
    ///
    /// The model targets log1p(TFLOPS), so the implementation applies
    /// ``std::expm1`` to the raw prediction (mirrors the
    /// ``log_targets`` flag in ``feature_spec.json``).
    [[nodiscard]] double predict(const ConvSelectionProblem& problem,
                                 const ConvImplicitGemmPerfKnobs& knobs) const;

    /// Test seam: exposes the 97-feature vector built for the booster.
    /// Used by the score-parity test to compare against Python's
    /// ``GroupedConvFeatureEngine.extract``. Production code never calls
    /// this -- ``predict`` consumes the same internal buffer in place,
    /// so there is no second-path drift to worry about; this just hands
    /// the buffer out for inspection.
    [[nodiscard]] std::vector<double> extractFeaturesForTest(
        const ConvSelectionProblem& problem, const ConvImplicitGemmPerfKnobs& knobs) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ck_dsl_provider
