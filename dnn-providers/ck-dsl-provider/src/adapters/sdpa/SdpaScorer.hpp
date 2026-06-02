// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ck_tile/dispatcher/fmha_kernel_key.hpp>
#include <ck_tile/dispatcher/fmha_problem.hpp>
#include <memory>
#include <string>

namespace ck_dsl_provider {

/// Thin RAII wrapper around the dispatcher's ``FmhaMLHeuristic``
/// (LightGBM-backed TFLOPS predictor for FMHA-forward kernels).
///
/// **Why a pimpl**: ``ck_tile/dispatcher/fmha_ml_heuristic.hpp``
/// transitively includes ``ck_tile/host/kernel_launch.hpp``, which is
/// HIP. Any TU that includes that header must be HIP-compiled. The plan
/// builder (``SdpaFwdPlanBuilder.cpp``) is a plain CXX TU and must stay
/// that way, so the heuristic is hidden behind an opaque ``Impl`` whose
/// definition lives only in the HIP-compiled ``SdpaScorer.cpp``. This
/// header therefore includes ONLY the HIP-free dispatcher key/problem
/// types plus ``<memory>`` / ``<string>``.
///
/// **Lifetime / cost**: constructing a scorer loads the ~11 MB gfx950
/// model via ``LGBM_BoosterCreateFromModelfile``. Construct it once and
/// reuse it (the plan builder holds a function-local ``static const``
/// instance). The booster is owned by the wrapped ``FmhaMLHeuristic``
/// and freed in its destructor; ``Impl`` is owned by ``unique_ptr`` here
/// -- no manual ``delete`` anywhere.
///
/// **Load failure is non-fatal**: a missing / unreadable model leaves
/// ``isLoaded()`` false (the underlying heuristic logs to stderr and
/// sets its booster to null without throwing). Callers fall back to the
/// analytic policy in that case (see ``selectPerfKnobs``).
class SdpaScorer {
   public:
    /// Loads the in-tree gfx950 FMHA-forward model from the path baked
    /// in at configure time (``CK_DSL_FMHA_FWD_MODEL_PATH``, generated
    /// into ``ckdsl_provider_paths.h``).
    SdpaScorer();

    /// Test seam: load from an explicit path. A bogus path yields a
    /// scorer with ``isLoaded() == false`` (no throw).
    explicit SdpaScorer(const std::string& modelPath);

    ~SdpaScorer();

    SdpaScorer(const SdpaScorer&) = delete;
    SdpaScorer& operator=(const SdpaScorer&) = delete;
    SdpaScorer(SdpaScorer&&) noexcept;
    SdpaScorer& operator=(SdpaScorer&&) noexcept;

    /// True iff the LightGBM booster loaded successfully.
    [[nodiscard]] bool isLoaded() const;

    /// Predicted TFLOPS for the (problem, kernel-key) pair. Returns 0.0
    /// when the model is not loaded (the heuristic short-circuits on a
    /// null booster). The wrapped heuristic does NOT dereference its
    /// registry on this path, so a null registry is fine.
    [[nodiscard]] double predict(const ck_tile::dispatcher::FmhaProblem& problem,
                                 const ck_tile::dispatcher::FmhaKernelKey& key) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ck_dsl_provider
