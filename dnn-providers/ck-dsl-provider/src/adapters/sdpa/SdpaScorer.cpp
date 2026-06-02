// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// HIP-compiled TU. fmha_ml_heuristic.hpp transitively pulls in
// ck_tile/host/kernel_launch.hpp (HIP), so this file is marked
// LANGUAGE HIP in src/CMakeLists.txt. Keep all FmhaMLHeuristic usage
// confined here; the header (SdpaScorer.hpp) stays HIP-free via pimpl.

#include "SdpaScorer.hpp"

#include <ck_tile/dispatcher/fmha_ml_heuristic.hpp>
#include <string>
#include <utility>

#include "ckdsl_provider_paths.h"

namespace ck_dsl_provider {

/// Opaque body: owns the LightGBM-backed heuristic. The registry is
/// always null here -- this provider scores a single self-built
/// candidate key per call (``predict_tflops``), and that path never
/// dereferences the registry. The registry is only used by the
/// heuristic's ``operator()`` ranking helper, which we do not call.
struct SdpaScorer::Impl {
    explicit Impl(const std::string& modelPath) : heuristic(modelPath, /*registry=*/nullptr) {}

    ck_tile::dispatcher::FmhaMLHeuristic heuristic;
};

SdpaScorer::SdpaScorer() : SdpaScorer(std::string(kCkDslFmhaFwdModelPath)) {}

SdpaScorer::SdpaScorer(const std::string& modelPath) : impl_(std::make_unique<Impl>(modelPath)) {}

SdpaScorer::~SdpaScorer() = default;

SdpaScorer::SdpaScorer(SdpaScorer&&) noexcept = default;
SdpaScorer& SdpaScorer::operator=(SdpaScorer&&) noexcept = default;

bool SdpaScorer::isLoaded() const {
    return impl_->heuristic.is_loaded();
}

double SdpaScorer::predict(const ck_tile::dispatcher::FmhaProblem& problem,
                           const ck_tile::dispatcher::FmhaKernelKey& key) const {
    return impl_->heuristic.predict_tflops(problem, key);
}

}  // namespace ck_dsl_provider
