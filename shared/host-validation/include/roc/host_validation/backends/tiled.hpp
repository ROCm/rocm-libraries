// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/validation.hpp>

namespace roc::host_validation {
class TiledGemmBackend final : public GemmBackendImplementation {
   public:
    GemmBackend backend() const override;
    GemmSupportInfo querySupport(const GemmProblem& problem) const override;
    GemmRunInfo run(const GemmProblem& problem) const override;
};
}  // namespace roc::host_validation
