// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/gemm.hpp>

namespace roc::host_validation {
class TiledGemmBackend final : public GemmBackendImplementation {
   public:
    GemmBackend backend() const override;
    GemmSupportInfo querySupport(const GemmRequest& request) const override;
    GemmRunInfo run(const GemmRequest& request) const override;
};
}  // namespace roc::host_validation
