// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/gemm.hpp>

namespace roc::host_validation {
// Reuses fixed-size A, B, and output blocks. For a partial output selection it
// accumulates every output position in each touched block and writes only the
// selected D coordinates.
class BlockedGemmBackend final : public GemmBackendImplementation {
   public:
    GemmBackend backend() const override;
    GemmSupportInfo querySupport(const GemmRequest& request) const override;
    GemmRunInfo run(const GemmRequest& request) const override;
};
}  // namespace roc::host_validation
