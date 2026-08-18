// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/gemm.hpp>

namespace roc::host_validation {
// Executes the subset of dense GEMM requests that CBLAS can consume directly.
// A, B, C, D, and the accumulator must already use one matching BLAS scalar
// type, and their views must satisfy the direct-layout and aliasing restrictions.
class BlasGemmBackend final : public GemmBackendImplementation {
   public:
    GemmBackend backend() const override;
    GemmSupportInfo querySupport(const GemmRequest& request) const override;
    GemmRunInfo run(const GemmRequest& request) const override;
};

// Extends the BLAS path to runtime-typed, scaled, or compute-quantized operands
// by materializing accumulator-typed scratch matrices. It delegates the dense
// multiply to BlasGemmBackend, then applies the supported output
// scaling/conversion while writing the caller-owned output tensor.
class TransformingBlasGemmBackend final : public GemmBackendImplementation {
   public:
    GemmBackend backend() const override;
    GemmSupportInfo querySupport(const GemmRequest& request) const override;
    GemmRunInfo run(const GemmRequest& request) const override;
};
}  // namespace roc::host_validation
