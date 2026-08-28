// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>

namespace roc::host_numerics {
// Configures reference execution and comparison for one observed GEMM output.
// The comparison selection is also the exact logical work requested from the
// reference implementation.
struct GemmValidationOptions {
    ComparisonOptions comparison;
    GemmBackend backend = GemmBackend::Automatic;
};

struct GemmValidationResult {
    GemmRunInfo reference;
    ComparisonResult comparison;
};

// Computes and compares the requested reference output. Partial selections are
// streamed into compact temporary storage rather than materializing complete D.
// GPU readback, tolerance selection, and result presentation remain caller-owned.
GemmValidationResult validateGemm(const GemmProblem& problem, const Tensor& observed,
                                  const GemmValidationOptions& options = {});
}  // namespace roc::host_numerics
