// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/verification.hpp>
#include <variant>

namespace roc::hostnumerics {
struct ExactVerification {
    VerificationStatus status = VerificationStatus::Unsupported;
    std::string reason;
    size_t uncertifiedElements = 0;
    ComparisonReport comparison;
    // For activation checks, max(abs(activation(auxiliary / auxiliaryScale))),
    // before output scaling. Absent when the auxiliary values are uncertified.
    std::optional<Tensor> maximumAbsolute;
};

// Check outputScale * activation(auxiliary / auxiliaryScale), with Float64
// reconstruction and nearest-even/saturating output conversion. Auxiliary
// values must already have been verified under an exact-integer contract;
// this check additionally rejects fractional/nonfinite values and values that
// could have been rounded or saturated. Identity, ReLU and finite clamp are
// supported. Scales are real rank-zero tensors; auxiliaryScale must be nonzero.
// Supports any rank and layout. Shape mismatches throw.
ExactVerification verifyExactActivation(const Tensor& observed, const Tensor& auxiliary,
                                        const Tensor& auxiliaryScale, const Tensor& outputScale,
                                        const ActivationFunction& activation = IdentityActivation{},
                                        size_t maxReportedMismatches = 10);

// Certify integral inputs and a strict absolute-sum bound below the compute
// type's exact-integer limit, then check the sum with nearest-even/saturating
// output conversion. Axes have referenceSum semantics. A zero other GEMM
// operand cannot hide an inexact reduction. Shape/axis errors throw.
ExactVerification verifyExactSum(const Tensor& observed, const Tensor& input,
                                 std::vector<size_t> axes, ScalarType computeType,
                                 size_t maxReportedMismatches = 10);

// Recover max(abs(verifiedOutput)) / abs(outputScale) from an already verified
// exact-integer result. Refuses rounded/saturated or nonfinite values and a
// zero/nonfinite scale. Success is a rank-zero Float64 Tensor; an empty output
// has maximum zero. This operation does not itself verify the original GEMM.
std::variant<Tensor, VerificationFailure> recoverExactMaximumAbsolute(const Tensor& verifiedOutput,
                                                                      const Tensor& outputScale);
}  // namespace roc::hostnumerics
