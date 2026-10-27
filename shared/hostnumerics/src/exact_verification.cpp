// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/hostnumerics/exact_verification.hpp>
#include <roc/hostnumerics/reduction.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>

#include "detail/exact_integer.hpp"
#include "detail/runtime_tensor_io.hpp"

namespace roc::hostnumerics {
namespace {
double realScalar(const Tensor& scalar) {
    if (scalar.shape().rank() != 0 ||
        scalarTypeInfo(scalar.type()).category == ScalarCategory::Complex)
        throw std::runtime_error("Exact verification scales must be real rank-zero tensors.");
    return scalar.item<double>();
}

size_t uncertifiedValues(const Tensor& input, double limit) {
    const detail::RuntimeTensorReader<double> reader(input);
    size_t count = 0;
    input.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(input.shape().elementCount()), [&](size_t, std::span<const size_t> at) {
            const double value = reader(at);
            count += !detail::isExactInteger(value) || !(std::abs(value) < limit);
        });
    return count;
}

void compareExact(ExactVerification& result, const Tensor& observed, const Tensor& expected,
                  size_t maxReportedMismatches) {
    ComparisonOptions options;
    options.computeFrobenius = false;
    options.maxReportedMismatches = maxReportedMismatches;
    result.comparison = compare(observed, expected, options);
    result.status =
        result.comparison.passed() ? VerificationStatus::Passed : VerificationStatus::Mismatch;
}
}  // namespace

ExactVerification verifyExactActivation(const Tensor& observed, const Tensor& auxiliary,
                                        const Tensor& auxiliaryScale, const Tensor& outputScale,
                                        const ActivationFunction& activation,
                                        size_t maxReportedMismatches) {
    if (observed.shape() != auxiliary.shape())
        throw std::runtime_error("Exact activation tensor shapes must match.");
    ExactVerification result;
    const double scaleE = realScalar(auxiliaryScale), scaleD = realScalar(outputScale);
    if (!std::isfinite(scaleE) || !std::isfinite(scaleD) || scaleE == 0) {
        result.reason =
            "An auxiliary scaled by 0 or nonfinite scales cannot reconstruct activation.";
        return result;
    }
    if (detail::exactIntegerLimit(auxiliary.type()) == 0 ||
        detail::exactIntegerLimit(observed.type()) == 0 ||
        !(std::holds_alternative<IdentityActivation>(activation) ||
          std::holds_alternative<ReluActivation>(activation) ||
          std::holds_alternative<ClampActivation>(activation))) {
        result.reason =
            "Exact activation requires supported real types and identity, ReLU or clamp.";
        return result;
    }
    if (const auto* clamp = std::get_if<ClampActivation>(&activation))
        if (!std::isfinite(clamp->minimum) || !std::isfinite(clamp->maximum) ||
            clamp->minimum > clamp->maximum) {
            result.reason = "Exact activation requires finite ordered clamp bounds.";
            return result;
        }
    result.uncertifiedElements =
        uncertifiedValues(auxiliary, detail::exactObservationLimit(auxiliary.type()));
    if (result.uncertifiedElements) {
        result.reason = "Auxiliary values are outside the certified exact-integer range.";
        return result;
    }
    Tensor pre(ScalarType::Float64, auxiliary.shape());
    const detail::RuntimeTensorReader<double> auxiliaryReader(auxiliary);
    const detail::RuntimeTensorWriter<double> preWriter(pre);
    bool finite = true;
    auxiliary.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(auxiliary.shape().elementCount()), [&](size_t, std::span<const size_t> at) {
            // Preserve division, including F64 rounding. Multiplication by a
            // rounded reciprocal is not equivalent.
            const double value = auxiliaryReader(at) / scaleE;
            finite &= std::isfinite(value);
            preWriter.store(at, value);
        });
    if (!finite) {
        result.reason = "Auxiliary scale reconstruction overflows Float64.";
        return result;
    }
    const auto activated = activate(pre, activation);
    result.maximumAbsolute =
        referenceMaximumAbsolute(activated, ScalarType::Float64, ScalarType::Float64);
    const auto scaled = multiply(activated, Tensor(scaleD));
    const auto expected = scaled.copyConvertedTo(observed.type(), detail::exactOutputConversion());
    compareExact(result, observed, expected, maxReportedMismatches);
    return result;
}

ExactVerification verifyExactSum(const Tensor& observed, const Tensor& input,
                                 std::vector<size_t> axes, ScalarType computeType,
                                 size_t maxReportedMismatches) {
    ExactVerification result;
    const double limit = detail::exactIntegerLimit(computeType);
    if (limit == 0 || detail::exactIntegerLimit(input.type()) == 0 ||
        detail::exactIntegerLimit(observed.type()) == 0) {
        result.reason = "Exact reduction requires supported real input, output and compute types.";
        return result;
    }
    result.uncertifiedElements = uncertifiedValues(input, limit);
    if (result.uncertifiedElements) {
        result.reason = "Exact reduction requires integral inputs inside the compute type's range.";
        return result;
    }
    const auto wide = input.copyConvertedTo(ScalarType::Float64);
    const auto bounds =
        referenceSum(wide.absolute(), axes, ScalarType::Float64, ScalarType::Float64);
    if (observed.shape() != bounds.shape())
        throw std::runtime_error(
            "Exact reduction observed shape does not match the reduced shape.");
    const detail::RuntimeTensorReader<double> boundsReader(bounds);
    bool exact = true;
    bounds.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(bounds.shape().elementCount()),
        [&](size_t, std::span<const size_t> at) { exact &= boundsReader(at) < limit; });
    if (!exact) {
        result.reason = "Exact reduction absolute-sum bound reaches the compute type's limit.";
        return result;
    }
    // The certificate makes every summation order exact in the requested
    // compute type. Float64 also represents every certified intermediate.
    const auto sum = referenceSum(wide, axes, ScalarType::Float64, ScalarType::Float64);
    compareExact(result, observed,
                 sum.copyConvertedTo(observed.type(), detail::exactOutputConversion()),
                 maxReportedMismatches);
    return result;
}

std::variant<Tensor, VerificationFailure> recoverExactMaximumAbsolute(const Tensor& verifiedOutput,
                                                                      const Tensor& outputScale) {
    const double scale = realScalar(outputScale);
    const double limit = detail::exactObservationLimit(verifiedOutput.type());
    if (limit == 0 || !std::isfinite(scale) || scale == 0 ||
        uncertifiedValues(verifiedOutput, limit))
        return VerificationFailure{VerificationStatus::Unsupported,
                                   "Maximum reconstruction requires exact finite integer output "
                                   "and a nonzero finite scale."};
    const auto top =
        referenceMaximumAbsolute(verifiedOutput, ScalarType::Float64, ScalarType::Float64);
    const double recovered = top.item<double>() / std::abs(scale);
    if (!std::isfinite(recovered))
        return VerificationFailure{VerificationStatus::Unsupported,
                                   "Maximum scale reconstruction overflows Float64."};
    return Tensor(recovered);
}
}  // namespace roc::hostnumerics
