// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <roc/host_validation/comparison.hpp>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_validation::python_bindings {
void registerComparisonBindings(nb::module_& module) {
    nb::enum_<UlpComparisonMode>(module, "UlpComparisonMode")
        .value("RelativeSpacing", UlpComparisonMode::RelativeSpacing)
        .value("EncodedDistance", UlpComparisonMode::EncodedDistance);

    nb::enum_<ComplexPointwiseMode>(module, "ComplexPointwiseMode")
        .value("Componentwise", ComplexPointwiseMode::Componentwise)
        .value("Magnitude", ComplexPointwiseMode::Magnitude);

    nb::class_<ComparisonSelection>(module, "ComparisonSelection")
        .def(nb::init<>())
        .def_rw("first", &ComparisonSelection::first)
        .def_rw("stride", &ComparisonSelection::stride)
        .def_rw("max_elements", &ComparisonSelection::maxElements)
        .def_rw("index_order", &ComparisonSelection::indexOrder);

    nb::class_<ComparisonOptions>(module, "ComparisonOptions")
        .def(nb::init<>())
        .def_rw("pointwise", &ComparisonOptions::pointwise)
        .def_rw("absolute_tolerance", &ComparisonOptions::absoluteTolerance)
        .def_rw("relative_tolerance", &ComparisonOptions::relativeTolerance)
        .def_rw("symmetric_relative_tolerance", &ComparisonOptions::symmetricRelativeTolerance)
        .def_rw("strict_tolerance", &ComparisonOptions::strictTolerance)
        .def_rw("equal_nans", &ComparisonOptions::equalNaNs)
        .def_rw("equal_signed_zero", &ComparisonOptions::equalSignedZero)
        .def_rw("complex_pointwise_mode", &ComparisonOptions::complexPointwiseMode)
        .def_rw("compute_pointwise_statistics", &ComparisonOptions::computePointwiseStatistics)
        .def_rw("compute_frobenius", &ComparisonOptions::computeFrobenius)
        .def_rw("compute_ulp", &ComparisonOptions::computeUlp)
        .def_rw("ulp_type", &ComparisonOptions::ulpType)
        .def_rw("ulp_mode", &ComparisonOptions::ulpMode)
        .def_rw("relative_frobenius_tolerance", &ComparisonOptions::relativeFrobeniusTolerance)
        .def_rw("maximum_ulp_tolerance", &ComparisonOptions::maximumUlpTolerance)
        .def_rw("report_matching_elements", &ComparisonOptions::reportMatchingElements)
        .def_rw("max_reported_mismatches", &ComparisonOptions::maxReportedMismatches)
        .def_rw("selection", &ComparisonOptions::selection);

    nb::class_<ComparisonValue>(module, "ComparisonValue")
        .def_ro("real", &ComparisonValue::real)
        .def_ro("imaginary", &ComparisonValue::imaginary)
        .def_ro("complex", &ComparisonValue::complex);

    nb::class_<Mismatch>(module, "Mismatch")
        .def_ro("index", &Mismatch::index)
        .def_ro("coordinates", &Mismatch::coordinates)
        .def_ro("observed_offset", &Mismatch::observedOffset)
        .def_ro("expected_offset", &Mismatch::expectedOffset)
        .def_ro("observed", &Mismatch::observed)
        .def_ro("expected", &Mismatch::expected)
        .def_ro("observed_imaginary", &Mismatch::observedImaginary)
        .def_ro("expected_imaginary", &Mismatch::expectedImaginary)
        .def_ro("absolute_difference", &Mismatch::absoluteDifference)
        .def_ro("tolerance", &Mismatch::tolerance)
        .def_ro("matched", &Mismatch::matched);

    nb::class_<ComparisonResult>(module, "ComparisonResult")
        .def_ro("compared", &ComparisonResult::compared)
        .def_ro("mismatches", &ComparisonResult::mismatches)
        .def_ro("matched_nans", &ComparisonResult::matchedNaNs)
        .def_ro("matched_infinities", &ComparisonResult::matchedInfinities)
        .def_ro("non_finite_mismatches", &ComparisonResult::nonFiniteMismatches)
        .def_ro("signed_zero_mismatches", &ComparisonResult::signedZeroMismatches)
        .def_ro("max_absolute_difference", &ComparisonResult::maxAbsoluteDifference)
        .def_ro("max_relative_difference", &ComparisonResult::maxRelativeDifference)
        .def_ro("max_symmetric_relative_difference",
                &ComparisonResult::maxSymmetricRelativeDifference)
        .def_ro("maximum_observed_magnitude", &ComparisonResult::maximumObservedMagnitude)
        .def_ro("maximum_expected_magnitude", &ComparisonResult::maximumExpectedMagnitude)
        .def_ro("frobenius_difference", &ComparisonResult::frobeniusDifference)
        .def_ro("frobenius_observed", &ComparisonResult::frobeniusObserved)
        .def_ro("frobenius_expected", &ComparisonResult::frobeniusExpected)
        .def_ro("relative_frobenius_error", &ComparisonResult::relativeFrobeniusError)
        .def_ro("maximum_ulp", &ComparisonResult::maximumUlp)
        .def_ro("sum_ulp", &ComparisonResult::sumUlp)
        .def_ro("average_ulp", &ComparisonResult::averageUlp)
        .def_ro("ulp_compared", &ComparisonResult::ulpCompared)
        .def_ro("pointwise_passed", &ComparisonResult::pointwisePassed)
        .def_ro("frobenius_passed", &ComparisonResult::frobeniusPassed)
        .def_ro("ulp_passed", &ComparisonResult::ulpPassed)
        .def_ro("reported_mismatches", &ComparisonResult::reportedMismatches)
        .def_ro("reported_comparisons", &ComparisonResult::reportedComparisons)
        .def_prop_ro("passed", &ComparisonResult::passed);

    nb::class_<ComparisonTolerance>(module, "ComparisonTolerance")
        .def_ro("absolute", &ComparisonTolerance::absolute)
        .def_ro("relative", &ComparisonTolerance::relative);

    nb::enum_<SentinelRegion>(module, "SentinelRegion")
        .value("Unspecified", SentinelRegion::Unspecified)
        .value("Before", SentinelRegion::Before)
        .value("Inside", SentinelRegion::Inside)
        .value("After", SentinelRegion::After);

    nb::class_<SentinelMismatch>(module, "SentinelMismatch")
        .def_ro("region", &SentinelMismatch::region)
        .def_ro("index", &SentinelMismatch::index)
        .def_ro("observed", &SentinelMismatch::observed);

    nb::class_<SentinelResult>(module, "SentinelResult")
        .def_ro("checked", &SentinelResult::checked)
        .def_ro("mismatches", &SentinelResult::mismatches)
        .def_ro("reported_mismatches", &SentinelResult::reportedMismatches)
        .def_prop_ro("passed", &SentinelResult::passed);

    module.def(
        "compare",
        [](const Tensor& observed, const Tensor& expected, const ComparisonOptions& options) {
            return compare(observed, expected, options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def("default_comparison_options", &defaultComparisonOptions, "type"_a,
               "symmetric_relative_tolerance"_a = std::optional<double>{});
    module.def("near_comparison_options", &nearComparisonOptions, "absolute_tolerance"_a);
    module.def("allclose_comparison_options", &allCloseComparisonOptions,
               "absolute_tolerance"_a = 1e-8, "relative_tolerance"_a = 1e-5,
               "equal_nans"_a = false);
    module.def("ulp_mantissa_bits", &ulpMantissaBits, "type"_a);
    module.def("ulp_distance", &ulpDistance, "exact"_a, "approximation"_a, "mantissa_bits"_a);
    module.def("encoded_ulp_distance", &encodedUlpDistance, "exact"_a, "approximation"_a, "type"_a);
    module.def(
        "find_allclose_tolerance",
        [](const Tensor& observed, const Tensor& expected,
           const std::vector<double>& absoluteCandidates,
           const std::vector<double>& relativeCandidates, const ComparisonOptions& options) {
            return findAllCloseTolerance(observed, expected,
                                         std::span<const double>(absoluteCandidates),
                                         std::span<const double>(relativeCandidates), options);
        },
        "observed"_a, "expected"_a, "absolute_candidates"_a, "relative_candidates"_a,
        "options"_a = allCloseComparisonOptions());
    module.def(
        "check_unwritten_sentinel",
        [](const Tensor& tensor, SentinelRegion region, size_t maxReportedMismatches) {
            return checkUnwrittenSentinel(tensor.type(), tensor.storage(), 0,
                                          tensor.shape().elementCount(), region,
                                          maxReportedMismatches);
        },
        "tensor"_a, "region"_a = SentinelRegion::Unspecified, "max_reported_mismatches"_a = 10);
    module.def(
        "check_unused_tensor_storage",
        [](const Tensor& tensor, size_t allocatedElements, SentinelRegion region,
           size_t maxReportedMismatches) {
            return checkUnusedTensorStorage(tensor, allocatedElements, region,
                                            maxReportedMismatches);
        },
        "tensor"_a, "allocated_elements"_a, "region"_a = SentinelRegion::Inside,
        "max_reported_mismatches"_a = 10);
}
}  // namespace roc::host_validation::python_bindings
