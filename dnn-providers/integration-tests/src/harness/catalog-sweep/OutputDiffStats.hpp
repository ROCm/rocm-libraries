// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

namespace hipdnn_integration_tests::catalog_sweep
{

/// How far one output tensor is from its reference, element by element.
///
/// The validators that decide pass or fail answer only yes or no; the report also wants
/// to say how far off a failing kernel is, and whether a passing one is close to the
/// edge, so this measures it.
struct OutputDiffStats
{
    std::size_t totalElements = 0;
    /// Elements whose difference exceeds atol + rtol * |expected|, or that hold a NaN.
    std::size_t mismatchedElements = 0;
    /// Of those, the elements of the *actual* output that are NaN.
    std::size_t nanElements = 0;
    /// Largest finite |actual - expected|, over the elements where both are numbers and
    /// the difference is finite (an infinity against a finite value is a mismatch, but
    /// not a number worth reporting as the maximum).
    double maxAbsErr = 0.0;

    double mismatchPct() const
    {
        return totalElements == 0 ? 0.0
                                  : 100.0 * static_cast<double>(mismatchedElements)
                                        / static_cast<double>(totalElements);
    }
};

/// Measures `actual` against `expected` on the host. Element types the report does not
/// measure (anything but FLOAT, HALF, BFLOAT16 and DOUBLE) and tensors whose shapes
/// differ return nullopt. An element that is infinite with the same sign on both sides
/// is not a difference.
std::optional<OutputDiffStats>
    computeOutputDiffStats(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                           hipdnn_data_sdk::utilities::ITensor& expected,
                           hipdnn_data_sdk::utilities::ITensor& actual,
                           float atol,
                           float rtol);

} // namespace hipdnn_integration_tests::catalog_sweep
