// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/host_numerics/validation.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "test_support.hpp"

#ifdef HOST_NUMERICS_TEST_OPENMP
#include <omp.h>
#endif

namespace host_numerics_test {

namespace {
using roc::host_numerics::test::require;
using roc::host_numerics::test::requireInvalidArgument;

std::vector<std::byte> copyRawEncodedBackingStorage(const roc::host_numerics::Tensor& tensor) {
    const std::span<const std::byte> storage = tensor.rawEncodedBackingStorage();
    return {storage.begin(), storage.end()};
}

void requireRawEncodedBackingStorageEquals(const roc::host_numerics::Tensor& tensor,
                                           std::span<const std::byte> expected,
                                           const char* message) {
    const std::span<const std::byte> actual = tensor.rawEncodedBackingStorage();
    require(actual.size() == expected.size() &&
                std::equal(actual.begin(), actual.end(), expected.begin()),
            message);
}
}  // namespace

void testRuntimeReferenceGemm();
void testRuntimeMixedAndBlockScaledGemm();
void testBlockedUnalignedScaleSegments();
void testExactIntegerGemm();
void testRuntimeComplexAndExplicitAxisGemm();
void testOutputSelection();
void testReferenceEpilogue();
void testReferenceReduction();
void testStructuredSparsity();
void testIndexedGeneration();
void testTensorOperations();
void testReferenceSoftmax();
void testReferenceLayerNorm();
void testReferenceOperationAliasing();
void testStridedAndOffsetViews();
void testGenerationAndComparison();
void testComparisonProgram();
void testComparisonMetrics();
}  // namespace host_numerics_test
