// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

int main() {
    using namespace host_numerics_test;
    testRuntimeReferenceGemm();
    testRuntimeMixedAndBlockScaledGemm();
    testBlockedUnalignedScaleSegments();
    testExactIntegerGemm();
    testRuntimeComplexAndExplicitAxisGemm();
    testOutputSelection();
    testReferenceEpilogue();
    testReferenceReduction();
    testStructuredSparsity();
    testIndexedGeneration();
    testTensorOperations();
    testReferenceSoftmax();
    testReferenceLayerNorm();
    testReferenceOperationAliasing();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    testComparisonProgram();
    testComparisonMetrics();
    return 0;
}
