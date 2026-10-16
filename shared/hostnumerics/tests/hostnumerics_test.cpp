// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "hostnumerics_test_support.hpp"

TEST(Operations, RuntimeReferenceGemm) {
    hostnumerics_test::testRuntimeReferenceGemm();
}

TEST(Operations, RuntimeMixedAndBlockScaledGemm) {
    hostnumerics_test::testRuntimeMixedAndBlockScaledGemm();
}

TEST(Operations, BlockedUnalignedScaleSegments) {
    hostnumerics_test::testBlockedUnalignedScaleSegments();
}

TEST(Operations, ExactIntegerGemm) {
    hostnumerics_test::testExactIntegerGemm();
}

TEST(Operations, RuntimeComplexAndExplicitAxisGemm) {
    hostnumerics_test::testRuntimeComplexAndExplicitAxisGemm();
}

TEST(Operations, OutputSelection) {
    hostnumerics_test::testOutputSelection();
}

TEST(Operations, ReferenceEpilogue) {
    hostnumerics_test::testReferenceEpilogue();
}

TEST(Operations, ReferenceReduction) {
    hostnumerics_test::testReferenceReduction();
}

TEST(Operations, StructuredSparsity) {
    hostnumerics_test::testStructuredSparsity();
}

TEST(Operations, IndexedGeneration) {
    hostnumerics_test::testIndexedGeneration();
}

TEST(Operations, TensorOperations) {
    hostnumerics_test::testTensorOperations();
}

TEST(Operations, ReferenceSoftmax) {
    hostnumerics_test::testReferenceSoftmax();
}

TEST(Operations, ReferenceLayerNorm) {
    hostnumerics_test::testReferenceLayerNorm();
}

TEST(Operations, ReferenceOperationAliasing) {
    hostnumerics_test::testReferenceOperationAliasing();
}

TEST(Operations, StridedAndOffsetViews) {
    hostnumerics_test::testStridedAndOffsetViews();
}

TEST(Operations, GenerationAndComparison) {
    hostnumerics_test::testGenerationAndComparison();
}

TEST(Operations, ComparisonProgram) {
    hostnumerics_test::testComparisonProgram();
}

TEST(Operations, ComparisonMetrics) {
    hostnumerics_test::testComparisonMetrics();
}
