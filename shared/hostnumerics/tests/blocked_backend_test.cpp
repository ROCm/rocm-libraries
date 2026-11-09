// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "_blocked_backend_test_support.hpp"

TEST(BlockedBackend, ReducedPrecisionAccumulators) {
    blocked_backend_test::testReducedPrecisionAccumulators();
}

TEST(BlockedBackend, SelectedBlockAccumulatorFamilies) {
    blocked_backend_test::testSelectedBlockAccumulatorFamilies();
}

TEST(BlockedBackend, SmallEdgeBlock) {
    blocked_backend_test::testSmallEdgeBlock();
}

TEST(BlockedBackend, ExplicitSelectionBlockPlan) {
    blocked_backend_test::testExplicitSelectionBlockPlan();
}

TEST(BlockedBackend, StridedSelectionBlockPlan) {
    blocked_backend_test::testStridedSelectionBlockPlan();
}

TEST(BlockedBackend, BlockScaledSelectionBlockPlan) {
    blocked_backend_test::testBlockScaledSelectionBlockPlan();
}

TEST(BlockedBackend, BlockScaleAppliedAfterCompleteScaleSegment) {
    blocked_backend_test::testBlockScaleAppliedAfterCompleteScaleSegment();
}

TEST(BlockedBackend, OneSidedBlockScaling) {
    blocked_backend_test::testOneSidedBlockScaling();
}

TEST(BlockedBackend, OneSidedBlockScalingWithZeroReductionExtent) {
    blocked_backend_test::testOneSidedBlockScalingWithZeroReductionExtent();
}

TEST(BlockedBackend, FullSelection) {
    blocked_backend_test::testFullSelection();
}

TEST(BlockedBackend, AutomaticSelectionUsesBlockedBackend) {
    blocked_backend_test::testAutomaticSelectionUsesBlockedBackend();
}

TEST(BlockedBackend, ParallelFullSelection) {
    blocked_backend_test::testParallelFullSelection();
}

TEST(BlockedBackend, OverlappingOutputIsRejectedAcrossBackends) {
    blocked_backend_test::testOverlappingOutputIsRejectedAcrossBackends();
}

TEST(BlockedBackend, OutputCannotAliasInputs) {
    blocked_backend_test::testOutputCannotAliasInputs();
}

TEST(BlockedBackend, ParallelSparseSelection) {
    blocked_backend_test::testParallelSparseSelection();
}
