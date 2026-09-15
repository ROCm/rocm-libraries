// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

int main() {
    using namespace blocked_backend_test;
    testReducedPrecisionAccumulators();
    testSelectedBlockAccumulatorFamilies();
    testSmallEdgeBlock();
    testExplicitSelectionBlockPlan();
    testStridedSelectionBlockPlan();
    testBlockScaledSelectionBlockPlan();
    testBlockScaleAppliedAfterCompleteScaleSegment();
    testOneSidedBlockScaling();
    testOneSidedBlockScalingWithZeroReductionExtent();
    testFullSelection();
    testAutomaticSelectionUsesBlockedBackend();
    testParallelFullSelection();
    testOverlappingOutputIsRejectedAcrossBackends();
    testOutputCannotAliasInputs();
    testParallelSparseSelection();
    return 0;
}
