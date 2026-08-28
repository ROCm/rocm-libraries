/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Unit tests for the SynchronizerSizeCheck predicate, which decides whether a
// solution's GSU (MBSK) flag usage fits the region it will be handed.
//
// The two callers get different regions, so they get different bounds:
//
//   * a non-grouped GEMM is handed the base of the whole buffer, so it may use
//     all GsuSynchronizerElements * SynchronizerGroupedSlots elements
//   * a grouped GEMM is handed the slot at its problem index, so it is bounded
//     by one slot, and a group with more problems than there are slots has no
//     slot of its own for every problem and must not run such a solution
//
// Applying the single-slot bound to a non-grouped problem is not a correctness
// bug, which is why this is a unit test rather than a GEMM: it silently drops
// MBSK candidates that the shipped logic files are tuned to select, and only a
// benchmark would notice.

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>

namespace
{
    using TensileLite::GsuSynchronizerElements;
    using TensileLite::SynchronizerGroupedSlots;
    using TensileLite::ContractionProblemGemm;
    using TensileLite::Predicates::Contraction::SynchronizerSizeCheck;

    // value is {MT0, MT1, globalWriteInstructions, waves, elementsPerThread,
    // defaultGsu}. All but the macro tile are 1 here so that the usage the
    // predicate computes is exactly ceil(m / MT0) * ceil(n / MT1) * batch, and
    // the bound under test is the only thing the outcome depends on.
    ContractionProblemGemm makeProblem(size_t m, size_t n, size_t batch, bool grouped, int count)
    {
        auto problem = ContractionProblemGemm::GEMM(false, false, m, n, 64, m, 64, m, 0.0, false, batch);
        problem.setParams().setGSU(2);
        problem.setGroupedGemm(grouped);
        if(grouped)
            problem.setGroupedGemmCount(count);
        return problem;
    }

    constexpr std::array<int, 6> kUnitTile = {1, 1, 1, 1, 1, 2};

    // Usage 1024 * 1024 sits above one slot and inside the whole buffer, which
    // is the range the two bounds disagree on.
    constexpr size_t kBetweenBounds = 1024;
    static_assert(kBetweenBounds * kBetweenBounds > GsuSynchronizerElements, "");
    static_assert(kBetweenBounds * kBetweenBounds
                      <= size_t(GsuSynchronizerElements) * SynchronizerGroupedSlots,
                  "");

    TEST(SynchronizerSizeCheck, NonGroupedMayUseTheWholeBuffer)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        EXPECT_TRUE(pred(makeProblem(kBetweenBounds, kBetweenBounds, 1, false, 0)));
    }

    TEST(SynchronizerSizeCheck, NonGroupedRejectedPastTheWholeBuffer)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        // One batch past the buffer.
        const size_t batch = size_t(GsuSynchronizerElements) * SynchronizerGroupedSlots
                                 / (kBetweenBounds * kBetweenBounds)
                             + 1;
        EXPECT_FALSE(pred(makeProblem(kBetweenBounds, kBetweenBounds, batch, false, 0)));
    }

    TEST(SynchronizerSizeCheck, GroupedBoundedByOneSlot)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        EXPECT_TRUE(pred(makeProblem(512, 512, 1, true, 2)));
        EXPECT_FALSE(pred(makeProblem(kBetweenBounds, kBetweenBounds, 1, true, 2)));
    }

    TEST(SynchronizerSizeCheck, GroupWiderThanTheSlotsIsRejected)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        EXPECT_TRUE(pred(makeProblem(512, 512, 1, true, SynchronizerGroupedSlots)));
        EXPECT_FALSE(pred(makeProblem(512, 512, 1, true, SynchronizerGroupedSlots + 1)));
    }

    TEST(SynchronizerSizeCheck, GsuOneUsesNoFlags)
    {
        // GSU 1 never reaches the flags, so the size is irrelevant.
        SynchronizerSizeCheck pred(0, {1, 1, 1, 1, 1, 1});
        auto problem = makeProblem(kBetweenBounds, kBetweenBounds, 64, true, 1024);
        problem.setParams().setGSU(1);
        EXPECT_TRUE(pred(problem));
    }
} // namespace
