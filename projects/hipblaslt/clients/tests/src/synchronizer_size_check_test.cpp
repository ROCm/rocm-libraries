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
//   * a grouped GEMM is handed the slot at its problem index, so one slot bounds
//     it, and a group with more problems than slots must not run such a solution
//
// A unit test rather than a GEMM because the single-slot bound on a non-grouped
// problem is not a correctness bug: it silently drops MBSK candidates the
// shipped logic files are tuned to select, and only a benchmark would notice.

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <sstream>

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
    //
    // `gsu` 0 leaves the problem's GSU unset, which is what makes the predicate
    // read value[5] instead; any other value is set on the problem.
    ContractionProblemGemm
        makeProblem(size_t m, size_t n, size_t batch, bool grouped, int count, int16_t gsu = 2)
    {
        auto problem = ContractionProblemGemm::GEMM(false, false, m, n, 64, m, 64, m, 0.0, false, batch);
        if(gsu != 0)
            problem.setParams().setGSU(gsu);
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

    // Exact bounds, one element apart. Every other test here sits an order of
    // magnitude away from a limit, so a `<` written for a `<=` would pass them
    // all. With the unit tile the usage is m * n * batch, so a single extra row
    // is the smallest step over each bound.
    constexpr size_t kWholeBufferSide = 2560; // 2560 * 2560 == 409600 * 16
    constexpr size_t kOneSlotSide     = 640; //   640 *  640 == 409600
    static_assert(kWholeBufferSide * kWholeBufferSide
                      == size_t(GsuSynchronizerElements) * SynchronizerGroupedSlots,
                  "");
    static_assert(kOneSlotSide * kOneSlotSide == size_t(GsuSynchronizerElements), "");

    TEST(SynchronizerSizeCheck, NonGroupedBoundIsInclusive)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        // 6553600 elements, exactly the buffer.
        EXPECT_TRUE(pred(makeProblem(kWholeBufferSide, kWholeBufferSide, 1, false, 0)));
        // 6556160, the next shape up.
        EXPECT_FALSE(pred(makeProblem(kWholeBufferSide + 1, kWholeBufferSide, 1, false, 0)));
    }

    TEST(SynchronizerSizeCheck, GroupedBoundIsInclusive)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        // 409600 elements, exactly one slot.
        EXPECT_TRUE(pred(makeProblem(kOneSlotSide, kOneSlotSide, 1, true, 2)));
        // 410240, the next shape up.
        EXPECT_FALSE(pred(makeProblem(kOneSlotSide + 1, kOneSlotSide, 1, true, 2)));
    }

    // A problem that never had its GSU set reaches the predicate with gsu() == 0,
    // so value[5] - the solution's declared GlobalSplitU - decides, and -1 there
    // is the "let the kernel choose" spelling that the shipped logic files carry
    // on nearly every MBSK solution. The predicate abstains on it before it
    // looks at the group width at all. That is intended, not an oversight, and
    // this test exists to keep it that way.
    //
    // The case is covered elsewhere: calculateAutoGSU early-returns whenever the
    // declared GlobalSplitU is not -1, so its clamp runs exactly when this
    // predicate abstains, applies the same fit test including the group-width
    // clause, and drops the solution to GSU 1, which reads no flags. Rejecting
    // here instead would refuse every auto-GSU MBSK solution on any group wider
    // than the slots, all of which run correctly at GSU 1.
    TEST(SynchronizerSizeCheck, AutoGsuFromValueAbstains)
    {
        SynchronizerSizeCheck pred(0, {1, 1, 1, 1, 1, -1});
        EXPECT_TRUE(pred(makeProblem(kBetweenBounds, kBetweenBounds, 64, true, 1024, 0)));
    }

    // debugEval is what --print_solution_rejection_reason drives. GSU -1 is the
    // other unsplit spelling - let the kernel choose - and is skipped the same way.
    //
    // This pins the verdict, not the short circuit: debugEvalCmp returns
    // operator()'s verdict, which short-circuits on GSU independently, so
    // dropping debugEval's own check would change only the printed row, and
    // PredicateDebugger drops passing rows outside TENSILE_DB verbose mode.
    // The guard is against a debugEval rewritten to judge for itself.
    TEST(SynchronizerSizeCheck, DebugEvalAcceptsUnsplitGsu)
    {
        SynchronizerSizeCheck pred(0, kUnitTile);
        for(int16_t gsu : {int16_t(1), int16_t(-1)})
        {
            auto problem = makeProblem(kBetweenBounds, kBetweenBounds, 64, true, 1024);
            problem.setParams().setGSU(gsu);
            ASSERT_TRUE(pred(problem)) << "gsu=" << gsu;

            std::ostringstream out;
            EXPECT_TRUE(pred.debugEval(problem, out)) << "gsu=" << gsu;
        }
    }
} // namespace
