// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the gemm that a grouped launch checks a hand-written
// kernel's problem predicates on. It must carry the group size and the
// cu-fallback status as grouped selection sets them, and leave the caller's
// gemm unchanged. The relative include avoids shadowing the clients'
// utility.hpp with the library's internal header of the same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/grouped_predicate_problem.hpp"

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace
{
    TensileLite::ContractionProblemGroupedGemm groupOf(size_t count)
    {
        auto gemm = TensileLite::ContractionProblemGemm::GEMM_Strides(false,
                                                                      false,
                                                                      rocisa::DataType::Half,
                                                                      rocisa::DataType::Half,
                                                                      rocisa::DataType::Half,
                                                                      rocisa::DataType::Half,
                                                                      256,
                                                                      256,
                                                                      256,
                                                                      1,
                                                                      256,
                                                                      256 * 256,
                                                                      256,
                                                                      256 * 256,
                                                                      256,
                                                                      256 * 256,
                                                                      256,
                                                                      256 * 256,
                                                                      0.0);
        gemm.setGroupedGemm(true);
        TensileLite::ContractionProblemGroupedGemm grouped;
        grouped.gemms.assign(count, gemm);
        return grouped;
    }

    // A 256x256 tile split two ways: one tile of flags, well inside one slot, so
    // only the group size decides.
    const TensileLite::Predicates::Contraction::SynchronizerSizeCheck splitTile(0,
                                                                              {256, 256, 1, 1, 1, 2});
}

TEST(HostUnitGroupedPredicateProblem, smoke_CarriesTheGroupSize)
{
    auto grouped = groupOf(3);
    ASSERT_EQ(grouped.gemms[1].groupedGemmCount(), std::numeric_limits<int>::max());
    EXPECT_FALSE(splitTile(grouped.gemms[1])) << "an unset group size reads as too many gemms";

    auto gemm = rocblaslt::groupedPredicateProblem(grouped, 1, false);
    EXPECT_EQ(gemm.groupedGemmCount(), 3);
    EXPECT_TRUE(splitTile(gemm));

    EXPECT_FALSE(splitTile(rocblaslt::groupedPredicateProblem(
        groupOf(TensileLite::SynchronizerGroupedSlots + 1), 0, false)))
        << "a group wider than the flag slots";
}

TEST(HostUnitGroupedPredicateProblem, smoke_CarriesTheFallbackStatus)
{
    // XCC 8 does not divide 38 CUs. A cu-fallback launches at XCC 1, which does.
    const TensileLite::Predicates::Contraction::WorkgroupMappingXCCCheck xcc8({8, -1}, 38);
    auto grouped = groupOf(2);
    EXPECT_FALSE(xcc8(grouped.gemms[0]));
    EXPECT_FALSE(xcc8(rocblaslt::groupedPredicateProblem(grouped, 0, false)));
    EXPECT_TRUE(xcc8(rocblaslt::groupedPredicateProblem(grouped, 0, true)));
}

TEST(HostUnitGroupedPredicateProblem, smoke_LeavesTheCallersGemmUnchanged)
{
    auto grouped = groupOf(2);
    static_cast<void>(rocblaslt::groupedPredicateProblem(grouped, 0, true));
    EXPECT_EQ(grouped.gemms[0].groupedGemmCount(), std::numeric_limits<int>::max());
    EXPECT_FALSE(grouped.gemms[0].getParams().fallbackStatus());
}
