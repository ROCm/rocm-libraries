// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ROCM-33326. An empty M or N is legal in a grouped GEMM and launches with zero
// tiles, but the distance search only stops early after some kernel accepts
// every member. No kernel accepts an empty M or N, so leaving that member in
// the ranked list evaluates every row. This counts those evaluations on the
// same search the grouped heuristic uses.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/PropertyMatching.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

using namespace TensileLite;

namespace
{
    struct Selected
    {
        int index = 0;
    };

    using Key = std::vector<int64_t>;

    struct KernelRow
    {
        Key key;
    };

    using KernelTable = Matching::DistanceMatchingTable<Key,
                                                        ContractionProblemGemm,
                                                        KernelRow,
                                                        std::shared_ptr<Selected>,
                                                        Matching::EuclideanDistance<Key>>;

    ContractionProblemGemm gemm(size_t m, size_t n, size_t k)
    {
        const size_t lda = std::max(m, size_t{1});
        const size_t ldb = std::max(k, size_t{1});
        return ContractionProblemGemm::GEMM(false, false, m, n, k, lda, ldb, lda, 0.0, false, 1);
    }

    KernelTable library()
    {
        KernelTable table{nullptr};
        for(int m = 16; m <= 256; m += 16)
        {
            for(int n = 16; n <= 256; n += 16)
            {
                Key key{m, n};
                table.table.push_back({key, KernelRow{key}, 1.0});
            }
        }
        std::sort(table.table.begin(), table.table.end(), [](auto const& a, auto const& b) {
            return a.key < b.key;
        });
        return table;
    }

    struct Scan
    {
        int    considered     = 0;
        int    index          = -1;
        size_t workspaceGroup = 0;
    };

    // The distance table is keyed by the member the search ranks. Production
    // keeps the full group for workspace (an empty member still takes an
    // argument slot) and ignores members with no output tiles. The unfiltered
    // path is the hang: the key is the first member, and a row is rejected
    // when any member has no tiles.
    Scan search(KernelTable const&                         table,
                std::vector<ContractionProblemGemm> const& members,
                bool                                       filter)
    {
        Scan scan;
        scan.workspaceGroup = members.size();

        ContractionProblemGemm const* keyProblem = nullptr;
        if(filter)
            keyProblem = groupedGemmMemberToRank(members);
        else if(!members.empty())
            keyProblem = &members[0];
        if(keyProblem == nullptr)
            return scan;

        Key key{static_cast<int64_t>(keyProblem->freeSizeA(0)),
                static_cast<int64_t>(keyProblem->freeSizeB(0))};

        auto found = table.findBestKeyMatch(key, [&](KernelRow const& row) {
            scan.considered++;
            if(row.key != key)
                return std::shared_ptr<Selected>{};
            for(auto const& member : members)
            {
                if(filter && !member.hasOutputTiles())
                    continue;
                if(!member.hasOutputTiles())
                    return std::shared_ptr<Selected>{};
            }
            return std::make_shared<Selected>(Selected{static_cast<int>(row.key[0])});
        });

        if(std::get<0>(found))
            scan.index = std::get<0>(found)->index;
        return scan;
    }
} // namespace

TEST(HostUnitGroupedHeuristic, smoke_EmptyMemberDoesNotVisitEveryKernel)
{
    const KernelTable table       = library();
    const int         everyKernel = static_cast<int>(table.table.size());
    ASSERT_GT(everyKernel, 1);

    const auto work   = gemm(128, 128, 128);
    const auto emptyM = gemm(0, 128, 128);
    const auto emptyN = gemm(128, 0, 128);
    const auto kZero  = gemm(128, 128, 0);

    // The unfiltered group is the hang: the search cannot stop, so it
    // evaluates every kernel. The assertion below is that bound.
    const Scan unfiltered = search(table, {work, emptyM}, false);
    ASSERT_EQ(unfiltered.considered, everyKernel);
    EXPECT_EQ(unfiltered.index, -1);

    auto expectSkipped = [&](const char*                                label,
                             std::vector<ContractionProblemGemm> const& members,
                             size_t                                     rankedCount) {
        EXPECT_EQ(groupedGemmMembersToRank(members).size(), rankedCount) << label;
        const Scan filtered = search(table, members, true);
        EXPECT_EQ(filtered.workspaceGroup, members.size()) << label;
        EXPECT_LT(filtered.considered, everyKernel) << label;
        EXPECT_EQ(filtered.considered, 1) << label;
        EXPECT_EQ(filtered.index, 128) << label;
    };

    expectSkipped("M=0 second", {work, emptyM}, 1);
    expectSkipped("M=0 first", {emptyM, work}, 1);
    expectSkipped("N=0 second", {work, emptyN}, 1);
    expectSkipped("N=0 first", {emptyN, work}, 1);
    expectSkipped("M=0 third", {work, work, emptyM}, 2);
    expectSkipped("K=0 is ranked", {work, kZero}, 2);

    EXPECT_TRUE(groupedGemmMembersToRank({emptyM, emptyN}).empty());
    EXPECT_EQ(search(table, {emptyM, emptyM}, true).considered, 0);
}
