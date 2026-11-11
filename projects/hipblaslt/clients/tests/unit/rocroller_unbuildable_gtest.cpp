// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The relative include avoids shadowing the clients' utility.hpp with the
// library's, as rocroller_kernel_select_gtest.cpp does.
#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_unbuildable.hpp"

#include "kernel_type.hpp"

#include <gtest/gtest.h>

#include <set>
#include <tuple>
#include <vector>

namespace
{
    using Format = RocblasltContractionProblem::ScalingFormat;
    using Row    = std::tuple<bool,
                              bool,
                              hipDataType,
                              hipDataType,
                              hipDataType,
                              hipDataType,
                              int,
                              int,
                              int,
                              bool,
                              bool>;

    constexpr RocRollerConfigKey recordedConfigs[] = {
#define X(transA, transB, typeA, typeB, typeC, typeD, tileM, tileN, tileK, wgm, streamK) \
    {transA, transB, typeA, typeB, typeC, typeD, tileM, tileN, tileK, wgm, streamK},
#include "unbuildable_configs.def"
#undef X
    };

    const std::vector<Row>& rows()
    {
        static const std::vector<Row> table = [] {
            std::vector<Row> r;
            for(const RocRollerConfigKey& c : recordedConfigs)
                r.emplace_back(c.transA,
                               c.transB,
                               c.typeA,
                               c.typeB,
                               c.typeC,
                               c.typeD,
                               c.tileM,
                               c.tileN,
                               c.tileK,
                               c.workgroupMapping,
                               c.streamK);
            return r;
        }();
        return table;
    }

    bool recorded(const Row& row)
    {
        static const std::set<Row> table(rows().begin(), rows().end());
        return table.count(row) != 0;
    }

    // The kernel type and ranked tile rankRocRollerSelectedTiles hands the
    // lookup for a problem with packed 32-element E8M0 scales.
    KernelType kernelType(const Row& row)
    {
        KernelType kernel{};
        std::tie(kernel.transA,
                 kernel.transB,
                 kernel.typeA,
                 kernel.typeB,
                 kernel.typeC,
                 kernel.typeD,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore)
            = row;
        kernel.scaleTypeA.format = Format::Block_32_UE8M0;
        kernel.scaleTypeB.format = Format::Block_32_UE8M0;
        return kernel;
    }

    RocRollerSelectedTile tile(const Row& row)
    {
        RocRollerSelectedTile t;
        std::tie(std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 t.m,
                 t.n,
                 t.k,
                 t.workgroupMapping,
                 t.streamK)
            = row;
        t.unroll = 2;
        return t;
    }

    bool unbuildable(const Row& row)
    {
        return rocRollerTileUnbuildable(kernelType(row), tile(row));
    }

    // The first ranked tile of the rocroller_streamk_under_uniform_summation_order
    // gtest (TN FP8 x FP8, bf16 output), and the Stream-K tile served after it.
    const Row fp8Tile16x16x256{true,
                               false,
                               HIP_R_8F_E4M3,
                               HIP_R_8F_E4M3,
                               HIP_R_16BF,
                               HIP_R_16BF,
                               16,
                               16,
                               256,
                               false,
                               false};
    const Row fp8StreamK16x16x128{true,
                                  false,
                                  HIP_R_8F_E4M3,
                                  HIP_R_8F_E4M3,
                                  HIP_R_16BF,
                                  HIP_R_16BF,
                                  16,
                                  16,
                                  128,
                                  false,
                                  true};
} // namespace

TEST(HostUnitRocRollerUnbuildable, smoke_EveryRecordedRowIsUnbuildable)
{
    ASSERT_FALSE(rows().empty());
    for(const Row& row : rows())
        EXPECT_TRUE(unbuildable(row)) << "row " << (&row - rows().data());
}

TEST(HostUnitRocRollerUnbuildable, smoke_TheStreamKCasePassesOverItsFirstTile)
{
    EXPECT_TRUE(unbuildable(fp8Tile16x16x256));
    EXPECT_FALSE(unbuildable(fp8StreamK16x16x128));
}

// Each field of the key counts: changing one gives the answer for the changed
// configuration, and for every field some change leaves the table. The
// recorded failures do not depend on the f16, bf16 or f32 output, so the
// output types change to ones no row has.
TEST(HostUnitRocRollerUnbuildable, smoke_EveryKeyFieldCounts)
{
    const std::vector<Row (*)(Row)> changes{
        [](Row r) {
            std::get<0>(r) = !std::get<0>(r);
            return r;
        },
        [](Row r) {
            std::get<1>(r) = !std::get<1>(r);
            return r;
        },
        [](Row r) {
            std::swap(std::get<2>(r), std::get<3>(r));
            return r;
        },
        [](Row r) {
            std::get<4>(r) = HIP_R_64F;
            return r;
        },
        [](Row r) {
            std::get<5>(r) = HIP_R_64F;
            return r;
        },
        [](Row r) {
            std::swap(std::get<6>(r), std::get<7>(r));
            return r;
        },
        [](Row r) {
            std::get<8>(r) *= 2;
            return r;
        },
        [](Row r) {
            std::get<10>(r) = !std::get<10>(r);
            std::get<9>(r)  = false;
            return r;
        },
    };
    for(size_t c = 0; c < changes.size(); ++c)
    {
        int leaves = 0;
        for(const Row& row : rows())
        {
            const Row changed = changes[c](row);
            EXPECT_EQ(unbuildable(changed), recorded(changed)) << "change " << c;
            leaves += !recorded(changed);
        }
        EXPECT_GT(leaves, 0) << "change " << c << " never leaves the table";
    }
}

// rocRoller turns workgroup mapping off for Stream-K, so a Stream-K tile with
// the mapping bit set is the recorded configuration without it.
TEST(HostUnitRocRollerUnbuildable, smoke_StreamKIgnoresWorkgroupMapping)
{
    int streamK = 0;
    for(Row row : rows())
    {
        if(!std::get<10>(row))
            continue;
        ++streamK;
        std::get<9>(row) = true;
        EXPECT_TRUE(unbuildable(row));
    }
    EXPECT_GT(streamK, 0);
}

TEST(HostUnitRocRollerUnbuildable, smoke_WorkgroupMappingCountsWithoutStreamK)
{
    int differs = 0;
    for(Row row : rows())
    {
        if(std::get<10>(row))
            continue;
        std::get<9>(row) = !std::get<9>(row);
        EXPECT_EQ(unbuildable(row), recorded(row));
        differs += !recorded(row);
    }
    EXPECT_GT(differs, 0);
}

TEST(HostUnitRocRollerUnbuildable, smoke_AnUnrecordedTileIsNot)
{
    Row row          = fp8Tile16x16x256;
    std::get<6>(row) = std::get<7>(row) = 48;
    EXPECT_FALSE(unbuildable(row));
}

// The table holds plain 32-element E8M0 scales only, so a kernel type with any
// other scales is never marked, even for a recorded tile.
TEST(HostUnitRocRollerUnbuildable, smoke_OnlyPlainE8M0ScalesAreRecorded)
{
    const RocRollerSelectedTile t = tile(fp8Tile16x16x256);
    ASSERT_TRUE(rocRollerTileUnbuildable(kernelType(fp8Tile16x16x256), t));

    std::vector<KernelType> others(8, kernelType(fp8Tile16x16x256));
    others[0].scaleTypeA.format         = Format::Block_32_UE8M0_32_8_EXT;
    others[1].scaleTypeB.format         = Format::Block_32_UE4M3;
    others[2].scaleTypeA.preSwizzleTile = {32, 4, 4};
    others[3].scaleTypeB.preSwizzleTile = {32, 4, 4};
    others[4].scaleTypeA.preTile        = {128, 128};
    others[5].scaleTypeB.preTile        = {128, 128};
    others[6].swizzleA                  = true;
    others[7].swizzleB                  = true;
    for(size_t i = 0; i < others.size(); ++i)
        EXPECT_FALSE(rocRollerTileUnbuildable(others[i], t)) << "kernel type " << i;
}
