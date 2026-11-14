// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the rocRoller-off walk over rocRoller's ranked tiles and
// of the checks that decide whether a checked-in kernel may serve a problem.
// The relative include avoids shadowing the clients' utility.hpp with the
// library's internal header of the same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_kernel_select.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace
{
    struct FakeProblem
    {
        int    id        = 0;
        size_t workspace = 0;
    };

    struct FakeHardware
    {
        int arch = 950;
    };

    struct FakeProblemPredicate
    {
        int  accepts;
        bool operator()(const FakeProblem& p) const
        {
            return p.id == accepts;
        }
    };

    struct FakeHardwarePredicate
    {
        int  arch;
        bool operator()(const FakeHardware& h) const
        {
            return h.arch == arch;
        }
    };

    struct FakeSolution
    {
        std::shared_ptr<FakeProblemPredicate>  problemPredicate;
        std::shared_ptr<FakeHardwarePredicate> hardwarePredicate;
        size_t                                 workspace = 0;

        size_t requiredWorkspaceSize(const FakeProblem&, const FakeHardware&) const
        {
            return workspace;
        }
    };

    // {tileM, tileN, tileK, unroll, workgroupMapping, streamK}
    constexpr RocRollerKernelFields k16{16, 16, 128, 2, false, false};
    constexpr RocRollerKernelFields k32{32, 32, 128, 2, false, false};
    constexpr RocRollerKernelFields k64sk{64, 64, 128, 2, false, true};
    constexpr RocRollerKernelFields k256{256, 256, 128, 2, false, false};

    // {m, n, k, workgroupMapping, streamK, nonTemporalA, nonTemporalB, unroll}
    constexpr RocRollerSelectedTile t16{16, 16, 128, false, false, false, false, 2};
    constexpr RocRollerSelectedTile t32{32, 32, 128, false, false, false, false, 2};
    constexpr RocRollerSelectedTile t64sk{64, 64, 128, false, true, false, false, 2};
    constexpr RocRollerSelectedTile t256{256, 256, 128, false, false, false, false, 2};
    constexpr RocRollerSelectedTile t48{48, 48, 128, false, false, false, false, 2};

    // The tile as recorded in unbuildable_configs.def.
    RocRollerSelectedTile unbuildable(RocRollerSelectedTile tile)
    {
        tile.unbuildable = true;
        return tile;
    }

    auto always = [](size_t) { return true; };
    auto never  = [](size_t) { return false; };

    RocRollerWalkResult walk(const std::vector<RocRollerSelectedTile>& ranked,
                             const std::vector<RocRollerKernelFields>& kernels,
                             int                                       count = -1,
                             size_t                                    m     = 1024,
                             size_t                                    n     = 1024,
                             size_t                                    k     = 1024)
    {
        return walkRocRollerTiles(ranked, m, n, k, kernels, count, always, never);
    }
}

TEST(HostUnitRocRollerKernelSelect, smoke_AcceptsRunsBothPredicates)
{
    FakeSolution solution;
    EXPECT_TRUE(rocRollerKernelAccepts(solution, FakeProblem{}, FakeHardware{}))
        << "a kernel without predicates accepts every problem";

    solution.problemPredicate = std::make_shared<FakeProblemPredicate>(FakeProblemPredicate{1});
    solution.hardwarePredicate
        = std::make_shared<FakeHardwarePredicate>(FakeHardwarePredicate{950});
    EXPECT_TRUE(rocRollerKernelAccepts(solution, FakeProblem{1}, FakeHardware{950}));
    EXPECT_FALSE(rocRollerKernelAccepts(solution, FakeProblem{2}, FakeHardware{950}))
        << "the problem predicate refuses";
    EXPECT_FALSE(rocRollerKernelAccepts(solution, FakeProblem{1}, FakeHardware{942}))
        << "the hardware predicate refuses";
}

TEST(HostUnitRocRollerKernelSelect, smoke_FitsWorkspaceIsInclusive)
{
    FakeSolution solution;
    solution.workspace = 4096;
    EXPECT_TRUE(rocRollerKernelFitsWorkspace(solution, FakeProblem{}, FakeHardware{}, 4096));
    EXPECT_TRUE(rocRollerKernelFitsWorkspace(solution, FakeProblem{}, FakeHardware{}, 8192));
    EXPECT_FALSE(rocRollerKernelFitsWorkspace(solution, FakeProblem{}, FakeHardware{}, 4095));
}

TEST(HostUnitRocRollerKernelSelect, smoke_KeepsTheKernelOfTheFirstTile)
{
    const auto r = walk({t32, t16}, {k16, k32}, 1);
    ASSERT_EQ(r.picks.size(), 1u);
    EXPECT_EQ(r.picks[0], 1u);
    EXPECT_EQ(r.stop, RocRollerWalkStop::CountReached);
    EXPECT_EQ(r.stopTile, 1u);
}

TEST(HostUnitRocRollerKernelSelect, smoke_KeepsRankOrder)
{
    const auto r = walk({t256, t16, t32}, {k16, k32, k256});
    EXPECT_EQ(r.picks, (std::vector<size_t>{2, 0, 1}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted);
    EXPECT_EQ(r.stopTile, 3u);
}

// rocRoller passes over a tile that does not divide the problem.
TEST(HostUnitRocRollerKernelSelect, smoke_PassesOverATileThatDoesNotDivide)
{
    const auto r = walk({t48, t32}, {k32}, 1);
    ASSERT_EQ(r.picks.size(), 1u);
    EXPECT_EQ(r.picks[0], 0u);

    const auto nOnly = walk({t256, t16}, {k16, k256}, 1, 1024, 48, 1024);
    ASSERT_EQ(nOnly.picks.size(), 1u);
    EXPECT_EQ(nOnly.picks[0], 0u) << "256 does not divide N = 48";

    const auto kNone = walk({t16}, {k16}, -1, 1024, 1024, 64);
    EXPECT_TRUE(kNone.picks.empty()) << "128 does not divide K = 64";
    EXPECT_EQ(kNone.stop, RocRollerWalkStop::Exhausted);
}

// When rocRoller's pick has no checked-in kernel the walk returns nothing,
// never the kernel of a tile rocRoller ranks lower.
TEST(HostUnitRocRollerKernelSelect, smoke_NoSubstituteForAMissingKernel)
{
    const auto r = walk({t256, t32}, {k32});
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::NoKernel);
    EXPECT_EQ(r.stopTile, 0u);
}

// The same holds further down: a later missing kernel ends the list there.
TEST(HostUnitRocRollerKernelSelect, smoke_AMissingKernelEndsTheList)
{
    const auto r = walk({t16, t256, t32}, {k16, k32});
    EXPECT_EQ(r.picks, (std::vector<size_t>{0}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::NoKernel);
    EXPECT_EQ(r.stopTile, 1u);
}

TEST(HostUnitRocRollerKernelSelect, smoke_TwoKernelsForOneTileEndTheWalk)
{
    const auto r = walk({t16, t32}, {k32, k16, k16});
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::Ambiguous);
}

// rocRoller passes over a kernel that needs more workspace than the caller
// gives, so the walk does too.
TEST(HostUnitRocRollerKernelSelect, smoke_PassesOverAKernelOverTheWorkspaceBudget)
{
    const auto r = walkRocRollerTiles(
        {t64sk, t16}, 1024, 1024, 1024, {k16, k64sk}, -1, [](size_t i) { return i != 1; }, never);
    EXPECT_EQ(r.picks, (std::vector<size_t>{0}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted);
}

TEST(HostUnitRocRollerKernelSelect, smoke_AKernelThatRefusesTheLaunchEndsTheWalk)
{
    const auto r = walkRocRollerTiles(
        {t64sk, t16}, 1024, 1024, 1024, {k16, k64sk}, -1, always, [](size_t i) { return i == 1; });
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::KernelRefusesProblem);
    EXPECT_EQ(r.stopTile, 0u);
}

// Over the workspace budget is checked first: rocRoller passes over such a
// tile without asking whether it could launch.
TEST(HostUnitRocRollerKernelSelect, smoke_WorkspaceIsCheckedBeforeTheLaunch)
{
    const auto r = walkRocRollerTiles(
        {t64sk, t16},
        1024,
        1024,
        1024,
        {k16, k64sk},
        -1,
        [](size_t i) { return i != 1; },
        [](size_t i) { return i == 1; });
    EXPECT_EQ(r.picks, (std::vector<size_t>{0}));
}

// Tiles that differ only in non-temporal hints share one kernel, kept once.
TEST(HostUnitRocRollerKernelSelect, smoke_KeepsAKernelOnce)
{
    RocRollerSelectedTile t16nt = t16;
    t16nt.nonTemporalA          = true;
    const auto r                = walk({t16, t16nt, t32}, {k16, k32});
    EXPECT_EQ(r.picks, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted);
}

// The walk refuses a real rocRoller Stream-K kernel exactly where its launch
// would run data-parallel tiles: at two or more tiles per workgroup. With no
// occupancy recorded its grid is the CU count capped by smCountTarget.
TEST(HostUnitRocRollerKernelSelect, smoke_StreamKIsRefusedFromTwoTilesPerWorkgroup)
{
    auto kernel = [](TensileLite::ContractionSolution& solution,
                     TensileLite::CustomWorkspaceType  workspaceType) {
        solution.customKernel.name          = "rr_probe";
        solution.customKernel.macrotile     = TensileLite::dim3(128, 128, 128);
        solution.customKernel.unroll        = 2;
        solution.customKernel.workspaceType = workspaceType;
    };
    TensileLite::ContractionSolution streamK;
    kernel(streamK, TensileLite::CustomWorkspaceType::RocRollerStreamK);
    TensileLite::ContractionSolution plain;
    kernel(plain, TensileLite::CustomWorkspaceType::None);

    const TensileLite::AMDGPU       device(TensileLite::AMDGPU::Processor::gfx950, 256, "gfx950");
    constexpr RocRollerKernelFields k128sk{128, 128, 128, 2, false, true};
    constexpr RocRollerSelectedTile t128sk{128, 128, 128, false, true, false, false, 2};

    // 1024x1024 in 128x128 tiles is 64 tiles.
    struct Case
    {
        int  grid;
        bool served;
    };
    for(Case c : {Case{0, true}, // 64 tiles on 256 workgroups
                  Case{64, true}, // tiles == grid
                  Case{48, true}, // grid < tiles < 2 * grid
                  Case{33, true}, // just below 2 * grid
                  Case{32, false}, // tiles == 2 * grid
                  Case{8, false}})
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 1024, 1024, 1024, 1024, 1024, 1024, 1.0, false, 1);
        problem.setParams().setSmCountTarget(c.grid);

        const auto r
            = walkRocRollerTiles({t128sk}, 1024, 1024, 1024, {k128sk}, -1, always, [&](size_t) {
                  return rocRollerKernelLaunchHasDataParallelTiles(streamK, problem, device);
              });
        if(c.served)
        {
            EXPECT_EQ(r.picks, (std::vector<size_t>{0})) << "grid " << c.grid;
            EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted) << "grid " << c.grid;
        }
        else
        {
            EXPECT_TRUE(r.picks.empty()) << "grid " << c.grid;
            EXPECT_EQ(r.stop, RocRollerWalkStop::KernelRefusesProblem) << "grid " << c.grid;
            EXPECT_EQ(r.stopTile, 0u) << "grid " << c.grid;
        }

        EXPECT_FALSE(rocRollerKernelLaunchHasDataParallelTiles(plain, problem, device))
            << "a kernel that is not Stream-K is never refused; grid " << c.grid;
    }
}

// A checked-in rocRoller kernel carries RocRollerCaptures in its hardware
// predicate. Only the build without rocRoller marks its hardware, so the same
// kernel is a candidate there and refused by a rocRoller build's Tensile paths;
// a kernel without the term is accepted by both.
TEST(HostUnitRocRollerKernelSelect, smoke_OnlyTheRocRollerOffBuildAcceptsCaptures)
{
    using AMDGPUPredicate
        = std::shared_ptr<TensileLite::Predicates::Predicate<TensileLite::AMDGPU>>;
    auto onGfx950 = [](std::vector<AMDGPUPredicate> terms) {
        terms.insert(terms.begin(),
                     std::make_shared<TensileLite::Predicates::GPU::ProcessorEqual>(
                         TensileLite::AMDGPU::Processor::gfx950));
        return std::make_shared<
            TensileLite::Predicates::IsSubclass<TensileLite::Hardware, TensileLite::AMDGPU>>(
            std::make_shared<TensileLite::Predicates::And<TensileLite::AMDGPU>>(terms));
    };
    TensileLite::ContractionSolution capture;
    capture.hardwarePredicate
        = onGfx950({std::make_shared<TensileLite::Predicates::GPU::RocRollerCaptures>()});
    TensileLite::ContractionSolution other;
    other.hardwarePredicate = onGfx950({});

    const auto problem = TensileLite::ContractionProblemGemm::GEMM(
        true, false, 256, 256, 256, 256, 256, 256, 1.0, false, 1);
    TensileLite::AMDGPU device(TensileLite::AMDGPU::Processor::gfx950, 256, "gfx950");
    EXPECT_FALSE(device.rocRollerCaptures);
    EXPECT_FALSE(rocRollerKernelAccepts(capture, problem, device));
    EXPECT_TRUE(rocRollerKernelAccepts(other, problem, device));

    const TensileLite::AMDGPU unmarked = device;
    allowRocRollerCaptures(device);
#ifdef HIPBLASLT_USE_ROCROLLER
    EXPECT_FALSE(device.rocRollerCaptures);
    EXPECT_FALSE(rocRollerKernelAccepts(capture, problem, device));
    EXPECT_TRUE(unmarked == device);
#else
    EXPECT_TRUE(device.rocRollerCaptures);
    EXPECT_TRUE(rocRollerKernelAccepts(capture, problem, device));
    // Selection caches key on the AMDGPU, so marked and unmarked never share an entry.
    EXPECT_FALSE(unmarked == device);
#endif
    EXPECT_TRUE(rocRollerKernelAccepts(other, problem, device));

    // An index lookup treats a capture as absent only where the term alone
    // refuses it. A kernel for another processor is not a capture refusal.
    EXPECT_TRUE(refusedOnlyForRocRollerCaptures(capture, unmarked));
    EXPECT_FALSE(refusedOnlyForRocRollerCaptures(other, unmarked));
    TensileLite::AMDGPU gfx942(TensileLite::AMDGPU::Processor::gfx942, 304, "gfx942");
    EXPECT_FALSE(refusedOnlyForRocRollerCaptures(capture, gfx942));
    EXPECT_FALSE(refusedOnlyForRocRollerCaptures(other, gfx942));
#ifndef HIPBLASLT_USE_ROCROLLER
    EXPECT_FALSE(refusedOnlyForRocRollerCaptures(capture, device));
#endif
}

TEST(HostUnitRocRollerKernelSelect, smoke_NoTilesNoKernels)
{
    const auto r = walk({}, {k16});
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted);

    const auto none = walk({t16}, {});
    EXPECT_TRUE(none.picks.empty());
    EXPECT_EQ(none.stop, RocRollerWalkStop::NoKernel);
}

TEST(HostUnitRocRollerKernelSelect, smoke_ZeroCountKeepsNothing)
{
    const auto r = walk({t16}, {k16}, 0);
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::CountReached);
}

// rocRoller passes over a tile it fails to build, so the walk passes over a
// recorded one and keeps the kernel of the next tile.
TEST(HostUnitRocRollerKernelSelect, smoke_PassesOverAnUnbuildableTile)
{
    const auto r = walk({unbuildable(t256), t16}, {k16});
    EXPECT_EQ(r.picks, (std::vector<size_t>{0}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::Exhausted);
}

// rocRoller never runs a configuration it fails to build, so a checked-in
// kernel for a recorded tile is passed over too.
TEST(HostUnitRocRollerKernelSelect, smoke_AnUnbuildableTileIsPassedOverEvenWithAKernel)
{
    const auto r = walk({unbuildable(t16), t32}, {k16, k32}, 1);
    EXPECT_EQ(r.picks, (std::vector<size_t>{1}));
}

// A tile that is neither recorded nor checked in still ends the walk, after
// the recorded tiles before it.
TEST(HostUnitRocRollerKernelSelect, smoke_AnUnrecordedTileWithoutAKernelStillStops)
{
    const auto r = walk({unbuildable(t32), t256, t16}, {k16});
    EXPECT_TRUE(r.picks.empty());
    EXPECT_EQ(r.stop, RocRollerWalkStop::NoKernel);
    EXPECT_EQ(r.stopTile, 1u);
}

// A recorded tile keeps nothing, so it does not count toward the request.
TEST(HostUnitRocRollerKernelSelect, smoke_AnUnbuildableTileDoesNotCountTowardTheRequest)
{
    const auto r = walk({unbuildable(t256), t32, t16}, {k16, k32}, 1);
    EXPECT_EQ(r.picks, (std::vector<size_t>{1}));
    EXPECT_EQ(r.stop, RocRollerWalkStop::CountReached);
    EXPECT_EQ(r.stopTile, 2u);
}
