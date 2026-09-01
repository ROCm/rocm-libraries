// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for the StreamK "launch summary" decision snapshot produced by
// ContractionSolution::computeStreamKDecisions(). This is the single source of
// truth solve() consumes to fill StreamKSettings, so asserting on it here is
// asserting on the real launch DECISIONS (mode / reduction / grid / tiles /
// split / workspace / partials / DP-only / fallbacks) made in the StreamK
// launch-parameter path -- without needing a GPU. Host-only: mock AMDGPU and
// hip::HipAMDGPU devices, no device library required.
//
// The partials-workspace behaviour asserted below is that the dynamic (SK4 /
// SK5-dynamic) path reserves the partials region based on tiles%grid
// divisibility, NOT on the skTiles*skSplit slot count. The tests near the bottom
// pin that relationship between dynamicPartialsSlots, tiles%grid divisibility,
// and whether a partials workspace is reserved.

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/streamk.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    constexpr size_t kGfx950AnalyticalCuCount = 256;

    // gfx950 analytical hardware advertising NUM_XCD=8 (matches the baked
    // per-XCD work-queue count), so the SK4 / SK5-dynamic work-stealing path is
    // supported. Mirrors makeGfx950AnalyticalHardware in CuCount_test.cpp.
    origami::hardware_t makeGfx950AnalyticalHardware()
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx950,
                                   kGfx950AnalyticalCuCount,
                                   163840,
                                   262144,
                                   8, // NUM_XCD
                                   1.0,
                                   1.0,
                                   1.0,
                                   4000000,
                                   1.2,
                                   1,
                                   std::make_tuple(0.0, 0.008, 0.0));
    }

    hip::HipAMDGPU makeHipDeviceWithAnalytical(origami::hardware_t const& hw)
    {
        hip::HipAMDGPU device;
        device.processor          = AMDGPU::Processor::gfx950;
        device.computeUnitCount   = static_cast<int>(hw.N_CU);
        device.deviceName         = "test-gfx950-analytical";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    void initStreamKSolution(ContractionSolution& solution, int streamK)
    {
        solution.sizeMapping.streamK               = streamK;
        solution.sizeMapping.streamKAtomic         = 0;
        solution.sizeMapping.streamKForceDPOnly    = 0;
        solution.sizeMapping.macroTile             = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU                = 64;
        solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
        solution.sizeMapping.CUOccupancy           = 1;
        solution.sizeMapping.workspaceSizePerElemC = 4;
    }

    ContractionProblemGemm makeGemmProblem(size_t m, size_t n, size_t k)
    {
        auto problem = ContractionProblemGemm::GEMM(false, false, m, n, k, m, n, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    struct AnalyticalEnv
    {
        AnalyticalEnv()
            : hw(makeGfx950AnalyticalHardware())
            , device(makeHipDeviceWithAnalytical(hw))
        {
        }
        origami::hardware_t hw;
        hip::HipAMDGPU      device;
    };

    // The summary is a deeply-indented "key = value" block whose column widths are
    // chosen per-section for alignment. Collapsing runs of spaces to a single space
    // lets the assertions below match on the stable tokens ("changedBy = ...",
    // "source = ...", etc.) without pinning exact column widths. Newlines are left
    // intact so a token can never be matched across a line break.
    std::string collapseSpaces(std::string const& s)
    {
        std::string out;
        out.reserve(s.size());
        bool prevSpace = false;
        for(char c : s)
        {
            if(c == ' ')
            {
                if(!prevSpace)
                    out.push_back(' ');
                prevSpace = true;
            }
            else
            {
                out.push_back(c);
                prevSpace = false;
            }
        }
        return out;
    }

    // Sets (or clears) TENSILE_DB for the lifetime of the object and refreshes the
    // Debug singleton from it, restoring the PREVIOUS value -- not merely unsetting
    // -- on scope exit. Debug::Instance() is a process-wide function-local static
    // (see Singleton.hpp, LazySingleton), so an unrestored TENSILE_DB would leak
    // into every later test in this binary; the destructor still runs when an
    // ASSERT_* aborts the enclosing test, which a TearDown() could not be relied on
    // to do for a scoped state change.
    //
    // reloadDebugBitsForTest() refreshes exactly TENSILE_DB, TENSILE_DB2 and
    // TENSILE_STREAMK5_FORCE_MODE. It deliberately does NOT refresh
    // TENSILE_STREAMK_DATA_PARALLEL, so nothing here can toggle that flag.
    class ScopedTensileDb
    {
    public:
        // value == nullptr means "unset TENSILE_DB", i.e. the shipped default.
        explicit ScopedTensileDb(const char* value)
        {
            const char* prior = std::getenv("TENSILE_DB");
            m_had             = (prior != nullptr);
            if(m_had)
                m_saved = prior; // copy BEFORE setenv invalidates the pointer

            if(value)
                setenv("TENSILE_DB", value, /*overwrite=*/1);
            else
                unsetenv("TENSILE_DB");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedTensileDb()
        {
            if(m_had)
                setenv("TENSILE_DB", m_saved.c_str(), /*overwrite=*/1);
            else
                unsetenv("TENSILE_DB");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ScopedTensileDb(ScopedTensileDb const&)            = delete;
        ScopedTensileDb& operator=(ScopedTensileDb const&) = delete;

    private:
        bool        m_had = false;
        std::string m_saved;
    };
} // namespace

// ---------------------------------------------------------------------------
// No-drift contract: the snapshot fields equal what the individual production
// helpers report, so the summary reflects the REAL decisions (not a re-derivation
// that could drift). This is the property that makes the summary trustworthy.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, SnapshotMatchesHelpersForDynamicPartialTiles)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // SK4 = unconditionally dynamic (tree, work-queue)

    // 4096x4224 -> tiles = 32*33 = 1056; grid = min(1056, 256) = 256; 1056 % 256
    // != 0 -> partial tiles -> partials workspace required.
    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto d = solution.computeStreamKDecisions(problem, env.device);

    const size_t tiles = problem.getNumTiles(solution.sizeMapping, 1);
    const auto   red   = solution.getSKReduction(problem, env.device);
    const size_t grid  = solution.getSKGrid(problem, env.device, tiles, red);

    EXPECT_EQ(d.streamKMode, 4);
    EXPECT_TRUE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.tiles, tiles);
    EXPECT_EQ(d.tiles, 1056u) << "4096/128 * 4224/128 = 32 * 33";
    // getSKGrid() reproduces the pre-fallback grid the snapshot records.
    EXPECT_EQ(d.skGridPreFallback, grid);
    EXPECT_EQ(d.skGridPreFallback, 256u) << "min(tiles, cuCount * CUOccupancy) = min(1056, 256)";
    ASSERT_NE(tiles % grid, 0u) << "test needs partial tiles";

    // No fallback fires here, so selected == pre-fallback == final launch grid.
    EXPECT_EQ(d.selectedGrid, d.skGridPreFallback);
    EXPECT_EQ(d.finalGrid, d.skGridPreFallback);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_FALSE(d.workspaceDPFallbackFired);
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.fixedGridUsed);

    // The authoritative workspace query must agree with the snapshot byte-for-byte.
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, env.device));
    EXPECT_GT(d.requiredWorkspaceBytes, 0u);
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.numQueues, 8u) << "gfx950 bakes 8 per-XCD work queues (NUM_XCD)";
}

// ---------------------------------------------------------------------------
// SK5 resolves to the dynamic (SK4) sub-path when the API mode is ON, with tree
// reduction; SK5 OFF stays static (SK3) with no work-queue.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk5OnResolvesDynamicTree)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 5);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(1); // ON

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.streamKMode, 5);
    EXPECT_TRUE(d.effectiveDynamic);
    EXPECT_TRUE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.numQueues, 8u);
    // With plenty of workspace the partials block is taken (no DP fallback).
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_FALSE(d.dpOnly);
}

TEST(StreamKLaunchSummaryTest, Sk5OffResolvesStaticSk3)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 5);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(0); // OFF (static, smCountTarget=0)

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.streamKMode, 5);
    EXPECT_FALSE(d.effectiveDynamic);
    EXPECT_FALSE(d.isDynamic) << "SK5-OFF must take the static (SK3) sub-path";
    EXPECT_EQ(d.numQueues, 8u); // baked count still reported (informational)
    // SK3-static: no per-XCD work-queue region in the workspace it reserves.
    ASSERT_TRUE(d.workspaceAllocated)
        << "scenario must actually reserve a workspace, otherwise the sizing check below "
           "would silently assert nothing";
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.partialTileSize(d.skGrid))
        << "static SK3 workspace = partialTileSize(grid), no work-queue region";
}

// ---------------------------------------------------------------------------
// SK3 static with partial tiles: partials present, skTiles>0, workspace>0,
// not dynamic, not DP-only.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3StaticPartialTilesReserveWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);

    // 4096x4224 -> tiles = 1056; grid = cuCount = 64; 1056 % 64 == 32 (!=0).
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    EXPECT_EQ(d.streamKMode, 3);
    EXPECT_FALSE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "test needs partial tiles";
    EXPECT_TRUE(d.partialsPresent);
    EXPECT_GT(d.skTiles, 0u);
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.numQueues, 0u) << "static SK3 mock device has no analytical work-queue count";
}

// ---------------------------------------------------------------------------
// Force-DP-only (SK3): every tile stays data-parallel. skTiles==0, no partials,
// workspace==0, dpOnly reported and sourced from the compile-time PARAM.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, ForceDpOnlyIsDpOnlyNoWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.streamKForceDPOnly = 1;

    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    EXPECT_TRUE(d.forceDPOnly);
    EXPECT_TRUE(d.dpOnly);
    // dp-only distinction: this is the PARAM source, NOT the runtime fallback.
    EXPECT_FALSE(d.workspaceDPFallbackFired) << "forceDPOnly is a param, not a runtime fallback";
    EXPECT_FALSE(d.streamKDP);
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_FALSE(d.partialsPresent);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // Force-DP-only launches on exactly the hardware CU count (mirrors
    // StreamKForceDPOnlyTest.UsesHardwareCuCount). With clusterDim == 1 the param
    // alone does not reset the grid to tiles -- only the cluster-multicast clamp
    // (clusterDim.x*clusterDim.y > 1) and the runtime workspace-DP fallback do --
    // so finalGrid == selectedGrid here.
    EXPECT_EQ(d.skGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(d.finalGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(d.finalGrid, d.selectedGrid);
}

// ---------------------------------------------------------------------------
// Workspace-starved SK4 falls back to a DP grid (grid=tiles, tree reduction):
// workspaceDPFallbackFired and dpOnly set, nothing reserved. This is the RUNTIME
// dp-only source, and it is the fallback that turns selectedGrid into finalGrid.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, WorkspaceDpFallbackFires)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(0); // no workspace at all -> must fall back to DP

    auto d = solution.computeStreamKDecisions(problem, env.device);

    ASSERT_NE(d.tiles % d.skGridPreFallback, 0u) << "test needs partial tiles pre-fallback";
    EXPECT_GT(d.idealWorkspaceBytes, 0u) << "the launch wanted a partials region";
    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_TRUE(d.dpOnly);
    // dp-only distinction: RUNTIME fallback, not the compile-time param.
    EXPECT_FALSE(d.forceDPOnly);
    // selected vs final: the fallback resets the grid to tiles.
    EXPECT_EQ(d.selectedGrid, d.skGridPreFallback) << "no tree-bounds fallback here";
    EXPECT_NE(d.selectedGrid, d.finalGrid) << "workspace-DP fallback changed the grid";
    EXPECT_EQ(d.finalGrid, d.tiles) << "DP fallback sets grid = tiles";
    EXPECT_EQ(d.skGrid, d.tiles);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // requiredWorkspaceSize agrees: nothing reserved when the workspace is too small.
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);
}

// ---------------------------------------------------------------------------
// Selected vs final grid + which-fallback attribution, exercised directly:
// the workspace-DP fallback is the mechanism that makes finalGrid != selectedGrid.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, SelectedVsFinalGridAttribution)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4);

    // Enough tiles for a real StreamK grid, but zero workspace forces the DP
    // fallback that overwrites the selected grid with tiles.
    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(0);

    auto d = solution.computeStreamKDecisions(problem, env.device);

    // "selected" = what StreamK wanted (CU/config), "final" = what launched.
    EXPECT_GT(d.selectedGrid, 0u);
    EXPECT_EQ(d.finalGrid, d.tiles);
    EXPECT_NE(d.selectedGrid, d.finalGrid);
    // The workspace-DP fallback is the flag responsible for the change here, and
    // the fixed-grid override did not fire.
    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.clusterDPGridClamped);
}

// ---------------------------------------------------------------------------
// DP-only source disambiguation: the snapshot distinguishes the compile-time
// PARAM (forceDPOnly) from the RUNTIME workspace-insufficient fallback. Both
// yield dpOnly, but only the runtime path sets workspaceDPFallbackFired and
// resets finalGrid to tiles.
// (The third source, the TENSILE_STREAMK_DATA_PARALLEL debug flag, is not
// toggleable in-process here: Debug caches it at construction and
// reloadDebugBitsForTest() intentionally does not refresh it. Its plumbing is
// still asserted-absent below via d.streamKDP.)
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, DpOnlySourceDistinguishesParamVsRuntime)
{
    AnalyticalEnv env;

    // (1) PARAM source: forceDPOnly on an SK4 solution with ample workspace.
    ContractionSolution paramSol;
    initStreamKSolution(paramSol, 4);
    paramSol.sizeMapping.streamKForceDPOnly = 1;
    auto paramProblem = makeGemmProblem(4096, 4224, 64);
    paramProblem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    auto pd = paramSol.computeStreamKDecisions(paramProblem, env.device);

    EXPECT_TRUE(pd.dpOnly);
    EXPECT_TRUE(pd.forceDPOnly);
    EXPECT_FALSE(pd.workspaceDPFallbackFired);
    EXPECT_FALSE(pd.streamKDP);
    // forceDPOnly does NOT reset the grid to tiles.
    EXPECT_EQ(pd.finalGrid, pd.selectedGrid);
    EXPECT_NE(pd.finalGrid, pd.tiles);

    // (2) RUNTIME source: no workspace forces the DP fallback (grid=tiles).
    ContractionSolution runSol;
    initStreamKSolution(runSol, 4);
    auto runProblem = makeGemmProblem(4096, 4224, 64);
    runProblem.setWorkspaceSize(0);
    auto rd = runSol.computeStreamKDecisions(runProblem, env.device);

    EXPECT_TRUE(rd.dpOnly);
    EXPECT_FALSE(rd.forceDPOnly);
    EXPECT_TRUE(rd.workspaceDPFallbackFired);
    EXPECT_EQ(rd.finalGrid, rd.tiles);
}

// ---------------------------------------------------------------------------
// Dynamic-path partials-workspace reservation rule.
//
// The dynamic (SK4 / SK5-dynamic) path reserves the partials workspace under the
// same guard the static path uses -- reduction==parallel OR tiles%grid!=0 -- and
// the dynamic path is always tree reduction, so divisibility alone decides. The
// reservation is independent of dynamicPartialsSlots (skTiles*skSplit). The three
// tests below pin that by varying dynamicPartialsSlots (0 vs >0) against tiles%grid
// (==0 vs !=0); the fourth combination (slots>0 and tiles%grid!=0) adds nothing,
// since indivisibility alone already forces the reservation.
//
// All three use a plain mock AMDGPU with no analyticalHardware, so
// streamKBakedQueueCount() is 0 and solve() would reject this device/solution pair
// at its dynamic-queue guard before ever sizing a workspace. Calling
// computeStreamKDecisions() directly is what lets the sizing rule be tested in
// isolation from that rejection.
// ---------------------------------------------------------------------------

// Case A: dynamicSlots == 0 (no split stream-k tiles) AND tiles % grid != 0.
// Workspace IS reserved (because tiles%grid!=0) even though the dynamic packing
// produced no partial tiles.
TEST(StreamKLaunchSummaryTest, DynamicNoSlotsButIndivisible_ReservesWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4224 -> tiles = 1056; grid = cuCount = 64; 1056 % 64 == 32 (!=0).
    // No skTiles override -> the dynamic packing yields skTiles == 0.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "case needs tiles % grid != 0";
    EXPECT_EQ(d.skTiles, 0u) << "no override -> dynamic packing produces no split tiles";
    EXPECT_EQ(d.dynamicPartialsSlots, 0u) << "skTiles*skSplit == 0";

    // Workspace is reserved because tiles%grid != 0, independent of the zero
    // dynamic slot count.
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_GT(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// Case B (complement): dynamicSlots == 0 AND tiles % grid == 0.
// No workspace reserved.
TEST(StreamKLaunchSummaryTest, DynamicNoSlotsAndDivisible_NoWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4096 -> tiles = 1024; grid = cuCount = 64; 1024 % 64 == 0.
    auto problem = makeGemmProblem(4096, 4096, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_EQ(d.tiles % d.skGrid, 0u) << "case needs tiles % grid == 0";
    EXPECT_EQ(d.dynamicPartialsSlots, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// Case C: dynamicSlots > 0 (skTiles override) BUT tiles % grid == 0.
// No workspace is reserved, because tiles%grid==0 gates the partials reservation
// independent of the positive dynamic slot count.
TEST(StreamKLaunchSummaryTest, DynamicSlotsPositiveButDivisible_NoWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4096 -> tiles = 1024; grid = cuCount = 64; 1024 % 64 == 0, so ONLY the
    // skTiles override creates partial (split) tiles.
    auto problem = makeGemmProblem(4096, 4096, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;
    device.skTiles       = 256; // override: number of split stream-k tiles
    device.skSplit       = 4;   // override: k-split factor per tile

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_EQ(d.tiles % d.skGrid, 0u) << "case needs tiles % grid == 0";
    EXPECT_EQ(d.skTiles, 256u);
    EXPECT_GE(d.skSplit, 1u);
    EXPECT_TRUE(d.partialsPresent) << "override produced split stream-k tiles";
    EXPECT_GT(d.dynamicPartialsSlots, 0u) << "skTiles*skSplit > 0";
    // totalItems = (tiles - skTiles) + skTiles*skSplit
    //            = (1024 - 256) + 256*4 = 1792.
    EXPECT_EQ(d.skSplit, 4u) << "itersPerTile=8, skSplit override 4 -> 4 work items per tile";
    EXPECT_EQ(d.totalItems, 1792u);

    // No workspace reserved despite dynamicSlots>0, because tiles%grid==0 gates
    // the partials reservation. requiredWorkspaceSize agrees (also 0).
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// ---------------------------------------------------------------------------
// Non-StreamK solutions produce an inert (mode==0) snapshot.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, NonStreamKProducesInertSnapshot)
{
    ContractionSolution solution; // streamK defaults to 0
    auto                problem = makeGemmProblem(512, 512, 512);
    auto                device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto d = solution.computeStreamKDecisions(problem, device);
    EXPECT_EQ(d.streamKMode, 0);
    EXPECT_FALSE(d.isDynamic);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.skGrid, 0u);
    EXPECT_EQ(d.finalGrid, 0u);
    EXPECT_EQ(d.selectedGrid, 0u);
}

// ---------------------------------------------------------------------------
// The printed summary is well-formed and reports the key fields, including the
// selected-vs-final grid attribution. Exercises the formatting path.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryEmitsFields)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_kernel";
    initStreamKSolution(solution, 4);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto               d = solution.computeStreamKDecisions(problem, env.device);
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    // Deeply-indented multi-line labeled block. The leading "LAUNCH SUMMARY" token
    // and the kernel name are emitted verbatim on the first line; the remaining
    // fields are aligned "key = value" pairs (whitespace-collapsed for matching).
    EXPECT_NE(line.find("LAUNCH SUMMARY"), std::string::npos);
    EXPECT_NE(line.find("test_streamk_kernel"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    // SK4 is unconditionally dynamic -> mode line reports it and the work-queue
    // line carries the real per-XCD counts (not NA).
    EXPECT_NE(line.find("isDynamic = yes"), std::string::npos);
    EXPECT_NE(line.find("selected = "), std::string::npos);
    EXPECT_NE(line.find("final = "), std::string::npos);
    EXPECT_NE(line.find("changedBy = "), std::string::npos);
    EXPECT_NE(line.find("source = "), std::string::npos);
    EXPECT_NE(line.find("numQueues(NUM_XCD) = 8"), std::string::npos);
    // Section headers are present on their own lines (multi-line block).
    EXPECT_NE(line.find("mode:"), std::string::npos);
    EXPECT_NE(line.find("grid:"), std::string::npos);
    EXPECT_NE(line.find("work-queue:"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The printed summary attributes the grid change to the workspace-DP fallback
// when it fires (selected vs final are both reported and differ).
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryReportsFallbackGridChange)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_fallback";
    initStreamKSolution(solution, 4);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(0); // force DP fallback

    auto               d = solution.computeStreamKDecisions(problem, env.device);
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = workspaceDP"), std::string::npos);
    EXPECT_NE(line.find("source = workspaceDP(runtime)"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = yes"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Work-queue fields are per-XCD dynamic-path only. On the SK5-static (SK3)
// sub-path (isDynamic == false) the summary must print the work-queue line as
// "NA (work-queues not used)" instead of a misleading numQueues value, while
// still reporting all the StreamK-wide fields (mode/grid/tiles/workspace).
// This is display-only: the struct still carries d.numQueues.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryNaWorkQueueWhenNotDynamic)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_static";
    initStreamKSolution(solution, 5);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(0); // OFF -> static (SK3), not dynamic

    auto d = solution.computeStreamKDecisions(problem, env.device);
    ASSERT_FALSE(d.isDynamic) << "SK5-OFF must resolve to the static (non-work-queue) path";
    // The struct still holds the baked count; only the DISPLAY is NA'd.
    EXPECT_EQ(d.numQueues, 8u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    // Non-dynamic -> work-queue fields are NA, and the misleading numeric
    // per-XCD field is NOT printed. The "work-queue:" header is on its own line
    // with the NA note indented beneath it.
    EXPECT_NE(line.find("isDynamic = no"), std::string::npos);
    EXPECT_NE(line.find("work-queue:"), std::string::npos);
    EXPECT_NE(line.find("NA (work-queues not used)"), std::string::npos);
    EXPECT_EQ(line.find("numQueues(NUM_XCD)"), std::string::npos)
        << "static path must not print a per-XCD work-queue count";
    EXPECT_EQ(line.find("dynamicPartialsSlots"), std::string::npos)
        << "dynamicPartialsSlots is a dynamic-path-only field";
    // StreamK-wide fields are still reported (not NA'd).
    EXPECT_NE(line.find("reduction = "), std::string::npos);
    EXPECT_NE(line.find("selected = "), std::string::npos);
    EXPECT_NE(line.find("tiles:"), std::string::npos);
    EXPECT_NE(line.find("workspace:"), std::string::npos);
    EXPECT_NE(line.find("fallbacks:"), std::string::npos);
}

// ---------------------------------------------------------------------------
// StreamKForceDPOnly cluster-multicast grid clamp (gfx1250 ClusterDim path).
//
// This clamp is the third "reset the grid to tiles" clamp inside getSKGridImpl,
// running AFTER the tree-fixup-bounds fallback. It fires on the conjunction of
// three SOLUTION-side properties -- SK3, streamKForceDPOnly, and a cluster whose
// x*y extent exceeds 1 -- so the launch runs exactly one work-group per output
// tile (a multicast broadcast, not a K-split). It consults no hardware field, so
// it is reproducible on the same mock AMDGPU the other static-SK3 tests use; no
// gfx1250 device and no analytical hardware are required. streamKForceDPOnly
// also short-circuits getSKReduction() to tree, which keeps the scenario fully
// deterministic.
//
// The tests below pin the resulting REPORT behaviour: the clamp is visible in
// the snapshot, it is what makes selectedGrid differ from finalGrid, and it wins
// the changedBy attribution over the earlier clamps it supersedes.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, ClusterDpMulticastClampsGridToTiles)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.streamKForceDPOnly = 1;
    // ClusterDim.x * ClusterDim.y == 4 > 1 -> multicast cluster launch.
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);

    // 4096x4224 -> tiles = 32*33 = 1056; the CU-count grid the selection logic
    // picks is 64, so the clamp is observable (64 != 1056).
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_EQ(d.streamKMode, 3);
    ASSERT_TRUE(d.forceDPOnly);

    // The clamp fired, and it is the thing that moved the grid.
    EXPECT_TRUE(d.clusterDPGridClamped);
    EXPECT_EQ(d.selectedGrid, static_cast<size_t>(_CPX_CU))
        << "selection still picks the CU-count grid before the clamp";
    EXPECT_NE(d.selectedGrid, d.finalGrid) << "cluster-DP clamp changed the grid";
    EXPECT_EQ(d.finalGrid, d.tiles) << "cluster multicast launches one work-group per tile";
    EXPECT_EQ(d.skGrid, d.finalGrid);
    // The clamp lives inside getSKGridImpl, so it is already reflected in the
    // pre-(workspace)-fallback grid.
    EXPECT_EQ(d.skGridPreFallback, d.tiles);

    // ...and no OTHER grid-changing fallback is credited for it.
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.workspaceDPFallbackFired)
        << "forceDPOnly suppresses the partials reservation, so no workspace fallback";

    // getSKGrid() -- the same helper solve() calls -- reproduces the clamped grid,
    // so the report is not describing a grid the launch does not use.
    EXPECT_EQ(solution.getSKGrid(problem, device, d.tiles, d.reduction), d.finalGrid);

    // Force-DP-only semantics are unchanged by the clamp: every tile stays whole.
    EXPECT_TRUE(d.dpOnly);
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_FALSE(d.partialsPresent);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// ---------------------------------------------------------------------------
// The clamp is a conjunction, and the report must not claim it on any of the
// near-miss configurations. Each sub-case below flips exactly one of the three
// conditions and asserts the grid is left alone.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, ClusterDpGridClampRequiresSk3ForceDpOnlyAndXyCluster)
{
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    // (1) Dynamic mode (SK4) with the same DP-only + cluster settings: the
    // multicast clamp is a static-SK3 construct, so the dynamic grid survives.
    {
        ContractionSolution solution;
        initStreamKSolution(solution, 4);
        solution.sizeMapping.streamKForceDPOnly = 1;
        solution.sizeMapping.clusterDim         = TensileLite::dim3(2, 2, 1);

        auto d = solution.computeStreamKDecisions(problem, device);

        EXPECT_FALSE(d.clusterDPGridClamped) << "SK4 is not the static multicast path";
        EXPECT_EQ(d.finalGrid, d.selectedGrid);
        EXPECT_NE(d.finalGrid, d.tiles);
    }

    // (2) SK3 with a multicast cluster but WITHOUT force-DP-only: the clamp is
    // only sound when every tile is already data-parallel, so it must not fire.
    {
        ContractionSolution solution;
        initStreamKSolution(solution, 3); // leaves streamKForceDPOnly == 0
        solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);

        auto d = solution.computeStreamKDecisions(problem, device);

        ASSERT_FALSE(d.forceDPOnly);
        EXPECT_FALSE(d.clusterDPGridClamped);
        EXPECT_EQ(d.finalGrid, d.selectedGrid);
        EXPECT_NE(d.finalGrid, d.tiles);
    }

    // (3) SK3 + force-DP-only, but the cluster only extends in z. Multicast peers
    // are laid out over the x/y tile grid, so a z-only cluster is not a multicast
    // launch and the grid must be left alone.
    {
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        solution.sizeMapping.streamKForceDPOnly = 1;
        solution.sizeMapping.clusterDim         = TensileLite::dim3(1, 1, 8);

        auto d = solution.computeStreamKDecisions(problem, device);

        EXPECT_FALSE(d.clusterDPGridClamped) << "clusterDim.z does not make a multicast cluster";
        EXPECT_EQ(d.finalGrid, d.selectedGrid);
        EXPECT_EQ(d.finalGrid, static_cast<size_t>(_CPX_CU));
    }
}

// ---------------------------------------------------------------------------
// The printed summary reports the cluster-multicast clamp: it is named in the
// fallbacks section, it is credited for the selected-vs-final grid change, and
// the preFallback grid is surfaced because the clamp moved the grid inside
// getSKGridImpl. The complementary run (no multicast cluster) prints the
// negative form and attributes nothing.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryReportsClusterDpMulticastClamp)
{
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    // (1) Multicast cluster -> clamp fires and is reported.
    ContractionSolution clustered;
    clustered.kernelName = "test_streamk_cluster_dp";
    initStreamKSolution(clustered, 3);
    clustered.sizeMapping.streamKForceDPOnly = 1;
    clustered.sizeMapping.clusterDim         = TensileLite::dim3(2, 2, 1);

    auto dc = clustered.computeStreamKDecisions(problem, device);
    ASSERT_TRUE(dc.clusterDPGridClamped) << "scenario must actually trip the clamp";

    std::ostringstream osc;
    clustered.printStreamKLaunchSummary(osc, problem, dc);
    const std::string clusteredLine = collapseSpaces(osc.str());

    EXPECT_NE(clusteredLine.find("clusterDPMulticast = yes"), std::string::npos);
    EXPECT_NE(clusteredLine.find("changedBy = clusterDPMulticast"), std::string::npos);
    // The other fallbacks are reported as not-fired, so the clamp is unambiguous.
    EXPECT_NE(clusteredLine.find("fixedGrid = no"), std::string::npos);
    EXPECT_NE(clusteredLine.find("workspaceDPFallback = no"), std::string::npos);
    EXPECT_NE(clusteredLine.find("treeBoundsFallback = no"), std::string::npos);
    // selected != final, and the intra-getSKGridImpl clamp surfaces preFallback.
    EXPECT_NE(clusteredLine.find("selected = " + std::to_string(dc.selectedGrid)),
              std::string::npos);
    EXPECT_NE(clusteredLine.find("final = " + std::to_string(dc.tiles)), std::string::npos);
    EXPECT_NE(clusteredLine.find("preFallback = " + std::to_string(dc.tiles)), std::string::npos);
    // dp-only is still sourced from the param, not from a runtime fallback.
    EXPECT_NE(clusteredLine.find("source = forceDPOnly(param)"), std::string::npos);

    // (2) Same solution without a multicast cluster -> negative form, and the
    // grid is not attributed to anything.
    ContractionSolution plain;
    plain.kernelName = "test_streamk_no_cluster";
    initStreamKSolution(plain, 3);
    plain.sizeMapping.streamKForceDPOnly = 1; // clusterDim stays {1, 1, 1}

    auto dp = plain.computeStreamKDecisions(problem, device);
    ASSERT_FALSE(dp.clusterDPGridClamped);

    std::ostringstream osp;
    plain.printStreamKLaunchSummary(osp, problem, dp);
    const std::string plainLine = collapseSpaces(osp.str());

    EXPECT_NE(plainLine.find("clusterDPMulticast = no"), std::string::npos);
    EXPECT_NE(plainLine.find("changedBy = none"), std::string::npos);
    EXPECT_EQ(plainLine.find("changedBy = clusterDPMulticast"), std::string::npos);
    // Nothing moved the grid, so preFallback is suppressed.
    EXPECT_EQ(plainLine.find("preFallback"), std::string::npos);
}

// ---------------------------------------------------------------------------
// changedBy attribution names the clamp that actually produced the launch grid,
// not merely the first one that fired. The skFixedGrid override picks 32 and the
// cluster-multicast clamp then overwrites it with tiles, so 32 is NOT the
// launched grid and crediting "fixedGrid" would be a lie.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, ClusterDpClampWinsAttributionOverFixedGrid)
{
    ContractionSolution solution;
    solution.kernelName = "test_streamk_cluster_over_fixed";
    initStreamKSolution(solution, 3);
    solution.sizeMapping.streamKForceDPOnly = 1;
    solution.sizeMapping.clusterDim         = TensileLite::dim3(2, 2, 1);

    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;
    device.skFixedGrid   = 32;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.fixedGridUsed);
    ASSERT_TRUE(d.clusterDPGridClamped);
    EXPECT_EQ(d.selectedGrid, 32u) << "the override is what selection produced";
    EXPECT_EQ(d.finalGrid, d.tiles) << "but the cluster clamp produced the launch grid";

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = clusterDPMulticast"), std::string::npos);
    EXPECT_EQ(line.find("changedBy = fixedGrid"), std::string::npos)
        << "the fixed grid was overwritten; it did not produce the launch grid";
    // Both are still reported as having fired in the fallbacks section.
    EXPECT_NE(line.find("fixedGrid = yes"), std::string::npos);
    EXPECT_NE(line.find("clusterDPMulticast = yes"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The parallel (split-K data-parallel) reduction path.
//
// Every other test in this file runs a tree-reduction launch. getSKReduction()
// delegates to origami's select_reduction() whenever the device's skDynamicGrid is
// k_split_aware (== 6, the AMDGPU default), and select_reduction returns parallel
// only on the narrow band "tiles < cuCount && itersPerTile >= 64 &&
// tiles <= cuCount/4". 256x4096x4096 with a 128x128x64 macro tile lands exactly
// there on the 256-CU analytical mock: tiles = 2*32 = 64 == cuCount/4, and
// itersPerTile = 4096/64 = 64. SK4 and SK5-dynamic are unconditionally tree, so
// this has to be SK3.
//
// Parallel is the one reduction that reserves a partials workspace unconditionally
// (the tree path only reserves when tiles % grid != 0). The snapshot sizes that
// reservation with partialTileSize(finalGrid), while the caller-facing
// requiredWorkspaceSize() sizes it with requiredWorkspaceSizeGsu(problem, hw,
// grid/tiles) -- two different formulas for the same region. They agree here
// because the parallel grid is an exact multiple of tiles (grid == tiles*skSplit
// == 64*4), which makes requiredWorkspaceSizeGsu's tiles*gsu equal
// partialTileSize's grid. They do NOT agree when the parallel grid comes back
// equal to tiles; see the comment on StreamKDecisions::requiredWorkspaceBytes.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelReductionReservesPartialsWorkspace)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel";
    initStreamKSolution(solution, 3); // SK3 static; SK4 is unconditionally tree

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    // TENSILE_STREAMK_DATA_PARALLEL is latched when Debug is constructed and is not
    // among the variables reloadDebugBitsForTest() refreshes, so it can only be
    // cleared before the process starts. Fail loudly rather than silently asserting
    // something else if it is set.
    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    // Anti-vacuity: the whole point of this test is the parallel branch.
    ASSERT_EQ(d.reduction, origami::reduction_t::parallel)
        << "scenario must actually select parallel reduction, otherwise every "
           "assertion below is about the already-covered tree path";
    // ...and it came from the real helper, not from a special case in the snapshot.
    EXPECT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel);

    EXPECT_EQ(d.streamKMode, 3);
    EXPECT_FALSE(d.isDynamic) << "isDynamic implies tree reduction";
    EXPECT_EQ(d.tiles, 64u) << "256/128 * 4096/128 = 2 * 32";
    EXPECT_EQ(problem.getItersPerTile(solution.sizeMapping), 64u)
        << "4096/64; select_reduction needs itersPerTile >= 64";

    // No clamp fires: selected == preFallback == final.
    EXPECT_EQ(d.selectedGrid, 256u);
    EXPECT_EQ(d.skGridPreFallback, 256u);
    EXPECT_EQ(d.finalGrid, 256u);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, env.device, d.tiles, d.reduction), d.finalGrid);
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.treeBoundsFallbackFired) << "tree-bounds fixup is tree-reduction only";
    EXPECT_FALSE(d.clusterDPGridClamped);
    EXPECT_FALSE(d.workspaceDPFallbackFired);

    // Parallel packing: skSplit = grid/tiles, and skTiles mirrors skSplit.
    EXPECT_EQ(d.skSplit, 4u) << "grid/tiles = 256/64";
    EXPECT_EQ(d.skTiles, d.skSplit) << "the parallel path packs skTiles = skSplit";
    EXPECT_EQ(d.totalItems, d.tiles);
    EXPECT_TRUE(d.partialsPresent);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_FALSE(d.forceDPOnly);
    EXPECT_FALSE(d.streamKDP);

    // Parallel reduction always reserves partials, sized by the FINAL grid.
    ASSERT_TRUE(d.workspaceAllocated)
        << "parallel reduction must reserve a partials region, otherwise the sizing "
           "checks below assert nothing";
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.partialTileSize(d.finalGrid));
    EXPECT_EQ(d.requiredWorkspaceBytes, 16777216u) << "128*128*4 bytes * 256 work-groups";
    EXPECT_EQ(d.idealWorkspaceBytes, d.requiredWorkspaceBytes) << "it fits, so nothing was trimmed";

    // In this regime -- and only because finalGrid is an exact multiple of tiles --
    // the two independent sizings land on the same byte count.
    EXPECT_EQ(d.finalGrid, d.tiles * d.skSplit) << "why the two sizings coincide here";
    EXPECT_GT(solution.requiredWorkspaceSize(problem, env.device), 0u);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), d.requiredWorkspaceBytes);

    // The report names the reduction and attributes nothing to a fallback.
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("reduction = parallel(DP)"), std::string::npos);
    EXPECT_NE(line.find("changedBy = none"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos) << "nothing moved the grid";
    EXPECT_NE(line.find("isDynamic = no"), std::string::npos);
    EXPECT_NE(line.find("NA (work-queues not used)"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The parallel branch's workspace-DP fallback.
//
// Same shape as the test above, but with no workspace at all. getSKReduction()
// still says parallel -- it never looks at the workspace -- while the snapshot
// demotes the launch to tree with grid = tiles because the partials region does
// not fit. This is the parallel-side complement of WorkspaceDpFallbackFires
// (which covers the tree / indivisible-tiles side), and it is also a second
// demonstration that changedBy is about ATTRIBUTION rather than about the numbers
// differing: with no workspace, origami's grid selection already returns tiles, so
// selectedGrid == finalGrid while the fallback is still correctly credited.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelReductionWorkspaceStarvedFallsBackToTree)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel_starved";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(0); // no workspace at all

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    // Anti-vacuity: the pre-fallback reduction really is parallel. (getSKReduction
    // is workspace-independent, so it reports what the snapshot started from.)
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel)
        << "scenario must start from parallel reduction for the fallback to be the "
           "parallel-path fallback";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.reduction, origami::reduction_t::tree) << "the fallback demotes to tree";
    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_TRUE(d.dpOnly);
    EXPECT_FALSE(d.forceDPOnly) << "runtime fallback, not the compile-time param";

    EXPECT_EQ(d.tiles, 64u);
    EXPECT_EQ(d.finalGrid, d.tiles) << "the DP fallback sets grid = tiles";
    // With no workspace the grid selection already returns tiles, so the fallback is
    // credited even though it did not move the number.
    EXPECT_EQ(d.selectedGrid, d.tiles);
    EXPECT_EQ(d.selectedGrid, d.finalGrid);

    EXPECT_EQ(d.idealWorkspaceBytes, solution.partialTileSize(d.tiles))
        << "the launch wanted a partials region sized by the parallel grid";
    EXPECT_EQ(d.idealWorkspaceBytes, 4194304u);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // Both sizings agree that nothing is reserved.
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("changedBy = workspaceDP"), std::string::npos)
        << "attribution is by which clamp produced the launch grid, not by whether "
           "the number changed";
    EXPECT_NE(line.find("source = workspaceDP(runtime)"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The launch summary is OFF unless TENSILE_DB bit 0x200000 is set.
//
// That the diagnostic is opt-in is the feature's central safety claim, so pin it
// directly on the Debug accessor the production gate reads rather than inferring
// it. Three states are checked: TENSILE_DB unset (the shipped default), the
// summary bit, and the neighbouring StreamK mode-selection bit 0x100000 -- which
// must NOT enable the summary.
//
// What this test cannot cover: the production gate is a call site inside
// ContractionSolution::solve() ("if(Debug::Instance().printStreamKLaunchSummary())
// printStreamKLaunchSummary(std::cerr, ...)"), and solve() needs real device input
// pointers and passes a work-queue guard that throws on these mock devices, so a
// host-only unit test cannot reach it. The two host-reachable halves of the claim
// are asserted instead: the predicate the gate reads is false by default, and the
// snapshot builder this suite drives writes nothing to stderr on its own.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, LaunchSummaryDebugBitIsOffByDefault)
{
    // Whatever TENSILE_DB the suite was launched with, the guards below must put it
    // back. Compared against the ambient value rather than against false, so that
    // running the whole suite under TENSILE_DB=0x200000 to eyeball the summaries
    // does not fail this test.
    const bool ambientSummaryBit = Debug::Instance().printStreamKLaunchSummary();

    // (1) Unset -> off. The compile-time debug mask default is 0, so an unset
    // TENSILE_DB leaves every debug bit clear.
    {
        ScopedTensileDb noDb(nullptr);
        EXPECT_FALSE(Debug::Instance().printStreamKLaunchSummary())
            << "the StreamK launch summary must be opt-in";
    }

    // (2) The documented bit turns it on.
    {
        ScopedTensileDb withBit("0x200000");
        ASSERT_TRUE(Debug::Instance().printStreamKLaunchSummary())
            << "TENSILE_DB=0x200000 must enable the launch summary; if this fails the "
               "rest of the test proves nothing about the bit being distinct";
    }

    // (3) The neighbouring StreamK bit does NOT turn it on -- they are distinct
    // opt-ins, so enabling mode-selection tracing does not also spam launch
    // summaries.
    {
        ScopedTensileDb siblingBit("0x100000");
        EXPECT_TRUE(Debug::Instance().printStreamKModeSelection());
        EXPECT_FALSE(Debug::Instance().printStreamKLaunchSummary());
    }

    // (4) State is restored: the guards above must not leak the bit into later
    // tests sharing this process-wide singleton.
    EXPECT_EQ(Debug::Instance().printStreamKLaunchSummary(), ambientSummaryBit)
        << "ScopedTensileDb must restore the prior TENSILE_DB value and reload";

    // (5) Building the snapshot is silent -- nothing reaches stderr unless the
    // caller explicitly prints.
    {
        ScopedTensileDb noDb(nullptr);

        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        auto problem = makeGemmProblem(4096, 4224, 512);
        problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
        auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
        device.skDynamicGrid = 0;

        std::ostringstream    captured;
        std::streambuf* const savedCerr = std::cerr.rdbuf(captured.rdbuf());
        auto                  d         = solution.computeStreamKDecisions(problem, device);
        std::cerr.rdbuf(savedCerr);

        EXPECT_TRUE(captured.str().empty())
            << "computeStreamKDecisions() must not write to stderr; got: " << captured.str();

        // ...and the printer is not self-gating: the bit gates the CALL SITE, so an
        // explicit call still emits. That is what lets every other test in this file
        // exercise the report with the bit clear.
        std::ostringstream os;
        solution.printStreamKLaunchSummary(os, problem, d);
        EXPECT_NE(os.str().find("LAUNCH SUMMARY"), std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// changedBy == treeBounds, as the WINNER of the attribution.
//
// The tree fixup indexes with 24-bit divide/remainder arithmetic, so the grid
// selection resets the grid to tiles when itersPerTile >= 65536, itersPerWG >=
// 65536, or tiles*itersPerTile >= 2^24. Existing tests only ever assert that this
// fallback did NOT fire; this one makes it the credited clamp.
//
// Shape: 1152x131072x131072 with a 128x128x64 macro tile.
//   tiles              = ceil(1152/128) * ceil(131072/128) = 9 * 1024 = 9216
//   itersPerTile       = ceil(131072/64)                              = 2048
//   tiles*itersPerTile = 18874368 >= 16777216  -> bound tripped
// (itersPerWG = 9216*2048/64 = 294912 also exceeds 65536; on a device with at most
// 256 CUs the two bounds cannot be tripped independently, since tiles*itersPerTile
// >= 2^24 forces itersPerWG >= 2^24/256 == 65536. itersPerTile itself stays under
// 65536, which the test asserts, so the trigger is not that bound.)
//
// n == k == 131072 keeps ldb == n >= k, which the {k, n} B descriptor built by
// makeGemmProblem with strides {1, ldb} requires.
//
// The clamp sets grid = tiles, so tiles % grid == 0 and the later workspace-DP
// fallback cannot fire and steal the attribution -- asserted below rather than
// assumed. A plain mock AMDGPU with skDynamicGrid = 0 is used so the pre-clamp grid
// is exactly the CU count rather than a performance-model output.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, TreeBoundsFallbackWinsGridAttribution)
{
    ContractionSolution solution;
    solution.kernelName = "test_streamk_tree_bounds";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(1152, 131072, 131072);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, device);

    // The arithmetic that trips the bound, spelled out so that a shape change fails
    // here with a legible message instead of silently un-tripping the fallback.
    const size_t itersPerTile = problem.getItersPerTile(solution.sizeMapping);
    ASSERT_EQ(d.tiles, 9216u) << "ceil(1152/128) * ceil(131072/128) = 9 * 1024";
    ASSERT_EQ(itersPerTile, 2048u) << "ceil(131072/64)";
    ASSERT_GE(d.tiles * itersPerTile, 16777216u) << "tiles*itersPerTile must reach 2^24";
    EXPECT_LT(itersPerTile, 65536u) << "itersPerTile alone is not what trips it";

    // Anti-vacuity: the fallback actually fired, and it is the credited one.
    ASSERT_TRUE(d.treeBoundsFallbackFired) << "scenario must actually trip the tree-bounds fixup";
    EXPECT_EQ(d.reduction, origami::reduction_t::tree) << "the fixup is tree-reduction only";
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.clusterDPGridClamped);
    // The clamp sets grid = tiles, which makes tiles % grid == 0, which is exactly
    // why the later workspace-DP fallback cannot fire and take the attribution.
    ASSERT_EQ(d.tiles % d.finalGrid, 0u) << "clamped grid must divide tiles";
    EXPECT_FALSE(d.workspaceDPFallbackFired)
        << "grid == tiles leaves no partial tiles, so nothing to reserve";

    // selected vs final: the clamp lives inside the grid selection, so the
    // pre-fallback grid moved with it.
    EXPECT_EQ(d.selectedGrid, static_cast<size_t>(_CPX_CU))
        << "selection picks the CU-count grid before the fixup";
    EXPECT_EQ(d.skGridPreFallback, d.tiles);
    EXPECT_EQ(d.finalGrid, d.tiles);
    EXPECT_NE(d.selectedGrid, d.finalGrid);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, device, d.tiles, d.reduction), d.finalGrid);

    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.idealWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = treeBounds"), std::string::npos);
    EXPECT_NE(line.find("treeBoundsFallback = yes"), std::string::npos);
    // Unambiguous: no other clamp is credited or reported as fired.
    EXPECT_NE(line.find("fixedGrid = no"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = no"), std::string::npos);
    EXPECT_NE(line.find("clusterDPMulticast = no"), std::string::npos);
    EXPECT_NE(line.find("selected = " + std::to_string(d.selectedGrid)), std::string::npos);
    EXPECT_NE(line.find("final = " + std::to_string(d.tiles)), std::string::npos);
    EXPECT_NE(line.find("preFallback = " + std::to_string(d.tiles)), std::string::npos);

    // Complement: the identical shape with a small K leaves the grid alone, so the
    // fallback is a property of the bound and not of the tile count.
    ContractionSolution small;
    small.kernelName = "test_streamk_tree_bounds_ok";
    initStreamKSolution(small, 3);
    auto smallProblem = makeGemmProblem(1152, 131072, 512);
    smallProblem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto ds = small.computeStreamKDecisions(smallProblem, device);
    EXPECT_LT(ds.tiles * smallProblem.getItersPerTile(small.sizeMapping), 16777216u);
    EXPECT_FALSE(ds.treeBoundsFallbackFired);
    EXPECT_EQ(ds.finalGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(ds.selectedGrid, ds.finalGrid);
}

// ---------------------------------------------------------------------------
// changedBy == fixedGrid, as the WINNER of the attribution.
//
// fixedGridUsed is otherwise only exercised by
// ClusterDpClampWinsAttributionOverFixedGrid, where the override deliberately
// LOSES to a later clamp. Here it is the last clamp standing: SK3, skFixedGrid
// set, clusterDim {1, 1, 1} (so no multicast clamp), K small enough that the tree
// bounds are nowhere near, and skFixedGrid chosen to divide tiles so the
// workspace-DP fallback has nothing to reserve and cannot steal the attribution.
//
// The subtlety this test exists to pin: the selected grid is captured AFTER the
// skFixedGrid override is applied, so selectedGrid == finalGrid == 32 and the
// printed "changedBy = fixedGrid" is NOT derived from selected != final. The
// evidence that the override did something is that the launch grid is 32 rather
// than the CU count the same solution and problem pick without it -- asserted via
// the baseline run at the end.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, FixedGridOverrideWinsGridAttribution)
{
    ContractionSolution solution;
    solution.kernelName = "test_streamk_fixed_grid";
    initStreamKSolution(solution, 3); // clusterDim stays {1, 1, 1}

    // 4096x4224 -> tiles = 32*33 = 1056, and 1056 % 32 == 0, so the fixed grid
    // leaves no partial tiles. K=512 -> itersPerTile = 8, far below every tree bound.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skDynamicGrid = 0;
    device.skFixedGrid   = 32;

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, device);

    // Anti-vacuity: the override actually ran.
    ASSERT_TRUE(d.fixedGridUsed) << "scenario must actually take the skFixedGrid override";
    EXPECT_EQ(d.streamKMode, 3);
    EXPECT_FALSE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.tiles, 1056u);

    // No later clamp supersedes it -- this is what makes fixedGrid the winner.
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.clusterDPGridClamped);
    ASSERT_EQ(d.tiles % d.finalGrid, 0u)
        << "the fixed grid must divide tiles, otherwise the workspace-DP fallback "
           "would reserve partials and could steal the attribution";
    EXPECT_FALSE(d.workspaceDPFallbackFired);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.idealWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
    EXPECT_FALSE(d.dpOnly);

    // The override is applied BEFORE the selected grid is captured, so "selected" is
    // already the overridden value: changedBy is attribution, not a diff.
    EXPECT_EQ(d.selectedGrid, 32u) << "selection reports the overridden grid";
    EXPECT_EQ(d.skGridPreFallback, 32u);
    EXPECT_EQ(d.finalGrid, 32u);
    EXPECT_EQ(d.selectedGrid, d.finalGrid)
        << "the fixed-grid override cannot make selected differ from final";
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, device, d.tiles, d.reduction), 32u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = fixedGrid"), std::string::npos);
    EXPECT_NE(line.find("fixedGrid = yes"), std::string::npos);
    EXPECT_NE(line.find("selected = 32"), std::string::npos);
    EXPECT_NE(line.find("final = 32"), std::string::npos);
    // preFallback is suppressed because no clamp inside the grid selection moved the
    // grid away from what selection produced.
    EXPECT_EQ(line.find("preFallback"), std::string::npos);
    EXPECT_NE(line.find("treeBoundsFallback = no"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = no"), std::string::npos);
    EXPECT_NE(line.find("clusterDPMulticast = no"), std::string::npos);

    // Baseline: same solution and problem, no override. This is the evidence that 32
    // came from skFixedGrid -- without it the launch uses the CU count.
    ContractionSolution baseline;
    baseline.kernelName = "test_streamk_no_fixed_grid";
    initStreamKSolution(baseline, 3);
    auto baseDevice          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    baseDevice.skDynamicGrid = 0; // skFixedGrid stays 0

    auto db = baseline.computeStreamKDecisions(problem, baseDevice);
    EXPECT_FALSE(db.fixedGridUsed);
    EXPECT_EQ(db.selectedGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(db.finalGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_NE(db.finalGrid, d.finalGrid) << "the override is what produced grid 32";

    std::ostringstream osb;
    baseline.printStreamKLaunchSummary(osb, problem, db);
    const std::string baseLine = collapseSpaces(osb.str());
    EXPECT_NE(baseLine.find("changedBy = none"), std::string::npos);
    EXPECT_NE(baseLine.find("fixedGrid = no"), std::string::npos);
}
