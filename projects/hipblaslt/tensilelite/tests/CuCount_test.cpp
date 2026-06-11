// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/streamk.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

// ===========================================================================
// CuCountPredicateTest -- basic CUCountEqual predicate behaviour
// ===========================================================================

TEST(CuCountPredicateTest, MatchesSPX)
{
    auto pred = std::make_shared<Predicates::GPU::CUCountEqual>(_SPX_CU);
    AMDGPU spx = makeDevice(_MI350_CHIP_ID, _SPX_CU, "spx");
    AMDGPU cpx = makeDevice(_MI350_CHIP_ID, _CPX_CU, "cpx");

    EXPECT_TRUE((*pred)(spx))  << "CUCountEqual(256) should match SPX (CU=256)";
    EXPECT_FALSE((*pred)(cpx)) << "CUCountEqual(256) should NOT match CPX (CU=64)";
}

TEST(CuCountPredicateTest, MatchesCPX)
{
    auto pred = std::make_shared<Predicates::GPU::CUCountEqual>(_CPX_CU);
    AMDGPU spx = makeDevice(_MI350_CHIP_ID, _SPX_CU, "spx");
    AMDGPU cpx = makeDevice(_MI350_CHIP_ID, _CPX_CU, "cpx");

    EXPECT_TRUE((*pred)(cpx))  << "CUCountEqual(64) should match CPX (CU=64)";
    EXPECT_FALSE((*pred)(spx)) << "CUCountEqual(64) should NOT match SPX (CU=256)";
}

TEST(CuCountPredicateTest, NoCuCheckMatchesBoth)
{
    // A hardware predicate with no CUCountEqual accepts any CU configuration.
    auto hwPred = makeHwPred(AMDGPU::Processor::gfx950, _MI350_CHIP_ID);
    AMDGPU spx = makeDevice(_MI350_CHIP_ID, _SPX_CU, "spx");
    AMDGPU cpx = makeDevice(_MI350_CHIP_ID, _CPX_CU, "cpx");

    EXPECT_TRUE((*hwPred.value)(spx)) << "Predicate without CU check should match SPX";
    EXPECT_TRUE((*hwPred.value)(cpx)) << "Predicate without CU check should match CPX";
}

// ===========================================================================
// CuCountFallbackTest fixture -- verifies CPX/SPX fallback patterns
// ===========================================================================
class CuCountFallbackTest : public ::testing::Test
{
protected:
    // Mock devices
    AMDGPU mi350spx = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    AMDGPU mi355spx = makeDevice(_MI355_CHIP_ID, _SPX_CU, "mi355spx");
    AMDGPU mi350cpx = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    AMDGPU mi355cpx = makeDevice(_MI355_CHIP_ID, _CPX_CU, "mi355cpx");

    static constexpr auto gfx950 = AMDGPU::Processor::gfx950;

    int nextIdx = 1;

    std::shared_ptr<ContractionSolution> sol(const std::string& name)
    {
        return makeSolution(name, nextIdx++);
    }

    void expectSelected(const ContractionHardwareSelectionLibrary& lib,
                        const AMDGPU&                              device,
                        const std::string&                         expectedName)
    {
        std::string got = selectSolution(lib, device, device.deviceName);
        EXPECT_EQ(got, expectedName)
            << "Device " << device.deviceName
            << " (chip=" << hexChipId(device.pciChipId().value())
            << ", CU=" << device.computeUnitCount
            << "): expected \"" << expectedName << "\", got \"" << got << "\"";
    }
};

// ---------------------------------------------------------------------------
// CPX falls back to SPX oob when no CPX-specific equality exists.
//
// Library has SPX equality + SPX oob only.  CPX devices skip the CU=256 rows
// (CUCountEqual(256) fails) and land on the no-CU catch-all with oob.
// ---------------------------------------------------------------------------
TEST_F(CuCountFallbackTest, CpxFallsBackToSpxOob)
{
    dbg("=== CpxFallsBackToSpxOob ===");

    auto spx_eq  = sol("mi350spx_eq");
    auto spx_oob = sol("mi350spx_oob");

    auto lib = buildHwLib({
        // Row 1: mi350, CU=256 -- SPX equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(spx_eq), singleLib(spx_oob))},

        // Row 2: mi350, any CU -- oob only (catch-all for CPX)
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(spx_oob))},

        // Row 3: gfx950 catch-all
        {makeHwPred(gfx950),
         buildProblemLib(singleLib(spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");
}

// ---------------------------------------------------------------------------
// CPX has its own equality row; CPX devices use it, SPX devices skip it.
// ---------------------------------------------------------------------------
TEST_F(CuCountFallbackTest, CpxWithOwnEq)
{
    dbg("=== CpxWithOwnEq ===");

    auto spx_eq  = sol("mi350spx_eq");
    auto spx_oob = sol("mi350spx_oob");
    auto cpx_eq  = sol("mi350cpx_eq");

    auto lib = buildHwLib({
        // Row 1: mi350, CU=256 -- SPX
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(spx_eq), singleLib(spx_oob))},

        // Row 2: mi350, CU=64 -- CPX
        {makeHwPred(gfx950, _MI350_CHIP_ID, _CPX_CU),
         buildProblemLib(singleLib(cpx_eq), singleLib(spx_oob))},

        // Row 3: mi350, any CU -- oob
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi350cpx, "mi350cpx_eq");
}

// ---------------------------------------------------------------------------
// mi355cpx falls to mi355spx oob (not mi350spx oob) when mi355 oob exists.
//
// Tests that CU-count fallback respects chip ID specificity: mi355cpx's
// no-CU catch-all row is chip-specific to mi355, so it gets mi355 oob.
// ---------------------------------------------------------------------------
TEST_F(CuCountFallbackTest, CpxFallsToSameChipOob)
{
    dbg("=== CpxFallsToSameChipOob ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");
    auto mi355spx_eq  = sol("mi355spx_eq");
    auto mi355spx_oob = sol("mi355spx_oob");

    auto lib = buildHwLib({
        // Row 1: mi355, CU=256
        {makeHwPred(gfx950, _MI355_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi355spx_eq), singleLib(mi355spx_oob))},

        // Row 2: mi350, CU=256
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_eq), singleLib(mi350spx_oob))},

        // Row 3: mi355, any CU -- mi355 oob
        {makeHwPred(gfx950, _MI355_CHIP_ID),
         buildProblemLib(singleLib(mi355spx_oob))},

        // Row 4: mi350, any CU -- mi350 oob
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 5: catch-all
        {makeHwPred(gfx950),
         buildProblemLib(singleLib(mi350spx_oob))},
    });

    // CPX devices skip the CU=256 rows, then hit their chip-specific no-CU row.
    expectSelected(*lib, mi355cpx, "mi355spx_oob");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");

    // SPX devices still get equality.
    expectSelected(*lib, mi355spx, "mi355spx_eq");
    expectSelected(*lib, mi350spx, "mi350spx_eq");
}

// ---------------------------------------------------------------------------
// When both CPX and SPX equality exist for the same chip, each mode selects
// its own equality solution independently.
// ---------------------------------------------------------------------------
TEST_F(CuCountFallbackTest, CpxAndSpxIndependent)
{
    dbg("=== CpxAndSpxIndependent ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");
    auto mi350cpx_eq  = sol("mi350cpx_eq");

    auto lib = buildHwLib({
        // Row 1: mi350, CU=256 -- SPX equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_eq), singleLib(mi350spx_oob))},

        // Row 2: mi350, CU=64 -- CPX equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _CPX_CU),
         buildProblemLib(singleLib(mi350cpx_eq), singleLib(mi350spx_oob))},

        // Row 3: mi350, any CU -- oob
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi350cpx, "mi350cpx_eq");

    // Verify each mode did NOT cross-select.
    // An mi350spx device should not get the CPX solution and vice versa.
    auto problem  = dummyProblem();
    auto spxResult = lib->findBestSolution(problem, mi350spx);
    auto cpxResult = lib->findBestSolution(problem, mi350cpx);

    ASSERT_NE(spxResult, nullptr);
    ASSERT_NE(cpxResult, nullptr);
    EXPECT_NE(spxResult->solutionName, cpxResult->solutionName)
        << "SPX and CPX should select different solutions";
}

TEST(StreamKForceDPOnlyTest, UsesHardwareCuCount)
{
    ContractionSolution solution;
    solution.sizeMapping.streamK               = 3;
    solution.sizeMapping.streamKForceDPOnly     = 1;
    solution.sizeMapping.macroTile             = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU                = 64;
    solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
    solution.sizeMapping.CUOccupancy           = 1;

    auto problem = dummyProblem();
    auto device  = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    auto tiles   = problem.getNumTiles(solution.sizeMapping, 1);

    EXPECT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::tree);
    EXPECT_EQ(solution.getSKGrid(problem, device, tiles, origami::reduction_t::tree), _CPX_CU);
}

TEST(StreamKForceDPOnlyTest, FixedGridOverridesForceDPOnlyGrid)
{
    ContractionSolution solution;
    solution.sizeMapping.streamK               = 3;
    solution.sizeMapping.streamKForceDPOnly     = 1;
    solution.sizeMapping.macroTile             = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU                = 64;
    solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
    solution.sizeMapping.CUOccupancy           = 1;

    auto problem       = dummyProblem();
    auto device        = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.skFixedGrid = 17;
    auto tiles         = problem.getNumTiles(solution.sizeMapping, 1);

    EXPECT_EQ(solution.getSKGrid(problem, device, tiles, origami::reduction_t::tree),
              device.skFixedGrid);
}

TEST(StreamKForceDPOnlyTest, DoesNotRequestPartialWorkspace)
{
    ContractionSolution solution;
    solution.sizeMapping.streamK               = 3;
    solution.sizeMapping.streamKForceDPOnly     = 1;
    solution.sizeMapping.streamKAtomic         = 0;
    solution.sizeMapping.macroTile             = TensileLite::dim3(256, 256, 1);
    solution.sizeMapping.depthU                = 64;
    solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
    solution.sizeMapping.CUOccupancy           = 1;
    solution.sizeMapping.workspaceSizePerElemC = 4;

    auto problem = dummyProblem();
    auto device  = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    auto tiles   = problem.getNumTiles(solution.sizeMapping, 1);

    ASSERT_NE(tiles % _CPX_CU, 0);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0);
}

// ===========================================================================
// StreamK5HybridModeTest -- the SK5 hybrid kernel must size its launch grid
// per the *effective* sub-mode: OFF/static borrows the SK3 grid, ON/dynamic
// borrows the SK4 persistent grid. The grid sizing (getSKGrid) and the
// kernel-arg packing (generateSingleCall) share streamK5EffectiveDynamic as
// the single source of truth; these tests lock that resolution so SK5-off
// can never silently launch the SK4 grid again (the original regression where
// SK5-off matched SK4's grid_size=256 instead of SK3's tile-count grid).
// AUTO (mode 2) routes through origami::streamk::select_hybrid_mode and
// requires HipAMDGPU::analyticalHardware; see the AUTO/smCountTarget tests
// below (mock analytical hardware, no GPU required).
// ===========================================================================

namespace
{
    constexpr size_t kGfx950AnalyticalCuCount = 256;

    origami::hardware_t makeGfx950AnalyticalHardware()
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx950,
                                   kGfx950AnalyticalCuCount,
                                   163840,
                                   8,
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
        device.processor        = AMDGPU::Processor::gfx950;
        device.computeUnitCount = static_cast<int>(hw.N_CU);
        device.deviceName       = "test-gfx950-analytical";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    ContractionSolution makeStreamK5Solution()
    {
        ContractionSolution solution;
        solution.sizeMapping.streamK           = 5;
        solution.sizeMapping.macroTile         = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU            = 64;
        solution.sizeMapping.matrixInstruction = {16, 16, 32, 1};
        solution.sizeMapping.CUOccupancy       = 1;
        return solution;
    }

    ContractionProblemGemm makeGemmProblem(size_t m, size_t n, size_t k)
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, n, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }
} // namespace

TEST(StreamK5HybridModeTest, ProblemParamsDefaultToAuto)
{
    auto problem = dummyProblem();
    EXPECT_EQ(problem.getParams().streamKTileSchedulingMode(), 2)
        << "StreamK=5 hybrid mode should default to AUTO (2)";
    EXPECT_EQ(problem.getParams().smCountTarget(), 0)
        << "smCountTarget should default to 0 (use all device CUs)";
}

TEST(StreamK5HybridModeTest, ProblemParamsRoundTripModeAndSmCountTarget)
{
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    problem.setParams().setSmCountTarget(128);
    EXPECT_EQ(problem.getParams().streamKTileSchedulingMode(), 1);
    EXPECT_EQ(problem.getParams().smCountTarget(), 128);
}

TEST(StreamK5HybridModeTest, TriStateOffResolvesStatic)
{
    ContractionSolution solution;
    solution.sizeMapping.streamK           = 5;
    solution.sizeMapping.macroTile         = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU            = 64;
    solution.sizeMapping.matrixInstruction = {16, 16, 32, 1};
    solution.sizeMapping.CUOccupancy       = 1;

    auto problem = dummyProblem();
    auto device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    problem.setParams().setStreamKTileSchedulingMode(0); // OFF -> static (SK3)
    EXPECT_FALSE(solution.streamK5EffectiveDynamic(problem, device))
        << "StreamK=5 OFF must resolve to the static (SK3) sub-path";
}

TEST(StreamK5HybridModeTest, TriStateOnResolvesDynamic)
{
    ContractionSolution solution;
    solution.sizeMapping.streamK           = 5;
    solution.sizeMapping.macroTile         = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU            = 64;
    solution.sizeMapping.matrixInstruction = {16, 16, 32, 1};
    solution.sizeMapping.CUOccupancy       = 1;

    auto problem = dummyProblem();
    auto device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    problem.setParams().setStreamKTileSchedulingMode(1); // ON -> dynamic (SK4)
    EXPECT_TRUE(solution.streamK5EffectiveDynamic(problem, device))
        << "StreamK=5 ON must resolve to the dynamic (SK4) sub-path";
}

TEST(StreamK5HybridModeTest, TriStateAutoResolvesStaticViaOrigami)
{
    auto solution = makeStreamK5Solution();
    auto hw       = makeGfx950AnalyticalHardware();
    auto device   = makeHipDeviceWithAnalytical(hw);

    // 2560^2 @ MT128 -> 400 tiles; tiles/CU ~= 1.56 < 2.08 threshold -> static.
    auto problem = makeGemmProblem(2560, 2560, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);

    EXPECT_FALSE(solution.streamK5EffectiveDynamic(problem, device))
        << "StreamK=5 AUTO must delegate to origami and pick static for low tiles/CU";
}

TEST(StreamK5HybridModeTest, TriStateAutoResolvesDynamicViaOrigami)
{
    auto solution = makeStreamK5Solution();
    auto hw       = makeGfx950AnalyticalHardware();
    auto device   = makeHipDeviceWithAnalytical(hw);

    // 4096^2 @ MT128 -> 1024 tiles; tiles/CU = 4 > 2.08 threshold -> dynamic.
    auto problem = makeGemmProblem(4096, 4096, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);

    EXPECT_TRUE(solution.streamK5EffectiveDynamic(problem, device))
        << "StreamK=5 AUTO must delegate to origami and pick dynamic for high tiles/CU";
}

TEST(StreamK5HybridModeTest, SmCountTargetForwardsIntoAutoHeuristic)
{
    auto solution = makeStreamK5Solution();
    auto hw       = makeGfx950AnalyticalHardware();
    auto device   = makeHipDeviceWithAnalytical(hw);

    // Borderline static with full CU count; halving effective CUs flips to dynamic.
    auto problem = makeGemmProblem(2560, 2560, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);
    problem.setParams().setSmCountTarget(0);

    EXPECT_FALSE(solution.streamK5EffectiveDynamic(problem, device))
        << "smCountTarget=0 should use full N_CU and keep AUTO static";

    problem.setParams().setSmCountTarget(static_cast<int>(kGfx950AnalyticalCuCount / 2));
    EXPECT_TRUE(solution.streamK5EffectiveDynamic(problem, device))
        << "smCountTarget must clamp available CUs and flip AUTO to dynamic";
}

TEST(StreamK5HybridModeTest, SmCountTargetZeroMatchesFullCuCount)
{
    auto solution = makeStreamK5Solution();
    auto hw       = makeGfx950AnalyticalHardware();
    auto device   = makeHipDeviceWithAnalytical(hw);

    auto problem = makeGemmProblem(4096, 4096, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);

    problem.setParams().setSmCountTarget(0);
    const bool withZero = solution.streamK5EffectiveDynamic(problem, device);

    problem.setParams().setSmCountTarget(static_cast<int>(kGfx950AnalyticalCuCount));
    const bool withFull = solution.streamK5EffectiveDynamic(problem, device);

    EXPECT_EQ(withZero, withFull)
        << "smCountTarget=0 and smCountTarget=N_CU must resolve AUTO identically";
}

// ===========================================================================
// Sk3Sk5OffPartition512Test -- dump and compare host partition state for the
// Equality MT64x64x16 kernel at 512^3 NN on the live device (MI355X/gfx950).
// ===========================================================================

namespace
{
    struct StreamKHostPack
    {
        origami::reduction_t reduction{};
        size_t               grid{};
        size_t               tiles{};
        size_t               itersPerTile{};
        uint32_t             skTiles{};
        uint32_t             skItersPerWG{};
        bool                 effectiveDynamic{};
    };

    void initEquality512Solution(ContractionSolution& solution, int streamK)
    {
        solution.sizeMapping.streamK            = streamK;
        solution.sizeMapping.macroTile          = TensileLite::dim3(64, 64, 1);
        solution.sizeMapping.depthU             = 16;
        solution.sizeMapping.matrixInstruction  = {16, 16, 4, 1};
        solution.sizeMapping.workGroupMapping   = 1;
        solution.sizeMapping.CUOccupancy        = -1;
        solution.sizeMapping.streamKForceDPOnly = 0;
        solution.sizeMapping.streamKAtomic      = 0;
    }

    ContractionProblemGemm make512Problem()
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, 512, 512, 512, 512, 512, 512, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    StreamKHostPack computeStreamKHostPack(ContractionSolution const& solution,
                                           ContractionProblemGemm&    problem,
                                           Hardware const&            hardware)
    {
        StreamKHostPack pack{};
        pack.tiles = problem.getNumTiles(solution.sizeMapping, 1);
        pack.itersPerTile
            = std::max(size_t{1}, problem.getItersPerTile(solution.sizeMapping));

        if(solution.sizeMapping.streamK == 5)
        {
            pack.effectiveDynamic = solution.streamK5EffectiveDynamic(problem, hardware);
            pack.reduction        = pack.effectiveDynamic
                                        ? origami::reduction_t::tree
                                        : solution.getSKReduction(problem, hardware);
        }
        else
        {
            pack.effectiveDynamic = false;
            pack.reduction        = solution.getSKReduction(problem, hardware);
        }

        pack.grid = solution.getSKGrid(problem, hardware, pack.tiles, pack.reduction);

        // Mirror the tree (non-parallel) SK3 arg branch used by both native SK3
        // and the SK5 static sub-path in ContractionSolution.cpp.
        if(pack.reduction == origami::reduction_t::parallel)
        {
            uint32_t skSplit      = static_cast<uint32_t>(pack.grid / pack.tiles);
            pack.skItersPerWG     = static_cast<uint32_t>(pack.itersPerTile) / skSplit;
            pack.skTiles          = skSplit;
        }
        else
        {
            AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
            assert(pAMDGPU != nullptr);
            int  fullTiles   = pAMDGPU->skFullTiles;
            bool bigEnough   = pack.tiles > pack.grid;
            bool forceDPOnly = solution.sizeMapping.streamKForceDPOnly != 0;
            pack.skTiles     = forceDPOnly ? 0u : static_cast<uint32_t>(pack.grid);
            if(!forceDPOnly && pack.tiles % pack.grid != 0)
            {
                pack.skTiles = bigEnough ? pack.grid * fullTiles + pack.tiles % pack.grid
                                         : pack.tiles;
                pack.skTiles = std::min(pack.skTiles, static_cast<uint32_t>(pack.tiles));
            }
            pack.skItersPerWG
                = static_cast<uint32_t>(pack.skTiles) * static_cast<uint32_t>(pack.itersPerTile)
                  / static_cast<uint32_t>(pack.grid);
        }

        return pack;
    }

    void printStreamKHostPack(char const* label, StreamKHostPack const& pack)
    {
        std::cout << label << ": tiles=" << pack.tiles << " itersPerTile=" << pack.itersPerTile
                  << " reduction=" << static_cast<int>(pack.reduction)
                  << " effectiveDynamic=" << (pack.effectiveDynamic ? "true" : "false")
                  << " grid=" << pack.grid << " skTiles=" << pack.skTiles
                  << " SKItersPerWG=" << pack.skItersPerWG << std::endl;
    }
} // namespace

TEST(Sk3Sk5OffPartition512Test, NativeSk3MatchesSk5OffHostPack)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0)
        GTEST_SKIP() << "No HIP device";

    auto hardware = hip::GetCurrentDevice();
    ASSERT_NE(hardware, nullptr);

    auto* amdgpu = dynamic_cast<AMDGPU*>(hardware.get());
    ASSERT_NE(amdgpu, nullptr);

    ContractionSolution sk3Solution;
    ContractionSolution sk5Solution;
    initEquality512Solution(sk3Solution, 3);
    initEquality512Solution(sk5Solution, 5);

    auto problemSk3 = make512Problem();
    auto problemSk5 = make512Problem();
    problemSk5.setParams().setStreamKTileSchedulingMode(0); // SK5-off

    auto sk3Pack = computeStreamKHostPack(sk3Solution, problemSk3, *hardware);
    auto sk5OffPack = computeStreamKHostPack(sk5Solution, problemSk5, *hardware);

    printStreamKHostPack("native SK3", sk3Pack);
    printStreamKHostPack("SK5-off", sk5OffPack);

    EXPECT_FALSE(sk5OffPack.effectiveDynamic);
    EXPECT_EQ(sk3Pack.reduction, sk5OffPack.reduction);
    EXPECT_EQ(sk3Pack.grid, sk5OffPack.grid);
    EXPECT_EQ(sk3Pack.skTiles, sk5OffPack.skTiles);
    EXPECT_EQ(sk3Pack.skItersPerWG, sk5OffPack.skItersPerWG);

    // Contrast: SK5-on (dynamic) should diverge at 512^3 when tiles < grid.
    problemSk5.setParams().setStreamKTileSchedulingMode(1);
    auto sk5OnPack = computeStreamKHostPack(sk5Solution, problemSk5, *hardware);
    printStreamKHostPack("SK5-on (contrast)", sk5OnPack);
    EXPECT_TRUE(sk5OnPack.effectiveDynamic);
    if(sk3Pack.grid > sk3Pack.tiles)
        EXPECT_NE(sk3Pack.grid, sk5OnPack.grid)
            << "512^3 static path oversubscribes; dynamic path should not match";
}
