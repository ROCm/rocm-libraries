// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <limits>
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
    device.skDynamicGrid = 0;
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
    device.skDynamicGrid = 0;
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
    device.skDynamicGrid = 0;
    auto tiles   = problem.getNumTiles(solution.sizeMapping, 1);

    ASSERT_NE(tiles % _CPX_CU, 0);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0);
}

// StreamK5HybridModeTest -- streamK5EffectiveDynamic drives grid sizing and
// host arg packing. OFF (0, default) is static unless smCountTarget()>0;
// AUTO (2) always uses origami::streamk::select_hybrid_mode;
// threshold/smCountTarget cases live in origami/tests/test_streamk.cpp.

namespace
{
    constexpr size_t kGfx950AnalyticalCuCount = 256;

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

    origami::hardware_t makeGfx950AnalyticalHardware()
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx950,
                                   kGfx950AnalyticalCuCount,
                                   163840,
                                   262144,  // rf_capacity: 65536 regs * 4 bytes
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
        device.processor          = AMDGPU::Processor::gfx950;
        device.computeUnitCount   = static_cast<int>(hw.N_CU);
        device.deviceName         = "test-gfx950-analytical";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    hip::HipAMDGPU makeHipDeviceWithoutAnalytical()
    {
        hip::HipAMDGPU device;
        device.processor        = AMDGPU::Processor::gfx950;
        device.computeUnitCount = static_cast<int>(kGfx950AnalyticalCuCount);
        device.deviceName       = "test-gfx950-no-analytical";
        return device;
    }

    void initStreamK5Solution(ContractionSolution& solution)
    {
        solution.sizeMapping.streamK           = 5;
        solution.sizeMapping.macroTile         = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU            = 64;
        solution.sizeMapping.matrixInstruction = {16, 16, 32, 1};
        solution.sizeMapping.CUOccupancy       = 1;
    }

    struct StreamK5AnalyticalEnv
    {
        StreamK5AnalyticalEnv()
            : hw(makeGfx950AnalyticalHardware())
            , device(makeHipDeviceWithAnalytical(hw))
        {
            initStreamK5Solution(solution);
        }

        ContractionSolution solution;
        origami::hardware_t hw;
        hip::HipAMDGPU      device;
    };

    ContractionProblemGemm makeGemmProblem(size_t m, size_t n, size_t k)
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, n, m, 1.0, false, 1);
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
} // namespace

TEST(StreamK5HybridModeTest, ProblemParamsDefaultToOff)
{
    auto problem = dummyProblem();
    EXPECT_EQ(problem.getParams().streamKTileSchedulingMode(), 0)
        << "StreamK=5 hybrid mode should default to OFF (0)";
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

struct StreamK5ExplicitModeParam
{
    int  mode;
    bool expectDynamic;
};

class StreamK5ExplicitModeTest : public ::testing::TestWithParam<StreamK5ExplicitModeParam>
{
};

TEST_P(StreamK5ExplicitModeTest, ResolvesEffectiveSubPath)
{
    auto const& param = GetParam();
    ContractionSolution solution;
    initStreamK5Solution(solution);
    auto        problem  = dummyProblem();
    auto        device   = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    problem.setParams().setStreamKTileSchedulingMode(param.mode);

    EXPECT_EQ(solution.streamK5EffectiveDynamic(problem, device), param.expectDynamic)
        << "StreamK=5 mode " << param.mode << " must resolve to the "
        << (param.expectDynamic ? "dynamic (SK4)" : "static (SK3)") << " sub-path";
}

INSTANTIATE_TEST_SUITE_P(
    StreamK5HybridModeTest,
    StreamK5ExplicitModeTest,
    ::testing::Values(StreamK5ExplicitModeParam{0, false}, StreamK5ExplicitModeParam{1, true}),
    [](::testing::TestParamInfo<StreamK5ExplicitModeParam> const& info) {
        return info.param.mode == 0 ? "OffStatic" : "OnDynamic";
    });

TEST(StreamK5HybridModeTest, TriStateAutoRequiresAnalyticalHardware)
{
    ContractionSolution solution;
    initStreamK5Solution(solution);
    auto device   = makeHipDeviceWithoutAnalytical();
    auto problem  = makeGemmProblem(4096, 4096, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);

    EXPECT_THROW(solution.streamK5EffectiveDynamic(problem, device), std::runtime_error)
        << "StreamK=5 AUTO must assert when analyticalHardware is null";
}

struct StreamK5AutoOrigamiParam
{
    size_t      m;
    size_t      n;
    bool        expectDynamic;
    const char* suffix;
};

class StreamK5AutoOrigamiTest : public ::testing::TestWithParam<StreamK5AutoOrigamiParam>
{
};

TEST_P(StreamK5AutoOrigamiTest, ResolvesViaOrigamiAndHostPack)
{
    auto const& param = GetParam();
    StreamK5AnalyticalEnv env;
    auto                  problem = makeGemmProblem(param.m, param.n, 64);
    problem.setParams().setStreamKTileSchedulingMode(2);
    // A non-zero smCountTarget signals a cotenant sharing the device, which is
    // required for the tiles-per-cu heuristic to engage (see the cotenant gate
    // in origami::streamk::select_hybrid_mode).
    problem.setParams().setSmCountTarget(128);

    EXPECT_EQ(env.solution.streamK5EffectiveDynamic(problem, env.device), param.expectDynamic)
        << "StreamK=5 AUTO " << param.suffix;

    StreamKHostPack pack = computeStreamKHostPack(env.solution, problem, env.device);
    EXPECT_EQ(pack.effectiveDynamic, param.expectDynamic);
    EXPECT_EQ(pack.grid,
              env.solution.getSKGrid(problem, env.device, pack.tiles, pack.reduction));
}

INSTANTIATE_TEST_SUITE_P(
    StreamK5HybridModeTest,
    StreamK5AutoOrigamiTest,
    ::testing::Values(
        StreamK5AutoOrigamiParam{2560, 2560, false, "BelowMinTilesGate"},
        StreamK5AutoOrigamiParam{4096, 4096, true, "AboveMinTilesGate"}),
    [](::testing::TestParamInfo<StreamK5AutoOrigamiParam> const& info) {
        return info.param.suffix;
    });

TEST(StreamK5HybridModeTest, OffWithSmCountTargetEngagesHeuristic)
{
    StreamK5AnalyticalEnv env;

    auto problemOffNoTarget = makeGemmProblem(4096, 4096, 64);
    problemOffNoTarget.setParams().setStreamKTileSchedulingMode(0);
    problemOffNoTarget.setParams().setSmCountTarget(0);
    EXPECT_FALSE(env.solution.streamK5EffectiveDynamic(problemOffNoTarget, env.device))
        << "OFF with smCountTarget=0 must stay on the static (SK3) sub-path";

    auto problemOffWithTarget = makeGemmProblem(4096, 4096, 64);
    problemOffWithTarget.setParams().setStreamKTileSchedulingMode(0);
    problemOffWithTarget.setParams().setSmCountTarget(128);

    auto problemAuto = makeGemmProblem(4096, 4096, 64);
    problemAuto.setParams().setStreamKTileSchedulingMode(2);
    problemAuto.setParams().setSmCountTarget(128);

    EXPECT_EQ(env.solution.streamK5EffectiveDynamic(problemOffWithTarget, env.device),
              env.solution.streamK5EffectiveDynamic(problemAuto, env.device))
        << "OFF + smCountTarget>0 must match AUTO heuristic for the same problem";
    EXPECT_TRUE(env.solution.streamK5EffectiveDynamic(problemOffWithTarget, env.device))
        << "4096x4096 with CU budget should select the dynamic (SK4) sub-path";

    auto problemOffLow = makeGemmProblem(2560, 2560, 64);
    problemOffLow.setParams().setStreamKTileSchedulingMode(0);
    problemOffLow.setParams().setSmCountTarget(128);
    auto problemAutoLow = makeGemmProblem(2560, 2560, 64);
    problemAutoLow.setParams().setStreamKTileSchedulingMode(2);
    problemAutoLow.setParams().setSmCountTarget(128);
    EXPECT_EQ(env.solution.streamK5EffectiveDynamic(problemOffLow, env.device),
              env.solution.streamK5EffectiveDynamic(problemAutoLow, env.device))
        << "OFF + smCountTarget>0 must match AUTO heuristic for the same problem";
}

// smCountTarget heuristic threshold behavior is covered by origami/tests/test_streamk.cpp.

// Guards the smCountTarget() -> origami_problem.num_cus wiring through the
// ContractionSolution path, on both outputs it feeds: getSKReduction (reduction
// strategy) and getSKGrid (grid size).
TEST(StreamKSmCountTargetTest, SmCountTargetChangesReductionAndGrid)
{
    // streamK=3 on the gfx950 analytical device (256 CUs), k_split_aware selector.
    StreamK5AnalyticalEnv env;
    env.solution.sizeMapping.streamK = 3;
    env.device.skDynamicGrid = static_cast<int>(origami::grid_selection_t::k_split_aware);

    // Make smCountTarget the sole grid budget source (AMDGPU defaults, explicit).
    env.device.skFixedGrid      = 0;
    env.device.skMaxCUs         = 0;
    env.device.skGridMultiplier = 1;

    // Scenario 1 - reduction: select_reduction picks parallel when tiles <=
    // cu_count/4. 512x512 => 16 tiles fits at 256 CUs (16<=64) but not at 32
    // (16>8), so the strategy flips parallel -> tree.
    {
        auto problem = makeGemmProblem(512, 512, 8192);

        problem.setParams().setSmCountTarget(0);  // use all device CUs (256)
        const auto reductionAllCUs = env.solution.getSKReduction(problem, env.device);

        problem.setParams().setSmCountTarget(32);  // tight CU budget
        const auto reductionCapped = env.solution.getSKReduction(problem, env.device);

        EXPECT_EQ(reductionAllCUs, origami::reduction_t::parallel)
            << "[reduction] With all 256 CUs, 16 tiles fit the parallel-reduction window";
        EXPECT_EQ(reductionCapped, origami::reduction_t::tree)
            << "[reduction] With smCountTarget=32, 16 tiles exceed cu_count/4, so tree";
        EXPECT_NE(reductionAllCUs, reductionCapped)
            << "[reduction] smCountTarget() must flow through to change the predicted reduction";
    }

    // Scenario 2 - grid size: getSKGrid folds smCountTarget into num_cus and
    // select_grid_size (k_split_aware) sizes the grid to the usable CU count.
    // 4096x4096 => 1024 tiles > CUs, so grid ~= cu_count: 256 at all CUs, 64 at
    // smCountTarget=64.
    {
        auto problem = makeGemmProblem(4096, 4096, 8192);

        const size_t tiles = problem.getNumTiles(env.solution.sizeMapping, 1);
        ASSERT_EQ(tiles, 1024u) << "[grid] 128x128 tiles over 4096x4096 => 32*32 = 1024";

        problem.setParams().setSmCountTarget(0); // use all device CUs (256)
        const auto   reductionAllCUs = env.solution.getSKReduction(problem, env.device);
        const size_t gridAllCUs
            = env.solution.getSKGrid(problem, env.device, tiles, reductionAllCUs);

        problem.setParams().setSmCountTarget(64); // tight CU budget
        const auto   reductionCapped = env.solution.getSKReduction(problem, env.device);
        const size_t gridCapped
            = env.solution.getSKGrid(problem, env.device, tiles, reductionCapped);

        EXPECT_EQ(gridAllCUs, 256u)
            << "[grid] With all 256 CUs, k_split_aware distributes 1024 tiles onto 256 CUs";
        EXPECT_EQ(gridCapped, 64u)
            << "[grid] smCountTarget=64 folds into num_cus and caps the grid at 64";
        EXPECT_LT(gridCapped, gridAllCUs)
            << "[grid] A tighter CU budget must shrink the predicted grid";
        EXPECT_NE(gridAllCUs, gridCapped)
            << "[grid] smCountTarget() must flow through to change the predicted StreamK grid";
    }
}

// ===========================================================================
// SK5 workspace sizing regression tests
//
// The under-provision regression occurs when the workspace query and launch
// disagree on the required size because they run with different SK5 tile
// scheduling modes. These tests verify that requiredWorkspaceSize returns
// a consistent, self-sufficient value for each mode so solve() never
// triggers a spurious DP-grid fallback.
// ===========================================================================

TEST(StreamK5WorkspaceRegressionTest, QueryAndLaunchAgreeForDynamicMode)
{
    StreamK5AnalyticalEnv env;
    env.solution.sizeMapping.workspaceSizePerElemC = 4;
    env.solution.sizeMapping.streamKAtomic        = 0;

    // Pick M/N so tiles % grid != 0 (partial tiles exist).
    // macroTile=128x128 → tiles = ceil(M/128)*ceil(N/128).
    // 4096x4224: tiles = 32*33 = 1056, grid = min(1056, 256) = 256,
    // 1056 % 256 != 0 → partial workspace required.
    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setParams().setStreamKTileSchedulingMode(1); // ON (dynamic)
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    size_t tiles = problem.getNumTiles(env.solution.sizeMapping, 1);
    ASSERT_GT(tiles, 0u);

    bool effectiveDynamic = env.solution.streamK5EffectiveDynamic(problem, env.device);
    ASSERT_TRUE(effectiveDynamic) << "4096x4224 with mode=ON must be dynamic";

    auto   reduction = env.solution.getSKReduction(problem, env.device);
    size_t grid      = env.solution.getSKGrid(problem, env.device, tiles, reduction);

    ASSERT_NE(tiles % grid, 0u) << "Need partial tiles for this test";

    size_t ws = env.solution.requiredWorkspaceSize(problem, env.device);
    EXPECT_GT(ws, 0u) << "Dynamic mode with partial tiles must request workspace";

    // The workspace must be at least partialTileSize; the per-XCD work-queue
    // region (cacheLineBytes * numXCD = 128 * 8 = 1024 B on gfx942/gfx950) is
    // included by both query and launch so they agree.
    EXPECT_GE(ws, env.solution.partialTileSize(grid))
        << "Workspace must cover at least partialTileSize(grid)";
}

TEST(StreamK5WorkspaceRegressionTest, StaticModeOmitsQueueRegion)
{
    StreamK5AnalyticalEnv env;
    env.solution.sizeMapping.workspaceSizePerElemC = 4;
    env.solution.sizeMapping.streamKAtomic        = 0;

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setParams().setStreamKTileSchedulingMode(0); // OFF (static)
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto tiles = problem.getNumTiles(env.solution.sizeMapping, 1);
    auto red   = env.solution.getSKReduction(problem, env.device);
    auto grid  = env.solution.getSKGrid(problem, env.device, tiles, red);

    if(tiles % grid == 0)
        GTEST_SKIP() << "No partial tiles for this config";

    size_t ws = env.solution.requiredWorkspaceSize(problem, env.device);

    // Static (SK3) path does not use the work-queue, so workspace
    // should be exactly partialTileSize — no per-XCD work-queue region.
    EXPECT_EQ(ws, env.solution.partialTileSize(grid))
        << "OFF workspace must equal partialTileSize(staticGrid)";
}

TEST(StreamK5WorkspaceRegressionTest, SufficientWorkspacePreventsDPFallback)
{
    StreamK5AnalyticalEnv env;
    env.solution.sizeMapping.workspaceSizePerElemC = 4;
    env.solution.sizeMapping.streamKAtomic        = 0;

    // Use dimensions that produce partial tiles.
    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setParams().setStreamKTileSchedulingMode(1); // ON (dynamic)

    size_t wsNeeded = env.solution.requiredWorkspaceSize(
        [&]{
            auto p = problem;
            p.setWorkspaceSize(std::numeric_limits<size_t>::max());
            return p;
        }(),
        env.device);

    ASSERT_GT(wsNeeded, 0u) << "Need non-zero workspace for this test";

    // Provide exactly the workspace that requiredWorkspaceSize reported.
    // The solve guard must not fall back to DP.
    problem.setWorkspaceSize(wsNeeded);
    size_t wsActual = env.solution.requiredWorkspaceSize(problem, env.device);
    EXPECT_EQ(wsActual, wsNeeded)
        << "With workspace >= required, requiredWorkspaceSize must return the "
        << "full amount (not 0 from DP fallback). ws=" << wsActual
        << " needed=" << wsNeeded;

    // Providing one byte less must trigger fallback (returns 0 for SK partial).
    auto problemShort = makeGemmProblem(4096, 4224, 64);
    problemShort.setParams().setStreamKTileSchedulingMode(1);
    problemShort.setWorkspaceSize(wsNeeded - 1);
    size_t wsShort = env.solution.requiredWorkspaceSize(problemShort, env.device);
    EXPECT_EQ(wsShort, 0u)
        << "With workspace < required, must fall back (return 0 for partial)";
}

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

    auto problemSk3 = makeGemmProblem(512, 512, 512);
    auto problemSk5 = makeGemmProblem(512, 512, 512);
    problemSk5.setParams().setStreamKTileSchedulingMode(0); // SK5-off

    auto sk3Pack = computeStreamKHostPack(sk3Solution, problemSk3, *hardware);
    auto sk5OffPack = computeStreamKHostPack(sk5Solution, problemSk5, *hardware);

    EXPECT_FALSE(sk5OffPack.effectiveDynamic);
    EXPECT_EQ(sk3Pack.reduction, sk5OffPack.reduction);
    EXPECT_EQ(sk3Pack.grid, sk5OffPack.grid);
    EXPECT_EQ(sk3Pack.skTiles, sk5OffPack.skTiles);
    EXPECT_EQ(sk3Pack.skItersPerWG, sk5OffPack.skItersPerWG);

    // Contrast: SK5-on (dynamic) should diverge at 512^3 when tiles < grid.
    problemSk5.setParams().setStreamKTileSchedulingMode(1);
    auto sk5OnPack = computeStreamKHostPack(sk5Solution, problemSk5, *hardware);
    EXPECT_TRUE(sk5OnPack.effectiveDynamic);
    if(sk3Pack.grid > sk3Pack.tiles)
        EXPECT_NE(sk3Pack.grid, sk5OnPack.grid)
            << "512^3 static path oversubscribes; dynamic path should not match";
}

// ===========================================================================
// StreamKDynamicQueueXcdGateTest -- MI300A (NUM_XCD=6) reject-and-continue.
//
// SK4 / SK5-dynamic work-stealing kernels bake a fixed power-of-two per-XCD
// queue count (8 for gfx942/gfx950) and mask indices with (Q-1), so they are
// valid only when the device's runtime NUM_XCD equals that baked count. MI300A
// reports gfx942 (baked 8) but has 6 XCDs; a mismatched partition (e.g. a 4-XCD
// slice of an 8-XCD gfx942) is likewise rejected. The host excludes such a
// solution from selection (streamKDynamicQueueSupported wired into
// softwarePredicate) and warns once instead of silently degrading. The
// production predicates live in a .cpp anonymous namespace, so -- like
// computeStreamKHostPack above -- this test mirrors them over a hip::HipAMDGPU
// mock (6 -> reject, 4 -> reject, 8 -> allow, unknown -> allow). Not run on
// real MI300A silicon.
// ===========================================================================
namespace
{
    // Mirror of the anonymous-namespace helper in ContractionSolution.cpp
    // (streamKBakedQueueCount): the baked per-XCD queue count comes from
    // origami's per-arch XCD count. Kept in lockstep with the production code.
    inline size_t streamKBakedQueueCountRef(Hardware const& hardware)
    {
        auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
            return 0;
        try
        {
            return origami::hardware_t::get_default_num_xcds(
                hipAMDGPU->analyticalHardware->arch);
        }
        catch(std::exception const&)
        {
            return 0;
        }
    }

    // Byte-for-byte mirror of the anonymous-namespace numeric predicate in
    // ContractionSolution.cpp (streamKDynamicQueueUnsupported). Kept in lockstep
    // with the production code; if that predicate changes, update this too.
    // Unknown hardware (not a HipAMDGPU, no analytical hardware, or no baked
    // per-XCD queue count) is treated as UNSUPPORTED (returns true).
    inline bool streamKDynamicQueueUnsupportedRef(Hardware const& hardware)
    {
        auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
            return true;
        size_t baked  = streamKBakedQueueCountRef(hardware);
        size_t numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
        return baked == 0 || numXCD == 0 || (numXCD & (numXCD - 1)) != 0
               || numXCD != baked;
    }

    // Mirror of ContractionSolution::streamKDynamicQueueSupported(). Returns
    // true when the solution is SELECTABLE, false when it must be EXCLUDED
    // (dynamic-queue / work-stealing on a non-power-of-two XCD device). streamK
    // is sizeMapping.streamK; effectiveDynamic is the SK5 sub-mode result
    // (ignored for streamK != 5). Kept in lockstep with the production member.
    inline bool streamKDynamicQueueSupportedRef(int             streamK,
                                                bool            effectiveDynamic,
                                                Hardware const& hardware)
    {
        if(streamK != 4 && streamK != 5)
            return true;
        if(!streamKDynamicQueueUnsupportedRef(hardware))
            return true;
        const bool dynamicQueue = (streamK == 4) || (streamK == 5 && effectiveDynamic);
        return !dynamicQueue; // dynamic-queue on non-pow2 XCD -> excluded
    }

    // gfx942 analytical hardware with a caller-chosen XCD count. NUM_XCD is the
    // 5th positional arg of origami::hardware_t (see origami/hardware.hpp).
    origami::hardware_t makeGfx942HardwareWithXcd(size_t numXCD)
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx942,
                                   304, // N_CU (MI300X SPX)
                                   163840,
                                   262144,
                                   numXCD,
                                   1.0,
                                   1.0,
                                   1.0,
                                   4000000,
                                   1.2,
                                   1,
                                   std::make_tuple(0.0, 0.008, 0.0));
    }

    hip::HipAMDGPU makeGfx942DeviceWithXcd(size_t numXCD)
    {
        hip::HipAMDGPU device;
        device.processor          = AMDGPU::Processor::gfx942;
        device.computeUnitCount   = 304;
        device.deviceName         = "test-gfx942-xcd";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(
            makeGfx942HardwareWithXcd(numXCD));
        return device;
    }
} // namespace

TEST(StreamKDynamicQueueXcdGateTest, RejectsMi300aSixXcd)
{
    hip::HipAMDGPU mi300a = makeGfx942DeviceWithXcd(6);
    Hardware const& hw    = mi300a;
    EXPECT_TRUE(streamKDynamicQueueUnsupportedRef(hw))
        << "MI300A (NUM_XCD=6, not a power of two) must flag the dynamic-queue "
           "work-stealing path as unsupported";
}

TEST(StreamKDynamicQueueXcdGateTest, AllowsMi300xEightXcd)
{
    hip::HipAMDGPU mi300x = makeGfx942DeviceWithXcd(8);
    Hardware const& hw    = mi300x;
    EXPECT_FALSE(streamKDynamicQueueUnsupportedRef(hw))
        << "MI300X (NUM_XCD=8, power of two) must keep the dynamic-queue path";
}

TEST(StreamKDynamicQueueXcdGateTest, RejectsGfx942FourXcdPowerOfTwoButMismatched)
{
    // A 4-XCD partition (e.g. a CPX-style slice) of an 8-XCD gfx942: 4 IS a
    // power of two, but the kernel bakes Q=8, so runtime NUM_XCD (4) != baked
    // (8) and the fixed Q=8 masking would mis-map queues. Must be rejected.
    hip::HipAMDGPU  gfx942Cpx = makeGfx942DeviceWithXcd(4);
    Hardware const& hw        = gfx942Cpx;
    EXPECT_EQ(streamKBakedQueueCountRef(hw), 8u)
        << "gfx942 must bake origami's per-arch XCD count (8)";
    EXPECT_TRUE(streamKDynamicQueueUnsupportedRef(hw))
        << "gfx942 with NUM_XCD=4 (power of two but != baked 8) must be rejected";
}

TEST(StreamKDynamicQueueXcdGateTest, AllowsGfx950EightXcd)
{
    // gfx950 (local MI355X) analytical hardware advertises 8 XCDs.
    hip::HipAMDGPU gfx950   = makeHipDeviceWithAnalytical(makeGfx950AnalyticalHardware());
    Hardware const& hw      = gfx950;
    EXPECT_FALSE(streamKDynamicQueueUnsupportedRef(hw))
        << "gfx950 (NUM_XCD=8) must keep the dynamic-queue work-stealing path";
}

TEST(StreamKDynamicQueueXcdGateTest, MissingAnalyticalHardwareIsUnsupported)
{
    // Unknown hardware (null analyticalHardware -> unknown NUM_XCD / baked
    // queue count == 0) must be treated as UNSUPPORTED so the dynamic-queue
    // solution is excluded from selection and a non-dynamic-queue solution
    // serves the GEMM, rather than staying selectable while the per-XCD counter
    // workspace is sized with an unknown (0) queue count (under-allocation).
    hip::HipAMDGPU noAnalytical;
    noAnalytical.processor     = AMDGPU::Processor::gfx942;
    noAnalytical.deviceName    = "test-gfx942-no-analytical";
    Hardware const& hwNoAnalyt = noAnalytical;
    ASSERT_EQ(noAnalytical.analyticalHardware, nullptr);
    EXPECT_TRUE(streamKDynamicQueueUnsupportedRef(hwNoAnalyt))
        << "Missing analyticalHardware (unknown NUM_XCD) must be treated as unsupported";
    // And the selection predicate must therefore EXCLUDE the dynamic-queue
    // solution (SK4) while keeping non-dynamic-queue solutions selectable.
    EXPECT_FALSE(streamKDynamicQueueSupportedRef(4, /*effectiveDynamic=*/false, hwNoAnalyt))
        << "SK4 work-stealing solution must be excluded when NUM_XCD is unknown";
    EXPECT_TRUE(streamKDynamicQueueSupportedRef(3, /*effectiveDynamic=*/false, hwNoAnalyt))
        << "SK3-static solution must remain selectable when NUM_XCD is unknown";
}

// Selection-predicate contract: on MI300A (6 XCD) the dynamic-queue solution is
// EXCLUDED from selection (supported == false) so a different solution serves
// the GEMM, while on MI300X (8 XCD) the identical solution stays selectable.
TEST(StreamKDynamicQueueXcdGateTest, ExcludesDynamicQueueSolutionOnMi300a)
{
    hip::HipAMDGPU  mi300a = makeGfx942DeviceWithXcd(6);
    hip::HipAMDGPU  mi300x = makeGfx942DeviceWithXcd(8);
    Hardware const& hwA    = mi300a;
    Hardware const& hwX    = mi300x;

    // SK4 is always dynamic-queue.
    EXPECT_FALSE(streamKDynamicQueueSupportedRef(4, /*effectiveDynamic=*/false, hwA))
        << "SK4 work-stealing solution must be excluded from selection on MI300A";
    EXPECT_TRUE(streamKDynamicQueueSupportedRef(4, /*effectiveDynamic=*/false, hwX))
        << "SK4 work-stealing solution must remain selectable on MI300X";

    // SK5 only takes the dynamic-queue path when it resolves to the dynamic
    // sub-mode; the static (SK3) sub-mode stays selectable even on MI300A.
    EXPECT_FALSE(streamKDynamicQueueSupportedRef(5, /*effectiveDynamic=*/true, hwA))
        << "SK5-dynamic must be excluded from selection on MI300A";
    EXPECT_TRUE(streamKDynamicQueueSupportedRef(5, /*effectiveDynamic=*/false, hwA))
        << "SK5-static (SK3 sub-path) must remain selectable on MI300A";
}

// Non-dynamic-queue solutions must never be excluded, so the GEMM still runs.
TEST(StreamKDynamicQueueXcdGateTest, KeepsNonDynamicQueueSolutionsOnMi300a)
{
    hip::HipAMDGPU  mi300a = makeGfx942DeviceWithXcd(6);
    Hardware const& hwA    = mi300a;

    EXPECT_TRUE(streamKDynamicQueueSupportedRef(0, /*effectiveDynamic=*/false, hwA))
        << "Non-StreamK solution must remain selectable on MI300A";
    EXPECT_TRUE(streamKDynamicQueueSupportedRef(3, /*effectiveDynamic=*/false, hwA))
        << "SK3-static solution must remain selectable on MI300A";
}
