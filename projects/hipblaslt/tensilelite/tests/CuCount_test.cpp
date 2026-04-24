// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>

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
