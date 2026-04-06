// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <set>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

// Verify that hipDeviceAttributePciChipId is correctly queried and matches
// a known chip ID.
TEST(PciChipIdTest, QueryDeviceChipId)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No HIP devices available";

    hipDeviceProp_t prop;
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess) << "Failed to get device properties";
    auto processor = AMDGPU::toProcessor(prop.gcnArchName);
    if(!ChipIdRegistry::supportsChipIdPredicate(processor))
        GTEST_SKIP() << "PCI chip ID querying is gfx950-only at runtime";

    // Query PCI Chip ID using hipDeviceGetAttribute directly
    int pciChipId = 0;
    err = hipDeviceGetAttribute(&pciChipId, hipDeviceAttributePciChipId, 0);
    ASSERT_EQ(err, hipSuccess) << "Failed to get PCI Chip ID attribute";

    auto isKnownChipId = ChipIdRegistry::isKnownChipId(pciChipId);
    EXPECT_TRUE(isKnownChipId) << "PCI Chip ID should be known";
}

// Verify that HipAMDGPU correctly populates pciChipId from HIP runtime
// and matches a known chip ID.
TEST(PciChipIdTest, HipHardwarePopulatesPciChipId)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No HIP devices available";

    // Get the current device using Tensile's hip::GetCurrentDevice()
    auto hardware = hip::GetCurrentDevice();
    ASSERT_NE(hardware, nullptr) << "Failed to get current device";

    // Cast to AMDGPU to access pciChipId
    auto* amdgpu = dynamic_cast<AMDGPU*>(hardware.get());
    ASSERT_NE(amdgpu, nullptr) << "Hardware is not an AMDGPU";

    if(!ChipIdRegistry::supportsChipIdPredicate(amdgpu->processor))
    {
        EXPECT_FALSE(amdgpu->pciChipId().has_value())
            << "pciChipId should be unset for non-gfx950 processors";
        return;
    }

    if(Debug::Instance().printDeviceSelection())
    {
        // Print device information
        std::cout << "\n=== AMDGPU Hardware Info ===" << std::endl;
        std::cout << "Description:    " << amdgpu->description() << std::endl;
        std::cout << "Processor:      " << AMDGPU::toString(amdgpu->processor) << std::endl;
        std::cout << "CU Count:       " << amdgpu->computeUnitCount << std::endl;
        if(amdgpu->pciChipId().has_value())
        {
            std::cout << "PCI Chip ID:    0x" << std::hex << amdgpu->pciChipId().value() << std::dec
                      << " (" << amdgpu->pciChipId().value() << ")" << std::endl;
        }
        else
        {
            std::cout << "PCI Chip ID:    (not set)" << std::endl;
        }
        std::cout << "=============================\n" << std::endl;
    }

    EXPECT_TRUE(amdgpu->pciChipId().has_value()) << "pciChipId should be populated from HIP runtime";
    EXPECT_GT(amdgpu->pciChipId().value(), 0) << "pciChipId should be a positive value";

    auto isKnownChipId = ChipIdRegistry::isKnownChipId(amdgpu->pciChipId().value());
    EXPECT_TRUE(isKnownChipId) << "PCI Chip ID should be known";
}

// Verify PciChipIdEqual predicate matches the correct device
TEST(PciChipIdTest, PciChipIdEqualPredicate)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No HIP devices available";

    // Get the current device
    auto hardware = hip::GetCurrentDevice();
    ASSERT_NE(hardware, nullptr);

    auto* amdgpu = dynamic_cast<AMDGPU*>(hardware.get());
    ASSERT_NE(amdgpu, nullptr);

    if(!ChipIdRegistry::supportsChipIdPredicate(amdgpu->processor))
        GTEST_SKIP() << "PciChipIdEqual runtime matching is gfx950-only";

    ASSERT_TRUE(amdgpu->pciChipId().has_value()) << "pciChipId must be set for this test";
    int actualPciChipId = amdgpu->pciChipId().value();
    ASSERT_GT(actualPciChipId, 0) << "pciChipId must be valid for this test";

    // Create predicate that matches the actual chip ID
    auto matchingPred = std::make_shared<Predicates::GPU::PciChipIdEqual>(actualPciChipId);

    // Create predicate that does NOT match (use a different ID)
    auto nonMatchingPred = std::make_shared<Predicates::GPU::PciChipIdEqual>(0x1234);

    EXPECT_TRUE((*matchingPred)(*amdgpu)) << "Predicate should match actual chip ID";
    EXPECT_FALSE((*nonMatchingPred)(*amdgpu)) << "Predicate should NOT match different chip ID";
}

TEST(PciChipIdTest, PciChipIdEqualIgnoredOnNonGfx950)
{
    AMDGPU gpuWithChipId(AMDGPU::Processor::gfx942, 120, "gfx942-mock", std::make_optional(0x75a0));
    auto   pred = std::make_shared<Predicates::GPU::PciChipIdEqual>(0x75a0);

    EXPECT_FALSE((*pred)(gpuWithChipId))
        << "PciChipIdEqual must ignore chip IDs for non-gfx950 processors";
    EXPECT_FALSE(pred->isFallbackMatch(gpuWithChipId))
        << "Fallback matching must also be disabled for non-gfx950 processors";
}

TEST(PciChipIdTest, OrPciChipIdPredicateExactForAllTargets)
{
    constexpr int chipA        = 0x75a0;
    constexpr int chipB        = 0x75a2;
    constexpr int chipFallback = 0x75a3; // Falls back to 0x75a0.

    auto isProcessor = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx950);
    auto isChipA     = std::make_shared<Predicates::GPU::PciChipIdEqual>(chipA);
    auto isChipB     = std::make_shared<Predicates::GPU::PciChipIdEqual>(chipB);
    auto isChipAOrB  = std::make_shared<Predicates::Or<AMDGPU>>(
        std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{isChipA, isChipB});
    auto isTargetDevice = std::make_shared<Predicates::And<AMDGPU>>(
        std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{isProcessor, isChipAOrB});

    HardwarePredicate pred(std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isTargetDevice),
                           std::set<int>{chipA, chipB});

    AMDGPU gpuA(AMDGPU::Processor::gfx950, 256, "gpuA", std::make_optional(chipA));
    AMDGPU gpuB(AMDGPU::Processor::gfx950, 256, "gpuB", std::make_optional(chipB));
    AMDGPU gpuFallback(AMDGPU::Processor::gfx950, 256, "gpuFallback", std::make_optional(chipFallback));

    // Both members of the Or chip-ID set are exact (not fallback).
    EXPECT_TRUE(pred(dummyProblem(), gpuA));
    EXPECT_FALSE(pred.isFallbackMatch(gpuA));

    EXPECT_TRUE(pred(dummyProblem(), gpuB));
    EXPECT_FALSE(pred.isFallbackMatch(gpuB));

    // A chip that matches through fallback should still be classified as fallback.
    EXPECT_TRUE(pred(dummyProblem(), gpuFallback));
    EXPECT_TRUE(pred.isFallbackMatch(gpuFallback));
}

// Test: Hardware selection with PciChipIdEqual in a library hierarchy
TEST(PciChipIdTest, HardwareSelectionWithPciChipId)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No HIP devices available";

    // Get the current device
    auto hardware = hip::GetCurrentDevice();
    ASSERT_NE(hardware, nullptr);

    auto* amdgpu = dynamic_cast<AMDGPU*>(hardware.get());
    ASSERT_NE(amdgpu, nullptr);

    if(!ChipIdRegistry::supportsChipIdPredicate(amdgpu->processor))
        GTEST_SKIP() << "HardwareSelectionWithPciChipId is gfx950-specific";

    ASSERT_TRUE(amdgpu->pciChipId().has_value()) << "pciChipId must be set for this test";
    int actualPciChipId = amdgpu->pciChipId().value();
    AMDGPU::Processor actualProcessor = amdgpu->processor;

    // Create solutions for different scenarios
    auto deviceSpecificSolution = std::make_shared<ContractionSolution>();
    deviceSpecificSolution->index = 1;
    deviceSpecificSolution->solutionName = "DeviceSpecific_0x" + 
        ([&]() { std::ostringstream ss; ss << std::hex << actualPciChipId; return ss.str(); })();

    auto fallbackSolution = std::make_shared<ContractionSolution>();
    fallbackSolution->index = 2;
    fallbackSolution->solutionName = "Fallback_" + AMDGPU::toString(actualProcessor);

    // Create libraries
    auto deviceSpecificLib = std::make_shared<SingleContractionLibrary>(deviceSpecificSolution);
    auto fallbackLib = std::make_shared<SingleContractionLibrary>(fallbackSolution);

    // Create hardware predicate for specific PCI Chip ID + Processor
    auto isPciChip = std::make_shared<Predicates::GPU::PciChipIdEqual>(actualPciChipId);
    auto isProcessor = std::make_shared<Predicates::GPU::ProcessorEqual>(actualProcessor);
    auto isSpecificDevice = std::make_shared<Predicates::And<AMDGPU>>(
        std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{isProcessor, isPciChip});
    
    HardwarePredicate deviceSpecificPred(
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isSpecificDevice),
        actualPciChipId);

    // Create fallback predicate (processor only, no chip ID)
    HardwarePredicate fallbackPred(
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isProcessor));

    // Build the hardware selection library (device-specific first, then fallback)
    ContractionHardwareSelectionLibrary::Row deviceRow(deviceSpecificPred, deviceSpecificLib);
    ContractionHardwareSelectionLibrary::Row fallbackRow(fallbackPred, fallbackLib);
    ContractionHardwareSelectionLibrary lib({deviceRow, fallbackRow});

    // Create a simple problem
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024, 1.0, false, 1);

    // Find best solution - should match device-specific
    auto solution = lib.findBestSolution(problem, *hardware);

    if(Debug::Instance().printDeviceSelection())
    {
        std::cout << "\n=== Hardware Selection with PCI Chip ID ===" << std::endl;
        std::cout << "Device: " << amdgpu->description() << std::endl;
        std::cout << "PCI Chip ID: 0x" << std::hex << actualPciChipId << std::dec << std::endl;
        if(solution)
        {
            std::cout << "Selected solution: " << solution->solutionName << " (index=" << solution->index << ")"
                      << std::endl;
        }
        else
        {
            std::cout << "No solution found!" << std::endl;
        }
        std::cout << "============================================\n" << std::endl;
    }

    ASSERT_NE(solution, nullptr) << "Should find a matching solution";
    EXPECT_EQ(solution->index, 1) << "Should select device-specific solution (index 1)";

    // Test with a different hardware (simulated) to verify fallback
    // Use std::make_optional to explicitly set a different chip ID
    AMDGPU differentDevice(actualProcessor, amdgpu->computeUnitCount, "Different Device", std::make_optional(0x1234));
    
    auto fallbackResult = lib.findBestSolution(problem, differentDevice);
    ASSERT_NE(fallbackResult, nullptr) << "Should find fallback solution";
    EXPECT_EQ(fallbackResult->index, 2) << "Should select fallback solution (index 2) for different chip ID";

    if(Debug::Instance().printDeviceSelection())
    {
        std::cout << "Fallback test with different PCI Chip ID (0x1234):" << std::endl;
        std::cout << "Selected solution: " << fallbackResult->solutionName << " (index=" << fallbackResult->index
                  << ")" << std::endl;
        std::cout << "=================================================\n" << std::endl;
    }
}

// Test: findAllSolutions with hardware containing pciChipId
TEST(PciChipIdTest, FindAllSolutionsWithPciChipId)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess);
    ASSERT_GT(deviceCount, 0);

    auto hardware = hip::GetCurrentDevice();
    ASSERT_NE(hardware, nullptr);

    auto* amdgpu = dynamic_cast<AMDGPU*>(hardware.get());
    ASSERT_NE(amdgpu, nullptr);

    // Create multiple solutions
    auto solution1 = std::make_shared<ContractionSolution>();
    solution1->index = 1;
    solution1->solutionName = "Solution_A";

    auto solution2 = std::make_shared<ContractionSolution>();
    solution2->index = 2;
    solution2->solutionName = "Solution_B";

    // Create libraries
    auto lib1 = std::make_shared<SingleContractionLibrary>(solution1);
    auto lib2 = std::make_shared<SingleContractionLibrary>(solution2);

    // Create hardware predicate that matches current device
    auto isProcessor = std::make_shared<Predicates::GPU::ProcessorEqual>(amdgpu->processor);
    HardwarePredicate hwPred(std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isProcessor));

    // Build library with multiple solutions for same hardware
    ContractionHardwareSelectionLibrary::Row row1(hwPred, lib1);
    ContractionHardwareSelectionLibrary::Row row2(hwPred, lib2);
    ContractionHardwareSelectionLibrary lib({row1, row2});

    // Create a problem
    auto problem = ContractionProblemGemm::GEMM(false, false, 512, 512, 512, 512, 512, 512, 1.0, false, 1);

    // Find all solutions
    auto solutions = lib.findAllSolutions(problem, *hardware);

    if(Debug::Instance().printDeviceSelection())
    {
        std::cout << "\n=== findAllSolutions Test ===" << std::endl;
        std::cout << "Hardware: " << amdgpu->description() << std::endl;
        if(amdgpu->pciChipId().has_value())
        {
            std::cout << "PCI Chip ID: 0x" << std::hex << amdgpu->pciChipId().value() << std::dec << std::endl;
        }
        else
        {
            std::cout << "PCI Chip ID: (not set)" << std::endl;
        }
        std::cout << "Found " << solutions.size() << " solution(s):" << std::endl;
        for(const auto& sol : solutions)
        {
            std::cout << "  - " << sol->solutionName << " (index=" << sol->index << ")" << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
    }

    EXPECT_EQ(solutions.size(), 2) << "Should find both solutions";
}

// ===========================================================================
// ChipIdFallbackTest fixture -- verifies the 4 kernel-availability scenarios
// described in the mi350/mi355 fallback specification.
//
// Each scenario constructs a HardwareSelectionLibrary whose rows are in the
// exact order the Python build system produces after the descending-chip-ID
// sort fix (higher chip IDs first, higher CU counts first within a chip,
// processor-only catch-all last).
//
// Row evaluation is first-match-wins (ExactLogicLibrary), and
// PciChipIdEqual includes one-directional fallback (mi355->mi350).
// ===========================================================================
class ChipIdFallbackTest : public ::testing::Test
{
protected:
    // Mock devices
    AMDGPU mi350spx = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    AMDGPU mi355spx = makeDevice(_MI355_CHIP_ID, _SPX_CU, "mi355spx");
    AMDGPU mi350cpx = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    AMDGPU mi355cpx = makeDevice(_MI355_CHIP_ID, _CPX_CU, "mi355cpx");

    static constexpr auto gfx950 = AMDGPU::Processor::gfx950;

    // Index counter for unique solution IDs
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
// Scenario 1: Only mi350 kernels (equality + Origami)
//
// Available kernels : mi350spx_eq, mi350spx_oob
// Expected:
//   mi350spx  -> mi350spx_eq  (exact chip match, SPX CU)
//   mi355spx  -> mi350spx_oob (mi355 falls through mi355-oob-only row)
//   mi350cpx  -> mi350spx_oob (CU=64 skips CU=256 rows)
//   mi355cpx  -> mi350spx_oob (CU=64 skips CU=256 rows)
// ---------------------------------------------------------------------------
TEST_F(ChipIdFallbackTest, Scenario1_MI350Only)
{
    dbg("=== Scenario 1: MI350 Only ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");

    // Row order mirrors the Python sort: higher chipId first, higher CU first.
    // Since only mi350 kernels exist, the mi355 rows map to mi350 oob.
    auto lib = buildHwLib({
        // Row 1: mi355, CU=256 -- no mi355-specific kernels, oob only
        {makeHwPred(gfx950, _MI355_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 2: mi350, CU=256 -- equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_eq), singleLib(mi350spx_oob))},

        // Row 3: mi355, any CU -- oob only
        {makeHwPred(gfx950, _MI355_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 4: mi350, any CU -- oob only
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 5: gfx950 catch-all
        {makeHwPred(gfx950),
         buildProblemLib(singleLib(mi350spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi355spx, "mi350spx_oob");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");
    expectSelected(*lib, mi355cpx, "mi350spx_oob");
}

// ---------------------------------------------------------------------------
// Scenario 2: mi355spx equality + mi350 (equality + Origami)
//
// Available kernels : mi350spx_eq, mi350spx_oob, mi355spx_eq
// Expected:
//   mi350spx  -> mi350spx_eq
//   mi355spx  -> mi355spx_eq  (exact chip match)
//   mi350cpx  -> mi350spx_oob
//   mi355cpx  -> mi350spx_oob
// ---------------------------------------------------------------------------
TEST_F(ChipIdFallbackTest, Scenario2_MI355Eq)
{
    dbg("=== Scenario 2: MI355 Eq ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");
    auto mi355spx_eq  = sol("mi355spx_eq");

    auto lib = buildHwLib({
        // Row 1: mi355, CU=256 -- mi355 equality + mi350 oob fallback
        {makeHwPred(gfx950, _MI355_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi355spx_eq), singleLib(mi350spx_oob))},

        // Row 2: mi350, CU=256 -- mi350 equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_eq), singleLib(mi350spx_oob))},

        // Row 3: mi355, any CU -- oob only
        {makeHwPred(gfx950, _MI355_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 4: mi350, any CU -- oob only
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 5: catch-all
        {makeHwPred(gfx950),
         buildProblemLib(singleLib(mi350spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi355spx, "mi355spx_eq");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");
    expectSelected(*lib, mi355cpx, "mi350spx_oob");
}

// ---------------------------------------------------------------------------
// Scenario 3: mi355spx (equality + Origami) + mi350 (equality + Origami)
//
// Available kernels : mi350spx_eq, mi350spx_oob, mi355spx_eq, mi355spx_oob
// Expected:
//   mi350spx  -> mi350spx_eq
//   mi355spx  -> mi355spx_eq
//   mi350cpx  -> mi350spx_oob
//   mi355cpx  -> mi355spx_oob
// ---------------------------------------------------------------------------
TEST_F(ChipIdFallbackTest, Scenario3_MI355EqOob)
{
    dbg("=== Scenario 3: MI355 Eq + Oob ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");
    auto mi355spx_eq  = sol("mi355spx_eq");
    auto mi355spx_oob = sol("mi355spx_oob");

    auto lib = buildHwLib({
        // Row 1: mi355, CU=256 -- mi355 equality + mi355 oob
        {makeHwPred(gfx950, _MI355_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi355spx_eq), singleLib(mi355spx_oob))},

        // Row 2: mi350, CU=256 -- mi350 equality + mi350 oob
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

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi355spx, "mi355spx_eq");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");
    expectSelected(*lib, mi355cpx, "mi355spx_oob");
}

// ---------------------------------------------------------------------------
// Scenario 4: mi355spx (equality + OOB) + mi355cpx equality + mi350 (eq + oob)
//
// Available kernels : mi350spx_eq, mi350spx_oob, mi355spx_eq, mi355spx_oob,
//                     mi355cpx_eq
// Expected:
//   mi350spx  -> mi350spx_eq
//   mi355spx  -> mi355spx_eq
//   mi350cpx  -> mi350spx_oob
//   mi355cpx  -> mi355cpx_eq   (has its own CPX equality kernels)
//
// NOTE: The original specification listed "mi355cpx: mi350cpx_eq -> mi355spx_oob"
// while the prose said "mi355cpx should use mi355cpx equality kernels first".
// We follow the prose and use mi355cpx_eq; rename if needed.
// ---------------------------------------------------------------------------
TEST_F(ChipIdFallbackTest, Scenario4_MI355CpxEq)
{
    dbg("=== Scenario 4: MI355 CPX Eq ===");

    auto mi350spx_eq  = sol("mi350spx_eq");
    auto mi350spx_oob = sol("mi350spx_oob");
    auto mi355spx_eq  = sol("mi355spx_eq");
    auto mi355spx_oob = sol("mi355spx_oob");
    auto mi355cpx_eq  = sol("mi355cpx_eq");

    auto lib = buildHwLib({
        // Row 1: mi355, CU=256 -- mi355 equality + oob
        {makeHwPred(gfx950, _MI355_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi355spx_eq), singleLib(mi355spx_oob))},

        // Row 2: mi350, CU=256 -- mi350 equality + oob
        {makeHwPred(gfx950, _MI350_CHIP_ID, _SPX_CU),
         buildProblemLib(singleLib(mi350spx_eq), singleLib(mi350spx_oob))},

        // Row 3: mi355, CU=64 -- mi355 cpx equality + mi355 spx oob fallback
        {makeHwPred(gfx950, _MI355_CHIP_ID, _CPX_CU),
         buildProblemLib(singleLib(mi355cpx_eq), singleLib(mi355spx_oob))},

        // Row 4: mi355, any CU -- mi355 oob
        {makeHwPred(gfx950, _MI355_CHIP_ID),
         buildProblemLib(singleLib(mi355spx_oob))},

        // Row 5: mi350, any CU -- mi350 oob
        {makeHwPred(gfx950, _MI350_CHIP_ID),
         buildProblemLib(singleLib(mi350spx_oob))},

        // Row 6: catch-all
        {makeHwPred(gfx950),
         buildProblemLib(singleLib(mi350spx_oob))},
    });

    expectSelected(*lib, mi350spx, "mi350spx_eq");
    expectSelected(*lib, mi355spx, "mi355spx_eq");
    expectSelected(*lib, mi350cpx, "mi350spx_oob");
    expectSelected(*lib, mi355cpx, "mi355cpx_eq");
}
