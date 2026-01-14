/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>

using namespace TensileLite;

// Test: Verify that hipDeviceAttributePciChipId is correctly queried and stored
TEST(PciChipIDTest, QueryDeviceChipId)
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No HIP devices available";

    // Query PCI Chip ID using hipDeviceGetAttribute directly
    int pciChipId = 0;
    err = hipDeviceGetAttribute(&pciChipId, hipDeviceAttributePciChipId, 0);
    ASSERT_EQ(err, hipSuccess) << "Failed to get PCI Chip ID attribute";

    // Print the device ID for manual verification
    std::cout << "\n=== PCI Device ID Test ===" << std::endl;
    std::cout << "Device 0 PCI Chip ID (decimal): " << pciChipId << std::endl;
    std::cout << "Device 0 PCI Chip ID (hex):     0x" << std::hex << pciChipId << std::dec << std::endl;
    std::cout << "Verify with: lspci -nn | grep -i " << std::hex << pciChipId << std::dec << std::endl;
    std::cout << "==========================\n" << std::endl;

    EXPECT_GT(pciChipId, 0) << "PCI Chip ID should be a positive value";
}

// Test: Verify that HipAMDGPU correctly populates pciChipId from HIP runtime
TEST(PciChipIDTest, HipHardwarePopulatesPciChipId)
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

    // Print device information
    std::cout << "\n=== AMDGPU Hardware Info ===" << std::endl;
    std::cout << "Description:    " << amdgpu->description() << std::endl;
    std::cout << "Processor:      " << AMDGPU::toString(amdgpu->processor) << std::endl;
    std::cout << "CU Count:       " << amdgpu->computeUnitCount << std::endl;
    if (amdgpu->pciChipId.has_value()) {
        std::cout << "PCI Chip ID:    0x" << std::hex << amdgpu->pciChipId.value() << std::dec
                  << " (" << amdgpu->pciChipId.value() << ")" << std::endl;
    } else {
        std::cout << "PCI Chip ID:    (not set)" << std::endl;
    }
    std::cout << "=============================\n" << std::endl;

    EXPECT_TRUE(amdgpu->pciChipId.has_value()) << "pciChipId should be populated from HIP runtime";
    EXPECT_GT(amdgpu->pciChipId.value(), 0) << "pciChipId should be a positive value";
}

// Test: Verify PciChipIDEqual predicate matches the correct device
TEST(PciChipIDTest, PciChipIDEqualPredicate)
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

    ASSERT_TRUE(amdgpu->pciChipId.has_value()) << "pciChipId must be set for this test";
    int actualPciChipId = amdgpu->pciChipId.value();
    ASSERT_GT(actualPciChipId, 0) << "pciChipId must be valid for this test";

    // Create predicate that matches the actual chip ID
    auto matchingPred = std::make_shared<Predicates::GPU::PciChipIDEqual>(actualPciChipId);

    // Create predicate that does NOT match (use a different ID)
    auto nonMatchingPred = std::make_shared<Predicates::GPU::PciChipIDEqual>(0x9999);

    // Test predicate evaluation
    EXPECT_TRUE((*matchingPred)(*amdgpu)) << "Predicate should match actual chip ID";
    EXPECT_FALSE((*nonMatchingPred)(*amdgpu)) << "Predicate should NOT match different chip ID";

    std::cout << "\n=== PciChipIDEqual Predicate Test ===" << std::endl;
    std::cout << "Actual PCI Chip ID: 0x" << std::hex << actualPciChipId << std::dec << std::endl;
    std::cout << "Matching predicate (0x" << std::hex << actualPciChipId << std::dec << "): PASS" << std::endl;
    std::cout << "Non-matching predicate (0x9999): CORRECTLY REJECTED" << std::endl;
    std::cout << "=======================================\n" << std::endl;
}

// Test: Hardware selection with PciChipIDEqual in a library hierarchy
TEST(PciChipIDTest, HardwareSelectionWithPciChipID)
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

    ASSERT_TRUE(amdgpu->pciChipId.has_value()) << "pciChipId must be set for this test";
    int actualPciChipId = amdgpu->pciChipId.value();
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
    auto isPciChip = std::make_shared<Predicates::GPU::PciChipIDEqual>(actualPciChipId);
    auto isProcessor = std::make_shared<Predicates::GPU::ProcessorEqual>(actualProcessor);
    auto isSpecificDevice = std::make_shared<Predicates::And<AMDGPU>>(
        std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{isProcessor, isPciChip});
    
    HardwarePredicate deviceSpecificPred(
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isSpecificDevice));

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

    std::cout << "\n=== Hardware Selection with PCI Chip ID ===" << std::endl;
    std::cout << "Device: " << amdgpu->description() << std::endl;
    std::cout << "PCI Chip ID: 0x" << std::hex << actualPciChipId << std::dec << std::endl;
    if (solution) {
        std::cout << "Selected solution: " << solution->solutionName << " (index=" << solution->index << ")" << std::endl;
    } else {
        std::cout << "No solution found!" << std::endl;
    }
    std::cout << "============================================\n" << std::endl;

    ASSERT_NE(solution, nullptr) << "Should find a matching solution";
    EXPECT_EQ(solution->index, 1) << "Should select device-specific solution (index 1)";

    // Test with a different hardware (simulated) to verify fallback
    // Use std::make_optional to explicitly set a different chip ID
    AMDGPU differentDevice(actualProcessor, amdgpu->computeUnitCount, "Different Device", std::make_optional(0x1234));
    
    auto fallbackResult = lib.findBestSolution(problem, differentDevice);
    ASSERT_NE(fallbackResult, nullptr) << "Should find fallback solution";
    EXPECT_EQ(fallbackResult->index, 2) << "Should select fallback solution (index 2) for different chip ID";

    std::cout << "Fallback test with different PCI Chip ID (0x1234):" << std::endl;
    std::cout << "Selected solution: " << fallbackResult->solutionName << " (index=" << fallbackResult->index << ")" << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

// Test: findAllSolutions with hardware containing pciChipId
TEST(PciChipIDTest, FindAllSolutionsWithPciChipID)
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

    std::cout << "\n=== findAllSolutions Test ===" << std::endl;
    std::cout << "Hardware: " << amdgpu->description() << std::endl;
    if (amdgpu->pciChipId.has_value()) {
        std::cout << "PCI Chip ID: 0x" << std::hex << amdgpu->pciChipId.value() << std::dec << std::endl;
    } else {
        std::cout << "PCI Chip ID: (not set)" << std::endl;
    }
    std::cout << "Found " << solutions.size() << " solution(s):" << std::endl;
    for (const auto& sol : solutions) {
        std::cout << "  - " << sol->solutionName << " (index=" << sol->index << ")" << std::endl;
    }
    std::cout << "==============================\n" << std::endl;

    EXPECT_EQ(solutions.size(), 2) << "Should find both solutions";
}

