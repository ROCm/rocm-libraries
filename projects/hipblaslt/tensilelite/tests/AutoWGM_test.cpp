// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ContractionSolution::calculateAutoWGM for non-persistent kernels: on gfx1151 a kernel with
// WorkGroupMapping 0 takes Origami's pick; a pinned mapping and other architectures do not.

#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

using namespace TensileLite;

namespace
{
    using arch_t = origami::hardware_t::architecture_t;

    hip::HipAMDGPU makeDevice(arch_t arch, AMDGPU::Processor processor, size_t numCUs)
    {
        hip::HipAMDGPU device;
        device.processor        = processor;
        device.computeUnitCount = static_cast<int>(numCUs);
        device.deviceName       = "auto-wgm-test";
        device.analyticalHardware
            = std::make_shared<origami::hardware_t>(origami::hardware_t::get_hardware_for_arch(
                arch, numCUs, 65536, 512 * 1024, 2 << 20, 1900000));
        return device;
    }

    hip::HipAMDGPU makeGfx1151()
    {
        return makeDevice(arch_t::gfx1151, AMDGPU::Processor::gfx1151, 40);
    }

    // bf16 TN kernel, MacroTile 96x128, DepthU 32, not persistent.
    void initSolution(ContractionSolution& solution, int workGroupMapping)
    {
        solution.sizeMapping.macroTile         = TensileLite::dim3(96, 128, 1);
        solution.sizeMapping.depthU            = 32;
        solution.sizeMapping.matrixInstruction = {16, 16, 16, 1};
        solution.sizeMapping.CUOccupancy       = 4;
        solution.sizeMapping.globalSplitU      = 1;
        solution.sizeMapping.workGroupMapping  = workGroupMapping;
    }

    // bf16 A^T B with natural lda/ldb and the given output stride.
    ContractionProblemGemm makeProblem(size_t m, size_t n, size_t k, size_t ldd)
    {
        return ContractionProblemGemm::GEMM_Strides(true,
                                                    false,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    m,
                                                    n,
                                                    k,
                                                    1,
                                                    k,
                                                    k * m,
                                                    k,
                                                    k * n,
                                                    ldd,
                                                    ldd * n,
                                                    ldd,
                                                    ldd * n,
                                                    0.0);
    }

    int32_t autoWGM(int                           workGroupMapping,
                    ContractionProblemGemm const& problem,
                    hip::HipAMDGPU const&         device)
    {
        ContractionSolution solution;
        initSolution(solution, workGroupMapping);
        return std::get<0>(solution.calculateAutoWGM(problem, &device, /*skgrid=*/0));
    }
} // namespace

TEST(AutoWGMTest, Gfx1151SmallAKeepsDispatchOrder)
{
    const auto device = makeGfx1151();
    // A = 1.25 MiB: re-reading it per column band is cheap.
    EXPECT_EQ(autoWGM(0, makeProblem(2560, 2880, 256, 2560), device), 1);
}

TEST(AutoWGMTest, Gfx1151LargeABandsAllColumns)
{
    const auto device = makeGfx1151();
    // A = 296 MiB, 8 column bands; the natural output row (303360 bytes) spreads over all
    // DRAM units, so one band takes every column tile.
    EXPECT_EQ(autoWGM(0, makeProblem(151680, 1024, 1024, 151680), device), 8);
}

TEST(AutoWGMTest, Gfx1151PageAlignedOutputCapsBand)
{
    const auto device = makeGfx1151();
    // Same problem with a page-multiple output row (307200 bytes): the band is capped.
    EXPECT_EQ(autoWGM(0, makeProblem(151680, 1024, 1024, 153600), device), 4);
}

TEST(AutoWGMTest, Gfx1151PinnedMappingIsKept)
{
    const auto device = makeGfx1151();
    EXPECT_EQ(autoWGM(5, makeProblem(151680, 1024, 1024, 151680), device), 5);
}

TEST(AutoWGMTest, OtherArchitectureKeepsSqrtDefault)
{
    const auto device = makeDevice(arch_t::gfx950, AMDGPU::Processor::gfx950, 256);
    const auto hw     = *device.analyticalHardware;
    const auto legacy = static_cast<int32_t>(std::ceil(std::sqrt(hw.N_CU / hw.NUM_XCD)));
    EXPECT_EQ(autoWGM(0, makeProblem(151680, 1024, 1024, 151680), device), legacy);
    EXPECT_EQ(autoWGM(5, makeProblem(151680, 1024, 1024, 151680), device), 5);
}
