// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// calculateAutoStaggerU on gfx1151: when the caller pads an A/B leading dimension that would
// alias DRAM channels (multiple of 2048 B) when packed, the solution's StaggerU mapping is
// replaced by 4 (all workgroups in phase). Host-only, no device library required.

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>

using namespace TensileLite;

namespace
{
    constexpr size_t kTunedMapping = 1;
    constexpr size_t kTunedStagger = 16;

    hip::HipAMDGPU makeDevice(AMDGPU::Processor processor)
    {
        hip::HipAMDGPU device;
        device.processor        = processor;
        device.computeUnitCount = 40;
        return device;
    }

    void initSolution(ContractionSolution& solution, size_t staggerU)
    {
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::None;
        solution.sizeMapping.staggerU               = staggerU;
        solution.sizeMapping.staggerUMapping        = kTunedMapping;
        solution.sizeMapping.staggerStrideShift     = 2;
    }

    // Float GEMM, n = 256. TN: packed lda = ldb = k. NN: packed lda = m, ldb = k.
    ContractionProblemGemm
        makeProblem(size_t k, size_t lda, size_t ldb, bool transA = true, size_t m = 256)
    {
        auto problem
            = ContractionProblemGemm::GEMM(transA, false, m, 256, k, lda, ldb, m, 1.0, false, 1);
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    size_t mappingFor(ContractionProblemGemm const& problem,
                      AMDGPU::Processor             processor = AMDGPU::Processor::gfx1151,
                      size_t                        staggerU  = kTunedStagger)
    {
        auto                device = makeDevice(processor);
        ContractionSolution solution;
        initSolution(solution, staggerU);
        return std::get<0>(solution.calculateAutoStaggerU(problem, &device, 0, 1));
    }
}

TEST(StaggerUStrideTest, PackedAliasedStridesKeepTunedMapping)
{
    // 512 floats = 2048 B: aliased, as tuned.
    EXPECT_EQ(mappingFor(makeProblem(512, 512, 512)), kTunedMapping);
}

TEST(StaggerUStrideTest, PaddedAwayAliasingUsesMapping4)
{
    EXPECT_EQ(mappingFor(makeProblem(512, 528, 528)), 4u);
    // NN, m = 500: only B would alias when packed.
    EXPECT_EQ(mappingFor(makeProblem(512, 500, 528, false, 500)), 4u);
}

TEST(StaggerUStrideTest, RemainingAliasedStrideKeepsTunedMapping)
{
    EXPECT_EQ(mappingFor(makeProblem(512, 528, 512)), kTunedMapping);
    EXPECT_EQ(mappingFor(makeProblem(512, 512, 528)), kTunedMapping);
}

TEST(StaggerUStrideTest, PaddingUnaliasedPackedStrideKeepsTunedMapping)
{
    // 500 floats = 2000 B is not aliased when packed, so padding changes nothing.
    EXPECT_EQ(mappingFor(makeProblem(500, 516, 516)), kTunedMapping);
}

TEST(StaggerUStrideTest, RuleLimitedToGfx1151AndStaggeredSolutions)
{
    auto padded = makeProblem(512, 528, 528);
    EXPECT_EQ(mappingFor(padded, AMDGPU::Processor::gfx950), kTunedMapping);
    EXPECT_EQ(mappingFor(padded, AMDGPU::Processor::gfx1151, 0), kTunedMapping);
}
