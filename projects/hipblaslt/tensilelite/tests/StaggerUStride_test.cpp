// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// calculateAutoStaggerU on gfx1151: when the caller pads an A/B leading dimension that would
// alias DRAM channels (multiple of 2048 B) when packed, the solution's StaggerU is disabled.
// Host-only, no device library required.

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>

using namespace TensileLite;

namespace
{
    using StaggerParams = std::tuple<size_t, size_t, size_t>; // mapping, StaggerU, stride shift

    constexpr size_t        kTunedStagger = 16;
    constexpr StaggerParams kTuned{StaggerUMapping::WG1, kTunedStagger, 2};
    constexpr StaggerParams kNoStagger{StaggerUMapping::WG0, 0, 0};

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
        solution.sizeMapping.staggerUMapping        = std::get<0>(kTuned);
        solution.sizeMapping.staggerStrideShift     = std::get<2>(kTuned);
    }

    // Float GEMM, n = 256. TN: packed lda = ldb = k. NN: packed lda = m, ldb = k.
    ContractionProblemGemm makeProblem(
        size_t k, size_t lda, size_t ldb, bool transA = true, size_t m = 256, bool transB = false)
    {
        auto problem
            = ContractionProblemGemm::GEMM(transA, transB, m, 256, k, lda, ldb, m, 1.0, false, 1);
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    StaggerParams staggerFor(ContractionProblemGemm const& problem,
                             AMDGPU::Processor             processor = AMDGPU::Processor::gfx1151,
                             size_t                        staggerU  = kTunedStagger)
    {
        auto                device = makeDevice(processor);
        ContractionSolution solution;
        initSolution(solution, staggerU);
        return solution.calculateAutoStaggerU(problem, &device, 0, 1);
    }
}

TEST(StaggerUStrideTest, PackedAliasedStridesKeepTunedStagger)
{
    // 512 floats = 2048 B: aliased, as tuned.
    EXPECT_EQ(staggerFor(makeProblem(512, 512, 512)), kTuned);
}

TEST(StaggerUStrideTest, PaddedAwayAliasingDisablesStagger)
{
    EXPECT_EQ(staggerFor(makeProblem(512, 528, 528)), kNoStagger);
    // NN, m = 500: only B would alias when packed.
    EXPECT_EQ(staggerFor(makeProblem(512, 500, 528, false, 500)), kNoStagger);
}

TEST(StaggerUStrideTest, RemainingAliasedStrideKeepsTunedStagger)
{
    EXPECT_EQ(staggerFor(makeProblem(512, 528, 512)), kTuned);
    EXPECT_EQ(staggerFor(makeProblem(512, 512, 528)), kTuned);
}

TEST(StaggerUStrideTest, PaddingUnaliasedPackedStrideKeepsTunedStagger)
{
    // 500 floats = 2000 B is not aliased when packed, so padding changes nothing.
    EXPECT_EQ(staggerFor(makeProblem(500, 516, 516)), kTuned);
}

TEST(StaggerUStrideTest, TTLayoutKeepsTunedStagger)
{
    // TT: packed lda = k = 512 floats = 2048 B, padded; ldb = n = 256 is packed.
    EXPECT_EQ(staggerFor(makeProblem(512, 528, 256, true, 256, true)), kTuned);
}

// Leading dimensions above 2^24 are not representable in float, where elementBytes() lives.
TEST(StaggerUStrideTest, LargeStridesUseExactByteArithmetic)
{
    constexpr size_t k = size_t{1} << 24;
    // Packed k + 1 floats = 2^26 + 4 B is not aliased (float math rounds it to 2^26 and would
    // fire); padding by 16 does not change that.
    EXPECT_EQ(staggerFor(makeProblem(k + 1, k + 16, k + 16)), kTuned);
    // Packed 2^24 floats = 2^26 B is aliased; the padded k + 1 is not (float math rounds the
    // padded stride back to 2^26 and would keep the stagger).
    EXPECT_EQ(staggerFor(makeProblem(k, k + 1, k + 1)), kNoStagger);
}

TEST(StaggerUStrideTest, RuleLimitedToGfx1151AndStaggeredSolutions)
{
    auto padded = makeProblem(512, 528, 528);
    EXPECT_EQ(staggerFor(padded, AMDGPU::Processor::gfx950), kTuned);
    EXPECT_EQ(std::get<1>(staggerFor(padded, AMDGPU::Processor::gfx1151, 0)), 0u);
    EXPECT_EQ(std::get<0>(staggerFor(padded, AMDGPU::Processor::gfx1151, 0)), std::get<0>(kTuned));
}
