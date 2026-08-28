/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Unit tests for the bound getSKGrid puts on a Stream-K grid.
//
// A Stream-K flag region is one block of StreamKFlagElements ints, private to
// one (stream, problem) pair. The dynamic-queue kernels (StreamK 4, and the
// StreamK 4 sub-path of StreamK 5) put the per-XCD work-queue counters at the
// base of that block and start the ready flags after them, so they can index
// fewer entries than the block holds. A grid sized against the whole block
// would run its last workgroups off the end and into the next stream's flags,
// which is the corruption the per-stream blocks exist to prevent.
//
// The heuristic grids stay far below either bound (gfx950 picks 224), so the
// bound is reached only through TENSILE_STREAMK_FIXED_GRID. That variable is
// latched into a function-local static on first read, so it cannot be set from
// inside a running test; these drive AMDGPU::skFixedGrid, which is the field it
// feeds, directly.

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <memory>
#include <tuple>

namespace
{
    using TensileLite::ContractionProblemGemm;
    using TensileLite::ContractionSolution;
    using TensileLite::StreamKFlagElements;

    constexpr size_t kCuCount = 256; // gfx950 SPX

    // gfx950 has 8 XCDs and a 128-byte cache line, so the counters take
    // 8 * 128 = 1024 bytes = 256 ints before the first flag.
    constexpr size_t kQueuePrefixElements = 256;

    TensileLite::hip::HipAMDGPU makeGfx950Device()
    {
        using arch_t = origami::hardware_t::architecture_t;
        origami::hardware_t hw(arch_t::gfx950,
                               kCuCount,
                               163840,
                               262144,
                               8,
                               1.0,
                               1.0,
                               1.0,
                               4000000,
                               1.2,
                               1,
                               std::make_tuple(0.0, 0.008, 0.0));

        TensileLite::hip::HipAMDGPU device;
        device.processor          = TensileLite::AMDGPU::Processor::gfx950;
        device.computeUnitCount   = static_cast<int>(kCuCount);
        device.deviceName         = "test-gfx950-analytical";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        // Pin the grid the clamp has to cut back. Anything that would pick a
        // grid of its own is turned off so the bound is the only thing acting.
        device.skDynamicGrid = 0;
        device.skFixedGrid   = static_cast<int>(StreamKFlagElements);
        return device;
    }

    // ContractionSolution is not copyable, so it is filled in place.
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

    // 64 x 63 tiles: not a multiple of the fixed grid, so partial tiles exist
    // and the flags are read. K is small enough that the tree-fixup bounds
    // above the clamp leave the grid alone.
    ContractionProblemGemm makeProblem()
    {
        return ContractionProblemGemm::GEMM(
            false, false, 8192, 8064, 512, 8192, 512, 8192, 1.0, false, 1);
    }

    TEST(StreamKFlagBound, DynamicQueueGridStopsBeforeTheNextBlock)
    {
        ContractionSolution solution;
        initStreamKSolution(solution, 4);
        auto device  = makeGfx950Device();
        auto problem = makeProblem();
        auto tiles   = problem.getNumTiles(solution.sizeMapping, 1);

        ASSERT_NE(tiles % StreamKFlagElements, 0u) << "grid must leave partial tiles to fix up";

        EXPECT_EQ(solution.getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  StreamKFlagElements - kQueuePrefixElements);
    }

    TEST(StreamKFlagBound, StaticGridKeepsTheWholeBlock)
    {
        // StreamK 3 indexes its flags from offset 0, so tightening it for the
        // work-queue prefix would cost it grid it is entitled to.
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        auto device  = makeGfx950Device();
        auto problem = makeProblem();
        auto tiles   = problem.getNumTiles(solution.sizeMapping, 1);

        ASSERT_NE(tiles % StreamKFlagElements, 0u);

        EXPECT_EQ(solution.getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  StreamKFlagElements);
    }
} // namespace
