// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/OrigamiConfig.hpp>

namespace TensileLite
{
    TEST(OrigamiConfig, PropagatesGeneratedResourcesAndScheduling)
    {
        ContractionSolution solution;
        auto& mapping                    = solution.sizeMapping;
        mapping.macroTile                = {128, 80, 1};
        mapping.depthU                   = 64;
        mapping.matrixInstruction        = {16, 16, 16, 1};
        mapping.CUOccupancy              = 2;
        mapping.workGroupMapping         = 8;
        mapping.workGroupSize            = {64, 2, 1};
        mapping.waveGroup                = {4, 1};
        mapping.waveNum                  = 4;
        mapping.PrefetchGlobalRead       = 1;
        mapping.prefetchLocalRead        = 1;
        mapping.scheduleIterAlg          = 3;
        mapping.oneLDSBuffer             = false;
        mapping.transposeLDS             = 1;
        mapping.sourceSwap               = true;
        mapping.localReadVectorWidth     = 16;
        mapping.streamK                  = 3;
        mapping.streamKForceDPOnly       = 0;
        mapping.staggerU                 = 32;
        mapping.staggerUMapping          = 1;
        mapping.totalVgprs               = 96;
        mapping.accumulatorVgprs         = 80;
        mapping.totalSgprs               = 94;
        mapping.ldsBytes                 = 62720;
        mapping.scratchBytes             = 0;
        mapping.grvwA                    = 8;
        mapping.grvwB                    = 8;
        mapping.gwvwD                    = 1;
        mapping.VectorWidthA             = 1;
        mapping.VectorWidthB             = 1;
        mapping.NumLoadsCoalescedA       = 1;
        mapping.NumLoadsCoalescedB       = 1;
        mapping.LocalSplitU              = 1;

        auto config = makeOrigamiConfig(solution, 7);
        auto const& tensile = config.tensile();

        EXPECT_EQ(config.index, 7);
        EXPECT_EQ(config.mt.m, 128);
        EXPECT_EQ(config.mt.n, 80);
        EXPECT_EQ(config.mt.k, 64);
        EXPECT_EQ(config.occupancy, 2);
        EXPECT_EQ(tensile.stream_k, 3);
        EXPECT_EQ(tensile.depth_u, 64);
        EXPECT_EQ(tensile.schedule_iter_alg, 3);
        EXPECT_EQ(tensile.prefetch_global_read, 1);
        EXPECT_EQ(tensile.prefetch_local_read, 1);
        EXPECT_EQ(tensile.transpose_lds, 1);
        EXPECT_TRUE(tensile.source_swap);
        EXPECT_EQ(tensile.local_read_vector_width, 16);
        EXPECT_EQ(tensile.total_vgprs, 96);
        EXPECT_EQ(tensile.accumulator_vgprs, 80);
        EXPECT_EQ(tensile.total_sgprs, 94);
        EXPECT_EQ(tensile.lds_bytes, 62720);
        EXPECT_EQ(tensile.scratch_bytes, 0);
        EXPECT_EQ(tensile.threads_per_workgroup, 128);
        EXPECT_EQ(tensile.compiled_cu_occupancy, 2);
    }
} // namespace TensileLite
