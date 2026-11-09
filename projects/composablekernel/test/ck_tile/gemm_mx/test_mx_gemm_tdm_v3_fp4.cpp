// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/**
 * @file
 * @brief MX FP4 GEMM on comp_tdm_v3 (gfx1250) at four waves (warps 2x2x1, warp tile 32x32x128)
 *        with the register double buffer (GemmPipelineAgBgCrCompTDMV3RegDoubleBufferPolicy) at
 *        block tiles 256x128x256 and 128x128x256, and with the scales staged through LDS as well
 *        (GemmPipelineAgBgCrCompTDMV3LdsScalePolicy) at 256x256x512 (four K sub-tiles) and
 *        256x256x256: fp4 x fp4 with 32-element E8M0 scales, fp16 C, TDM epilogue.
 *
 * Shapes are sized for the gfx1250 emulator smoke stage: one K tile (no K loop), and a 2x2 grid of
 * workgroups with three and four K tiles. The K loop holds two K tiles per trip, so three K tiles
 * run the hot loop once and end on one tile, four run it once and end on two: both tail paths
 * after a hot loop. (One or two K tiles never run the hot loop.)
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmV3Fp4
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3Fp4<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3Fp4<T>>::check_data_type(),
        "every comp_tdm_v3 MX FP4 row must have WMMA traits");
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmV3Fp4, KernelTypesMxGemmTdmV3Fp4);

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4, HotLoopOneTileTail)
{
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, 3 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4, HotLoopTwoTileTail)
{
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
}
