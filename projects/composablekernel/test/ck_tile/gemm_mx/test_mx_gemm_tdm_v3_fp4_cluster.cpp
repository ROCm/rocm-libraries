// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/**
 * @file
 * @brief MX FP4 GEMM on comp_tdm_v3 (gfx1250) under a 4x4 cluster launch, with the register
 *        double buffer and the scales staged through LDS
 *        (GemmPipelineAgBgCrCompTDMV3LdsScalePolicy): the A data and scale TDMs multicast to
 *        the workgroups of the cluster that share the A tile, the B ones to those that share the
 *        B tile, and the K loop has no cluster barrier. fp4 x fp4 with 32-element E8M0 scales,
 *        fp16 C, TDM epilogue, warps 2x2x1, block tiles 256x256x512 and 256x256x256.
 *
 * Shapes are sized for the gfx1250 emulator smoke stage, each with three K tiles (the hot loop
 * runs once and ends on one tile): one full cluster (4x4 tiles); a 3x3 tile grid padded to one
 * 4x4 cluster (the out-of-range workgroups exit, and their bits stay in their peers' masks); and
 * two clusters along M, the second one padded (5x4 tiles), so that a nonzero cluster index runs.
 *
 * The emulator does not model the workgroup mask (it delivers every TDM to its issuer only), so
 * these rows check the cluster tile walk, the descriptors and the results, not multicast
 * delivery.
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmV3Fp4Cluster
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3Fp4Cluster<T>>
{
    using Base = TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3Fp4Cluster<T>>;

    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(Base::check_data_type(),
                  "every comp_tdm_v3 MX FP4 cluster row must have WMMA traits");
    static_assert(Base::ClusterLaunch && Base::M_Cluster == 4 && Base::N_Cluster == 4,
                  "every row launches 4x4 clusters");
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmV3Fp4Cluster, KernelTypesMxGemmTdmV3Fp4Cluster);

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4Cluster, OneCluster)
{
    this->Run(4 * TestFixture::M_Tile, 4 * TestFixture::N_Tile, 3 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4Cluster, PaddedCluster)
{
    this->Run(3 * TestFixture::M_Tile, 3 * TestFixture::N_Tile, 3 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmV3Fp4Cluster, TwoClustersAlongM)
{
    this->Run(5 * TestFixture::M_Tile, 4 * TestFixture::N_Tile, 3 * TestFixture::K_Tile);
}
