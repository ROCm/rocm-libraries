// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of the cluster launch guard in ck_tile/host/kernel_launch.hpp and of how
// TDMConfig takes its workgroup mask.
//
// A kernel whose TDM loads multicast across a workgroup cluster must be launched as a cluster of
// more than one workgroup. kernel_requires_cluster_launch_v names those kernels, and the cluster
// overload of make_kernel throws for them when the cluster has a single workgroup, before any HIP
// call. Nothing is launched: make_kernel only returns the launch callable, these tests never call
// it, the fake kernels have empty bodies, and the TDM GEMM pipelines and kernels are only
// inspected.
//
// TDMConfig holds the mask as a TDMWorkgroupMask, which only TDMWorkgroupMask::checked makes from
// an integer. That check reads a device register and so cannot run here (tdm_hw_cluster_id.cpp
// reads the register on the device); the last test checks that the mask defaults to 0 and cannot
// be constructed or converted from an integer.

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/tdm.hpp"

namespace ck_tile {
namespace test {
namespace {

template <bool Value>
struct FakePipeline
{
    static constexpr bool UseClusterLaunch = Value;
};

// A pipeline without TDM multicast has no UseClusterLaunch member.
struct FakePipelineWithoutClusterLaunch
{
};

struct FakeKernel
{
    static constexpr ck_tile::index_t kBlockSize = 32;
    // Only the device pass odr-uses this; see the note in tdm_descriptor_stride.cpp.
    [[maybe_unused]] CK_TILE_DEVICE void operator()() const {}
};

template <typename Pipeline>
struct FakeGemmKernel : FakeKernel
{
    using GemmPipeline = Pipeline;
};

template <typename Pipeline>
struct FakeFlatmmKernel : FakeKernel
{
    using FlatmmPipeline = Pipeline;
};

// Mirrors MXFlatmmKernel, which derives from FlatmmKernel and inherits FlatmmPipeline.
template <typename Pipeline>
struct FakeMXFlatmmKernel : FakeFlatmmKernel<Pipeline>
{
    using MXFlatmmPipeline = Pipeline;
};

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using F16 = ck_tile::half_t;

// The real TDM copy kernel, which reports cluster launch through ClusterEnable. It is only
// inspected, never passed to make_kernel.
template <bool ClusterEnable>
using CopyKernel = ck_tile::TDMCopyKernel<
    ck_tile::TDMPipelineProblem<ck_tile::TDMTileShape<2,
                                                      ck_tile::sequence<16, 16>,
                                                      ck_tile::sequence<1, 1>,
                                                      ck_tile::sequence<16, 16>>,
                                ck_tile::TDMPipelineTraits<F16,
                                                           Row,
                                                           uint16_t,
                                                           false, /*AtomicBarrierEnable_*/
                                                           false, /*IsGatherMode_*/
                                                           false, /*IterateEnable_*/
                                                           false, /*PadEnable_*/
                                                           false, /*EarlyTimeOutEnable_*/
                                                           ClusterEnable>>>;

// The f16 TDM V1/V2/V3 GEMM pipelines of the TDM GEMM tests (64x64x32 block tile, 2x2x1 warps,
// 16x16x32 warp tile, row-major A, column-major B) over a plain or a cluster tile shape.
using TDMBlockTile  = ck_tile::sequence<64, 64, 32>;
using TDMBlockWarps = ck_tile::sequence<2, 2, 1>;
using TDMWarpTile   = ck_tile::sequence<16, 16, 32>;
using TDMPlainShape = ck_tile::TileGemmShape<TDMBlockTile, TDMBlockWarps, TDMWarpTile>;
template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
using TDMClusterShape = ck_tile::
    ClusterTileGemmShape<ck_tile::sequence<M, N, K>, TDMBlockTile, TDMBlockWarps, TDMWarpTile>;

// The 2x2x1 cluster of the TDM GEMM cluster tests, a cluster along one dimension, and a cluster
// of a single workgroup.
using Cluster2x2x1Shape = TDMClusterShape<2, 2, 1>;
using Cluster1x2x1Shape = TDMClusterShape<1, 2, 1>;
using Cluster1x1x1Shape = TDMClusterShape<1, 1, 1>;

// No padding and a double LDS buffer, as in the TDM GEMM tests.
using TDMTraits = ck_tile::TileGemmUniversalTraits<false, false, false, true, Row, Col, Row>;
template <typename Shape>
using TDMProblem = ck_tile::UniversalGemmPipelineProblem<F16, F16, float, Shape, TDMTraits>;
template <typename Shape>
using TDMV1Kernel = FakeGemmKernel<ck_tile::GemmPipelineAgBgCrCompTDMV1<TDMProblem<Shape>>>;
template <typename Shape>
using TDMV2Kernel = FakeGemmKernel<ck_tile::GemmPipelineAgBgCrCompTDMV2<TDMProblem<Shape>>>;
template <typename Shape>
using TDMV3Kernel = FakeGemmKernel<ck_tile::GemmPipelineAgBgCrCompTDMV3<TDMProblem<Shape>>>;

// The C-shuffle epilogue for the 64x64 block tile, 2x2 warps and 16x16x32 warp tile above.
using TDMEpilogue =
    ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<F16,
                                                               F16,
                                                               ck_tile::tuple<>,
                                                               float,
                                                               F16,
                                                               ck_tile::tuple<>,
                                                               Row,
                                                               ck_tile::element_wise::PassThrough,
                                                               64, /*kM_*/
                                                               64, /*kN_*/
                                                               2,  /*MWave_*/
                                                               2,  /*NWave_*/
                                                               16, /*MPerXdl_*/
                                                               16, /*NPerXdl_*/
                                                               32, /*KPerXdl_*/
                                                               false /*isCTransposed_*/>>;
// The real GEMM kernel over TDM V1, its pipeline over the tile shape of the partitioner, as in the
// TDM GEMM tests. It is only inspected, never passed to make_kernel.
template <typename Partitioner>
using TDMV1GemmKernel = ck_tile::GemmKernel<
    Partitioner,
    ck_tile::GemmPipelineAgBgCrCompTDMV1<TDMProblem<typename Partitioner::BlockGemmShape>>,
    TDMEpilogue>;

using ClusterKernel    = FakeGemmKernel<FakePipeline<true>>;
using NonClusterKernel = FakeGemmKernel<FakePipeline<false>>;

constexpr dim3 kGrid(4);
constexpr dim3 kBlock(FakeKernel::kBlockSize);

TEST(TDMClusterLaunchGuard, DetectsClusterKernels)
{
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<FakeGemmKernel<FakePipeline<true>>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<FakeFlatmmKernel<FakePipeline<true>>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<FakeMXFlatmmKernel<FakePipeline<true>>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<CopyKernel<true>>);
}

TEST(TDMClusterLaunchGuard, IgnoresNonClusterKernels)
{
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<FakeGemmKernel<FakePipeline<false>>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<FakeFlatmmKernel<FakePipeline<false>>>);
    EXPECT_FALSE(
        ck_tile::kernel_requires_cluster_launch_v<FakeMXFlatmmKernel<FakePipeline<false>>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<CopyKernel<false>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<
                 FakeGemmKernel<FakePipelineWithoutClusterLaunch>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<FakeKernel>);
}

// The TDM GEMM pipelines multicast only in a cluster of more than one workgroup, so a cluster tile
// shape of a single workgroup does not make them cluster kernels.
TEST(TDMClusterLaunchGuard, DetectsTDMGemmPipelinesByClusterSize)
{
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TDMV1Kernel<Cluster2x2x1Shape>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TDMV2Kernel<Cluster2x2x1Shape>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TDMV3Kernel<Cluster2x2x1Shape>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TDMV1Kernel<Cluster1x2x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV1Kernel<Cluster1x1x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV2Kernel<Cluster1x1x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV3Kernel<Cluster1x1x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV1Kernel<TDMPlainShape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV2Kernel<TDMPlainShape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TDMV3Kernel<TDMPlainShape>>);
}

// The trait reads the pipeline through the kernel's GemmPipeline alias, so it must see it on the
// real GemmKernel too, and agree with the ClusterSize() that callers launch the kernel with.
TEST(TDMClusterLaunchGuard, DetectsRealGemmKernel)
{
    using ClusterGemmKernel =
        TDMV1GemmKernel<ck_tile::GemmClusterTilePartitioner<Cluster2x2x1Shape>>;
    using PlainGemmKernel =
        TDMV1GemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<TDMPlainShape, 8, 4>>;
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<ClusterGemmKernel>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<PlainGemmKernel>);

    const dim3 cluster = ClusterGemmKernel::ClusterSize();
    EXPECT_GT(cluster.x * cluster.y * cluster.z, 1u);
    const dim3 plain = PlainGemmKernel::ClusterSize();
    EXPECT_EQ(plain.x * plain.y * plain.z, 1u);
}

// The message names the cluster that make_kernel received.
TEST(TDMClusterLaunchGuard, ClusterKernelRejectsSingleWorkgroupCluster)
{
    try
    {
        ck_tile::make_kernel(ClusterKernel{}, dim3(1, 1, 1), kGrid, kBlock, 0);
        FAIL() << "make_kernel accepted a 1x1x1 cluster for a kernel that requires a cluster";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("1x1x1"), std::string::npos) << e.what();
    }
}

TEST(TDMClusterLaunchGuard, ClusterKernelAcceptsClusterAlongAnyDim)
{
    for(const dim3& cluster : {dim3(2, 1, 1), dim3(1, 2, 1), dim3(1, 1, 2), dim3(2, 2, 1)})
    {
        EXPECT_NO_THROW(ck_tile::make_kernel(ClusterKernel{}, cluster, kGrid, kBlock, 0))
            << cluster.x << "x" << cluster.y << "x" << cluster.z;
    }
}

TEST(TDMClusterLaunchGuard, NonClusterKernelAcceptsSingleWorkgroupCluster)
{
    EXPECT_NO_THROW(ck_tile::make_kernel(NonClusterKernel{}, dim3(1, 1, 1), kGrid, kBlock, 0));
}

// Control for the construction checks below: constructible from an integer through a public
// constructor that, like the private one of TDMWorkgroupMask, is device-only.
struct FakeOpenMask
{
    [[maybe_unused]] CK_TILE_DEVICE explicit FakeOpenMask(uint16_t) {}
};

TEST(TDMWorkgroupMask, NonZeroMaskOnlyThroughChecked)
{
    EXPECT_TRUE(
        (std::is_same_v<decltype(ck_tile::TDMConfig::workgroup_mask), ck_tile::TDMWorkgroupMask>));
    EXPECT_TRUE((std::is_constructible_v<FakeOpenMask, uint16_t>));
    // The static_assert next to TDMWorkgroupMask already enforces these two at compile time; they
    // are restated here beside the control.
    EXPECT_FALSE((std::is_constructible_v<ck_tile::TDMWorkgroupMask, uint16_t>));
    EXPECT_FALSE((std::is_convertible_v<uint16_t, ck_tile::TDMWorkgroupMask>));
    EXPECT_EQ(ck_tile::TDMConfig{}.workgroup_mask.value(), 0);
}

} // namespace
} // namespace test
} // namespace ck_tile
