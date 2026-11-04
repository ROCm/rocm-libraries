// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of the cluster launch guard in ck_tile/host/kernel_launch.hpp and of how
// TDMConfig takes its workgroup mask.
//
// A kernel whose TDM loads multicast across a workgroup cluster must be launched as a cluster of
// more than one workgroup. kernel_requires_cluster_launch_v names those kernels, and the cluster
// overload of make_kernel throws for them when the cluster has a single workgroup, before any HIP
// call. Nothing is launched: make_kernel only returns the launch callable, these tests never call
// it, the fake kernels have empty bodies, and the TDM GEMM pipelines are only inspected.
//
// TDMConfig holds the mask as a TDMWorkgroupMask, which only TDMWorkgroupMask::checked makes from
// an integer. Its check reads a device register and so cannot run here; the last test checks that
// TDMWorkgroupMask cannot be constructed from or converted from an integer.

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/tdm.hpp"

// Kernels have external linkage: in an anonymous namespace the host pass, which never sees the
// device call, rejects their operator() as an unused member function (-Werror).
namespace tdm_cluster_launch_guard_test {

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
    CK_TILE_DEVICE void operator()() const {}
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

// The real TDM copy kernel, which reports cluster launch through ClusterEnable. It is only
// inspected, never passed to make_kernel.
template <bool ClusterEnable>
using CopyKernel = ck_tile::TDMCopyKernel<
    ck_tile::TDMPipelineProblem<ck_tile::TDMTileShape<2,
                                                      ck_tile::sequence<16, 16>,
                                                      ck_tile::sequence<1, 1>,
                                                      ck_tile::sequence<16, 16>>,
                                ck_tile::TDMPipelineTraits<ck_tile::half_t,
                                                           ck_tile::tensor_layout::gemm::RowMajor,
                                                           uint16_t,
                                                           false, /*AtomicBarrierEnable_*/
                                                           false, /*IsGatherMode_*/
                                                           false, /*IterateEnable_*/
                                                           false, /*PadEnable_*/
                                                           false, /*EarlyTimeOutEnable_*/
                                                           ClusterEnable>>>;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using F16 = ck_tile::half_t;

// The f16 TDM V1/V2 GEMM pipelines of the TDM GEMM tests (64x64x32 block tile, 2x2x1 warps,
// 16x16x32 warp tile, row-major A, column-major B) over a plain or a cluster tile shape.
using TdmBlockTile  = ck_tile::sequence<64, 64, 32>;
using TdmBlockWarps = ck_tile::sequence<2, 2, 1>;
using TdmWarpTile   = ck_tile::sequence<16, 16, 32>;
using TdmPlainShape = ck_tile::TileGemmShape<TdmBlockTile, TdmBlockWarps, TdmWarpTile>;
template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
using TdmClusterShape = ck_tile::
    ClusterTileGemmShape<ck_tile::sequence<M, N, K>, TdmBlockTile, TdmBlockWarps, TdmWarpTile>;

// The 2x2x1 cluster of the TDM GEMM cluster tests, a cluster along one dimension, and a cluster
// of a single workgroup.
using Cluster2x2x1Shape = TdmClusterShape<2, 2, 1>;
using Cluster1x2x1Shape = TdmClusterShape<1, 2, 1>;
using Cluster1x1x1Shape = TdmClusterShape<1, 1, 1>;

// No padding and a double LDS buffer, as in the TDM GEMM tests.
using TdmTraits = ck_tile::TileGemmUniversalTraits<false, false, false, true, Row, Col, Row>;
template <typename Shape>
using TdmProblem = ck_tile::UniversalGemmPipelineProblem<F16, F16, float, Shape, TdmTraits>;
template <typename Shape>
using TdmV1Kernel = FakeGemmKernel<ck_tile::GemmPipelineAgBgCrCompTDMV1<TdmProblem<Shape>>>;
template <typename Shape>
using TdmV2Kernel = FakeGemmKernel<ck_tile::GemmPipelineAgBgCrCompTDMV2<TdmProblem<Shape>>>;

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
TEST(TDMClusterLaunchGuard, DetectsTdmGemmPipelinesByClusterSize)
{
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TdmV1Kernel<Cluster2x2x1Shape>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TdmV2Kernel<Cluster2x2x1Shape>>);
    EXPECT_TRUE(ck_tile::kernel_requires_cluster_launch_v<TdmV1Kernel<Cluster1x2x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TdmV1Kernel<Cluster1x1x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TdmV2Kernel<Cluster1x1x1Shape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TdmV1Kernel<TdmPlainShape>>);
    EXPECT_FALSE(ck_tile::kernel_requires_cluster_launch_v<TdmV2Kernel<TdmPlainShape>>);
}

TEST(TDMClusterLaunchGuard, ClusterKernelRejectsSingleWorkgroupCluster)
{
    EXPECT_THROW(ck_tile::make_kernel(ClusterKernel{}, dim3(1, 1, 1), kGrid, kBlock, 0),
                 std::runtime_error);
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
    CK_TILE_DEVICE explicit FakeOpenMask(uint16_t) {}
};

TEST(TDMWorkgroupMask, NonZeroMaskOnlyThroughChecked)
{
    EXPECT_TRUE(
        (std::is_same_v<decltype(ck_tile::TDMConfig::workgroup_mask), ck_tile::TDMWorkgroupMask>));
    EXPECT_TRUE((std::is_constructible_v<FakeOpenMask, uint16_t>));
    EXPECT_FALSE((std::is_constructible_v<ck_tile::TDMWorkgroupMask, uint16_t>));
    EXPECT_FALSE((std::is_convertible_v<uint16_t, ck_tile::TDMWorkgroupMask>));
    EXPECT_EQ(ck_tile::TDMConfig{}.workgroup_mask.value(), 0);
}

} // namespace tdm_cluster_launch_guard_test
