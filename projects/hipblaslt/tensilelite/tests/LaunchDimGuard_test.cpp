// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Guards the 32-bit narrowing at the launch boundary (ROCM-31290).
//
// KernelInvocation carries its grid in dim3, which is vector3<size_t>, but both
// launch APIs take 32-bit grid parameters: hipExtModuleLaunchKernel's
// globalWorkSize arguments are `unsigned int`, and HIP_LAUNCH_CONFIG's gridDim
// fields are too. A grid that does not fit used to be narrowed silently rather
// than refused. Measured on gfx950 at 256 threads, 2^24 + 16 workgroups wrapped
// to a 16-workgroup launch and left the other 16,777,216 workgroups' worth of D
// holding whatever was there before, with no error raised.
//
// Solution selection cannot see every path to an oversized grid: LaunchLimits
// exempts Stream-K, whose insufficient-workspace fallback still launches one
// workgroup per output tile, and skFixedGrid / skGridMultiplier override the
// grid after selection. SolutionAdapter::launchKernel is where Tensile's own
// launches narrow the grid, so that is where the refusal lives. (rocblaslt's
// rocRoller custom kernels launch directly and are not covered here.)
//
// These tests drive launchKernel directly with a synthetic KernelInvocation.
// The check runs before the code object is looked up, so no kernel needs to
// exist and no GPU work is dispatched; an oversized grid must come back as
// hipErrorInvalidValue rather than reaching HIP at all.
//
// The negative direction (safe grids still launch) is covered by the rest of
// this suite and by hipblaslt-test, every passing case of which launches a real
// kernel through this same function.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

#include <cstdint>
#include <limits>
#include <sstream>

namespace
{
    constexpr size_t kUintMax = std::numeric_limits<unsigned int>::max();

    // A kernel name that no code object provides. The guard runs ahead of the
    // lookup, so a refused launch never gets far enough to care; if the guard
    // ever regressed, the lookup would fail with some other error instead,
    // which is what the assertions below distinguish.
    TensileLite::KernelInvocation makeInvocation(TensileLite::dim3 const& workGroupSize,
                                                 TensileLite::dim3 const& numWorkGroups)
    {
        TensileLite::KernelInvocation k;
        k.kernelName     = "LaunchDimGuard_test_no_such_kernel";
        k.workGroupSize  = workGroupSize;
        k.numWorkGroups  = numWorkGroups;
        k.numWorkItems   = {workGroupSize.x * numWorkGroups.x,
                            workGroupSize.y * numWorkGroups.y,
                            workGroupSize.z * numWorkGroups.z};
        k.sharedMemBytes = 0;
        return k;
    }
}

// The shape ROCM-31290 was measured at: 256 threads and 2^24 workgroups is
// exactly 2^32 work items, one past what the launch parameter can hold.
TEST(LaunchDimGuard, WorkItemProductAtTwoPow32_Refused)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    auto kernel = makeInvocation(/*workGroupSize=*/{256, 1, 1},
                                 /*numWorkGroups=*/{size_t(1) << 24, 1, 1});

    ASSERT_EQ(kernel.numWorkItems.x, size_t(1) << 32)
        << "the fixture must actually land on 2^32 work items";

    EXPECT_EQ(adapter.launchKernel(kernel), hipErrorInvalidValue)
        << "a work-item count of 2^32 narrows to a zero-sized grid and must be refused "
           "rather than passed to hipExtModuleLaunchKernel";
}

// The silent case: 2^24 + 16 workgroups at 256 threads narrowed to 4096 work
// items, so 16 workgroups ran and the rest of D was never written.
TEST(LaunchDimGuard, WorkItemProductPastTwoPow32_Refused)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    auto kernel = makeInvocation(/*workGroupSize=*/{256, 1, 1},
                                 /*numWorkGroups=*/{(size_t(1) << 24) + 16, 1, 1});

    ASSERT_GT(kernel.numWorkItems.x, kUintMax);

    EXPECT_EQ(adapter.launchKernel(kernel), hipErrorInvalidValue)
        << "this is the shape that silently dropped stores on gfx950";
}

// The largest grid that still fits must not be caught by the guard. It cannot
// launch here (the kernel does not exist), but it has to fail for that reason
// and not this one, which is what pins the boundary to the right value.
TEST(LaunchDimGuard, LargestFittingWorkItemProduct_NotRefusedByTheGuard)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    // 2^32 - 256 work items: the largest 256-thread grid under the limit.
    auto kernel = makeInvocation(/*workGroupSize=*/{256, 1, 1},
                                 /*numWorkGroups=*/{(size_t(1) << 24) - 1, 1, 1});

    ASSERT_LE(kernel.numWorkItems.x, kUintMax) << "the fixture must stay under the limit";

    EXPECT_NE(adapter.launchKernel(kernel), hipErrorInvalidValue)
        << "a grid that fits in 32 bits must reach the kernel lookup instead of being "
           "rejected by the launch-dimension guard";
}

// The y and z dimensions feed the same 32-bit parameters as x.
TEST(LaunchDimGuard, OversizedYDimension_Refused)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    auto kernel = makeInvocation(/*workGroupSize=*/{1, 256, 1},
                                 /*numWorkGroups=*/{1, (size_t(1) << 24) + 16, 1});

    ASSERT_GT(kernel.numWorkItems.y, kUintMax);

    EXPECT_EQ(adapter.launchKernel(kernel), hipErrorInvalidValue);
}

// Cluster launches enumerate the grid in workgroups rather than work items, so
// that path has to be bounded against numWorkGroups. A workgroup count that
// fits must pass even when the work-item product does not, since the work-item
// product is never handed to hipDrvLaunchKernelEx.
TEST(LaunchDimGuard, ClusterLaunchIsBoundedByWorkgroupsNotWorkItems)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    auto kernel = makeInvocation(/*workGroupSize=*/{256, 1, 1},
                                 /*numWorkGroups=*/{(size_t(1) << 24) + 16, 1, 1});
    kernel.clusterDim = {2, 1, 1};

    ASSERT_GT(kernel.numWorkItems.x, kUintMax) << "work items overflow";
    ASSERT_LE(kernel.numWorkGroups.x, kUintMax) << "but workgroups do not";

#ifdef HIP_HAS_CLUSTER_LAUNCH
    EXPECT_NE(adapter.launchKernel(kernel), hipErrorInvalidValue)
        << "the cluster path passes numWorkGroups, which fits here, so the work-item "
           "product must not gate it";
#else
    // Without cluster-launch support the invocation falls through to the
    // work-item path, where the same grid does overflow.
    EXPECT_EQ(adapter.launchKernel(kernel), hipErrorInvalidValue);
#endif
}

TEST(LaunchDimGuard, ClusterLaunchWithOversizedWorkgroupCount_Refused)
{
    using namespace TensileLite;

    hip::SolutionAdapter adapter;
    auto kernel       = makeInvocation(/*workGroupSize=*/{1, 1, 1},
                                 /*numWorkGroups=*/{(size_t(1) << 32) + 16, 1, 1});
    kernel.clusterDim = {2, 1, 1};

    ASSERT_GT(kernel.numWorkGroups.x, kUintMax);

    EXPECT_EQ(adapter.launchKernel(kernel), hipErrorInvalidValue);
}

#ifdef HIP_HAS_CLUSTER_LAUNCH
// AIHPBLAS-5190: validate each axis before kernel lookup; a divisible total
// workgroup count alone does not guarantee complete clusters.
TEST(LaunchDimGuard, ClusterDimensionsValidatedBeforeKernelLookup)
{
    using namespace TensileLite;
    const struct
    {
        TensileLite::dim3 grid, cluster;
        bool invalid;
    } cases[] = {
        {{86, 1, 1}, {4, 1, 1}, true},
        {{84, 1, 1}, {4, 1, 1}, false},
        {{3, 4, 1}, {2, 2, 1}, true},
        {{4, 3, 1}, {2, 2, 1}, true},
        {{4, 4, 1}, {2, 2, 1}, false},
        {{4, 4, 1}, {0, 2, 1}, true},
        {{4, 4, 1}, {2, 0, 1}, true},
    };
    hip::SolutionAdapter adapter;
    for(auto const& test : cases)
    {
        SCOPED_TRACE(::testing::Message() << "grid " << test.grid << " cluster " << test.cluster);
        auto kernel = makeInvocation({128, 1, 1}, test.grid);
        kernel.clusterDim = test.cluster;
        EXPECT_EQ(adapter.launchKernel(kernel) == hipErrorInvalidValue, test.invalid);
    }
}
#endif

#ifdef HIP_HAS_CLUSTER_LAUNCH
// A cluster launch delivers the workgroup id in y and z to the kernel as 16-bit
// values. A grid beyond 65535 in either dimension used to launch successfully
// and wrap on the device, leaving part of D unwritten.
TEST(LaunchDimGuard, ClusterLaunchGridYZBeyond16Bits_Refused)
{
    using namespace TensileLite;
    const struct
    {
        TensileLite::dim3 grid;
        bool              invalid;
    } cases[] = {
        {{2, 2, 65535}, false},
        {{2, 2, 65536}, true},
        {{2, 2, 98666}, true},
        {{2, 65534, 1}, false},
        {{2, 65536, 1}, true},
    };
    hip::SolutionAdapter adapter;
    for(auto const& test : cases)
    {
        SCOPED_TRACE(::testing::Message() << "grid " << test.grid);
        auto kernel       = makeInvocation({128, 1, 1}, test.grid);
        kernel.clusterDim = {2, 2, 1};
        EXPECT_EQ(adapter.launchKernel(kernel) == hipErrorInvalidValue, test.invalid);
    }
}
#endif

namespace
{
    // Non-persistent 192x192 tile solution, as generated for
    // MatrixInstruction [16, 16, 32, 1, 1, 6, 6, 2, 2].
    void initClusterSolution(TensileLite::ContractionSolution& solution,
                             TensileLite::dim3 const&          clusterDim)
    {
        solution.kernelName                  = "LaunchDimGuard_test_cluster_solution";
        solution.sizeMapping.workGroupSize   = {128, 1, 1};
        solution.sizeMapping.macroTile       = {192, 192, 1};
        solution.sizeMapping.depthU          = 64;
        solution.sizeMapping.globalSplitU    = 1;
        solution.sizeMapping.clusterDim      = clusterDim;
    }

    bool launchLimitsAccept(TensileLite::ContractionSolution const& solution,
                            size_t                                  m,
                            size_t                                  n,
                            size_t                                  batch)
    {
        using namespace TensileLite;
        AMDGPU hardware(AMDGPU::Processor::gfx1250, 256, "gfx1250");
        auto   problem = ContractionProblemGemm::GEMM(true, false, m, n, 96, 96, 96, m, 0.0, false, batch);
        Task   task(hardware, problem, solution);
        Predicates::Contraction::LaunchLimits predicate;
        std::ostringstream                     diagnostic;
        bool                                   rv = predicate(task);
        EXPECT_EQ(predicate.debugEval(task, diagnostic), rv) << diagnostic.str();
        return rv;
    }
}

// Selection must not pick a cluster solution whose launch would exceed the
// cluster grid y/z limit, so another solution serves the problem instead.
TEST(LaunchDimGuard, LaunchLimitsRejectsClusterGridBeyond16Bits)
{
    TensileLite::ContractionSolution cluster, linear;
    initClusterSolution(cluster, {2, 2, 1});
    initClusterSolution(linear, {1, 1, 1});

    // z carries the batch count.
    EXPECT_TRUE(launchLimitsAccept(cluster, 224, 128, 65535));
    EXPECT_FALSE(launchLimitsAccept(cluster, 224, 128, 65536));
    EXPECT_FALSE(launchLimitsAccept(cluster, 224, 128, 98666));
    EXPECT_TRUE(launchLimitsAccept(linear, 224, 128, 98666));

    // y carries the N tiles, rounded up to whole clusters.
    EXPECT_TRUE(launchLimitsAccept(cluster, 192, 192 * 65534, 1));
    EXPECT_FALSE(launchLimitsAccept(cluster, 192, 192 * 65535, 1));
    EXPECT_TRUE(launchLimitsAccept(linear, 192, 192 * 65535, 1));
}
