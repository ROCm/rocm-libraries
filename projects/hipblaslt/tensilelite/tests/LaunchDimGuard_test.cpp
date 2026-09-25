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

#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

#include <cstdint>
#include <limits>

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
