/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Host-only smoke tests for cgroup-aware host memory probing in d_vector.hpp.
// No GPU allocations: reads sysinfo and cgroup sysfs only.

#include "d_vector.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace
{
    // hip_memory's constructor is protected; a trivial subclass reaches the probe API.
    struct HostMemoryProbe : hip_memory
    {
        HostMemoryProbe()
            : hip_memory(0, 0)
        {
        }

        using hip_memory::cgroup_available_memory;
    };
} // namespace

TEST(HostMemoryProbeSmoke, CgroupAvailableMemoryNonZero)
{
#ifndef __linux__
    GTEST_SKIP() << "Linux-only: cgroup probe reads /proc/self/cgroup and cgroupfs";
#else
    EXPECT_GT(HostMemoryProbe::cgroup_available_memory(), 0u);
#endif
}

TEST(HostMemoryProbeSmoke, GetAvailableHostMemoryNonZero)
{
#ifndef __linux__
    GTEST_SKIP() << "Linux-only: host memory probe uses sysinfo(2)";
#else
    HostMemoryProbe probe;
    EXPECT_GT(probe.get_available_host_memory(), 0u);
#endif
}

TEST(HostMemoryProbeSmoke, GetAvailableHostMemoryWithinCgroupBudget)
{
#ifndef __linux__
    GTEST_SKIP() << "Linux-only: min(freeram, cgroup headroom) is Linux-specific";
#else
    HostMemoryProbe probe;
    size_t const cgroup = probe.cgroup_available_memory();
    size_t const host   = probe.get_available_host_memory();

    EXPECT_GT(host, 0u);
    // get_available_host_memory() is min(freeram, cgroup_available_memory()).
    EXPECT_LE(host, cgroup);
#endif
}
