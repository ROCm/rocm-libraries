// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// d_vector.hpp integration for the cgroup host-memory probe: proves hip_memory
// delegates to cgroup_memory_probe.hpp on Linux. Live proc/mountinfo behaviour
// is covered in cgroup_memory_probe_gtest.cpp.

#include "cgroup_memory_probe.hpp"
#include "d_vector.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sys/sysinfo.h>

namespace
{
    struct HostMemoryProbe : hip_memory
    {
        HostMemoryProbe()
            : hip_memory(0, 0)
        {
        }

        using hip_memory::cgroup_available_memory;
    };
} // namespace

TEST(HostMemoryProbeLive, CgroupDelegateMatchesLiveProbe)
{
#ifndef __linux__
    GTEST_SKIP() << "Linux-only: hip_memory delegates to cgroup_available_memory_live()";
#else
    EXPECT_EQ(HostMemoryProbe::cgroup_available_memory(),
              hipblaslt_client::cgroup_available_memory_live());
#endif
}

TEST(HostMemoryProbeLive, GetAvailableHostMemoryMinFreeramAndCgroup)
{
#ifndef __linux__
    GTEST_SKIP() << "Linux-only: get_available_host_memory uses sysinfo(2)";
#else
    struct sysinfo info{};
    ASSERT_EQ(sysinfo(&info), 0);

    HostMemoryProbe probe;
    size_t const cgroup = probe.cgroup_available_memory();
    size_t const host   = probe.get_available_host_memory();

    EXPECT_EQ(host, std::min(static_cast<size_t>(info.freeram), cgroup));
#endif
}
