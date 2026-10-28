// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

// The files the Linux host-memory query reads. Tests point them at temporary files.
struct HostMemorySources
{
    std::string meminfo     = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

// Host memory this process can still allocate, in bytes: the smaller of MemAvailable and the
// headroom under each cgroup memory limit from the process's cgroup up to the root. Inside a
// container /proc/meminfo describes the whole host, so the cgroup limit is often the smaller.
// Returns 0 when /proc/meminfo cannot be read. On Windows, returns the available physical
// memory.
size_t available_host_memory_bytes(const HostMemorySources& sources = {});
