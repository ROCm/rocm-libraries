// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string>

// The files the Linux host-memory query reads. Tests point them at temporary files.
struct HostMemorySources
{
    std::string meminfo     = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string mountinfo   = "/proc/self/mountinfo";
    // Where the cgroup hierarchy is assumed to be when mountinfo does not say.
    std::string cgroup_root = "/sys/fs/cgroup";
};

// Host memory this process can still allocate, in bytes: the smaller of MemAvailable and the
// headroom under each cgroup memory limit from the process's cgroup up to the root. Inside a
// container /proc/meminfo describes the whole host, so the cgroup limit is often the smaller.
// On Windows, the available physical memory. Either way, HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT caps
// the result. Returns nothing when the system cannot be read and no limit is set; 0 means that
// no memory is left, for example in a full container.
std::optional<size_t> available_host_memory_bytes(const HostMemorySources& sources = {});

// HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT, when set, caps the free host memory that each query reports
// to the test and bench clients: on a machine with no cgroup limit, or to make a test skip. It can
// only lower that figure. It is not a limit on what the process allocates in total, and it
// reserves nothing against other processes: each check compares one case or one buffer with it.
// The value is a byte count with an optional K, M, G or T suffix in powers of 1024, such as 64G.
// An unset variable gives nothing; an unreadable value gives nothing and a warning.
std::optional<size_t> host_memory_limit_override();

// The byte count `text` names in the format above, or nothing when it names none.
std::optional<size_t> parse_host_memory_size(const char* text);
