// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Which HIP devices a PartitionedGemm run uses.
//
// A GPU in DPX mode shows up as two HIP devices that are the two partitions of one physical GPU.
// They share the PCI domain, bus and device number and differ only in the PCI function
// (0001:01:00.0 and 0001:01:00.1 on a DPX gfx1250), and they can reach each other's memory.
// Two separate cards (SPX) have different buses and cannot be relied on to reach each other, so
// the number of devices says nothing about DPX.
//
// partitionDevices(count): the partitions of one GPU when the machine has a GPU with exactly
// `count` partitions, otherwise `count` times the same device (SPX: no speed benefit, but the
// result is the same). dpxPair() is the pair itself, for the tests that only mean something
// across two partitions.

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt_partitions
{
    // The visible HIP devices grouped by physical GPU: a group is the devices that share the PCI
    // domain, bus and device ("dddd:bb:dd") and differ in the function. Only groups with two or
    // more devices are returned; the devices of a group are in increasing order.
    inline std::vector<std::vector<int>> partitionGroups()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess)
            return {};
        std::map<std::string, std::vector<int>> byGpu;
        for(int device = 0; device < count; device++)
        {
            char id[64] = {};
            if(hipDeviceGetPCIBusId(id, sizeof(id), device) != hipSuccess)
                continue;
            const std::string bdf(id); // "0001:01:00.1"
            const auto        dot = bdf.rfind('.');
            if(dot == std::string::npos)
                continue;
            byGpu[bdf.substr(0, dot)].push_back(device);
        }
        std::vector<std::vector<int>> groups;
        for(auto& entry : byGpu)
            if(entry.second.size() >= 2)
                groups.push_back(entry.second);
        std::sort(groups.begin(), groups.end());
        return groups;
    }

    // The two partitions of a GPU in DPX mode, or nothing.
    inline std::optional<std::pair<int, int>> dpxPair(int prefer = -1)
    {
        std::optional<std::pair<int, int>> first;
        for(auto const& group : partitionGroups())
        {
            if(group.size() != 2)
                continue;
            if(prefer < 0 || group[0] == prefer || group[1] == prefer)
                return std::make_pair(group[0], group[1]);
            if(!first)
                first = std::make_pair(group[0], group[1]);
        }
        return first;
    }

    // The devices `count` partitions run on, one per partition. Partition 0 is `first` when it is
    // one of the partitions found.
    inline std::vector<int> partitionDevices(size_t count, int first = 0)
    {
        if(count <= 1)
            return std::vector<int>(count, first);
        const auto              groups = partitionGroups();
        const std::vector<int>* chosen = nullptr;
        for(auto const& group : groups)
        {
            if(group.size() != count)
                continue;
            if(std::find(group.begin(), group.end(), first) != group.end())
            {
                chosen = &group;
                break;
            }
            if(chosen == nullptr)
                chosen = &group;
        }
        if(chosen == nullptr)
            return std::vector<int>(count, first); // SPX: every partition on the same device
        std::vector<int> devices = *chosen;
        const auto       it      = std::find(devices.begin(), devices.end(), first);
        if(it != devices.end())
            std::iter_swap(devices.begin(), it); // partition 0 runs on `first`
        return devices;
    }
} // namespace hipblaslt_partitions
