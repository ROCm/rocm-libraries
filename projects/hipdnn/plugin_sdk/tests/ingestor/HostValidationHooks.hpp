// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>

/**
 * @file HostValidationHooks.hpp
 * @brief Host stand-ins for BenchmarkPlan's output-validation hooks: memset and memcpy over
 *        buffers that live in host memory, so validation runs with no device, plus the small
 *        helpers the validation tests share.
 */
namespace hipdnn_plugin_sdk::ingestor::host_hooks
{

/// The raw bytes of @p values, the image a float output buffer holds.
inline std::vector<uint8_t> floatBytes(std::initializer_list<float> values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t offset = 0;
    for(const float value : values)
    {
        std::memcpy(bytes.data() + offset, &value, sizeof(float));
        offset += sizeof(float);
    }
    return bytes;
}

/// The kernel ids of @p record's entries, in record order.
inline std::vector<DescriptorId> entryIds(const WinnerRecord& record)
{
    std::vector<DescriptorId> ids;
    ids.reserve(record.entries.size());
    for(const auto& entry : record.entries)
    {
        ids.push_back(entry.kernelId);
    }
    return ids;
}

/// How many times the host hooks ran, and optional scripted snapshot faults.
struct HookCounters
{
    int fills = 0;
    int snapshots = 0;
    /// 1-based snapshot call that reports a failed copy; 0 never fails.
    int failSnapshotAtCall = 0;
    /// 1-based snapshot call that reports success with one image fewer than it was asked
    /// for, breaking the hook contract; 0 never does.
    int shortSnapshotAtCall = 0;
};

/// Fills each output with its dtype's fill byte and copies each one back, in host memory.
inline BenchmarkValidationHooks hostValidationHooks(HookCounters& counters)
{
    BenchmarkValidationHooks hooks;
    hooks.fill = [&counters](hipStream_t,
                             const hipdnnPluginDeviceBuffer_t* buffers,
                             uint32_t count,
                             const std::vector<OutputBufferSpec>& specs) {
        ++counters.fills;
        for(const auto& spec : specs)
        {
            void* ptr = detail::findDeviceBuffer(buffers, count, spec.uid);
            if(ptr != nullptr)
            {
                std::memset(ptr, output_agreement::fillByte(spec.dataType), spec.byteSize);
            }
        }
    };
    hooks.snapshot = [&counters](hipStream_t,
                                 const hipdnnPluginDeviceBuffer_t* buffers,
                                 uint32_t count,
                                 const std::vector<OutputBufferSpec>& specs,
                                 OutputImages& images) {
        ++counters.snapshots;
        if(counters.snapshots == counters.failSnapshotAtCall)
        {
            return false;
        }
        images.assign(specs.size(), {});
        for(size_t i = 0; i < specs.size(); ++i)
        {
            const void* ptr = detail::findDeviceBuffer(buffers, count, specs[i].uid);
            if(ptr == nullptr)
            {
                return false;
            }
            images[i].resize(specs[i].byteSize);
            std::memcpy(images[i].data(), ptr, specs[i].byteSize);
        }
        if(counters.snapshots == counters.shortSnapshotAtCall && !images.empty())
        {
            images.pop_back();
        }
        return true;
    };
    return hooks;
}

} // namespace hipdnn_plugin_sdk::ingestor::host_hooks

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
