// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// The seam through which BenchmarkPlan checks a candidate's output before the candidate may
// win: which buffers a graph's nodes write, how to fill and read them on a device, and where
// the validated record goes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One tensor some node writes: the buffer to fill and read back.
struct OutputBufferSpec
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    /// ceil(span * elementBits / 8), where span = 1 + sum((dims[i] - 1) * strides[i]).
    size_t byteSize = 0;
};

/// One host image per OutputBufferSpec, in the same order.
using OutputImages = std::vector<std::vector<uint8_t>>;

/// The device operations validation needs, injectable so tests run without a device.
struct BenchmarkValidationHooks
{
    /// Stream-ordered: sets the first `byteSize` bytes of every spec'd buffer to
    /// `output_agreement::fillByte(dataType)`.
    std::function<void(hipStream_t,
                       const hipdnnPluginDeviceBuffer_t*,
                       uint32_t,
                       const std::vector<OutputBufferSpec>&)>
        fill;
    /// Synchronizes the stream, then copies every spec'd buffer device-to-host. False when a
    /// copy failed.
    std::function<bool(hipStream_t,
                       const hipdnnPluginDeviceBuffer_t*,
                       uint32_t,
                       const std::vector<OutputBufferSpec>&,
                       OutputImages&)>
        snapshot;
};

namespace detail
{

/// The device pointer bound to @p uid, or nullptr when the caller supplied none.
inline void*
    findDeviceBuffer(const hipdnnPluginDeviceBuffer_t* buffers, uint32_t count, int64_t uid)
{
    const auto* end = buffers + count;
    const auto* found = std::find_if(buffers, end, [uid](const hipdnnPluginDeviceBuffer_t& buffer) {
        return buffer.uid == uid;
    });
    return found != end ? found->ptr : nullptr;
}

} // namespace detail

/// HIP-backed hooks: hipMemsetAsync to fill, hipStreamSynchronize then hipMemcpyAsync to
/// snapshot. Makes no HIP call until a hook is invoked.
inline BenchmarkValidationHooks hipValidationHooks()
{
    BenchmarkValidationHooks hooks;
    hooks.fill = [](hipStream_t stream,
                    const hipdnnPluginDeviceBuffer_t* buffers,
                    uint32_t count,
                    const std::vector<OutputBufferSpec>& specs) {
        for(const auto& spec : specs)
        {
            void* ptr = detail::findDeviceBuffer(buffers, count, spec.uid);
            if(ptr == nullptr || spec.byteSize == 0)
            {
                continue;
            }
            // A failed fill leaves the buffer as the kernel finds it; the snapshot that
            // follows reports the copy failure if the device is unusable.
            static_cast<void>(
                hipMemsetAsync(ptr,
                               hipdnn_plugin_sdk::output_agreement::fillByte(spec.dataType),
                               spec.byteSize,
                               stream));
        }
    };
    hooks.snapshot = [](hipStream_t stream,
                        const hipdnnPluginDeviceBuffer_t* buffers,
                        uint32_t count,
                        const std::vector<OutputBufferSpec>& specs,
                        OutputImages& images) {
        if(hipStreamSynchronize(stream) != hipSuccess)
        {
            return false;
        }
        // Resolve and size every image before the first copy is enqueued, so nothing returns
        // or throws while a copy into `images` is in flight.
        std::vector<void*> sources(specs.size(), nullptr);
        images.assign(specs.size(), {});
        for(size_t i = 0; i < specs.size(); ++i)
        {
            sources[i] = detail::findDeviceBuffer(buffers, count, specs[i].uid);
            if(sources[i] == nullptr)
            {
                return false;
            }
            images[i].resize(specs[i].byteSize);
        }
        bool copied = true;
        for(size_t i = 0; i < specs.size() && copied; ++i)
        {
            copied = specs[i].byteSize == 0
                     || hipMemcpyAsync(images[i].data(),
                                       sources[i],
                                       specs[i].byteSize,
                                       hipMemcpyDeviceToHost,
                                       stream)
                            == hipSuccess;
        }
        // The copies are stream-ordered to a pageable host buffer. Drain on every path: the
        // caller reads the images on success and destroys them on failure.
        const bool drained = hipStreamSynchronize(stream) == hipSuccess;
        return copied && drained;
    };
    return hooks;
}

/// Everything BenchmarkPlan needs to validate one sweep. A plan holding one is in
/// validation mode, even when `outputs` is empty (the check is then inconclusive).
struct BenchmarkValidation
{
    /// Non-virtual tensors some node writes; may be empty.
    std::vector<OutputBufferSpec> outputs;
    /// Non-virtual tensors no node writes, for the aliasing check.
    std::vector<int64_t> inputUids;
    BenchmarkValidationHooks hooks;
    /// Receives the validated record; replaces RecordRankingFn when validation is active.
    std::function<void(WinnerRecord)> record;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
