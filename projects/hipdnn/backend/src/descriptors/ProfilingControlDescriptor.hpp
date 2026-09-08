// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>

#include <memory>
#include <type_traits>

struct hipdnnHandle;

namespace hipdnn_backend
{

/**
 * @brief Deleter that destroys a HIP event, enabling RAII ownership via unique_ptr.
 */
struct HipEventDeleter
{
    void operator()(hipEvent_t event) const
    {
        if(event != nullptr)
        {
            static_cast<void>(hipEventDestroy(event));
        }
    }
};

using HipEventGuard = std::unique_ptr<std::remove_pointer_t<hipEvent_t>, HipEventDeleter>;

/**
 * @brief Backend descriptor for GPU timing via HIP events
 *
 * Provides event-based GPU profiling for autotuning workloads.
 * The lifecycle is:
 *   1. setAttribute(PROFILING_HANDLE_EXT)        -- store handle, extract stream, create events
 *   2. setAttribute(PROFILING_DEVICE_SYNC_EXT)   -- optional: sync device before benchmark
 *   3. setAttribute(PROFILING_STALL_ARM_EXT)     -- optional: stall the stream
 *   4. setAttribute(PROFILING_START_EXT)         -- record start event
 *   5. (run kernel on the same stream)
 *   6. setAttribute(PROFILING_STOP_EXT)          -- record stop event
 *   7. setAttribute(PROFILING_STALL_RELEASE_EXT) -- release the stall
 *   8. finalize()                                -- synchronize stop event, compute elapsed time
 *   9. getAttribute(PROFILING_ELAPSED_MS_EXT)    -- read elapsed milliseconds
 */
class ProfilingControlDescriptor : public HipdnnBackendDescriptorImpl<ProfilingControlDescriptor>
{
public:
    ProfilingControlDescriptor() = default;
    ~ProfilingControlDescriptor() override;

    ProfilingControlDescriptor(const ProfilingControlDescriptor&) = delete;
    ProfilingControlDescriptor& operator=(const ProfilingControlDescriptor&) = delete;
    ProfilingControlDescriptor(ProfilingControlDescriptor&&) = delete;
    ProfilingControlDescriptor& operator=(ProfilingControlDescriptor&&) = delete;

    void finalize() override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnnHandle* _handle = nullptr;
    hipStream_t _stream = nullptr;
    HipEventGuard _startEvent;
    HipEventGuard _stopEvent;
    float _elapsedMs = 0.0F;
    bool _startRecorded = false;
    bool _stopRecorded = false;
    // Destroyed with the descriptor, which releases the stall if the caller never did.
    hipdnn_data_sdk::utilities::StallGate _stallGate;

    void createEvents();
};

} // namespace hipdnn_backend
