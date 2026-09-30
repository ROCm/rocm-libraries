// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>

#include <memory>
#include <optional>
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
 *
 * setAttribute(PROFILING_RESET_EXT) reuses this context for another measurement: it is
 * the only attribute accepted once finalized, and is also valid on a fresh or partially
 * executed descriptor. It releases the gate, drains any outstanding stream work not
 * already covered by a prior successful finalize(), and clears the finalized/start/
 * stop/elapsed/stall-used/timed-out state, returning the descriptor to step 3. The
 * handle, stream, events, and gate are retained; rebinding to a different handle or
 * stream requires a new descriptor.
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
    // Events and signal memory stay on the device where the handle was bound.
    int _device = 0;
    HipEventGuard _startEvent;
    HipEventGuard _stopEvent;
    float _elapsedMs = 0.0F;
    bool _startRecorded = false;
    bool _stopRecorded = false;
    // Latched result of the most recent STALL_ARM_EXT attempt: true only when arm()
    // actually stalled the stream for this measurement, not the current armed state --
    // it stays readable after release()/finalize() (both of which always release the
    // gate). Exposed via STALL_USED_EXT.
    bool _stallUsed = false;
    // Latched at finalize() from the gate's per-attempt timedOut(), so a later
    // measurement that skips STALL_ARM_EXT entirely cannot inherit a stale timeout from
    // an earlier one that reused this same gate. Exposed via STALL_TIMED_OUT_EXT and
    // cleared by reset().
    bool _timedOut = false;
    // Created on the first STALL_ARM_EXT, so a descriptor that only times or only syncs
    // never acquires signal memory or a watchdog thread. Destroyed with the descriptor,
    // which releases the stall if the caller never did.
    std::optional<hipdnn_data_sdk::utilities::StallGate> _stallGate;

    void createEvents();
    void checkBinding() const;
    // Handles PROFILING_RESET_EXT: releases the gate, drains outstanding stream work
    // not already covered by a prior successful finalize(), and clears per-measurement
    // state so the same handle/stream/events/gate can be reused.
    void reset();
};

} // namespace hipdnn_backend
