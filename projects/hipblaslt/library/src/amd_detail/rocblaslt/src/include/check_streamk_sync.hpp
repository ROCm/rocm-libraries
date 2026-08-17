// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the StreamK Synchronizer buffer.
 *        Enabled by HIPBLASLT_CHECK_STREAMK_SYNC env var (read once in handle ctor).
 *
 *        Covers rocblaslt_matmul_impl only; the ext and user-argument launch
 *        paths share the same buffer but are not scanned, so residue they leave
 *        is reported against the next scanned matmul.
 *
 *        Single-threaded debugging only: the scan synchronizes the stream and
 *        zeroes a handle-wide buffer, which would disrupt a concurrent kernel
 *        on another stream. Skipped entirely during HIP graph capture, since
 *        both operations are illegal there.
 */

#pragma once
#ifndef HIPBLASLT_CHECK_STREAMK_SYNC_HPP
#define HIPBLASLT_CHECK_STREAMK_SYNC_HPP

#include "handle.h"
#include "rocblaslt-types.h"
#include "utility.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <vector>

// Blocks on `stream` to read the buffer back, and reports it if any int is
// nonzero. Gated on the env var, so the default path costs nothing.
inline void hipblaslt_check_streamk_sync_scan(rocblaslt_handle handle,
                                              hipStream_t      stream,
                                              const char*      label)
{
    if(!handle || !handle->check_streamk_sync || !handle->Synchronizer)
        return;

    // Skip during HIP graph capture: the sync and memset below cannot be
    // sequenced into a captured graph.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess && cap != hipStreamCaptureStatusNone)
        return;

    constexpr size_t count = hipblaslt_streamk_synchronizer_ints;
    constexpr size_t bytes = count * sizeof(int);

    if(handle->check_streamk_sync_host.size() != count)
        handle->check_streamk_sync_host.assign(count, 0);
    std::vector<int>& host = handle->check_streamk_sync_host;

    hipError_t err = hipStreamSynchronize(stream);
    if(err == hipSuccess)
        err = hipMemcpy(host.data(), handle->Synchronizer, bytes, hipMemcpyDeviceToHost);
    // `host` may hold a previous scan's contents on failure, so an unreported
    // failure here could read as a stale (possibly clean) buffer.
    if(err != hipSuccess)
    {
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << "[hipBLASLt CHECK_STREAMK_SYNC] " << label << ": readback failed ("
              << hipGetErrorString(err) << "); buffer not checked." << std::endl;
        return;
    }

    // Ints are both the scan step and the reported unit: the head of the buffer
    // is one work-queue counter per XCD, so the offset names the counter left set.
    size_t nonzero = 0, first = count;
    for(size_t i = 0; i < count; ++i)
        if(host[i] != 0)
        {
            if(nonzero == 0)
                first = i;
            ++nonzero;
        }

    if(nonzero == 0)
        return;

    {
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << "[hipBLASLt CHECK_STREAMK_SYNC] " << label << ": Synchronizer left dirty ("
              << nonzero << "/" << count << " ints nonzero, first at offset " << first
              << ") -- the kernel did not self-clean its work-queue state." << std::endl;
    }

    // hipblasLtCreate zeroes the buffer once and every matmul is scanned, so
    // restoring zero here keeps the next call's baseline clean and stops this
    // residue being re-reported by every call after it.
    static_cast<void>(hipMemset(handle->Synchronizer, 0, bytes));
}

#endif // HIPBLASLT_CHECK_STREAMK_SYNC_HPP
