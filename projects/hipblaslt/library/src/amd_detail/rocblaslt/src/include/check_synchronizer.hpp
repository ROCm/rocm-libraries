// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the shared Synchronizer buffer.
 *        StreamK (work-queue / fixup Flags) and GSU MultipleBufferSingleKernel
 *        both use it and both must leave it at zero on exit.
 *        Enabled by HIPBLASLT_CHECK_SYNCHRONIZER env var (read once in handle ctor).
 *
 *        Covers rocblaslt_matmul_impl only. The ext and user-argument paths
 *        share the buffer but are not scanned, so residue they leave is
 *        reported against the next scanned matmul.
 *
 *        Single-threaded debugging only: the scan synchronizes the stream and
 *        zeroes a handle-wide buffer, disrupting any concurrent kernel on
 *        another stream. Skipped during HIP graph capture, where both
 *        operations are illegal.
 */

#pragma once
#ifndef HIPBLASLT_CHECK_SYNCHRONIZER_HPP
#define HIPBLASLT_CHECK_SYNCHRONIZER_HPP

#include "handle.h"
#include "rocblaslt-types.h"
#include "utility.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <vector>

// Blocks on `stream` to read the buffer back and reports any nonzero int.
inline void hipblaslt_check_synchronizer_scan(rocblaslt_handle handle,
                                              hipStream_t      stream,
                                              const char*      label)
{
    if(!handle || !handle->check_synchronizer || !handle->Synchronizer)
        return;

    // Skip during HIP graph capture: the sync and memset below cannot be
    // sequenced into a captured graph.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess && cap != hipStreamCaptureStatusNone)
        return;

    constexpr size_t count = hipblaslt_synchronizer_ints;
    constexpr size_t bytes = count * sizeof(int);

    // Thread-local, so two threads sharing a handle do not race on it.
    static thread_local std::vector<int> staging;
    if(staging.size() != count)
        staging.assign(count, 0);
    std::vector<int>& host = staging;

    hipError_t err = hipStreamSynchronize(stream);
    if(err == hipSuccess)
        err = hipMemcpy(host.data(), handle->Synchronizer, bytes, hipMemcpyDeviceToHost);
    // Reported, not swallowed: `host` still holds the previous scan, which
    // would otherwise read as a clean buffer.
    if(err != hipSuccess)
    {
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": readback failed ("
              << hipGetErrorString(err) << "); buffer not checked." << std::endl;
        return;
    }

    // Every consumer writes 32-bit counters, so an int offset names the counter
    // left set. StreamK and MBSK both work from the head, except MBSK on the
    // user-argument path, which is offset by 1638400 bytes (int 409600).
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
        *sink << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": Synchronizer left dirty ("
              << nonzero << "/" << count << " ints nonzero, first at int offset " << first
              << ") -- the kernel did not reset the shared Synchronizer buffer on exit."
              << std::endl;
    }

    // Restore the zero baseline, so this residue is reported once rather than
    // by every call after it. A failure here would re-report it forever.
    if(hipError_t merr = hipMemset(handle->Synchronizer, 0, bytes); merr != hipSuccess)
    {
        std::lock_guard<std::mutex> lk(log_mutex);
        std::ostream*               sink = get_logger_os();
        if(!sink)
            sink = &std::cerr;
        *sink << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": could not clear the buffer ("
              << hipGetErrorString(merr) << "); residue will be re-reported." << std::endl;
    }
}

#endif // HIPBLASLT_CHECK_SYNCHRONIZER_HPP
