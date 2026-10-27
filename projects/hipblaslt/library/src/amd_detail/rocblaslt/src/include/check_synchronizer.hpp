// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the handle's inter-workgroup flag
 *        buffers. Enabled by the HIPBLASLT_CHECK_SYNCHRONIZER env var (read
 *        once in the handle ctor).
 *
 *        There are two such buffers and both must read back all-zero once a
 *        kernel that used them has retired:
 *          - `Synchronizer`: the GSU MultipleBufferSingleKernel reduction flags
 *            and the amaxD counter, one slot per problem, shared across
 *            streams. Scanned whole.
 *          - `StreamKFlags`: the Stream-K work-queue and fixup flags, one block
 *            per (stream, problem). Only the block bound to this launch is
 *            scanned. Another stream may be mid-kernel with legitimately
 *            nonzero flags in its own block, so scanning the whole table would
 *            report that as residue; the per-(stream, problem) block is the
 *            only region this launch could have dirtied.
 *        Each buffer is reported and re-zeroed on its own, and either may be
 *        absent, in which case that half of the scan is skipped.
 *
 *        Covers rocblaslt_matmul_impl only. The ext and user-argument paths use
 *        the same buffers but are not scanned, so residue they leave is
 *        reported against the next scanned matmul.
 *
 *        The scan synchronizes the stream, stages the readback in a
 *        thread-local host buffer, and zeroes whichever region it found dirty.
 *        The two buffers differ in how safe that is. The Stream-K block is
 *        private to its (stream, problem), so scanning and zeroing it is safe
 *        with other streams running. The GSU `Synchronizer` region is shared
 *        across streams and un-partitioned by design, so scanning and zeroing
 *        it assumes no other stream is concurrently in an MBSK launch -- the
 *        same assumption that region's own design already makes. Skipped during
 *        HIP graph capture, where the synchronize and the memset are illegal.
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

namespace hipblaslt_check_synchronizer_detail
{
    inline std::ostream& sink()
    {
        std::ostream* os = get_logger_os();
        return os ? *os : std::cerr;
    }

    // Reads `count` ints back from `region` and reports any nonzero one against
    // `buffer`. No-op when that buffer is absent.
    inline void scan_region(const char* label, const char* buffer, void* region, size_t count)
    {
        if(!region)
            return;

        const size_t bytes = count * sizeof(int);

        // Staging for the readback. Thread-local, and so unshared: one handle
        // serves many streams, and two threads driving their own Stream-K
        // blocks through the same handle both reach this scan. It grows to the
        // larger of the two regions and is reused by both.
        static thread_local std::vector<int> host;
        if(host.size() < count)
            host.assign(count, 0);

        // Reported, not swallowed: `host` still holds the previous scan, which
        // would otherwise read as a clean buffer.
        if(hipError_t err = hipMemcpy(host.data(), region, bytes, hipMemcpyDeviceToHost);
           err != hipSuccess)
        {
            std::lock_guard<std::mutex> lk(log_mutex);
            sink() << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": " << buffer
                   << " readback failed (" << hipGetErrorString(err) << "); buffer not checked."
                   << std::endl;
            return;
        }

        // Every consumer writes 32-bit counters, so an int offset names the
        // counter left set. Stream-K indexes its block by workgroup id. MBSK
        // works from the head of the problem's Synchronizer slot, except on the
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
            sink() << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": " << buffer
                   << " left dirty (" << nonzero << "/" << count
                   << " ints nonzero, first at int offset " << first
                   << ") -- the kernel did not reset its flags on exit." << std::endl;
        }

        // Restore the zero baseline, so this residue is reported once rather
        // than by every call after it. A failure here would re-report it
        // forever.
        if(hipError_t merr = hipMemset(region, 0, bytes); merr != hipSuccess)
        {
            std::lock_guard<std::mutex> lk(log_mutex);
            sink() << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": could not clear " << buffer
                   << " (" << hipGetErrorString(merr) << "); residue will be re-reported."
                   << std::endl;
        }
    }
}

// Blocks on `stream` to read both flag buffers back and reports any nonzero int.
inline void hipblaslt_check_synchronizer_scan(rocblaslt_handle handle,
                                              hipStream_t      stream,
                                              const char*      label)
{
    if(!handle || !handle->check_synchronizer)
        return;

    // Skip during HIP graph capture: the synchronize and the memsets below
    // cannot be sequenced into a captured graph.
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &cap) == hipSuccess && cap != hipStreamCaptureStatusNone)
        return;

    // The Stream-K block this launch was handed, at the problem index
    // rocblaslt_matmul_impl binds. Null when StreamKFlags was never allocated or
    // no block was left; the status is ignored because the caller has already
    // acted on it and this check stays passive.
    void* streamKFlags = nullptr;
    static_cast<void>(handle->streamKFlagsForStream(stream, 0, &streamKFlags));

    if(!handle->Synchronizer && !streamKFlags)
        return;

    constexpr size_t gsuCount = _rocblaslt_handle::c_syncGsuTotalElements;
    constexpr size_t skCount  = _rocblaslt_handle::c_syncSkSlotElements;

    if(hipError_t err = hipStreamSynchronize(stream); err != hipSuccess)
    {
        std::lock_guard<std::mutex> lk(log_mutex);
        hipblaslt_check_synchronizer_detail::sink()
            << "[hipBLASLt CHECK_SYNCHRONIZER] " << label << ": readback failed ("
            << hipGetErrorString(err) << "); buffers not checked." << std::endl;
        return;
    }

    hipblaslt_check_synchronizer_detail::scan_region(
        label, "Synchronizer", handle->Synchronizer, gsuCount);
    hipblaslt_check_synchronizer_detail::scan_region(label, "StreamKFlags", streamKFlags, skCount);
}

#endif // HIPBLASLT_CHECK_SYNCHRONIZER_HPP
