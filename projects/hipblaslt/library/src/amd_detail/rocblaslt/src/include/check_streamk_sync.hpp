// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the StreamK Synchronizer buffer.
 *        Enabled by HIPBLASLT_CHECK_STREAMK_SYNC env var (read once in handle ctor).
 */

#pragma once
#ifndef HIPBLASLT_CHECK_STREAMK_SYNC_HPP
#define HIPBLASLT_CHECK_STREAMK_SYNC_HPP

#include "handle.h"
#include "rocblaslt-types.h"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

// Zeroes the buffer for a known-clean baseline before the launch. Pair with
// hipblaslt_check_streamk_sync_scan to report residue left by that launch.
inline void hipblaslt_check_streamk_sync_reset(rocblaslt_handle handle, hipStream_t stream)
{
    if(!handle || !handle->check_streamk_sync || !handle->Synchronizer)
        return;
    static_cast<void>(hipMemsetAsync(
        handle->Synchronizer, 0, hipblaslt_streamk_synchronizer_ints * sizeof(int), stream));
}

// Blocks on `stream` to read the buffer back. Gated on the env var, so the
// default path costs nothing.
inline void hipblaslt_check_streamk_sync_scan(rocblaslt_handle handle,
                                              hipStream_t      stream,
                                              const char*      label)
{
    if(!handle || !handle->check_streamk_sync || !handle->Synchronizer)
        return;

    std::vector<int> host(hipblaslt_streamk_synchronizer_ints);
    hipError_t       err = hipStreamSynchronize(stream);
    if(err == hipSuccess)
        err = hipMemcpy(host.data(),
                        handle->Synchronizer,
                        hipblaslt_streamk_synchronizer_ints * sizeof(int),
                        hipMemcpyDeviceToHost);
    // `host` is zero-initialized, so a silent readback failure would read as a
    // clean buffer. Report it instead.
    if(err != hipSuccess)
    {
        fprintf(stderr,
                "[hipBLASLt CHECK_STREAMK_SYNC] %s: readback failed (%s); buffer not checked.\n",
                label,
                hipGetErrorString(err));
        return;
    }

    size_t nonzero = 0, first = hipblaslt_streamk_synchronizer_ints;
    for(size_t i = 0; i < hipblaslt_streamk_synchronizer_ints; ++i)
        if(host[i] != 0)
        {
            if(nonzero == 0)
                first = i;
            ++nonzero;
        }

    if(nonzero)
        fprintf(stderr,
                "[hipBLASLt CHECK_STREAMK_SYNC] %s: Synchronizer left dirty "
                "(%zu/%zu ints nonzero, first at offset %zu) -- the kernel did "
                "not self-clean its work-queue state.\n",
                label,
                nonzero,
                hipblaslt_streamk_synchronizer_ints,
                first);
}

// Resets the Synchronizer buffer on construction and scans it on
// destruction, reporting any residue left by whatever ran in between.
class hipblaslt_check_streamk_sync_scope
{
public:
    hipblaslt_check_streamk_sync_scope(rocblaslt_handle handle,
                                       hipStream_t      stream,
                                       const char*      label)
        : handle_(handle)
        , stream_(stream)
        , label_(label)
    {
        hipblaslt_check_streamk_sync_reset(handle_, stream_);
    }

    ~hipblaslt_check_streamk_sync_scope()
    {
        hipblaslt_check_streamk_sync_scan(handle_, stream_, label_);
    }

private:
    rocblaslt_handle handle_;
    hipStream_t      stream_;
    const char*      label_;
};

#endif // HIPBLASLT_CHECK_STREAMK_SYNC_HPP
