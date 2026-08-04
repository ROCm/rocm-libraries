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

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

// Blocks on `stream` to read the buffer back, and reports it if any int is
// nonzero. Gated on the env var, so the default path costs nothing.
inline void hipblaslt_check_streamk_sync_scan(rocblaslt_handle handle,
                                              hipStream_t      stream,
                                              const char*      label)
{
    if(!handle || !handle->check_streamk_sync || !handle->Synchronizer)
        return;

    constexpr size_t count = hipblaslt_streamk_synchronizer_ints;
    constexpr size_t bytes = count * sizeof(int);

    std::vector<int> host(count);
    hipError_t       err = hipStreamSynchronize(stream);
    if(err == hipSuccess)
        err = hipMemcpy(host.data(), handle->Synchronizer, bytes, hipMemcpyDeviceToHost);
    // `host` is zero-initialized, so an unreported failure here would read as a
    // clean buffer.
    if(err != hipSuccess)
    {
        fprintf(stderr,
                "[hipBLASLt CHECK_STREAMK_SYNC] %s: readback failed (%s); buffer not checked.\n",
                label,
                hipGetErrorString(err));
        return;
    }

    // Word-at-a-time on the clean path; the per-int tally below only runs once
    // a nonzero word is found.
    const uint64_t* w     = reinterpret_cast<const uint64_t*>(host.data());
    bool            dirty = false;
    for(size_t i = 0; i < count / 2 && !dirty; ++i)
        dirty = (w[i] != 0);
    if(!dirty && (count % 2) != 0)
        dirty = (host[count - 1] != 0);

    if(!dirty)
        return;

    size_t nonzero = 0, first = count;
    for(size_t i = 0; i < count; ++i)
        if(host[i] != 0)
        {
            if(nonzero == 0)
                first = i;
            ++nonzero;
        }

    fprintf(stderr,
            "[hipBLASLt CHECK_STREAMK_SYNC] %s: Synchronizer left dirty "
            "(%zu/%zu ints nonzero, first at offset %zu) -- the kernel did "
            "not self-clean its work-queue state.\n",
            label,
            nonzero,
            count,
            first);

    // hipblasLtCreate zeroes the buffer once and every matmul is scanned, so
    // restoring zero here keeps the next call's baseline clean and stops this
    // residue being re-reported by every call after it.
    static_cast<void>(hipMemset(handle->Synchronizer, 0, bytes));
}

#endif // HIPBLASLT_CHECK_STREAMK_SYNC_HPP
