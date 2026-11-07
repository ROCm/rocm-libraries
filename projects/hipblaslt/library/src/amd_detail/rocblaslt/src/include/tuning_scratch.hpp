// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace rocblaslt
{
    // Gradient epilogues read E. Every scratch block needs the caller's input.
    inline bool seedGradientAuxiliary(void*       destination,
                                      const void* source,
                                      size_t      bytes,
                                      size_t      stride,
                                      size_t      blocks,
                                      hipStream_t stream)
    {
        for(size_t block = 0; block < blocks; ++block)
            if(hipMemcpyAsync(static_cast<char*>(destination) + block * stride,
                              source,
                              bytes,
                              hipMemcpyDeviceToDevice,
                              stream)
               != hipSuccess)
                return false;
        return true;
    }
}
