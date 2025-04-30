/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef TESTING_HIP_BACKEND_HPP
#define TESTING_HIP_BACKEND_HPP

#include <hip/hip_runtime.h>

#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

using namespace Tensile;

struct HipBackend
{
    // Types
    using SolutionAdapter = hip::SolutionAdapter;
    using Stream          = hipStream_t;
    using Event           = hipEvent_t;
    template <typename T>
    using BufferObj = T*;

    // Device mgmt
    static inline void setDefaultDevice(int deviceId)
    {
        HIP_CHECK_EXC(hipSetDevice(deviceId));
    }

    static inline std::shared_ptr<Hardware> getCurrentDevice()
    {
        return hip::GetCurrentDevice();
    }

    static inline void deviceReset()
    {
        HIP_CHECK_EXC(hipDeviceReset());
    }

    static inline int32_t offsetAlignment()
    {
        return sizeof(double);
    }

    // Buffer mgmt
    template <typename T>
    static inline void malloc(BufferObj<T>& bufferObj, size_t bytes)
    {
        HIP_CHECK_EXC(hipMalloc(&bufferObj, bytes));
    }

    template <typename T>
    static inline void free(BufferObj<T>& bufferObj)
    {
        HIP_CHECK_EXC(hipFree(bufferObj));
    }

    template <typename T>
    static inline T* dataPtr(BufferObj<T>& bufferObj, size_t offsetBytes = 0)
    {
        assert((0 == ((offsetAlignment() - 1) & offsetBytes))
               && "Offset alignment requirement not met");
        return bufferObj + offsetBytes / sizeof(T);
    }

    template <typename T>
    static inline void copyHostToDevice(BufferObj<T>& bufferObj,
                                        size_t        byteOffset,
                                        T const*      hostData,
                                        size_t        byteCount)
    {
        HIP_CHECK_EXC(hipMemcpy(
            bufferObj + (byteOffset / sizeof(T)), hostData, byteCount, hipMemcpyHostToDevice));
    }

    template <typename T>
    static inline void copyDeviceToHost(T*                  hostData,
                                        BufferObj<T> const& bufferObj,
                                        size_t              byteOffset,
                                        size_t              byteCount)
    {
        HIP_CHECK_EXC(hipMemcpy(
            hostData, bufferObj + (byteOffset / sizeof(T)), byteCount, hipMemcpyDeviceToHost));
    }

    // Flag for the KernelArguments debug mode
    static inline bool kernelArgsLog()
    {
        return false;
    }
};

#endif // TESTING_HIP_BACKEND_HPP
