// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Warm-up probe: after a GPU memory fault the first process doing real GPU work can compute wrong
// results; this one does that work (allocate, run a kernel, read back and verify) so the next
// client starts on a healthy device. Exit status: 0 verified, 1 failure, 2 usage.
//
// Written without exceptions: every step returns false and fills in the failure reason.

#include "GpuProbeArgs.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>

static constexpr int64_t bufferBytes   = 128LL * 1024 * 1024;
static constexpr int64_t elementCount  = bufferBytes / static_cast<int64_t>(sizeof(uint32_t));
static constexpr int32_t blockSize     = 256;
static constexpr int32_t usageExitCode = 2;

struct HipFree
{
    void operator()(void* pointer) const
    {
        // The buffers are released on every exit path, where a failure cannot be acted upon.
        (void)hipFree(pointer);
    }
};

using DeviceBuffer = std::unique_ptr<uint32_t, HipFree>;
using HostBuffer   = std::unique_ptr<uint32_t[]>;

static __global__ void copyKernel(const uint32_t* src, uint32_t* dst, int64_t count)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count)
        dst[index] = src[index];
}

static uint32_t pattern(int64_t index)
{
    return static_cast<uint32_t>(index) * 2654435761u + 12345u;
}

static bool checkHip(hipError_t error, char const* what, std::string& failure)
{
    if(error == hipSuccess)
        return true;
    std::ostringstream message;
    message << "hip error " << static_cast<int>(error) << " (" << hipGetErrorName(error) << ") at "
            << what << ": " << hipGetErrorString(error);
    failure = message.str();
    return false;
}

static bool allocateHostBuffer(HostBuffer& buffer, std::string& failure)
{
    buffer.reset(new(std::nothrow) uint32_t[elementCount]);
    if(buffer)
        return true;
    failure = "host allocation failed";
    return false;
}

static bool allocateDeviceBuffer(DeviceBuffer& buffer, std::string& failure)
{
    void* pointer = nullptr;
    if(!checkHip(hipMalloc(&pointer, bufferBytes), "hipMalloc", failure))
        return false;
    buffer.reset(static_cast<uint32_t*>(pointer));
    return true;
}

static bool copyOnDevice(const HostBuffer& input, HostBuffer& output, std::string& failure)
{
    DeviceBuffer src;
    DeviceBuffer dst;
    if(!allocateDeviceBuffer(src, failure) || !allocateDeviceBuffer(dst, failure))
        return false;
    if(!checkHip(hipMemcpy(src.get(), input.get(), bufferBytes, hipMemcpyHostToDevice),
                 "hipMemcpy (host to device)",
                 failure))
        return false;
    if(!checkHip(hipMemset(dst.get(), 0xFF, bufferBytes), "hipMemset", failure))
        return false;

    int64_t blocks = (elementCount + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(copyKernel,
                       dim3(static_cast<uint32_t>(blocks)),
                       dim3(blockSize),
                       0,
                       0,
                       src.get(),
                       dst.get(),
                       elementCount);
    return checkHip(hipGetLastError(), "kernel launch", failure)
           && checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize", failure)
           && checkHip(hipMemcpy(output.get(), dst.get(), bufferBytes, hipMemcpyDeviceToHost),
                       "hipMemcpy (device to host)",
                       failure);
}

static bool verifyCopy(const HostBuffer& output, std::string& failure)
{
    int64_t mismatches = 0;
    int64_t firstIndex = -1;
    for(int64_t i = 0; i < elementCount; ++i)
    {
        if(output[i] == pattern(i))
            continue;
        if(mismatches == 0)
            firstIndex = i;
        ++mismatches;
    }
    if(mismatches == 0)
        return true;
    std::ostringstream message;
    message << mismatches << " of " << elementCount << " elements differ (first at index "
            << firstIndex << ")";
    failure = message.str();
    return false;
}

static bool runProbe(int32_t deviceIdx, std::string& failure)
{
    if(!checkHip(hipSetDevice(deviceIdx), "hipSetDevice", failure))
        return false;
    HostBuffer input;
    HostBuffer output;
    if(!allocateHostBuffer(input, failure) || !allocateHostBuffer(output, failure))
        return false;
    for(int64_t i = 0; i < elementCount; ++i)
        input[i] = pattern(i);
    return copyOnDevice(input, output, failure) && verifyCopy(output, failure);
}

int main(int argc, char** argv)
{
    std::optional<int32_t> deviceIdx = parseGpuProbeArgs(argc, argv);
    if(!deviceIdx)
    {
        std::cerr << "usage: " << argv[0] << " [--device-idx N]" << std::endl;
        return usageExitCode;
    }
    std::string failure;
    if(!runProbe(*deviceIdx, failure))
    {
        std::cerr << "gpu probe failed on device " << *deviceIdx << ": " << failure << std::endl;
        return 1;
    }
    std::cout << "gpu probe passed on device " << *deviceIdx << " (128 MiB copy verified)"
              << std::endl;
    return 0;
}
