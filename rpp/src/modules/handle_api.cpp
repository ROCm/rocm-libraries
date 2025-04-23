/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cstdio>
#include "errors.hpp"
#include "handle.hpp"

extern "C" rppStatus_t rppCreate(rppHandle_t* handle, size_t nBatchSize, Rpp32u numThreads, void* stream, RppBackend backend)
{
    if(backend == RppBackend::RPP_HOST_BACKEND)
        return rpp::try_([&] { rpp::deref(handle) = new rpp::Handle(nBatchSize, numThreads); });
#if GPU_SUPPORT
    else if(backend == RppBackend::RPP_HIP_BACKEND || backend == RppBackend::RPP_OCL_BACKEND)
    {
            return rpp::try_([&] {
            rpp::deref(handle) = new rpp::Handle(nBatchSize, reinterpret_cast<rppAcceleratorQueue_t>(stream));
        });
    }
#endif // GPU_SUPPORT
    else
        return rppStatusNotImplemented;

}

extern "C" rppStatus_t rppDestroy(rppHandle_t handle, RppBackend backend)
{
    if(backend == RppBackend::RPP_HOST_BACKEND)
    {
        return rpp::try_([&] { rpp::deref(handle).rpp_destroy_object_host(); });
    }
#if GPU_SUPPORT
    else if(backend == RppBackend::RPP_HIP_BACKEND || backend == RppBackend::RPP_OCL_BACKEND)
    {
        return rpp::try_([&] { rpp::deref(handle).rpp_destroy_object_gpu(); });
    }
#endif // GPU_SUPPORT
    else
        return rppStatusNotImplemented;
}

extern "C" rppStatus_t rppSetBatchSize(rppHandle_t handle, size_t batchSize)
{
    return rpp::try_([&] { rpp::deref(handle).SetBatchSize(batchSize); });
}

extern "C" rppStatus_t rppGetBatchSize(rppHandle_t handle, size_t *batchSize)
{
    return rpp::try_([&] { rpp::deref(batchSize) = rpp::deref(handle).GetBatchSize(); });
}

#if GPU_SUPPORT

extern "C" rppStatus_t rppSetStream(rppHandle_t handle, rppAcceleratorQueue_t streamID)
{
    return rpp::try_([&] { rpp::deref(handle).SetStream(streamID); });
}

extern "C" rppStatus_t rppGetStream(rppHandle_t handle, rppAcceleratorQueue_t* streamID)
{
    return rpp::try_([&] { rpp::deref(streamID) = rpp::deref(handle).GetStream(); });
}

extern "C" rppStatus_t rppSetAllocator(rppHandle_t handle, rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext)
{
    return rpp::try_([&] { rpp::deref(handle).SetAllocator(allocator, deallocator, allocatorContext); });
}

extern "C" rppStatus_t rppGetKernelTime(rppHandle_t handle, float* time)
{
    return rpp::try_([&] { rpp::deref(time) = rpp::deref(handle).GetKernelTime(); });
}

extern "C" rppStatus_t rppEnableProfiling(rppHandle_t handle, bool enable)
{
    return rpp::try_([&] { rpp::deref(handle).EnableProfiling(enable); });
}

#endif // GPU_SUPPORT
