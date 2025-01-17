// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_runtime.h>

#define CUDART_VERSION 0

// types
#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif
#ifndef cudaError_t
#  define cudaError_t hipError_t
#endif
#ifndef cudaEvent_t
#  define cudaEvent_t hipEvent_t
#endif
#ifndef cudaMemPool_t
#  define cudaMemPool_t hipMemPool_t
#endif
#ifndef cudaStream_t
#  define cudaStream_t hipStream_t
#endif
#ifndef cudaMemPoolAttr
#  define cudaMemPoolAttr hipMemPoolAttr
#endif
#ifndef cudaMemPoolProps
#  define cudaMemPoolProps hipMemPoolProps
#endif
#ifndef cudaMemAllocationHandleType
#  define cudaMemAllocationHandleType hipMemAllocationHandleType
#endif
#ifndef cudaPointerAttributes
#  define cudaPointerAttributes hipPointerAttribute_t
#endif
// macros, enum constant definitions
// NOTE: C++ `constexpr` might cause redefinition errors while #define only results in a warning in this case.
//       Such redefinitions might happen when a code includes multiple "reverse hipfication" header files
//       like this one from a number of other projects. Therefore, we prefer to use #define.
#ifndef cudaStreamLegacy
#  define cudaStreamLegacy ((hipStream_t) nullptr)
#endif
#ifndef cudaStreamPerThread
#  define cudaStreamPerThread hipStreamPerThread
#endif
#ifndef cudaMemcpy
#  define cudaMemcpy hipMemcpy
#endif
#ifndef cudaMemcpyDefault
#  define cudaMemcpyDefault hipMemcpyDefault
#endif
#ifndef cudaMemcpyDeviceToHost 
#  define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost 
#endif
#ifndef cudaMemcpyHostToDevice 
#  define cudaMemcpyHostToDevice hipMemcpyHostToDevice 
#endif
#ifndef cudaMemPoolAttrReleaseThreshold
#  define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif
#ifndef cudaDevAttrMemoryPoolSupportedHandleTypes
#  define cudaDevAttrMemoryPoolSupportedHandleTypes hipDeviceAttributeMemoryPoolSupportedHandleTypes
#endif
#ifndef cudaDevAttrMemoryPoolsSupported
#  define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#endif
#ifndef cudaDevAttrL2CacheSize
#  define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#endif
#ifndef cudaDevAttrConcurrentManagedAccess
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#endif
#ifndef cudaDevAttrManagedMemory
#define cudaDevAttrManagedMemory hipDeviceAttributeManagedMemory
#endif
#ifndef cudaErrorInvalidValue
#  define cudaErrorInvalidValue hipErrorInvalidValue
#endif
#ifndef cudaErrorMemoryAllocation
#  define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif
#ifndef cudaMemAllocationTypePinned
#  define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#endif
#ifndef cudaMemPoolAttrReleaseThreshold
#  define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif
#ifndef cudaMemHandleTypeNone
#  define cudaMemHandleTypeNone hipMemHandleTypeNone
#endif
#ifndef cudaMemLocationTypeDevice
#  define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif
#ifndef cudaMemPoolReuseAllowOpportunistic
#  define cudaMemPoolReuseAllowOpportunistic hipMemPoolReuseAllowOpportunistic
#endif
#ifndef cudaEventDisableTiming
#  define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaMemoryTypeDevice
#  define cudaMemoryTypeDevice hipMemoryTypeDevice
#endif
#ifndef cudaMemoryTypeHost
#  define cudaMemoryTypeHost hipMemoryTypeHost
#endif
#ifndef cudaMemoryTypeManaged
#  define cudaMemoryTypeManaged hipMemoryTypeManaged
#endif
// functions
#ifndef cudaDeviceGetAttribute
#  define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDeviceGetDefaultMemPool
#  define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#endif
#ifndef cudaDeviceSynchronize
#  define cudaDeviceSynchronize hipDeviceSynchronize
#endif

#ifndef cudaDriverGetVersion
#  define cudaDriverGetVersion hipDriverGetVersion
#endif

#ifndef cudaEventCreateWithFlags
#  define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDestroy
#  define cudaEventDestroy hipEventDestroy
#endif
#ifndef cudaEventRecord
#  define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventSynchronize
#  define cudaEventSynchronize hipEventSynchronize
#endif

#ifndef cudaFree
#  define cudaFree hipFree
#endif
#ifndef cudaFreeAsync
#  define cudaFreeAsync hipFreeAsync
#endif
#ifndef cudaFreeHost
#  define cudaFreeHost hipHostFree
#endif

#ifndef cudaGetDevice
#  define cudaGetDevice hipGetDevice
#endif
#ifndef cudaGetDeviceCount
#  define cudaGetDeviceCount hipGetDeviceCount
#endif
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif
#ifndef cudaGetErrorName
#  define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetErrorString
#  define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetLastError
#  define cudaGetLastError hipGetLastError
#endif

#ifndef cudaMallocAsync
#  define cudaMallocAsync hipMallocAsync
#endif
#ifndef cudaMalloc
#  define cudaMalloc hipMalloc
#endif
#ifndef cudaMallocFromPoolAsync
#  define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#endif
#ifndef cudaMallocHost
#  define cudaMallocHost hipHostMalloc
#endif
#ifndef cudaMallocManaged
#  define cudaMallocManaged hipMallocManaged
#endif

#ifndef cudaMemGetInfo
#  define cudaMemGetInfo hipMemGetInfo
#endif
#ifndef cudaMemPoolCreate
#  define cudaMemPoolCreate hipMemPoolCreate
#endif
#ifndef cudaMemPoolDestroy
#  define cudaMemPoolDestroy hipMemPoolDestroy
#endif
#ifndef cudaMemPoolSetAttribute
#  define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif

#ifndef cudaMemcpyAsync
#  define cudaMemcpyAsync hipMemcpyAsync
#endif
#ifndef cudaMemsetAsync
#  define cudaMemsetAsync hipMemsetAsync
#endif

#ifndef cudaSetDevice
#  define cudaSetDevice hipSetDevice
#endif

#ifndef cudaStreamCreate
#  define cudaStreamCreate hipStreamCreate
#endif
#ifndef cudaStreamDestroy
#  define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#  define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef cudaStreamWaitEvent
#  define cudaStreamWaitEvent(a,b,c) hipStreamWaitEvent(a,b,c)
#endif
#ifndef cudaEventCreate
#  define cudaEventCreate hipEventCreate
#endif
#ifndef cudaPointerGetAttributes
#  define cudaPointerGetAttributes hipPointerGetAttributes
#endif
#ifndef cudaEventElapsedTime
#  define cudaEventElapsedTime hipEventElapsedTime
#endif

#ifndef cudaStreamQuery
#  define cudaStreamQuery hipStreamQuery
#endif
