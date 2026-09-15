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
#include <hip/hiprtc.h>

#define CUDART_VERSION 0

// types
#ifndef nvrtcProgram
#define nvrtcProgram hiprtcProgram
#endif
#ifndef nvrtcResult
#define nvrtcResult hiprtcResult
#endif
#ifndef cudaError_t
#  define cudaError_t hipError_t
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif

// macros
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif
#ifndef cudaSuccess
#  define cudaSuccess hipSuccess
#endif
#ifndef CUDA_SUCCESS
#  define CUDA_SUCCESS hipSuccess
#endif
#ifndef CUresult
#  define CUresult hipError_t
#endif
#ifndef CUdevice
#  define CUdevice hipDevice_t
#endif

#ifndef CUcontext
#  define CUcontext hipCtx_t
#endif
#ifndef CUmodule
#  define CUmodule hipModule_t
#endif
#ifndef CUfunction
#  define CUfunction hipFunction_t
#endif


// functions
#ifndef nvrtcCreateProgram
#  define nvrtcCreateProgram hiprtcCreateProgram
#endif
#ifndef nvrtcDestroyProgram
#  define nvrtcDestroyProgram hiprtcDestroyProgram
#endif
#ifndef nvrtcGetPTX
#  define nvrtcGetPTX hiprtcGetCode
#endif
#ifndef nvrtcGetPTXSize
#  define nvrtcGetPTXSize hiprtcGetCodeSize
#endif
#ifndef nvrtcGetCUBIN
#  define nvrtcGetCUBIN hiprtcGetBitcode
#endif
#ifndef nvrtcGetCUBINSize
#  define nvrtcGetCUBINSize hiprtcGetBitcodeSize
#endif
#ifndef nvrtcGetProgramLog
#  define nvrtcGetProgramLog hiprtcGetProgramLog
#endif
#ifndef nvrtcGetProgramLogSize
#  define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#endif
#ifndef nvrtcCompileProgram
#  define nvrtcCompileProgram hiprtcCompileProgram
#endif
#ifndef nvrtcGetErrorString
#  define nvrtcGetErrorString hiprtcGetErrorString
#endif
#ifndef cuLaunchKernel
#  define cuLaunchKernel hipModuleLaunchKernel
#endif
#ifndef cuInit
#  define cuInit hipInit
#endif
#ifndef cuCtxCreate
#  define cuCtxCreate hipCtxCreate
#endif
#ifndef cuModuleGetFunction  
#  define cuModuleGetFunction hipModuleGetFunction
#endif
#ifndef cuModuleLoadDataEx
#  define cuModuleLoadDataEx hipModuleLoadDataEx
#endif

#ifndef cudaGetErrorString
#  define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetErrorName
#  define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetLastError
#  define cudaGetLastError hipGetLastError
#endif
#ifndef cudaDeviceSynchronize
#  define cudaDeviceSynchronize hipDeviceSynchronize
#endif
#ifndef cuGetErrorName
#  define cuGetErrorName hipDrvGetErrorName
#endif
#ifndef cuDeviceGet
#  define cuDeviceGet hipDeviceGet
#endif
#ifndef cudaGetDevice
#  define cudaGetDevice hipGetDevice
#endif
#ifndef cudaDeviceProp
#  define cudaDeviceProp hipDeviceProp_t
#endif
#ifndef cudaGetDeviceProperties
#  define cudaGetDeviceProperties hipGetDeviceProperties
#endif
