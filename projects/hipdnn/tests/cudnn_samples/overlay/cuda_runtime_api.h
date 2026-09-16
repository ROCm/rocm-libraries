#pragma once
// CUDA runtime -> HIP. Host-only corpus, so hip_runtime_api.h (not hip_runtime.h).
#include <hip/hip_runtime_api.h>

// 17 TUs gate on CUDART_VERSION, and an undefined macro evaluates to 0 -- which would
// vacuously skip all of them. Derive it from the shim's own claim rather than repeating
// the number here; cudnn_runtime_version.h asks consumers to do exactly this, so the two
// cannot drift apart. The manifest asserts the resulting value.
#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDNN_CUDART_VERSION
#endif

// Enumerated by hand: cudaDeviceProp -> hipDeviceProp_t is a suffix change a
// prefix rule would miss.
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t; // must be hipStream_t: cudnnSetStream takes it
using cudaDeviceProp = hipDeviceProp_t;
using cudaMemcpyKind = hipMemcpyKind;
static const cudaError_t cudaSuccess = hipSuccess;
static const cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
static const cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static const cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
// Deliberately NO cudaGraph* here: the shim declares `using cudaGraph_t = void*`
// globally and it appears in four Graph signatures.
