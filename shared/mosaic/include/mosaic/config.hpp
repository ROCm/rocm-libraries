// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Platform detection and portability macros
// ---------------------------------------------------------------------------

// TODO: Should these be __forceinline__?
//
// hip::host defines __HIP_PLATFORM_AMD__ for every dependent, including
// host C++ translation units. __host__ and __device__ are keywords only
// under a HIP compiler, so gate on that instead of the platform macro.

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define MOSAIC_HOST_DEVICE __host__ __device__
#else
#define MOSAIC_HOST_DEVICE
#endif
