// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Platform detection and portability macros
// ---------------------------------------------------------------------------

// TODO: Should these be __forceinline__?

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define MOSAIC_HOST_DEVICE __host__ __device__
#else
#define MOSAIC_HOST_DEVICE
#endif
