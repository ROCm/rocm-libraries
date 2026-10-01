// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(HOST_DEVICE)
#if defined(__HIPCC__) || defined(__HIP__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

namespace hipdnn_gpu_ref::types::detail
{

HOST_DEVICE inline unsigned int floatToBits(float f) noexcept
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(unsigned int, f);
#else
    unsigned int b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
#endif
#else
    unsigned int b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
#endif
}

HOST_DEVICE inline float bitsToFloat(unsigned int b) noexcept
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(float, b);
#else
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
#endif
#else
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
#endif
}

} // namespace hipdnn_gpu_ref::types::detail
