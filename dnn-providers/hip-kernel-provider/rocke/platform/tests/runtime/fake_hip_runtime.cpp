// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * fake_hip_runtime.cpp -- minimal dynamically loaded HIP query surface for
 * host-only loader integration tests.
 */
#include <cstring>

#if defined(_WIN32)
#define ROCKE_FAKE_HIP_EXPORT extern "C" __declspec(dllexport)
#else
#define ROCKE_FAKE_HIP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifndef ROCKE_FAKE_HIP_ARCH
#define ROCKE_FAKE_HIP_ARCH "gfx942"
#endif

ROCKE_FAKE_HIP_EXPORT int hipGetDevice(int* device)
{
    if(device == nullptr)
    {
        return 1;
    }
    *device = 0;
    return 0;
}

ROCKE_FAKE_HIP_EXPORT int hipGetDevicePropertiesR0600(void* raw, int device)
{
    if(raw == nullptr || device != 0)
    {
        return 1;
    }
    constexpr std::size_t arch_offset = 512;
    const char* arch = ROCKE_FAKE_HIP_ARCH;
    std::memcpy(static_cast<unsigned char*>(raw) + arch_offset, arch, std::strlen(arch) + 1);
    return 0;
}

ROCKE_FAKE_HIP_EXPORT const char* hipGetErrorString(int status)
{
    return status == 0 ? "success" : "fake HIP failure";
}
