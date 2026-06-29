// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hip/hip_runtime.h>

#if defined(_WIN32)
#define SKIP_IF_WINDOWS()                               \
    do                                                  \
    {                                                   \
        GTEST_SKIP() << "Disable this test in Windows"; \
    } while(0)
#else
#define SKIP_IF_WINDOWS() \
    do                    \
    {                     \
    } while(0)
#endif

#ifdef ADDRESS_SANITIZER
#define SKIP_IF_ASAN()                                            \
    do                                                            \
    {                                                             \
        GTEST_SKIP() << "Disable this test when ASAN is Enabled"; \
    } while(0)
#else
#define SKIP_IF_ASAN() \
    do                 \
    {                  \
    } while(0)
#endif

// Skips the test when running under ASAN on a gfx90a device. rocBLAS/Tensile produce a
// heap-buffer-overflow under ASAN on gfx90a that aborts the process before the test can
// complete; the failure is in the GPU math libraries, not in hipDNN. Other architectures
// (e.g. gfx942) are unaffected by this overflow and continue to run the test.
// Tracking issue: https://github.com/ROCm/rocm-libraries/issues/8869
#ifdef ADDRESS_SANITIZER
#define SKIP_IF_ASAN_ON_GFX90A()                                                        \
    do                                                                                  \
    {                                                                                   \
        int _asanDev = 0;                                                               \
        hipDeviceProp_t _asanProps{};                                                   \
        if(hipGetDevice(&_asanDev) == hipSuccess                                        \
           && hipGetDeviceProperties(&_asanProps, _asanDev) == hipSuccess)              \
        {                                                                               \
            if(std::string(_asanProps.gcnArchName).find("gfx90a") != std::string::npos) \
            {                                                                           \
                GTEST_SKIP() << "Disable this test when ASAN is Enabled on gfx90a";     \
            }                                                                           \
        }                                                                               \
    } while(0)
#else
#define SKIP_IF_ASAN_ON_GFX90A() \
    do                           \
    {                            \
    } while(0)
#endif

#define SKIP_IF_NO_DEVICES()                                        \
    do                                                              \
    {                                                               \
        int device_count;                                           \
        auto result = hipGetDeviceCount(&device_count);             \
        if(result == hipErrorNoDevice || device_count == 0)         \
        {                                                           \
            GTEST_SKIP() << "No devices available. Skipping test."; \
        }                                                           \
    } while(0)

#ifdef THREAD_SANITIZER
#define SKIP_IF_TSAN()                                            \
    do                                                            \
    {                                                             \
        GTEST_SKIP() << "Disable this test when TSAN is Enabled"; \
    } while(0)
#else
#define SKIP_IF_TSAN() \
    do                 \
    {                  \
    } while(0)
#endif
