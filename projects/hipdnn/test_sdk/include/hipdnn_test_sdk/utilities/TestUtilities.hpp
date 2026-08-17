// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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
