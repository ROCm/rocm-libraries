// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef ADDRESS_SANITIZER
#define SKIP_IF_NO_DEVICES()                              \
    do                                                    \
    {                                                     \
        GTEST_SKIP() << "Disable device tests with ASAN"; \
    } while(0)

#define SKIP_IF_ASAN()                                            \
    do                                                            \
    {                                                             \
        GTEST_SKIP() << "Disable this test when ASAN is Enabled"; \
    } while(0)

#elif _WIN32
#define SKIP_IF_NO_DEVICES()                               \
    do                                                     \
    {                                                      \
        GTEST_SKIP() << "Disable device tests with WIN32"; \
    } while(0)

#define SKIP_IF_ASAN() \
    do                 \
    {                  \
    } while(0)

#else
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

#define SKIP_IF_ASAN() \
    do                 \
    {                  \
    } while(0)

#endif
