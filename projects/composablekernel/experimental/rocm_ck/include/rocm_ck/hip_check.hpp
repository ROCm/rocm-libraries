// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — HIP error checking. Requires HIP runtime.
//
// HIP error checking macro for rocm_ck examples.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                  \
    do                                                   \
    {                                                    \
        hipError_t err = (call);                         \
        if(err != hipSuccess)                            \
        {                                                \
            std::fprintf(stderr,                         \
                         "HIP error %d (%s) at %s:%d\n", \
                         err,                            \
                         hipGetErrorString(err),         \
                         __FILE__,                       \
                         __LINE__);                      \
            std::exit(1);                                \
        }                                                \
    } while(0)
