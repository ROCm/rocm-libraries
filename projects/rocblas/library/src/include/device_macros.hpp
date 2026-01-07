/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

// do not use pragma once

/*******************************************************************************
 * Macros
 ******************************************************************************/

#ifdef DEVICE_GRID_YZ_16BIT
#undef DEVICE_GRID_YZ_16BIT
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#define DEVICE_GRID_YZ_16BIT 1
#else
#define DEVICE_GRID_YZ_16BIT 0
#endif

#define DEVICE_GRID_SETUP                                                                    \
    uint32_t dc_YZ_grid_launch_limit = (uint32_t)0x7fffffff;                                 \
    if(__builtin_amdgcn_processor_is("gfx1200") || __builtin_amdgcn_processor_is("gfx1201")) \
        dc_YZ_grid_launch_limit = c_YZ_grid_launch_limit;

#define DEVICE_GRID_CONTINUE \
    (__builtin_amdgcn_processor_is("gfx1200") || __builtin_amdgcn_processor_is("gfx1201"))

#define WARP_32 32
#define WARP_64 64
