/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse-debugging.h>
#include <rocsparse/rocsparse.h>
#include <stdio.h>

#define HIP_CHECK(f)                                                             \
    {                                                                            \
        const hipError_t stat = (f);                                             \
        if(stat != hipSuccess)                                                   \
        {                                                                        \
            fprintf(stderr, "Error: hip error %d in line %d\n", stat, __LINE__); \
            return -1;                                                           \
        }                                                                        \
    }

#define ROCSPARSE_CHECK(f)                                                             \
    {                                                                                  \
        const rocsparse_status stat = (f);                                             \
        if(stat != rocsparse_status_success)                                           \
        {                                                                              \
            fprintf(stderr, "Error: rocsparse error %d in line %d\n", stat, __LINE__); \
            return -1;                                                                 \
        }                                                                              \
    }

//! [doc example]
int main()
{
    rocsparse_error* p_error = NULL;
    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

#ifdef ROCSPARSE_DEBUGGING
    ROCSPARSE_CHECK(rocsparse_hip_debug_enable());
#endif
    float *        dx, *dy;
    rocsparse_int* dx_ind;
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(rocsparse_int)));
    HIP_CHECK(hipMemset(dx, 0, sizeof(float)));
    HIP_CHECK(hipMemset(dx_ind, 0, sizeof(rocsparse_int)));

#ifdef ROCSPARSE_DEBUGGING
    ROCSPARSE_CHECK(rocsparse_hip_debug_start(handle, p_error));
#endif
    ROCSPARSE_CHECK(rocsparse_sgthr(handle, 1, dy, dx, dx_ind, rocsparse_index_base_zero));

#ifdef ROCSPARSE_DEBUGGING
    rocsparse_hip_debug_api_history api_history;
    ROCSPARSE_CHECK(rocsparse_hip_debug_info_get(
        handle, rocsparse_hip_debug_info_api, &api_history, sizeof(api_history), p_error));

    fprintf(stdout, "api_history: %d\n", api_history);
    ROCSPARSE_CHECK(rocsparse_hip_debug_disable());
#endif
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    return 0;
}
//! [doc example]
