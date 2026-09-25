// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Shared body for the rocRAND C-compilation checks. Each ordering-specific
// translation unit (c_compile_*.c) sets ROCRAND_C_COMPILE_FN and includes the
// public headers in a particular order *before* including this file. The point
// of the test is that the header set compiles cleanly as C and that the C-mode
// fallback typedefs (uint4 / __half / hipStream_t) do not collide with HIP's
// real definitions regardless of include order -- see ROCM-28020.

#ifndef ROCRAND_C_COMPILE_FN
    #error "define ROCRAND_C_COMPILE_FN before including c_compile_body.h"
#endif

int ROCRAND_C_COMPILE_FN(void);

int ROCRAND_C_COMPILE_FN(void)
{
    // Naming each of these types is what regressed in ROCM-28020: rocRAND's
    // C-only fallback typedefs must resolve to a single, consistent definition
    // whether or not <hip/hip_runtime.h> is also in the translation unit.
    uint4       vec;
    __half      half_value;
    hipStream_t stream;
    int         version = 0;

    (void)vec;
    (void)half_value;
    (void)stream;

    // Host-only entry point: exercises linkage against the real library without
    // requiring a GPU at run time.
    if(rocrand_get_version(&version) != ROCRAND_STATUS_SUCCESS)
    {
        return -1;
    }
    return version;
}
