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

// Shared body for the hipRAND C-compilation checks. Each ordering-specific
// translation unit (c_compile_*.c) sets HIPRAND_C_COMPILE_FN and includes the
// public headers in a particular order *before* including this file. The point
// of the test is that the header set compiles cleanly as C and that hipRAND's
// derived types (notably `half`, which is `typedef __half half;`) resolve
// whether or not <hip/hip_runtime.h> is pulled in first.

#ifndef HIPRAND_C_COMPILE_FN
    #error "define HIPRAND_C_COMPILE_FN before including c_compile_body.h"
#endif

int HIPRAND_C_COMPILE_FN(void);

int HIPRAND_C_COMPILE_FN(void)
{
    // Naming these types verifies they are available and unambiguous in a C
    // translation unit regardless of header include order.
    hiprandGenerator_t generator;
    half               half_value;
    int                version = 0;

    (void)generator;
    (void)half_value;

    // Host-only entry point: exercises linkage against the real library without
    // requiring a GPU at run time.
    if(hiprandGetVersion(&version) != HIPRAND_STATUS_SUCCESS)
    {
        return -1;
    }
    return version;
}
