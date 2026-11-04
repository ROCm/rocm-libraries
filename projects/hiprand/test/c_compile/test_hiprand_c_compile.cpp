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

#include <gtest/gtest.h>

// The real coverage of this test happens at build time: the sources below are
// compiled as C (see the c_compile_*.c translation units) and will fail to
// build if the public hipRAND headers are not C-clean under any of the tested
// include orders. This driver additionally links and runs them so the checks
// also appear as CTest cases. Each function returns the hipRAND version (> 0)
// on success, or -1 on failure.
extern "C" {
int hiprand_c_compile_hiprand_only(void);
int hiprand_c_compile_hiprand_then_hip(void);
int hiprand_c_compile_hip_then_hiprand(void);
}

TEST(hiprand_c_compile_tests, hiprand_header_only)
{
    EXPECT_GT(hiprand_c_compile_hiprand_only(), 0);
}

TEST(hiprand_c_compile_tests, hiprand_then_hip_runtime)
{
    EXPECT_GT(hiprand_c_compile_hiprand_then_hip(), 0);
}

TEST(hiprand_c_compile_tests, hip_runtime_then_hiprand)
{
    EXPECT_GT(hiprand_c_compile_hip_then_hiprand(), 0);
}
