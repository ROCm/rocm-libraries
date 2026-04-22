/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "hiptensor_options.hpp"
#include "utils.hpp"

#include <cstdio>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "llvm/command_line_parser.hpp"

int main(int argc, char** argv)
{
    // Parse hiptensor test options
    hiptensor::parseOptions(argc, argv);

    // Force HIP runtime initialization before any test fixture runs.
    // If HIP fails here we get a clean error rather than an SEH crash inside a
    // fixture, which would leave the C++ static-initialization guard locked and
    // cause all subsequent tests to deadlock ("resource deadlock would occur").
    int        deviceCount = 0;
    hipError_t hipErr      = hipGetDeviceCount(&deviceCount);
    if((hipErr != hipSuccess) || (deviceCount <= 0))
    {
        fprintf(
            stderr,
            "hipGetDeviceCount failed (%d: %s) — Device count: %d — aborting before tests run.\n",
            static_cast<int>(hipErr),
            hipGetErrorString(hipErr),
            deviceCount);
        return 1;
    }

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Run the tests
    return RUN_ALL_TESTS();
}
