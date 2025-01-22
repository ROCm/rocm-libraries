/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#ifndef HIPTENSOR_SAMPLES_CONTRACTION_COMMON_HPP
#define HIPTENSOR_SAMPLES_CONTRACTION_COMMON_HPP

#define MAX_ELEMENTS_PRINT_COUNT 512

#define HIPTENSOR_FREE_DEVICE(ptr)     \
    if(ptr != nullptr)                 \
    {                                  \
        CHECK_HIP_ERROR(hipFree(ptr)); \
    }

#define HIPTENSOR_FREE_HOST(ptr)           \
    if(ptr != nullptr)                     \
    {                                      \
        CHECK_HIP_ERROR(hipHostFree(ptr)); \
    }

inline bool isF32Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx908") != std::string::npos)
           || (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx940") != std::string::npos)
           || (deviceName.find("gfx941") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos)
           || (deviceName.find("gfx950") != std::string::npos);
}

inline bool isF64Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx940") != std::string::npos)
           || (deviceName.find("gfx941") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos)
           || (deviceName.find("gfx950") != std::string::npos);
}

#endif // HIPTENSOR_SAMPLES_CONTRACTION_COMMON_HPP
