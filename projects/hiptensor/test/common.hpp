/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hiptensor/hiptensor.h>

#include <hiptensor/internal/hiptensor_utility.hpp>

//! @brief hipTensor data types
typedef enum
{
    HIPTENSOR_MEMORY_LAYOUT_DEFAULT      = 0, // Use the default memory layout
    HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR = 1, // Strides increase monotonically from left to right
    HIPTENSOR_MEMORY_LAYOUT_ROW_MAJOR    = 2, // Strides increase monotonically from right to left
    HIPTENSOR_MEMORY_LAYOUT_OTHER        = 3, // Other memory layout
} hiptensorMemoryLayout_t;

namespace hiptensor
{
    std::string hipMemoryLayoutToString(hiptensorMemoryLayout_t hipMemoryLayout);

    struct HostDeleter
    {
        void operator()(void* ptr)
        {
            operator delete(ptr);
        }
    };

    struct DeviceDeleter
    {
        void operator()(void* ptr)
        {
            CHECK_HIP_ERROR(hipFree(ptr));
        }
    };

    namespace test
    {
        // Cross-platform safe temporary filename generator
        inline std::string generateTempFilename(const std::string& prefix = "hiptensor_test_")
        {
            std::random_device              rd;
            std::mt19937                    gen(rd());
            std::uniform_int_distribution<> dis(100000, 999999);

            std::filesystem::path temp_dir    = std::filesystem::temp_directory_path();
            constexpr int         maxAttempts = 10;
            for(int i = 0; i < maxAttempts; i++)
            {
                std::filesystem::path candidate
                    = temp_dir / (prefix + std::to_string(dis(gen)) + ".tmp");
                if(!std::filesystem::exists(candidate))
                {
                    return candidate.string();
                }
            }
            throw std::runtime_error("Failed to generate a unique temporary filename after "
                                     + std::to_string(maxAttempts) + " attempts");
        }

        // Cross-platform safe file open for test code (does not depend on the library's internal symbol)
        inline FILE* safeFopen(const char* filename, const char* mode)
        {
#ifdef _WIN32
            FILE* file = nullptr;
            if(fopen_s(&file, filename, mode) == 0)
                return file;
            return nullptr;
#else
            return fopen(filename, mode);
#endif
        }

        // Redirect logger output to the null device to silence it during tests
        inline void silenceLogger()
        {
#ifdef _WIN32
            const char* nullDevice = "NUL";
#else
            const char* nullDevice = "/dev/null";
#endif
            if(hiptensorLoggerOpenFile(nullDevice) != HIPTENSOR_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    std::string("Failed to silence hiptensor logger: could not open ")
                    + nullDevice);
            }
        }

    } // namespace test

} // namespace hiptensor
