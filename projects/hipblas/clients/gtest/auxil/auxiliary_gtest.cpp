/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "utility.h"
#include <math.h>
#include <stdexcept>
#include <vector>

namespace
{

    TEST(hipblas_auxiliary, statusToString)
    {
        EXPECT_EQ(0,
                  strcmp("HIPBLAS_STATUS_ALLOC_FAILED",
                         hipblasStatusToString(HIPBLAS_STATUS_ALLOC_FAILED)));
    }

    // Test hipblasGetVersion with valid handle
    TEST(hipblas_version, getVersion_validHandle)
    {
        hipblasHandle_t handle;
        ASSERT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasCreate(&handle));

        int version = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetVersion(handle, &version));

        // Verify version format

        assert(hipblasVersionMinor < hipblasVersionK);
        assert(hipblasVersionPatch < hipblasVersionK);

        int expected_version
            = (hipblasVersionMajor * hipblasVersionK + hipblasVersionMinor) * hipblasVersionK
              + hipblasVersionPatch;
        EXPECT_EQ(expected_version, version);

        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasDestroy(handle));
    }

    // Test hipblasGetVersion with NULL handle (should still work)
    TEST(hipblas_version, getVersion_nullHandle)
    {
        int version = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetVersion(nullptr, &version));

        // Verify version format
        int expected_version
            = (hipblasVersionMajor * hipblasVersionK + hipblasVersionMinor) * hipblasVersionK
              + hipblasVersionPatch;
        EXPECT_EQ(expected_version, version);
    }

    // Test hipblasGetVersion with NULL pointer (should fail)
    TEST(hipblas_version, getVersion_nullPointer)
    {
        hipblasHandle_t handle;
        ASSERT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasCreate(&handle));

        EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, hipblasGetVersion(handle, nullptr));

        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasDestroy(handle));
    }

    // Test hipblasGetProperty for major version
    TEST(hipblas_version, getProperty_major)
    {
        int value = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_MAJOR_VERSION, &value));
        EXPECT_EQ(hipblasVersionMajor, value);
    }

    // Test hipblasGetProperty for minor version
    TEST(hipblas_version, getProperty_minor)
    {
        int value = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_MINOR_VERSION, &value));
        EXPECT_EQ(hipblasVersionMinor, value);
    }

    // Test hipblasGetProperty for patch level
    TEST(hipblas_version, getProperty_patch)
    {
        int value = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_PATCH_LEVEL, &value));
        EXPECT_EQ(hipblasVersionPatch, value);
    }

    // Test hipblasGetProperty with NULL pointer (should fail)
    TEST(hipblas_version, getProperty_nullPointer)
    {
        EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, hipblasGetProperty(HIPBLAS_MAJOR_VERSION, nullptr));
    }

    // Test hipblasGetProperty with invalid type (should fail)
    TEST(hipblas_version, getProperty_invalidType)
    {
        int value = 0;
        // Use an invalid enum value (casting to force invalid type)
        EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE,
                  hipblasGetProperty(static_cast<hipblasLibraryProperty_t>(999), &value));
    }

    // Test consistency between GetVersion and GetProperty
    TEST(hipblas_version, version_property_consistency)
    {
        int version = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetVersion(nullptr, &version));

        int major = 0, minor = 0, patch = 0;
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_MAJOR_VERSION, &major));
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_MINOR_VERSION, &minor));
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, hipblasGetProperty(HIPBLAS_PATCH_LEVEL, &patch));

        // Verify that reconstructed version matches
        int reconstructed_version = (major * hipblasVersionK + minor) * hipblasVersionK + patch;
        EXPECT_EQ(version, reconstructed_version);
    }

} // namespace
