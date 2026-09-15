/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <gtest/gtest.h>

#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>

#include <array>
#include <cstddef>
#include <string>

// Tests for ValidateGlobalWorkSize(), the pre-launch guard that rejects a global
// work size the HIP module launch APIs cannot express.
//
// These are pure CPU tests operating on the validator directly: no GPU, no
// allocation, and no large tensors. The geometries below are only ever *described*,
// never launched, so a grid that would need billions of work-items costs nothing
// to test.

namespace {

constexpr std::size_t k32BitLimit = 1ULL << 32;

void ExpectAccepted(const std::array<std::size_t, 3>& gdims)
{
    EXPECT_NO_THROW(miopen::ValidateGlobalWorkSize(gdims, "test_kernel"));
}

void ExpectRejected(const std::array<std::size_t, 3>& gdims)
{
    EXPECT_THROW(miopen::ValidateGlobalWorkSize(gdims, "test_kernel"), miopen::Exception);
}

} // namespace

// NOLINTBEGIN(google-readability-avoid-underscore-in-googletest-name)

// The largest expressible global work size must be accepted: the limit is
// exclusive, so 2^32 - 1 is still a valid launch.
TEST(CPU_LaunchGeometry_NONE, AcceptsLargestExpressibleGlobalWorkSize)
{
    ExpectAccepted({k32BitLimit - 1, 1, 1});
}

TEST(CPU_LaunchGeometry_NONE, AcceptsTypicalGeometry)
{
    ExpectAccepted({std::size_t{4096} * 256, 1, 1});
    ExpectAccepted({1024, 512, 64});
}

// A global work size that is an exact multiple of 2^32 truncates to zero. HIP
// rejects this, but only asynchronously and from an unrelated later call.
TEST(CPU_LaunchGeometry_NONE, RejectsGlobalWorkSizeTruncatingToZero)
{
    ExpectRejected({k32BitLimit, 1, 1});
    ExpectRejected({2 * k32BitLimit, 1, 1});
}

// The dangerous case: a global work size just over 2^32 truncates to a smaller
// non-zero grid that HIP accepts without error, so the kernel silently covers
// only part of the problem. Must be rejected.
TEST(CPU_LaunchGeometry_NONE, RejectsGlobalWorkSizeTruncatingToNonZeroGrid)
{
    ExpectRejected({k32BitLimit + 256, 1, 1});
}

// Every dimension is validated, not just x.
TEST(CPU_LaunchGeometry_NONE, RejectsOverflowInAnyDimension)
{
    ExpectRejected({k32BitLimit, 1, 1});
    ExpectRejected({1, k32BitLimit, 1});
    ExpectRejected({1, 1, k32BitLimit});
}

// The check bounds the global work size only. A partial trailing workgroup is a
// valid launch on this path -- hipExtModuleLaunchKernel takes work-item counts and
// derives the workgroup count itself -- so geometry whose global work size is not
// a multiple of the local work size must NOT be rejected here.
TEST(CPU_LaunchGeometry_NONE, AcceptsGlobalWorkSizeNotMultipleOfLocalWorkSize)
{
    ExpectAccepted({255, 1, 1});
    ExpectAccepted({1000, 1, 1});
    ExpectAccepted({100, 200, 300});
}

// Regression guard for the naive convolution solver, which builds a 1-D grid as
// g_wk[0] = grid_size * block_size with block_size = 256. Chunking bounds
// grid_size by MAX_GRID_SIZE, so the largest geometry the solver can emit must
// remain launchable; if that bound is ever raised to allow grid_size * 256 to
// reach 2^32, this test fails rather than the launch silently truncating.
TEST(CPU_LaunchGeometry_NONE, AcceptsMaxNaiveConvSolverGeometry)
{
    constexpr std::size_t block_size = 256;
    constexpr std::size_t max_grid   = k32BitLimit / block_size;

    // The largest grid that still fits: one workgroup below the limit.
    ExpectAccepted({(max_grid - 1) * block_size, 1, 1});

    // Exactly at the limit the launch is not expressible.
    ExpectRejected({max_grid * block_size, 1, 1});
}

// The exception must name the kernel so a rejected launch is attributable during
// Find, where many candidates are evaluated in sequence.
TEST(CPU_LaunchGeometry_NONE, ReportsKernelNameOnRejection)
{
    try
    {
        miopen::ValidateGlobalWorkSize({k32BitLimit, 1, 1}, "naive_conv_fwd");
        FAIL() << "expected the global work size guard to throw";
    }
    catch(const miopen::Exception& e)
    {
        EXPECT_NE(std::string{e.what()}.find("naive_conv_fwd"), std::string::npos)
            << "exception should name the offending kernel: " << e.what();
    }
}

// NOLINTEND(google-readability-avoid-underscore-in-googletest-name)
