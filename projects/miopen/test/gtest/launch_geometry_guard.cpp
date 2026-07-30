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

// Regression test for the pre-launch geometry guard in HIPOCKernelInvoke::run().
//
// hipExtModuleLaunchKernel() takes the global work sizes as uint32_t and the
// hardware caps gridDim.x at 2^31-1 and gridDim.y/z at 65535. A solver can
// compute an out-of-range geometry - e.g. the naive convolution solver sets
// g_wk[0] = grid_size * block_size (= n*k*256 for NCHW), which silently
// overflows 32 bits once the batch N is large enough (the ViT patch-embed
// Conv2d(3, 1024, kernel=14, stride=14) reaches this at N = 32768). Before the
// fix such a launch truncated to a wrong (possibly zero) grid and made HIP
// raise an asynchronous "invalid configuration argument" that corrupted the
// whole context. The guard rejects the launch (throwing miopen::Exception)
// BEFORE issuing any HIP call, so the candidate is dropped cleanly during
// Find/EvaluateInvokers and the context is never poisoned.
//
// These are pure CPU tests: the guard runs before the kernel function pointer
// is dereferenced, so a null function is used on purpose and no GPU launch is
// performed.

#include <gtest/gtest.h>

#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>

#include <array>
#include <string>

namespace {

// Build an invoke object with the given launch geometry. The kernel function
// pointer is intentionally null: a valid geometry would be dereferenced at the
// HIP launch, but every case below is out-of-range and must be rejected by the
// guard before that point.
miopen::HIPOCKernelInvoke MakeInvoke(std::array<size_t, 3> gdims, std::array<size_t, 3> ldims)
{
    return miopen::HIPOCKernelInvoke{/*pstream*/ nullptr,
                                     /*pfun*/ nullptr,
                                     /*pldims*/ ldims,
                                     /*pgdims*/ gdims,
                                     /*pname*/ "launch_geometry_guard_test",
                                     /*pcallback*/ nullptr,
                                     /*pcoop_launch*/ false};
}

// Invoke and assert the guard rejected the geometry. We check the exception
// message so the failure is attributed to the guard and not to some later HIP
// error from the null function pointer.
void ExpectGuardRejects(const miopen::HIPOCKernelInvoke& invoke)
{
    try
    {
        // A single dummy scalar argument is enough to reach run().
        invoke(int{0});
        FAIL() << "expected the launch-geometry guard to throw";
    }
    catch(const miopen::Exception& e)
    {
        EXPECT_NE(std::string{e.what()}.find("Invalid launch"), std::string::npos)
            << "threw, but not from the geometry guard: " << e.what();
    }
}

} // namespace

// The suite name follows MIOpen's mandatory gtest naming scheme
// (HW_Name_DATATYPE, enforced by test/gtest/check_names.py), which requires
// underscores. Silence the Googletest-underscore check for these registrations.
// NOLINTBEGIN(google-readability-avoid-underscore-in-googletest-name)

// n*k*256 overflows uint32_t. For the ViT patch-embed conv at N = 32768,
// K = 1024, g_wk[0] = 32768 * 1024 * 256 = 2^33, which truncates to 0.
TEST(CPU_LaunchGeometryGuard_NONE, RejectsGlobalWorkSizeOverflowTruncatingToZero)
{
    constexpr size_t block = 256;
    constexpr size_t n     = 32768;
    constexpr size_t k     = 1024;
    constexpr size_t gwk0  = n * k * block; // 2^33
    static_assert(gwk0 >= (1ULL << 32), "expected a 32-bit overflow");
    static_assert((gwk0 & 0xffffffffULL) == 0, "expected truncation to zero");

    auto invoke = MakeInvoke({gwk0, 1, 1}, {block, 1, 1});
    ExpectGuardRejects(invoke);
}

// A non-multiple-of-2^32 overflow (e.g. the observed N = 66096) truncates to a
// wrong non-zero value instead of zero; it must also be rejected.
TEST(CPU_LaunchGeometryGuard_NONE, RejectsGlobalWorkSizeOverflowTruncatingToNonZero)
{
    constexpr size_t block = 256;
    constexpr size_t n     = 66096;
    constexpr size_t k     = 1024;
    constexpr size_t gwk0  = n * k * block;
    static_assert(gwk0 >= (1ULL << 32), "expected a 32-bit overflow");
    static_assert((gwk0 & 0xffffffffULL) != 0, "expected truncation to non-zero");

    auto invoke = MakeInvoke({gwk0, 1, 1}, {block, 1, 1});
    ExpectGuardRejects(invoke);
}

// Global work sizes fit in 32 bits, but gridDim.y exceeds the 65535 hardware cap.
TEST(CPU_LaunchGeometryGuard_NONE, RejectsGridDimYExceedingHwLimit)
{
    constexpr size_t grid_y = 70000; // > 65535
    auto invoke             = MakeInvoke({1, grid_y, 1}, {1, 1, 1});
    ExpectGuardRejects(invoke);
}

// Same for gridDim.z.
TEST(CPU_LaunchGeometryGuard_NONE, RejectsGridDimZExceedingHwLimit)
{
    constexpr size_t grid_z = 70000; // > 65535
    auto invoke             = MakeInvoke({1, 1, grid_z}, {1, 1, 1});
    ExpectGuardRejects(invoke);
}

// gridDim.x above 2^31-1 (with global work size still under the 2^32 uint32_t
// limit) must also be rejected.
TEST(CPU_LaunchGeometryGuard_NONE, RejectsGridDimXExceedingHwLimit)
{
    constexpr size_t gwk0 = (1ULL << 32) - 2; // < 2^32, passes the uint32_t check
    constexpr size_t ldim = 1;                // grid_x == gwk0 > 2^31-1
    static_assert(gwk0 < (1ULL << 32), "must stay within the 32-bit launch limit");
    static_assert(gwk0 > 0x7fffffffULL, "grid_x must exceed 2^31-1");

    auto invoke = MakeInvoke({gwk0, 1, 1}, {ldim, 1, 1});
    ExpectGuardRejects(invoke);
}

// NOLINTEND(google-readability-avoid-underscore-in-googletest-name)
