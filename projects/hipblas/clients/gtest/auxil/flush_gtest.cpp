/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_common.hpp"

#include <vector>

namespace
{
    // Test names carry pre_checkin so every CTest category (test_categories.yaml) runs them.
    std::vector<float> copy_to_host(const float* device, size_t n)
    {
        std::vector<float> host(n);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(host.data(), device, n * sizeof(float), hipMemcpyDeviceToHost));
        return host;
    }

    // Seven copies: the doubling fill does a partial last transfer.
    TEST(pre_checkin_hipblas_flush, rotating_buffer_copies_match_the_operand)
    {
        constexpr size_t   n = 1000;
        std::vector<float> values(n);
        for(size_t i = 0; i < n; i++)
            values[i] = float(i);
        device_vector<float> dx(n);
        ASSERT_EQ(hipSuccess,
                  hipMemcpy((float*)dx, values.data(), n * sizeof(float), hipMemcpyHostToDevice));

        device_rotating_buffer<float> rotating(dx, dx.nmemb(), 7);
        ASSERT_EQ(hipSuccess, rotating.memcheck());
        EXPECT_EQ((float*)dx, rotating[0]);
        for(size_t i = 1; i < 7; i++)
        {
            // each copy starts one 256-byte-aligned operand size after the previous one; their
            // base follows the gtest guard padding (g_DVEC_PAD), so the spacing is checked rather
            // than the absolute address
            uintptr_t offset = reinterpret_cast<uintptr_t>(rotating[i])
                               - reinterpret_cast<uintptr_t>(rotating[1]);
            EXPECT_EQ((i - 1) * hipblas_flush_align(n * sizeof(float)), offset) << "copy " << i;
            EXPECT_EQ(values, copy_to_host(rotating[i], n)) << "copy " << i;
        }

        device_rotating_buffer<float> single(dx, dx.nmemb(), 1);
        EXPECT_EQ(hipSuccess, single.memcheck());
        EXPECT_EQ((float*)dx, single[0]);
    }

    TEST(pre_checkin_hipblas_flush, batch_count_uses_the_bytes_a_call_accesses)
    {
        Arguments arg{};
        arg.flush_memory_size = 1 << 20;

        // a 64 x 64 matrix with lda 8192 accesses 16 KiB of its 2 MiB allocation
        device_matrix<float> dA(64, 64, 8192);
        EXPECT_EQ(64u, hipblas_flush_batch_count(arg, dA));

        // a 4-byte operand occupies the 256 byte block its copies start on
        device_vector<float> dx(1);
        EXPECT_EQ(4096u, hipblas_flush_batch_count(arg, dx));

        // vector increments are gaps too: 1024 elements with inc 4 access 4 KiB
        device_vector<float> dy(1024, 4);
        EXPECT_EQ(256u, hipblas_flush_batch_count(arg, dy));
        // operands add up: 256 + 3 x 4096 = 12544 bytes, so 1 MiB takes 84 copies
        EXPECT_EQ(84u, hipblas_flush_batch_count(arg, dx, dy, dy, dy));

        arg.flush_memory_size = 0;
        EXPECT_EQ(1u, hipblas_flush_batch_count(arg, dA, dx));
        arg.flush_batch_count = 5;
        EXPECT_EQ(5u, hipblas_flush_batch_count(arg, dA, dx));
    }

    TEST(pre_checkin_hipblas_flush, batch_count_fits_in_device_memory)
    {
        size_t free_bytes = 0, total_bytes = 0;
        ASSERT_EQ(hipSuccess, hipMemGetInfo(&free_bytes, &total_bytes));

        Arguments arg{};
        // ask for four times the device memory through a small operand
        arg.flush_memory_size = 4 * total_bytes;
        device_vector<float> dx(1 << 20);
        size_t               count = hipblas_flush_batch_count(arg, dx);
        EXPECT_LE((count - 1) * dx.nmemb() * sizeof(float), free_bytes);
    }
}
