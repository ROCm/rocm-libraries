/*******************************************************************************
 *
 * MIT License
 *
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
 *******************************************************************************/
// Compile-time coverage checks backend support, register sizes and test dispatch.
// Runtime coverage detects no-op instructions and incorrect packed lane layouts.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include "../gemm/gemm_predicates_base.hpp"

namespace
{
    using namespace rocwmma;

#if ROCWMMA_ARCH_GFX117X
    static_assert(ROCWMMA_ARCH_GFX12 && !ROCWMMA_ARCH_GFX11);
    static_assert(Constants::AMDGCN_CURRENT_ARCH_ID
                  == (ROCWMMA_ARCH_GFX1170 ? Constants::AMDGCN_ARCH_ID_GFX1170
                                          : Constants::AMDGCN_ARCH_ID_GFX1171));
    static_assert(ROCWMMA_FP8);
    static_assert(Constants::AMDGCN_WAVE_SIZE == 32);

    template <typename A, typename B, typename C, uint32_t K>
    constexpr bool checkBackend()
    {
        using Op = detail::amdgcn_wmma<A, B, C, 16, 16, K>;
        static_assert(MmaTraits<Op>::is_supported);
        static_assert(sizeof(typename Op::ARegsT) == 16 * K * sizeof(A) / 32);
        static_assert(sizeof(typename Op::BRegsT) == 16 * K * sizeof(B) / 32);
        static_assert(sizeof(typename Op::CRegsT) == 16 * 16 * sizeof(C) / 32);
        return true;
    }

    template <typename Input, typename Compute>
    constexpr bool checkDispatch()
    {
        using Predicates = GemmPredicatesBase<16, 16, 16, Input, Compute, Compute,
                                              1, 1, 32, 1, 32, Constants::AMDGCN_CURRENT_ARCH_ID>;
        static_assert(Predicates::enableBuild());
        static_assert(Predicates::enableRun());
        return true;
    }

    static_assert(checkBackend<float16_t, float16_t, float32_t, 8>());
    static_assert(checkBackend<float16_t, float16_t, float32_t, 16>());
    static_assert(checkBackend<float16_t, float16_t, float16_t, 8>());
    static_assert(checkBackend<float16_t, float16_t, float16_t, 16>());
    static_assert(checkBackend<bfloat16_t, bfloat16_t, float32_t, 8>());
    static_assert(checkBackend<bfloat16_t, bfloat16_t, float32_t, 16>());
    static_assert(checkBackend<bfloat16_t, bfloat16_t, bfloat16_t, 8>());
    static_assert(checkBackend<bfloat16_t, bfloat16_t, bfloat16_t, 16>());
#if !ROCWMMA_NO_HALF
    static_assert(checkBackend<hfloat16_t, hfloat16_t, float32_t, 8>());
    static_assert(checkBackend<hfloat16_t, hfloat16_t, float32_t, 16>());
    static_assert(checkBackend<hfloat16_t, hfloat16_t, hfloat16_t, 8>());
    static_assert(checkBackend<hfloat16_t, hfloat16_t, hfloat16_t, 16>());
#endif
    static_assert(checkBackend<int8_t, int8_t, int32_t, 8>());
    static_assert(checkBackend<int8_t, int8_t, int32_t, 16>());
    static_assert(checkBackend<uint8_t, uint8_t, int32_t, 16>());
    static_assert(checkBackend<float8_t, float8_t, float32_t, 16>());
    static_assert(checkBackend<bfloat8_t, bfloat8_t, float32_t, 16>());
    static_assert(checkBackend<float8_t, bfloat8_t, float32_t, 16>());
    static_assert(checkBackend<bfloat8_t, float8_t, float32_t, 16>());
    static_assert(checkDispatch<float16_t, float32_t>());
    static_assert(checkDispatch<bfloat16_t, float32_t>());
    static_assert(checkDispatch<int8_t, int32_t>());
    static_assert(checkDispatch<float8_t, float32_t>());
    static_assert(checkDispatch<bfloat8_t, float32_t>());
#endif

    template <typename A, typename B, typename C, uint32_t K, typename D>
    __global__ void wmmaTile(A const* a, B const* b, D* d)
    {
#if ROCWMMA_ARCH_GFX117X
        fragment<matrix_a, 16, 16, K, A, row_major> fa;
        fragment<matrix_b, 16, 16, K, B, col_major> fb;
        fragment<accumulator, 16, 16, K, C> fc;
        load_matrix_sync(fa, a, K);
        load_matrix_sync(fb, b, K);
        fill_fragment(fc, C(1));
        mma_sync(fc, fa, fb, fc);
        fragment<accumulator, 16, 16, K, D> fd;
        for(uint32_t i = 0; i < fd.num_elements; ++i)
            fd.x[i] = D(fc.x[i]);
        store_matrix_sync(d, fd, 16, mem_row_major);
#endif
    }

    template <typename A, typename B, typename C, uint32_t K = 16, typename D = C>
    void checkTile()
    {
        int count = 0;
        auto status = hipGetDeviceCount(&count);
        if(status == hipErrorNoDevice || (status == hipSuccess && count == 0))
            GTEST_SKIP() << "No ROCm-capable device available";
        ASSERT_EQ(status, hipSuccess);
        int device = 0;
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        std::string name(props.gcnArchName);
        if(name.substr(0, 7) != "gfx1170" && name.substr(0, 7) != "gfx1171")
            GTEST_SKIP() << "Requires gfx1170 or gfx1171";
        A* a = nullptr;
        B* b = nullptr;
        D* d = nullptr;
        ASSERT_EQ(hipMallocManaged(&a, sizeof(A) * 16 * K), hipSuccess);
        ASSERT_EQ(hipMallocManaged(&b, sizeof(B) * 16 * K), hipSuccess);
        ASSERT_EQ(hipMallocManaged(&d, sizeof(D) * 256), hipSuccess);
        for(uint32_t i = 0; i < 16; ++i)
            for(uint32_t k = 0; k < K; ++k)
            {
                a[i * K + k] = A(int((i + k) % 3) - (is_same_v<A, int8_t> ? 1 : 0));
                b[i * K + k] = B(int((2 * i + k) % 3));
            }
        for(uint32_t i = 0; i < 256; ++i)
            d[i] = D(0);
        hipLaunchKernelGGL((wmmaTile<A, B, C, K, D>), dim3(1), dim3(32), 0, 0, a, b, d);
        EXPECT_EQ(hipGetLastError(), hipSuccess);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        for(uint32_t i = 0; i < 16; ++i)
            for(uint32_t j = 0; j < 16; ++j)
            {
                int expected = 1;
                for(uint32_t k = 0; k < K; ++k)
                    expected += (int((i + k) % 3) - (is_same_v<A, int8_t> ? 1 : 0))
                                * int((2 * j + k) % 3);
                EXPECT_EQ(float(d[i * 16 + j]), float(expected)) << "row=" << i << ", col=" << j;
            }
        EXPECT_EQ(hipFree(a), hipSuccess);
        EXPECT_EQ(hipFree(b), hipSuccess);
        EXPECT_EQ(hipFree(d), hipSuccess);
    }
}

TEST(Gfx117xWmma, F16F32)
{
    checkTile<float16_t, float16_t, float32_t, 16>();
}

TEST(Gfx117xWmma, F16F16)
{
    checkTile<float16_t, float16_t, float16_t, 16>();
}

TEST(Gfx117xWmma, BF16F32)
{
    checkTile<bfloat16_t, bfloat16_t, float32_t, 16>();
}

TEST(Gfx117xWmma, BF16BF16)
{
    checkTile<bfloat16_t, bfloat16_t, bfloat16_t, 16>();
}

TEST(Gfx117xWmma, I8)
{
    checkTile<int8_t, int8_t, int32_t, 16>();
}

TEST(Gfx117xWmma, U8)
{
    checkTile<uint8_t, uint8_t, int32_t, 16>();
}

TEST(Gfx117xWmma, F8F8)
{
    checkTile<float8_t, float8_t, float32_t, 16>();
}

TEST(Gfx117xWmma, BF8BF8)
{
    checkTile<bfloat8_t, bfloat8_t, float32_t, 16>();
}

TEST(Gfx117xWmma, F8BF8)
{
    checkTile<float8_t, bfloat8_t, float32_t, 16>();
}

TEST(Gfx117xWmma, BF8F8)
{
    checkTile<bfloat8_t, float8_t, float32_t, 16>();
}

TEST(Gfx117xWmma, SmallK)
{
    checkTile<float16_t, float16_t, float32_t, 8>();
}

TEST(Gfx117xWmma, RepeatedK)
{
    checkTile<float16_t, float16_t, float32_t, 32>();
}

#if !ROCWMMA_NO_HALF
TEST(Gfx117xWmma, HalfForwarding)
{
    checkTile<hfloat16_t, hfloat16_t, hfloat16_t, 16>();
}

#endif

TEST(Gfx117xWmma, F8HalfOutput)
{
    checkTile<float8_t, float8_t, float32_t, 16, float16_t>();
}
