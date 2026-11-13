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
// uint8 WMMA accumulates into signed int32 at every supported BlockK, including the
// zero-padded K=8 form. The IU8 WMMA instructions have no unsigned accumulator.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <string>

namespace
{
    using namespace rocwmma;

#if ROCWMMA_ARCH_GFX12
    template <typename InputT, typename ComputeT, uint32_t BlockK>
    using SelectedOp = typename WmmaSelector<InputT, InputT, ComputeT, 16, 16, BlockK>::SelectedOp;

    template <typename Op, uint32_t BlockK>
    constexpr bool isSupportedWithBlockK()
    {
        return MmaTraits<Op>::is_supported && MmaTraits<Op>::BlockK == BlockK;
    }

    static_assert(isSupportedWithBlockK<SelectedOp<uint8_t, int32_t, 8>, 8>());
    static_assert(isSupportedWithBlockK<SelectedOp<uint8_t, int32_t, 16>, 16>());
    static_assert(isSupportedWithBlockK<SelectedOp<int8_t, int32_t, 8>, 8>());
    static_assert(!MmaTraits<SelectedOp<uint8_t, uint32_t, 8>>::is_supported);
    static_assert(!MmaTraits<SelectedOp<uint8_t, uint32_t, 16>>::is_supported);
#endif

    uint8_t valueA(uint32_t row, uint32_t k)
    {
        // Values above 127 detect inputs treated as signed.
        return uint8_t(128u + (row * 7u + k * 3u) % 128u);
    }

    uint8_t valueB(uint32_t col, uint32_t k)
    {
        return uint8_t(255u - (col * 5u + k) % 128u);
    }

    template <uint32_t K>
    __global__ void wmmaUint8Tile(uint8_t const* a, uint8_t const* b, int32_t* d)
    {
#if ROCWMMA_ARCH_GFX12
        fragment<matrix_a, 16, 16, K, uint8_t, row_major> fa;
        fragment<matrix_b, 16, 16, K, uint8_t, col_major> fb;
        fragment<accumulator, 16, 16, K, int32_t>         fc;
        load_matrix_sync(fa, a, K);
        load_matrix_sync(fb, b, K);
        fill_fragment(fc, int32_t(1));
        mma_sync(fc, fa, fb, fc);
        store_matrix_sync(d, fc, 16, mem_row_major);
#endif
    }

    template <uint32_t K>
    void checkUint8Tile()
    {
        int  count  = 0;
        auto status = hipGetDeviceCount(&count);
        if(status == hipErrorNoDevice || (status == hipSuccess && count == 0))
            GTEST_SKIP() << "No ROCm-capable device available";
        ASSERT_EQ(status, hipSuccess);
        int             device = 0;
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        std::string name(props.gcnArchName);
        name = name.substr(0, name.find(':'));
        if(name != "gfx1170" && name != "gfx1171" && name != "gfx1200" && name != "gfx1201"
           && name != "gfx1250")
            GTEST_SKIP() << "Requires gfx1170, gfx1171, gfx1200, gfx1201 or gfx1250";
        uint8_t* a = nullptr;
        uint8_t* b = nullptr;
        int32_t* d = nullptr;
        ASSERT_EQ(hipMallocManaged(&a, 16 * K), hipSuccess);
        ASSERT_EQ(hipMallocManaged(&b, 16 * K), hipSuccess);
        ASSERT_EQ(hipMallocManaged(&d, sizeof(int32_t) * 256), hipSuccess);
        for(uint32_t i = 0; i < 16; ++i)
            for(uint32_t k = 0; k < K; ++k)
            {
                a[i * K + k] = valueA(i, k);
                b[i * K + k] = valueB(i, k);
            }
        for(uint32_t i = 0; i < 256; ++i)
            d[i] = 0;
        hipLaunchKernelGGL((wmmaUint8Tile<K>), dim3(1), dim3(32), 0, 0, a, b, d);
        EXPECT_EQ(hipGetLastError(), hipSuccess);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        for(uint32_t i = 0; i < 16; ++i)
            for(uint32_t j = 0; j < 16; ++j)
            {
                int32_t expected = 1;
                for(uint32_t k = 0; k < K; ++k)
                    expected += int32_t(valueA(i, k)) * int32_t(valueB(j, k));
                EXPECT_EQ(d[i * 16 + j], expected) << "row=" << i << ", col=" << j;
            }
        EXPECT_EQ(hipFree(a), hipSuccess);
        EXPECT_EQ(hipFree(b), hipSuccess);
        EXPECT_EQ(hipFree(d), hipSuccess);
    }
}

TEST(WmmaUint8, K8)
{
    checkUint8Tile<8>();
}

TEST(WmmaUint8, K16)
{
    checkUint8Tile<16>();
}
