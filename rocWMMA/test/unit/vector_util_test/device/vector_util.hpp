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

#ifndef ROCWMMA_DEVICE_VECTOR_UTIL_TEST_HPP
#define ROCWMMA_DEVICE_VECTOR_UTIL_TEST_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline DataT get(VecT<DataT, VecSize> const& v, uint32_t idx)
    {
        return v.data[idx];
    }

    template <typename DataT, uint32_t VecSize, uint32_t Start = 0>
    ROCWMMA_DEVICE static inline auto generateSeqVec()
    {
        auto buildSeq = [](auto&& idx) {
            constexpr auto Index = std::decay_t<decltype(idx)>::value;
            return static_cast<DataT>(Index) + static_cast<DataT>(Start);
        };

        return vector_generator<DataT, VecSize>()(buildSeq);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool vectorGeneratorTestBasic()
    {
        bool err = false;

        auto res = generateSeqVec<DataT, VecSize>();

        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(i));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool vectorGeneratorTestWithArgs()
    {
        bool err = false;

        auto sum = [](auto&& idx, auto&& v0, auto&& v1) {
            constexpr auto Index = std::decay_t<decltype(idx)>::value;
            return get<Index>(v0) + get<Index>(v1);
        };

        auto v0 = VecT<DataT, VecSize>{static_cast<DataT>(1.0f)};
        auto v1 = VecT<DataT, VecSize>{static_cast<DataT>(2.0f)};

        auto res = vector_generator<DataT, VecSize>()(sum, v0, v1);

        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != (static_cast<DataT>(1.0f) + static_cast<DataT>(2.0f)));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool extractEvenTest()
    {
        bool err = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = extractEven(v);

        // Handle the general case
        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize / 2; i++)
            {
                err |= (get(res, i) != static_cast<DataT>(i * 2));
            }
        }
        // Handle case where VecSize == 1
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool extractOddTest()
    {
        bool err = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = extractOdd(v);

        // Handle general case
        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize / 2; i++)
            {
                err |= (get(res, i) != static_cast<DataT>(i * 2 + 1));
            }
        }
        // Handle case where VecSize == 1
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool reorderEvenOddTest()
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;
        bool err         = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = reorderEvenOdd(v);

        // Handle general case
        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize; i++)
            {
                if(i < VecSize / 2)
                {
                    err |= (get(res, i) != static_cast<DataT>(i * 2));
                }
                else
                {
                    err |= (get(res, i) != static_cast<DataT>((i - VecSize / 2) * 2 + 1));
                }
            }
        }
        // Handle case where VecSize == 1
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool reorderOddEvenTest()
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;
        bool err         = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = reorderOddEven(v);

        // Handle general case
        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize; i++)
            {
                if(i < VecSize / 2)
                {
                    err |= (get(res, i) != static_cast<DataT>(i * 2 + 1));
                }
                else
                {
                    err |= (get(res, i) != static_cast<DataT>((i - VecSize / 2) * 2));
                }
            }
        }
        // Handle case where VecSize == 1
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool extractLoTest()
    {
        bool err = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = extractLo(v);

        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize / 2; i++)
            {
                err |= (get(res, i) != static_cast<DataT>(i));
            }
        }
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool extractHiTest()
    {
        bool err = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = extractHi(v);

        if constexpr(VecSize > 1)
        {
            for(uint32_t i = 0; i < VecSize / 2; i++)
            {
                err |= (get(res, i) != static_cast<DataT>(i + VecSize / 2));
            }
        }
        else
        {
            err |= (v != res);
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool concatTest()
    {
        bool err = false;

        auto v   = generateSeqVec<DataT, VecSize>();
        auto res = concat(v, v);

        for(uint32_t i = 0; i < VecSize * 2; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(i % VecSize));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool zipTest()
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;
        bool err         = false;

        auto v0  = generateSeqVec<DataT, VecSize>();
        auto v1  = generateSeqVec<DataT, VecSize, VecSize>();
        auto res = zip(v0, v1);

        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(i + VecSize * (i & 1)));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool unpackLoTest()
    {
        bool err = false;

        auto v0  = generateSeqVec<DataT, VecSize>();
        auto v1  = generateSeqVec<DataT, VecSize, VecSize>();
        auto res = unpackLo(v0, v1);

        // 0, VecSize , 1, VecSize + 1, ...
        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(i / 2 + VecSize * (i & 1)));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool unpackHiTest()
    {
        bool err = false;

        auto v0  = generateSeqVec<DataT, VecSize>();
        auto v1  = generateSeqVec<DataT, VecSize, VecSize>();
        auto res = unpackHi(v0, v1);

        // VecSize / 2, VecSize / 2 + VecSize , VecSize / 2 + 1, VecSize / 2 + VecSize + 1, ...
        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(VecSize / 2 + i / 2 + VecSize * (i & 1)));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_KERNEL void vectorUtilTest(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        err = err ? err : vectorGeneratorTestBasic<DataT, VecSize>();
        err = err ? err : vectorGeneratorTestWithArgs<DataT, VecSize>();
        err = err ? err : extractEvenTest<DataT, VecSize>();
        err = err ? err : extractOddTest<DataT, VecSize>();
        err = err ? err : reorderEvenOddTest<DataT, VecSize>();
        err = err ? err : reorderOddEvenTest<DataT, VecSize>();
        err = err ? err : extractLoTest<DataT, VecSize>();
        err = err ? err : extractHiTest<DataT, VecSize>();
        err = err ? err : concatTest<DataT, VecSize>();
        err = err ? err : zipTest<DataT, VecSize>();
        err = err ? err : unpackLoTest<DataT, VecSize>();
        err = err ? err : unpackHiTest<DataT, VecSize>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_VECTOR_UTIL_TEST_HPP
