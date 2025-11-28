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

#ifndef ROCWMMA_DEVICE_VECTOR_TEST_HPP
#define ROCWMMA_DEVICE_VECTOR_TEST_HPP

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool bcastCtorTest()
    {
        bool err = false;

        auto vec = make_vector<DataT, VecSize>(5.0f);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec) != static_cast<DataT>(5.0f));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool copyCtorTest()
    {
        bool err = false;

        auto                 vec0 = make_vector<DataT, VecSize>(5.0f);
        VecT<DataT, VecSize> vec1{vec0};

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != static_cast<DataT>(5.0f));
            err |= (get<i>(vec1) != static_cast<DataT>(5.0f));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool moveCtorTest()
    {
        bool err = false;

        auto                 vec0 = make_vector<DataT, VecSize>(5.0f);
        VecT<DataT, VecSize> vec1{std::move(vec0)};

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec1) != static_cast<DataT>(5.0f));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool assignmentTest()
    {
        bool err = false;

        auto                 vec0 = make_vector<DataT, VecSize>(5.0f);
        VecT<DataT, VecSize> vec1;

        vec1 = vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != static_cast<DataT>(5.0f));
            err |= (get<i>(vec1) != static_cast<DataT>(5.0f));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool assignmentMoveTest()
    {
        bool err = false;

        auto                 vec0 = make_vector<DataT, VecSize>(5.0f);
        VecT<DataT, VecSize> vec1;

        vec1 = std::move(vec0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec1) != static_cast<DataT>(5.0f));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorPlusEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        vec0 += vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) + static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorPlusS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = vec0 + static_cast<DataT>(3.0f);
        auto vec2 = static_cast<DataT>(3.0f) + vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) + static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) + static_cast<DataT>(3.0f)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorPlusV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = vec0 + vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) + static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMinusEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        vec0 -= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) - static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMinusS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = vec0 - static_cast<DataT>(3.0f);
        auto vec2 = static_cast<DataT>(3.0f) - vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) - static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(3.0f) - static_cast<DataT>(5.0f)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMinusV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = vec0 - vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) - static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMultEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        vec0 *= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) * static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMultS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = vec0 * static_cast<DataT>(3.0f);
        auto vec2 = static_cast<DataT>(3.0f) * vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) * static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) * static_cast<DataT>(3.0f)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorMultV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = vec0 * vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(5.0f) * static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorDivEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        vec0 /= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(6.0f) / static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorDivS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6.0f);
        auto vec1 = vec0 / static_cast<DataT>(3.0f);
        auto vec2 = static_cast<DataT>(3.0f) / vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(6.0f)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(6.0f) / static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(3.0f) / static_cast<DataT>(6.0f)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorDivV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = vec0 / vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(6.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(6.0f) / static_cast<DataT>(3.0f)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorInc()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = vec0++;
        auto vec2 = ++vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(7.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec2) != (static_cast<DataT>(7.0f)));
        });

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorDec()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(5.0f);
        auto vec1 = vec0--;
        auto vec2 = --vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(5.0f)));
            err |= (get<i>(vec2) != (static_cast<DataT>(3.0f)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6.0f);
        auto vec1 = make_vector<DataT, VecSize>(4.0f);
        vec0 %= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(6u) % static_cast<DataT>(4u)));
            err |= (get<i>(vec1) != (static_cast<DataT>(4u)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6u);
        auto vec1 = vec0 % static_cast<DataT>(4u);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(6u)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(6u) % static_cast<DataT>(4u)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(6u);
        auto vec1 = make_vector<DataT, VecSize>(4u);
        auto vec2 = vec0 % vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(6u)));
            err |= (get<i>(vec1) != (static_cast<DataT>(4u)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(6u) % static_cast<DataT>(4u)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorModV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        vec0 &= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) & static_cast<DataT>(0xF0)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = vec0 & static_cast<DataT>(0xF0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) & static_cast<DataT>(0xF0)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        auto vec2 = vec0 & vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) & static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitAndV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        vec0 |= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) | static_cast<DataT>(0xF0)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = vec0 | static_cast<DataT>(0xF0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) | static_cast<DataT>(0xF0)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        auto vec2 = vec0 | vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) | static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitOrV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        vec0 ^= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) ^ static_cast<DataT>(0xF0)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = vec0 ^ static_cast<DataT>(0xF0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) ^ static_cast<DataT>(0xF0)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0xF0);
        auto vec2 = vec0 ^ vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0xF0)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) ^ static_cast<DataT>(0xF0)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitXorV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0x03);
        vec0 >>= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) >> static_cast<DataT>(0x03)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0x03)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = vec0 >> static_cast<DataT>(0x03);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) >> static_cast<DataT>(0x03)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0x03);
        auto vec2 = vec0 >> vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0x03)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) >> static_cast<DataT>(0x03)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShrV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0x03);
        vec0 <<= vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) << static_cast<DataT>(0x03)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0x03)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = vec0 << static_cast<DataT>(0x03);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) << static_cast<DataT>(0x03)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = make_vector<DataT, VecSize>(0x03);
        auto vec2 = vec0 << vec1;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != (static_cast<DataT>(0x03)));
            err |= (get<i>(vec2)
                    != static_cast<DataT>(static_cast<DataT>(0x0F) << static_cast<DataT>(0x03)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
            err |= (get<i>(vec1) == (get<i>(vec2)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlEqV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlS()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitShlV()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitInv()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(0x0F);
        auto vec1 = ~vec0;

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(0x0F)));
            err |= (get<i>(vec1) != static_cast<DataT>(static_cast<DataT>(~0x0F)));
            err |= (get<i>(vec0) == (get<i>(vec1)));
        });

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              typename std::enable_if_t<!std::is_integral<DataT>{}>* = nullptr>
    __host__ __device__ static inline bool operatorBitInv()
    {
        // Non-integral
        bool err = false;
        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorBoolEqS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(3.0f);
        auto res1 = (vec0 == static_cast<DataT>(3.0f));
        auto res2 = (static_cast<DataT>(11.0f) == vec0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(3.0f)));
        });

        err |= (res1 != true);
        err |= (res2 != false);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorBoolEqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(3.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = make_vector<DataT, VecSize>(5.0f);
        auto res1 = (vec0 == vec1);
        auto res2 = (vec1 == vec2);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2) != (static_cast<DataT>(5.0f)));
        });

        err |= (res1 != true);
        err |= (res2 != false);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorBoolNeqS()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(3.0f);
        auto res1 = (vec0 != static_cast<DataT>(3.0f));
        auto res2 = (static_cast<DataT>(11.0f) != vec0);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(3.0f)));
        });

        err |= (res1 != false);
        err |= (res2 != true);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __host__ __device__ static inline bool operatorBoolNeqV()
    {
        bool err = false;

        auto vec0 = make_vector<DataT, VecSize>(3.0f);
        auto vec1 = make_vector<DataT, VecSize>(3.0f);
        auto vec2 = make_vector<DataT, VecSize>(5.0f);
        auto res1 = (vec0 != vec1);
        auto res2 = (vec1 != vec2);

        static_for<0, VecSize, 1>([&](auto&& Idx) {
            constexpr uint32_t i = decay_t<decltype(Idx)>::value;
            err |= (get<i>(vec0) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec1) != (static_cast<DataT>(3.0f)));
            err |= (get<i>(vec2) != (static_cast<DataT>(5.0f)));
        });

        err |= (res1 != false);
        err |= (res2 != true);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __global__ void vectorTest(uint32_t     m,
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

        err = err ? err : bcastCtorTest<DataT, VecSize>();

        err = err ? err : copyCtorTest<DataT, VecSize>();

        err = err ? err : moveCtorTest<DataT, VecSize>();

        err = err ? err : assignmentTest<DataT, VecSize>();

        err = err ? err : assignmentMoveTest<DataT, VecSize>();

        err = err ? err : operatorPlusEqV<DataT, VecSize>();

        err = err ? err : operatorPlusS<DataT, VecSize>();

        err = err ? err : operatorPlusV<DataT, VecSize>();

        err = err ? err : operatorMinusEqV<DataT, VecSize>();

        err = err ? err : operatorMinusS<DataT, VecSize>();

        err = err ? err : operatorMinusV<DataT, VecSize>();

        err = err ? err : operatorMultEqV<DataT, VecSize>();

        err = err ? err : operatorMultS<DataT, VecSize>();

        err = err ? err : operatorMultV<DataT, VecSize>();

        err = err ? err : operatorDivEqV<DataT, VecSize>();

        err = err ? err : operatorDivS<DataT, VecSize>();

        err = err ? err : operatorDivV<DataT, VecSize>();

        err = err ? err : operatorModEqV<DataT, VecSize>();

        err = err ? err : operatorModS<DataT, VecSize>();

        err = err ? err : operatorModV<DataT, VecSize>();

        err = err ? err : operatorBitAndEqV<DataT, VecSize>();

        err = err ? err : operatorBitAndS<DataT, VecSize>();

        err = err ? err : operatorBitAndV<DataT, VecSize>();

        err = err ? err : operatorBitOrEqV<DataT, VecSize>();

        err = err ? err : operatorBitOrS<DataT, VecSize>();

        err = err ? err : operatorBitOrV<DataT, VecSize>();

        err = err ? err : operatorBitXorEqV<DataT, VecSize>();

        err = err ? err : operatorBitXorS<DataT, VecSize>();

        err = err ? err : operatorBitXorV<DataT, VecSize>();

        err = err ? err : operatorBitShrEqV<DataT, VecSize>();

        err = err ? err : operatorBitShrS<DataT, VecSize>();

        err = err ? err : operatorBitShrV<DataT, VecSize>();

        err = err ? err : operatorBitShlEqV<DataT, VecSize>();

        err = err ? err : operatorBitShlS<DataT, VecSize>();

        err = err ? err : operatorBitShlV<DataT, VecSize>();

        err = err ? err : operatorBitInv<DataT, VecSize>();

        err = err ? err : operatorBoolEqS<DataT, VecSize>();

        err = err ? err : operatorBoolEqV<DataT, VecSize>();

        err = err ? err : operatorBoolNeqS<DataT, VecSize>();

        err = err ? err : operatorBoolNeqV<DataT, VecSize>();

        err = err ? err : operatorInc<DataT, VecSize>();

        err = err ? err : operatorDec<DataT, VecSize>();

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

    template <typename DataT, uint32_t VecSize>
    auto inline vectorHostTest()
    {
        bool err = true;

        bcastCtorTest<DataT, VecSize>();

        copyCtorTest<DataT, VecSize>();

        moveCtorTest<DataT, VecSize>();

        assignmentTest<DataT, VecSize>();

        assignmentMoveTest<DataT, VecSize>();

        operatorPlusEqV<DataT, VecSize>();

        operatorPlusS<DataT, VecSize>();

        operatorPlusV<DataT, VecSize>();

        operatorMinusEqV<DataT, VecSize>();

        operatorMinusS<DataT, VecSize>();

        operatorMinusV<DataT, VecSize>();

        operatorMultEqV<DataT, VecSize>();

        operatorMultS<DataT, VecSize>();

        operatorMultV<DataT, VecSize>();

        operatorDivEqV<DataT, VecSize>();

        operatorDivS<DataT, VecSize>();

        operatorDivV<DataT, VecSize>();

        operatorModEqV<DataT, VecSize>();

        operatorModS<DataT, VecSize>();

        operatorModV<DataT, VecSize>();

        operatorBitAndEqV<DataT, VecSize>();

        operatorBitAndS<DataT, VecSize>();

        operatorBitAndV<DataT, VecSize>();

        operatorBitOrEqV<DataT, VecSize>();

        operatorBitOrS<DataT, VecSize>();

        operatorBitOrV<DataT, VecSize>();

        operatorBitXorEqV<DataT, VecSize>();

        operatorBitXorS<DataT, VecSize>();

        operatorBitXorV<DataT, VecSize>();

        operatorBitShrEqV<DataT, VecSize>();

        operatorBitShrS<DataT, VecSize>();

        operatorBitShrV<DataT, VecSize>();

        operatorBitShlEqV<DataT, VecSize>();

        operatorBitShlS<DataT, VecSize>();

        operatorBitShlV<DataT, VecSize>();

        operatorBitInv<DataT, VecSize>();

        operatorBoolEqS<DataT, VecSize>();

        operatorBoolEqV<DataT, VecSize>();

        operatorBoolNeqS<DataT, VecSize>();

        operatorBoolNeqV<DataT, VecSize>();

        operatorInc<DataT, VecSize>();

        operatorDec<DataT, VecSize>();

        return err;
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_VECTOR_TEST_HPP
