/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>

namespace TensileLite
{

    // Type wrapper that can be copied or assigned to in a threadsafe manner. Value cannot be modified
    // Intended for using value semantics with non-trivially copyable data
    template <typename T>
    class ThreadSafeValue
    {
    private:
        mutable std::mutex m_access;
        T                  m_value;

    public:
        ThreadSafeValue() {}

        ThreadSafeValue(const ThreadSafeValue<T>& other)
        {
            std::lock_guard<std::mutex> lock(other.m_access);
            m_value = other.m_value;
        }

        ThreadSafeValue(const T& other)
            : m_value(other)
        {
        }

        ThreadSafeValue<T>& operator=(const ThreadSafeValue<T>& other)
        {
            if(this != &other)
            {
                std::lock_guard<std::mutex> otherLock(other.m_access);
                std::lock_guard<std::mutex> selfLock(m_access);
                m_value = other.m_value;
            }

            return *this;
        }

        ThreadSafeValue<T>& operator=(const T& other)
        {
            std::lock_guard<std::mutex> lock(m_access);
            m_value = other;

            return *this;
        }

        T load() const
        {
            std::lock_guard<std::mutex> lock(m_access);
            return m_value;
        }

        T operator*() const
        {
            return load();
        }

        ~ThreadSafeValue() = default;
    };

    /**
 * \ingroup Tensile
 * \addtogroup Utilities
 * @{
 */

    template <typename T>
    T CeilDivide(T num, T den)
    {
        return (num + (den - 1)) / den;
    }

    template <typename T>
    T RoundUpToMultiple(T val, T den)
    {
        return CeilDivide(val, den) * den;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
    T IsPrime(T val)
    {
        if(val < 2)
            return false;
        if(val < 4)
            return true;

        T end = sqrt(val);

        for(T i = 2; i <= end; i++)
            if(val % i == 0)
                return false;
        return true;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
    T NextPrime(T val)
    {
        if(val < 2)
            return 2;
        while(!IsPrime(val))
            val++;
        return val;
    }

    template <typename Container, typename Joiner>
    void streamJoin(std::ostream& stream, Container const& c, Joiner const& j)
    {
        bool first = true;
        for(auto const& item : c)
        {
            if(!first)
                stream << j;
            stream << item;
            first = false;
        }
    }

    template <typename T, size_t N>
    inline std::ostream& operator<<(std::ostream& stream, std::array<T, N> const& array)
    {
        streamJoin(stream, array, ", ");
        return stream;
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& stream, std::vector<T> const& array)
    {
        streamJoin(stream, array, ", ");
        return stream;
    }

    template <typename T>
    inline std::ostream& stream_write(std::ostream& stream, T&& val)
    {
        return stream << std::forward<T>(val);
    }

    template <typename T, typename... Ts>
    inline std::ostream& stream_write(std::ostream& stream, T&& val, Ts&&... vals)
    {
        return stream_write(stream << std::forward<T>(val), std::forward<Ts>(vals)...);
    }

    template <typename... Ts>
    inline std::string concatenate(Ts&&... vals)
    {
        std::ostringstream msg;
        stream_write(msg, std::forward<Ts>(vals)...);

        return msg.str();
    }

    template <bool T_Enable, typename... Ts>
    inline std::string concatenate_if(Ts&&... vals)
    {
        if(!T_Enable)
            return "";

        return concatenate(std::forward<Ts>(vals)...);
    }

    class StreamRead
    {
    public:
        StreamRead(std::string const& value, bool except = true);
        ~StreamRead();

        bool read(std::istream& stream);

    private:
        std::string const& m_value;
        bool               m_except;
        bool               m_success = false;
    };

    // inline std::istream & operator>>(std::istream & stream, StreamRead & value);
    inline std::istream& operator>>(std::istream& stream, StreamRead& value)
    {
        value.read(stream);
        return stream;
    }

    struct BitFieldGenerator
    {
        constexpr static uint32_t maxBitFieldWidth = 64;

        // Get the minimum width of the given maxVal in bits.
        constexpr static uint64_t ElementWidth(uint64_t maxVal)
        {
            return maxVal ? 1 + ElementWidth(maxVal >> 1) : 0;
        }

        // Get the bit mask for the element size in bits.
        constexpr static uint64_t BitMask(uint64_t elementWidth)
        {
            if(elementWidth == 1)
                return (uint64_t)0x1;
            return (BitMask(elementWidth - 1) << 1) | (uint64_t)0x1;
        }

        // Generate a 64 bit field containing val0 in the LSB, occupying the first
        // elementWidth bits.
        constexpr static uint32_t GenerateBitField(uint64_t elementWidth, uint64_t val0)
        {
            uint64_t mask = BitMask(elementWidth);
            return mask & val0;
        }

        // Generate a 64 bit field containing val0... valN in order starting from LSB, each
        // value occupying elementWidth bits of the field.
        template <typename... ArgsT>
        constexpr static uint64_t
            GenerateBitField(uint64_t elementWidth, uint64_t val0, ArgsT... valN)
        {
            uint64_t mask = BitMask(elementWidth);
            return (GenerateBitField(elementWidth, valN...) << elementWidth) | (mask & val0);
        }
    };

    std::vector<std::string> generateArgNameList(size_t length, const char* name = "");

    size_t greekToIndex(std::string name);
    /**
 * @}
 */
} // namespace TensileLite

/**
 * \addtogroup Utilities
 * @{
 */
#define TENSILE_STR_(x) #x
#define TENSILE_STR(x) TENSILE_STR_(x)
#define TENSILE_LINENO TENSILE_STR(__LINE__)
#define TENSILE_LINEINFO __FILE__ ":" TENSILE_LINENO

#define TENSILE_ASSERT_EXC(exp)                                               \
    do                                                                        \
    {                                                                         \
        if(!(exp))                                                            \
        {                                                                     \
            throw std::runtime_error("Error in " TENSILE_LINEINFO ": " #exp); \
        }                                                                     \
    } while(false)

/**
 * @}
 */
