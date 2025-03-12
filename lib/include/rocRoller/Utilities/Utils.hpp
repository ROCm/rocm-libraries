/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <rocRoller/Utilities/Concepts.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{

    /**
     * \ingroup rocRoller
     * \addtogroup Utilities
     * @{
     */

    template <typename T>
    constexpr T CeilDivide(T num, T den)
    {
        return (num + (den - 1)) / den;
    }

    template <typename T>
    constexpr T RoundUpToMultiple(T val, T den)
    {
        return CeilDivide(val, den) * den;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
    constexpr T IsPrime(T val)
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
    constexpr T NextPrime(T val)
    {
        if(val < 2)
            return 2;
        while(!IsPrime(val))
            val++;
        return val;
    }

    template <typename T>
    std::variant<T> singleVariant(T value)
    {
        return std::variant<T>(std::move(value));
    }

    /**
     * Returns a copy of `sets`, with any sets that have elements in common
     * merged together.
     *
     * Postconditions:
     * - Union of return value sets is the same as the union of the input sets.
     * - No two sets in return value have any common elements.
     * - mergeSets(mergeSets(x)) == mergeSets(x) for any x.
     */
    std::vector<std::set<int>> mergeSets(std::vector<std::set<int>> sets);

    inline std::string toString(int x)
    {
        return std::to_string(x);
    }

    template <CHasToString T>
    requires(!std::is_arithmetic_v<T>) inline std::ostream& operator<<(std::ostream& stream,
                                                                       T const&      x)
    {
        return stream << toString(x);
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

    template <CHasToString T>
    inline std::ostream& operator<<(std::ostream& stream, std::set<T> const& x)
    {
        stream << "set{";
        streamJoin(stream, x, ", ");
        stream << "}";

        return stream;
    }

    template <CHasToString T1, CHasToString T2>
    inline std::ostream& operator<<(std::ostream& stream, std::pair<T1, T2> const& x)
    {
        stream << "[" << x.first << ", " << x.second << "]";

        return stream;
    }

    template <CHasToString T>
    inline std::ostream& operator<<(std::ostream& stream, std::vector<T> const& xs)
    {
        auto iter = xs.begin();
        stream << "[";
        if(iter != xs.end())
        {
            stream << *iter;
        }
        iter++;

        for(; iter != xs.end(); iter++)
        {
            stream << ", " << *iter;
        }

        stream << "]";

        return stream;
    }

    template <int Idx = 0, typename... Ts>
    inline void
        streamJoinTuple(std::ostream& stream, std::string const& sep, std::tuple<Ts...> const& tup)
    {
        if constexpr(Idx < sizeof...(Ts))
        {
            stream << std::get<Idx>(tup);

            if constexpr((Idx + 1) < (sizeof...(Ts)))
            {
                stream << sep;
                streamJoinTuple<Idx + 1>(stream, sep, tup);
            }
        }
    }

    template <typename... Ts>
    inline std::string concatenate_join(std::string const& sep, Ts const&... vals)
    {
        std::ostringstream msg;
        msg.setf(std::ios::showpoint);
        streamJoinTuple(msg, sep, std::forward_as_tuple(vals...));

        return msg.str();
    }

    template <typename... Ts>
    inline std::string concatenate(Ts const&... vals)
    {
        std::ostringstream msg;
        msg.setf(std::ios::showpoint);
        ((msg << (vals)), ...);

        return msg.str();
    }

    template <>
    inline std::string concatenate<std::string>(std::string const& val)
    {
        return val;
    }

    template <bool T_Enable, typename... Ts>
    inline std::string concatenate_if(Ts const&... vals)
    {
        if constexpr(!T_Enable)
            return "";

        return concatenate(vals...);
    }

    class StreamRead
    {
    public:
        explicit StreamRead(std::string const& value, bool except = true);
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
        constexpr static uint32_t maxBitFieldWidth = 32;

        // Get the minimum width of the given maxVal in bits.
        constexpr static uint32_t ElementWidth(uint32_t maxVal)
        {
            return maxVal ? 1 + ElementWidth(maxVal >> 1) : 0;
        }

        // Get the bit mask for the element size in bits.
        constexpr static uint32_t BitMask(uint32_t elementWidth)
        {
            if(elementWidth == 1)
                return (uint32_t)0x1;
            return (BitMask(elementWidth - 1) << 1) | (uint32_t)0x1;
        }

        // Generate a 32 bit field containing val0 in the LSB, occupying the first
        // elementWidth bits.
        constexpr static uint32_t GenerateBitField(uint32_t elementWidth, uint32_t val0)
        {
            int mask = BitMask(elementWidth);
            return mask & val0;
        }

        // Generate a 32 bit field containing val0... valN in order starting from LSB, each
        // value occupying elementWidth bits of the field.
        template <typename... ArgsT>
        constexpr static uint32_t
            GenerateBitField(uint32_t elementWidth, uint32_t val0, ArgsT... valN)
        {
            int mask = BitMask(elementWidth);
            return (GenerateBitField(elementWidth, valN...) << elementWidth) | (mask & val0);
        }
    };

    template <std::integral T>
    Generator<T> iota(T begin, T end, T inc)
    {
        for(; begin < end; begin += inc)
            co_yield begin;
    }

    template <std::integral T>
    Generator<T> iota(T begin, T end)
    {
        co_yield iota<T>(begin, end, 1);
    }

    template <std::integral T>
    Generator<T> iota(T begin)
    {
        for(;; ++begin)
            co_yield begin;
    }

    inline constexpr auto Generated(auto gen)
    {
        return std::vector(gen.begin(), gen.end());
    }

    /**
     * Returns true if begin...end represents a contiguous, increasing range of integer values.
     */
    template <typename Iter, typename End>
    inline bool IsContiguousRange(Iter begin, End end)
    {
        if(begin == end)
            return true;

        auto value = *begin;
        begin++;

        for(; begin != end; begin++)
        {
            if(*begin != value + 1)
                return false;
            value = *begin;
        }

        return true;
    }

    /**
     * Returns the product of the elements of the input.
     */
    template <std::ranges::forward_range T>
    inline auto product(T const& x)
    {
        using Value = std::decay_t<decltype(*x.cbegin())>;
        Value init  = 1;
        return std::accumulate(x.cbegin(), x.cend(), init, std::multiplies<Value>());
    }

    template <typename T>
    constexpr inline bool contains(std::vector<T> x, T val)
    {
        return std::any_of(x.begin(), x.end(), [val](T elem) { return elem == val; });
    }

    // helper for visitor
    template <class... Ts>
    struct overloaded : Ts...
    {
        // cppcheck-suppress syntaxError
        using Ts::operator()...;
    };

    // explicit deduction
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    /**
     * Converts a string value to an enum by comparing against each toString conversion.
     */
    template <CCountedEnum T>
    T fromString(std::string const& str);

    template <CHasName T>
    requires(std::default_initializable<T>) std::string name()
    {
        T obj;
        return name(obj);
    }

    std::vector<char> readFile(std::string const&);

    std::string readMetaDataFromCodeObject(std::string const& fileName);

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    template <typename... Ts>
    inline std::ostream& operator<<(std::ostream& stream, std::tuple<Ts...> const& tup)
    {
        stream << "[";
        rocRoller::streamJoinTuple(stream, ", ", tup);
        return stream << "]";
    }

    template <typename T, size_t N>
    inline std::ostream& operator<<(std::ostream& stream, std::array<T, N> const& array)
    {
        rocRoller::streamJoin(stream, array, ", ");
        return stream;
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& stream, std::set<T> const& array)
    {
        stream << "{";
        rocRoller::streamJoin(stream, array, ", ");
        stream << "}";
        return stream;
    }
}

#include <rocRoller/Utilities/Utils_impl.hpp>
