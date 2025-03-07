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

#include <type_traits>

namespace rocRoller
{
    /**
 * \ingroup Utilities
 * \defgroup Comparison Comparison
 */

    /**
 * \addtogroup Comparison
 * @{
 */

    /**
 * Lexicographically compares two lists of values, one pair of values at a
 * time.
 *
 * \return -1: The first list is lesser.
 *          1: The second list is lesser.
 *          0: The lists are equal.
 */
    inline int LexicographicCompare()
    {
        return 0;
    }

    template <typename A>
    inline int LexicographicCompare(A const& lhs, A const& rhs)
    {
        if(lhs < rhs)
            return -1;
        if(lhs > rhs)
            return 1;
        return 0;
    }

    template <typename A,
              typename... Args,
              typename = typename std::enable_if<sizeof...(Args) % 2 == 0>>
    inline int LexicographicCompare(A const& lhs, A const& rhs, Args const&... rest)
    {
        if(lhs < rhs)
            return -1;
        if(lhs > rhs)
            return 1;
        return LexicographicCompare(rest...);
    }

    /**
 * @brief Traits class which enables comparison operators to be implemented
 * based on a `compare()` function.
 *
 * Specializations must implement a `compare(lhs, rhs)` function which returns:
 *  - -1: lhs is lesser
 *  -  0: values are equal
 *  -  1: rhs is lesser
 */
    template <typename T, typename U = T>
    struct Comparison
    {
        enum
        {
            implemented = false
        };
    };

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator==(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) == 0;
    }

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator!=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) != 0;
    }

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator<(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) < 0;
    }

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator<=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) <= 0;
    }

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator>(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) > 0;
    }

    template <typename T,
              typename U,
              typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator>=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) >= 0;
    }

    /**
     * Combines a number of already-hashed values.
     */
    template <typename... Ts>
    inline size_t combine_hashes(size_t a, Ts... rest)
    {
        return combine_hashes(a, combine_hashes(rest...));
    }

    template <>
    inline size_t combine_hashes(size_t a, size_t b)
    {
        return b ^ (a + 0x9b9773e99e3779b9 + (b << 6) + (b >> 2));
    }

    template <typename T>
    inline size_t hash_combine(T const& val)
    {
        return std::hash<T>()(val);
    }

    template <typename T, typename... Ts>
    inline size_t hash_combine(T const& val, Ts const&... more)
    {
        //size_t mine = sizeof(val);
        size_t mine = std::hash<T>()(val);
        size_t rest = hash_combine(more...);

        //return mine + rest;
        return combine_hashes(mine, rest);
    }

    template <typename Iter>
    inline size_t hash_combine_iter(Iter begin, Iter end)
    {
        size_t rv = 0;
        while(begin != end)
        {
            rv = combine_hashes(hash_combine(*begin), rv);
            begin++;
        }

        return rv;
    }

    template <size_t N, class... Types>
    struct tuple_hash
    {
        using MyTuple = std::tuple<Types...>;
        using TypeN   = typename std::tuple_element<N, MyTuple>::type;
        static size_t apply(std::tuple<Types...> const& tup)
        {
            size_t mine = std::hash<TypeN>()(std::get<N>(tup));
            size_t rest = tuple_hash<N - 1, Types...>::apply(tup);

            return combine_hashes(mine, rest);
        }
    };

    template <class... Types>
    struct tuple_hash<0, Types...>
    {
        using MyTuple = std::tuple<Types...>;
        using Type0   = typename std::tuple_element<0, MyTuple>::type;
        static size_t apply(std::tuple<Types...> const& tup)
        {
            return std::hash<Type0>()(std::get<0>(tup));
        }
    };

    template <class... Types>
    size_t hash_tuple(std::tuple<Types...> const& tup)
    {
        return tuple_hash<sizeof...(Types) - 1, Types...>::apply(tup);
    }

    /**
 * @}
 */
} // namespace rocRoller

namespace std
{
    template <class... Types>
    struct hash<tuple<Types...>>
    {
        inline size_t operator()(tuple<Types...> const& tup) const
        {
            return rocRoller::hash_tuple(tup);
        }
    };
} // namespace std
