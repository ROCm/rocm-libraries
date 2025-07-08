/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <concepts>
#include <deque>
#include <ranges>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace rocRoller
{
    template <typename T, typename... U>
    concept CIsAnyOf = (std::same_as<T, U> || ...);

    static_assert(CIsAnyOf<int, int, float>);
    static_assert(CIsAnyOf<std::string, int, std::string>);
    static_assert(!CIsAnyOf<std::string, int, float>);

    // clang-format off
    template <typename T>
    concept CHasToStringMember = requires(T const& x)
    {
        !std::convertible_to<std::string, T>;

        { x.toString() } -> std::convertible_to<std::string>;
    };

    template <typename T>
    concept CHasToString = requires(T const& x)
    {
        !std::convertible_to<std::string, T>;

        { toString(x) } -> std::convertible_to<std::string>;
    };

    /**
     * Matches enumerations that are scoped, that have a Count member, and that
     * can be converted to string with toString().
     */
    template <typename T>
    concept CCountedEnum = requires()
    {
        requires std::regular<T>;
        requires CHasToString<T>;

        { T::Count } -> std::convertible_to<T>;

        {
            static_cast<std::underlying_type_t<T>>(T::Count)
        } -> std::convertible_to<std::underlying_type_t<T>>;


    };

    template <typename Range, typename Of>
    concept CForwardRangeOf = requires()
    {
        requires std::ranges::forward_range<Range>;
        requires std::convertible_to<std::ranges::range_value_t<Range>, Of>;
    };

    template <typename Range, typename Of>
    concept CInputRangeOf = requires()
    {
        requires std::ranges::input_range<std::remove_reference_t<Range>>;
        requires std::convertible_to<std::ranges::range_value_t<std::remove_reference_t<Range>>, Of>;
    };

    template <typename T>
    concept CHasName = requires(T const& obj)
    {
        { name(obj) } -> std::convertible_to<std::string>;
    };

    // clang-format on

    static_assert(CForwardRangeOf<std::vector<int>, int>);
    static_assert(CForwardRangeOf<std::vector<short>, int>);
    static_assert(CForwardRangeOf<std::vector<float>, int>);
    static_assert(CForwardRangeOf<std::deque<int>, int>);
    static_assert(CForwardRangeOf<std::set<int>, int>);
    static_assert(!CForwardRangeOf<std::set<std::string>, int>);
    static_assert(!CForwardRangeOf<int, int>);

    template <typename T>
    concept CPointer = requires()
    {
        requires std::is_pointer_v<T>;
    };

    static_assert(!CPointer<int>);
    static_assert(CPointer<int*>);
    static_assert(CPointer<int const*>);
    static_assert(CPointer<char const*>);
    static_assert(CPointer<char*>);
}
