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

}
