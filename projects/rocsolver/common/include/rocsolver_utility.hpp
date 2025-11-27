#pragma once

#include <string>
#include <string_view>

#ifdef USE_FMT_LIB
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#else
#include <format>
#include <ostream>
#include <iostream>
#include <cstdio>
#include <ranges>
#endif

// Aliased printing
#ifndef USE_FMT_LIB
template <typename Range>
struct joinable_range
{
    const Range& range;
    const std::string_view sep;
};

template <typename Range, typename CharT>
struct std::formatter<joinable_range<Range>, CharT>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        auto end = ctx.end();

        if(it != ctx.end() && *it != '}')
            throw std::format_error(
                "Invalid format args provided for rocsolver::formatting::join. The format string '{}' must be used. \n \
					The joinable_range type should be privately used by rocsolver::formatting::join.");

        return it;
    }

    template <typename FormatContext>
    auto format(const joinable_range<Range>& jr, FormatContext& ctx) const
    {
        auto it = std::begin(jr.range);
        auto end = std::end(jr.range);

        auto out = ctx.out();
        if(it != end)
        {
            out = std::format_to(out, "{}", *it);
            ++it;
        }

        for(; it != end; ++it)
        {
            out = std::format_to(out, "{}", jr.sep);
            out = std::format_to(out, "{}", *it);
        }

        return out;
    }
};

template <typename... T, typename CharT>
struct std::formatter<joinable_range<std::tuple<T...>>, CharT>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        auto end = ctx.end();

        if(it != ctx.end() && *it != '}')
            throw std::format_error(
                "Invalid format args provided for rocsolver::formatting::join. The format string '{}' must be used. \n \
					The joinable_range type should be privately used by rocsolver::formatting::join.");

        return it;
    }

    template <typename FormatContext>
    auto format(const joinable_range<std::tuple<T...>>& jr, FormatContext& ctx) const
    {
        std::string result = "";
        std::apply(
            [&](auto&&... args) {
                size_t i = 0;
                ((result += std::format("{}", args),
                  ++i != sizeof...(args) ? result += jr.sep : result += ""),
                 ...);
            },
            jr.range);
        result += "";
        return std::format_to(ctx.out(), "{}", result);
    }
};
#endif

namespace rocsolver
{
namespace formatting
{
#ifndef USE_FMT_LIB
    template <class... Args>
    void print(std::format_string<Args...> fmt_string, Args&&... args)
    {

        std::cout << std::format(fmt_string, std::forward<Args>(args)...);
    }

    template <class... Args>
    void print(FILE* stream, std::format_string<Args...> fmt_string, Args&&... args)
    {
        fprintf(stream, "%s", std::format(fmt_string, std::forward<Args>(args)...).c_str());
    }

    template <class... Args>
    void print(std::ostream& stream, std::format_string<Args...> fmt_string, Args&&... args)
    {
        stream << std::format(fmt_string, std::forward<Args>(args)...);
    }

    template <class... Args>
    std::string format(std::format_string<Args...> fmt_string, Args&&... args)
    {
        return std::format(fmt_string, std::forward<Args>(args)...);
    }
#else

    template <class T, class... Args>
    void print(T&& fmt_string, Args&&... args)
    {
        fmt::print(std::forward<T>(fmt_string), std::forward<Args>(args)...);
    }

    template <class T, class... Args>
    void print(FILE* stream, T&& fmt_string, Args&&... args)
    {
        fmt::print(stream, std::forward<T>(fmt_string), std::forward<Args>(args)...);
    }

    template <class T, class... Args>
    void print(std::ostream& stream, T&& fmt_string, Args&&... args)
    {
        fmt::print(stream, std::forward<T>(fmt_string), std::forward<Args>(args)...);
    }

    template <class T, class... Args>
    std::string format(T&& fmt_string, Args&&... args)
    {
        return fmt::format(std::forward<T>(fmt_string), std::forward<Args>(args)...);
    }
#endif
    template <typename Range>
    std::string join(Range&& r, std::string_view sep)
    {
#ifdef USE_FMT_LIB
        return fmt::format("{}", fmt::join(r, sep));
#else
        joinable_range range{r, sep};
        return std::format("{}", range);
#endif
    }
}
}
