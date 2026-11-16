#pragma once

// A value of type T or an error of type E; the part of C++23's std::expected that hipconv uses.

#include <utility>
#include <variant>

namespace hipconv
{

template <typename T, typename E>
struct Expected : std::variant<T, E>
{
    constexpr Expected(T&& t) : std::variant<T, E>{std::move(t)} {}
    constexpr Expected(E&& e) : std::variant<T, E>{std::move(e)} {}

    constexpr auto has_value() const -> bool { return std::holds_alternative<T>(*this); }
    constexpr auto& operator*() const { return std::get<T>(*this); }
    constexpr auto* operator->() const { return &std::get<T>(*this); }
    constexpr auto& value() const { return std::get<T>(*this); }
    constexpr auto& error() const { return std::get<E>(*this); }
};

} // namespace hipconv
