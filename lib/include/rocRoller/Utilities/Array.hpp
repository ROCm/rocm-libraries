
#pragma once

#include <array>
#include <concepts>
#include <memory>
#include <ranges>

#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    /**
     * A collection of helper functions to deal with the sparsely populated,
     * statically allocated std::array of shared_ptr and optional values used
     * in rocRoller.
     */

    /**
     * Returns an iterator to the first empty slot in `array`, or `end` if the
     * array is full.
     */
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N> const& array);

    /**
     * Returns an iterator to the first empty slot in `array`, or `end` if the
     * array is full.
     */
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N>& array);

    /**
     * Places `value` in the first empty slot in `array`, or throws if the
     * array is full.
     */
    template <typename T, size_t N>
    void append(std::array<T, N>& array, T value);

    /**
     * Places `values` in the first empty slots in `array`, or throws if the
     * array is full.
     */
    template <typename T, size_t N, typename C>
    requires CInputRangeOf<C, T>
    void append(std::array<T, N>& array, C values);
}

#include "Array_impl.hpp"
