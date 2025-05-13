
#pragma once

#include "Array.hpp"

namespace rocRoller
{
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N> const& array)
    {
        for(auto iter = array.begin(); iter != array.end(); iter++)
        {
            if(!*iter)
                return iter;
        }
        return array.end();
    }

    template <typename T, size_t N>
    auto emptySlot(std::array<T, N>& array)
    {
        for(auto iter = array.begin(); iter != array.end(); iter++)
        {
            if(!*iter)
                return iter;
        }
        return array.end();
    }

    template <typename T, size_t N>
    void append(std::array<T, N>& array, T value)
    {
        auto iter = emptySlot(array);
        AssertFatal(iter != array.end(), "Array is full!");
        *iter = std::move(value);
    }

    template <typename T, size_t N, typename C>
    requires CInputRangeOf<C, T>
    void append(std::array<T, N>& array, C values)
    {
        auto arrayIter = emptySlot(array);

        if constexpr(std::ranges::sized_range<C>)
        {
            auto remainingSlots = array.end() - arrayIter;
            auto numValues      = std::ranges::size(values);
            AssertFatal(remainingSlots >= numValues,
                        "Array is full!",
                        ShowValue(remainingSlots),
                        ShowValue(numValues));
        }

        auto valueIter = values.begin();
        for(; valueIter != values.end() && arrayIter != array.end(); ++arrayIter, ++valueIter)
        {
            AssertFatal(!*arrayIter, "Array values not contiguous!");
            *arrayIter = std::move(*valueIter);
        }

        AssertFatal(valueIter == values.end(), "Array is full!");
    }

}
