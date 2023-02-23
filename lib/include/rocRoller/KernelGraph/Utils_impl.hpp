
#pragma once

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{

    template <typename T>
    std::optional<T> only(std::vector<T> v)
    {
        if(v.size() == 1)
            return v[0];
        return {};
    }

    std::optional<int> only(Generator<int> g)
    {
        auto it = g.begin();

        if(it == g.end())
            return {};

        auto first = *it;

        it = std::next(it);
        if(it == g.end())
            return first;

        return {};
    }

}
