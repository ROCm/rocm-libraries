#pragma once

#include <array>
#include <numeric>

struct tensor_elem_gen_checkboard_sign
{
    template <class... Ts>
    double operator()(Ts... Xs) const
    {
        std::array<uint64_t, sizeof...(Ts)> dims = {{Xs...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](int init, uint64_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};
