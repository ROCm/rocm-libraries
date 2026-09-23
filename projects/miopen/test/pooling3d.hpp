// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pooling_common.hpp"

template <class T>
struct pooling3d_driver : pooling_driver<T>
{
    std::vector<std::vector<int>> get_3d_pooling_input_shapes()
    {
        return {{16, 64, 3, 4, 4},
                {16, 32, 4, 9, 9},
                {8, 512, 3, 14, 14},
                {8, 512, 4, 28, 28},
                {16, 64, 56, 56, 56},
                {4, 3, 4, 227, 227},
                {4, 4, 4, 161, 700},
                {1, 3, 4, 4, 4},
                {2, 8, 2, 8, 8},
                {1, 16, 3, 5, 5},
                {1, 32, 4, 14, 14},
                {2, 64, 8, 8, 8},
                {1, 16, 4, 28, 28},
                {1, 3, 8, 56, 56},
                {2, 64, 4, 28, 28},
                {1, 32, 16, 32, 32}};
    }

    pooling3d_driver() : pooling_driver<T>()
    {
        this->add(
            this->in_shape, "input", this->generate_data_limited(get_3d_pooling_input_shapes(), 4));
        this->add(this->lens, "lens", this->generate_data({{2, 2, 2}, {3, 3, 3}, {1, 2, 2}}));
        this->add(this->strides, "strides", this->generate_data({{2, 2, 2}, {1, 1, 1}, {1, 2, 2}}));
        this->add(this->pads, "pads", this->generate_data({{0, 0, 0}, {1, 1, 1}}));
        this->add(this->wsidx, "wsidx", this->generate_data({1}));
    }
};
