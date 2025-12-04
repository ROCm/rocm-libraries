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
                {4, 4, 4, 161, 700}};
    }

    pooling3d_driver() : pooling_driver<T>()
    {
        this->add(
            this->in_shape, "input", this->generate_data_limited(get_3d_pooling_input_shapes(), 4));
        this->add(this->lens, "lens", this->generate_data({{2, 2, 2}, {3, 3, 3}}));
        this->add(this->strides, "strides", this->generate_data({{2, 2, 2}, {1, 1, 1}}));
        this->add(this->pads, "pads", this->generate_data({{0, 0, 0}, {1, 1, 1}}));
        this->add(this->wsidx, "wsidx", this->generate_data({1}));
    }
};

int main(int argc, const char* argv[]) { test_drive<pooling3d_driver<bfloat16>>(argc, argv); }
