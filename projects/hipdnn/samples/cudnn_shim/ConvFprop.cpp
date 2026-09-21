// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-and-run smoke of the cuDNN-shaped convolution-fprop surface a hipified
// consumer exercises. Mirrors cuDNN FE's samples/cpp/convolution/fprop.cpp:
// build a conv-fprop graph through the shim using only cuDNN-spelled types, then
// validate. Host-only (no device), so it runs as a CTest.

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <exception>
#include <iostream>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

namespace
{
int fail(const char* step, const cudnn_frontend::error_t& error)
{
    std::cerr << step << " failed: " << error.get_message() << '\n';
    return 1;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        static_cast<void>(argc);
        static_cast<void>(argv);

        const int64_t n = 16;
        const int64_t c = 128;
        const int64_t h = 64;
        const int64_t w = 64;
        const int64_t k = 256;
        const int64_t r = 1;
        const int64_t s = 1;

        cudnn_frontend::graph::Graph graph;
        graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        auto x = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_name("image")
                                  .set_dim({n, c, h, w})
                                  .set_stride({c * h * w, 1, c * w, c})
                                  .set_uid(1));
        auto weight = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                       .set_name("filter")
                                       .set_dim({k, c, r, s})
                                       .set_stride({c * r * s, 1, c * s, c})
                                       .set_uid(2));

        auto y = graph.conv_fprop(x,
                                  weight,
                                  cudnn_frontend::graph::Conv_fprop_attributes{}
                                      .set_padding({0, 0})
                                      .set_stride({1, 1})
                                      .set_dilation({1, 1}));
        y->set_output(true).set_uid(3);

        if(auto error = graph.validate(); error.is_bad())
        {
            return fail("validate", error);
        }

        std::cout << "cuDNN-shim conv_fprop graph validated\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << '\n';
        return 1;
    }
    catch(...)
    {
        std::cerr << "Unhandled unknown exception\n";
        return 1;
    }
}
