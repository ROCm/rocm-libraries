// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-and-run smoke of the cuDNN-shaped SDPA surface a hipified consumer
// (e.g. PyTorch's aten/src/ATen/native/cudnn/MHA.cpp) exercises: build an SDPA
// forward graph through the shim using only cuDNN-spelled setters, then validate.
// Host-only (no device), so it runs as a CTest.

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>

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

        cudnn_frontend::graph::Graph graph;
        graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT);

        auto q = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_dim({2, 8, 16, 64})
                                  .set_stride({8192, 1024, 64, 1})
                                  .set_data_type(cudnn_frontend::DataType_t::HALF)
                                  .set_uid(1));
        auto k = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_dim({2, 8, 32, 64})
                                  .set_stride({16384, 2048, 64, 1})
                                  .set_data_type(cudnn_frontend::DataType_t::HALF)
                                  .set_uid(2));
        auto v = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_dim({2, 8, 32, 64})
                                  .set_stride({16384, 2048, 64, 1})
                                  .set_data_type(cudnn_frontend::DataType_t::HALF)
                                  .set_uid(3));

        auto [o, stats] = graph.sdpa(q,
                                     k,
                                     v,
                                     cudnn_frontend::graph::SDPA_attributes{}
                                         .set_name("mha")
                                         .set_attn_scale(0.125F)
                                         .set_causal_mask(true));
        static_cast<void>(stats);
        o->set_output(true).set_uid(4);

        if(auto error = graph.validate(); error.is_bad())
        {
            return fail("validate", error);
        }

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
