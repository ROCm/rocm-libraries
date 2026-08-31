// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-and-run smoke of the cuDNN-shaped layernorm surface a hipified
// consumer exercises. Mirrors cuDNN FE's samples/cpp/norm/layernorm.cpp: build a
// layernorm-forward graph through the shim using only cuDNN-spelled types, then
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

        const int64_t b = 4;
        const int64_t s = 1024;
        const int64_t d = 128;

        cudnn_frontend::graph::Graph graph;
        graph.set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
            .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        auto x = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_name("X")
                                  .set_dim({b * s, d, 1, 1})
                                  .set_stride({d, 1, d, d})
                                  .set_uid(1));
        auto scale = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                      .set_name("scale")
                                      .set_dim({1, d, 1, 1})
                                      .set_stride({d, 1, d, d})
                                      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                                      .set_uid(2));
        auto bias = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                     .set_name("bias")
                                     .set_dim({1, d, 1, 1})
                                     .set_stride({d, 1, d, d})
                                     .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                                     .set_uid(3));
        auto epsilon = graph.tensor(1e-05F, cudnn_frontend::graph::ScalarType::COMPILE_TIME_CONST);

        auto [y, mean, invVariance]
            = graph.layernorm(x,
                              scale,
                              bias,
                              cudnn_frontend::graph::Layernorm_attributes{}
                                  .set_forward_phase(cudnn_frontend::NormFwdPhase_t::TRAINING)
                                  .set_epsilon(epsilon));
        y->set_output(true).set_uid(4);
        mean->set_output(true).set_data_type(cudnn_frontend::DataType_t::FLOAT).set_uid(5);
        invVariance->set_output(true).set_data_type(cudnn_frontend::DataType_t::FLOAT).set_uid(6);

        if(auto error = graph.validate(); error.is_bad())
        {
            return fail("validate", error);
        }

        std::cout << "cuDNN-shim layernorm graph validated\n";
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
