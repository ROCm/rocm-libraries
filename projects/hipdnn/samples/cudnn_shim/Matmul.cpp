// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-and-run smoke of the cuDNN-shaped matmul surface a hipified consumer
// exercises. Mirrors cuDNN FE's samples/cpp/matmul/matmuls.cpp: build a batched
// matmul graph through the shim using only cuDNN-spelled types, then validate.
// Host-only (no device), so it runs as a CTest.

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
        const int64_t m = 64;
        const int64_t n = 32;
        const int64_t k = 128;

        cudnn_frontend::graph::Graph graph;
        graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        auto a = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                  .set_name("A")
                                  .set_dim({b, m, k})
                                  .set_stride({m * k, k, 1})
                                  .set_uid(1));
        auto bMat = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                                     .set_name("B")
                                     .set_dim({b, k, n})
                                     .set_stride({k * n, n, 1})
                                     .set_uid(2));

        auto c
            = graph.matmul(a, bMat, cudnn_frontend::graph::Matmul_attributes{}.set_name("matmul"));
        c->set_output(true).set_uid(3);

        if(auto error = graph.validate(); error.is_bad())
        {
            return fail("validate", error);
        }

        std::cout << "cuDNN-shim matmul graph validated\n";
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
