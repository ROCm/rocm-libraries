// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <iostream>
#include <vector>

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
    static_cast<void>(argc);
    static_cast<void>(argv);

    cudnn_frontend::graph::Graph graph;
    graph.tensor(cudnn_frontend::graph::Tensor_attributes{}
                     .set_dim({1})
                     .set_stride({1})
                     .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                     .set_uid(1)
                     .set_output(true));

    std::vector<uint8_t> data;
    if(auto error = graph.serialize(data); error.is_bad())
    {
        return fail("serialize", error);
    }

    cudnn_frontend::graph::Graph roundTripped;
    if(auto error = roundTripped.deserialize(data); error.is_bad())
    {
        return fail("deserialize", error);
    }

    if(auto error = roundTripped.validate(); error.is_bad())
    {
        return fail("validate", error);
    }

    return 0;
}
