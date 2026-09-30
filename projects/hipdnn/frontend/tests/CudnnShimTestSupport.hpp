// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared, host-only test scaffolding for the cuDNN-shaped graph wrapper tests.
// Deliberately does NOT pull samples/utils/Helpers.hpp (that drags in
// hip_runtime.h); the shim graph surface validates without a device.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace hipdnn_shim_test
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

inline std::shared_ptr<fe::graph::Tensor_attributes> makeTensor(fe::graph::Graph& graph,
                                                                const std::vector<int64_t>& dim,
                                                                const std::vector<int64_t>& stride,
                                                                int64_t uid)
{
    return graph.tensor(fe::graph::Tensor_attributes{}
                            .set_dim(dim)
                            .set_stride(stride)
                            .set_data_type(fe::DataType_t::FLOAT)
                            .set_uid(uid));
}

// Fills q/k/v (BHSD) with the canonical shapes used by the native SDPA node
// tests: batch 2, 8 heads, S_q 16, S_kv 32, head-dim 64.
inline void addForwardInputs(fe::graph::Graph& graph,
                             std::shared_ptr<fe::graph::Tensor_attributes>& q,
                             std::shared_ptr<fe::graph::Tensor_attributes>& k,
                             std::shared_ptr<fe::graph::Tensor_attributes>& v)
{
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT);
    q = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 1);
    k = makeTensor(graph, {2, 8, 32, 64}, {16384, 2048, 64, 1}, 2);
    v = makeTensor(graph, {2, 8, 32, 64}, {16384, 2048, 64, 1}, 3);
}

} // namespace hipdnn_shim_test
