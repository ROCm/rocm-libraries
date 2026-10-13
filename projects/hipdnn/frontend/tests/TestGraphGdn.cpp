// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphGdn, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({8, 2, 8}).set_stride({16, 8, 1}).set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({8, 2, 8}).set_stride({16, 8, 1}).set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim({8, 2, 8}).set_stride({16, 8, 1}).set_data_type(DataType::FLOAT);

    auto g = std::make_shared<TensorAttributes>();
    g->set_dim({8, 2}).set_stride({2, 1}).set_data_type(DataType::FLOAT);

    auto beta = std::make_shared<TensorAttributes>();
    beta->set_dim({8, 2}).set_stride({2, 1}).set_data_type(DataType::FLOAT);

    auto cuSeqlens = std::make_shared<TensorAttributes>();
    cuSeqlens->set_dim({3}).set_stride({1}).set_data_type(DataType::INT32);

    // Create attributes
    GdnAttributes attributes;
    attributes.set_name("GdnNode");
    attributes.set_scale(0.35355339F);
    attributes.set_output_final_state(true);
    attributes.set_use_qk_l2norm(true);
    attributes.set_checkpoint_every_n_tokens(4);
    attributes.set_use_beta_sigmoid(true);
    attributes.set_allow_neg_eigval(true);
    attributes.set_safe_gate(true);
    attributes.set_gate_domain(GateDomain::LOG);
    attributes.set_batch_invariant(true);
    attributes.set_overwrite_initial_state(false);

    // Call graph method
    // TODO: Handle graph_return_type 'array'
    graph.gdn_ext(q, k, v, g, beta, cuSeqlens, attributes);

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
