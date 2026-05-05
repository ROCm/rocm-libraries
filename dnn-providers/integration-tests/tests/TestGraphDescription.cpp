// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <string>

#include "harness/GraphDescription.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestGraphDescription, SingleConvFprop)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 16, 16, 16}).set_stride({4096, 256, 16, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({1, 16, 3, 3}).set_stride({144, 9, 3, 1}).set_data_type(DataType::FLOAT);

    graph.conv_fprop(x, w, ConvFpropAttributes());

    auto desc = hipdnn_integration_tests::describeGraph(graph);
    EXPECT_NE(desc.find("ConvFprop"), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("io="), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("compute="), std::string::npos) << "desc=" << desc;
}

TEST(TestGraphDescription, ConvPlusPointwiseRelu)
{
    Graph graph;
    graph.set_io_data_type(DataType::HALF).set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 16, 16, 16}).set_stride({4096, 256, 16, 1}).set_data_type(DataType::HALF);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({1, 16, 3, 3}).set_stride({144, 9, 3, 1}).set_data_type(DataType::HALF);

    auto y = graph.conv_fprop(x, w, ConvFpropAttributes());

    PointwiseAttributes pw;
    pw.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(y, pw);

    auto desc = hipdnn_integration_tests::describeGraph(graph);
    EXPECT_NE(desc.find("ConvFprop"), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("Pointwise:RELU_FWD"), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("+"), std::string::npos) << "desc=" << desc;
}

TEST(TestGraphDescription, DataTypeAnnotations)
{
    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 16, 16, 16}).set_stride({4096, 256, 16, 1}).set_data_type(DataType::HALF);

    PointwiseAttributes pw;
    pw.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pw);

    auto desc = hipdnn_integration_tests::describeGraph(graph);
    EXPECT_NE(desc.find("io=fp16"), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("compute=fp32"), std::string::npos) << "desc=" << desc;
    EXPECT_NE(desc.find("intermediate=fp16"), std::string::npos) << "desc=" << desc;
}

// NOLINTEND(readability-identifier-naming)
