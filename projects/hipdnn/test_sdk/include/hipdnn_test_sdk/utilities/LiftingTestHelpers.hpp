// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

namespace hipdnn_tests
{

/// Validates a graph, lowers via build_operation_graph(handle), retrieves the
/// raw backend descriptor, and lifts into a new TestableGraphLifting via
/// fromBackendDescriptor(). Returns nullptr on failure; internal EXPECT
/// macros report the specific step that failed.
inline std::shared_ptr<TestableGraphLifting> liftGraph(TestableGraphLifting& graph,
                                                       hipdnnHandle_t handle)
{
    using hipdnn_frontend::ErrorCode;

    auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return nullptr;
    }

    result = graph.build_operation_graph(handle);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return nullptr;
    }

    auto rawDesc = graph.get_raw_graph_descriptor();
    EXPECT_NE(rawDesc, nullptr); // NOLINT(readability-implicit-bool-conversion)
    if(rawDesc == nullptr)
    {
        return nullptr;
    }

    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return nullptr;
    }

    return liftedGraph;
}

/// Validates a graph, serializes to binary, creates a backend descriptor from
/// the raw bytes (no handle, no finalize), and lifts into a new
/// TestableGraphLifting via fromBackendDescriptor(). Returns nullptr on failure.
inline std::shared_ptr<TestableGraphLifting>
    liftGraphWithoutFinalization(TestableGraphLifting& graph)
{
    using hipdnn_frontend::ErrorCode;

    auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return nullptr;
    }

    auto [data, serErr] = graph.to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    if(!serErr.is_good())
    {
        return nullptr;
    }
    EXPECT_FALSE(data.empty());
    if(data.empty())
    {
        return nullptr;
    }

    const hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(),
                                                                           data.size());
    EXPECT_TRUE(graphDesc.valid()) // NOLINT(readability-implicit-bool-conversion)
        << "Failed to create backend graph descriptor"; // NOLINT(readability-implicit-bool-conversion)
    if(!graphDesc.valid())
    {
        return nullptr;
    }

    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return nullptr;
    }

    return liftedGraph;
}

/// Builds a standard conv fprop graph using the integration test constants.
/// Used by multiple integration test fixtures for round-trip (lower/lift) testing.
inline std::shared_ptr<TestableGraphLifting>
    buildConvFpropGraph(const std::string& graphName = "ConvFpropTestGraph",
                        hipdnn_frontend::DataType computeType = hipdnn_frontend::DataType::FLOAT,
                        hipdnn_frontend::DataType intermediateType
                        = hipdnn_frontend::DataType::FLOAT,
                        hipdnn_frontend::DataType ioType = hipdnn_frontend::DataType::FLOAT)
{
    using namespace hipdnn_frontend::graph;
    using namespace constants;

    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name(graphName)
        .set_compute_data_type(computeType)
        .set_intermediate_data_type(intermediateType)
        .set_io_data_type(ioType);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_FPROP_TENSOR_X_UID).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    x->set_dim(toVec(K_FPROP_TENSOR_X_DIMS)).set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_FPROP_TENSOR_W_UID).set_name("W").set_data_type(hipdnn_frontend::DataType::FLOAT);
    w->set_dim(toVec(K_FPROP_TENSOR_W_DIMS)).set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fprop_op");
    convAttrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_FPROP_CONV_DILATION));
    convAttrs.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_uid(K_FPROP_TENSOR_Y_UID).set_output(true).set_name("Y");

    return graph;
}

/// Verify that a tensor with the given UID exists in the graph's tensor map
/// and matches expected attributes (name, dims, strides, data type).
inline void verifyTensorInGraph(
    const std::unordered_map<int64_t, std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>>&
        tensorMap,
    int64_t expectedUid,
    const std::string& expectedName,
    const std::vector<int64_t>& expectedDims,
    const std::vector<int64_t>& expectedStrides,
    hipdnn_frontend::DataType expectedDataType)
{
    auto it = tensorMap.find(expectedUid);
    ASSERT_NE(it, tensorMap.end()) << "Tensor with UID " << expectedUid << " not found";
    const auto& tensor = it->second;
    EXPECT_EQ(tensor->get_name(), expectedName) << "Name mismatch for UID " << expectedUid;
    EXPECT_EQ(tensor->get_dim(), expectedDims) << "Dims mismatch for UID " << expectedUid;
    EXPECT_EQ(tensor->get_stride(), expectedStrides) << "Strides mismatch for UID " << expectedUid;
    EXPECT_EQ(tensor->get_data_type(), expectedDataType)
        << "DataType mismatch for UID " << expectedUid;
}

} // namespace hipdnn_tests
