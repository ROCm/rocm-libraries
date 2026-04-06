// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>

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

    auto data = graph.toBinary();
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

} // namespace hipdnn_tests
