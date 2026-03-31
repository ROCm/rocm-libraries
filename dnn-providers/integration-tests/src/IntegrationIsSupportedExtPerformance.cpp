// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <chrono>
#include <hipdnn_frontend.hpp>
#include <iostream>

#include "harness/SharedHandle.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr int K_ITERATIONS = 1000;

class IntegrationIsSupportedExtPerformance : public ::testing::Test
{
protected:
    static Graph createSimpleBatchnormGraph()
    {
        // NCHW tensor: [N=2, C=3, H=4, W=4]
        const std::vector<int64_t> dims = {2, 3, 4, 4};
        const std::vector<int64_t> strides
            = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
        // Per-channel: [1, C, 1, 1]
        const std::vector<int64_t> channelDims = {1, dims[1], 1, 1};
        const std::vector<int64_t> channelStrides = {dims[1], 1, 1, 1};

        Graph graph;
        graph.set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X").set_uid(1).set_dim(dims).set_stride(strides).set_data_type(
            DataType::FLOAT);

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_name("mean")
            .set_uid(2)
            .set_dim(channelDims)
            .set_stride(channelStrides)
            .set_data_type(DataType::FLOAT);

        auto invVar = std::make_shared<TensorAttributes>();
        invVar->set_name("inv_variance")
            .set_uid(3)
            .set_dim(channelDims)
            .set_stride(channelStrides)
            .set_data_type(DataType::FLOAT);

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_name("scale")
            .set_uid(4)
            .set_dim(channelDims)
            .set_stride(channelStrides)
            .set_data_type(DataType::FLOAT);

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_name("bias")
            .set_uid(5)
            .set_dim(channelDims)
            .set_stride(channelStrides)
            .set_data_type(DataType::FLOAT);

        const BatchnormInferenceAttributes bnAttrs;
        auto y = graph.batchnorm_inference(x, mean, invVar, scale, bias, bnAttrs);
        y->set_name("Y").set_uid(6).set_data_type(DataType::FLOAT).set_output(true);

        return graph;
    }

    hipdnnHandle_t _handle = hipdnn_integration_tests::getSharedHandle();
};

TEST_F(IntegrationIsSupportedExtPerformance, ColdCallCompletesWithinThreshold)
{
    const auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < K_ITERATIONS; ++i)
    {
        Graph graph = createSimpleBatchnormGraph();
        auto result = graph.is_supported_ext(_handle);
        ASSERT_TRUE(result.is_good()) << "Iteration " << i << ": " << result.get_message();
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(end - start);
    const auto avgUs
        = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / K_ITERATIONS;

    std::cout << "[  PERF    ] Cold is_supported_ext: " << elapsed.count() << "s total, " << avgUs
              << "us avg per call (" << K_ITERATIONS << " iterations)" << '\n';

    EXPECT_LT(elapsed.count(), 10.0) << "Cold is_supported_ext took " << elapsed.count() << "s for "
                                     << K_ITERATIONS << " iterations (threshold: 10s)";
}

TEST_F(IntegrationIsSupportedExtPerformance, HotCallCompletesWithinThreshold)
{
    Graph graph = createSimpleBatchnormGraph();

    auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.get_message();

    result = graph.build_operation_graph(_handle);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    const auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < K_ITERATIONS; ++i)
    {
        result = graph.is_supported_ext(_handle);
        ASSERT_TRUE(result.is_good()) << "Iteration " << i << ": " << result.get_message();
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(end - start);
    const auto avgNs
        = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / K_ITERATIONS;

    std::cout << "[  PERF    ] Hot is_supported_ext: " << elapsed.count() << "s total, " << avgNs
              << "ns avg per call (" << K_ITERATIONS << " iterations)" << '\n';

    EXPECT_LT(elapsed.count(), 1.0) << "Hot is_supported_ext took " << elapsed.count() << "s for "
                                    << K_ITERATIONS << " iterations (threshold: 1s)";
}

} // namespace
