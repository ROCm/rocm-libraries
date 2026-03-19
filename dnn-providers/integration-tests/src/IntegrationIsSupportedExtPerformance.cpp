// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <chrono>
#include <hipdnn_frontend.hpp>
#include <iostream>

#include "harness/SharedHandle.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

constexpr int kIterations = 1000;

class IntegrationIsSupportedExtPerformance : public ::testing::Test {
   protected:
    static Graph createSimplePointwiseGraph() {
        std::vector<int64_t> const dims = {2, 3, 4, 4};

        Graph graph;
        graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X")
            .set_uid(1)
            .set_dim(dims)
            .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
            .set_data_type(DataType::FLOAT);

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);

        auto y = graph.pointwise(x, attrs);
        y->set_name("Y").set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

        return graph;
    }

    hipdnnHandle_t handle_ = hipdnn_integration_tests::getSharedHandle();
};

TEST_F(IntegrationIsSupportedExtPerformance, ColdCallCompletesWithinThreshold) {
    auto const start = std::chrono::steady_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        Graph graph = createSimplePointwiseGraph();
        auto result = graph.is_supported_ext(handle_);
        ASSERT_TRUE(result.is_good()) << "Iteration " << i << ": " << result.get_message();
    }

    auto const end = std::chrono::steady_clock::now();
    auto const elapsed = std::chrono::duration<double>(end - start);
    auto const avgUs =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / kIterations;

    std::cout << "[  PERF   ] Cold is_supported_ext: " << elapsed.count() << "s total, " << avgUs
              << "us avg per call (" << kIterations << " iterations)" << std::endl;

    EXPECT_LT(elapsed.count(), 10.0) << "Cold is_supported_ext took " << elapsed.count() << "s for "
                                     << kIterations << " iterations (threshold: 10s)";
}

TEST_F(IntegrationIsSupportedExtPerformance, HotCallCompletesWithinThreshold) {
    Graph graph = createSimplePointwiseGraph();

    auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.get_message();

    result = graph.build_operation_graph(handle_);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto const start = std::chrono::steady_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        result = graph.is_supported_ext(handle_);
        ASSERT_TRUE(result.is_good()) << "Iteration " << i << ": " << result.get_message();
    }

    auto const end = std::chrono::steady_clock::now();
    auto const elapsed = std::chrono::duration<double>(end - start);
    auto const avgNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / kIterations;

    std::cout << "[  PERF   ] Hot is_supported_ext: " << elapsed.count() << "s total, " << avgNs
              << "ns avg per call (" << kIterations << " iterations)" << std::endl;

    EXPECT_LT(elapsed.count(), 1.0) << "Hot is_supported_ext took " << elapsed.count() << "s for "
                                    << kIterations << " iterations (threshold: 1s)";
}

}  // namespace
