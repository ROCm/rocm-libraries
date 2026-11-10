// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides,
                                             DataType dataType)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_stride(strides).set_data_type(dataType);
    return tensor;
}

} // namespace

TEST(TestGraphPagedCacheStore, BuildGraphReturnsCallerContainer)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = makeTensor({4, 2, 8}, {16, 8, 1}, DataType::FLOAT);
    auto slotMapping = makeTensor({4}, {1}, DataType::INT32);
    auto container = makeTensor({3, 2, 4, 8}, {64, 32, 8, 1}, DataType::FLOAT);

    PagedCacheStoreAttributes attributes;
    attributes.set_name("PagedCacheStoreNode");

    auto result = graph.paged_cache_store(x, slotMapping, container, attributes);

    ASSERT_EQ(result, container);
    EXPECT_EQ(container->get_name(), "PagedCacheStoreNode::CONTAINER");
    EXPECT_EQ(x->get_name(), "PagedCacheStoreNode::X");
    EXPECT_EQ(slotMapping->get_name(), "PagedCacheStoreNode::SLOT_MAPPING");
    EXPECT_FALSE(container->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphPagedCacheStore, BuildGraphSdpaNativeLayoutWithScale)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT);

    // x [B=2, H=2, S=3, D=8] stored in BSHD memory order.
    auto x = makeTensor({2, 2, 3, 8}, {48, 8, 16, 1}, DataType::HALF);
    auto slotMapping = makeTensor({2, 3}, {3, 1}, DataType::INT64);
    auto container = makeTensor({3, 2, 4, 8}, {64, 32, 8, 1}, DataType::HALF);
    auto scale = makeTensor({1}, {1}, DataType::FLOAT);

    PagedCacheStoreAttributes attributes;
    attributes.set_name("store").set_scale(scale);

    auto result = graph.paged_cache_store(x, slotMapping, container, attributes);
    ASSERT_EQ(result, container);
    EXPECT_EQ(scale->get_name(), "store::SCALE");

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphPagedCacheStore, NullInputsReportedByValidate)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto slotMapping = makeTensor({4}, {1}, DataType::INT32);
    auto container = makeTensor({3, 2, 4, 8}, {64, 32, 8, 1}, DataType::FLOAT);

    auto result
        = graph.paged_cache_store(nullptr, slotMapping, container, PagedCacheStoreAttributes{});
    EXPECT_EQ(result, container);

    auto validationResult = graph.validate();
    EXPECT_FALSE(validationResult.is_good());
}

TEST(TestGraphPagedCacheStore, ContainerHeadMismatchFailsValidate)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = makeTensor({4, 2, 8}, {16, 8, 1}, DataType::FLOAT);
    auto slotMapping = makeTensor({4}, {1}, DataType::INT32);
    auto container = makeTensor({3, 1, 4, 8}, {32, 32, 8, 1}, DataType::FLOAT);

    graph.paged_cache_store(x, slotMapping, container, PagedCacheStoreAttributes{});

    auto validationResult = graph.validate();
    EXPECT_FALSE(validationResult.is_good());
}
