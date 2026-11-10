// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/node/PagedCacheStoreNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

PagedCacheStoreAttributes createValidAttributes()
{
    PagedCacheStoreAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({4, 2, 8});
    xTensor->set_stride({16, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_dim({4});
    slotMappingTensor->set_stride({1});
    slotMappingTensor->set_data_type(DataType::INT32);
    attrs.set_slot_mapping(slotMappingTensor);
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_dim({3, 2, 4, 8});
    containerTensor->set_stride({64, 32, 8, 1});
    containerTensor->set_data_type(DataType::FLOAT);
    attrs.set_container(containerTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestPagedCacheStoreNode, GetNodeTypeReturnsPagedCacheStore)
{
    const GraphAttributes graphAttrs;
    const PagedCacheStoreNode node(PagedCacheStoreAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::PAGED_CACHE_STORE);
}

// --- PreValidateNode (success case) ---

TEST(TestPagedCacheStoreNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestPagedCacheStoreNode, PreValidateNodeMissingXTensor)
{
    PagedCacheStoreAttributes attrs;

    // Set all required tensors except x
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_dim({4});
    slotMappingTensor->set_stride({1});
    slotMappingTensor->set_data_type(DataType::INT32);
    attrs.set_slot_mapping(slotMappingTensor);
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_dim({3, 2, 4, 8});
    containerTensor->set_stride({64, 32, 8, 1});
    containerTensor->set_data_type(DataType::FLOAT);
    attrs.set_container(containerTensor);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestPagedCacheStoreNode, PreValidateNodeMissingSlotMappingTensor)
{
    PagedCacheStoreAttributes attrs;

    // Set all required tensors except slot_mapping
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({4, 2, 8});
    xTensor->set_stride({16, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_dim({3, 2, 4, 8});
    containerTensor->set_stride({64, 32, 8, 1});
    containerTensor->set_data_type(DataType::FLOAT);
    attrs.set_container(containerTensor);

    // slot_mapping tensor is missing
    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestPagedCacheStoreNode, PreValidateNodeMissingContainerTensor)
{
    PagedCacheStoreAttributes attrs;

    // Set all required tensors except container
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({4, 2, 8});
    xTensor->set_stride({16, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_dim({4});
    slotMappingTensor->set_stride({1});
    slotMappingTensor->set_data_type(DataType::INT32);
    attrs.set_slot_mapping(slotMappingTensor);

    // container tensor is missing
    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestPagedCacheStoreNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode (rule failures) ---

namespace
{

ErrorCode preValidate(PagedCacheStoreAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node().code;
}

std::shared_ptr<TensorAttributes> makeScale(const std::vector<int64_t>& dims, DataType dataType)
{
    auto scale = std::make_shared<TensorAttributes>();
    const auto rank = dims.size();
    scale->set_dim(dims);
    scale->set_stride(std::vector<int64_t>(rank, 1));
    scale->set_data_type(dataType);
    return scale;
}

} // namespace

TEST(TestPagedCacheStoreNode, PreValidateAcceptsInt64SlotMapping)
{
    auto attrs = createValidAttributes();
    attrs.get_slot_mapping()->set_data_type(DataType::INT64);
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::OK);
}

TEST(TestPagedCacheStoreNode, PreValidateAcceptsFloatScalarScale)
{
    auto attrs = createValidAttributes();
    attrs.set_scale(makeScale({1}, DataType::FLOAT));
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::OK);
}

TEST(TestPagedCacheStoreNode, PreValidateAcceptsSdpaNativeLayout)
{
    auto attrs = createValidAttributes();
    // x [B=2, H=2, S=3, D=8] in BSHD memory order, slot_mapping [B, S].
    attrs.get_x()->set_dim({2, 2, 3, 8}).set_stride({48, 8, 16, 1});
    attrs.get_slot_mapping()->set_dim({2, 3}).set_stride({3, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::OK);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsBadXRank)
{
    auto attrs = createValidAttributes();
    attrs.get_x()->set_dim({4, 16}).set_stride({16, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsSlotMappingLengthMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_slot_mapping()->set_dim({5});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsRank2SlotMappingWithTokenPackedX)
{
    auto attrs = createValidAttributes();
    attrs.get_slot_mapping()->set_dim({4, 1}).set_stride({1, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsSdpaNativeSlotMappingMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_x()->set_dim({2, 2, 3, 8}).set_stride({48, 24, 8, 1});
    attrs.get_slot_mapping()->set_dim({2, 4}).set_stride({4, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsFloatSlotMapping)
{
    auto attrs = createValidAttributes();
    attrs.get_slot_mapping()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsContainerRank)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_dim({3, 2, 32}).set_stride({64, 32, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsContainerHeadCountMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_dim({3, 4, 4, 8}).set_stride({128, 32, 8, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsContainerHeadSizeMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_dim({3, 2, 4, 16}).set_stride({128, 64, 16, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsEmptyContainerBlocks)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_dim({3, 2, 0, 8}).set_stride({16, 8, 8, 1});
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsVirtualContainer)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_is_virtual(true);
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsNonScalarScale)
{
    auto attrs = createValidAttributes();
    attrs.set_scale(makeScale({2}, DataType::FLOAT));
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

TEST(TestPagedCacheStoreNode, PreValidateRejectsHalfScale)
{
    auto attrs = createValidAttributes();
    attrs.set_scale(makeScale({1}, DataType::HALF));
    EXPECT_EQ(preValidate(std::move(attrs)), error_code_t::INVALID_VALUE);
}

// --- InferPropertiesNode ---

TEST(TestPagedCacheStoreNode, InferPropertiesKeepsCallerContainerLayout)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_stride({64, 8, 16, 1});

    const GraphAttributes graphAttributes;
    PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_container()->get_dim(), (std::vector<int64_t>{3, 2, 4, 8}));
    EXPECT_EQ(node.attributes.get_container()->get_stride(), (std::vector<int64_t>{64, 8, 16, 1}));
}

TEST(TestPagedCacheStoreNode, InferPropertiesFillsPackedContainerStrides)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_stride({});

    const GraphAttributes graphAttributes;
    PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_container()->get_stride(), (std::vector<int64_t>{64, 32, 8, 1}));
}

TEST(TestPagedCacheStoreNode, InferPropertiesRejectsContainerWithoutDims)
{
    auto attrs = createValidAttributes();
    attrs.get_container()->set_dim({}).set_stride({});

    const GraphAttributes graphAttributes;
    PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.infer_properties_node().code, error_code_t::ATTRIBUTE_NOT_SET);
}

// --- GatherHipdnnTensors ---

TEST(TestPagedCacheStoreNode, GatherHipdnnTensor)
{
    PagedCacheStoreAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9500).set_name("XTensor");
    attrs.set_x(xTensor);
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_uid(9501).set_name("SlotMappingTensor");
    attrs.set_slot_mapping(slotMappingTensor);
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_uid(9502).set_name("ContainerTensor");
    attrs.set_container(containerTensor);

    const GraphAttributes graphAttributes;
    const PagedCacheStoreNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(slotMappingTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(containerTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

// NOLINTEND(misc-const-correctness)
