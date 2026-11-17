// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulBwdNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

MoeGroupedMatmulBwdAttributes createValidAttributes()
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910);
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911);
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912);
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913);
    dweightTensor->set_dim({2, 16, 32});
    // Column-major [K*N, 1, K], matching the layout infer_properties_node() generates.
    dweightTensor->set_stride({512, 1, 16});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    return attrs;
}

class TestMoeGroupedMatmulBwdNodeCreateOperation : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;
    std::array<char, 5> _fakeDescriptors{};
    size_t _nextDescriptor = 0;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    static MoeGroupedMatmulBwdNode makeNode()
    {
        auto attrs = createValidAttributes();
        attrs.set_compute_data_type(DataType::FLOAT);
        return {std::move(attrs), GraphAttributes{}};
    }

    hipdnnStatus_t createFakeDescriptor(hipdnnBackendDescriptor_t* descriptor)
    {
        *descriptor = reinterpret_cast<hipdnnBackendDescriptor_t>(
            std::addressof(_fakeDescriptors[_nextDescriptor++]));
        return HIPDNN_STATUS_SUCCESS;
    }
};

} // namespace

// --- GetNodeType ---

TEST(TestMoeGroupedMatmulBwdNode, GetNodeTypeReturnsMoeGroupedMatmulBwd)
{
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulBwdNode node(MoeGroupedMatmulBwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_GROUPED_MATMUL_BWD);
}

// --- PreValidateNode (success case) ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingDoutputTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except doutput
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 1, 16});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // doutput tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingTokenTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except token
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 1, 16});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // token tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingFirstTokenOffsetTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except first_token_offset
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 1, 16});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // first_token_offset tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingDweightTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except dweight
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);

    // dweight tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

// --- PreValidateNode: empty tensor dimensions ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsEmptyDoutputDims)
{
    auto attrs = createValidAttributes();
    attrs.get_doutput()->set_dim({});
    attrs.get_doutput()->set_stride({});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsEmptyTokenDims)
{
    auto attrs = createValidAttributes();
    attrs.get_token()->set_dim({});
    attrs.get_token()->set_stride({});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsEmptyFirstTokenOffsetDims)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_dim({});
    attrs.get_first_token_offset()->set_stride({});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeAllowsEmptyDweightDims)
{
    auto attrs = createValidAttributes();
    // Unlike the inputs above, dweight may arrive with empty dims: its shape is inferred in
    // infer_properties_node(), so pre_validate_node() must not reject it here.
    attrs.get_dweight()->set_dim({});
    attrs.get_dweight()->set_stride({});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: token/doutput/first_token_offset consistency checks ---
// dweight shape consistency is checked in infer_properties_node(), not here.

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsTokenCountMismatch)
{
    auto attrs = createValidAttributes();
    // doutput token-count dimension (dim[1] == 8) no longer matches token's dim[1]
    attrs.get_token()->set_dim({1, 4, 16});
    attrs.get_token()->set_stride({64, 16, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsNonInt32FirstTokenOffset)
{
    auto attrs = createValidAttributes();
    // first_token_offset must be INT32; this is a frontend-only check (the YAML's
    // expected_data_type only generates a backend finalize() check).
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsDoutputNonSingletonLeadingDimension)
{
    auto attrs = createValidAttributes();
    // doutput dim[0] must be 1; real tokens are flattened into dim[1].
    attrs.get_doutput()->set_dim({2, 8, 32});
    attrs.get_doutput()->set_stride({256, 32, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsTokenNonSingletonLeadingDimension)
{
    auto attrs = createValidAttributes();
    // token dim[0] must be 1; real tokens are flattened into dim[1].
    attrs.get_token()->set_dim({2, 8, 16});
    attrs.get_token()->set_stride({128, 16, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsZeroExpertCount)
{
    auto attrs = createValidAttributes();
    // first_token_offset dim[0] is the expert count and must describe at least one expert;
    // it becomes dweight's leading dimension in infer_properties_node().
    attrs.get_first_token_offset()->set_dim({0, 1, 1});
    attrs.get_first_token_offset()->set_stride({1, 1, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsDoutputRankMismatch)
{
    auto attrs = createValidAttributes();
    // doutput rank 4 passes the minimum-rank (>= 3) check but must still be rejected by the
    // exact-rank-3 check.
    attrs.get_doutput()->set_dim({1, 8, 32, 1});
    attrs.get_doutput()->set_stride({256, 32, 1, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsTokenRankMismatch)
{
    auto attrs = createValidAttributes();
    // token rank 4 passes the minimum-rank (>= 3) check but must still be rejected by the
    // exact-rank-3 check.
    attrs.get_token()->set_dim({1, 8, 16, 1});
    attrs.get_token()->set_stride({128, 16, 1, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsFirstTokenOffsetRankMismatch)
{
    auto attrs = createValidAttributes();
    // first_token_offset rank 4 passes the minimum-rank (>= 3) check but must still be
    // rejected by the exact-rank-3 check.
    attrs.get_first_token_offset()->set_dim({2, 1, 1, 1});
    attrs.get_first_token_offset()->set_stride({1, 1, 1, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsFirstTokenOffsetTrailingShapeMismatch)
{
    auto attrs = createValidAttributes();
    // first_token_offset must have trailing dimensions [1, 1]; mutate dim[1].
    attrs.get_first_token_offset()->set_dim({2, 2, 1});
    attrs.get_first_token_offset()->set_stride({2, 1, 1});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

// --- InferPropertiesNode ---

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeGeneratesDweightStrides)
{
    auto attrs = createValidAttributes();
    attrs.get_dweight()->set_stride({});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Column-major [K*N, 1, K] for dims [2, 16, 32]
    const std::vector<int64_t> expectedStrides{512, 1, 16};
    EXPECT_EQ(node.attributes.get_dweight()->get_stride(), expectedStrides);
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeInfersDweightDimsWhenUnset)
{
    auto attrs = createValidAttributes();
    // dweight = [first_token_offset.dim[0], token.dim[2], doutput.dim[2]] == [2, 16, 32]
    attrs.get_dweight()->set_dim({});
    attrs.get_dweight()->set_stride({});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

    const std::vector<int64_t> expectedDims{2, 16, 32};
    const std::vector<int64_t> expectedStrides{512, 1, 16};
    EXPECT_EQ(node.attributes.get_dweight()->get_dim(), expectedDims);
    EXPECT_EQ(node.attributes.get_dweight()->get_stride(), expectedStrides);
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeRejectsDweightKDimensionMismatch)
{
    auto attrs = createValidAttributes();
    // dweight K dimension (dim[1]) no longer matches token K dimension (dim[2] == 16)
    attrs.get_dweight()->set_dim({2, 8, 32});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeRejectsDweightNDimensionMismatch)
{
    auto attrs = createValidAttributes();
    // dweight N dimension (dim[2]) no longer matches doutput N dimension (dim[2] == 32)
    attrs.get_dweight()->set_dim({2, 16, 64});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeRejectsExpertCountMismatch)
{
    auto attrs = createValidAttributes();
    // dweight expert count (dim[0]) no longer matches first_token_offset's dim[0] (== 2)
    attrs.get_dweight()->set_dim({3, 16, 32});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeRejectsDweightRankMismatch)
{
    auto attrs = createValidAttributes();
    // A caller-supplied dweight of the wrong rank cannot match the inferred rank-3 shape.
    attrs.get_dweight()->set_dim({2, 16});
    attrs.get_dweight()->set_stride({16, 1});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

// --- GatherHipdnnTensors ---

TEST(TestMoeGroupedMatmulBwdNode, GatherHipdnnTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910).set_name("DoutputTensor");
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911).set_name("TokenTensor");
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912).set_name("FirstTokenOffsetTensor");
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913).set_name("DweightTensor");
    attrs.set_dweight(dweightTensor);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(doutputTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(tokenTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(firstTokenOffsetTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dweightTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}

// --- CreateOperation ---

TEST_F(TestMoeGroupedMatmulBwdNodeCreateOperation, PropagatesBackendError)
{
    EXPECT_CALL(
        *_mockBackend,
        backendCreateDescriptor(HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto node = makeNode();
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(tensorDescs.empty());
    EXPECT_TRUE(operations.empty());
}

TEST_F(TestMoeGroupedMatmulBwdNodeCreateOperation, SuccessCreatesFourTensorsAndOneOperation)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .Times(5)
        .WillRepeatedly(
            [this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                return createFakeDescriptor(descriptor);
            });
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_))
        .Times(5)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(5)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto node = makeNode();
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(tensorDescs.size(), 4u);
    EXPECT_EQ(operations.size(), 1u);
}

// NOLINTEND(misc-const-correctness)
