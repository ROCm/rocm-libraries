// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

namespace
{

MoeGroupedMatmulAttributes createValidAttributes()
{
    MoeGroupedMatmulAttributes attrs;

    auto token = std::make_shared<TensorAttributes>();
    token->set_uid(1900)
        .set_dim({1, 8, 16})
        .set_stride({128, 16, 1})
        .set_data_type(DataType::FLOAT);
    attrs.set_token(token);

    auto weight = std::make_shared<TensorAttributes>();
    weight->set_uid(1901)
        .set_dim({2, 16, 32})
        .set_stride({512, 32, 1})
        .set_data_type(DataType::FLOAT);
    attrs.set_weight(weight);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_uid(1902).set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(
        DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffset);

    auto output = std::make_shared<TensorAttributes>();
    output->set_uid(1905)
        .set_dim({1, 8, 32})
        .set_stride({256, 32, 1})
        .set_data_type(DataType::FLOAT);
    attrs.set_output(output);
    attrs.set_compute_data_type(DataType::FLOAT);
    return attrs;
}

std::shared_ptr<TensorAttributes> makeRoutingTensor(int64_t routedTokens = 8,
                                                    DataType dataType = DataType::INT32)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(1903)
        .set_dim({1, routedTokens, 1})
        .set_stride({routedTokens, 1, 1})
        .set_data_type(dataType);
    return tensor;
}

template <size_t N>
using DescriptorStorage = std::array<char, N>;

template <size_t N>
hipdnnBackendDescriptor_t descriptorAt(DescriptorStorage<N>& storage, size_t index)
{
    return reinterpret_cast<hipdnnBackendDescriptor_t>(std::addressof(storage[index]));
}

constexpr std::array<hipdnnBackendAttributeName_t, 4> K_REQUIRED_TENSOR_ATTRS
    = {HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
       HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
       HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
       HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC};

inline void writeBackendValue(void* destination, hipdnnBackendDescriptor_t value)
{
    auto* output = static_cast<hipdnnBackendDescriptor_t*>(destination);
    *output = value;
}

template <typename T>
void writeBackendValue(void* destination, const T& value)
{
    std::memcpy(destination, static_cast<const void*>(std::addressof(value)), sizeof(T));
}

class MoeMockBackendFixture : public ::testing::Test
{
protected:
    std::shared_ptr<NiceMock<Mock_hipdnn_backend>> _mockBackend;
    DescriptorStorage<64> _fakeDescriptors{};
    size_t _nextDescriptor = 0;

    void SetUp() override
    {
        _mockBackend = std::make_shared<NiceMock<Mock_hipdnn_backend>>();
        IHipdnnBackend::setInstance(_mockBackend);
        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t,
                                  hipdnnBackendDescriptor_t* descriptor) {
                *descriptor
                    = descriptorAt(_fakeDescriptors, _nextDescriptor++ % _fakeDescriptors.size());
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .Times(AnyNumber())
            .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, getLastErrorString(_, _))
            .WillByDefault([](char* message, size_t size) {
                if(size > 0)
                {
                    message[0] = '\0';
                }
            });
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    static MoeGroupedMatmulNode makeNode(MoeGroupedMatmulMode mode)
    {
        auto attrs = createValidAttributes();
        attrs.set_mode(mode);
        if(mode != MoeGroupedMatmulMode::NONE)
        {
            attrs.set_token_index(makeRoutingTensor());
        }
        if(mode == MoeGroupedMatmulMode::SCATTER)
        {
            auto tokenKs = makeRoutingTensor();
            tokenKs->set_uid(1904);
            attrs.set_token_ks(tokenKs).set_top_k(2);
        }
        return {std::move(attrs), GraphAttributes{}};
    }

    void expectOperationAttribute(hipdnnBackendAttributeName_t attrName, int expectedCount = 1)
    {
        EXPECT_CALL(*_mockBackend, backendSetAttribute(_, attrName, _, _, _))
            .Times(expectedCount)
            .RetiresOnSaturation();
    }

    void expectNoOperationAttribute(hipdnnBackendAttributeName_t attrName)
    {
        EXPECT_CALL(*_mockBackend, backendSetAttribute(_, attrName, _, _, _)).Times(0);
    }
};

struct UnpackScenario
{
    MoeGroupedMatmulMode frontendMode;
    hipdnnMoeGroupedMatmulMode_t backendMode;
    bool hasTokenIndex;
    bool hasTokenKs;
    int32_t topK;
};

class MoeUnpackModeTest : public MoeMockBackendFixture,
                          public ::testing::WithParamInterface<UnpackScenario>
{
protected:
    DescriptorStorage<1> _operationStorage{};
    DescriptorStorage<6> _tensorDescriptorStorage{};

    hipdnnBackendDescriptor_t operationDesc()
    {
        return descriptorAt(_operationStorage, 0);
    }

    void expectTensorReference(hipdnnBackendAttributeName_t attrName,
                               hipdnnBackendDescriptor_t tensorDesc,
                               int64_t uid)
    {
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(operationDesc(), attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _, _))
            .WillOnce(Invoke([tensorDesc](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t* count,
                                          void* value) {
                *count = 1;
                writeBackendValue(value, tensorDesc);
                return HIPDNN_STATUS_SUCCESS;
            }));
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        tensorDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
            .WillOnce(Invoke([uid](hipdnnBackendDescriptor_t,
                                   hipdnnBackendAttributeName_t,
                                   hipdnnBackendAttributeType_t,
                                   int64_t,
                                   int64_t* count,
                                   void* value) {
                *count = 1;
                writeBackendValue(value, uid);
                return HIPDNN_STATUS_SUCCESS;
            }));
        EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(tensorDesc))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    }

    void expectCommonTensorReferences(
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        constexpr std::array<int64_t, 4> K_UIDS = {1900, 1901, 1902, 1905};
        for(size_t index = 0; index < K_REQUIRED_TENSOR_ATTRS.size(); ++index)
        {
            auto tensor = std::make_shared<TensorAttributes>();
            tensor->set_uid(K_UIDS[index]);
            tensorMap.emplace(K_UIDS[index], tensor);
            expectTensorReference(K_REQUIRED_TENSOR_ATTRS[index],
                                  descriptorAt(_tensorDescriptorStorage, index),
                                  K_UIDS[index]);
        }
    }
};

} // namespace

TEST(TestMoeGroupedMatmulNode, GetNodeTypeReturnsMoeGroupedMatmul)
{
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulNode node(MoeGroupedMatmulAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_GROUPED_MATMUL);
}

TEST(TestMoeGroupedMatmulNode, NoneModeAcceptsRequiredInputs)
{
    auto attrs = createValidAttributes();
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
}

TEST(TestMoeGroupedMatmulNode, NotSetModeIsRejected)
{
    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::NOT_SET);

    const GraphAttributes graphAttrs;
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, RequiredTensorsMustBePresent)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_token(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_weight(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_first_token_offset(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_output(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, RequiredTensorShapesAndTypesMustBeConsistent)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.get_token()->set_dim({1, 8, 15});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_dim({3, 1, 1});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherRequiresInt32TokenIndex)
{
    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER);

    const GraphAttributes graphAttrs;
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER)
        .set_token_index(makeRoutingTensor(8, DataType::FLOAT));
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterValidatesRoutingAndTopK)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(2);
    EXPECT_TRUE(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().is_good());

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(0);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor(8))
        .set_token_ks(makeRoutingTensor(7))
        .set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterRequiresValidRoutingAndBoundedTopK)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER).set_token_index(makeRoutingTensor()).set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor(8, DataType::FLOAT))
        .set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(3);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherInfersOutputDimensionsAndStrides)
{
    auto attrs = createValidAttributes();
    auto output = std::make_shared<TensorAttributes>();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER)
        .set_token_index(makeRoutingTensor(6))
        .set_output(output);

    const GraphAttributes graphAttrs;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
    const auto error = node.infer_properties_node();
    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 6, 32}));
    EXPECT_EQ(output->get_stride(), (std::vector<int64_t>{192, 32, 1}));
}

TEST(TestMoeGroupedMatmulNode, ScatterInfersSourceTokenOutputDimensionsAndStrides)
{
    auto attrs = createValidAttributes();
    auto output = std::make_shared<TensorAttributes>();
    auto tokenKs = makeRoutingTensor(6);
    tokenKs->set_uid(1904);
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor(6))
        .set_token_ks(tokenKs)
        .set_top_k(2)
        .set_output(output);

    const GraphAttributes graphAttrs;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
    const auto error = node.infer_properties_node();
    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 8, 32}));
    EXPECT_EQ(output->get_stride(), (std::vector<int64_t>{256, 32, 1}));
}

TEST(TestMoeGroupedMatmulNode, NoneInfersOutputDimensionsAndRejectsMismatch)
{
    auto attrs = createValidAttributes();
    auto output = std::make_shared<TensorAttributes>();
    attrs.set_output(output);

    const GraphAttributes graphAttrs;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
    EXPECT_TRUE(node.infer_properties_node().is_good());
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 8, 32}));

    attrs = createValidAttributes();
    attrs.get_output()->set_dim({1, 7, 32});
    MoeGroupedMatmulNode mismatchedNode(std::move(attrs), graphAttrs);
    EXPECT_TRUE(mismatchedNode.pre_validate_node().is_good());
    EXPECT_EQ(mismatchedNode.infer_properties_node().code, ErrorCode::INVALID_VALUE);
}

TEST_F(MoeMockBackendFixture, CreateOperationUsesCanonicalNONEFootprint)
{
    expectNoOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC);
    expectNoOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC);
    expectNoOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K);

    auto node = makeNode(MoeGroupedMatmulMode::NONE);
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(tensorDescs.size(), 4u);
    EXPECT_EQ(operations.size(), 1u);
}

TEST_F(MoeMockBackendFixture, CreateOperationUsesCanonicalGATHERFootprint)
{
    expectOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC);
    expectNoOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC);
    expectNoOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K);

    auto node = makeNode(MoeGroupedMatmulMode::GATHER);
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(tensorDescs.size(), 5u);
    EXPECT_EQ(operations.size(), 1u);
}

TEST_F(MoeMockBackendFixture, CreateOperationUsesCanonicalSCATTERFootprint)
{
    expectOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC);
    expectOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC);
    expectOperationAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K);

    auto node = makeNode(MoeGroupedMatmulMode::SCATTER);
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(tensorDescs.size(), 6u);
    EXPECT_EQ(operations.size(), 1u);
}

TEST_F(MoeMockBackendFixture, CreateOperationPropagatesBackendCreationError)
{
    EXPECT_CALL(*_mockBackend,
                backendCreateDescriptor(HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto node = makeNode(MoeGroupedMatmulMode::NONE);
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<ScopedHipdnnBackendDescriptor> operations;
    const auto error = node.create_operation(tensorDescs, operations);

    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(operations.empty());
}

TEST_P(MoeUnpackModeTest, UnpackFromDescriptorReadsOnlyModeSpecificAttributes)
{
    const auto scenario = GetParam();
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    expectCommonTensorReferences(tensorMap);

    if(scenario.hasTokenIndex)
    {
        auto tokenIndex = makeRoutingTensor();
        tensorMap.emplace(1903, tokenIndex);
        expectTensorReference(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                              descriptorAt(_tensorDescriptorStorage, 4),
                              1903);
    }
    else
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(operationDesc(),
                                        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                        _,
                                        _,
                                        _,
                                        _))
            .Times(0);
    }
    if(scenario.hasTokenKs)
    {
        auto tokenKs = makeRoutingTensor();
        tokenKs->set_uid(1904);
        tensorMap.emplace(1904, tokenKs);
        expectTensorReference(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                              descriptorAt(_tensorDescriptorStorage, 5),
                              1904);
    }
    else
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(operationDesc(),
                                        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                        _,
                                        _,
                                        _,
                                        _))
            .Times(0);
    }

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(operationDesc(),
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                    HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                    1,
                                    _,
                                    _))
        .WillOnce(Invoke([scenario](hipdnnBackendDescriptor_t,
                                    hipdnnBackendAttributeName_t,
                                    hipdnnBackendAttributeType_t,
                                    int64_t,
                                    int64_t* count,
                                    void* value) {
            *count = 1;
            writeBackendValue(value, scenario.backendMode);
            return HIPDNN_STATUS_SUCCESS;
        }));
    if(scenario.frontendMode == MoeGroupedMatmulMode::SCATTER)
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(operationDesc(),
                                        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                                        HIPDNN_TYPE_INT32,
                                        1,
                                        _,
                                        _))
            .WillOnce(Invoke([scenario](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t,
                                        int64_t* count,
                                        void* value) {
                *count = 1;
                writeBackendValue(value, scenario.topK);
                return HIPDNN_STATUS_SUCCESS;
            }));
    }
    else
    {
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(
                operationDesc(), HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, _, _, _, _))
            .Times(0);
    }
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(operationDesc(),
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                    HIPDNN_TYPE_DATA_TYPE,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}), Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(operationDesc(),
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                    HIPDNN_TYPE_DATA_TYPE,
                                    1,
                                    _,
                                    _))
        .WillOnce(Invoke([](hipdnnBackendDescriptor_t,
                            hipdnnBackendAttributeName_t,
                            hipdnnBackendAttributeType_t,
                            int64_t,
                            int64_t* count,
                            void* value) {
            *count = 1;
            constexpr auto K_DATA_TYPE = HIPDNN_DATA_FLOAT;
            writeBackendValue(value, K_DATA_TYPE);
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            operationDesc(), HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    MoeGroupedMatmulNode node(MoeGroupedMatmulAttributes{}, GraphAttributes{});
    const auto error = node.unpack_from_descriptor(operationDesc(), tensorMap);

    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(node.attributes.get_mode(), scenario.frontendMode);
    EXPECT_EQ(node.attributes.get_token_index() != nullptr, scenario.hasTokenIndex);
    EXPECT_EQ(node.attributes.get_token_ks() != nullptr, scenario.hasTokenKs);
    EXPECT_EQ(node.attributes.get_top_k(), scenario.topK);
}

TEST_F(MoeUnpackModeTest, UnpackFromDescriptorPreservesAttributesOnFailure)
{
    auto original = createValidAttributes();
    original.set_name("preserved");
    MoeGroupedMatmulNode node(std::move(original), GraphAttributes{});
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(operationDesc(),
                                    HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const auto error = node.unpack_from_descriptor(operationDesc(), tensorMap);

    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(node.attributes.get_name(), "preserved");
    EXPECT_NE(node.attributes.get_token(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AllModes,
    MoeUnpackModeTest,
    ::testing::Values(
        UnpackScenario{
            MoeGroupedMatmulMode::NONE, HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE, false, false, 0},
        UnpackScenario{
            MoeGroupedMatmulMode::GATHER, HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER, true, false, 0},
        UnpackScenario{
            MoeGroupedMatmulMode::SCATTER, HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER, true, true, 2}));
