// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

#include <hipdnn_frontend/detail/OperationUnpacker.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

namespace
{

// Concrete INode subclass that does NOT override unpack_from_descriptor.
// Used to verify the default implementation returns an error.
class FakeNodeNoUnpack : public INode
{
public:
    explicit FakeNodeNoUnpack(GraphAttributes attrs = GraphAttributes())
        : INode(std::move(attrs))
    {
    }

    std::string getNodeName() const override
    {
        return "FakeNodeNoUnpack";
    }
};

} // namespace

// ---------------------------------------------------------------------------
// INode::unpack_from_descriptor default implementation tests
// ---------------------------------------------------------------------------

TEST(TestOperationUnpacker, DefaultUnpackFromDescriptorReturnsError)
{
    FakeNodeNoUnpack node;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;

    auto err = node.unpack_from_descriptor(nullptr, tensorMap);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("not implemented") != std::string::npos);
    EXPECT_TRUE(err.get_message().find("FakeNodeNoUnpack") != std::string::npos);
}

TEST(TestOperationUnpacker, DefaultUnpackFromDescriptorIncludesNodeName)
{
    FakeNodeNoUnpack node;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;

    auto err = node.unpack_from_descriptor(nullptr, tensorMap);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("FakeNodeNoUnpack") != std::string::npos);
}

// ---------------------------------------------------------------------------
// queryOperationType tests
// ---------------------------------------------------------------------------

class TestQueryOperationType : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

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
};

TEST_F(TestQueryOperationType, QueryReturnsConvForwardType)
{
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto value = HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD;
                            std::memcpy(arrayOfElements, &value, sizeof(hipdnnOperationType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [result, err] = queryOperationType(desc);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(result, HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD);
}

TEST_F(TestQueryOperationType, ReturnsErrorWhenQueryFails)
{
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_OPERATION_TYPE_EXT, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    // Need to also expect getLastErrorString to be called
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [result, err] = queryOperationType(desc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
    EXPECT_EQ(result, HIPDNN_OPERATION_TYPE_NOT_SET);
}

TEST_F(TestQueryOperationType, PassesThroughUnknownOperationType)
{
    auto unknownType = static_cast<hipdnnOperationType_t>(999);

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([unknownType](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                            std::memcpy(
                                arrayOfElements, &unknownType, sizeof(hipdnnOperationType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [result, err] = queryOperationType(desc);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(result, unknownType);
}

// ---------------------------------------------------------------------------
// unpackOperation tests
// ---------------------------------------------------------------------------

class TestUnpackOperation : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

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
};

TEST_F(TestUnpackOperation, FailsWhenTypeCannotBeDetermined)
{
    // Query fails
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_OPERATION_TYPE_EXT, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("Failed to determine operation type") != std::string::npos);
}

TEST_F(TestUnpackOperation, FailsForUnsupportedOperationType)
{
    auto unknownType = static_cast<hipdnnOperationType_t>(999);

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([unknownType](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                            std::memcpy(
                                arrayOfElements, &unknownType, sizeof(hipdnnOperationType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("Unsupported operation type") != std::string::npos);
    EXPECT_TRUE(err.get_message().find("999") != std::string::npos)
        << "Error should include the unsupported type id, got: " << err.get_message();
}

TEST_F(TestUnpackOperation, FailsImmediatelyOnUnpackError)
{
    // Type query succeeds with CONV_FORWARD
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto value = HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD;
                            std::memcpy(arrayOfElements, &value, sizeof(hipdnnOperationType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Unpacking the ConvFwd operation fails (getDescriptorAttrDesc for X tensor fails)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_,
                                    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

TEST_F(TestUnpackOperation, UnpackOperationSuccessConvFprop)
{
    // Test constants
    constexpr int64_t K_X_UID = 10;
    constexpr int64_t K_W_UID = 20;
    constexpr int64_t K_Y_UID = 30;
    constexpr std::array<int64_t, 4> K_X_DIMS = {1, 3, 32, 32};
    constexpr std::array<int64_t, 4> K_X_STRIDES = {3072, 1024, 32, 1};
    constexpr std::array<int64_t, 4> K_W_DIMS = {64, 3, 3, 3};
    constexpr std::array<int64_t, 4> K_W_STRIDES = {27, 9, 3, 1};
    constexpr std::array<int64_t, 4> K_Y_DIMS = {1, 64, 32, 32};
    constexpr std::array<int64_t, 4> K_Y_STRIDES = {65536, 1024, 32, 1};
    constexpr std::array<int64_t, 2> K_PADDING = {1, 1};
    constexpr std::array<int64_t, 2> K_STRIDE = {1, 1};
    constexpr std::array<int64_t, 2> K_DILATION = {1, 1};

    // Fake descriptor pointers for the 3 tensors
    int xDescPlaceholder = 0;
    int wDescPlaceholder = 0;
    int yDescPlaceholder = 0;
    auto* xDescFake = reinterpret_cast<hipdnnBackendDescriptor_t>(&xDescPlaceholder);
    auto* wDescFake = reinterpret_cast<hipdnnBackendDescriptor_t>(&wDescPlaceholder);
    auto* yDescFake = reinterpret_cast<hipdnnBackendDescriptor_t>(&yDescPlaceholder);

    // --- 1. Mock operation type query ---
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto value = HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD;
                            std::memcpy(arrayOfElements, &value, sizeof(hipdnnOperationType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Helper lambda: mocks one tensor descriptor retrieval and all its attribute queries.
    // The tensor attr calls are dispatched by descriptor pointer, so we use specific
    // matchers on arg0.
    auto expectTensorDescGet = [&](hipdnnBackendAttributeName_t tensorAttrName,
                                   hipdnnBackendDescriptor_t fakeDesc) {
        // backendGetAttribute for the tensor descriptor from the operation
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(_, tensorAttrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([fakeDesc](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* arrayOfElements) {
                                auto descPtr
                                    = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                                *descPtr = fakeDesc;
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    };

    // Helper lambda to mock tensor attribute queries on a specific fake descriptor.
    // Each tensor needs: UID (scalar), name (count=0 for empty), data type (scalar),
    // dims (count + data), strides (count + data), is_virtual (scalar).
    auto expectTensorAttrs = [&](hipdnnBackendDescriptor_t fakeDesc,
                                 int64_t uid,
                                 const std::array<int64_t, 4>& dims,
                                 const std::array<int64_t, 4>& strides) {
        // UID query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
            .WillRepeatedly(DoAll(SetArgPointee<4>(int64_t{1}),
                                  Invoke([uid](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t*,
                                               void* arrayOfElements) {
                                      std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                                  }),
                                  Return(HIPDNN_STATUS_SUCCESS)));

        // Name count query (empty name: count = 0)
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

        // Data type query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto dt = HIPDNN_DATA_FLOAT;
                                std::memcpy(arrayOfElements, &dt, sizeof(hipdnnDataType_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Dims: count query then data query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([dims](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t*,
                                          void* arrayOfElements) {
                                std::memcpy(arrayOfElements, dims.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Strides: count query then data query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([strides](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                                std::memcpy(arrayOfElements, strides.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // is_virtual query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto val = false;
                                std::memcpy(arrayOfElements, &val, sizeof(bool));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    };

    // --- 2. Mock tensor descriptor retrieval for X, W, Y ---
    expectTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, xDescFake);
    expectTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wDescFake);
    expectTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, yDescFake);

    // --- 3. Mock tensor attribute queries for each tensor ---
    expectTensorAttrs(xDescFake, K_X_UID, K_X_DIMS, K_X_STRIDES);
    expectTensorAttrs(wDescFake, K_W_UID, K_W_DIMS, K_W_STRIDES);
    expectTensorAttrs(yDescFake, K_Y_UID, K_Y_DIMS, K_Y_STRIDES);

    // --- 4. Mock convolution parameters ---

    // pre_padding: count query then data query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                        Invoke([K_PADDING](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_PADDING.data(), 2 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // post_padding: count query then data query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                        Invoke([K_PADDING](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_PADDING.data(), 2 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // stride: count query then data query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                        Invoke([K_STRIDE](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t*,
                                          void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_STRIDE.data(), 2 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // dilation: count query then data query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                        Invoke([K_DILATION](hipdnnBackendDescriptor_t,
                                            hipdnnBackendAttributeName_t,
                                            hipdnnBackendAttributeType_t,
                                            int64_t,
                                            int64_t*,
                                            void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_DILATION.data(), 2 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // conv_mode: scalar query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto mode = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;
                            std::memcpy(arrayOfElements, &mode, sizeof(hipdnnConvolutionMode_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // compute_type: scalar query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto dt = HIPDNN_DATA_FLOAT;
                            std::memcpy(arrayOfElements, &dt, sizeof(hipdnnDataType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // --- 5. Mock operation name query ---
    // getDescriptorAttrString makes two calls: count query then data query
    const std::string opName = "test_conv_op";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(opName.size() + 1)),
                        Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(opName.size() + 1)),
                        Invoke([opName](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t,
                                        int64_t*,
                                        void* arrayOfElements) {
                            std::memcpy(arrayOfElements, opName.c_str(), opName.size() + 1);
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // --- 6. Mock destroy for the 3 tensor descriptors ---
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // --- Execute ---
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);

    // --- Verify ---
    EXPECT_TRUE(err.is_good()) << "Error: " << err.get_message();
    ASSERT_NE(node, nullptr);

    auto convNode = std::dynamic_pointer_cast<ConvolutionFpropNode>(node);
    ASSERT_NE(convNode, nullptr);

    // Verify tensors
    auto xTensor = convNode->attributes.get_x();
    auto wTensor = convNode->attributes.get_w();
    auto yTensor = convNode->attributes.get_y();
    ASSERT_NE(xTensor, nullptr);
    ASSERT_NE(wTensor, nullptr);
    ASSERT_NE(yTensor, nullptr);

    EXPECT_EQ(xTensor->get_uid(), K_X_UID);
    EXPECT_EQ(wTensor->get_uid(), K_W_UID);
    EXPECT_EQ(yTensor->get_uid(), K_Y_UID);

    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{K_X_DIMS.begin(), K_X_DIMS.end()}));
    EXPECT_EQ(xTensor->get_stride(),
              (std::vector<int64_t>{K_X_STRIDES.begin(), K_X_STRIDES.end()}));
    EXPECT_EQ(wTensor->get_dim(), (std::vector<int64_t>{K_W_DIMS.begin(), K_W_DIMS.end()}));
    EXPECT_EQ(wTensor->get_stride(),
              (std::vector<int64_t>{K_W_STRIDES.begin(), K_W_STRIDES.end()}));
    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{K_Y_DIMS.begin(), K_Y_DIMS.end()}));
    EXPECT_EQ(yTensor->get_stride(),
              (std::vector<int64_t>{K_Y_STRIDES.begin(), K_Y_STRIDES.end()}));

    EXPECT_EQ(xTensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(wTensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(yTensor->get_data_type(), DataType::FLOAT);

    EXPECT_FALSE(xTensor->get_is_virtual());
    EXPECT_FALSE(wTensor->get_is_virtual());
    EXPECT_FALSE(yTensor->get_is_virtual());

    // Verify conv params
    EXPECT_EQ(convNode->attributes.get_pre_padding(),
              (std::vector<int64_t>{K_PADDING.begin(), K_PADDING.end()}));
    EXPECT_EQ(convNode->attributes.get_post_padding(),
              (std::vector<int64_t>{K_PADDING.begin(), K_PADDING.end()}));
    EXPECT_EQ(convNode->attributes.get_stride(),
              (std::vector<int64_t>{K_STRIDE.begin(), K_STRIDE.end()}));
    EXPECT_EQ(convNode->attributes.get_dilation(),
              (std::vector<int64_t>{K_DILATION.begin(), K_DILATION.end()}));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.compute_data_type, DataType::FLOAT);

    // Verify operation name
    EXPECT_EQ(convNode->attributes.get_name(), "test_conv_op");

    // Verify tensors were registered in the tensor map
    EXPECT_EQ(tensorMap.size(), 3u);
    EXPECT_NE(tensorMap.find(K_X_UID), tensorMap.end());
    EXPECT_NE(tensorMap.find(K_W_UID), tensorMap.end());
    EXPECT_NE(tensorMap.find(K_Y_UID), tensorMap.end());
}

// ---------------------------------------------------------------------------
// unpackConvFpropOperation error-path tests
// ---------------------------------------------------------------------------

class TestUnpackConvFpropErrors : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    // Test constants matching the success-path test
    static constexpr int64_t K_X_UID = 10;
    static constexpr int64_t K_W_UID = 20;
    static constexpr int64_t K_Y_UID = 30;
    static constexpr std::array<int64_t, 4> K_X_DIMS = {1, 3, 32, 32};
    static constexpr std::array<int64_t, 4> K_X_STRIDES = {3072, 1024, 32, 1};
    static constexpr std::array<int64_t, 4> K_W_DIMS = {64, 3, 3, 3};
    static constexpr std::array<int64_t, 4> K_W_STRIDES = {27, 9, 3, 1};
    static constexpr std::array<int64_t, 4> K_Y_DIMS = {1, 64, 32, 32};
    static constexpr std::array<int64_t, 4> K_Y_STRIDES = {65536, 1024, 32, 1};
    static constexpr std::array<int64_t, 2> K_PADDING = {1, 1};
    static constexpr std::array<int64_t, 2> K_STRIDE = {1, 1};
    static constexpr std::array<int64_t, 2> K_DILATION = {1, 1};

    // Fake descriptor pointers for the 3 tensors
    int _xDescPlaceholder = 0;
    int _wDescPlaceholder = 0;
    int _yDescPlaceholder = 0;
    hipdnnBackendDescriptor_t _xDescFake
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_xDescPlaceholder);
    hipdnnBackendDescriptor_t _wDescFake
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_wDescPlaceholder);
    hipdnnBackendDescriptor_t _yDescFake
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_yDescPlaceholder);

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

    // Mocks the operation type query to return CONV_FORWARD
    void mockOperationTypeQuery()
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto value = HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD;
                                std::memcpy(arrayOfElements, &value, sizeof(hipdnnOperationType_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks backendGetAttribute for retrieving a tensor descriptor from the operation
    void mockTensorDescGet(hipdnnBackendAttributeName_t tensorAttrName,
                           hipdnnBackendDescriptor_t fakeDesc)
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(_, tensorAttrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([fakeDesc](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* arrayOfElements) {
                                auto descPtr
                                    = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                                *descPtr = fakeDesc;
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks all tensor attribute queries on a specific fake descriptor
    void mockTensorAttrs(hipdnnBackendDescriptor_t fakeDesc,
                         int64_t uid,
                         const std::array<int64_t, 4>& dims,
                         const std::array<int64_t, 4>& strides)
    {
        // UID query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
            .WillRepeatedly(DoAll(SetArgPointee<4>(int64_t{1}),
                                  Invoke([uid](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t*,
                                               void* arrayOfElements) {
                                      std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                                  }),
                                  Return(HIPDNN_STATUS_SUCCESS)));

        // Name count query (empty name: count = 0)
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

        // Data type query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto dt = HIPDNN_DATA_FLOAT;
                                std::memcpy(arrayOfElements, &dt, sizeof(hipdnnDataType_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Dims: count query then data query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([dims](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t*,
                                          void* arrayOfElements) {
                                std::memcpy(arrayOfElements, dims.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Strides: count query then data query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([strides](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                                std::memcpy(arrayOfElements, strides.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // is_virtual query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        fakeDesc, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto val = false;
                                std::memcpy(arrayOfElements, &val, sizeof(bool));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks all convolution vector parameter queries (pre_padding, post_padding, stride, dilation)
    void mockConvVectorParams()
    {
        // pre_padding: count query then data query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(arrayOfElements, K_PADDING.data(), 2 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // post_padding: count query then data query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(arrayOfElements, K_PADDING.data(), 2 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // stride: count query then data query
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(arrayOfElements, K_STRIDE.data(), 2 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // dilation: count query then data query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(
                                    arrayOfElements, K_DILATION.data(), 2 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks the conv_mode scalar query to return CROSS_CORRELATION
    void mockConvMode()
    {
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(
                _, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto mode = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;
                                std::memcpy(
                                    arrayOfElements, &mode, sizeof(hipdnnConvolutionMode_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks the compute type scalar query to return FLOAT
    void mockComputeType()
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto dt = HIPDNN_DATA_FLOAT;
                                std::memcpy(arrayOfElements, &dt, sizeof(hipdnnDataType_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Mocks the operation name string query
    void mockOperationName()
    {
        const std::string opName = "test_conv_op";
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(_, HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(opName.size() + 1)),
                            Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(opName.size() + 1)),
                            Invoke([opName](hipdnnBackendDescriptor_t,
                                            hipdnnBackendAttributeName_t,
                                            hipdnnBackendAttributeType_t,
                                            int64_t,
                                            int64_t*,
                                            void* arrayOfElements) {
                                std::memcpy(arrayOfElements, opName.c_str(), opName.size() + 1);
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Sets up all 3 tensor descriptor gets and their attribute queries
    void mockAllTensors()
    {
        mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, _xDescFake);
        mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, _wDescFake);
        mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, _yDescFake);
        mockTensorAttrs(_xDescFake, K_X_UID, K_X_DIMS, K_X_STRIDES);
        mockTensorAttrs(_wDescFake, K_W_UID, K_W_DIMS, K_W_STRIDES);
        mockTensorAttrs(_yDescFake, K_Y_UID, K_Y_DIMS, K_Y_STRIDES);
    }

    // Executes unpackOperation and returns the result
    static std::pair<std::shared_ptr<graph::INode>, Error> executeUnpack()
    {
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
        GraphAttributes graphAttrs;
        hipdnnBackendDescriptor_t desc = nullptr;
        return unpackOperation(desc, tensorMap, graphAttrs);
    }
};

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropWTensorFails)
{
    mockOperationTypeQuery();

    // X tensor succeeds
    mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, _xDescFake);
    mockTensorAttrs(_xDescFake, K_X_UID, K_X_DIMS, K_X_STRIDES);

    // W tensor descriptor retrieval fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_,
                                    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // X tensor descriptor is cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(1)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropYTensorFails)
{
    mockOperationTypeQuery();

    // X and W tensors succeed
    mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, _xDescFake);
    mockTensorDescGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, _wDescFake);
    mockTensorAttrs(_xDescFake, K_X_UID, K_X_DIMS, K_X_STRIDES);
    mockTensorAttrs(_wDescFake, K_W_UID, K_W_DIMS, K_W_STRIDES);

    // Y tensor descriptor retrieval fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_,
                                    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // X and W tensor descriptors are cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropConvModeFails)
{
    mockOperationTypeQuery();
    mockAllTensors();
    mockConvVectorParams();

    // Conv mode query fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // All 3 tensor descriptors are cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropInvalidConvMode)
{
    mockOperationTypeQuery();
    mockAllTensors();
    mockConvVectorParams();

    // Conv mode query succeeds but returns an invalid value
    auto invalidMode = static_cast<hipdnnConvolutionMode_t>(999);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([invalidMode](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                            std::memcpy(
                                arrayOfElements, &invalidMode, sizeof(hipdnnConvolutionMode_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // All 3 tensor descriptors are cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("999") != std::string::npos)
        << "Error should include the invalid mode value, got: " << err.get_message();
}

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropComputeTypeFails)
{
    mockOperationTypeQuery();
    mockAllTensors();
    mockConvVectorParams();
    mockConvMode();

    // Compute type query fails
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // All 3 tensor descriptors are cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

TEST_F(TestUnpackConvFpropErrors, UnpackConvFpropOperationNameFails)
{
    mockOperationTypeQuery();
    mockAllTensors();
    mockConvVectorParams();
    mockConvMode();
    mockComputeType();

    // Operation name count query fails with a non-NOT_SUPPORTED error
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // All 3 tensor descriptors are cleaned up via RAII
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto [node, err] = executeUnpack();
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

// ---------------------------------------------------------------------------
// createNodeForType tests
// ---------------------------------------------------------------------------

TEST(TestCreateNodeForType, CreatesConvFpropNode)
{
    GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto convNode = std::dynamic_pointer_cast<ConvolutionFpropNode>(node);
    EXPECT_NE(convNode, nullptr);
}

TEST(TestCreateNodeForType, ReturnsErrorForUnsupportedType)
{
    GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE, graphAttrs);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.get_message().find("Unsupported operation type") != std::string::npos);
    EXPECT_TRUE(err.get_message().find(
                    std::to_string(static_cast<int>(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE)))
                != std::string::npos)
        << "Error should include the unsupported type id, got: " << err.get_message();
}
