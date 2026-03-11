// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

class TestDescriptorUnpackHelpers : public ::testing::Test
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

// ---------------------------------------------------------------------------
// getDescriptorAttrVec tests
// ---------------------------------------------------------------------------

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrVecSuccess)
{
    // Count query returns 3
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{3}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query returns {1, 2, 3}
    constexpr std::array<int64_t, 3> K_DATA = {1, 2, 3};
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 3, _, Ne(nullptr)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{3}),
                        Invoke([K_DATA](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t,
                                        int64_t*,
                                        void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_DATA.data(), 3 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::vector<int64_t> values;
    auto err = getDescriptorAttrVec(desc, HIPDNN_ATTR_TENSOR_DIMENSIONS, values, "test dims");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 2);
    EXPECT_EQ(values[2], 3);
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrVecZeroCount)
{
    // Count query returns 0
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::vector<int64_t> values;
    auto err = getDescriptorAttrVec(desc, HIPDNN_ATTR_TENSOR_DIMENSIONS, values, "test dims");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(values.empty());
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrVecNegativeCount)
{
    // Count query returns -1 (treated as <= 0 by the guard in production code)
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{-1}), Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::vector<int64_t> values;
    auto err = getDescriptorAttrVec(desc, HIPDNN_ATTR_TENSOR_DIMENSIONS, values, "test dims");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(values.empty());
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrVecCountFails)
{
    // Count query fails
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    std::vector<int64_t> values;
    auto err = getDescriptorAttrVec(desc, HIPDNN_ATTR_TENSOR_DIMENSIONS, values, "test dims");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrVecCountMismatch)
{
    // Count query returns 3
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{3}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query returns actualCount=5 (mismatches count=3)
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 3, _, Ne(nullptr)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{5}), Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::vector<int64_t> values;
    auto err = getDescriptorAttrVec(desc, HIPDNN_ATTR_TENSOR_DIMENSIONS, values, "test dims");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// getDescriptorAttrScalar tests
// ---------------------------------------------------------------------------

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrScalarSuccess)
{
    constexpr int64_t K_VALUE = 42;

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([K_VALUE](hipdnnBackendDescriptor_t,
                                         hipdnnBackendAttributeName_t,
                                         hipdnnBackendAttributeType_t,
                                         int64_t,
                                         int64_t*,
                                         void* arrayOfElements) {
                            std::memcpy(arrayOfElements, &K_VALUE, sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    int64_t value = 0;
    auto err = getDescriptorAttrScalar(
        desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, value, "test uid");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(value, 42);
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrScalarFails)
{
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    int64_t value = 0;
    auto err = getDescriptorAttrScalar(
        desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, value, "test uid");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// unpackTensorAttributes tests
// ---------------------------------------------------------------------------

class TestUnpackTensorAttributes : public TestDescriptorUnpackHelpers
{
protected:
    // Fake descriptor pointer for the tensor
    int _descPlaceholder = 0;
    hipdnnBackendDescriptor_t _fakeDesc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_descPlaceholder);

    static constexpr int64_t K_UID = 42;
    static constexpr std::array<int64_t, 4> K_DIMS = {1, 3, 32, 32};
    static constexpr std::array<int64_t, 4> K_STRIDES = {3072, 1024, 32, 1};

    void expectFullTensorMocks()
    {
        // UID scalar
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                auto uid = K_UID;
                                std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Name count query (empty name: count = 0)
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

        // Data type scalar
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
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
                        _fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(arrayOfElements, K_DIMS.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Strides: count query then data query
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                            Invoke([](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                                std::memcpy(arrayOfElements, K_STRIDES.data(), 4 * sizeof(int64_t));
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // is_virtual scalar
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(
                        _fakeDesc, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, _, _))
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
};

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesSuccess)
{
    expectFullTensorMocks();

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->get_uid(), K_UID);
    EXPECT_EQ(tensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensor->get_dim(), (std::vector<int64_t>{K_DIMS.begin(), K_DIMS.end()}));
    EXPECT_EQ(tensor->get_stride(), (std::vector<int64_t>{K_STRIDES.begin(), K_STRIDES.end()}));
    EXPECT_FALSE(tensor->get_is_virtual());
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesMissingDimsFails)
{
    // UID scalar succeeds
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto uid = K_UID;
                            std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Data type scalar succeeds
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
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

    // Dims count query fails
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// unpackAndRegisterTensor tests
// ---------------------------------------------------------------------------

class TestUnpackAndRegisterTensor : public TestUnpackTensorAttributes
{
protected:
    void expectDescriptorGet(hipdnnBackendAttributeName_t tensorAttrName)
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(_, tensorAttrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _, _))
            .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                            Invoke([this](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t*,
                                          void* arrayOfElements) {
                                auto descPtr
                                    = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                                *descPtr = _fakeDesc;
                            }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }
};

TEST_F(TestUnpackAndRegisterTensor, UnpackAndRegisterTensorNewTensor)
{
    expectDescriptorGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);

    // UID query is called twice: once by unpackAndRegisterTensor to check the map,
    // and once by unpackTensorAttributes.
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
        .WillRepeatedly(DoAll(SetArgPointee<4>(int64_t{1}),
                              Invoke([](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t,
                                        int64_t*,
                                        void* arrayOfElements) {
                                  auto uid = K_UID;
                                  std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                              }),
                              Return(HIPDNN_STATUS_SUCCESS)));

    // Name count query (empty name)
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data type
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
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

    // Dims
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_DIMS.data(), 4 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Strides
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{4}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            std::memcpy(arrayOfElements, K_STRIDES.data(), 4 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // is_virtual
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, _, _))
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

    // Destroy for the RAII wrapper
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    std::shared_ptr<TensorAttributes> outTensor;
    hipdnnBackendDescriptor_t opDesc = nullptr;

    auto err = unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, outTensor, "conv X");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(outTensor, nullptr);
    EXPECT_EQ(outTensor->get_uid(), K_UID);
    EXPECT_EQ(tensorMap.size(), 1u);
    EXPECT_EQ(tensorMap[K_UID], outTensor);
}

TEST_F(TestUnpackAndRegisterTensor, UnpackAndRegisterTensorExistingUid)
{
    // Pre-populate the tensor map
    auto existingTensor = std::make_shared<TensorAttributes>();
    existingTensor->set_uid(K_UID)
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1});

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[K_UID] = existingTensor;

    // Mock getting the tensor descriptor from the operation
    expectDescriptorGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);

    // Mock reading UID from the tensor descriptor
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto uid = K_UID;
                            std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Destroy for the RAII wrapper
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    std::shared_ptr<TensorAttributes> outTensor;
    hipdnnBackendDescriptor_t opDesc = nullptr;

    auto err = unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, outTensor, "conv X");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(outTensor, nullptr);
    EXPECT_EQ(outTensor, existingTensor);
    EXPECT_EQ(tensorMap.size(), 1u);
}

TEST_F(TestUnpackAndRegisterTensor, UnpackAndRegisterTensorNullDescFails)
{
    // Mock getting the tensor descriptor from the operation, returning null
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_,
                                    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto descPtr = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                            *descPtr = nullptr;
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    std::shared_ptr<TensorAttributes> outTensor;
    hipdnnBackendDescriptor_t opDesc = nullptr;

    auto err = unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, outTensor, "conv X");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("Null") != std::string::npos
                || err.get_message().find("null") != std::string::npos);
}
