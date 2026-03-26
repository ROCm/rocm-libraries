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

        // Is-by-value query: return false (no scalar value set)
        EXPECT_CALL(
            *_mockBackend,
            backendGetAttribute(
                _fakeDesc, HIPDNN_ATTR_TENSOR_IS_BY_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, _, _))
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
    EXPECT_FALSE(tensor->get_pass_by_value());
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesRestoresPassByValue)
{
    static constexpr float K_SCALAR_VALUE = 1e-5f;

    // UID
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

    // Name (empty)
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data type (FLOAT)
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

    // Dims: scalar tensor has dims = {1}
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            int64_t dim = 1;
                            std::memcpy(arrayOfElements, &dim, sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Strides: scalar tensor has strides = {1}
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}), Return(HIPDNN_STATUS_SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            int64_t stride = 1;
                            std::memcpy(arrayOfElements, &stride, sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // is_virtual: false
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

    // IS_BY_VALUE: true
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_IS_BY_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto val = true;
                            std::memcpy(arrayOfElements, &val, sizeof(bool));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // TENSOR_VALUE: return the raw float bytes
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(sizeof(float))),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto val = K_SCALAR_VALUE;
                            std::memcpy(arrayOfElements, &val, sizeof(float));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->get_uid(), K_UID);
    EXPECT_EQ(tensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensor->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_EQ(tensor->get_stride(), (std::vector<int64_t>{1}));
    EXPECT_TRUE(tensor->get_pass_by_value());
    ASSERT_TRUE(tensor->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(tensor->get_pass_by_value<float>().value(), K_SCALAR_VALUE);
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesPassByValuePreserves4dDims)
{
    static constexpr float K_SCALAR_VALUE = 1e-5f;
    static constexpr std::array<int64_t, 4> K_SCALAR_DIMS = {1, 1, 1, 1};
    static constexpr std::array<int64_t, 4> K_SCALAR_STRIDES = {1, 1, 1, 1};

    // UID
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

    // Name (empty)
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data type (FLOAT)
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

    // Dims: 4D {1, 1, 1, 1}
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
                            std::memcpy(arrayOfElements, K_SCALAR_DIMS.data(), 4 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Strides: 4D {1, 1, 1, 1}
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
                            std::memcpy(
                                arrayOfElements, K_SCALAR_STRIDES.data(), 4 * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // is_virtual: false
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

    // IS_BY_VALUE: true
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_IS_BY_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto val = true;
                            std::memcpy(arrayOfElements, &val, sizeof(bool));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // TENSOR_VALUE: return the raw float bytes
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(sizeof(float))),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto val = K_SCALAR_VALUE;
                            std::memcpy(arrayOfElements, &val, sizeof(float));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->get_uid(), K_UID);
    EXPECT_EQ(tensor->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(tensor->get_pass_by_value());
    ASSERT_TRUE(tensor->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(tensor->get_pass_by_value<float>().value(), K_SCALAR_VALUE);
    // Verify dims and strides are preserved (not reset to {1} by set_value)
    EXPECT_EQ(tensor->get_dim(),
              (std::vector<int64_t>{K_SCALAR_DIMS.begin(), K_SCALAR_DIMS.end()}));
    EXPECT_EQ(tensor->get_stride(),
              (std::vector<int64_t>{K_SCALAR_STRIDES.begin(), K_SCALAR_STRIDES.end()}));
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

    // Is-by-value query: return false (no scalar value set)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_IS_BY_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, _, _))
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

TEST_F(TestUnpackAndRegisterTensor, UnpackAndRegisterTensorUIDFails)
{
    // Descriptor fetch succeeds
    expectDescriptorGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);

    // UID scalar query fails
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // Destroy for the RAII wrapper
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    std::shared_ptr<TensorAttributes> outTensor;
    hipdnnBackendDescriptor_t opDesc = nullptr;

    auto err = unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, outTensor, "conv X");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestUnpackAndRegisterTensor, UnpackAndRegisterTensorUnpackFails)
{
    // Descriptor fetch succeeds
    expectDescriptorGet(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);

    // UID scalar succeeds (new UID not in map)
    constexpr int64_t K_NEW_UID = 99;
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
                                  auto uid = K_NEW_UID;
                                  std::memcpy(arrayOfElements, &uid, sizeof(int64_t));
                              }),
                              Return(HIPDNN_STATUS_SUCCESS)));

    // Data type query fails during unpackTensorAttributes
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // Destroy for the RAII wrapper
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    std::shared_ptr<TensorAttributes> outTensor;
    hipdnnBackendDescriptor_t opDesc = nullptr;

    auto err = unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, outTensor, "conv X");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// getDescriptorAttrString tests
// ---------------------------------------------------------------------------

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrStringSuccess)
{
    const std::string kExpected = "test_tensor";

    // Count query returns string length + 1 (for null terminator)
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(kExpected.size() + 1)),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Data query returns the string
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, Ne(0), _, Ne(nullptr)))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(kExpected.size() + 1)),
                        Invoke([kExpected](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* arrayOfElements) {
                            std::memcpy(arrayOfElements, kExpected.c_str(), kExpected.size() + 1);
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::string value = "initial";
    auto err = getDescriptorAttrString(desc, HIPDNN_ATTR_TENSOR_NAME_EXT, value, "tensor name");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(value, kExpected);
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrStringNotSupported)
{
    // Count query returns NOT_SUPPORTED
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::string value = "initial";
    auto err = getDescriptorAttrString(desc, HIPDNN_ATTR_TENSOR_NAME_EXT, value, "tensor name");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(value.empty());
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrStringEmptyCount)
{
    // Count query succeeds but count=0
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    std::string value = "initial";
    auto err = getDescriptorAttrString(desc, HIPDNN_ATTR_TENSOR_NAME_EXT, value, "tensor name");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(value.empty());
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrStringCountFails)
{
    // Count query returns a non-NOT_SUPPORTED error
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    std::string value = "initial";
    auto err = getDescriptorAttrString(desc, HIPDNN_ATTR_TENSOR_NAME_EXT, value, "tensor name");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorUnpackHelpers, GetDescriptorAttrStringDataFails)
{
    // Count query succeeds with count > 0
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{10}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query fails
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_CHAR, 10, _, Ne(nullptr)))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    std::string value = "initial";
    auto err = getDescriptorAttrString(desc, HIPDNN_ATTR_TENSOR_NAME_EXT, value, "tensor name");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// getDescriptorAttrDescArray tests
// ---------------------------------------------------------------------------

class TestGetDescriptorAttrDescArray : public TestDescriptorUnpackHelpers
{
protected:
    // Fake descriptor pointers for the source and returned descriptors
    int _sourcePlaceholder = 0;
    hipdnnBackendDescriptor_t _fakeSource
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_sourcePlaceholder);

    int _desc1Placeholder = 1;
    hipdnnBackendDescriptor_t _fakeDesc1
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_desc1Placeholder);

    int _desc2Placeholder = 2;
    hipdnnBackendDescriptor_t _fakeDesc2
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_desc2Placeholder);
};

TEST_F(TestGetDescriptorAttrDescArray, GetDescriptorAttrDescArraySuccess)
{
    // Count query returns 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query returns 2 descriptors
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, _, Ne(nullptr)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}),
                        Invoke([this](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                            auto descs = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                            descs[0] = _fakeDesc1;
                            descs[1] = _fakeDesc2;
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // RAII destructors call backendDestroyDescriptor for each returned descriptor
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [result, err]
        = getDescriptorAttrDescArray(_fakeSource, HIPDNN_ATTR_TENSOR_DIMENSIONS, "test array");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].get(), _fakeDesc1);
    EXPECT_EQ(result[1].get(), _fakeDesc2);
}

TEST_F(TestGetDescriptorAttrDescArray, GetDescriptorAttrDescArrayCountFails)
{
    // Count query fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto [result, err]
        = getDescriptorAttrDescArray(_fakeSource, HIPDNN_ATTR_TENSOR_DIMENSIONS, "test array");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestGetDescriptorAttrDescArray, GetDescriptorAttrDescArrayZeroCount)
{
    // Count query returns 0
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    auto [result, err]
        = getDescriptorAttrDescArray(_fakeSource, HIPDNN_ATTR_TENSOR_DIMENSIONS, "test array");

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(result.empty());
}

TEST_F(TestGetDescriptorAttrDescArray, GetDescriptorAttrDescArrayDataFails)
{
    // Count query returns 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query fails, but returns partially-filled descriptors
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, _, Ne(nullptr)))
        .WillOnce(DoAll(Invoke([this](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                            auto descs = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                            descs[0] = _fakeDesc1;
                            descs[1] = _fakeDesc2;
                        }),
                        Return(HIPDNN_STATUS_INTERNAL_ERROR)));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    // The cleanup path destroys both descriptors
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [result, err]
        = getDescriptorAttrDescArray(_fakeSource, HIPDNN_ATTR_TENSOR_DIMENSIONS, "test array");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestGetDescriptorAttrDescArray, GetDescriptorAttrDescArrayCountMismatch)
{
    // Count query returns 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{2}), Return(HIPDNN_STATUS_SUCCESS)));

    // Data query succeeds but actualCount > count (mismatch)
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeSource, _, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, _, Ne(nullptr)))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{5}),
                        Invoke([this](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
                            auto descs = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
                            descs[0] = _fakeDesc1;
                            descs[1] = _fakeDesc2;
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // The cleanup path destroys both descriptors
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [result, err]
        = getDescriptorAttrDescArray(_fakeSource, HIPDNN_ATTR_TENSOR_DIMENSIONS, "test array");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(result.empty());
}

// ---------------------------------------------------------------------------
// Additional unpackTensorAttributes error-path tests
// ---------------------------------------------------------------------------

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesUIDFails)
{
    // UID scalar query fails
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesDataTypeFails)
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

    // Data type query fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesStridesFails)
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

    // Dims: count query then data query succeed
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

    // Strides count query fails
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestUnpackTensorAttributes, UnpackTensorAttributesInvalidDataType)
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

    // Data type scalar returns an unrecognized value
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
                            auto dt = static_cast<hipdnnDataType_t>(9999);
                            std::memcpy(arrayOfElements, &dt, sizeof(hipdnnDataType_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Dims: count query then data query succeed
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

    // Strides: count query then data query succeed
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

    // is_virtual scalar succeeds
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

    // Name count query (empty name)
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(_fakeDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, _, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    std::shared_ptr<TensorAttributes> tensor;
    auto err = unpackTensorAttributes(_fakeDesc, tensor);

    // fromHipdnnDataType returns an error for unknown data types
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// ---------------------------------------------------------------------------
// unpackGraphDataType tests
// ---------------------------------------------------------------------------

TEST_F(TestDescriptorUnpackHelpers, UnpackGraphDataTypeBackendFails)
{
    // getDescriptorAttrScalar for the compute data type returns error
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, HIPDNN_TYPE_DATA_TYPE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [dt, err] = unpackGraphDataType(desc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, "compute type");

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(dt, DataType::NOT_SET);
}
