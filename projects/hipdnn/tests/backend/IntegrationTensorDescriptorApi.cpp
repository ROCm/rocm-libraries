// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendTestHelpers.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <vector>

using namespace backend_test;

namespace
{

constexpr int64_t K_TENSOR_UID = 1;
constexpr std::array<int64_t, 4> K_TENSOR_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_STRIDES = {3072, 1024, 32, 1};

constexpr std::array<int64_t, 4> K_SCALAR_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SCALAR_STRIDES = {1, 1, 1, 1};
constexpr float K_SCALAR_VALUE = 2.5f;

constexpr std::array<int64_t, 4> K_SMALL_TENSOR_DIMS = {1, 3, 4, 4};
constexpr std::array<int64_t, 4> K_SMALL_TENSOR_STRIDES = {48, 16, 4, 1};

// Matches the default isVirtual parameter in setAllTensorAttributes
constexpr bool K_IS_VIRTUAL = false;

} // namespace

class IntegrationTensorDescriptorApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override {}

    static hipdnnBackendDescriptor_t createTensorDesc()
    {
        hipdnnBackendDescriptor_t desc = nullptr;
        auto status = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &desc);
        EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
        return desc;
    }

    /// Set up a scalar tensor descriptor with the given C API data type.
    static void setScalarTensorAttributes(hipdnnBackendDescriptor_t desc,
                                          hipdnnDataType_t apiDataType)
    {
        setAllTensorAttributes(desc,
                               K_TENSOR_UID,
                               "scalar",
                               toVec(K_SCALAR_DIMS),
                               toVec(K_SCALAR_STRIDES),
                               false,
                               apiDataType);
    }
};

TEST_F(IntegrationTensorDescriptorApi, SetAndGetAllAttributes)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    auto dims = toVec(K_TENSOR_DIMS);
    auto strides = toVec(K_TENSOR_STRIDES);

    ASSERT_NO_FATAL_FAILURE(
        setAllTensorAttributes(desc, K_TENSOR_UID, "test_tensor", dims, strides));

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Read back and verify all attributes
    int64_t elementCount = 0;

    int64_t gotUid = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &elementCount, &gotUid),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotUid, K_TENSOR_UID);

    std::array<char, 256> gotName{};
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_NAME_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(gotName.size()),
                                        &elementCount,
                                        gotName.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_STREQ(gotName.data(), "test_tensor");

    hipdnnDataType_t gotDataType = HIPDNN_DATA_FLOAT;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_DATA_TYPE,
                                        HIPDNN_TYPE_DATA_TYPE,
                                        1,
                                        &elementCount,
                                        &gotDataType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotDataType, HIPDNN_DATA_FLOAT);

    std::vector<int64_t> gotDims(4);
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(gotDims.size()),
                                        &elementCount,
                                        gotDims.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotDims, dims);

    std::vector<int64_t> gotStrides(4);
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(gotStrides.size()),
                                        &elementCount,
                                        gotStrides.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotStrides, strides);

    bool gotIsVirtual = true;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_IS_VIRTUAL,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        &elementCount,
                                        &gotIsVirtual),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotIsVirtual, K_IS_VIRTUAL);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValue)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    float tensorValue = K_SCALAR_VALUE;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    float gotValue = 0.0f;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(gotValue, K_SCALAR_VALUE);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(float)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueDouble)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_DOUBLE));

    double tensorValue = 2.718281828;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(double)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    double gotValue = 0.0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(double)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(gotValue, 2.718281828);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(double)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueInt32)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_INT32));

    int32_t tensorValue = 42;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(int32_t)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    int32_t gotValue = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(int32_t)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotValue, 42);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(int32_t)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueHalf)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_HALF));

    auto tensorValue = hipdnn_data_sdk::types::half(1.5f);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(tensorValue)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    hipdnn_data_sdk::types::half gotValue(0.0f);
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(gotValue)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(static_cast<float>(gotValue), 1.5f);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(gotValue)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueBFloat16)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_BFLOAT16));

    auto tensorValue = hipdnn_data_sdk::types::bfloat16(1.5f);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(tensorValue)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    hipdnn_data_sdk::types::bfloat16 gotValue(0.0f);
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(gotValue)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(static_cast<float>(gotValue), 1.5f);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(gotValue)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueTypeMismatchFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    float tensorValue = 1.0f;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    double gotValue = 0.0;
    EXPECT_NE(
        hipdnnBackendGetAttribute(
            desc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_DOUBLE, 1, &elementCount, &gotValue),
        HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueUnsupportedTypeFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    bool tensorValue = true;
    EXPECT_NE(hipdnnBackendSetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueWithoutDataTypeFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    float tensorValue = 1.0f;
    EXPECT_NE(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueUint8)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_UINT8));

    uint8_t tensorValue = 200;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(uint8_t)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    uint8_t gotValue = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(uint8_t)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotValue, 200);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(uint8_t)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueInt8)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_INT8));

    int8_t tensorValue = -42;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(int8_t)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    int8_t gotValue = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(int8_t)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotValue, -42);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(int8_t)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueFp8E4M3)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FP8_E4M3));

    auto tensorValue = hipdnn_data_sdk::types::fp8_e4m3(1.5f);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(tensorValue)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    hipdnn_data_sdk::types::fp8_e4m3 gotValue;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(gotValue)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotValue.data, tensorValue.data);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(gotValue)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetAndGetTensorValueFp8E5M2)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FP8_E5M2));

    auto tensorValue = hipdnn_data_sdk::types::fp8_e5m2(1.5f);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(tensorValue)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    hipdnn_data_sdk::types::fp8_e5m2 gotValue;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(gotValue)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gotValue.data, tensorValue.data);
    EXPECT_EQ(elementCount, static_cast<int64_t>(sizeof(gotValue)));

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueOverwritesPrevious)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    float val1 = 1.0f;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &val1),
              HIPDNN_STATUS_SUCCESS);

    float val2 = 2.0f;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &val2),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    float gotValue = 0.0f;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(gotValue, 2.0f);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueCopiesData)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    {
        float val = 3.14f;
        ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                            HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                            HIPDNN_TYPE_CHAR,
                                            static_cast<int64_t>(sizeof(float)),
                                            &val),
                  HIPDNN_STATUS_SUCCESS);
    }
    // val is out of scope — descriptor must have its own copy

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    float gotValue = 0.0f;
    EXPECT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(gotValue, 3.14f);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, SetTensorValueWrongElementCountFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    float tensorValue = 1.0f;
    EXPECT_NE(hipdnnBackendSetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, 2, &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, GetTensorValueNotSetFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    float gotValue = 0.0f;
    EXPECT_NE(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &elementCount,
                                        &gotValue),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, GetTensorValueWrongAttributeTypeFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setScalarTensorAttributes(desc, HIPDNN_DATA_FLOAT));

    float tensorValue = 1.0f;
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(sizeof(float)),
                                        &tensorValue),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    float gotValue = 0.0f;
    EXPECT_NE(
        hipdnnBackendGetAttribute(
            desc, HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &gotValue),
        HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, GetAttributeBeforeFinalizeFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    ASSERT_NO_FATAL_FAILURE(setAllTensorAttributes(desc,
                                                   K_TENSOR_UID,
                                                   "unfinalized",
                                                   toVec(K_SMALL_TENSOR_DIMS),
                                                   toVec(K_SMALL_TENSOR_STRIDES)));

    int64_t elementCount = 0;
    int64_t gotUid = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &elementCount, &gotUid),
              HIPDNN_STATUS_NOT_INITIALIZED);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationTensorDescriptorApi, FinalizeWithoutRequiredAttributesFails)
{
    auto desc = createTensorDesc();
    ASSERT_NE(desc, nullptr);

    int64_t uid = K_TENSOR_UID;
    ASSERT_EQ(
        hipdnnBackendSetAttribute(desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid),
        HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(desc);
}
