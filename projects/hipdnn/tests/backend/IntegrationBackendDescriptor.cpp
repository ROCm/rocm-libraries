// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <array>
#include <gtest/gtest.h>
#include <test_plugins/TestPluginConstants.hpp>

class IntegrationBackendDescriptor : public ::testing::Test
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
};

TEST_F(IntegrationBackendDescriptor, CreateAndDestroy)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationBackendDescriptor, CreateWithNullptr)
{
    const hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, WillNotCreateDescriptorIfTypeNotSupported)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    const hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE_EXT, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationBackendDescriptor, WontDestroyDescriptorIfNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    const hipdnnStatus_t status = hipdnnBackendDestroyDescriptor(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, Finalize)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    const hipdnnStatus_t status = hipdnnBackendFinalize(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, GetAttributeWithNullDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    const hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    const int64_t requestedElementCount = 0;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    const hipdnnStatus_t status = hipdnnBackendGetAttribute(descriptor,
                                                            attributeName,
                                                            attributeType,
                                                            requestedElementCount,
                                                            &elementCount,
                                                            arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, SetAttributeWithNullDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    const hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    const int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    const hipdnnStatus_t status = hipdnnBackendSetAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, GetAttributeRejectsNegativeRequestedCount)
{
    int64_t elementCount = 42;

    // An invalid descriptor is reported before the count is looked at.
    EXPECT_EQ(hipdnnBackendGetAttribute(nullptr,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        -1,
                                        &elementCount,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor),
              HIPDNN_STATUS_SUCCESS);

    // A null array is a count query for this attribute, so the descriptor alone
    // would answer it; a negative count is still refused.
    EXPECT_EQ(hipdnnBackendGetAttribute(descriptor,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        -1,
                                        &elementCount,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(elementCount, 42);

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationBackendDescriptor, SetAttributeRejectsNegativeElementCount)
{
    // An invalid descriptor is reported before the count is looked at.
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  nullptr, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, -1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &descriptor),
              HIPDNN_STATUS_SUCCESS);

    // The descriptor alone would report the null array first; the negative count
    // is refused before the descriptor sees the call.
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  descriptor, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, -1, nullptr),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
}
