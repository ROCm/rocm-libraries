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
    hipdnnStatus_t const status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, WillNotCreateDescriptorIfTypeNotSupported)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t const status = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationBackendDescriptor, WontDestroyDescriptorIfNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t const status = hipdnnBackendDestroyDescriptor(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, Finalize)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t const status = hipdnnBackendFinalize(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendDescriptor, GetAttributeWithNullDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t const attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t const attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t const requestedElementCount = 0;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t const status = hipdnnBackendGetAttribute(descriptor,
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
    hipdnnBackendAttributeName_t const attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t const attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t const elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t const status = hipdnnBackendSetAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}
