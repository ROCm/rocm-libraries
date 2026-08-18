// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_backend.h"

#include <gtest/gtest.h>

TEST(TestExecutionPlanSerializationApi, GetSerializedExecutionPlanRejectsNullDescriptor)
{
    size_t planByteSize = 0;

    EXPECT_EQ(hipdnnBackendGetSerializedExecutionPlan_ext(nullptr, 0, &planByteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestExecutionPlanSerializationApi, CreateAndDeserializeExecutionPlanRejectsNullHandle)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const uint8_t serializedPlan = 0;

    EXPECT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  nullptr, &descriptor, &serializedPlan, sizeof(serializedPlan)),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestExecutionPlanSerializationApi, CreateAndDeserializeExecutionPlanRejectsNullDescriptor)
{
    const auto handle = reinterpret_cast<hipdnnHandle_t>(0x1);
    const uint8_t serializedPlan = 0;

    EXPECT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  handle, nullptr, &serializedPlan, sizeof(serializedPlan)),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestExecutionPlanSerializationApi, GetSerializedBinaryGraphAndPlanRejectsNullGraphDescriptor)
{
    size_t blobByteSize = 0;

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  nullptr, nullptr, 0, &blobByteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestExecutionPlanSerializationApi, GetSerializedBinaryGraphAndPlanRejectsNullSize)
{
    // A real (valid) graph descriptor is required: throwIfInvalidDescriptor runs
    // before the null-size check and dereferences the descriptor.
    hipdnnBackendDescriptor_t graphDescriptor = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graphDescriptor),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(graphDescriptor, nullptr);

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(
                  graphDescriptor, nullptr, 0, nullptr, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(graphDescriptor), HIPDNN_STATUS_SUCCESS);
}

TEST(TestExecutionPlanSerializationApi, GetSerializedBinaryContentsRejectsNullBlob)
{
    int contentFlags = 0;

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(nullptr, 0, &contentFlags),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestExecutionPlanSerializationApi, GetSerializedBinaryContentsRejectsNullFlags)
{
    const uint8_t someByte = 0;

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryContents_ext(&someByte, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}
