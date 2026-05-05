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
