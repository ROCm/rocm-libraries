// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

#include "hipdnn_backend.h"

// NOLINTBEGIN(modernize-avoid-c-arrays)
TEST(IntegrationGetErrorString, AllStatusCodes)
{
    std::vector<std::tuple<hipdnnStatus_t, std::string>> statusPairs
        = {{HIPDNN_STATUS_SUCCESS, "HIPDNN_STATUS_SUCCESS"},
           {HIPDNN_STATUS_NOT_INITIALIZED, "HIPDNN_STATUS_NOT_INITIALIZED"},
           {HIPDNN_STATUS_BAD_PARAM, "HIPDNN_STATUS_BAD_PARAM"},
           {HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "HIPDNN_STATUS_BAD_PARAM_NULL_POINTER"},
           {HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED, "HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED"},
           {HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND, "HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND"},
           {HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT, "HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT"},
           {HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH, "HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH"},
           {HIPDNN_STATUS_NOT_SUPPORTED, "HIPDNN_STATUS_NOT_SUPPORTED"},
           {HIPDNN_STATUS_INTERNAL_ERROR, "HIPDNN_STATUS_INTERNAL_ERROR"},
           {HIPDNN_STATUS_ALLOC_FAILED, "HIPDNN_STATUS_ALLOC_FAILED"},
           {HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
            "HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"},
           {HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
            "HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED"},
           {HIPDNN_STATUS_EXECUTION_FAILED, "HIPDNN_STATUS_EXECUTION_FAILED"},
           {static_cast<hipdnnStatus_t>(-999), "HIPDNN_STATUS_UNKNOWN"}};

    for(const auto& [status, text] : statusPairs)
    {
        const char* str = hipdnnGetErrorString(status);
        ASSERT_STREQ(str, text.c_str());
    }
}

TEST(IntegrationGetLastErrorString, ReturnLastError)
{
    char buffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnGetLastErrorString(buffer, sizeof(buffer));
    ASSERT_NE(std::string(buffer), "");
}

TEST(IntegrationGetLastErrorString, NullBufferOrZeroSize)
{
    // Should not crash or throw
    hipdnnGetLastErrorString(nullptr, 10);
    char buf[8];
    hipdnnGetLastErrorString(buf, 0);
}

TEST(IntegrationGetLastErrorString, BufferTruncationAndNullTermination)
{
    hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    char buffer[2];
    hipdnnGetLastErrorString(buffer, sizeof(buffer));

    ASSERT_EQ(buffer[sizeof(buffer) - 1], '\0');
    ASSERT_LT(strlen(buffer), sizeof(buffer));
    std::string bufferStr(buffer);
    ASSERT_NE(bufferStr, "");
}

TEST(IntegrationGetLastErrorString, PerThreadErrorIsolation)
{
    // Set error in main thread
    hipdnnDestroy(nullptr);
    char mainBuf[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(mainBuf, sizeof(mainBuf));

    std::string threadError;
    std::thread t([&threadError]() {
        char buf[HIPDNN_ERROR_STRING_MAX_LENGTH];
        hipdnnGetLastErrorString(buf, sizeof(buf));
        threadError = buf;
    });
    t.join();

    // Main thread error should be unchanged
    char mainBuf2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(mainBuf2, sizeof(mainBuf2));
    ASSERT_STREQ(mainBuf, mainBuf2);
    ASSERT_TRUE(threadError.empty());
}

TEST(IntegrationGetLastErrorString, BufferLargerThanMax)
{
    hipdnnDestroy(nullptr);
    char mainBuf[1028];
    hipdnnGetLastErrorString(mainBuf, sizeof(mainBuf));

    char mainBuf2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(mainBuf2, sizeof(mainBuf2));
    ASSERT_STREQ(mainBuf, mainBuf2);
}
// NOLINTEND(modernize-avoid-c-arrays)
