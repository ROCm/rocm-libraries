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
    const std::vector<std::tuple<hipdnnStatus_t, std::string>> statusPairs
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
    const hipdnnStatus_t status = hipdnnDestroy(nullptr);
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
    const hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    char buffer[2];
    hipdnnGetLastErrorString(buffer, sizeof(buffer));

    ASSERT_EQ(buffer[sizeof(buffer) - 1], '\0');
    ASSERT_LT(strlen(buffer), sizeof(buffer));
    const std::string bufferStr(buffer);
    ASSERT_NE(bufferStr, "");
}

TEST(IntegrationGetLastErrorString, PerThreadErrorIsolation)
{
    // Set error in main thread
    hipdnnDestroy(nullptr);
    char mainBuf[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(mainBuf, sizeof(mainBuf));

    std::string threadError;
    std::thread t([&threadError]() {
        char buf[HIPDNN_ERROR_STRING_MAX_LENGTH];
        hipdnnPeekLastErrorString_ext(buf, sizeof(buf));
        threadError = buf;
    });
    t.join();

    // Main thread error should be unchanged (using peek doesn't clear)
    char mainBuf2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(mainBuf2, sizeof(mainBuf2));
    ASSERT_STREQ(mainBuf, mainBuf2);
    ASSERT_TRUE(threadError.empty());
}

TEST(IntegrationGetLastErrorString, BufferLargerThanMax)
{
    hipdnnDestroy(nullptr);
    char mainBuf[1028];
    hipdnnPeekLastErrorString_ext(mainBuf, sizeof(mainBuf));

    char mainBuf2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(mainBuf2, sizeof(mainBuf2));
    ASSERT_STREQ(mainBuf, mainBuf2);
}

TEST(IntegrationGetLastErrorString, GetClearsError)
{
    // Generate an error
    const hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // First call should return the error
    char buffer1[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(buffer1, sizeof(buffer1));
    ASSERT_STRNE(buffer1, "");

    // Second call should return empty (error was cleared)
    char buffer2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(buffer2, sizeof(buffer2));
    ASSERT_STREQ(buffer2, "");
}

TEST(IntegrationGetLastErrorString, PeekDoesNotClearError)
{
    // Generate an error
    const hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // First peek should return the error
    char buffer1[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(buffer1, sizeof(buffer1));
    ASSERT_STRNE(buffer1, "");

    // Second peek should return the same error (not cleared)
    char buffer2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(buffer2, sizeof(buffer2));
    ASSERT_STREQ(buffer1, buffer2);

    // Third peek should still return the same error
    char buffer3[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(buffer3, sizeof(buffer3));
    ASSERT_STREQ(buffer1, buffer3);
}

TEST(IntegrationGetLastErrorString, PeekThenGetClears)
{
    // Generate an error
    hipdnnDestroy(nullptr);

    // Peek should return the error without clearing
    char peekBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(peekBuffer, sizeof(peekBuffer));
    ASSERT_STRNE(peekBuffer, "");

    // Get should return the same error and clear it
    char getBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnGetLastErrorString(getBuffer, sizeof(getBuffer));
    ASSERT_STREQ(peekBuffer, getBuffer);

    // Next peek should return empty (error was cleared by previous get)
    char emptyBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(emptyBuffer, sizeof(emptyBuffer));
    ASSERT_STREQ(emptyBuffer, "");
}

TEST(IntegrationGetLastErrorString, ClearingIsThreadLocal)
{
    // Set error in main thread
    hipdnnDestroy(nullptr);
    char mainBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(mainBuffer, sizeof(mainBuffer));
    ASSERT_STRNE(mainBuffer, "");

    // Worker thread generates and clears its own error
    std::thread t([]() {
        hipdnnDestroy(nullptr);
        char workerBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
        hipdnnGetLastErrorString(workerBuffer, sizeof(workerBuffer));
        ASSERT_STRNE(workerBuffer, "");

        // Error should be cleared after get
        char emptyBuffer[HIPDNN_ERROR_STRING_MAX_LENGTH];
        hipdnnGetLastErrorString(emptyBuffer, sizeof(emptyBuffer));
        ASSERT_STREQ(emptyBuffer, "");
    });
    t.join();

    // Main thread error should still be present (clearing is thread-local)
    char mainBuffer2[HIPDNN_ERROR_STRING_MAX_LENGTH];
    hipdnnPeekLastErrorString_ext(mainBuffer2, sizeof(mainBuffer2));
    ASSERT_STREQ(mainBuffer, mainBuffer2);
}
// NOLINTEND(modernize-avoid-c-arrays)
