// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LastErrorManager.hpp"

#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(TestLastErrorManager, StaticSetLastError)
{
    // Test setting a success status
    hipdnnStatus_t status
        = LastErrorManager::setLastError(HIPDNN_STATUS_SUCCESS, "Operation successful");
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Test setting a valid error
    const std::string errorMessage = "An error occurred";
    status = LastErrorManager::setLastError(HIPDNN_STATUS_NOT_SUPPORTED, errorMessage.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(LastErrorManager::getLastError(), errorMessage.c_str());
}

TEST(TestLastErrorManager, ErrorCStringMessagePerThread)
{
    const std::string mainError = "Main thread error";
    const std::string workerError = "Worker thread error";
    LastErrorManager::setLastError(HIPDNN_STATUS_BAD_PARAM, mainError.c_str());

    std::string threadError;
    std::thread t([&threadError, workerError]() {
        LastErrorManager::setLastError(HIPDNN_STATUS_NOT_SUPPORTED, workerError.c_str());
        threadError = LastErrorManager::getLastError();
    });
    t.join();

    EXPECT_EQ(LastErrorManager::getLastError(), mainError);
    EXPECT_EQ(threadError, workerError);
}

TEST(TestLastErrorManager, ErrorSTDStringMessagePerThread)
{
    const std::string mainError = "Main thread error";
    const std::string workerError = "Worker thread error";
    LastErrorManager::setLastError(HIPDNN_STATUS_BAD_PARAM, mainError);

    std::string threadError;
    std::thread t([&threadError, workerError]() {
        LastErrorManager::setLastError(HIPDNN_STATUS_NOT_SUPPORTED, workerError);
        threadError = LastErrorManager::getLastError();
    });
    t.join();

    EXPECT_EQ(LastErrorManager::getLastError(), mainError);
    EXPECT_EQ(threadError, workerError);
}

TEST(TestLastErrorManager, SetSuccessSTDStringDoesNotSetErrorMessage)
{
    const std::string errorMessage = "This message should not be set";
    const hipdnnStatus_t status
        = LastErrorManager::setLastError(HIPDNN_STATUS_SUCCESS, errorMessage);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(LastErrorManager::getLastError(), errorMessage);
}

TEST(TestLastErrorManager, SetSuccessCStringDoesNotSetErrorMessage)
{
    const std::string errorMessage = "This message should not be set";
    const hipdnnStatus_t status
        = LastErrorManager::setLastError(HIPDNN_STATUS_SUCCESS, errorMessage.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(LastErrorManager::getLastError(), errorMessage);
}

TEST(TestLastErrorManager, ClearLastErrorEmptiesBuffer)
{
    // Set an error
    const std::string errorMessage = "Test error";
    LastErrorManager::setLastError(HIPDNN_STATUS_BAD_PARAM, errorMessage);
    EXPECT_STREQ(LastErrorManager::getLastError(), errorMessage.c_str());

    // Clear the error
    LastErrorManager::clearLastError();

    // Error should now be empty
    EXPECT_STREQ(LastErrorManager::getLastError(), "");
}

TEST(TestLastErrorManager, ClearLastErrorIsThreadLocal)
{
    const std::string mainError = "Main thread error";
    const std::string workerError = "Worker thread error";

    // Set error in main thread
    LastErrorManager::setLastError(HIPDNN_STATUS_BAD_PARAM, mainError);

    // Set and clear error in worker thread
    std::thread t([workerError]() {
        LastErrorManager::setLastError(HIPDNN_STATUS_NOT_SUPPORTED, workerError);
        EXPECT_STREQ(LastErrorManager::getLastError(), workerError.c_str());

        // Clear in worker thread
        LastErrorManager::clearLastError();
        EXPECT_STREQ(LastErrorManager::getLastError(), "");
    });
    t.join();

    // Main thread error should still be present (clearing is thread-local)
    EXPECT_STREQ(LastErrorManager::getLastError(), mainError.c_str());
}

TEST(TestLastErrorManager, MultipleClears)
{
    // Set an error
    LastErrorManager::setLastError(HIPDNN_STATUS_BAD_PARAM, "Error message");
    EXPECT_STRNE(LastErrorManager::getLastError(), "");

    // Clear it
    LastErrorManager::clearLastError();
    EXPECT_STREQ(LastErrorManager::getLastError(), "");

    // Clearing again should be safe
    LastErrorManager::clearLastError();
    EXPECT_STREQ(LastErrorManager::getLastError(), "");
}
