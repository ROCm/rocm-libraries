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
    std::string errorMessage = "An error occurred";
    status = LastErrorManager::setLastError(HIPDNN_STATUS_NOT_SUPPORTED, errorMessage.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(LastErrorManager::getLastError(), errorMessage.c_str());
}

TEST(TestLastErrorManager, ErrorCStringMessagePerThread)
{
    std::string mainError = "Main thread error";
    std::string workerError = "Worker thread error";
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
    std::string mainError = "Main thread error";
    std::string workerError = "Worker thread error";
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
    std::string errorMessage = "This message should not be set";
    hipdnnStatus_t status = LastErrorManager::setLastError(HIPDNN_STATUS_SUCCESS, errorMessage);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(LastErrorManager::getLastError(), errorMessage);
}

TEST(TestLastErrorManager, SetSuccessCStringDoesNotSetErrorMessage)
{
    std::string errorMessage = "This message should not be set";
    hipdnnStatus_t status
        = LastErrorManager::setLastError(HIPDNN_STATUS_SUCCESS, errorMessage.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(LastErrorManager::getLastError(), errorMessage);
}
