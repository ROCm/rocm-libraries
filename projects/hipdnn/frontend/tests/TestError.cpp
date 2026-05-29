// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>

TEST(TestError, DefaultConstructor)
{
    const hipdnn_frontend::Error error;
    EXPECT_EQ(error.get_code(), hipdnn_frontend::ErrorCode::OK);
    EXPECT_TRUE(error.is_good());
    EXPECT_FALSE(error.is_bad());
    EXPECT_EQ(error.get_message(), "");
}

TEST(TestError, ParameterizedConstructor)
{
    const hipdnn_frontend::Error error(hipdnn_frontend::ErrorCode::INVALID_VALUE,
                                       "Invalid value provided");
    EXPECT_EQ(error.get_code(), hipdnn_frontend::ErrorCode::INVALID_VALUE);
    EXPECT_FALSE(error.is_good());
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_message(), "Invalid value provided");
}

TEST(TestError, EqualityOperators)
{
    const hipdnn_frontend::Error error1(hipdnn_frontend::ErrorCode::INVALID_VALUE, "Error 1");
    const hipdnn_frontend::Error error2(hipdnn_frontend::ErrorCode::INVALID_VALUE, "Error 2");
    const hipdnn_frontend::Error error3(hipdnn_frontend::ErrorCode::ATTRIBUTE_NOT_SET, "Error 3");

    EXPECT_TRUE(error1 == error2);
    EXPECT_FALSE(error1 == error3);
    EXPECT_TRUE(error1 != error3);
    EXPECT_FALSE(error1 != error2);
}

TEST(TestError, CodeEqualityOperators)
{
    const hipdnn_frontend::Error error(hipdnn_frontend::ErrorCode::INVALID_VALUE,
                                       "Invalid value provided");

    EXPECT_TRUE(error == hipdnn_frontend::ErrorCode::INVALID_VALUE);
    EXPECT_FALSE(error == hipdnn_frontend::ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error != hipdnn_frontend::ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_FALSE(error != hipdnn_frontend::ErrorCode::INVALID_VALUE);
}

TEST(TestError, ToStringReturnsCorrectStringForAllErrorCodes)
{
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::OK), "OK");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::INVALID_VALUE),
              "INVALID_VALUE");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR),
              "HIPDNN_BACKEND_ERROR");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::ATTRIBUTE_NOT_SET),
              "ATTRIBUTE_NOT_SET");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED),
              "GRAPH_NOT_SUPPORTED");
}

TEST(TestError, ToStringReturnsUnknownForInvalidCode)
{
    // Cast an out-of-range value to ErrorCode to verify the default case
    auto invalidCode = static_cast<hipdnn_frontend::ErrorCode>(999);
    EXPECT_EQ(hipdnn_frontend::to_string(invalidCode), "UNKNOWN_ERROR");
}

TEST(TestError, CheckHipdnnErrorMacro)
{
    auto successFunction
        = []() -> hipdnn_frontend::Error { return {hipdnn_frontend::ErrorCode::OK, "Success"}; };

    auto failureFunction = []() -> hipdnn_frontend::Error {
        return {hipdnn_frontend::ErrorCode::INVALID_VALUE, "Failure"};
    };

    auto testFunction = [&]() -> hipdnn_frontend::Error {
        HIPDNN_CHECK_ERROR(successFunction());
        HIPDNN_CHECK_ERROR(failureFunction());
        return {hipdnn_frontend::ErrorCode::OK, "Should not reach here"};
    };

    const hipdnn_frontend::Error result = testFunction();
    EXPECT_EQ(result.get_code(), hipdnn_frontend::ErrorCode::INVALID_VALUE);
    EXPECT_EQ(result.get_message(), "Failure");
}
