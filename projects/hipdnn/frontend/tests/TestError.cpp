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
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::SHAPE_DEDUCTION_FAILED),
              "SHAPE_DEDUCTION_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::INVALID_TENSOR_NAME),
              "INVALID_TENSOR_NAME");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::INVALID_VARIANT_PACK),
              "INVALID_VARIANT_PACK");
    EXPECT_EQ(hipdnn_frontend::to_string(
                  hipdnn_frontend::ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED),
              "GRAPH_EXECUTION_PLAN_CREATION_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::GRAPH_EXECUTION_FAILED),
              "GRAPH_EXECUTION_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::HEURISTIC_QUERY_FAILED),
              "HEURISTIC_QUERY_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::UNSUPPORTED_GRAPH_FORMAT),
              "UNSUPPORTED_GRAPH_FORMAT");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::CUDA_API_FAILED),
              "CUDA_API_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::CUDNN_BACKEND_API_FAILED),
              "CUDNN_BACKEND_API_FAILED");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::INVALID_CUDA_DEVICE),
              "INVALID_CUDA_DEVICE");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::HANDLE_ERROR), "HANDLE_ERROR");
    EXPECT_EQ(hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::NVRTC_COMPILATION_FAILED),
              "NVRTC_COMPILATION_FAILED");
}

TEST(TestError, CudnnCompatCodesConstructAndCompare)
{
    for(auto code : {hipdnn_frontend::ErrorCode::SHAPE_DEDUCTION_FAILED,
                     hipdnn_frontend::ErrorCode::INVALID_TENSOR_NAME,
                     hipdnn_frontend::ErrorCode::INVALID_VARIANT_PACK,
                     hipdnn_frontend::ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                     hipdnn_frontend::ErrorCode::GRAPH_EXECUTION_FAILED,
                     hipdnn_frontend::ErrorCode::HEURISTIC_QUERY_FAILED,
                     hipdnn_frontend::ErrorCode::UNSUPPORTED_GRAPH_FORMAT,
                     hipdnn_frontend::ErrorCode::CUDA_API_FAILED,
                     hipdnn_frontend::ErrorCode::CUDNN_BACKEND_API_FAILED,
                     hipdnn_frontend::ErrorCode::INVALID_CUDA_DEVICE,
                     hipdnn_frontend::ErrorCode::HANDLE_ERROR,
                     hipdnn_frontend::ErrorCode::NVRTC_COMPILATION_FAILED})
    {
        const hipdnn_frontend::Error error(code, "msg");
        EXPECT_EQ(error.get_code(), code);
        EXPECT_TRUE(error.is_bad());
        EXPECT_TRUE(error == code);
        EXPECT_NE(hipdnn_frontend::to_string(code), "UNKNOWN_ERROR");
    }
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
