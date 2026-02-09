/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>

#include <data_types.hpp>
#include <data_types_impl.hpp>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <logger.hpp>
#include <util.hpp>

TEST(stridesFromLengthsTest, UtilTest)
{
    std::vector<int> input = {3, 4, 5};

    std::vector<int> expected = {20, 5, 1};
    auto             output   = hiptensor::stridesFromLengths(input, false); // col_major == false
    EXPECT_EQ(output, expected);

    expected = {1, 3, 12};
    output   = hiptensor::stridesFromLengths(input, true); // col_major == true
    EXPECT_EQ(output, expected);

    input    = {};
    expected = {};
    output   = hiptensor::stridesFromLengths(input);
    EXPECT_EQ(output, expected);
}

TEST(CheckApiParamsTest, UtilTest)
{
    using hiptensor::Logger;
    auto&             logger      = Logger::instance();
    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, nullptr);
    EXPECT_EQ(checkResult, HIPTENSOR_STATUS_NOT_INITIALIZED);
}

TEST(hiptensorGetVersionTest, UtilTest)
{
    EXPECT_EQ(hiptensorGetVersion(), 2002000);
}

TEST(logLevelToStringTest, UtilTest)
{
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_OFF).c_str(),
                 "HIPTENSOR_LOG_LEVEL_OFF");
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_ERROR).c_str(),
                 "HIPTENSOR_LOG_LEVEL_ERROR");
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_PERF_TRACE).c_str(),
                 "HIPTENSOR_LOG_LEVEL_PERF_TRACE");
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_PERF_HINT).c_str(),
                 "HIPTENSOR_LOG_LEVEL_PERF_HINT");
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE).c_str(),
                 "HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE");
    EXPECT_STREQ(hiptensor::logLevelToString(HIPTENSOR_LOG_LEVEL_API_TRACE).c_str(),
                 "HIPTENSOR_LOG_LEVEL_API_TRACE");
}

TEST(opTypeToStringTest, UtilTest)
{
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_IDENTITY).c_str(), "HIPTENSOR_OP_IDENTITY");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_SQRT).c_str(), "HIPTENSOR_OP_SQRT");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_RELU).c_str(), "HIPTENSOR_OP_RELU");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_CONJ).c_str(), "HIPTENSOR_OP_CONJ");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_RCP).c_str(), "HIPTENSOR_OP_RCP");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_SIGMOID).c_str(), "HIPTENSOR_OP_SIGMOID");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_TANH).c_str(), "HIPTENSOR_OP_TANH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_EXP).c_str(), "HIPTENSOR_OP_EXP");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_LOG).c_str(), "HIPTENSOR_OP_LOG");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ABS).c_str(), "HIPTENSOR_OP_ABS");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_NEG).c_str(), "HIPTENSOR_OP_NEG");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_SIN).c_str(), "HIPTENSOR_OP_SIN");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_COS).c_str(), "HIPTENSOR_OP_COS");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_TAN).c_str(), "HIPTENSOR_OP_TAN");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_SINH).c_str(), "HIPTENSOR_OP_SINH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_COSH).c_str(), "HIPTENSOR_OP_COSH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ASIN).c_str(), "HIPTENSOR_OP_ASIN");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ACOS).c_str(), "HIPTENSOR_OP_ACOS");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ATAN).c_str(), "HIPTENSOR_OP_ATAN");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ASINH).c_str(), "HIPTENSOR_OP_ASINH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ACOSH).c_str(), "HIPTENSOR_OP_ACOSH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ATANH).c_str(), "HIPTENSOR_OP_ATANH");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_CEIL).c_str(), "HIPTENSOR_OP_CEIL");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_FLOOR).c_str(), "HIPTENSOR_OP_FLOOR");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_ADD).c_str(), "HIPTENSOR_OP_ADD");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_MUL).c_str(), "HIPTENSOR_OP_MUL");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_MAX).c_str(), "HIPTENSOR_OP_MAX");
    EXPECT_STREQ(hiptensor::opTypeToString(HIPTENSOR_OP_MIN).c_str(), "HIPTENSOR_OP_MIN");
}

TEST(convertToHipTensorDataTypeTypeToStringTest, UtilTest)
{
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_16BF), HIPTENSOR_R_16BF);
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_16F), HIPTENSOR_R_16F);
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_32F), HIPTENSOR_R_32F);
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_64F), HIPTENSOR_R_64F);
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_C32F), HIPTENSOR_C_32F);
    EXPECT_EQ(hiptensor::convertToHipTensorDataType(HIPTENSOR_COMPUTE_DESC_C64F), HIPTENSOR_C_64F);
}

TEST(algoTypeToStringTest, UtilTest)
{
    EXPECT_STREQ(hiptensor::algoTypeToString(HIPTENSOR_ALGO_ACTOR_CRITIC).c_str(),
                 "HIPTENSOR_ALGO_ACTOR_CRITIC");
    EXPECT_STREQ(hiptensor::algoTypeToString(HIPTENSOR_ALGO_DEFAULT).c_str(),
                 "HIPTENSOR_ALGO_DEFAULT");
    EXPECT_STREQ(hiptensor::algoTypeToString(HIPTENSOR_ALGO_DEFAULT_PATIENT).c_str(),
                 "HIPTENSOR_ALGO_DEFAULT_PATIENT");
}

TEST(workSizePrefToStringTest, UtilTest)
{
    EXPECT_STREQ(hiptensor::workSizePrefToString(HIPTENSOR_WORKSPACE_MIN).c_str(),
                 "HIPTENSOR_WORKSPACE_MIN");
    EXPECT_STREQ(hiptensor::workSizePrefToString(HIPTENSOR_WORKSPACE_DEFAULT).c_str(),
                 "HIPTENSOR_WORKSPACE_DEFAULT");
    EXPECT_STREQ(hiptensor::workSizePrefToString(HIPTENSOR_WORKSPACE_MAX).c_str(),
                 "HIPTENSOR_WORKSPACE_MAX");
}
