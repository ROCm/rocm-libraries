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

#include <cstring>
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

TEST(alignStridesToModesTest, UtilTest)
{
    std::vector<std::size_t> const strides = {1, 3, 12}; // packed col-major for {3, 4, 5}

    // Reference order equal to the tensor's own order leaves the strides untouched.
    EXPECT_EQ(hiptensor::alignStridesToModes({'a', 'b', 'c'}, {'a', 'b', 'c'}, strides),
              (std::vector<std::size_t>{1, 3, 12}));

    // A reordered tensor is permuted so that entry i is the stride of the reference mode i.
    EXPECT_EQ(hiptensor::alignStridesToModes({'c', 'a', 'b'}, {'a', 'b', 'c'}, strides),
              (std::vector<std::size_t>{12, 1, 3}));

    // A reference mode the tensor doesn't carry gets stride 0, broadcasting it along that mode.
    EXPECT_EQ(hiptensor::alignStridesToModes({'a', 'b', 'd'}, {'a', 'b', 'c'}, strides),
              (std::vector<std::size_t>{1, 3, 0}));

    // A tensor of lower rank than the reference is broadcast along every mode it lacks.
    EXPECT_EQ(hiptensor::alignStridesToModes<std::size_t>({'a', 'b', 'c'}, {'c'}, {1}),
              (std::vector<std::size_t>{0, 0, 1}));

    // Mode and stride counts must agree.
    EXPECT_TRUE(hiptensor::alignStridesToModes({'a', 'b'}, {'a', 'b'}, strides).empty());
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
    EXPECT_EQ(hiptensorGetVersion(), 2005000);
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

// The elementwise and reduction entry points read alpha, beta and gamma through readVal with a
// compute descriptor, so every real compute type they accept must decode to the value written.
// 1.25 is exact in every one of these types.
TEST(readValComputeDescriptorTest, UtilTest)
{
    hip_bfloat16 bf16Value(1.25f);
    _Float16     f16Value = 1.25f;
    float        f32Value = 1.25f;
    double       f64Value = 1.25;

    EXPECT_EQ(hiptensor::readVal<float>(&bf16Value, HIPTENSOR_COMPUTE_DESC_16BF), 1.25f);
    EXPECT_EQ(hiptensor::readVal<double>(&bf16Value, HIPTENSOR_COMPUTE_DESC_16BF), 1.25);
    EXPECT_EQ(hiptensor::readVal<float>(&f16Value, HIPTENSOR_COMPUTE_DESC_16F), 1.25f);
    EXPECT_EQ(hiptensor::readVal<double>(&f16Value, HIPTENSOR_COMPUTE_DESC_16F), 1.25);
    EXPECT_EQ(hiptensor::readVal<float>(&f32Value, HIPTENSOR_COMPUTE_DESC_32F), 1.25f);
    EXPECT_EQ(hiptensor::readVal<double>(&f32Value, HIPTENSOR_COMPUTE_DESC_32F), 1.25);
    EXPECT_EQ(hiptensor::readVal<float>(&f64Value, HIPTENSOR_COMPUTE_DESC_64F), 1.25f);
    EXPECT_EQ(hiptensor::readVal<double>(&f64Value, HIPTENSOR_COMPUTE_DESC_64F), 1.25);
}

// Host scalars come from user pointers with no alignment guarantee: a complex double passed as
// double[2] is only 8-byte aligned. Scalar reads must not fault or misread at any byte offset.
TEST(readValUnalignedTest, UtilTest)
{
    alignas(16) unsigned char buffer[64] = {};
    for(size_t offset = 0; offset < 16; offset++)
    {
        unsigned char* p = buffer + offset;

        hipDoubleComplex const z = make_hipDoubleComplex(1.25, -2.5);
        std::memcpy(p, &z, sizeof(z));
        auto const zr = hiptensor::loadUnaligned<hipDoubleComplex>(p);
        EXPECT_EQ(zr.x, 1.25);
        EXPECT_EQ(zr.y, -2.5);

        hipFloatComplex const c = make_hipFloatComplex(0.5f, 3.0f);
        std::memcpy(p, &c, sizeof(c));
        auto const cr = hiptensor::loadUnaligned<hipFloatComplex>(p);
        EXPECT_EQ(cr.x, 0.5f);
        EXPECT_EQ(cr.y, 3.0f);

        double const d = -7.75;
        std::memcpy(p, &d, sizeof(d));
        EXPECT_EQ(hiptensor::readVal<double>(p, HIPTENSOR_COMPUTE_DESC_64F), -7.75);
        EXPECT_EQ(hiptensor::readVal<double>(p, HIPTENSOR_R_64F), -7.75);

        float const f = 6.5f;
        std::memcpy(p, &f, sizeof(f));
        EXPECT_EQ(hiptensor::readVal<float>(p, HIPTENSOR_COMPUTE_DESC_32F), 6.5f);
    }
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
