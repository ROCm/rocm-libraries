/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_gtsv2_strided_batch.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int> gtsv2_strided_batch_tuple;

int gtsv2_strided_batch_M_range[]           = {512};
int gtsv2_strided_batch_batch_count_range[] = {512};

class parameterized_gtsv2_strided_batch : public testing::TestWithParam<gtsv2_strided_batch_tuple>
{
protected:
    parameterized_gtsv2_strided_batch() {}
    virtual ~parameterized_gtsv2_strided_batch() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gtsv2_strided_batch_arguments(gtsv2_strided_batch_tuple tup)
{
    Arguments arg;
    arg.M           = std::get<0>(tup);
    arg.batch_count = std::get<1>(tup);
    arg.timing      = 0;
    return arg;
}

TEST(gtsv2_strided_batch_bad_arg, gtsv2_strided_batch_float)
{
    testing_gtsv2_strided_batch_bad_arg<float>();
}

TEST_P(parameterized_gtsv2_strided_batch, gtsv2_strided_batch_float)
{
    Arguments arg = setup_gtsv2_strided_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_strided_batch<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_strided_batch, gtsv2_strided_batch_double)
{
    Arguments arg = setup_gtsv2_strided_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_strided_batch<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_strided_batch, gtsv2_strided_batch_float_complex)
{
    Arguments arg = setup_gtsv2_strided_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_strided_batch<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_strided_batch, gtsv2_strided_batch_double_complex)
{
    Arguments arg = setup_gtsv2_strided_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_strided_batch<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    gtsv2_strided_batch,
    parameterized_gtsv2_strided_batch,
    testing::Combine(testing::ValuesIn(gtsv2_strided_batch_M_range),
                     testing::ValuesIn(gtsv2_strided_batch_batch_count_range)));
