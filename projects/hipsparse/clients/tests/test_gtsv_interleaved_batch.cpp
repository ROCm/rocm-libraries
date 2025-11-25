/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_gtsv_interleaved_batch.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int, int> gtsv_interleaved_batch_tuple;

int gtsv_interleaved_batch_M_range[]           = {512};
int gtsv_interleaved_batch_batch_count_range[] = {512};
int gtsv_interleaved_batch_algo_range[]        = {0};

class parameterized_gtsv_interleaved_batch
    : public testing::TestWithParam<gtsv_interleaved_batch_tuple>
{
protected:
    parameterized_gtsv_interleaved_batch() {}
    virtual ~parameterized_gtsv_interleaved_batch() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gtsv_interleaved_batch_arguments(gtsv_interleaved_batch_tuple tup)
{
    Arguments arg;
    arg.M           = std::get<0>(tup);
    arg.batch_count = std::get<1>(tup);
    arg.gtsv_alg    = std::get<2>(tup);
    arg.timing      = 0;
    return arg;
}

TEST(gtsv_interleaved_batch_bad_arg, gtsv_interleaved_batch_float)
{
    testing_gtsv_interleaved_batch_bad_arg<float>();
}

TEST_P(parameterized_gtsv_interleaved_batch, gtsv_interleaved_batch_float)
{
    Arguments arg = setup_gtsv_interleaved_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv_interleaved_batch<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv_interleaved_batch, gtsv_interleaved_batch_double)
{
    Arguments arg = setup_gtsv_interleaved_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv_interleaved_batch<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv_interleaved_batch, gtsv_interleaved_batch_float_complex)
{
    Arguments arg = setup_gtsv_interleaved_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv_interleaved_batch<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv_interleaved_batch, gtsv_interleaved_batch_double_complex)
{
    Arguments arg = setup_gtsv_interleaved_batch_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv_interleaved_batch<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    gtsv_interleaved_batch,
    parameterized_gtsv_interleaved_batch,
    testing::Combine(testing::ValuesIn(gtsv_interleaved_batch_M_range),
                     testing::ValuesIn(gtsv_interleaved_batch_batch_count_range),
                     testing::ValuesIn(gtsv_interleaved_batch_algo_range)));
