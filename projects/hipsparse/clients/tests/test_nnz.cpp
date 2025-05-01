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

#include "testing_nnz.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int> nnz_tuple;

int nnz_M_range[]  = {0, 10, 500, 872, 1000};
int nnz_N_range[]  = {0, 33, 242, 623, 1000};
int nnz_LD_range[] = {1000};

class parameterized_nnz : public testing::TestWithParam<nnz_tuple>
{
protected:
    parameterized_nnz() {}
    virtual ~parameterized_nnz() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_nnz_arguments(nnz_tuple tup)
{
    Arguments arg;
    arg.M   = std::get<0>(tup);
    arg.N   = std::get<1>(tup);
    arg.lda = std::get<2>(tup);
    return arg;
}

TEST(nnz_bad_arg, nnz)
{
    testing_nnz_bad_arg<float>();
}

TEST_P(parameterized_nnz, nnz_float)
{
    Arguments arg = setup_nnz_arguments(GetParam());

    hipsparseStatus_t status = testing_nnz<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_nnz, nnz_double)
{
    Arguments arg = setup_nnz_arguments(GetParam());

    hipsparseStatus_t status = testing_nnz<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_nnz, nnz_float_complex)
{
    Arguments arg = setup_nnz_arguments(GetParam());

    hipsparseStatus_t status = testing_nnz<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_nnz, nnz_double_complex)
{
    Arguments arg = setup_nnz_arguments(GetParam());

    hipsparseStatus_t status = testing_nnz<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(nnz,
                         parameterized_nnz,
                         testing::Combine(testing::ValuesIn(nnz_M_range),
                                          testing::ValuesIn(nnz_N_range),
                                          testing::ValuesIn(nnz_LD_range)));
