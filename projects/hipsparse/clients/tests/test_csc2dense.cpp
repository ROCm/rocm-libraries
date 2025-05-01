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

#include "testing_csc2dense.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>
typedef hipsparseIndexBase_t            base;
typedef std::tuple<int, int, int, base> csc2dense_tuple;

int  csc2dense_M_range[]        = {0, 10, 500, 872, 1000};
int  csc2dense_N_range[]        = {0, 33, 242, 623, 1000};
int  csc2dense_LD_range[]       = {1000};
base csc2dense_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_csc2dense : public testing::TestWithParam<csc2dense_tuple>
{
protected:
    parameterized_csc2dense() {}
    virtual ~parameterized_csc2dense() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csc2dense_arguments(csc2dense_tuple tup)
{
    Arguments arg;
    arg.M     = std::get<0>(tup);
    arg.N     = std::get<1>(tup);
    arg.lda   = std::get<2>(tup);
    arg.baseA = std::get<3>(tup);
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(csc2dense_bad_arg, csc2dense)
{
    testing_csc2dense_bad_arg<float>();
}

TEST_P(parameterized_csc2dense, csc2dense_float)
{
    Arguments arg = setup_csc2dense_arguments(GetParam());

    hipsparseStatus_t status = testing_csc2dense<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csc2dense, csc2dense_double)
{
    Arguments arg = setup_csc2dense_arguments(GetParam());

    hipsparseStatus_t status = testing_csc2dense<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csc2dense, csc2dense_float_complex)
{
    Arguments arg = setup_csc2dense_arguments(GetParam());

    hipsparseStatus_t status = testing_csc2dense<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csc2dense, csc2dense_double_complex)
{
    Arguments arg = setup_csc2dense_arguments(GetParam());

    hipsparseStatus_t status = testing_csc2dense<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csc2dense,
                         parameterized_csc2dense,
                         testing::Combine(testing::ValuesIn(csc2dense_M_range),
                                          testing::ValuesIn(csc2dense_N_range),
                                          testing::ValuesIn(csc2dense_LD_range),
                                          testing::ValuesIn(csc2dense_idx_base_range)));
#endif
