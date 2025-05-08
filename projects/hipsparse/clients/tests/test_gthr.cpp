/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_gthr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <vector>

typedef hipsparseIndexBase_t       base;
typedef std::tuple<int, int, base> gthr_tuple;

int gthr_N_range[]   = {12000, 15332, 22031};
int gthr_nnz_range[] = {0, 5, 10, 500, 1000, 7111, 10000};

base gthr_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_gthr : public testing::TestWithParam<gthr_tuple>
{
protected:
    parameterized_gthr() {}
    virtual ~parameterized_gthr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gthr_arguments(gthr_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(gthr_bad_arg, gthr_float)
{
    testing_gthr_bad_arg<float>();
}

TEST_P(parameterized_gthr, gthr_float)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    hipsparseStatus_t status = testing_gthr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthr, gthr_double)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    hipsparseStatus_t status = testing_gthr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthr, gthr_float_complex)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    hipsparseStatus_t status = testing_gthr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthr, gthr_double_complex)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    hipsparseStatus_t status = testing_gthr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gthr,
                         parameterized_gthr,
                         testing::Combine(testing::ValuesIn(gthr_N_range),
                                          testing::ValuesIn(gthr_nnz_range),
                                          testing::ValuesIn(gthr_idx_base_range)));
#endif