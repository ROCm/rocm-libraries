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

#include "testing_dense_to_sparse_csc.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t, hipsparseOrder_t, hipsparseDenseToSparseAlg_t>
    dense_to_sparse_csc_tuple;

int dense_to_sparse_csc_M_range[] = {100};
int dense_to_sparse_csc_N_range[] = {10};

hipsparseIndexBase_t dense_to_sparse_csc_base[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseOrder_t dense_to_sparse_csc_order[]          = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseDenseToSparseAlg_t dense_to_sparse_csc_alg[] = {HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT};

class parameterized_dense_to_sparse_csc : public testing::TestWithParam<dense_to_sparse_csc_tuple>
{
protected:
    parameterized_dense_to_sparse_csc() {}
    virtual ~parameterized_dense_to_sparse_csc() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_dense_to_sparse_csc_arguments(dense_to_sparse_csc_tuple tup)
{
    Arguments arg;
    arg.M                = std::get<0>(tup);
    arg.N                = std::get<1>(tup);
    arg.baseA            = std::get<2>(tup);
    arg.orderA           = std::get<3>(tup);
    arg.dense2sparse_alg = std::get<4>(tup);
    arg.timing           = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
TEST(dense_to_sparse_csc_bad_arg, dense_to_sparse_csc)
{
    testing_dense_to_sparse_csc_bad_arg();
}

TEST_P(parameterized_dense_to_sparse_csc, dense_to_sparse_csc_float)
{
    Arguments arg = setup_dense_to_sparse_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_dense_to_sparse_csc<int, int, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(dense_to_sparse_csc,
                         parameterized_dense_to_sparse_csc,
                         testing::Combine(testing::ValuesIn(dense_to_sparse_csc_M_range),
                                          testing::ValuesIn(dense_to_sparse_csc_N_range),
                                          testing::ValuesIn(dense_to_sparse_csc_base),
                                          testing::ValuesIn(dense_to_sparse_csc_order),
                                          testing::ValuesIn(dense_to_sparse_csc_alg)));
#endif
