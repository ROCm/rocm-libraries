/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_spmm_batched_coo.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t>
    spmm_batched_coo_tuple;
typedef std::tuple<int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   std::string>
    spmm_batched_coo_bin_tuple;

int spmm_batched_coo_M_range[] = {50};
int spmm_batched_coo_N_range[] = {5};
int spmm_batched_coo_K_range[] = {23};

std::vector<double> spmm_batched_coo_alpha_range = {2.0};
std::vector<double> spmm_batched_coo_beta_range  = {1.0};

hipsparseOperation_t spmm_batched_coo_transA_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOperation_t spmm_batched_coo_transB_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOrder_t     spmm_batched_coo_orderB_range[]  = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseOrder_t     spmm_batched_coo_orderC_range[]  = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseIndexBase_t spmm_batched_coo_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

std::string spmm_batched_coo_bin[] = {"shipsec1.bin"};

class parameterized_spmm_batched_coo : public testing::TestWithParam<spmm_batched_coo_tuple>
{
protected:
    parameterized_spmm_batched_coo() {}
    virtual ~parameterized_spmm_batched_coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spmm_batched_coo_bin : public testing::TestWithParam<spmm_batched_coo_bin_tuple>
{
protected:
    parameterized_spmm_batched_coo_bin() {}
    virtual ~parameterized_spmm_batched_coo_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spmm_batched_coo_arguments(spmm_batched_coo_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.K      = std::get<2>(tup);
    arg.alpha  = std::get<3>(tup);
    arg.beta   = std::get<4>(tup);
    arg.transA = std::get<5>(tup);
    arg.transB = std::get<6>(tup);
    arg.orderB = std::get<7>(tup);
    arg.orderC = std::get<8>(tup);
    arg.baseA  = std::get<9>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_spmm_batched_coo_arguments(spmm_batched_coo_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = std::get<0>(tup);
    arg.K      = -99;
    arg.alpha  = std::get<1>(tup);
    arg.beta   = std::get<2>(tup);
    arg.transA = std::get<3>(tup);
    arg.transB = std::get<4>(tup);
    arg.orderB = std::get<5>(tup);
    arg.orderC = std::get<6>(tup);
    arg.baseA  = std::get<7>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<8>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

// batched_coo format not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST(spmm_batched_coo_bad_arg, spmm_batched_coo_float)
{
    testing_spmm_batched_coo_bad_arg();
}

TEST_P(parameterized_spmm_batched_coo, spmm_batched_coo_i32_float)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_coo, spmm_batched_coo_i32_float_complex)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_coo_bin, spmm_batched_coo_bin_i32_float)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

// 64 bit indices not supported in cusparse for all algorithms
#if(!defined(CUDART_VERSION))
TEST_P(parameterized_spmm_batched_coo, spmm_batched_coo_i64_double)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_coo, spmm_batched_coo_i64_double_complex)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_coo_bin, spmm_batched_coo_bin_i64_double)
{
    Arguments arg = setup_spmm_batched_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(spmm_batched_coo,
                         parameterized_spmm_batched_coo,
                         testing::Combine(testing::ValuesIn(spmm_batched_coo_M_range),
                                          testing::ValuesIn(spmm_batched_coo_N_range),
                                          testing::ValuesIn(spmm_batched_coo_K_range),
                                          testing::ValuesIn(spmm_batched_coo_alpha_range),
                                          testing::ValuesIn(spmm_batched_coo_beta_range),
                                          testing::ValuesIn(spmm_batched_coo_transA_range),
                                          testing::ValuesIn(spmm_batched_coo_transB_range),
                                          testing::ValuesIn(spmm_batched_coo_orderB_range),
                                          testing::ValuesIn(spmm_batched_coo_orderC_range),
                                          testing::ValuesIn(spmm_batched_coo_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(spmm_batched_coo_bin,
                         parameterized_spmm_batched_coo_bin,
                         testing::Combine(testing::ValuesIn(spmm_batched_coo_N_range),
                                          testing::ValuesIn(spmm_batched_coo_alpha_range),
                                          testing::ValuesIn(spmm_batched_coo_beta_range),
                                          testing::ValuesIn(spmm_batched_coo_transA_range),
                                          testing::ValuesIn(spmm_batched_coo_transB_range),
                                          testing::ValuesIn(spmm_batched_coo_orderB_range),
                                          testing::ValuesIn(spmm_batched_coo_orderC_range),
                                          testing::ValuesIn(spmm_batched_coo_idxbase_range),
                                          testing::ValuesIn(spmm_batched_coo_bin)));
#endif
