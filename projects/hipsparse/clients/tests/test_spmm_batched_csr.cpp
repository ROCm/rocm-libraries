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

#include "testing_spmm_batched_csr.hpp"

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
    spmm_batched_csr_tuple;
typedef std::tuple<int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   std::string>
    spmm_batched_csr_bin_tuple;

int spmm_batched_csr_M_range[] = {50};
int spmm_batched_csr_N_range[] = {5};
int spmm_batched_csr_K_range[] = {84};

std::vector<double> spmm_batched_csr_alpha_range = {2.0};
std::vector<double> spmm_batched_csr_beta_range  = {1.0};

hipsparseOperation_t spmm_batched_csr_transA_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOperation_t spmm_batched_csr_transB_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOrder_t     spmm_batched_csr_orderB_range[]  = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseOrder_t     spmm_batched_csr_orderC_range[]  = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseIndexBase_t spmm_batched_csr_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

std::string spmm_batched_csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_spmm_batched_csr : public testing::TestWithParam<spmm_batched_csr_tuple>
{
protected:
    parameterized_spmm_batched_csr() {}
    virtual ~parameterized_spmm_batched_csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spmm_batched_csr_bin : public testing::TestWithParam<spmm_batched_csr_bin_tuple>
{
protected:
    parameterized_spmm_batched_csr_bin() {}
    virtual ~parameterized_spmm_batched_csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spmm_batched_csr_arguments(spmm_batched_csr_tuple tup)
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

Arguments setup_spmm_batched_csr_arguments(spmm_batched_csr_bin_tuple tup)
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

// batched_csr format not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST(spmm_batched_csr_bad_arg, spmm_batched_csr_float)
{
    testing_spmm_batched_csr_bad_arg();
}

TEST_P(parameterized_spmm_batched_csr, spmm_batched_csr_i32_float)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_csr, spmm_batched_csr_i32_float_complex)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_csr_bin, spmm_batched_csr_bin_i32_float)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

// 64 bit indices not supported in cusparse for all algorithms
#if(!defined(CUDART_VERSION))
TEST_P(parameterized_spmm_batched_csr, spmm_batched_csr_i64_double)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_csr, spmm_batched_csr_i64_double_complex)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmm_batched_csr_bin, spmm_batched_csr_bin_i64_double)
{
    Arguments arg = setup_spmm_batched_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spmm_batched_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(spmm_batched_csr,
                         parameterized_spmm_batched_csr,
                         testing::Combine(testing::ValuesIn(spmm_batched_csr_M_range),
                                          testing::ValuesIn(spmm_batched_csr_N_range),
                                          testing::ValuesIn(spmm_batched_csr_K_range),
                                          testing::ValuesIn(spmm_batched_csr_alpha_range),
                                          testing::ValuesIn(spmm_batched_csr_beta_range),
                                          testing::ValuesIn(spmm_batched_csr_transA_range),
                                          testing::ValuesIn(spmm_batched_csr_transB_range),
                                          testing::ValuesIn(spmm_batched_csr_orderB_range),
                                          testing::ValuesIn(spmm_batched_csr_orderC_range),
                                          testing::ValuesIn(spmm_batched_csr_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(spmm_batched_csr_bin,
                         parameterized_spmm_batched_csr_bin,
                         testing::Combine(testing::ValuesIn(spmm_batched_csr_N_range),
                                          testing::ValuesIn(spmm_batched_csr_alpha_range),
                                          testing::ValuesIn(spmm_batched_csr_beta_range),
                                          testing::ValuesIn(spmm_batched_csr_transA_range),
                                          testing::ValuesIn(spmm_batched_csr_transB_range),
                                          testing::ValuesIn(spmm_batched_csr_orderB_range),
                                          testing::ValuesIn(spmm_batched_csr_orderC_range),
                                          testing::ValuesIn(spmm_batched_csr_idxbase_range),
                                          testing::ValuesIn(spmm_batched_csr_bin)));
#endif
