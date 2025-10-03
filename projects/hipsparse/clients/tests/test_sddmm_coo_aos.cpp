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

#include "testing_sddmm_coo_aos.hpp"

#include <hipsparse.h>

struct alpha_beta
{
    double alpha;
    double beta;
};

typedef std::tuple<int,
                   int,
                   int,
                   alpha_beta,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   hipsparseSDDMMAlg_t>
    sddmm_coo_aos_tuple;
typedef std::tuple<int,
                   alpha_beta,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   hipsparseSDDMMAlg_t,
                   std::string>
    sddmm_coo_aos_bin_tuple;

int sddmm_coo_aos_M_range[] = {50};
int sddmm_coo_aos_N_range[] = {84};
int sddmm_coo_aos_K_range[] = {5};

alpha_beta sddmm_coo_aos_alpha_beta_range[] = {{2.0, 1.0}};

hipsparseOperation_t sddmm_coo_aos_transA_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOperation_t sddmm_coo_aos_transB_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOrder_t     sddmm_coo_aos_orderA_range[] = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseOrder_t     sddmm_coo_aos_orderB_range[] = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseIndexBase_t sddmm_coo_aos_idxbase_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseSDDMMAlg_t sddmm_coo_aos_alg_range[] = {HIPSPARSE_SDDMM_ALG_DEFAULT};

std::string sddmm_coo_aos_bin[] = {"nos1.bin", "nos3.bin", "nos5.bin", "nos7.bin", "shipsec1.bin"};

class parameterized_sddmm_coo_aos : public testing::TestWithParam<sddmm_coo_aos_tuple>
{
protected:
    parameterized_sddmm_coo_aos() {}
    virtual ~parameterized_sddmm_coo_aos() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_sddmm_coo_aos_bin : public testing::TestWithParam<sddmm_coo_aos_bin_tuple>
{
protected:
    parameterized_sddmm_coo_aos_bin() {}
    virtual ~parameterized_sddmm_coo_aos_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_sddmm_coo_aos_arguments(sddmm_coo_aos_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.K         = std::get<2>(tup);
    arg.alpha     = std::get<3>(tup).alpha;
    arg.beta      = std::get<3>(tup).beta;
    arg.transA    = std::get<4>(tup);
    arg.transB    = std::get<5>(tup);
    arg.orderA    = std::get<6>(tup);
    arg.orderB    = std::get<7>(tup);
    arg.baseC     = std::get<8>(tup);
    arg.sddmm_alg = std::get<9>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_sddmm_coo_aos_arguments(sddmm_coo_aos_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.K         = std::get<0>(tup);
    arg.alpha     = std::get<1>(tup).alpha;
    arg.beta      = std::get<1>(tup).beta;
    arg.transA    = std::get<2>(tup);
    arg.transB    = std::get<3>(tup);
    arg.orderA    = std::get<4>(tup);
    arg.orderB    = std::get<5>(tup);
    arg.baseC     = std::get<6>(tup);
    arg.sddmm_alg = std::get<7>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<8>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

// COO_AOS format not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST(sddmm_coo_aos_bad_arg, sddmm_coo_aos_float)
{
    testing_sddmm_coo_aos_bad_arg();
}

TEST_P(parameterized_sddmm_coo_aos, sddmm_coo_aos_i32_float)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_aos, sddmm_coo_aos_i64_double)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_aos, sddmm_coo_aos_i32_float_complex)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_aos, sddmm_coo_aos_i64_double_complex)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_aos_bin, sddmm_coo_aos_bin_i32_float)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_aos_bin, sddmm_coo_aos_bin_i64_double)
{
    Arguments arg = setup_sddmm_coo_aos_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo_aos<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(sddmm_coo_aos,
                         parameterized_sddmm_coo_aos,
                         testing::Combine(testing::ValuesIn(sddmm_coo_aos_M_range),
                                          testing::ValuesIn(sddmm_coo_aos_N_range),
                                          testing::ValuesIn(sddmm_coo_aos_K_range),
                                          testing::ValuesIn(sddmm_coo_aos_alpha_beta_range),
                                          testing::ValuesIn(sddmm_coo_aos_transA_range),
                                          testing::ValuesIn(sddmm_coo_aos_transB_range),
                                          testing::ValuesIn(sddmm_coo_aos_orderA_range),
                                          testing::ValuesIn(sddmm_coo_aos_orderB_range),
                                          testing::ValuesIn(sddmm_coo_aos_idxbase_range),
                                          testing::ValuesIn(sddmm_coo_aos_alg_range)));

INSTANTIATE_TEST_SUITE_P(sddmm_coo_aos_bin,
                         parameterized_sddmm_coo_aos_bin,
                         testing::Combine(testing::ValuesIn(sddmm_coo_aos_K_range),
                                          testing::ValuesIn(sddmm_coo_aos_alpha_beta_range),
                                          testing::ValuesIn(sddmm_coo_aos_transA_range),
                                          testing::ValuesIn(sddmm_coo_aos_transB_range),
                                          testing::ValuesIn(sddmm_coo_aos_orderA_range),
                                          testing::ValuesIn(sddmm_coo_aos_orderB_range),
                                          testing::ValuesIn(sddmm_coo_aos_idxbase_range),
                                          testing::ValuesIn(sddmm_coo_aos_alg_range),
                                          testing::ValuesIn(sddmm_coo_aos_bin)));
#endif
