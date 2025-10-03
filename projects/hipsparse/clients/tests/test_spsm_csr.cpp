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

#include "testing_spsm_csr.hpp"

#include <hipsparse.h>

struct M_N_K_alpha
{
    int    M;
    int    N;
    int    K;
    double alpha;
};

typedef std::tuple<M_N_K_alpha,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSMAlg_t>
    spsm_csr_tuple;
typedef std::tuple<M_N_K_alpha,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSMAlg_t,
                   std::string>
    spsm_csr_bin_tuple;

M_N_K_alpha spsm_csr_M_N_K_alpha_range[] = {{50, 50, 22, 2.0}};

hipsparseOperation_t spsm_csr_transA_range[]    = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseOperation_t spsm_csr_transB_range[]    = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseOrder_t     spsm_csr_orderB_range[]    = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseOrder_t     spsm_csr_orderC_range[]    = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseIndexBase_t spsm_csr_idxbase_range[]   = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseDiagType_t  spsm_csr_diag_type_range[] = {HIPSPARSE_DIAG_TYPE_NON_UNIT};
hipsparseFillMode_t  spsm_csr_fill_mode_range[]
    = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
hipsparseSpSMAlg_t spsm_csr_alg_range[] = {HIPSPARSE_SPSM_ALG_DEFAULT};

std::string spsm_csr_bin[] = {"nos1.bin", "nos3.bin", "nos5.bin", "scircuit.bin"};

class parameterized_spsm_csr : public testing::TestWithParam<spsm_csr_tuple>
{
protected:
    parameterized_spsm_csr() {}
    virtual ~parameterized_spsm_csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spsm_csr_bin : public testing::TestWithParam<spsm_csr_bin_tuple>
{
protected:
    parameterized_spsm_csr_bin() {}
    virtual ~parameterized_spsm_csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spsm_csr_arguments(spsm_csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup).M;
    arg.N         = std::get<0>(tup).N;
    arg.K         = std::get<0>(tup).K;
    arg.alpha     = std::get<0>(tup).alpha;
    arg.transA    = std::get<1>(tup);
    arg.transB    = std::get<2>(tup);
    arg.orderB    = std::get<3>(tup);
    arg.orderC    = std::get<4>(tup);
    arg.baseA     = std::get<5>(tup);
    arg.diag_type = std::get<6>(tup);
    arg.fill_mode = std::get<7>(tup);
    arg.spsm_alg  = std::get<8>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_spsm_csr_arguments(spsm_csr_bin_tuple tup)
{
    Arguments arg;
    arg.K         = std::get<0>(tup).K;
    arg.alpha     = std::get<0>(tup).alpha;
    arg.transA    = std::get<1>(tup);
    arg.transB    = std::get<2>(tup);
    arg.orderB    = std::get<3>(tup);
    arg.orderC    = std::get<4>(tup);
    arg.baseA     = std::get<5>(tup);
    arg.diag_type = std::get<6>(tup);
    arg.fill_mode = std::get<7>(tup);
    arg.spsm_alg  = std::get<8>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<9>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spsm_csr_bad_arg, spsm_csr_float)
{
    testing_spsm_csr_bad_arg();
}

TEST_P(parameterized_spsm_csr, spsm_csr_i32_float)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsm_csr, spsm_csr_i64_double)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsm_csr, spsm_csr_i32_float_complex)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsm_csr, spsm_csr_i64_double_complex)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsm_csr_bin, spsm_csr_bin_i32_float)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsm_csr_bin, spsm_csr_bin_i64_double)
{
    Arguments arg = setup_spsm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsm_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spsm_csr,
                         parameterized_spsm_csr,
                         testing::Combine(testing::ValuesIn(spsm_csr_M_N_K_alpha_range),
                                          testing::ValuesIn(spsm_csr_transA_range),
                                          testing::ValuesIn(spsm_csr_transB_range),
                                          testing::ValuesIn(spsm_csr_orderB_range),
                                          testing::ValuesIn(spsm_csr_orderC_range),
                                          testing::ValuesIn(spsm_csr_idxbase_range),
                                          testing::ValuesIn(spsm_csr_diag_type_range),
                                          testing::ValuesIn(spsm_csr_fill_mode_range),
                                          testing::ValuesIn(spsm_csr_alg_range)));

INSTANTIATE_TEST_SUITE_P(spsm_csr_bin,
                         parameterized_spsm_csr_bin,
                         testing::Combine(testing::ValuesIn(spsm_csr_M_N_K_alpha_range),
                                          testing::ValuesIn(spsm_csr_transA_range),
                                          testing::ValuesIn(spsm_csr_transB_range),
                                          testing::ValuesIn(spsm_csr_orderB_range),
                                          testing::ValuesIn(spsm_csr_orderC_range),
                                          testing::ValuesIn(spsm_csr_idxbase_range),
                                          testing::ValuesIn(spsm_csr_diag_type_range),
                                          testing::ValuesIn(spsm_csr_fill_mode_range),
                                          testing::ValuesIn(spsm_csr_alg_range),
                                          testing::ValuesIn(spsm_csr_bin)));
#endif
