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

#include "hipsparse_arguments.hpp"
#include "testing_spsv_csr.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSVAlg_t>
    spsv_csr_tuple;
typedef std::tuple<double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSVAlg_t,
                   std::string>
    spsv_csr_bin_tuple;

int spsv_csr_M_range[] = {50};
int spsv_csr_N_range[] = {50};

std::vector<double> spsv_csr_alpha_range = {2.0};

hipsparseOperation_t spsv_csr_transA_range[]    = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseIndexBase_t spsv_csr_idxbase_range[]   = {HIPSPARSE_INDEX_BASE_ONE};
hipsparseDiagType_t  spsv_csr_diag_type_range[] = {HIPSPARSE_DIAG_TYPE_NON_UNIT};
hipsparseFillMode_t  spsv_csr_fill_mode_range[]
    = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
hipsparseSpSVAlg_t spsv_csr_alg_range[] = {HIPSPARSE_SPSV_ALG_DEFAULT};

std::string spsv_csr_bin[] = {"nos1.bin",
                              "nos2.bin",
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin",
                              "scircuit.bin"};

class parameterized_spsv_csr : public testing::TestWithParam<spsv_csr_tuple>
{
protected:
    parameterized_spsv_csr() {}
    virtual ~parameterized_spsv_csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spsv_csr_bin : public testing::TestWithParam<spsv_csr_bin_tuple>
{
protected:
    parameterized_spsv_csr_bin() {}
    virtual ~parameterized_spsv_csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spsv_csr_arguments(spsv_csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.transA    = std::get<3>(tup);
    arg.baseA     = std::get<4>(tup);
    arg.diag_type = std::get<5>(tup);
    arg.fill_mode = std::get<6>(tup);
    arg.spsv_alg  = std::get<7>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_spsv_csr_arguments(spsv_csr_bin_tuple tup)
{
    Arguments arg;
    arg.alpha     = std::get<0>(tup);
    arg.transA    = std::get<1>(tup);
    arg.baseA     = std::get<2>(tup);
    arg.diag_type = std::get<3>(tup);
    arg.fill_mode = std::get<4>(tup);
    arg.spsv_alg  = std::get<5>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spsv_csr_bad_arg, spsv_csr_float)
{
    testing_spsv_csr_bad_arg();
}

TEST_P(parameterized_spsv_csr, spsv_csr_i32_float)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_csr, spsv_csr_i64_double)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_csr, spsv_csr_i32_float_complex)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_csr, spsv_csr_i64_double_complex)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_csr_bin, spsv_csr_bin_i32_float)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_csr_bin, spsv_csr_bin_i64_double)
{
    Arguments arg = setup_spsv_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spsv_csr,
                         parameterized_spsv_csr,
                         testing::Combine(testing::ValuesIn(spsv_csr_M_range),
                                          testing::ValuesIn(spsv_csr_N_range),
                                          testing::ValuesIn(spsv_csr_alpha_range),
                                          testing::ValuesIn(spsv_csr_transA_range),
                                          testing::ValuesIn(spsv_csr_idxbase_range),
                                          testing::ValuesIn(spsv_csr_diag_type_range),
                                          testing::ValuesIn(spsv_csr_fill_mode_range),
                                          testing::ValuesIn(spsv_csr_alg_range)));

INSTANTIATE_TEST_SUITE_P(spsv_csr_bin,
                         parameterized_spsv_csr_bin,
                         testing::Combine(testing::ValuesIn(spsv_csr_alpha_range),
                                          testing::ValuesIn(spsv_csr_transA_range),
                                          testing::ValuesIn(spsv_csr_idxbase_range),
                                          testing::ValuesIn(spsv_csr_diag_type_range),
                                          testing::ValuesIn(spsv_csr_fill_mode_range),
                                          testing::ValuesIn(spsv_csr_alg_range),
                                          testing::ValuesIn(spsv_csr_bin)));
#endif
