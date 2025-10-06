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

#include "testing_spsv_coo.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSVAlg_t>
    spsv_coo_tuple;
typedef std::tuple<double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseDiagType_t,
                   hipsparseFillMode_t,
                   hipsparseSpSVAlg_t,
                   std::string>
    spsv_coo_bin_tuple;

int spsv_coo_M_range[] = {50};
int spsv_coo_N_range[] = {50};

std::vector<double> spsv_coo_alpha_range = {2.0};

hipsparseOperation_t spsv_coo_transA_range[]    = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseIndexBase_t spsv_coo_idxbase_range[]   = {HIPSPARSE_INDEX_BASE_ONE};
hipsparseDiagType_t  spsv_coo_diag_type_range[] = {HIPSPARSE_DIAG_TYPE_NON_UNIT};
hipsparseFillMode_t  spsv_coo_fill_mode_range[]
    = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
hipsparseSpSVAlg_t spsv_coo_alg_range[] = {HIPSPARSE_SPSV_ALG_DEFAULT};

std::string spsv_coo_bin[] = {"nos1.bin",
                              "nos2.bin",
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin",
                              "scircuit.bin"};

class parameterized_spsv_coo : public testing::TestWithParam<spsv_coo_tuple>
{
protected:
    parameterized_spsv_coo() {}
    virtual ~parameterized_spsv_coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spsv_coo_bin : public testing::TestWithParam<spsv_coo_bin_tuple>
{
protected:
    parameterized_spsv_coo_bin() {}
    virtual ~parameterized_spsv_coo_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spsv_coo_arguments(spsv_coo_tuple tup)
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

Arguments setup_spsv_coo_arguments(spsv_coo_bin_tuple tup)
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
TEST(spsv_coo_bad_arg, spsv_coo_float)
{
    testing_spsv_coo_bad_arg();
}

TEST_P(parameterized_spsv_coo, spsv_coo_i32_float)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_coo, spsv_coo_i64_double)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_coo, spsv_coo_i32_float_complex)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_coo, spsv_coo_i64_double_complex)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_coo_bin, spsv_coo_bin_i32_float)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spsv_coo_bin, spsv_coo_bin_i64_double)
{
    Arguments arg = setup_spsv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spsv_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spsv_coo,
                         parameterized_spsv_coo,
                         testing::Combine(testing::ValuesIn(spsv_coo_M_range),
                                          testing::ValuesIn(spsv_coo_N_range),
                                          testing::ValuesIn(spsv_coo_alpha_range),
                                          testing::ValuesIn(spsv_coo_transA_range),
                                          testing::ValuesIn(spsv_coo_idxbase_range),
                                          testing::ValuesIn(spsv_coo_diag_type_range),
                                          testing::ValuesIn(spsv_coo_fill_mode_range),
                                          testing::ValuesIn(spsv_coo_alg_range)));

INSTANTIATE_TEST_SUITE_P(spsv_coo_bin,
                         parameterized_spsv_coo_bin,
                         testing::Combine(testing::ValuesIn(spsv_coo_alpha_range),
                                          testing::ValuesIn(spsv_coo_transA_range),
                                          testing::ValuesIn(spsv_coo_idxbase_range),
                                          testing::ValuesIn(spsv_coo_diag_type_range),
                                          testing::ValuesIn(spsv_coo_fill_mode_range),
                                          testing::ValuesIn(spsv_coo_alg_range),
                                          testing::ValuesIn(spsv_coo_bin)));
#endif
