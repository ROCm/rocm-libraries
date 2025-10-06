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

#include "testing_bsrmm.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t                                                          base;
typedef hipsparseDirection_t                                                          direction;
typedef hipsparseOperation_t                                                          trans;
typedef std::tuple<int, int, int, int, double, double, direction, base, trans, trans> bsrmm_tuple;
typedef std::tuple<int, int, double, double, direction, base, trans, trans, std::string>
    bsrmm_bin_tuple;

int bsrmm_M_range[]         = {42, 2059};
int bsrmm_N_range[]         = {7, 78};
int bsrmm_K_range[]         = {50, 1375};
int bsrmm_block_dim_range[] = {4, 7, 16};

double bsrmm_alpha_range[] = {-0.5};
double bsrmm_beta_range[]  = {0.5};

direction bsrmm_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};
base      bsrmm_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
trans     bsrmm_transA_range[]  = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
trans     bsrmm_transB_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};

int bsrmm_N_range_bin[]         = {9, 23};
int bsrmm_block_dim_range_bin[] = {5};

double bsrmm_alpha_range_bin[] = {0.75};
double bsrmm_beta_range_bin[]  = {-0.5};

direction bsrmm_dir_range_bin[]     = {HIPSPARSE_DIRECTION_COLUMN};
base      bsrmm_idxbase_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};
trans     bsrmm_transA_range_bin[]  = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
trans bsrmm_transB_range_bin[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};

std::string bsrmm_bin[]
    = {"rma10.bin", "scircuit.bin", "nos1.bin", "nos3.bin", "nos5.bin", "nos7.bin"};

class parameterized_bsrmm : public testing::TestWithParam<bsrmm_tuple>
{
protected:
    parameterized_bsrmm() {}
    virtual ~parameterized_bsrmm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsrmm_bin : public testing::TestWithParam<bsrmm_bin_tuple>
{
protected:
    parameterized_bsrmm_bin() {}
    virtual ~parameterized_bsrmm_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsrmm_arguments(bsrmm_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.K         = std::get<2>(tup);
    arg.block_dim = std::get<3>(tup);
    arg.alpha     = std::get<4>(tup);
    arg.beta      = std::get<5>(tup);
    arg.dirA      = std::get<6>(tup);
    arg.baseA     = std::get<7>(tup);
    arg.transA    = std::get<8>(tup);
    arg.transB    = std::get<9>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_bsrmm_arguments(bsrmm_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = std::get<0>(tup);
    arg.K         = -99;
    arg.block_dim = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.beta      = std::get<3>(tup);
    arg.dirA      = std::get<4>(tup);
    arg.baseA     = std::get<5>(tup);
    arg.transA    = std::get<6>(tup);
    arg.transB    = std::get<7>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<8>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(bsrmm_bad_arg, bsrmm_float)
{
    testing_bsrmm_bad_arg<float>();
}

TEST_P(parameterized_bsrmm, bsrmm_float)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmm, bsrmm_double)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmm, bsrmm_float_complex)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmm, bsrmm_double_complex)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmm_bin, bsrmm_bin_float)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmm_bin, bsrmm_bin_double)
{
    Arguments arg = setup_bsrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(bsrmm,
                         parameterized_bsrmm,
                         testing::Combine(testing::ValuesIn(bsrmm_M_range),
                                          testing::ValuesIn(bsrmm_N_range),
                                          testing::ValuesIn(bsrmm_K_range),
                                          testing::ValuesIn(bsrmm_block_dim_range),
                                          testing::ValuesIn(bsrmm_alpha_range),
                                          testing::ValuesIn(bsrmm_beta_range),
                                          testing::ValuesIn(bsrmm_dir_range),
                                          testing::ValuesIn(bsrmm_idxbase_range),
                                          testing::ValuesIn(bsrmm_transA_range),
                                          testing::ValuesIn(bsrmm_transB_range)));

INSTANTIATE_TEST_SUITE_P(bsrmm_bin,
                         parameterized_bsrmm_bin,
                         testing::Combine(testing::ValuesIn(bsrmm_N_range_bin),
                                          testing::ValuesIn(bsrmm_block_dim_range_bin),
                                          testing::ValuesIn(bsrmm_alpha_range_bin),
                                          testing::ValuesIn(bsrmm_beta_range_bin),
                                          testing::ValuesIn(bsrmm_dir_range_bin),
                                          testing::ValuesIn(bsrmm_idxbase_range_bin),
                                          testing::ValuesIn(bsrmm_transA_range_bin),
                                          testing::ValuesIn(bsrmm_transB_range_bin),
                                          testing::ValuesIn(bsrmm_bin)));
