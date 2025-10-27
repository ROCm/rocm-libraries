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

#include "testing_bsrilu02.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t                                                       base;
typedef hipsparseDirection_t                                                       dir;
typedef hipsparseSolvePolicy_t                                                     solve_policy;
typedef std::tuple<int, int, int, double, double, double, dir, base, solve_policy> bsrilu02_tuple;
typedef std::tuple<int, int, double, double, double, dir, base, solve_policy, std::string>
    bsrilu02_bin_tuple;

int bsrilu02_M_range[]   = {0, 50, 426};
int bsrilu02_dim_range[] = {1, 3, 5, 9};

int    bsrilu02_boost_range[]      = {0};
double bsrilu02_boost_tol_range[]  = {1.1};
double bsrilu02_boost_val_range[]  = {0.3};
double bsrilu02_boost_vali_range[] = {0.2};

base         bsrilu02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
dir          bsrilu02_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};
solve_policy bsrilu02_solve_policy_range[]
    = {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL};

std::string bsrilu02_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_bsrilu02 : public testing::TestWithParam<bsrilu02_tuple>
{
protected:
    parameterized_bsrilu02() {}
    virtual ~parameterized_bsrilu02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsrilu02_bin : public testing::TestWithParam<bsrilu02_bin_tuple>
{
protected:
    parameterized_bsrilu02_bin() {}
    virtual ~parameterized_bsrilu02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsrilu02_arguments(bsrilu02_tuple tup)
{
    Arguments arg;
    arg.M            = std::get<0>(tup);
    arg.block_dim    = std::get<1>(tup);
    arg.numericboost = std::get<2>(tup);
    arg.boosttol     = std::get<3>(tup);
    arg.boostval     = std::get<4>(tup);
    arg.boostvali    = std::get<5>(tup);
    arg.dirA         = std::get<6>(tup);
    arg.baseA        = std::get<7>(tup);
    arg.solve_policy = std::get<8>(tup);
    arg.timing       = 0;
    return arg;
}

Arguments setup_bsrilu02_arguments(bsrilu02_bin_tuple tup)
{
    Arguments arg;
    arg.M            = -99;
    arg.block_dim    = std::get<0>(tup);
    arg.numericboost = std::get<1>(tup);
    arg.boosttol     = std::get<2>(tup);
    arg.boostval     = std::get<3>(tup);
    arg.boostvali    = std::get<4>(tup);
    arg.dirA         = std::get<5>(tup);
    arg.baseA        = std::get<6>(tup);
    arg.solve_policy = std::get<7>(tup);
    arg.timing       = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<8>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(bsrilu02_bad_arg, bsrilu02_float)
{
    testing_bsrilu02_bad_arg<float>();
}

TEST_P(parameterized_bsrilu02, bsrilu02_float)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_double)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_float_complex)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_double_complex)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02_bin, bsrilu02_bin_float)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02_bin, bsrilu02_bin_double)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(bsrilu02,
                         parameterized_bsrilu02,
                         testing::Combine(testing::ValuesIn(bsrilu02_M_range),
                                          testing::ValuesIn(bsrilu02_dim_range),
                                          testing::ValuesIn(bsrilu02_boost_range),
                                          testing::ValuesIn(bsrilu02_boost_tol_range),
                                          testing::ValuesIn(bsrilu02_boost_val_range),
                                          testing::ValuesIn(bsrilu02_boost_vali_range),
                                          testing::ValuesIn(bsrilu02_dir_range),
                                          testing::ValuesIn(bsrilu02_idxbase_range),
                                          testing::ValuesIn(bsrilu02_solve_policy_range)));

INSTANTIATE_TEST_SUITE_P(bsrilu02_bin,
                         parameterized_bsrilu02_bin,
                         testing::Combine(testing::ValuesIn(bsrilu02_dim_range),
                                          testing::ValuesIn(bsrilu02_boost_range),
                                          testing::ValuesIn(bsrilu02_boost_tol_range),
                                          testing::ValuesIn(bsrilu02_boost_val_range),
                                          testing::ValuesIn(bsrilu02_boost_vali_range),
                                          testing::ValuesIn(bsrilu02_dir_range),
                                          testing::ValuesIn(bsrilu02_idxbase_range),
                                          testing::ValuesIn(bsrilu02_solve_policy_range),
                                          testing::ValuesIn(bsrilu02_bin)));
#endif
