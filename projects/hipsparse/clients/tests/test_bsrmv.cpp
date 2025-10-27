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

#include "testing_bsrmv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                                    base;
typedef hipsparseDirection_t                                    dir;
typedef std::tuple<int, int, double, double, int, dir, base>    bsrmv_tuple;
typedef std::tuple<double, double, int, dir, base, std::string> bsrmv_bin_tuple;

int bsr_M_range[]   = {0, 500, 7111};
int bsr_N_range[]   = {0, 842, 4441};
int bsr_dim_range[] = {1, 3, 5, 9};

std::vector<double> bsr_alpha_range = {3.0};
std::vector<double> bsr_beta_range  = {1.0};

base bsr_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
dir  bsr_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string bsr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_bsrmv : public testing::TestWithParam<bsrmv_tuple>
{
protected:
    parameterized_bsrmv() {}
    virtual ~parameterized_bsrmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsrmv_bin : public testing::TestWithParam<bsrmv_bin_tuple>
{
protected:
    parameterized_bsrmv_bin() {}
    virtual ~parameterized_bsrmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsrmv_arguments(bsrmv_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.beta      = std::get<3>(tup);
    arg.block_dim = std::get<4>(tup);
    arg.dirA      = std::get<5>(tup);
    arg.baseA     = std::get<6>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_bsrmv_arguments(bsrmv_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.alpha     = std::get<0>(tup);
    arg.beta      = std::get<1>(tup);
    arg.block_dim = std::get<2>(tup);
    arg.dirA      = std::get<3>(tup);
    arg.baseA     = std::get<4>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(bsrmv_bad_arg, bsrmv_float)
{
    testing_bsrmv_bad_arg<float>();
}

TEST_P(parameterized_bsrmv, bsrmv_float)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmv, bsrmv_double)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmv, bsrmv_float_complex)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmv, bsrmv_double_complex)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmv_bin, bsrmv_bin_float)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrmv_bin, bsrmv_bin_double)
{
    Arguments arg = setup_bsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(bsrmv,
                         parameterized_bsrmv,
                         testing::Combine(testing::ValuesIn(bsr_M_range),
                                          testing::ValuesIn(bsr_N_range),
                                          testing::ValuesIn(bsr_alpha_range),
                                          testing::ValuesIn(bsr_beta_range),
                                          testing::ValuesIn(bsr_dim_range),
                                          testing::ValuesIn(bsr_dir_range),
                                          testing::ValuesIn(bsr_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(bsrmv_bin,
                         parameterized_bsrmv_bin,
                         testing::Combine(testing::ValuesIn(bsr_alpha_range),
                                          testing::ValuesIn(bsr_beta_range),
                                          testing::ValuesIn(bsr_dim_range),
                                          testing::ValuesIn(bsr_dir_range),
                                          testing::ValuesIn(bsr_idxbase_range),
                                          testing::ValuesIn(bsr_bin)));
