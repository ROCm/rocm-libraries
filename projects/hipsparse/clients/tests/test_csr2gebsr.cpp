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

#include "testing_csr2gebsr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::
    tuple<int, int, int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t>
        csr2gebsr_tuple;
typedef std::
    tuple<int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t, std::string>
        csr2gebsr_bin_tuple;

// Random matrices
int csr2gebsr_M_range[]             = {0, 13095};
int csr2gebsr_N_range[]             = {0, 12766};
int csr2gebsr_row_block_dim_range[] = {1, 2, 7};
int csr2gebsr_col_block_dim_range[] = {1, 2};

hipsparseIndexBase_t csr2gebsr_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

hipsparseIndexBase_t csr2gebsr_bsr_base_range[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t csr2gebsr_dir_range[] = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

// Matrices from files (float and double)
int csr2gebsr_row_block_dim_range_bin[] = {5};
int csr2gebsr_col_block_dim_range_bin[] = {5};

hipsparseIndexBase_t csr2gebsr_csr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t csr2gebsr_bsr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t csr2gebsr_dir_range_bin[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string csr2gebsr_bin[] = {"rma10.bin",
                               "mac_econ_fwd500.bin",
                               "scircuit.bin",
                               "bmwcra_1.bin",
                               "nos1.bin",
                               "nos6.bin",
                               "nos7.bin"};

class parameterized_csr2gebsr : public testing::TestWithParam<csr2gebsr_tuple>
{
protected:
    parameterized_csr2gebsr() {}
    virtual ~parameterized_csr2gebsr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2gebsr_bin : public testing::TestWithParam<csr2gebsr_bin_tuple>
{
protected:
    parameterized_csr2gebsr_bin() {}
    virtual ~parameterized_csr2gebsr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2gebsr_arguments(csr2gebsr_tuple tup)
{
    Arguments arg;
    arg.M              = std::get<0>(tup);
    arg.N              = std::get<1>(tup);
    arg.row_block_dimA = std::get<2>(tup);
    arg.col_block_dimA = std::get<3>(tup);
    arg.baseA          = std::get<4>(tup);
    arg.baseB          = std::get<5>(tup);
    arg.dirA           = std::get<6>(tup);
    arg.timing         = 0;
    return arg;
}

Arguments setup_csr2gebsr_arguments(csr2gebsr_bin_tuple tup)
{
    Arguments arg;
    arg.M              = -99;
    arg.N              = -99;
    arg.row_block_dimA = std::get<0>(tup);
    arg.col_block_dimA = std::get<1>(tup);
    arg.baseA          = std::get<2>(tup);
    arg.baseB          = std::get<3>(tup);
    arg.dirA           = std::get<4>(tup);
    arg.timing         = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(csr2gebsr_bad_arg, csr2gebsr)
{
    testing_csr2gebsr_bad_arg<float>();
}

TEST_P(parameterized_csr2gebsr, csr2gebsr_float)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2gebsr, csr2gebsr_double)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2gebsr, csr2gebsr_float_complex)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2gebsr, csr2gebsr_double_complex)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2gebsr_bin, csr2gebsr_bin_float)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2gebsr_bin, csr2gebsr_bin_double)
{
    Arguments arg = setup_csr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2gebsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csr2gebsr,
                         parameterized_csr2gebsr,
                         testing::Combine(testing::ValuesIn(csr2gebsr_M_range),
                                          testing::ValuesIn(csr2gebsr_N_range),
                                          testing::ValuesIn(csr2gebsr_row_block_dim_range),
                                          testing::ValuesIn(csr2gebsr_col_block_dim_range),
                                          testing::ValuesIn(csr2gebsr_bsr_base_range),
                                          testing::ValuesIn(csr2gebsr_csr_base_range),
                                          testing::ValuesIn(csr2gebsr_dir_range)));

INSTANTIATE_TEST_SUITE_P(csr2gebsr_bin,
                         parameterized_csr2gebsr_bin,
                         testing::Combine(testing::ValuesIn(csr2gebsr_row_block_dim_range_bin),
                                          testing::ValuesIn(csr2gebsr_col_block_dim_range_bin),
                                          testing::ValuesIn(csr2gebsr_bsr_base_range_bin),
                                          testing::ValuesIn(csr2gebsr_csr_base_range_bin),
                                          testing::ValuesIn(csr2gebsr_dir_range_bin),
                                          testing::ValuesIn(csr2gebsr_bin)));
