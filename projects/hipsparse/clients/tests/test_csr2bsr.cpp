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

#include "testing_csr2bsr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t>
    csr2bsr_tuple;
typedef std::
    tuple<int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t, std::string>
        csr2bsr_bin_tuple;

// Random matrices
int csr2bsr_M_range[]         = {0, 872, 21453};
int csr2bsr_N_range[]         = {0, 623, 29285};
int csr2bsr_block_dim_range[] = {1, 2, 4, 7, 16};

hipsparseIndexBase_t csr2bsr_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

hipsparseIndexBase_t csr2bsr_bsr_base_range[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t csr2bsr_dir_range[] = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

// Matrices from files (float and double)
int csr2bsr_block_dim_range_bin[] = {5};

hipsparseIndexBase_t csr2bsr_csr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t csr2bsr_bsr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t csr2bsr_dir_range_bin[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string csr2bsr_bin[] = {"scircuit.bin", "nos2.bin", "nos4.bin", "nos6.bin", "sme3Dc.bin"};

class parameterized_csr2bsr : public testing::TestWithParam<csr2bsr_tuple>
{
protected:
    parameterized_csr2bsr() {}
    virtual ~parameterized_csr2bsr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2bsr_bin : public testing::TestWithParam<csr2bsr_bin_tuple>
{
protected:
    parameterized_csr2bsr_bin() {}
    virtual ~parameterized_csr2bsr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2bsr_arguments(csr2bsr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.block_dim = std::get<2>(tup);
    arg.baseA     = std::get<3>(tup);
    arg.baseB     = std::get<4>(tup);
    arg.dirA      = std::get<5>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csr2bsr_arguments(csr2bsr_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.block_dim = std::get<0>(tup);
    arg.baseA     = std::get<1>(tup);
    arg.baseB     = std::get<2>(tup);
    arg.dirA      = std::get<3>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(csr2bsr_bad_arg, csr2bsr)
{
    testing_csr2bsr_bad_arg<float>();
}

TEST_P(parameterized_csr2bsr, csr2bsr_float)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2bsr, csr2bsr_double)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2bsr, csr2bsr_float_complex)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2bsr, csr2bsr_double_complex)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2bsr_bin, csr2bsr_bin_float)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2bsr_bin, csr2bsr_bin_double)
{
    Arguments arg = setup_csr2bsr_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2bsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csr2bsr,
                         parameterized_csr2bsr,
                         testing::Combine(testing::ValuesIn(csr2bsr_M_range),
                                          testing::ValuesIn(csr2bsr_N_range),
                                          testing::ValuesIn(csr2bsr_block_dim_range),
                                          testing::ValuesIn(csr2bsr_csr_base_range),
                                          testing::ValuesIn(csr2bsr_bsr_base_range),
                                          testing::ValuesIn(csr2bsr_dir_range)));

INSTANTIATE_TEST_SUITE_P(csr2bsr_bin,
                         parameterized_csr2bsr_bin,
                         testing::Combine(testing::ValuesIn(csr2bsr_block_dim_range_bin),
                                          testing::ValuesIn(csr2bsr_csr_base_range_bin),
                                          testing::ValuesIn(csr2bsr_bsr_base_range_bin),
                                          testing::ValuesIn(csr2bsr_dir_range_bin),
                                          testing::ValuesIn(csr2bsr_bin)));
