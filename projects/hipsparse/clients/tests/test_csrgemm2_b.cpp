/* ************************************************************************
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csrgemm2_b.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t base;

typedef std::tuple<int, int, double, base, base>    csrgemm2_b_tuple;
typedef std::tuple<double, base, base, std::string> csrgemm2_b_bin_tuple;

double csrgemm2_b_beta_range[] = {1.3};

int csrgemm2_b_M_range[] = {50, 647, 1799};
int csrgemm2_b_N_range[] = {0, 13, 523, 3712};

base csrgemm2_b_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgemm2_b_idxbaseD_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrgemm2_b_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csrgemm2_b : public testing::TestWithParam<csrgemm2_b_tuple>
{
protected:
    parameterized_csrgemm2_b() {}
    virtual ~parameterized_csrgemm2_b() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgemm2_b_bin : public testing::TestWithParam<csrgemm2_b_bin_tuple>
{
protected:
    parameterized_csrgemm2_b_bin() {}
    virtual ~parameterized_csrgemm2_b_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgemm2_b_arguments(csrgemm2_b_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.beta   = std::get<2>(tup);
    arg.baseC  = std::get<3>(tup);
    arg.baseD  = std::get<4>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrgemm2_b_arguments(csrgemm2_b_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.beta   = std::get<0>(tup);
    arg.baseC  = std::get<1>(tup);
    arg.baseD  = std::get<2>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(csrgemm2_b_bad_arg, csrgemm2_b_float)
{
    testing_csrgemm2_b_bad_arg<float>();
}

TEST_P(parameterized_csrgemm2_b, csrgemm2_b_float)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_b, csrgemm2_b_double)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_b, csrgemm2_b_float_complex)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_b, csrgemm2_b_double_complex)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_b_bin, csrgemm2_b_bin_float)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_b_bin, csrgemm2_b_bin_double)
{
    Arguments arg = setup_csrgemm2_b_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_b<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrgemm2_b,
                         parameterized_csrgemm2_b,
                         testing::Combine(testing::ValuesIn(csrgemm2_b_M_range),
                                          testing::ValuesIn(csrgemm2_b_N_range),
                                          testing::ValuesIn(csrgemm2_b_beta_range),
                                          testing::ValuesIn(csrgemm2_b_idxbaseC_range),
                                          testing::ValuesIn(csrgemm2_b_idxbaseD_range)));

INSTANTIATE_TEST_SUITE_P(csrgemm2_b_bin,
                         parameterized_csrgemm2_b_bin,
                         testing::Combine(testing::ValuesIn(csrgemm2_b_beta_range),
                                          testing::ValuesIn(csrgemm2_b_idxbaseC_range),
                                          testing::ValuesIn(csrgemm2_b_idxbaseD_range),
                                          testing::ValuesIn(csrgemm2_b_bin)));
#endif
