/* ************************************************************************
 * Copyright (C) 2018 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csrgeam2.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t base;

typedef std::tuple<int, int, double, double, base, base, base>    csrgeam2_tuple;
typedef std::tuple<double, double, base, base, base, std::string> csrgeam2_bin_tuple;

double csrgeam2_alpha_range[] = {0.0, 2.0};
double csrgeam2_beta_range[]  = {0.0, 1.0};

int csrgeam2_M_range[] = {0, 50, 647, 1799};
int csrgeam2_N_range[] = {0, 13, 523, 3712};

base csrgeam2_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgeam2_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgeam2_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrgeam2_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csrgeam2 : public testing::TestWithParam<csrgeam2_tuple>
{
protected:
    parameterized_csrgeam2() {}
    virtual ~parameterized_csrgeam2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgeam2_bin : public testing::TestWithParam<csrgeam2_bin_tuple>
{
protected:
    parameterized_csrgeam2_bin() {}
    virtual ~parameterized_csrgeam2_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgeam2_arguments(csrgeam2_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.beta   = std::get<3>(tup);
    arg.baseA  = std::get<4>(tup);
    arg.baseB  = std::get<5>(tup);
    arg.baseC  = std::get<6>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrgeam2_arguments(csrgeam2_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.alpha  = std::get<0>(tup);
    arg.beta   = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.baseB  = std::get<3>(tup);
    arg.baseC  = std::get<4>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION))
TEST(csrgeam2_bad_arg, csrgeam2_float)
{
    testing_csrgeam2_bad_arg<float>();
}

TEST_P(parameterized_csrgeam2, csrgeam2_float)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam2, csrgeam2_double)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam2, csrgeam2_float_complex)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam2, csrgeam2_double_complex)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam2_bin, csrgeam2_bin_float)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam2_bin, csrgeam2_bin_double)
{
    Arguments arg = setup_csrgeam2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrgeam2,
                         parameterized_csrgeam2,
                         testing::Combine(testing::ValuesIn(csrgeam2_M_range),
                                          testing::ValuesIn(csrgeam2_N_range),
                                          testing::ValuesIn(csrgeam2_alpha_range),
                                          testing::ValuesIn(csrgeam2_beta_range),
                                          testing::ValuesIn(csrgeam2_idxbaseA_range),
                                          testing::ValuesIn(csrgeam2_idxbaseB_range),
                                          testing::ValuesIn(csrgeam2_idxbaseC_range)));

INSTANTIATE_TEST_SUITE_P(csrgeam2_bin,
                         parameterized_csrgeam2_bin,
                         testing::Combine(testing::ValuesIn(csrgeam2_alpha_range),
                                          testing::ValuesIn(csrgeam2_beta_range),
                                          testing::ValuesIn(csrgeam2_idxbaseA_range),
                                          testing::ValuesIn(csrgeam2_idxbaseB_range),
                                          testing::ValuesIn(csrgeam2_idxbaseC_range),
                                          testing::ValuesIn(csrgeam2_bin)));
#endif
