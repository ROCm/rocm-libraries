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

#include "testing_csrgemm2_a.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t base;

typedef std::tuple<int, int, int, double, base, base, base> csrgemm2_a_tuple;
typedef std::tuple<double, base, base, base, std::string>   csrgemm2_a_bin_tuple;

double csrgemm2_a_alpha_range[] = {2.0};

int csrgemm2_a_M_range[] = {0, 50, 647, 1799};
int csrgemm2_a_N_range[] = {13, 523, 3712};
int csrgemm2_a_K_range[] = {0, 50, 254, 1942};

base csrgemm2_a_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgemm2_a_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgemm2_a_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrgemm2_a_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csrgemm2_a : public testing::TestWithParam<csrgemm2_a_tuple>
{
protected:
    parameterized_csrgemm2_a() {}
    virtual ~parameterized_csrgemm2_a() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgemm2_a_bin : public testing::TestWithParam<csrgemm2_a_bin_tuple>
{
protected:
    parameterized_csrgemm2_a_bin() {}
    virtual ~parameterized_csrgemm2_a_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgemm2_a_arguments(csrgemm2_a_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.K      = std::get<2>(tup);
    arg.alpha  = std::get<3>(tup);
    arg.baseA  = std::get<4>(tup);
    arg.baseB  = std::get<5>(tup);
    arg.baseC  = std::get<6>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrgemm2_a_arguments(csrgemm2_a_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.K      = -99;
    arg.alpha  = std::get<0>(tup);
    arg.baseA  = std::get<1>(tup);
    arg.baseB  = std::get<2>(tup);
    arg.baseC  = std::get<3>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(csrgemm2_a_bad_arg, csrgemm2_a_float)
{
    testing_csrgemm2_a_bad_arg<float>();
}

TEST_P(parameterized_csrgemm2_a, csrgemm2_a_float)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_a, csrgemm2_a_double)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_a, csrgemm2_a_float_complex)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_a, csrgemm2_a_double_complex)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_a_bin, csrgemm2_a_bin_float)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm2_a_bin, csrgemm2_a_bin_double)
{
    Arguments arg = setup_csrgemm2_a_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm2_a<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrgemm2_a,
                         parameterized_csrgemm2_a,
                         testing::Combine(testing::ValuesIn(csrgemm2_a_M_range),
                                          testing::ValuesIn(csrgemm2_a_N_range),
                                          testing::ValuesIn(csrgemm2_a_K_range),
                                          testing::ValuesIn(csrgemm2_a_alpha_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseA_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseB_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseC_range)));

INSTANTIATE_TEST_SUITE_P(csrgemm2_a_bin,
                         parameterized_csrgemm2_a_bin,
                         testing::Combine(testing::ValuesIn(csrgemm2_a_alpha_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseA_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseB_range),
                                          testing::ValuesIn(csrgemm2_a_idxbaseC_range),
                                          testing::ValuesIn(csrgemm2_a_bin)));
#endif
