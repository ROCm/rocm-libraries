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

#include "testing_gemmi.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int, int, double, double>    gemmi_tuple;
typedef std::tuple<int, double, double, std::string> gemmi_bin_tuple;

int gemmi_M_range[] = {0, 19, 78, 482};
int gemmi_N_range[] = {0, 42, 275, 759};
int gemmi_K_range[] = {0, 50, 173, 1375};

double gemmi_alpha_range[] = {-0.5};
double gemmi_beta_range[]  = {0.5};

std::string gemmi_bin[] = {"nos1.bin", "nos3.bin", "nos5.bin", "nos7.bin"};

class parameterized_gemmi : public testing::TestWithParam<gemmi_tuple>
{
protected:
    parameterized_gemmi() {}
    virtual ~parameterized_gemmi() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gemmi_bin : public testing::TestWithParam<gemmi_bin_tuple>
{
protected:
    parameterized_gemmi_bin() {}
    virtual ~parameterized_gemmi_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gemmi_arguments(gemmi_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.K      = std::get<2>(tup);
    arg.alpha  = std::get<3>(tup);
    arg.beta   = std::get<4>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_gemmi_arguments(gemmi_bin_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = -99;
    arg.K      = -99;
    arg.alpha  = std::get<1>(tup);
    arg.beta   = std::get<2>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(gemmi_bad_arg, gemmi_float)
{
    testing_gemmi_bad_arg<float>();
}

TEST_P(parameterized_gemmi, gemmi_float)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemmi, gemmi_double)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemmi, gemmi_float_complex)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemmi, gemmi_double_complex)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemmi_bin, gemmi_bin_float)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemmi_bin, gemmi_bin_double)
{
    Arguments arg = setup_gemmi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemmi<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gemmi,
                         parameterized_gemmi,
                         testing::Combine(testing::ValuesIn(gemmi_M_range),
                                          testing::ValuesIn(gemmi_N_range),
                                          testing::ValuesIn(gemmi_K_range),
                                          testing::ValuesIn(gemmi_alpha_range),
                                          testing::ValuesIn(gemmi_beta_range)));

INSTANTIATE_TEST_SUITE_P(gemmi_bin,
                         parameterized_gemmi_bin,
                         testing::Combine(testing::ValuesIn(gemmi_N_range),
                                          testing::ValuesIn(gemmi_alpha_range),
                                          testing::ValuesIn(gemmi_beta_range),
                                          testing::ValuesIn(gemmi_bin)));
#endif
