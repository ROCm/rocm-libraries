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

#include "testing_csrgeam.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t base;

typedef std::tuple<int, int, double, double, base, base, base>    csrgeam_tuple;
typedef std::tuple<double, double, base, base, base, std::string> csrgeam_bin_tuple;

double csrgeam_alpha_range[] = {0.0, 1.0};
double csrgeam_beta_range[]  = {0.0, 2.0};

int csrgeam_M_range[] = {0, 50, 647, 1799};
int csrgeam_N_range[] = {0, 13, 523, 3712};

base csrgeam_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgeam_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
base csrgeam_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrgeam_bin[] = {/*"rma10.bin",*/
                             "mac_econ_fwd500.bin",
                             /*"bibd_22_8.bin",*/
                             "mc2depi.bin",
                             "scircuit.bin",
                             /*"bmwcra_1.bin",*/
                             "nos1.bin",
                             "nos2.bin",
                             "nos3.bin",
                             "nos4.bin",
                             "nos5.bin",
                             "nos6.bin",
                             "nos7.bin"};

class parameterized_csrgeam : public testing::TestWithParam<csrgeam_tuple>
{
protected:
    parameterized_csrgeam() {}
    virtual ~parameterized_csrgeam() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgeam_bin : public testing::TestWithParam<csrgeam_bin_tuple>
{
protected:
    parameterized_csrgeam_bin() {}
    virtual ~parameterized_csrgeam_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgeam_arguments(csrgeam_tuple tup)
{
    Arguments arg;
    arg.M     = std::get<0>(tup);
    arg.N     = std::get<1>(tup);
    arg.alpha = std::get<2>(tup);
    arg.beta  = std::get<3>(tup);
    arg.baseA = std::get<4>(tup);
#ifdef __HIP_PLATFORM_NVIDIA__
    // There is a bug with index base in cusparse
    arg.baseB = std::get<4>(tup);
    arg.baseC = std::get<4>(tup);
#else
    arg.baseB = std::get<5>(tup);
    arg.baseC = std::get<6>(tup);
#endif
    arg.timing = 0;
    return arg;
}

Arguments setup_csrgeam_arguments(csrgeam_bin_tuple tup)
{
    Arguments arg;
    arg.M     = -99;
    arg.N     = -99;
    arg.alpha = std::get<0>(tup);
    arg.beta  = std::get<1>(tup);
    arg.baseA = std::get<2>(tup);
#ifdef __HIP_PLATFORM_NVIDIA__
    // There is a bug with index base in cusparse
    arg.baseB = std::get<2>(tup);
    arg.baseC = std::get<2>(tup);
#else
    arg.baseB = std::get<3>(tup);
    arg.baseC = std::get<4>(tup);
#endif
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(csrgeam_bad_arg, csrgeam_float)
{
    testing_csrgeam_bad_arg<float>();
}

TEST_P(parameterized_csrgeam, csrgeam_float)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam, csrgeam_double)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam, csrgeam_float_complex)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam, csrgeam_double_complex)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam_bin, csrgeam_bin_float)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgeam_bin, csrgeam_bin_double)
{
    Arguments arg = setup_csrgeam_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgeam<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrgeam,
                         parameterized_csrgeam,
                         testing::Combine(testing::ValuesIn(csrgeam_M_range),
                                          testing::ValuesIn(csrgeam_N_range),
                                          testing::ValuesIn(csrgeam_alpha_range),
                                          testing::ValuesIn(csrgeam_beta_range),
                                          testing::ValuesIn(csrgeam_idxbaseA_range),
                                          testing::ValuesIn(csrgeam_idxbaseB_range),
                                          testing::ValuesIn(csrgeam_idxbaseC_range)));

INSTANTIATE_TEST_SUITE_P(csrgeam_bin,
                         parameterized_csrgeam_bin,
                         testing::Combine(testing::ValuesIn(csrgeam_alpha_range),
                                          testing::ValuesIn(csrgeam_beta_range),
                                          testing::ValuesIn(csrgeam_idxbaseA_range),
                                          testing::ValuesIn(csrgeam_idxbaseB_range),
                                          testing::ValuesIn(csrgeam_idxbaseC_range),
                                          testing::ValuesIn(csrgeam_bin)));
#endif
