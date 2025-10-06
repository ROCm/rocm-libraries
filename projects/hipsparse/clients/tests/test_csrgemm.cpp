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

#include "testing_csrgemm.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t base;
typedef hipsparseOperation_t trans;

typedef std::tuple<int, int, int, base, base, base, trans, trans> csrgemm_tuple;
typedef std::tuple<base, base, base, trans, trans, std::string>   csrgemm_bin_tuple;

int csrgemm_M_range[] = {0, 647, 1799};
int csrgemm_N_range[] = {0, 523, 3712};
int csrgemm_K_range[] = {0, 254, 1942};

base csrgemm_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
base csrgemm_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ONE};
base csrgemm_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

trans csrgemm_transA_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
trans csrgemm_transB_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};

std::string csrgemm_bin[] = {"mc2depi.bin",
                             "scircuit.bin",
                             "nos1.bin",
                             "nos2.bin",
                             "nos3.bin",
                             "nos4.bin",
                             "nos5.bin",
                             "nos6.bin",
                             "nos7.bin"};

class parameterized_csrgemm : public testing::TestWithParam<csrgemm_tuple>
{
protected:
    parameterized_csrgemm() {}
    virtual ~parameterized_csrgemm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgemm_bin : public testing::TestWithParam<csrgemm_bin_tuple>
{
protected:
    parameterized_csrgemm_bin() {}
    virtual ~parameterized_csrgemm_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgemm_arguments(csrgemm_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.K      = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.baseB  = std::get<4>(tup);
    arg.baseC  = std::get<5>(tup);
    arg.transA = std::get<6>(tup);
    arg.transB = std::get<7>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrgemm_arguments(csrgemm_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.K      = -99;
    arg.baseA  = std::get<0>(tup);
    arg.baseB  = std::get<1>(tup);
    arg.baseC  = std::get<2>(tup);
    arg.transA = std::get<3>(tup);
    arg.transB = std::get<4>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(csrgemm_bad_arg, csrgemm_float)
{
    testing_csrgemm_bad_arg<float>();
}

TEST_P(parameterized_csrgemm, csrgemm_float)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm, csrgemm_double)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm, csrgemm_float_complex)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm, csrgemm_double_complex)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm_bin, csrgemm_bin_float)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrgemm_bin, csrgemm_bin_double)
{
    Arguments arg = setup_csrgemm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrgemm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrgemm,
                         parameterized_csrgemm,
                         testing::Combine(testing::ValuesIn(csrgemm_M_range),
                                          testing::ValuesIn(csrgemm_N_range),
                                          testing::ValuesIn(csrgemm_K_range),
                                          testing::ValuesIn(csrgemm_idxbaseA_range),
                                          testing::ValuesIn(csrgemm_idxbaseB_range),
                                          testing::ValuesIn(csrgemm_idxbaseC_range),
                                          testing::ValuesIn(csrgemm_transA_range),
                                          testing::ValuesIn(csrgemm_transB_range)));

INSTANTIATE_TEST_SUITE_P(csrgemm_bin,
                         parameterized_csrgemm_bin,
                         testing::Combine(testing::ValuesIn(csrgemm_idxbaseA_range),
                                          testing::ValuesIn(csrgemm_idxbaseB_range),
                                          testing::ValuesIn(csrgemm_idxbaseC_range),
                                          testing::ValuesIn(csrgemm_transA_range),
                                          testing::ValuesIn(csrgemm_transB_range),
                                          testing::ValuesIn(csrgemm_bin)));
#endif
