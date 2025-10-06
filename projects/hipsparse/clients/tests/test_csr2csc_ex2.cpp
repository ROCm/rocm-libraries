/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csr2csc_ex2.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseAction_t, hipsparseIndexBase_t, hipsparseCsr2CscAlg_t>
    csr2csc_ex2_tuple;
typedef std::tuple<hipsparseAction_t, hipsparseIndexBase_t, hipsparseCsr2CscAlg_t, std::string>
    csr2csc_ex2_bin_tuple;

int csr2csc_ex2_M_range[] = {0, 10, 500, 872, 1000};
int csr2csc_ex2_N_range[] = {0, 33, 242, 623, 1000};

hipsparseAction_t csr2csc_ex2_action_range[]
    = {HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_SYMBOLIC};

hipsparseIndexBase_t csr2csc_ex2_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

hipsparseCsr2CscAlg_t csr2csc_ex2_alg_range[] = {HIPSPARSE_CSR2CSC_ALG1};

std::string csr2csc_ex2_bin[] = {"rma10.bin",
                                 "mc2depi.bin",
                                 "scircuit.bin",
                                 "nos1.bin",
                                 "nos2.bin",
                                 "nos3.bin",
                                 "nos4.bin",
                                 "nos5.bin",
                                 "nos6.bin",
                                 "nos7.bin",
                                 "webbase-1M.bin",
                                 "shipsec1.bin"};

class parameterized_csr2csc_ex2 : public testing::TestWithParam<csr2csc_ex2_tuple>
{
protected:
    parameterized_csr2csc_ex2() {}
    virtual ~parameterized_csr2csc_ex2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2csc_ex2_bin : public testing::TestWithParam<csr2csc_ex2_bin_tuple>
{
protected:
    parameterized_csr2csc_ex2_bin() {}
    virtual ~parameterized_csr2csc_ex2_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2csc_ex2_arguments(csr2csc_ex2_tuple tup)
{
    Arguments arg;
    arg.M           = std::get<0>(tup);
    arg.N           = std::get<1>(tup);
    arg.action      = std::get<2>(tup);
    arg.baseA       = std::get<3>(tup);
    arg.csr2csc_alg = std::get<4>(tup);
    arg.timing      = 0;
    return arg;
}

Arguments setup_csr2csc_ex2_arguments(csr2csc_ex2_bin_tuple tup)
{
    Arguments arg;
    arg.M           = -99;
    arg.N           = -99;
    arg.action      = std::get<0>(tup);
    arg.baseA       = std::get<1>(tup);
    arg.csr2csc_alg = std::get<2>(tup);
    arg.timing      = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
TEST(csr2csc_ex2_bad_arg, csr2csc_ex2)
{
    testing_csr2csc_ex2_bad_arg<float>();
}

TEST_P(parameterized_csr2csc_ex2, csr2csc_ex2_int8)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<int8_t>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2, csr2csc_ex2_float)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2, csr2csc_ex2_double)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2, csr2csc_ex2_float_complex)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2, csr2csc_ex2_double_complex)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2_bin, csr2csc_ex2_bin_float)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc_ex2_bin, csr2csc_ex2_bin_double)
{
    Arguments arg = setup_csr2csc_ex2_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc_ex2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csr2csc_ex2,
                         parameterized_csr2csc_ex2,
                         testing::Combine(testing::ValuesIn(csr2csc_ex2_M_range),
                                          testing::ValuesIn(csr2csc_ex2_N_range),
                                          testing::ValuesIn(csr2csc_ex2_action_range),
                                          testing::ValuesIn(csr2csc_ex2_base_range),
                                          testing::ValuesIn(csr2csc_ex2_alg_range)));

INSTANTIATE_TEST_SUITE_P(csr2csc_ex2_bin,
                         parameterized_csr2csc_ex2_bin,
                         testing::Combine(testing::ValuesIn(csr2csc_ex2_action_range),
                                          testing::ValuesIn(csr2csc_ex2_base_range),
                                          testing::ValuesIn(csr2csc_ex2_alg_range),
                                          testing::ValuesIn(csr2csc_ex2_bin)));
#endif
