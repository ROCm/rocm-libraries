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

#include "testing_csric02.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                        base;
typedef hipsparseSolvePolicy_t                      solve_policy;
typedef std::tuple<int, base, solve_policy>         csric02_tuple;
typedef std::tuple<base, solve_policy, std::string> csric02_bin_tuple;

int csric02_M_range[] = {0, 50, 426};

base         csric02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
solve_policy csric02_solve_policy_range[]
    = {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL};

std::string csric02_bin[] = {"nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csric02 : public testing::TestWithParam<csric02_tuple>
{
protected:
    parameterized_csric02() {}
    virtual ~parameterized_csric02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csric02_bin : public testing::TestWithParam<csric02_bin_tuple>
{
protected:
    parameterized_csric02_bin() {}
    virtual ~parameterized_csric02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csric02_arguments(csric02_tuple tup)
{
    Arguments arg;
    arg.M            = std::get<0>(tup);
    arg.baseA        = std::get<1>(tup);
    arg.solve_policy = std::get<2>(tup);
    arg.timing       = 0;
    return arg;
}

Arguments setup_csric02_arguments(csric02_bin_tuple tup)
{
    Arguments arg;
    arg.M            = -99;
    arg.baseA        = std::get<0>(tup);
    arg.solve_policy = std::get<1>(tup);
    arg.timing       = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<2>(tup);

    // Get current executables absolute path

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(csric02_bad_arg, csric02_float)
{
    testing_csric02_bad_arg<float>();
}

TEST_P(parameterized_csric02, csric02_float)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_double)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_float_complex)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_double_complex)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02_bin, csric02_bin_float)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02_bin, csric02_bin_double)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csric02,
                         parameterized_csric02,
                         testing::Combine(testing::ValuesIn(csric02_M_range),
                                          testing::ValuesIn(csric02_idxbase_range),
                                          testing::ValuesIn(csric02_solve_policy_range)));

INSTANTIATE_TEST_SUITE_P(csric02_bin,
                         parameterized_csric02_bin,
                         testing::Combine(testing::ValuesIn(csric02_idxbase_range),
                                          testing::ValuesIn(csric02_solve_policy_range),
                                          testing::ValuesIn(csric02_bin)));
#endif
