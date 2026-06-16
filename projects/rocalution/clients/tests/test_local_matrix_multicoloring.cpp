/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_local_matrix_multicoloring.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> local_matrix_multicoloring_tuple;

int          local_matrix_multicoloring_size[]   = {10, 17, 21};
std::string  local_matrix_multicoloring_type[]   = {"Laplacian2D", "PermutedIdentity", "Random"};
unsigned int local_matrix_multicoloring_format[] = {1};

class parameterized_local_matrix_multicoloring
    : public testing::TestWithParam<local_matrix_multicoloring_tuple>
{
protected:
    parameterized_local_matrix_multicoloring() {}
    virtual ~parameterized_local_matrix_multicoloring() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_local_matrix_multicoloring_arguments(local_matrix_multicoloring_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.matrix_type = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_local_matrix_multicoloring, local_matrix_multicoloring_float)
{
    Arguments arg = setup_local_matrix_multicoloring_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_multicoloring<float>(arg), true);
}

TEST_P(parameterized_local_matrix_multicoloring, local_matrix_multicoloring_double)
{
    Arguments arg = setup_local_matrix_multicoloring_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_multicoloring<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_multicoloring,
                        parameterized_local_matrix_multicoloring,
                        testing::Combine(testing::ValuesIn(local_matrix_multicoloring_size),
                                         testing::ValuesIn(local_matrix_multicoloring_type),
                                         testing::ValuesIn(local_matrix_multicoloring_format)));
