/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_mixed_precision.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, std::string, int> mixed_precision_tuple;

std::vector<int>          mixed_precision_size           = {7};
std::vector<unsigned int> mixed_precision_format         = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  mixed_precision_matrix_type    = {"Laplacian2D"};
std::vector<std::string>  mixed_precision_precond        = {"None", "FSAI"};
std::vector<int>          mixed_precision_rebuildnumeric = {0};

// Function to update tests if environment variable is set
void update_mixed_precision() {}

struct MPInitializer
{
    MPInitializer()
    {
        update_mixed_precision();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
MPInitializer mixed_precision_initializer;

class parameterized_mixed_precision : public testing::TestWithParam<mixed_precision_tuple>
{
protected:
    parameterized_mixed_precision() {}
    virtual ~parameterized_mixed_precision() {}
    virtual void SetUp()
    {
        if(!is_env_var_set("ROCALUTION_CODE_COVERAGE"))
        {
            GTEST_SKIP()
                << "Skipping mixed precision tests as ROCALUTION_CODE_COVERAGE is not set.";
        }
    }
    virtual void TearDown() {}
};

Arguments setup_mixed_precision_arguments(mixed_precision_tuple tup)
{
    Arguments arg;
    arg.size           = std::get<0>(tup);
    arg.format         = std::get<1>(tup);
    arg.matrix_type    = std::get<2>(tup);
    arg.precond        = std::get<3>(tup);
    arg.rebuildnumeric = std::get<4>(tup);
    return arg;
}

TEST_P(parameterized_mixed_precision, mixed_precision_double_float)
{
    Arguments arg = setup_mixed_precision_arguments(GetParam());
    ASSERT_EQ(testing_mixed_precision<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(mixed_precision,
                        parameterized_mixed_precision,
                        testing::Combine(testing::ValuesIn(mixed_precision_size),
                                         testing::ValuesIn(mixed_precision_format),
                                         testing::ValuesIn(mixed_precision_matrix_type),
                                         testing::ValuesIn(mixed_precision_precond),
                                         testing::ValuesIn(mixed_precision_rebuildnumeric)));
