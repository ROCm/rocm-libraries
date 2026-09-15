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

#include "testing_chebyshev.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, std::string, int, int> chebyshev_tuple;

std::vector<int>          chebyshev_size           = {7};
std::vector<unsigned int> chebyshev_format         = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  chebyshev_matrix_type    = {"Laplacian2D"};
std::vector<std::string>  chebyshev_precond        = {"None", "FSAI"};
std::vector<int>          chebyshev_rebuildnumeric = {0};
std::vector<int>          chebyshev_use_acc        = {0, 1};

// Function to update tests if environment variable is set
void update_chebyshev() {}

struct ChebyshevInitializer
{
    ChebyshevInitializer()
    {
        update_chebyshev();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
ChebyshevInitializer chebyshev_initializer;

class parameterized_chebyshev : public testing::TestWithParam<chebyshev_tuple>
{
protected:
    parameterized_chebyshev() {}
    virtual ~parameterized_chebyshev() {}
    virtual void SetUp()
    {
        if(!is_env_var_set("ROCALUTION_CODE_COVERAGE"))
        {
            GTEST_SKIP() << "Skipping Chebyshev tests as ROCLUTION_CODE_COVERAGE is not set.";
        }
    }
    virtual void TearDown() {}
};

Arguments setup_chebyshev_arguments(chebyshev_tuple tup)
{
    Arguments arg;
    arg.size           = std::get<0>(tup);
    arg.format         = std::get<1>(tup);
    arg.matrix_type    = std::get<2>(tup);
    arg.precond        = std::get<3>(tup);
    arg.rebuildnumeric = std::get<4>(tup);
    arg.use_acc        = std::get<5>(tup);
    return arg;
}

TEST_P(parameterized_chebyshev, chebyshev_float)
{
    Arguments arg = setup_chebyshev_arguments(GetParam());
    ASSERT_EQ(testing_chebyshev<float>(arg), true);
}

TEST_P(parameterized_chebyshev, chebyshev_double)
{
    Arguments arg = setup_chebyshev_arguments(GetParam());
    ASSERT_EQ(testing_chebyshev<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(chebyshev,
                        parameterized_chebyshev,
                        testing::Combine(testing::ValuesIn(chebyshev_size),
                                         testing::ValuesIn(chebyshev_format),
                                         testing::ValuesIn(chebyshev_matrix_type),
                                         testing::ValuesIn(chebyshev_precond),
                                         testing::ValuesIn(chebyshev_rebuildnumeric),
                                         testing::ValuesIn(chebyshev_use_acc)));
