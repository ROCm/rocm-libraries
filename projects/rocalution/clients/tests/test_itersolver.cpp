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

#include "testing_itsolver.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, int> itsolver_tuple;

std::vector<int>          itsolver_size        = {7};
std::vector<unsigned int> itsolver_format      = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  itsolver_matrix_type = {"Laplacian2D"};
std::vector<int>          itsolver_use_acc     = {0, 1};

// Function to update tests if environment variable is set
void update_itsolver() {}

struct ItsolverInitializer
{
    ItsolverInitializer()
    {
        update_itsolver();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
ItsolverInitializer itsolver_initializer;

class parameterized_itsolver : public testing::TestWithParam<itsolver_tuple>
{
protected:
    parameterized_itsolver() {}
    virtual ~parameterized_itsolver() {}
    virtual void SetUp()
    {
        if(!is_env_var_set("ROCALUTION_CODE_COVERAGE"))
        {
            GTEST_SKIP() << "Skipping tests as ROCALUTION_CODE_COVERAGE is not set.";
        }
    }
    virtual void TearDown() {}
};

Arguments setup_itsolver_arguments(itsolver_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.format      = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    arg.use_acc     = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_itsolver, itsolver_float)
{
    Arguments arg = setup_itsolver_arguments(GetParam());
    ASSERT_EQ(testing_itsolver<float>(arg), true);
}

TEST_P(parameterized_itsolver, itsolver_double)
{
    Arguments arg = setup_itsolver_arguments(GetParam());
    ASSERT_EQ(testing_itsolver<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(itsolver,
                        parameterized_itsolver,
                        testing::Combine(testing::ValuesIn(itsolver_size),
                                         testing::ValuesIn(itsolver_format),
                                         testing::ValuesIn(itsolver_matrix_type),
                                         testing::ValuesIn(itsolver_use_acc)));
