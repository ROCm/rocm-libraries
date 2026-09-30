/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_lu.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, int> lu_tuple;

std::vector<int>          lu_size             = {7, 16, 21};
std::vector<unsigned int> lu_format           = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  lu_matrix_type      = {"Laplacian2D"};
std::vector<int>          lu_use_host_and_acc = {0};

// Function to update tests if environment variable is set
void update_lu()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        lu_size.clear();
        lu_format.clear();
        lu_use_host_and_acc.clear();

        lu_size.push_back(16);
        lu_format.push_back(2);
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        lu_size.clear();
        lu_format.clear();

        lu_size.push_back(7);
        lu_format.insert(lu_format.end(), {1, 2, 3, 4, 5, 6, 7});
        lu_use_host_and_acc.push_back(1);
    }
    else if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                                "ROCALUTION_EMULATION_REGRESSION",
                                "ROCALUTION_EMULATION_EXTENDED"}))
    {
        lu_use_host_and_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        lu_size.push_back(16);
        lu_format.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        lu_size.insert(lu_size.end(), {7, 16});
        lu_format.insert(lu_format.end(), {1, 3});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        lu_size.insert(lu_size.end(), {16, 21});
        lu_format.insert(lu_format.end(), {4, 5, 6, 7});
    }
}

struct LUInitializer
{
    LUInitializer()
    {
        update_lu();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
LUInitializer lu_initializer;

class parameterized_lu : public testing::TestWithParam<lu_tuple>
{
protected:
    parameterized_lu() {}
    virtual ~parameterized_lu() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_lu_arguments(lu_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.format      = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    arg.use_acc     = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_lu, lu_float)
{
    Arguments arg = setup_lu_arguments(GetParam());
    ASSERT_EQ(testing_lu<float>(arg), true);
}

TEST_P(parameterized_lu, lu_double)
{
    Arguments arg = setup_lu_arguments(GetParam());
    ASSERT_EQ(testing_lu<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(lu,
                        parameterized_lu,
                        testing::Combine(testing::ValuesIn(lu_size),
                                         testing::ValuesIn(lu_format),
                                         testing::ValuesIn(lu_matrix_type),
                                         testing::ValuesIn(lu_use_host_and_acc)));
