/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_bicgstabl.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int, int> bicgstabl_tuple;

std::vector<int>         bicgstabl_size = {7, 63};
std::vector<std::string> bicgstabl_precond
    = {"None", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ItILU0", "ILUT", "IC", "MCGS", "MCILU"};
std::vector<unsigned int> bicgstabl_format  = {1, 4, 5, 6, 7};
std::vector<int>          bicgstabl_level   = {1, 2, 4};
std::vector<int>          bicgstabl_use_acc = {1};

// Function to update tests if environment variable is set
void update_bicgstabl()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        bicgstabl_size.clear();
        bicgstabl_precond.clear();
        bicgstabl_format.clear();
        bicgstabl_level.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        bicgstabl_size.push_back(7);
        bicgstabl_precond.insert(bicgstabl_precond.end(), {"None", "Jacobi"});
        bicgstabl_format.push_back(1);
        bicgstabl_level.push_back(1);
        bicgstabl_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        bicgstabl_size.push_back(63);
        bicgstabl_precond.insert(bicgstabl_precond.end(), {"None", "Jacobi"});
        bicgstabl_format.push_back(1);
        bicgstabl_level.push_back(4);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        bicgstabl_size.insert(bicgstabl_size.end(), {7, 63});
        bicgstabl_precond.insert(bicgstabl_precond.end(), {"SPAI", "TNS"});
        bicgstabl_format.insert(bicgstabl_format.end(), {4, 5});
        bicgstabl_level.insert(bicgstabl_level.end(), {1, 2});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        bicgstabl_size.insert(bicgstabl_size.end(), {7, 63});
        bicgstabl_precond.insert(bicgstabl_precond.end(), {"ILU", "IC"});
        bicgstabl_format.insert(bicgstabl_format.end(), {6, 7});
        bicgstabl_level.insert(bicgstabl_level.end(), {2, 4});
    }
}

struct BiCGStabLInitializer
{
    BiCGStabLInitializer()
    {
        update_bicgstabl();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
BiCGStabLInitializer bicgstabl_initializer;

class parameterized_bicgstabl : public testing::TestWithParam<bicgstabl_tuple>
{
protected:
    parameterized_bicgstabl() {}
    virtual ~parameterized_bicgstabl() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bicgstabl_arguments(bicgstabl_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.index   = std::get<3>(tup);
    arg.use_acc = std::get<4>(tup);
    return arg;
}

TEST_P(parameterized_bicgstabl, bicgstabl_float)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<float>(arg), true);
}

TEST_P(parameterized_bicgstabl, bicgstabl_double)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(bicgstabl,
                        parameterized_bicgstabl,
                        testing::Combine(testing::ValuesIn(bicgstabl_size),
                                         testing::ValuesIn(bicgstabl_precond),
                                         testing::ValuesIn(bicgstabl_format),
                                         testing::ValuesIn(bicgstabl_level),
                                         testing::ValuesIn(bicgstabl_use_acc)));
