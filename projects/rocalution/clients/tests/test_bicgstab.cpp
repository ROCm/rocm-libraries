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

#include "testing_bicgstab.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> bicgstab_tuple;

std::vector<int>         bicgstab_size = {7, 63};
std::vector<std::string> bicgstab_precond
    = {"None", "Chebyshev", "TNS", "Jacobi", "ItILU0", "ILUT", "MCGS", "MCILU"};
std::vector<unsigned int> bicgstab_format  = {1, 2, 4, 6};
std::vector<int>          bicgstab_use_acc = {1};

// Function to update tests if environment variable is set
void update_bicgstab()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        bicgstab_size.clear();
        bicgstab_precond.clear();
        bicgstab_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        bicgstab_size.push_back(7);
        bicgstab_precond.insert(
            bicgstab_precond.end(),
            {"None", "Chebyshev", "TNS", "Jacobi", "ItILU0", "ILUT", "MCGS", "MCILU"});
        bicgstab_format.insert(bicgstab_format.end(), {1, 2, 4, 6});
        bicgstab_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        bicgstab_size.push_back(63);
        bicgstab_precond.insert(bicgstab_precond.end(), {"None", "Chebyshev"});
        bicgstab_format.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        bicgstab_size.insert(bicgstab_size.end(), {7, 63});
        bicgstab_precond.insert(bicgstab_precond.end(), {"TNS", "MCILU"});
        bicgstab_format.insert(bicgstab_format.end(), {1, 4});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        bicgstab_size.insert(bicgstab_size.end(), {7, 63});
        bicgstab_precond.insert(bicgstab_precond.end(), {"ItILU0", "ILUT"});
        bicgstab_format.push_back(6);
    }
}

struct BiCGStabInitializer
{
    BiCGStabInitializer()
    {
        update_bicgstab();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
BiCGStabInitializer bicgstab_initializer;

class parameterized_bicgstab : public testing::TestWithParam<bicgstab_tuple>
{
protected:
    parameterized_bicgstab() {}
    virtual ~parameterized_bicgstab() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bicgstab_arguments(bicgstab_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.use_acc = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_bicgstab, bicgstab_float)
{
    Arguments arg = setup_bicgstab_arguments(GetParam());
    ASSERT_EQ(testing_bicgstab<float>(arg), true);
}

TEST_P(parameterized_bicgstab, bicgstab_double)
{
    Arguments arg = setup_bicgstab_arguments(GetParam());
    ASSERT_EQ(testing_bicgstab<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(bicgstab,
                        parameterized_bicgstab,
                        testing::Combine(testing::ValuesIn(bicgstab_size),
                                         testing::ValuesIn(bicgstab_precond),
                                         testing::ValuesIn(bicgstab_format),
                                         testing::ValuesIn(bicgstab_use_acc)));
