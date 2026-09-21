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

#include "testing_cg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> cg_tuple;

std::vector<int>          cg_size    = {7, 63};
std::vector<std::string>  cg_precond = {"None", "FSAI", "SPAI", "TNS", "Jacobi", "IC", "MCSGS"};
std::vector<unsigned int> cg_format  = {1, 3, 4, 6};
std::vector<int>          cg_use_acc = {1};

// Function to update tests if environment variable is set
void update_cg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        cg_size.clear();
        cg_precond.clear();
        cg_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        cg_size.push_back(7);
        cg_precond.insert(cg_precond.end(),
                          {"None", "FSAI", "SPAI", "TNS", "Jacobi", "IC", "MCSGS"});
        cg_format.insert(cg_format.end(), {1, 3, 4, 6});
        cg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        cg_size.push_back(63);
        cg_precond.insert(cg_precond.end(), {"None", "FSAI"});
        cg_format.push_back(3);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        cg_size.insert(cg_size.end(), {7, 63});
        cg_precond.insert(cg_precond.end(), {"SPAI", "TNS"});
        cg_format.push_back(1);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        cg_size.insert(cg_size.end(), {7, 63});
        cg_precond.insert(cg_precond.end(), {"Jacobi", "IC", "MCSGS"});
        cg_format.insert(cg_format.end(), {4, 6});
    }
}

struct CGInitializer
{
    CGInitializer()
    {
        update_cg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
CGInitializer cg_initializer;

class parameterized_cg : public testing::TestWithParam<cg_tuple>
{
protected:
    parameterized_cg() {}
    virtual ~parameterized_cg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cg_arguments(cg_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.use_acc = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_cg, cg_float)
{
    Arguments arg = setup_cg_arguments(GetParam());
    ASSERT_EQ(testing_cg<float>(arg), true);
}

TEST_P(parameterized_cg, cg_double)
{
    Arguments arg = setup_cg_arguments(GetParam());
    ASSERT_EQ(testing_cg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(cg,
                        parameterized_cg,
                        testing::Combine(testing::ValuesIn(cg_size),
                                         testing::ValuesIn(cg_precond),
                                         testing::ValuesIn(cg_format),
                                         testing::ValuesIn(cg_use_acc)));
