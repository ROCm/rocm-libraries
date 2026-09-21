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

#include "testing_cr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> cr_tuple;

std::vector<int>         cr_size = {7, 63};
std::vector<std::string> cr_precond
    = {"None", "Chebyshev", "FSAI", "Jacobi", "SGS", "ILU", "ItILU0", "IC", "MCSGS"};
std::vector<unsigned int> cr_format  = {2, 4, 7};
std::vector<int>          cr_use_acc = {1};

// Function to update tests if environment variable is set
void update_cr()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        cr_size.clear();
        cr_precond.clear();
        cr_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        cr_size.push_back(7);
        cr_precond.insert(
            cr_precond.end(),
            {"None", "Chebyshev", "FSAI", "Jacobi", "SGS", "ILU", "ItILU0", "IC", "MCSGS"});
        cr_format.insert(cr_format.end(), {2, 4, 7});
        cr_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        cr_size.push_back(63);
        cr_precond.insert(cr_precond.end(), {"None", "SGS"});
        cr_format.push_back(4);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        cr_size.insert(cr_size.end(), {7, 63});
        cr_precond.insert(cr_precond.end(), {"Chebyshev", "FSAI", "Jacobi"});
        cr_format.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        cr_size.insert(cr_size.end(), {7, 63});
        cr_precond.insert(cr_precond.end(), {"ILU", "ItILU0", "IC", "MCSGS"});
        cr_format.push_back(7);
    }
}

struct CRInitializer
{
    CRInitializer()
    {
        update_cr();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
CRInitializer cr_initializer;

class parameterized_cr : public testing::TestWithParam<cr_tuple>
{
protected:
    parameterized_cr() {}
    virtual ~parameterized_cr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cr_arguments(cr_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.use_acc = std::get<3>(tup);
    return arg;
}
/* TODO there _MIGHT_ be some issue with float accuracy
TEST_P(parameterized_cr, cr_float)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<float>(arg), true);
}
*/
TEST_P(parameterized_cr, cr_double)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(cr,
                        parameterized_cr,
                        testing::Combine(testing::ValuesIn(cr_size),
                                         testing::ValuesIn(cr_precond),
                                         testing::ValuesIn(cr_format),
                                         testing::ValuesIn(cr_use_acc)));
