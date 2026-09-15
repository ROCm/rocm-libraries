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

#include "testing_fcg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> fcg_tuple;

std::vector<int>         fcg_size = {7, 63};
std::vector<std::string> fcg_precond
    = {"None", "Chebyshev", "SPAI", "TNS", "ItILU0", "ILUT", "MCSGS"};
std::vector<unsigned int> fcg_format  = {2, 5, 6, 7};
std::vector<int>          fcg_use_acc = {1};

// Function to update tests if environment variable is set
void update_fcg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        fcg_size.clear();
        fcg_precond.clear();
        fcg_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        fcg_size.push_back(7);
        fcg_precond.insert(fcg_precond.end(),
                           {"None", "Chebyshev", "SPAI", "TNS", "ItILU0", "ILUT", "MCSGS"});
        fcg_format.insert(fcg_format.end(), {2, 5, 6, 7});
        fcg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        fcg_size.push_back(63);
        fcg_precond.insert(fcg_precond.end(), {"None", "SPAI"});
        fcg_format.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        fcg_size.insert(fcg_size.end(), {7, 63});
        fcg_precond.insert(fcg_precond.end(), {"TNS", "MCSGS"});
        fcg_format.push_back(5);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        fcg_size.insert(fcg_size.end(), {7, 63});
        fcg_precond.insert(fcg_precond.end(), {"ILUT", "Chebyshev"});
        fcg_format.insert(fcg_format.end(), {6, 7});
    }
}

struct FCGInitializer
{
    FCGInitializer()
    {
        update_fcg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
FCGInitializer fcg_initializer;

class parameterized_fcg : public testing::TestWithParam<fcg_tuple>
{
protected:
    parameterized_fcg() {}
    virtual ~parameterized_fcg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fcg_arguments(fcg_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.use_acc = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_fcg, fcg_float)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<float>(arg), true);
}

TEST_P(parameterized_fcg, fcg_double)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fcg,
                        parameterized_fcg,
                        testing::Combine(testing::ValuesIn(fcg_size),
                                         testing::ValuesIn(fcg_precond),
                                         testing::ValuesIn(fcg_format),
                                         testing::ValuesIn(fcg_use_acc)));
