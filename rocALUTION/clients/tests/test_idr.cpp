/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_idr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> idr_tuple;

std::vector<int>          idr_size    = {7, 63};
std::vector<std::string>  idr_precond = {"None", "SPAI", "GS", "ILU", "MCILU"};
std::vector<unsigned int> idr_format  = {1, 4, 5, 6};
std::vector<int>          idr_level   = {1, 2};

// Function to update tests if environment variable is set
void update_idr()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED"}))
    {
        idr_size.clear();
        idr_precond.clear();
        idr_format.clear();
        idr_level.clear();
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        idr_size.push_back(63);
        idr_precond.insert(idr_precond.end(), {"None", "MCILU"});
        idr_format.push_back(6);
        idr_level.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        idr_size.insert(idr_size.end(), {7, 63});
        idr_precond.insert(idr_precond.end(), {"MCILU"});
        idr_format.push_back(1);
        idr_level.push_back(2);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        idr_size.insert(idr_size.end(), {7, 63});
        idr_precond.insert(idr_precond.end(), {"SPAI", "GS", "ILU"});
        idr_format.insert(idr_format.end(), {4, 5});
        idr_level.insert(idr_level.end(), {1, 2});
    }
}

struct IDRInitializer
{
    IDRInitializer()
    {
        update_idr();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
IDRInitializer idr_initializer;

class parameterized_idr : public testing::TestWithParam<idr_tuple>
{
protected:
    parameterized_idr() {}
    virtual ~parameterized_idr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_idr_arguments(idr_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.index   = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_idr, idr_float)
{
    Arguments arg = setup_idr_arguments(GetParam());
    ASSERT_EQ(testing_idr<float>(arg), true);
}

TEST_P(parameterized_idr, idr_double)
{
    Arguments arg = setup_idr_arguments(GetParam());
    ASSERT_EQ(testing_idr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(idr,
                        parameterized_idr,
                        testing::Combine(testing::ValuesIn(idr_size),
                                         testing::ValuesIn(idr_precond),
                                         testing::ValuesIn(idr_format),
                                         testing::ValuesIn(idr_level)));
