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

#include "testing_qmrcgstab.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int> qmrcgstab_tuple;

std::vector<int>         qmrcgstab_size = {7, 63};
std::vector<std::string> qmrcgstab_precond
    = {"None", "Chebyshev", "SPAI", "Jacobi", "GS", "ILU", "ItILU0"};
std::vector<unsigned int> qmrcgstab_format  = {1, 2, 3, 7};
std::vector<int>          qmrcgstab_use_acc = {1};

// Function to update tests if environment variable is set
void update_qmrcgstab()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        qmrcgstab_size.clear();
        qmrcgstab_precond.clear();
        qmrcgstab_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        qmrcgstab_size.push_back(7);
        qmrcgstab_precond.insert(qmrcgstab_precond.end(),
                                 {"None", "Chebyshev", "SPAI", "Jacobi", "GS", "ILU", "ItILU0"});
        qmrcgstab_format.insert(qmrcgstab_format.end(), {1, 2, 3, 7});
        qmrcgstab_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        qmrcgstab_size.push_back(63);
        qmrcgstab_precond.insert(qmrcgstab_precond.end(), {"None", "GS"});
        qmrcgstab_format.push_back(7);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        qmrcgstab_size.insert(qmrcgstab_size.end(), {7, 63});
        qmrcgstab_precond.insert(qmrcgstab_precond.end(), {"Chebyshev", "ItILU0"});
        qmrcgstab_format.insert(qmrcgstab_format.end(), {1, 2});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        qmrcgstab_size.insert(qmrcgstab_size.end(), {7, 63});
        qmrcgstab_precond.insert(qmrcgstab_precond.end(), {"SPAI", "ILU"});
        qmrcgstab_format.insert(qmrcgstab_format.end(), {1, 3});
    }
}

struct QMRCGStabInitializer
{
    QMRCGStabInitializer()
    {
        update_qmrcgstab();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
QMRCGStabInitializer qmrcgstab_initializer;

class parameterized_qmrcgstab : public testing::TestWithParam<qmrcgstab_tuple>
{
protected:
    parameterized_qmrcgstab() {}
    virtual ~parameterized_qmrcgstab() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_qmrcgstab_arguments(qmrcgstab_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.use_acc = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_qmrcgstab, qmrcgstab_float)
{
    Arguments arg = setup_qmrcgstab_arguments(GetParam());
    ASSERT_EQ(testing_qmrcgstab<float>(arg), true);
}

TEST_P(parameterized_qmrcgstab, qmrcgstab_double)
{
    Arguments arg = setup_qmrcgstab_arguments(GetParam());
    ASSERT_EQ(testing_qmrcgstab<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(qmrcgstab,
                        parameterized_qmrcgstab,
                        testing::Combine(testing::ValuesIn(qmrcgstab_size),
                                         testing::ValuesIn(qmrcgstab_precond),
                                         testing::ValuesIn(qmrcgstab_format),
                                         testing::ValuesIn(qmrcgstab_use_acc)));
