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

#include "testing_preconditioner.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string> preconditioner_tuple;

int         preconditioner_size[] = {7};
std::string preconditioner_type[] = {"Jacobi",
                                     "GS",
                                     "SGS",
                                     "ILU",
                                     "ILUT",
                                     "IC",
                                     "ItILU0",
                                     "AIChebyshev",
                                     "FSAI",
                                     "SPAI",
                                     "TNS",
                                     "AS",
                                     "RAS",
                                     "MultiColoredSGS",
                                     "MultiColoredGS",
                                     "MultiColoredILU",
                                     "MultiElimination",
                                     "VariablePreconditioner"};

class parameterized_preconditioner : public testing::TestWithParam<preconditioner_tuple>
{
protected:
    parameterized_preconditioner() {}
    virtual ~parameterized_preconditioner() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_preconditioner_arguments(preconditioner_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    return arg;
}

#define TEST_P_CASE(function, type, disable_accelerator)        \
    disable_accelerator_rocalution(disable_accelerator);        \
    set_device_rocalution(0);                                   \
    init_rocalution();                                          \
    Arguments arg = setup_preconditioner_arguments(GetParam()); \
    function<type>(arg);                                        \
    stop_rocalution();                                          \
    disable_accelerator_rocalution(false);

#define GENERATE_P_TEST_CASES_DEVICE(test_fixture, function, name) \
    TEST_P(test_fixture, name##_device_float)                      \
    {                                                              \
        TEST_P_CASE(function, float, false);                       \
    }                                                              \
    TEST_P(test_fixture, name##_device_double)                     \
    {                                                              \
        TEST_P_CASE(function, double, false);                      \
    }

#define GENERATE_P_TEST_CASES_HOST(test_fixture, function, name) \
    TEST_P(test_fixture, name##_host_float)                      \
    {                                                            \
        TEST_P_CASE(function, float, true);                      \
    }                                                            \
    TEST_P(test_fixture, name##_host_double)                     \
    {                                                            \
        TEST_P_CASE(function, double, true);                     \
    }
#define GENERATE_P_TEST_CASES(test_fixture, function, name)    \
    GENERATE_P_TEST_CASES_DEVICE(test_fixture, function, name) \
    GENERATE_P_TEST_CASES_HOST(test_fixture, function, name)

GENERATE_P_TEST_CASES_DEVICE(parameterized_preconditioner, testing_preconditioner_all, all);

INSTANTIATE_TEST_CASE_P(preconditioner,
                        parameterized_preconditioner,
                        testing::Combine(testing::ValuesIn(preconditioner_size),
                                         testing::ValuesIn(preconditioner_type)));
