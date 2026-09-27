/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_rsamg_extpi_host_fallback.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int> rsamg_extpi_fallback_tuple;

// Number of coarse points the single fine point is strongly connected to. 512 stays inside the
// LDS capacity of the fill kernel, 4200 forces the host fallback.
std::vector<int> rsamg_extpi_fallback_size = {512, 4200};

class parameterized_rsamg_extpi_host_fallback
    : public testing::TestWithParam<rsamg_extpi_fallback_tuple>
{
protected:
    parameterized_rsamg_extpi_host_fallback() {}
    virtual ~parameterized_rsamg_extpi_host_fallback() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_extpi_fallback_arguments(rsamg_extpi_fallback_tuple tup)
{
    Arguments arg;
    arg.size = std::get<0>(tup);
    return arg;
}

TEST_P(parameterized_rsamg_extpi_host_fallback, rsamg_extpi_host_fallback_float)
{
    Arguments arg = setup_rsamg_extpi_fallback_arguments(GetParam());
    testing_rsamg_extpi_host_fallback<float>(arg);
}

TEST_P(parameterized_rsamg_extpi_host_fallback, rsamg_extpi_host_fallback_double)
{
    Arguments arg = setup_rsamg_extpi_fallback_arguments(GetParam());
    testing_rsamg_extpi_host_fallback<double>(arg);
}

INSTANTIATE_TEST_CASE_P(rsamg_extpi_host_fallback,
                        parameterized_rsamg_extpi_host_fallback,
                        testing::Combine(testing::ValuesIn(rsamg_extpi_fallback_size)));
