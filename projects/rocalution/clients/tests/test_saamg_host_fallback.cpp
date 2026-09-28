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

#include "testing_saamg_host_fallback.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, double> saamg_fallback_tuple;

// Number of rays of the star matrix. 200 keeps the prolongation fill inside the LDS capacity,
// 600 forces the host fallback while the nnz count still runs on the accelerator.
std::vector<int>    saamg_fallback_size  = {200, 600};
std::vector<double> saamg_fallback_relax = {0.6667};

class parameterized_saamg_host_fallback : public testing::TestWithParam<saamg_fallback_tuple>
{
protected:
    parameterized_saamg_host_fallback() {}
    virtual ~parameterized_saamg_host_fallback() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_saamg_fallback_arguments(saamg_fallback_tuple tup)
{
    Arguments arg;
    arg.size  = std::get<0>(tup);
    arg.alpha = std::get<1>(tup);
    return arg;
}

TEST_P(parameterized_saamg_host_fallback, saamg_host_fallback_float)
{
    Arguments arg = setup_saamg_fallback_arguments(GetParam());
    testing_saamg_host_fallback<float>(arg);
}

TEST_P(parameterized_saamg_host_fallback, saamg_host_fallback_double)
{
    Arguments arg = setup_saamg_fallback_arguments(GetParam());
    testing_saamg_host_fallback<double>(arg);
}

INSTANTIATE_TEST_CASE_P(saamg_host_fallback,
                        parameterized_saamg_host_fallback,
                        testing::Combine(testing::ValuesIn(saamg_fallback_size),
                                         testing::ValuesIn(saamg_fallback_relax)));
