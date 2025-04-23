/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "common/auxiliary/testing_lacgv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using lacgv_tuple = vector<I>;

// each range is a {n,inc}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {1, 0},
    // normal (valid) samples
    {10, 1},
    {10, -1},
    {20, 2},
    {30, 3},
    {30, -3}};

const vector<vector<int64_t>> range_64 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {1, 0},
    // normal (valid) samples
    {10, 1},
    {10, -1},
    {20, 2},
    {30, 3},
    {30, -3}};

// for daily_lapack tests
const vector<vector<int>> large_range
    = {{192, 10}, {192, -10}, {250, 20}, {500, 30}, {1500, 40}, {1500, -40}};

const vector<vector<int64_t>> large_range_64
    = {{192, 10}, {192, -10}, {250, 20}, {500, 30}, {1500, 40}, {1500, -40}};

template <typename I>
Arguments lacgv_setup_arguments(lacgv_tuple<I> tup)
{
    Arguments arg;

    arg.set<I>("n", tup[0]);
    arg.set<I>("incx", tup[1]);

    return arg;
}

template <typename I>
class LACGV_BASE : public ::TestWithParam<lacgv_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lacgv_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0)
            testing_lacgv_bad_arg<T, I>();

        testing_lacgv<T, I>(arg);
    }
};

class LACGV : public LACGV_BASE<rocblas_int>
{
};
class LACGV_64 : public LACGV_BASE<int64_t>
{
};

// non-batch tests

TEST_P(LACGV, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LACGV, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LACGV_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LACGV_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LACGV, ValuesIn(large_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LACGV, ValuesIn(range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, LACGV_64, ValuesIn(large_range_64));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LACGV_64, ValuesIn(range_64));
