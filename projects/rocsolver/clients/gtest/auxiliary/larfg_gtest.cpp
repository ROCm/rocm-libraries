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

#include "common/auxiliary/testing_larfg.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using larfg_tuple = std::tuple<I, I>;

// case when n = 0 and incx = 0 also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<int> incx_range = {
    // invalid
    -1,
    0,
    // normal (valid) samples
    1,
    5,
    8,
    10,
};

const vector<int64_t> incx_range_64 = {
    // invalid
    -1,
    0,
    // normal (valid) samples
    1,
    5,
    8,
    10,
};

// for checkin_lapack tests
const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    1,
    12,
    20,
    35,
};

const vector<int64_t> n_size_range_64 = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    1,
    12,
    20,
    35,
};

// for daily_lapack tests
const vector<int> large_n_size_range = {
    192,
    640,
    1024,
    2547,
};

const vector<int64_t> large_n_size_range_64 = {
    192,
    640,
    1024,
    2547,
};

template <typename I>
Arguments larfg_setup_arguments(larfg_tuple<I> tup)
{
    I n_size = std::get<0>(tup);
    I inc = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", n_size);
    arg.set<I>("incx", inc);

    arg.timing = 0;

    return arg;
}

template <typename I>
class LARFG_BASE : public ::TestWithParam<larfg_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = larfg_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<I>("incx") == 0)
            testing_larfg_bad_arg<T, I>();

        testing_larfg<T, I>(arg);
    }
};

class LARFG : public LARFG_BASE<rocblas_int>
{
};
class LARFG_64 : public LARFG_BASE<int64_t>
{
};

// non-batch tests

TEST_P(LARFG, __float)
{
    run_tests<float>();
}

TEST_P(LARFG, __double)
{
    run_tests<double>();
}

TEST_P(LARFG, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARFG, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LARFG_64, __float)
{
    run_tests<float>();
}

TEST_P(LARFG_64, __double)
{
    run_tests<double>();
}

TEST_P(LARFG_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARFG_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFG,
                         Combine(ValuesIn(large_n_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LARFG, Combine(ValuesIn(n_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFG_64,
                         Combine(ValuesIn(large_n_size_range_64), ValuesIn(incx_range_64)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARFG_64,
                         Combine(ValuesIn(n_size_range_64), ValuesIn(incx_range_64)));
