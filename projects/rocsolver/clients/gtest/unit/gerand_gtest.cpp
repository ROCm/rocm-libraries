/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/unit/testing_gerand.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// tuple: {M, N, lda}
using gerand_tuple = std::tuple<vector<int>, int>;

// each size_range vector is {M, lda}; N is the second element of the tuple.

// case when M == 0 also executes the bad-arguments test

// for checkin_lapack tests
const vector<vector<int>> gerand_size_range = {
    // quick return
    {0,  1},
    // normal (valid) samples
    {1,  1},
    {10, 10},
    {12, 15},  // lda > m (tests padding)
    {20, 20},
    {35, 50},  // lda > m (tests padding)
};

// for daily_lapack tests
const vector<vector<int>> large_gerand_size_range = {
    {512,  512},
    {1024, 1024},
    {2000, 2048},
};

const vector<int> n_range       = {0, 1, 15, 20};
const vector<int> large_n_range = {512, 1024, 2000};

Arguments gerand_setup_arguments(gerand_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    int n = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m",   size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    arg.set<rocblas_int>("n",   n);

    arg.timing = 0;

    return arg;
}

class GERAND : public ::TestWithParam<gerand_tuple>
{
protected:
    void TearDown() override {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = gerand_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0)
        {
            gerand_checkBadArgs<T>();
        }

        testing_gerand<T>(arg);
    }
};

// non-batch tests (gerand is a pure host function, no batching)

TEST_P(GERAND, __float)
{
    run_tests<float>();
}

TEST_P(GERAND, __double)
{
    run_tests<double>();
}

TEST_P(GERAND, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GERAND, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GERAND,
                         Combine(ValuesIn(large_gerand_size_range), ValuesIn(large_n_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GERAND,
                         Combine(ValuesIn(gerand_size_range), ValuesIn(n_range)));
