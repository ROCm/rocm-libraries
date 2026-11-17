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

#include "common/unit/testing_hbrand.hpp"

using ::testing::TestWithParam;
using ::testing::ValuesIn;
using namespace std;

// each size_range vector is a {n, kl, ku, ldab}

// case when n == 0 also executes the bad arguments test

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 2, 2, 5},
    // invalid
    {-1, 2, 2, 5},
    {10, 2, 2, 4}, // ldab < kl+ku+1
    // kl == ku (fully symmetric)
    {10, 0, 0, 1}, // diagonal only
    {10, 2, 2, 5},
    {20, 3, 3, 7},
    // kl > ku
    {10, 3, 1, 5},
    {20, 4, 2, 7},
    // kl < ku
    {10, 1, 3, 5},
    {20, 2, 4, 7},
    // ldab > kl+ku+1 (extra padding rows)
    {15, 2, 2, 8},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {512, 5, 5, 11},
    {1024, 10, 10, 21},
    {2000, 8, 3, 12},
    {2000, 3, 8, 12},
};

Arguments hbrand_setup_arguments(vector<int> sz)
{
    Arguments arg;

    arg.set<rocblas_int>("n", sz[0]);
    arg.set<rocblas_int>("kl", sz[1]);
    arg.set<rocblas_int>("ku", sz[2]);
    arg.set<rocblas_int>("ldab", sz[3]);

    arg.timing = 0;

    return arg;
}

class HBRAND : public ::TestWithParam<vector<int>>
{
protected:
    void TearDown() override {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = hbrand_setup_arguments(this->GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
        {
            testing_hbrand_bad_arg<T>();
        }

        testing_hbrand<T>(arg);
    }
};

// non-batch tests (hbrand is a pure host function, no batching)

TEST_P(HBRAND, __float)
{
    run_tests<float>();
}

TEST_P(HBRAND, __double)
{
    run_tests<double>();
}

TEST_P(HBRAND, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(HBRAND, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, HBRAND, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HBRAND, ValuesIn(matrix_size_range));
