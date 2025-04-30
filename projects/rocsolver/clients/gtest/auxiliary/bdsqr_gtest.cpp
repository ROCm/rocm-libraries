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

#include "common/auxiliary/testing_bdsqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> bdsqr_tuple;

// each size_range is a {n, nv, nu, nc}

// each opt_range is a {uplo, ldv, ldu, ldc}
// if uplo = 0, then is upper bidiagonal
// if uplo = 1, then is lower bidiagonal
// if ldx = -1, then ldx < limit (invalid size)
// if ldx = 0, then ldx = limit
// if ldx = 1, then ldx > limit

// case when n = 0 and uplo = 'L' will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 1},
    // invalid
    {-1, 1, 1, 1},
    {1, -1, 1, 1},
    {1, 1, -1, 1},
    {1, 1, 1, -1},
    // normal (valid) samples
    {1, 1, 1, 1},
    {15, 10, 10, 10},
    {20, 0, 0, 15},
    {20, 0, 15, 0},
    {30, 50, 0, 0},
    {50, 60, 20, 0},
    {70, 0, 0, 0},
};

const vector<vector<int>> opt_range = {
    // invalid
    {0, -1, 0, 0},
    {0, 0, -1, 0},
    {0, 0, 0, -1},
    // normal (valid) samples
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{152, 152, 152, 152}, {640, 640, 656, 700}, {1000, 1024, 1000, 80}, {2000, 0, 0, 0}};

const vector<vector<int>> large_opt_range = {{0, 0, 0, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 0}};

Arguments bdsqr_setup_arguments(bdsqr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt = std::get<1>(tup);

    Arguments arg;

    rocblas_int n = size[0];
    rocblas_int nv = size[1];
    rocblas_int nu = size[2];
    rocblas_int nc = size[3];
    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("nv", nv);
    arg.set<rocblas_int>("nu", nu);
    arg.set<rocblas_int>("nc", nc);

    arg.set<char>("uplo", opt[0] ? 'L' : 'U');

    arg.set<rocblas_int>("ldv", (nv > 0 ? n : 1) + opt[1] * 10);
    arg.set<rocblas_int>("ldu", (nu > 0 ? nu : 1) + opt[2] * 10);
    arg.set<rocblas_int>("ldc", (nc > 0 ? n : 1) + opt[3] * 10);

    arg.timing = 0;

    return arg;
}

template <rocblas_int MODE>
class BDSQR_BASE : public ::TestWithParam<bdsqr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = bdsqr_setup_arguments(GetParam());
        arg.alg_mode = MODE;

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("uplo") == 'L')
            testing_bdsqr_bad_arg<T>();

        testing_bdsqr<T>(arg);
    }
};

class BDSQR : public BDSQR_BASE<0>
{
};

class BDSQR_HYBRID : public BDSQR_BASE<1>
{
};

// non-batch tests

TEST_P(BDSQR, __float)
{
    run_tests<float>();
}

TEST_P(BDSQR, __double)
{
    run_tests<double>();
}

TEST_P(BDSQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(BDSQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(BDSQR_HYBRID, __float)
{
    run_tests<float>();
}

TEST_P(BDSQR_HYBRID, __double)
{
    run_tests<double>();
}

TEST_P(BDSQR_HYBRID, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(BDSQR_HYBRID, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         BDSQR,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, BDSQR, Combine(ValuesIn(size_range), ValuesIn(opt_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         BDSQR_HYBRID,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         BDSQR_HYBRID,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));
