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

#include "common/auxiliary/testing_geequ.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using geequ_tuple = std::tuple<vector<I>, char>;

// each size_range vector is a {M, N, lda}

// each case is one of:
// 'N': entries of normal magnitude (rows and columns scaled by up to 2^20),
// 'L': same, with the largest entries close to overflow,
// 'S': same, with the smallest entries close to underflow,
// 'P': same as 'N', with all magnitudes exact powers of 2 (where LAPACK's rounding of the
//      logarithms in xGEEQUB/xPOEQUB matters),
// 'D': same, with the largest magnitudes subnormal,
// 'R': a zero row (in the middle),
// 'C': a zero column (the last one),
// 'B': a zero row (the last one) and a zero column (the first one)

// case when M == 0 and case == 'N' also executes the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> case_range = {'N', 'L', 'S', 'P', 'D', 'R', 'C', 'B'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5},
    // normal (valid) samples
    {1, 1, 1},
    {1, 7, 1},
    {7, 1, 7},
    {7, 33, 10},
    {33, 7, 40},
    {33, 33, 33},
    {33, 100, 33},
    {100, 33, 120},
    {100, 100, 100}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5},
    // normal (valid) samples
    {1, 1, 1},
    {1, 7, 1},
    {7, 1, 7},
    {7, 33, 10},
    {33, 7, 40},
    {33, 33, 33},
    {33, 100, 33},
    {100, 33, 120},
    {100, 100, 100}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{300, 2000, 300}, {2000, 300, 2048}, {1000, 1000, 1000}, {2000, 2000, 2000}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{300, 2000, 300}, {2000, 300, 2048}, {1000, 1000, 1000}, {2000, 2000, 2000}};

template <typename I>
Arguments geequ_setup_arguments(geequ_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char test_case = std::get<1>(tup);

    I m = matrix_size[0];
    I n = matrix_size[1];

    Arguments arg;

    arg.set<I>("m", m);
    arg.set<I>("n", n);
    arg.set<I>("lda", matrix_size[2]);

    char scale = 'N';
    I zero_row = 0;
    I zero_col = 0;
    if(test_case == 'L' || test_case == 'S' || test_case == 'P' || test_case == 'D')
        scale = test_case;
    else if(test_case == 'R')
        zero_row = (m + 1) / 2;
    else if(test_case == 'C')
        zero_col = n;
    else if(test_case == 'B')
    {
        zero_row = m;
        zero_col = 1;
    }

    arg.set<char>("scale", scale);
    arg.set<I>("zero_row", std::max(zero_row, I(0)));
    arg.set<I>("zero_col", std::max(zero_col, I(0)));

    arg.timing = 0;

    return arg;
}

template <bool POW2, typename I>
class GEEQU_BASE : public ::TestWithParam<geequ_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = geequ_setup_arguments(this->GetParam());

        if(arg.peek<I>("m") == 0 && std::get<1>(this->GetParam()) == 'N')
            testing_geequ_bad_arg<POW2, T, I>();

        testing_geequ<POW2, T, I>(arg);
    }
};

class GEEQU : public GEEQU_BASE<false, rocblas_int>
{
};

class GEEQUB : public GEEQU_BASE<true, rocblas_int>
{
};

class GEEQU_64 : public GEEQU_BASE<false, int64_t>
{
};

class GEEQUB_64 : public GEEQU_BASE<true, int64_t>
{
};

// non-batch tests

TEST_P(GEEQU, __float)
{
    run_tests<float>();
}

TEST_P(GEEQU, __double)
{
    run_tests<double>();
}

TEST_P(GEEQU, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEEQU, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(GEEQUB, __float)
{
    run_tests<float>();
}

TEST_P(GEEQUB, __double)
{
    run_tests<double>();
}

TEST_P(GEEQUB, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEEQUB, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(GEEQU_64, __float)
{
    run_tests<float>();
}

TEST_P(GEEQU_64, __double)
{
    run_tests<double>();
}

TEST_P(GEEQU_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEEQU_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(GEEQUB_64, __float)
{
    run_tests<float>();
}

TEST_P(GEEQUB_64, __double)
{
    run_tests<double>();
}

TEST_P(GEEQUB_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEEQUB_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEQU,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEQU,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEQUB,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEQUB,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEQU_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEQU_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEQUB_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEQUB_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));
