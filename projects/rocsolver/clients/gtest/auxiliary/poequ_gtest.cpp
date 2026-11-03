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

#include "common/auxiliary/testing_poequ.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using poequ_tuple = std::tuple<vector<I>, char>;

// each size_range vector is a {N, lda}

// each case is one of:
// 'N': positive diagonal of normal magnitude (spanning 2^-40 to 2^40),
// 'L': same, with the largest elements close to overflow,
// 'S': same, with the smallest elements close to underflow,
// 'P': same as 'N', with all magnitudes exact powers of 2 (where LAPACK's rounding of the
//      logarithms in xGEEQUB/xPOEQUB matters),
// 'D': same, with the largest magnitudes subnormal,
// 'Z': a zero diagonal element (in the middle),
// 'M': a negative diagonal element (the last one)

// case when N == 0 and case == 'N' also executes the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> case_range = {'N', 'L', 'S', 'P', 'D', 'Z', 'M'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {7, 7},
    {7, 10},
    {33, 33},
    {33, 40},
    {100, 100}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {7, 7},
    {7, 10},
    {33, 33},
    {33, 40},
    {100, 100}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{500, 500}, {1000, 1024}, {2000, 2000}};

const vector<vector<int64_t>> large_matrix_size_range_64 = {{500, 500}, {1000, 1024}, {2000, 2000}};

template <typename I>
Arguments poequ_setup_arguments(poequ_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char test_case = std::get<1>(tup);

    I n = matrix_size[0];

    Arguments arg;

    arg.set<I>("n", n);
    arg.set<I>("lda", matrix_size[1]);

    char scale = 'N';
    I zero_diag = 0;
    I neg_diag = 0;
    if(test_case == 'L' || test_case == 'S' || test_case == 'P' || test_case == 'D')
        scale = test_case;
    else if(test_case == 'Z')
        zero_diag = (n + 1) / 2;
    else if(test_case == 'M')
        neg_diag = n;

    arg.set<char>("scale", scale);
    arg.set<I>("zero_diag", std::max(zero_diag, I(0)));
    arg.set<I>("neg_diag", std::max(neg_diag, I(0)));

    arg.timing = 0;

    return arg;
}

template <bool POW2, typename I>
class POEQU_BASE : public ::TestWithParam<poequ_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = poequ_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && std::get<1>(this->GetParam()) == 'N')
            testing_poequ_bad_arg<POW2, T, I>();

        testing_poequ<POW2, T, I>(arg);
    }
};

class POEQU : public POEQU_BASE<false, rocblas_int>
{
};

class POEQUB : public POEQU_BASE<true, rocblas_int>
{
};

class POEQU_64 : public POEQU_BASE<false, int64_t>
{
};

class POEQUB_64 : public POEQU_BASE<true, int64_t>
{
};

// non-batch tests

TEST_P(POEQU, __float)
{
    run_tests<float>();
}

TEST_P(POEQU, __double)
{
    run_tests<double>();
}

TEST_P(POEQU, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(POEQU, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(POEQUB, __float)
{
    run_tests<float>();
}

TEST_P(POEQUB, __double)
{
    run_tests<double>();
}

TEST_P(POEQUB, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(POEQUB, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(POEQU_64, __float)
{
    run_tests<float>();
}

TEST_P(POEQU_64, __double)
{
    run_tests<double>();
}

TEST_P(POEQU_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(POEQU_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(POEQUB_64, __float)
{
    run_tests<float>();
}

TEST_P(POEQUB_64, __double)
{
    run_tests<double>();
}

TEST_P(POEQUB_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(POEQUB_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POEQU,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POEQU,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POEQUB,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POEQUB,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POEQU_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POEQU_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POEQUB_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POEQUB_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));
