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

#include "common/auxiliary/testing_laqsy_laqhe.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using laqsy_laqhe_tuple = std::tuple<vector<I>, string>;

// each size_range vector is a {N, lda, uplo}
// if uplo = 0 then lower
// if uplo = 1 then upper
// if uplo = 2 then full (invalid; checked before the sizes)

// each case is a string {equed, equ_mode}, where equed is the expected equilibration ('N' or 'Y';
// '-' when it is not known beforehand) and equ_mode selects how it is obtained:
// 'F': forced through scond (0.5 or 0.05),
// 'T': forced through a tiny amax,
// 'H': forced through a huge amax,
// 'E': scond and amax exactly at the thresholds (no scaling),
// 'G': scalings computed by POEQU from A,
// 'P': scalings computed by POEQUB from A

// case when N == 0 and case == "NF" also executes the bad arguments test
// (null handle, null pointers and invalid values)

const vector<string> case_range = {"NF", "YF", "YT", "YH", "NE", "-G", "-P"};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    {20, 20, 2},
    {-1, 1, 2},
    // normal (valid) samples
    {1, 1, 0},
    {1, 1, 1},
    {7, 7, 0},
    {7, 10, 1},
    {33, 33, 0},
    {33, 40, 1},
    {100, 100, 0},
    {100, 100, 1}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    {20, 20, 2},
    {-1, 1, 2},
    // normal (valid) samples
    {1, 1, 0},
    {1, 1, 1},
    {7, 7, 0},
    {7, 10, 1},
    {33, 33, 0},
    {33, 40, 1},
    {100, 100, 0},
    {100, 100, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{500, 500, 0}, {1000, 1024, 1}, {2000, 2000, 0}, {2000, 2000, 1}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{500, 500, 0}, {1000, 1024, 1}, {2000, 2000, 0}, {2000, 2000, 1}};

template <typename I>
Arguments laqsy_laqhe_setup_arguments(laqsy_laqhe_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    string test_case = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("lda", matrix_size[1]);

    if(matrix_size[2] == 0)
        arg.set<char>("uplo", 'L');
    else if(matrix_size[2] == 1)
        arg.set<char>("uplo", 'U');
    else
        arg.set<char>("uplo", 'F');

    if(test_case[0] != '-')
        arg.set<char>("equed", test_case[0]);
    arg.set<char>("equ_mode", test_case[1]);

    arg.timing = 0;

    return arg;
}

template <bool HERM, typename I>
class LAQSY_LAQHE : public ::TestWithParam<laqsy_laqhe_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = laqsy_laqhe_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && std::get<1>(this->GetParam()) == "NF")
            testing_laqsy_laqhe_bad_arg<HERM, T, I>();

        testing_laqsy_laqhe<HERM, T, I>(arg);
    }
};

class LAQSY : public LAQSY_LAQHE<false, rocblas_int>
{
};

class LAQHE : public LAQSY_LAQHE<true, rocblas_int>
{
};

class LAQSY_64 : public LAQSY_LAQHE<false, int64_t>
{
};

class LAQHE_64 : public LAQSY_LAQHE<true, int64_t>
{
};

// non-batch tests

TEST_P(LAQSY, __float)
{
    run_tests<float>();
}

TEST_P(LAQSY, __double)
{
    run_tests<double>();
}

TEST_P(LAQSY, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAQSY, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LAQHE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAQHE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LAQSY_64, __float)
{
    run_tests<float>();
}

TEST_P(LAQSY_64, __double)
{
    run_tests<double>();
}

TEST_P(LAQSY_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAQSY_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LAQHE_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAQHE_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LAQSY,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LAQSY,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LAQHE,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LAQHE,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LAQSY_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LAQSY_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LAQHE_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LAQHE_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(case_range)));
