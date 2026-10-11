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

#include "common/auxiliary/testing_lansy_lanhe.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using lansy_lanhe_tuple = std::tuple<vector<I>, printable_char, string>;

// each size_range vector is a {N, lda}

// each uplo is one of: 'U', 'L'

// each norm_case is a {norm_type, magnitude} pair, where norm_type is one of '1', 'I', 'F', 'M'
// and magnitude is one of:
// 'N' = random matrix with entries of moderate size,
// 'B' = entries scaled close to the overflow threshold,
// 'S' = entries scaled close to the underflow threshold
// 'Q' = one referenced entry is NaN (the norm must be NaN, as in LAPACK)
// (the scaled matrices are only used with the Frobenius and max norms)

// case when N == 0, uplo == 'U' and norm_case == "1N" also executes the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'U', 'L'};

const vector<string> norm_case_range
    = {"1N", "IN", "FN", "MN", "FB", "FS", "MB", "MS", "1Q", "IQ", "FQ", "MQ"};

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
    {33, 40},
    {100, 100},
    {1100, 1100}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {7, 7},
    {33, 40},
    {100, 100},
    {1100, 1100}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192}, {640, 700}, {1000, 1024}, {2547, 2550}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{192, 192}, {640, 700}, {1000, 1024}, {2547, 2550}};

template <typename I>
Arguments lansy_lanhe_setup_arguments(lansy_lanhe_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);
    string norm_case = std::get<2>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);
    arg.set<char>("norm_type", norm_case[0]);
    arg.set<char>("magnitude", norm_case[1]);

    arg.timing = 0;

    return arg;
}

template <bool HERM, typename I>
class LANSY_LANHE_BASE : public ::TestWithParam<lansy_lanhe_tuple<I>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lansy_lanhe_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<char>("uplo") == 'U'
           && arg.peek<char>("norm_type") == '1' && arg.peek<char>("magnitude") == 'N')
            testing_lansy_lanhe_bad_arg<HERM, T, I>();

        testing_lansy_lanhe<HERM, T, I>(arg);
    }
};

class LANSY : public LANSY_LANHE_BASE<false, rocblas_int>
{
};

class LANSY_64 : public LANSY_LANHE_BASE<false, int64_t>
{
};

class LANHE : public LANSY_LANHE_BASE<true, rocblas_int>
{
};

class LANHE_64 : public LANSY_LANHE_BASE<true, int64_t>
{
};

// non-batch tests

TEST_P(LANSY, __float)
{
    run_tests<float>();
}

TEST_P(LANSY, __double)
{
    run_tests<double>();
}

TEST_P(LANSY, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANSY, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LANSY_64, __float)
{
    run_tests<float>();
}

TEST_P(LANSY_64, __double)
{
    run_tests<double>();
}

TEST_P(LANSY_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANSY_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LANHE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANHE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LANHE_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANHE_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANSY,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANSY,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANSY_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANSY_64,
                         Combine(ValuesIn(matrix_size_range_64),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANHE,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANHE,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANHE_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANHE_64,
                         Combine(ValuesIn(matrix_size_range_64),
                                 ValuesIn(uplo_range),
                                 ValuesIn(norm_case_range)));
