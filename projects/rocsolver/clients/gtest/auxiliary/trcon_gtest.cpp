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

#include "common/auxiliary/testing_trcon.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using trcon_tuple = std::tuple<vector<I>, printable_char, printable_char, printable_char>;

// each size_range vector is a {N, lda, singular}, or {N, lda, singular, 1} to run with the handle
// in device pointer mode
// if singular = 1, then the tests are also run with a matrix that has a zero diagonal element
// (the estimate must be exactly zero if diag is non-unit)

// each norm_type is one of: '1', 'I'

// each uplo is one of: 'U', 'L'

// each diag is one of: 'N', 'U'

// case when N == 0, norm_type == '1', uplo == 'U' and diag == 'N' also executes the bad
// arguments test (null handle, null pointers and invalid values)

const vector<printable_char> norm_range = {'1', 'I'};

const vector<printable_char> uplo_range = {'U', 'L'};

const vector<printable_char> diag_range = {'N', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid sizes
    {-1, 1, 0},
    {10, 5, 0},
    // normal (valid) samples
    {1, 1, 1},
    {7, 7, 1},
    {20, 25, 0},
    {33, 50, 1},
    {100, 100, 0},
    // device pointer mode (with n large enough for rocBLAS to be called)
    {500, 500, 0, 1}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 0},
    // invalid sizes
    {-1, 1, 0},
    {10, 5, 0},
    // normal (valid) samples
    {1, 1, 1},
    {7, 7, 1},
    {20, 25, 0},
    {33, 50, 1},
    {100, 100, 0},
    // device pointer mode (with n large enough for rocBLAS to be called)
    {500, 500, 0, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 1}, {640, 800, 0}, {1024, 1200, 1}, {2048, 2050, 0}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{192, 192, 1}, {640, 800, 0}, {1024, 1200, 1}, {2048, 2050, 0}};

template <typename I>
Arguments trcon_setup_arguments(trcon_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char norm_type = std::get<1>(tup);
    char uplo = std::get<2>(tup);
    char diag = std::get<3>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("lda", matrix_size[1]);
    arg.set<char>("norm_type", norm_type);
    arg.set<char>("uplo", uplo);
    arg.set<char>("diag", diag);

    arg.singular = (matrix_size[2] == 1) ? 1 : 0;

    // device pointer mode
    if(matrix_size.size() > 3 && matrix_size[3] == 1)
        arg.set<char>("pointer_mode", 'D');

    arg.timing = 0;

    return arg;
}

template <typename I>
class TRCON_BASE : public ::TestWithParam<trcon_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = trcon_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<char>("norm_type") == '1'
           && arg.peek<char>("uplo") == 'U' && arg.peek<char>("diag") == 'N')
            testing_trcon_bad_arg<T, I>();

        if(arg.singular == 1)
            testing_trcon<T, I>(arg);

        arg.singular = 0;
        testing_trcon<T, I>(arg);
    }
};

class TRCON : public TRCON_BASE<rocblas_int>
{
};

class TRCON_64 : public TRCON_BASE<int64_t>
{
};

TEST_P(TRCON, __float)
{
    run_tests<float>();
}

TEST_P(TRCON, __double)
{
    run_tests<double>();
}

TEST_P(TRCON, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(TRCON, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(TRCON_64, __float)
{
    run_tests<float>();
}

TEST_P(TRCON_64, __double)
{
    run_tests<double>();
}

TEST_P(TRCON_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(TRCON_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TRCON,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(norm_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(diag_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRCON,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(norm_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(diag_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TRCON_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(norm_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(diag_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRCON_64,
                         Combine(ValuesIn(matrix_size_range_64),
                                 ValuesIn(norm_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(diag_range)));
