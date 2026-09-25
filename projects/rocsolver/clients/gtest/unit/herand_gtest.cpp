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

#include "common/unit/testing_herand.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::ValuesIn;
using namespace std;

// each size_range vector is a {N, lda}
// uplo_range: 'L' = lower, 'U' = upper, 'F' = full

using herand_tuple = std::tuple<vector<int>, char>;

// case when N == 0 also executes the bad arguments test

const vector<char> uplo_range = {'L', 'U', 'F'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5}, // lda < n
    // normal (valid) samples
    {1, 1},
    {10, 10},
    {20, 20},
    {35, 50}, // lda > n (tests padding)
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192}, {640, 700}, {1024, 1024}, {2547, 2550}};

Arguments herand_setup_arguments(herand_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<char>("uplo", uplo);

    arg.timing = 0;

    return arg;
}

class HERAND : public ::TestWithParam<herand_tuple>
{
protected:
    void TearDown() override {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = herand_setup_arguments(this->GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
        {
            testing_herand_bad_arg<T>();
        }

        testing_herand<T>(arg);
    }
};

// non-batch tests (herand is a pure host function, no batching)

TEST_P(HERAND, __float)
{
    run_tests<float>();
}

TEST_P(HERAND, __double)
{
    run_tests<double>();
}

TEST_P(HERAND, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(HERAND, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HERAND,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HERAND,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
