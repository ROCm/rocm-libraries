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

#include "common/lapack/testing_sytf2_sytrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, printable_char> sytrf_tuple;

// each matrix_size_range vector is a {n, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// each uplo_range is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {32, 32, 1},
    {50, 50, 0},
    {70, 100, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 1},
    {640, 640, 0},
    {1000, 1024, 1},
};

Arguments sytrf_setup_arguments(sytrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

template <bool BLOCKED>
class SYTF2_SYTRF : public ::TestWithParam<sytrf_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sytrf_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<rocblas_int>("n") == 0)
            testing_sytf2_sytrf_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_sytf2_sytrf<BATCHED, STRIDED, BLOCKED, T>(arg);

        arg.singular = 0;
        testing_sytf2_sytrf<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class SYTF2 : public SYTF2_SYTRF<false>
{
};

class SYTRF : public SYTF2_SYTRF<true>
{
};

// non-batch tests

TEST_P(SYTF2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTF2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYTF2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYTF2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYTRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYTRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYTRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYTF2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYTF2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(SYTF2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(SYTF2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(SYTRF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYTRF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(SYTRF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(SYTRF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYTF2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYTF2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(SYTF2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(SYTF2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(SYTRF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYTRF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(SYTRF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(SYTRF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
