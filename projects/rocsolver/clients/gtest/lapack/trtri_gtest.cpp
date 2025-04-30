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

#include "common/lapack/testing_trtri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, printable_char> trtri_tuple;

// each matrix_size_range vector is a {n, lda, singular/diag}
// if singular = 0, then the used matrix for the tests is triangular unit
// if singular = 1, then the used matrix for the tests is triangular non-unit and singular
// otherwise, the used matrix is triangular non-unit and not singular

// each uplo_range is {uplo}

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
    {20, 32, 0},
    {30, 30, 1},
    {40, 60, 2},
    {80, 80, 2},
    {90, 100, 1},
    {100, 150, 0}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 1}, {500, 600, 2}, {640, 640, 0}, {1000, 1024, 1}, {1200, 1230, 2}};

Arguments trtri_setup_arguments(trtri_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    if(matrix_size[2] == 0)
        arg.set<char>("diag", 'U');
    else
        arg.set<char>("diag", 'N');

    if(matrix_size[2] == 1)
        arg.singular = 1;
    else
        arg.singular = 0;

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class TRTRI : public ::TestWithParam<trtri_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = trtri_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("uplo") == 'L')
            testing_trtri_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_trtri<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_trtri<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(TRTRI, __float)
{
    run_tests<false, false, float>();
}

TEST_P(TRTRI, __double)
{
    run_tests<false, false, double>();
}

TEST_P(TRTRI, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TRTRI, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(TRTRI, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(TRTRI, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(TRTRI, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(TRTRI, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(TRTRI, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(TRTRI, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(TRTRI, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(TRTRI, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TRTRI,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRTRI,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
