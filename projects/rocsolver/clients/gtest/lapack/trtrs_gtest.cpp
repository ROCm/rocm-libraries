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

#include "common/lapack/testing_trtrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using trtrs_tuple = tuple<vector<I>, vector<I>, vector<printable_char>>;

// each A_range vector is a {N, lda, ldb}

// each B_range vector is a {nrhs, singular}
// if singular = 1, then the tests are also run with a matrix A that has zero diagonal elements
// in one instance of the batch (info must be reported if diag is non-unit, even if nrhs = 0,
// and the right-hand sides of that instance must not be modified)

// each op_range vector is a {uplo, trans, diag}

// case when N = 0, nrhs = 1 and op = {U, N, N} will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> op_range
    = {{'U', 'N', 'N'}, {'U', 'N', 'U'}, {'U', 'T', 'N'}, {'U', 'T', 'U'},
       {'U', 'C', 'N'}, {'U', 'C', 'U'}, {'L', 'N', 'N'}, {'L', 'N', 'U'},
       {'L', 'T', 'N'}, {'L', 'T', 'U'}, {'L', 'C', 'N'}, {'L', 'C', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {10, 5, 10},
    {10, 10, 5},
    // normal (valid) samples
    {1, 1, 1},
    {7, 7, 7},
    {20, 25, 20},
    {33, 33, 40}};
const vector<vector<int>> matrix_sizeB_range = {
    // invalid
    {-1, 0},
    // normal (valid) samples
    {0, 1},
    {1, 0},
    {10, 1},
    {16, 0}};
const vector<vector<int64_t>> matrix_sizeA_range_64 = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {10, 5, 10},
    {10, 10, 5},
    // normal (valid) samples
    {1, 1, 1},
    {7, 7, 7},
    {20, 25, 20},
    {33, 33, 40}};
const vector<vector<int64_t>> matrix_sizeB_range_64 = {
    // invalid
    {-1, 0},
    // normal (valid) samples
    {0, 1},
    {1, 0},
    {10, 1},
    {16, 0}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range
    = {{64, 64, 64}, {192, 192, 200}, {500, 600, 500}, {1000, 1000, 1024}};
const vector<vector<int>> large_matrix_sizeB_range = {{1, 0}, {100, 1}, {250, 0}};
const vector<vector<int64_t>> large_matrix_sizeA_range_64
    = {{64, 64, 64}, {192, 192, 200}, {500, 600, 500}, {1000, 1000, 1024}};
const vector<vector<int64_t>> large_matrix_sizeB_range_64 = {{1, 0}, {100, 1}, {250, 0}};

template <typename I>
Arguments trtrs_setup_arguments(trtrs_tuple<I> tup)
{
    vector<I> matrix_sizeA = std::get<0>(tup);
    vector<I> matrix_sizeB = std::get<1>(tup);
    vector<printable_char> op = std::get<2>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_sizeA[0]);
    arg.set<I>("nrhs", matrix_sizeB[0]);
    arg.set<I>("lda", matrix_sizeA[1]);
    arg.set<I>("ldb", matrix_sizeA[2]);

    arg.set<char>("uplo", op[0]);
    arg.set<char>("trans", op[1]);
    arg.set<char>("diag", op[2]);

    arg.singular = (matrix_sizeB[1] == 1) ? 1 : 0;

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <typename I>
class TRTRS_BASE : public ::TestWithParam<trtrs_tuple<I>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = trtrs_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<I>("nrhs") == 1 && arg.peek<char>("uplo") == 'U'
           && arg.peek<char>("trans") == 'N' && arg.peek<char>("diag") == 'N')
            testing_trtrs_bad_arg<BATCHED, STRIDED, T, I>();

        // the batched variants are tested with one and three instances
        // (the singular instance is the only one, or the second one)
        const bool singular = (arg.singular == 1);
        for(I bc : (BATCHED || STRIDED ? vector<I>{1, 3} : vector<I>{1}))
        {
            arg.batch_count = bc;
            if(singular)
            {
                arg.singular = 1;
                testing_trtrs<BATCHED, STRIDED, T, I>(arg);
            }

            arg.singular = 0;
            testing_trtrs<BATCHED, STRIDED, T, I>(arg);
        }
    }
};

class TRTRS : public TRTRS_BASE<rocblas_int>
{
};

class TRTRS_64 : public TRTRS_BASE<int64_t>
{
};

// non-batch tests

TEST_P(TRTRS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(TRTRS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(TRTRS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TRTRS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(TRTRS_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(TRTRS_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(TRTRS_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TRTRS_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(TRTRS, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(TRTRS, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(TRTRS, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(TRTRS, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(TRTRS_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(TRTRS_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(TRTRS_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(TRTRS_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(TRTRS, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(TRTRS, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(TRTRS, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(TRTRS, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(TRTRS_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(TRTRS_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(TRTRS_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(TRTRS_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TRTRS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range),
                                 ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TRTRS_64,
                         Combine(ValuesIn(large_matrix_sizeA_range_64),
                                 ValuesIn(large_matrix_sizeB_range_64),
                                 ValuesIn(op_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRTRS,
                         Combine(ValuesIn(matrix_sizeA_range),
                                 ValuesIn(matrix_sizeB_range),
                                 ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRTRS_64,
                         Combine(ValuesIn(matrix_sizeA_range_64),
                                 ValuesIn(matrix_sizeB_range_64),
                                 ValuesIn(op_range)));
