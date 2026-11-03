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

#include "common/lapack/testing_sysv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int, printable_char> sysv_tuple;

// each matrix_size_range vector is a {n, lda, ldb, singular} or {n, lda, ldb, singular, 1}
// if singular = 1, then the matrix in the middle of the batch is singular
// a fifth element 1 uses random symmetric indefinite matrices (irregular 1x1 and 2x2 pivots)

// each nrhs_range is a {nrhs}
// nrhs = 0 still factorizes A

// each uplo_range is a {uplo}

// case when n = 0, nrhs = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {20, 5, 20, 0},
    {20, 20, 5, 0},
    // normal (valid) samples
    {1, 1, 1, 1},
    {15, 15, 15, 0},
    {32, 32, 40, 1},
    {50, 60, 50, 0},
    {70, 100, 70, 1},
    {130, 130, 135, 1},
    {2, 2, 2, 0, 1},
    {9, 9, 9, 1, 1},
    {40, 40, 41, 0, 1},
    {150, 150, 150, 1, 1}};

const vector<int> nrhs_range = {
    // invalid
    -1,
    // normal (valid) samples
    0, 1, 10, 31};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192, 1},
    {257, 300, 260, 0},
    {640, 640, 650, 1},
    {1000, 1024, 1000, 1},
};

const vector<int> large_nrhs_range = {0, 1, 100};

Arguments sysv_setup_arguments(sysv_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int nrhs = std::get<1>(tup);
    char uplo = std::get<2>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);
    arg.set<rocblas_int>("nrhs", nrhs);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[3];
    if(matrix_size.size() > 4 && matrix_size[4] == 1)
        arg.set<char>("pivots", 'R');

    return arg;
}

class SYSV : public ::TestWithParam<sysv_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sysv_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<rocblas_int>("n") == 0
           && arg.peek<rocblas_int>("nrhs") == 0)
            testing_sysv_bad_arg<BATCHED, STRIDED, T>();

        // batched and strided_batched are tested with batch_count = 3 and 1
        const bool singular = arg.singular;
        for(rocblas_int bc : {3, 1})
        {
            if(!(BATCHED || STRIDED) && bc != 1)
                continue;

            Arguments argb = arg;
            argb.batch_count = bc;
            if(singular)
                testing_sysv<BATCHED, STRIDED, T>(argb);

            argb = arg;
            argb.batch_count = bc;
            argb.singular = 0;
            testing_sysv<BATCHED, STRIDED, T>(argb);
        }
    }
};

// non-batch tests

TEST_P(SYSV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYSV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYSV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYSV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYSV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYSV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(SYSV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(SYSV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYSV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYSV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(SYSV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(SYSV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYSV,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(large_nrhs_range),
                                 ValuesIn(uplo_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYSV,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(nrhs_range),
                                 ValuesIn(uplo_range)));
