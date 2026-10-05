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

#include "common/lapack/testing_trexc.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> trexc_tuple;

// each matrix_size_range vector is a {n, ldt, ldq, ifst, ilst, mtype}
// mtype = 0: random upper triangular T,
// mtype = 1: T scaled by a tiny factor,
// mtype = 2: T scaled by a huge factor,
// mtype = 3: T with repeated diagonal entries
// (see trexc_initData)

// each compq_range value is the compq option (N or V)

// case when n = 0 and compq = V will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> compq_range = {'N', 'V'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 0},
    {10, 10, 10, 4, 4, 0},
    // invalid
    {-1, 1, 1, 1, 1, 0},
    {10, 5, 10, 1, 2, 0},
    {10, 10, 5, 1, 2, 0},
    {10, 10, 10, 0, 2, 0},
    {10, 10, 10, 1, 11, 0},
    // normal (valid) samples
    {2, 2, 2, 1, 2, 0},
    {2, 2, 2, 2, 1, 0},
    {10, 10, 10, 1, 10, 0},
    {10, 12, 10, 10, 1, 0},
    {35, 35, 40, 5, 30, 0},
    {35, 35, 35, 30, 5, 0},
    {64, 64, 64, 1, 64, 1},
    {64, 64, 64, 64, 1, 2},
    {40, 40, 40, 3, 37, 3},
    {300, 300, 300, 1, 300, 0},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {1000, 1000, 1000, 1, 1000, 0},
    {1000, 1024, 1000, 900, 50, 0},
};

Arguments trexc_setup_arguments(trexc_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char compq = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("ldt", matrix_size[1]);
    arg.set<rocblas_int>("ldq", matrix_size[2]);
    arg.set<rocblas_int>("ifst", matrix_size[3]);
    arg.set<rocblas_int>("ilst", matrix_size[4]);
    arg.set<rocblas_int>("mtype", matrix_size[5]);
    arg.set<char>("compq", compq);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class TREXC : public ::TestWithParam<trexc_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = trexc_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("compq") == 'V')
            testing_trexc_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_trexc<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(TREXC, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TREXC, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(TREXC, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(TREXC, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(TREXC, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(TREXC, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TREXC,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(compq_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TREXC,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(compq_range)));
