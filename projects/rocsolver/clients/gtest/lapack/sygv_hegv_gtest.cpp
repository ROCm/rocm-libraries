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

#include "common/lapack/testing_sygv_hegv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> sygv_tuple;

// each matrix_size_range is a {n, lda, ldb, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each type_range is a {itype, evect, uplo}

// case when n = 0, itype = 1, evect = 'N', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> type_range
    = {{'1', 'N', 'U'}, {'2', 'N', 'L'}, {'3', 'N', 'U'},
       {'1', 'V', 'L'}, {'2', 'V', 'U'}, {'3', 'V', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {20, 5, 5, 0},
    // normal (valid) samples
    {20, 30, 20, 1},
    {35, 35, 35, 0},
    {50, 50, 60, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192, 0},
    {256, 270, 256, 0},
    {300, 300, 310, 0},
};

Arguments sygv_setup_arguments(sygv_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<printable_char> type = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("evect", type[1]);
    arg.set<char>("uplo", type[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[3];

    return arg;
}

class SYGV_HEGV : public ::TestWithParam<sygv_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygv_setup_arguments(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("uplo") == 'U' && arg.peek<rocblas_int>("n") == 0)
            testing_sygv_hegv_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_sygv_hegv<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_sygv_hegv<BATCHED, STRIDED, T>(arg);
    }
};

class SYGV : public SYGV_HEGV
{
};

class HEGV : public SYGV_HEGV
{
};

// non-batch tests

TEST_P(SYGV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYGV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYGV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEGV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEGV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYGV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYGV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEGV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEGV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGV,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGV,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGV,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGV,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
