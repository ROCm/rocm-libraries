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

#include "common/lapack/testing_sygvx_hegvx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> sygvx_tuple;

// each matrix_size_range is a {n, lda, ldb, ldz, vl, vu, il, iu, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each type_range is a {itype, evect, erange, uplo}

// case when n = 0, itype = 1, evect = 'N', erange = 'A', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> type_range
    = {{'1', 'N', 'A', 'U'}, {'2', 'N', 'V', 'L'}, {'3', 'N', 'I', 'U'},
       {'1', 'V', 'V', 'L'}, {'2', 'V', 'I', 'U'}, {'3', 'V', 'A', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1, 0, 10, 1, 0, 0},
    // invalid
    {-1, 1, 1, 1, 0, 10, 1, 1, 0},
    {20, 5, 5, 20, 0, 10, 1, 1, 0},
    {20, 20, 20, 5, 0, 10, 1, 1, 0},
    // valid only when erange=A
    {20, 20, 20, 20, 10, 0, 10, 1, 0},
    // normal (valid) samples
    {20, 30, 20, 20, 5, 15, 1, 10, 1},
    {35, 35, 35, 35, -10, 10, 1, 35, 0},
    {50, 50, 60, 70, -15, -5, 25, 50, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192, 192, 5, 15, 100, 150, 0},
    {256, 270, 256, 260, -10, 10, 1, 100, 0},
    {300, 300, 310, 320, -15, -5, 200, 300, 0},
};

template <typename T>
Arguments sygvx_setup_arguments(sygvx_tuple tup)
{
    using S = decltype(std::real(T{}));

    vector<int> matrix_size = std::get<0>(tup);
    vector<printable_char> type = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);
    arg.set<rocblas_int>("ldz", matrix_size[3]);
    arg.set<double>("vl", matrix_size[4]);
    arg.set<double>("vu", matrix_size[5]);
    arg.set<rocblas_int>("il", matrix_size[6]);
    arg.set<rocblas_int>("iu", matrix_size[7]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("evect", type[1]);
    arg.set<char>("erange", type[2]);
    arg.set<char>("uplo", type[3]);

    arg.set<double>("abstol", 0);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[8];

    return arg;
}

class SYGVX_HEGVX : public ::TestWithParam<sygvx_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygvx_setup_arguments<T>(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("erange") == 'A' && arg.peek<char>("uplo") == 'U'
           && arg.peek<rocblas_int>("n") == 0)
            testing_sygvx_hegvx_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_sygvx_hegvx<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_sygvx_hegvx<BATCHED, STRIDED, T>(arg);
    }
};

class SYGVX : public SYGVX_HEGVX
{
};

class HEGVX : public SYGVX_HEGVX
{
};

// non-batch tests

TEST_P(SYGVX, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVX, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVX, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVX, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYGVX, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYGVX, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEGVX, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEGVX, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYGVX, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYGVX, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEGVX, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEGVX, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGVX,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVX,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGVX,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVX,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
