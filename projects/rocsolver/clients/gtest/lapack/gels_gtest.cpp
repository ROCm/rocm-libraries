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

#include "common/lapack/testing_gels.hpp"
#include "common/lapack/testing_gels_outofplace.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int, int, int, int> gels_params_A;
typedef std::tuple<int, printable_char> gels_params_B;

typedef std::tuple<gels_params_A, gels_params_B> gels_tuple;

// each A_range tuple is a {M, N, lda, ldb, singular};
// if singular = 1, then the used matrix for the tests is singular

// each B_range tuple is a {nrhs, trans};

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<gels_params_A> matrix_sizeA_range = {
    // quick return
    {0, 0, 0, 0, 0},
    // invalid
    {-1, 1, 1, 1, 0},
    {1, -1, 1, 1, 0},
    {10, 10, 10, 1, 0},
    {10, 10, 1, 10, 0},
    // normal (valid) samples
    {20, 20, 20, 20, 1},
    {30, 20, 40, 30, 0},
    {20, 30, 30, 40, 0},
    {40, 20, 40, 40, 1},
    {20, 40, 40, 40, 1},
};
const vector<gels_params_B> matrix_sizeB_range = {
    // quick return
    {0, 'N'},
    // invalid
    {-1, 'N'},
    // normal (valid) samples
    {10, 'N'},
    {20, 'N'},
    {30, 'N'},
    // invalid for complex precision
    {10, 'T'},
    {30, 'T'},
    // invalid for real precision
    {20, 'C'},
};

// for daily_lapack tests
const vector<gels_params_A> large_matrix_sizeA_range = {
    {75, 25, 75, 75, 1},    {25, 75, 75, 75, 1},    {150, 150, 150, 150, 1},
    {500, 50, 600, 600, 0}, {50, 500, 600, 600, 0},
};
const vector<gels_params_B> large_matrix_sizeB_range = {
    {100, 'N'},
    {200, 'T'},
    {500, 'C'},
    {1000, 'N'},
};

Arguments gels_setup_arguments(gels_tuple tup, bool outofplace)
{
    gels_params_A matrix_sizeA = std::get<0>(tup);
    gels_params_B matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", std::get<0>(matrix_sizeA));
    arg.set<rocblas_int>("n", std::get<1>(matrix_sizeA));
    arg.set<rocblas_int>("lda", std::get<2>(matrix_sizeA));
    arg.set<rocblas_int>("ldb", std::get<3>(matrix_sizeA));

    if(outofplace)
        arg.set<rocblas_int>("ldx", std::get<3>(matrix_sizeA));

    arg.set<rocblas_int>("nrhs", std::get<0>(matrix_sizeB));
    arg.set<char>("trans", std::get<1>(matrix_sizeB));

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = std::get<4>(matrix_sizeA);

    return arg;
}

class GELS : public ::TestWithParam<gels_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gels_setup_arguments(GetParam(), false);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nrhs") == 0)
            testing_gels_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_gels<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_gels<BATCHED, STRIDED, T>(arg);
    }
};

class GELS_OUTOFPLACE : public ::TestWithParam<gels_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gels_setup_arguments(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nrhs") == 0)
            testing_gels_outofplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_gels_outofplace<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_gels_outofplace<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GELS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GELS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GELS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GELS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GELS_OUTOFPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GELS_OUTOFPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GELS_OUTOFPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GELS_OUTOFPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GELS, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GELS, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GELS, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GELS, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GELS, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GELS, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GELS, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GELS, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELS_OUTOFPLACE,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS_OUTOFPLACE,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
