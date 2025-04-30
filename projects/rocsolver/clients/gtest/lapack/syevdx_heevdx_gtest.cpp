/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/lapack/testing_syevdx_heevdx.hpp"
#include "common/lapack/testing_syevdx_heevdx_inplace.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> syevdx_heevdx_tuple;

// each size_range vector is a {n, lda, ldz, vl, vu, il, iu}

// each op_range vector is a {evect, erange, uplo}

// case when n == 0, evect == N, erange == V and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> op_range
    = {{'N', 'V', 'L'}, {'V', 'A', 'U'}, {'V', 'V', 'L'}, {'V', 'I', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 0, 10, 1, 0},
    // invalid
    {-1, 1, 1, 0, 10, 1, 1},
    {10, 5, 10, 0, 10, 1, 1},
    // valid only when evect=N
    {10, 10, 5, 0, 10, 1, 1},
    // valid only when erange=A
    {10, 10, 10, 10, 0, 10, 1},
    // normal (valid) samples
    {1, 1, 1, 0, 10, 1, 1},
    {12, 12, 15, -20, 20, 10, 12},
    {20, 30, 30, 5, 15, 1, 20},
    {35, 35, 35, -10, 10, 1, 15},
    {50, 60, 50, -15, -5, 20, 30}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192, 192, 5, 15, 100, 170},
                                              {256, 270, 256, -10, 10, 1, 256},
                                              {300, 300, 330, -15, -5, 200, 300}};

template <typename T>
Arguments syevdx_heevdx_setup_arguments(syevdx_heevdx_tuple tup, bool inplace)
{
    using S = decltype(std::real(T{}));

    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    if(!inplace)
        arg.set<rocblas_int>("ldz", size[2]);
    arg.set<double>("vl", size[3]);
    arg.set<double>("vu", size[4]);
    arg.set<rocblas_int>("il", size[5]);
    arg.set<rocblas_int>("iu", size[6]);

    arg.set<char>("evect", op[0]);
    arg.set<char>("erange", op[1]);
    arg.set<char>("uplo", op[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class SYEVDX_HEEVDX : public ::TestWithParam<syevdx_heevdx_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        using S = decltype(std::real(T{}));

        Arguments arg = syevdx_heevdx_setup_arguments<T>(GetParam(), false);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("erange") == 'V' && arg.peek<char>("uplo") == 'L')
            testing_syevdx_heevdx_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevdx_heevdx<BATCHED, STRIDED, T>(arg);
    }
};

class SYEVDX : public SYEVDX_HEEVDX
{
};

class HEEVDX : public SYEVDX_HEEVDX
{
};

class SYEVDX_HEEVDX_INPLACE : public ::TestWithParam<syevdx_heevdx_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        using S = decltype(std::real(T{}));

        Arguments arg = syevdx_heevdx_setup_arguments<T>(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("erange") == 'V' && arg.peek<char>("uplo") == 'L')
            testing_syevdx_heevdx_inplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevdx_heevdx_inplace<BATCHED, STRIDED, T>(arg);
    }
};

class SYEVDX_INPLACE : public SYEVDX_HEEVDX_INPLACE
{
};

class HEEVDX_INPLACE : public SYEVDX_HEEVDX_INPLACE
{
};

// non-batch tests

TEST_P(SYEVDX, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVDX, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVDX, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVDX, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVDX_INPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVDX_INPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVDX_INPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVDX_INPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYEVDX, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYEVDX, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEEVDX, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEEVDX, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEVDX, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVDX, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVDX, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVDX, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYEVDX,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVDX, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEEVDX,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVDX, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYEVDX_INPLACE,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVDX_INPLACE,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEEVDX_INPLACE,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVDX_INPLACE,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
