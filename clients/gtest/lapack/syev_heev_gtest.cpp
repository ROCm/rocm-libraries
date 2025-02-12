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

#include "common/lapack/testing_syev_heev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> syev_heev_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {evect, uplo}

// case when n == 0, evect == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments syev_heev_setup_arguments(syev_heev_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("evect", op[0]);
    arg.set<char>("uplo", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <rocblas_int MODE>
class SYEV_HEEV : public ::TestWithParam<syev_heev_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syev_heev_setup_arguments(GetParam());
        arg.alg_mode = MODE;

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syev_heev_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syev_heev<BATCHED, STRIDED, T>(arg);
    }
};

class SYEV : public SYEV_HEEV<0>
{
};

class HEEV : public SYEV_HEEV<0>
{
};

class SYEV_HYBRID : public SYEV_HEEV<1>
{
};

class HEEV_HYBRID : public SYEV_HEEV<1>
{
};

// non-batch tests

TEST_P(SYEV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEV_HYBRID, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEV_HYBRID, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEV_HYBRID, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEV_HYBRID, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYEV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYEV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEEV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEEV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(SYEV_HYBRID, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYEV_HYBRID, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEEV_HYBRID, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEEV_HYBRID, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(SYEV_HYBRID, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEV_HYBRID, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEV_HYBRID, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEV_HYBRID, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack, SYEV, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, HEEV, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYEV_HYBRID,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEEV_HYBRID,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEV, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEV, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEV_HYBRID,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEV_HYBRID,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
