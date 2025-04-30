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

#include "common/auxiliary/testing_ormxl_unmxl.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormql_tuple;

// each size_range vector is a {M, N, K};

// each op_range is a {lda, ldc, s, t}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if ldc = -1, then ldc < limit (invalid size)
// if ldc = 0, then ldc = limit
// if ldc = 1, then ldc > limit
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'
// if t = 0, then trans = 'N'
// if t = 1, then trans = 'T'
// if t = 2, then trans = 'C'

// case when m = 0, side = 'L' and trans = 'T' will also execute the bad
// arguments test (null handle, null pointers and invalid values)

const vector<vector<int>> op_range = {
    // invalid
    {-1, 0, 0, 0},
    {0, -1, 0, 0},
    // normal (valid) samples
    {0, 0, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 0, 2},
    {0, 0, 1, 0},
    {0, 0, 1, 1},
    {0, 0, 1, 2},
    {1, 1, 0, 0}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 0},
    {1, 0, 0},
    {30, 30, 0},
    // always invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // invalid for side = 'R'
    {20, 10, 20},
    // invalid for side = 'L'
    {15, 25, 25},
    // normal (valid) samples
    {40, 40, 40},
    {45, 40, 30},
    {50, 50, 20}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{100, 100, 100}, {150, 100, 80}, {300, 400, 300}, {1024, 1000, 950}, {1500, 1500, 1000}};

Arguments ormql_setup_arguments(ormql_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    Arguments arg;

    rocblas_int m = size[0];
    rocblas_int n = size[1];
    rocblas_int k = size[2];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("k", k);

    if(op[2] == 0)
        arg.set<rocblas_int>("lda", m + op[0] * 10);
    else
        arg.set<rocblas_int>("lda", n + op[0] * 10);
    arg.set<rocblas_int>("ldc", m + op[1] * 10);
    arg.set<char>("side", op[2] == 0 ? 'L' : 'R');
    arg.set<char>("trans", (op[3] == 0 ? 'N' : (op[3] == 1 ? 'T' : 'C')));

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class ORMXL_UNMXL : public ::TestWithParam<ormql_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = ormql_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<char>("side") == 'L'
           && arg.peek<char>("trans") == 'T')
            testing_ormxl_unmxl_bad_arg<T, BLOCKED>();

        testing_ormxl_unmxl<T, BLOCKED>(arg);
    }
};

class ORM2L : public ORMXL_UNMXL<false>
{
};

class UNM2L : public ORMXL_UNMXL<false>
{
};

class ORMQL : public ORMXL_UNMXL<true>
{
};

class UNMQL : public ORMXL_UNMXL<true>
{
};

// non-batch tests

TEST_P(ORM2L, __float)
{
    run_tests<float>();
}

TEST_P(ORM2L, __double)
{
    run_tests<double>();
}

TEST_P(UNM2L, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNM2L, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORMQL, __float)
{
    run_tests<float>();
}

TEST_P(ORMQL, __double)
{
    run_tests<double>();
}

TEST_P(UNMQL, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMQL, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORM2L, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORM2L, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNM2L, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNM2L, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORMQL, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMQL, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNMQL, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMQL, Combine(ValuesIn(size_range), ValuesIn(op_range)));
