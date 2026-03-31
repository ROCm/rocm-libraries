/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_ormtr_unmtr_hb2st.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormtr_hb2st_tuple;

// each size_range vector is a {M, N, KD}

// each store_range vector is a {ldv, ldc, side, trans}
// if ldv = -1, then ldv < limit (invalid size)
// if ldv = 0, then ldv = limit
// if ldv = 1, then ldv > limit
// if ldc = -1, then ldc < limit (invalid size)
// if ldc = 0, then ldc = limit
// if ldc = 1, then ldc > limit
// if side = 0, then side = 'L'
// if side = 1, then side = 'R'
// if trans = 0, then trans = 'N'
// if trans = 1, then trans = 'T' or 'C' (transpose or conjugate transpose)

// case when m = 0, n = 1, kd = 0, side = 'L', and trans = 'N'
// will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> store_range = {
    // invalid
    {-1, 0, 0, 0},
    {0, -1, 0, 0},
    // normal (valid) samples
    {1, 1, 0, 0},
    {1, 1, 1, 0},
    {0, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 1},
};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 0},
    {1, 0, 0},
    // invalid
    {-1, 1, 0},
    {1, -1, 0},
    {1, 1, -1},
    // normal (valid) samples
    {10, 30, 4},
    {20, 5, 2},
    {20, 20, 5},
    {50, 50, 10},
    {70, 40, 8},
};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {
    {200, 150, 20}, {270, 270, 30}, {400, 400, 40}, {800, 500, 50}, {1500, 1000, 100},
};

Arguments ormtr_hb2st_setup_arguments(ormtr_hb2st_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    rocblas_int m = size[0];
    rocblas_int n = size[1];
    rocblas_int kd = size[2];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("kd", kd);

    arg.set<rocblas_int>("ldv", m + store[0] * 10);
    arg.set<rocblas_int>("ldc", m + store[1] * 10);
    arg.set<char>("side", store[2] == 0 ? 'L' : 'R');
    arg.set<char>("trans", store[3] == 0 ? 'N' : 'T');

    arg.timing = 0;

    return arg;
}

class ORMTR_UNMTR_HB2ST : public ::TestWithParam<ormtr_hb2st_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = ormtr_hb2st_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 1
           && arg.peek<rocblas_int>("kd") == 0 && arg.peek<char>("side") == 'L'
           && arg.peek<char>("trans") == 'N')
            testing_ormtr_unmtr_hb2st_bad_arg<T>();

        testing_ormtr_unmtr_hb2st<T>(arg);
    }
};

class ORMTR_HB2ST : public ORMTR_UNMTR_HB2ST
{
};

class UNMTR_HB2ST : public ORMTR_UNMTR_HB2ST
{
};

// non-batch tests

TEST_P(ORMTR_HB2ST, __float)
{
    run_tests<float>();
}

TEST_P(ORMTR_HB2ST, __double)
{
    run_tests<double>();
}

TEST_P(UNMTR_HB2ST, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMTR_HB2ST, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORMTR_HB2ST,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORMTR_HB2ST,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNMTR_HB2ST,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNMTR_HB2ST,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));
