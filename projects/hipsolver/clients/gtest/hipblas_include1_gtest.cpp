/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_hipblas_include1.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> hipblas_include1_tuple;

// each size_range vector is a {M, N}

// each store_range vector is a {lda, ldc, s, t, u}
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
// if u = 0, then uplo = 'U'
// if u = 1, then uplo = 'L'

// case when m = -1, n = 1, side = 'L', trans = 'T' and uplo = 'U'
// will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> store_range = {
    // invalid
    {-1, 0, 0, 0, 0},
    {0, -1, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 0, 0, 0},
    {1, 1, 0, 0, 1},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 2, 1},
    {0, 0, 1, 0, 0},
    {0, 0, 1, 0, 1},
    {0, 0, 1, 1, 0},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 2, 0},
    {0, 0, 1, 2, 1},
};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1},
    {1, -1},
    // normal (valid) samples
    {10, 30},
    {20, 5},
    {20, 20},
    {50, 50},
    {70, 40},
};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {
//     {200, 150},
//     {270, 270},
//     {400, 400},
//     {800, 500},
//     {1500, 1000},
// };

Arguments hipblas_include1_setup_arguments(hipblas_include1_tuple tup)
{
    vector<int> size  = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    int nq = store[2] == 0 ? m : n;

    arg.set<rocblas_int>("lda", nq + store[0] * 10);
    arg.set<rocblas_int>("ldc", m + store[1] * 10);
    arg.set<char>("side", store[2] == 0 ? 'L' : 'R');
    arg.set<char>("trans", (store[3] == 0 ? 'N' : (store[3] == 1 ? 'T' : 'C')));
    arg.set<char>("uplo", store[4] == 0 ? 'U' : 'L');

    arg.timing = 0;

    return arg;
}

class HIPBLAS_INCLUDE1 : public ::TestWithParam<hipblas_include1_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = hipblas_include1_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_hipblas_include1<T>(arg);
    }
};

// non-batch tests

TEST_P(HIPBLAS_INCLUDE1, __float)
{
    run_tests<float>();
}

TEST_P(HIPBLAS_INCLUDE1, __double)
{
    run_tests<double>();
}

TEST_P(HIPBLAS_INCLUDE1, __float_complex)
{
    run_tests<hipsolverComplex>();
}

TEST_P(HIPBLAS_INCLUDE1, __double_complex)
{
    run_tests<hipsolverDoubleComplex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HIPBLAS_INCLUDE1,
//                          Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HIPBLAS_INCLUDE1,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));
