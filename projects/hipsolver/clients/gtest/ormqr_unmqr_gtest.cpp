/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_ormqr_unmqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormqr_tuple;

// each size_range vector is a {M, N, K}

// each op_range vector is a {lda, ldc, s, t}
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

// case when m = -1, side = L and trans = T will also execute the bad arguments
// test (null handle, null pointers and invalid values)

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

// // for daily_lapack tests
// const vector<vector<int>> large_size_range
//     = {{100, 100, 100}, {150, 100, 80}, {300, 400, 300}, {1024, 1000, 950}, {1500, 1500, 1000}};

Arguments ormqr_setup_arguments(ormqr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> op   = std::get<1>(tup);

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

template <testAPI_t API>
class ORMQR_UNMQR : public ::TestWithParam<ormqr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = ormqr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == -1 && arg.peek<char>("side") == 'L'
           && arg.peek<char>("trans") == 'T')
            testing_ormqr_unmqr_bad_arg<API, T>();

        testing_ormqr_unmqr<API, T>(arg);
    }
};

class ORMQR : public ORMQR_UNMQR<API_NORMAL>
{
};

class UNMQR : public ORMQR_UNMQR<API_NORMAL>
{
};

class ORMQR_FORTRAN : public ORMQR_UNMQR<API_FORTRAN>
{
};

class UNMQR_FORTRAN : public ORMQR_UNMQR<API_FORTRAN>
{
};

class ORMQR_COMPAT : public ORMQR_UNMQR<API_COMPAT>
{
};

class UNMQR_COMPAT : public ORMQR_UNMQR<API_COMPAT>
{
};

// non-batch tests

TEST_P(ORMQR, __float)
{
    run_tests<float>();
}

TEST_P(ORMQR, __double)
{
    run_tests<double>();
}

TEST_P(UNMQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORMQR_FORTRAN, __float)
{
    run_tests<float>();
}

TEST_P(ORMQR_FORTRAN, __double)
{
    run_tests<double>();
}

TEST_P(UNMQR_FORTRAN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMQR_FORTRAN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORMQR_COMPAT, __float)
{
    run_tests<float>();
}

TEST_P(ORMQR_COMPAT, __double)
{
    run_tests<double>();
}

TEST_P(UNMQR_COMPAT, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMQR_COMPAT, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORMQR,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMQR, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNMQR,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMQR, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORMQR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORMQR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNMQR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNMQR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORMQR_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORMQR_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNMQR_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNMQR_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
