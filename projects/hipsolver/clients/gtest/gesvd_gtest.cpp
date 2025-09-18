/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_gesvd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvd_tuple;

// each size_range vector is a {m, n, fa};
// if fa = 0 then no fast algorithm is allowed
// if fa = 1 fast algorithm is used when possible

// each opt_range vector is a {lda, ldu, ldv, leftsv, rightsv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit
// if leftsv (rightsv) = 0 then overwrite singular vectors
// if leftsv (rightsv) = 1 then compute singular vectors
// if leftsv (rightsv) = 2 then compute all orthogonal matrix
// if leftsv (rightsv) = 3 then no singular vectors are computed

// case when m = -1, n = 1, and rightsv = leftsv = 3 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1, 0},
    {1, -1, 0},
    // normal (valid) samples
    {1, 1, 0},
    {20, 20, 0},
    {40, 30, 0},
    {60, 30, 0}};

const vector<vector<int>> opt_range = {
    // invalid
    {-1, 0, 0, 2, 2},
    {0, -1, 0, 1, 2},
    {0, 0, -1, 2, 1},
    {0, 0, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 1, 3, 3},
    {0, 0, 1, 3, 2},
    {0, 1, 0, 3, 1},
    {0, 1, 1, 3, 0},
    {1, 0, 0, 2, 3},
    {1, 0, 1, 2, 2},
    {1, 1, 0, 2, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 1, 3},
    {0, 0, 0, 1, 2},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 3},
    {0, 0, 0, 0, 2},
    {0, 0, 0, 0, 1}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{120, 100, 0}, {300, 120, 0}};

// const vector<vector<int>> large_opt_range = {{0, 0, 0, 3, 3},
//                                              {1, 0, 0, 0, 1},
//                                              {0, 1, 0, 1, 0},
//                                              {0, 0, 1, 1, 1},
//                                              {0, 0, 0, 3, 0},
//                                              {0, 0, 0, 1, 3},
//                                              {0, 0, 0, 3, 2}};

Arguments gesvd_setup_arguments(gesvd_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt  = std::get<1>(tup);

    Arguments arg;

    // sizes
    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    // // fast algorithm
    // if(size[2] == 0)
    //     arg.set<char>("fast_alg", 'I');
    // else
    //     arg.set<char>("fast_alg", 'O');

    // leading dimensions
    arg.set<rocblas_int>("lda", m + opt[0] * 10);
    arg.set<rocblas_int>("ldu", m + opt[1] * 10);
    if(opt[4] == 2)
        arg.set<rocblas_int>("ldv", n + opt[2] * 10);
    else
        arg.set<rocblas_int>("ldv", min(m, n) + opt[2] * 10);

    // vector options
    if(opt[3] == 0)
        arg.set<char>("jobu", 'O');
    else if(opt[3] == 1)
        arg.set<char>("jobu", 'S');
    else if(opt[3] == 2)
        arg.set<char>("jobu", 'A');
    else
        arg.set<char>("jobu", 'N');

    if(opt[4] == 0)
        arg.set<char>("jobv", 'O');
    else if(opt[4] == 1)
        arg.set<char>("jobv", 'S');
    else if(opt[4] == 2)
        arg.set<char>("jobv", 'A');
    else
        arg.set<char>("jobv", 'N');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, bool NRWK>
class GESVD_BASE : public ::TestWithParam<gesvd_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == -1 && arg.peek<rocblas_int>("n") == 1
           && arg.peek<char>("jobu") == 'N' && arg.peek<char>("jobv") == 'N')
            testing_gesvd_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_gesvd<API, BATCHED, STRIDED, NRWK, T>(arg);
    }
};

class GESVD : public GESVD_BASE<API_NORMAL, false>
{
};

class GESVD_FORTRAN : public GESVD_BASE<API_FORTRAN, false>
{
};

class GESVD_COMPAT : public GESVD_BASE<API_COMPAT, false>
{
};

class GESVD_NRWK : public GESVD_BASE<API_NORMAL, true>
{
};

// non-batch tests

TEST_P(GESVD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVD_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVD_NRWK, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD_NRWK, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD_NRWK, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD_NRWK, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVD,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GESVD, Combine(ValuesIn(size_range), ValuesIn(opt_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVD_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVD_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVD_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVD_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVD_NRWK,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVD_NRWK,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));
