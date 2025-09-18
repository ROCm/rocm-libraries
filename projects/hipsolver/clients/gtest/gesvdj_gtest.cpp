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

#include "testing_gesvdj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvdj_tuple;

// each size_range vector is a {m, n, fa};
// if fa = 0 then no fast algorithm is allowed
// if fa = 1 fast algorithm is used when possible

// each opt_range vector is a {lda, ldu, ldv, jobz, econ};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit
// if jobz = 0 then no singular vectors are computed
// if jobz = 1 then compute singular vectors

// case when m = 1, n = 1, jobz = 3, and econ = 0 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // normal (valid) samples
    {1, 1, 0},
    {20, 20, 0},
    {30, 30, 0},
    {32, 30, 0},
    {4, 32, 0},
    {32, 4, 0},
};

const vector<vector<int>> opt_range = {
    // normal (valid) samples
    {1, 1, 1, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 1, 0, 0, 0},
    {1, 0, 0, 1, 1},
    {1, 0, 1, 1, 0},
    {1, 1, 0, 1, 0},
    {0, 0, 0, 1, 1},
};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{120, 100, 0}, {300, 120, 0}};

// const vector<vector<int>> large_opt_range = {{0, 0, 0, 0, 0},
//                                              {0, 0, 1, 1, 1},
//                                              {0, 1, 0, 1, 0},
//                                              {1, 0, 0, 0, 0}};

template <typename T>
Arguments gesvdj_setup_arguments(gesvdj_tuple tup, bool STRIDED)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt  = std::get<1>(tup);

    Arguments arg;

    // sizes
    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    // leading dimensions
    arg.set<rocblas_int>("lda", m + opt[0] * 10);
    arg.set<rocblas_int>("ldu", m + opt[1] * 10);
    arg.set<rocblas_int>("ldv", n + opt[2] * 10);

    // vector options
    if(opt[3] == 0)
        arg.set<char>("jobz", 'N');
    else
        arg.set<char>("jobz", 'V');

    if(!STRIDED)
        arg.set<rocblas_int>("econ", opt[4]);

    arg.set<double>("tolerance", 2 * get_epsilon<T>());
    arg.set<rocblas_int>("max_sweeps", 100);
    arg.set<rocblas_int>("sort_eig", 1);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class GESVDJ_BASE : public ::TestWithParam<gesvdj_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvdj_setup_arguments<T>(GetParam(), STRIDED);

        if(arg.peek<rocblas_int>("m") == 1 && arg.peek<rocblas_int>("n") == 1
           && arg.peek<char>("jobz") == 'N' && (STRIDED || arg.peek<rocblas_int>("econ") == 0))
            testing_gesvdj_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gesvdj<API, BATCHED, STRIDED, T>(arg);
    }
};

class GESVDJ : public GESVDJ_BASE<API_NORMAL>
{
};

class GESVDJ_FORTRAN : public GESVDJ_BASE<API_FORTRAN>
{
};

class GESVDJ_COMPAT : public GESVDJ_BASE<API_COMPAT>
{
};

// non-batch tests

TEST_P(GESVDJ, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVDJ, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVDJ, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVDJ, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVDJ_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVDJ_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVDJ_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVDJ_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVDJ_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVDJ_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVDJ_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVDJ_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GESVDJ, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVDJ, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVDJ, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVDJ, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GESVDJ_FORTRAN, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVDJ_FORTRAN, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVDJ_FORTRAN, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVDJ_FORTRAN, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GESVDJ_COMPAT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVDJ_COMPAT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVDJ_COMPAT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVDJ_COMPAT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}
// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVDJ,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVDJ,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVDJ_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVDJ_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVDJ_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVDJ_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));
