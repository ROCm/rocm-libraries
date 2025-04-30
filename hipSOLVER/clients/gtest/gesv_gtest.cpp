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

#include "testing_gesv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gesv_tuple;

// each A_range vector is a {N, lda, ldb/ldx};

// each B_range vector is a {nrhs};

// case when N = nrhs = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};
const vector<int> matrix_sizeB_range = {
    // invalid
    -1,
    // normal (valid) samples
    10,
    20,
    30,
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_sizeA_range
//     = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};
// const vector<int> large_matrix_sizeB_range = {
//     100,
//     150,
//     200,
//     524,
//     1000,
// };

Arguments gesv_setup_arguments(gesv_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    int         matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);
    arg.set<rocblas_int>("ldx", matrix_sizeA[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, bool INPLACE>
class GESV_BASE : public ::TestWithParam<gesv_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesv_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<rocblas_int>("nrhs") == -1)
            testing_gesv_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_gesv<API, BATCHED, STRIDED, INPLACE, T>(arg);
    }
};

class GESV : public GESV_BASE<API_NORMAL, false>
{
};

class GESV_FORTRAN : public GESV_BASE<API_FORTRAN, false>
{
};

class GESV_COMPAT : public GESV_BASE<API_COMPAT, false>
{
};

class GESV_INPLACE : public GESV_BASE<API_NORMAL, true>
{
};

// non-batch tests

TEST_P(GESV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESV_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESV_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
TEST_P(GESV_INPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV_INPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV_INPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV_INPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}
#endif

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESV,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESV_FORTRAN,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV_FORTRAN,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESV_COMPAT,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV_COMPAT,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESV_INPLACE,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV_INPLACE,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
#endif
