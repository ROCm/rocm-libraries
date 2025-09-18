/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_syevd_heevd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> syevd_heevd_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {jobz, uplo}

// case when n == -1, jobz == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments syevd_heevd_setup_arguments(syevd_heevd_tuple tup)
{
    vector<int>  size = std::get<0>(tup);
    vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("jobz", op[0]);
    arg.set<char>("uplo", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYEVD_HEEVD : public ::TestWithParam<syevd_heevd_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syevd_heevd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syevd_heevd_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_syevd_heevd<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYEVD : public SYEVD_HEEVD<API_NORMAL>
{
};

class HEEVD : public SYEVD_HEEVD<API_NORMAL>
{
};

class SYEVD_FORTRAN : public SYEVD_HEEVD<API_FORTRAN>
{
};

class HEEVD_FORTRAN : public SYEVD_HEEVD<API_FORTRAN>
{
};

class SYEVD_COMPAT : public SYEVD_HEEVD<API_COMPAT>
{
};

class HEEVD_COMPAT : public SYEVD_HEEVD<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYEVD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVD_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVD_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVD_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVD_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVD,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVD, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVD,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVD, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVD_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVD_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVD_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVD_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVD_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVD_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVD_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVD_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
