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

#include "testing_syevj_heevj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> syevj_heevj_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {jobz, uplo}

// case when n == 1, jobz == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{192, 192}, {256, 270}, {300, 300}};

template <typename T>
Arguments syevj_heevj_setup_arguments(syevj_heevj_tuple tup)
{
    vector<int>  size = std::get<0>(tup);
    vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("jobz", op[0]);
    arg.set<char>("uplo", op[1]);

    arg.set<double>("tolerance", 2 * get_epsilon<T>());
    arg.set<rocblas_int>("max_sweeps", 100);
    arg.set<rocblas_int>("sort_eig", 1);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYEVJ_HEEVJ : public ::TestWithParam<syevj_heevj_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syevj_heevj_setup_arguments<T>(GetParam());

        if(arg.peek<rocblas_int>("n") == 1 && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syevj_heevj_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevj_heevj<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYEVJ : public SYEVJ_HEEVJ<API_NORMAL>
{
};

class HEEVJ : public SYEVJ_HEEVJ<API_NORMAL>
{
};

class SYEVJ_FORTRAN : public SYEVJ_HEEVJ<API_FORTRAN>
{
};

class HEEVJ_FORTRAN : public SYEVJ_HEEVJ<API_FORTRAN>
{
};

class SYEVJ_COMPAT : public SYEVJ_HEEVJ<API_COMPAT>
{
};

class HEEVJ_COMPAT : public SYEVJ_HEEVJ<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYEVJ, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVJ, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVJ, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVJ, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVJ_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVJ_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVJ_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVJ_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVJ_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVJ_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVJ_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVJ_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEVJ, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVJ, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVJ, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVJ, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(SYEVJ_FORTRAN, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVJ_FORTRAN, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVJ_FORTRAN, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVJ_FORTRAN, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(SYEVJ_COMPAT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVJ_COMPAT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVJ_COMPAT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVJ_COMPAT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVJ,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVJ, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVJ,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVJ, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVJ_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVJ_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVJ_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVJ_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVJ_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVJ_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVJ_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVJ_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
