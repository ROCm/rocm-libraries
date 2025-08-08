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

#include "testing_sygvj_hegvj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> sygvj_tuple;

// each matrix_size_range is a {n, lda, ldb}

// each type_range is a {itype, jobz, uplo}

// case when n = -1, itype = 1, jobz = 'N', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> type_range = {{'1', 'N', 'U'},
                                         {'2', 'N', 'L'},
                                         {'3', 'N', 'U'},
                                         {'1', 'V', 'L'},
                                         {'2', 'V', 'U'},
                                         {'3', 'V', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1, 1},
    {20, 5, 5},
    // normal (valid) samples
    {20, 30, 20},
    {35, 35, 35},
    {50, 50, 60}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192, 192},
//     {256, 270, 256},
//     {300, 300, 310},
// };

template <typename T>
Arguments sygvj_setup_arguments(sygvj_tuple tup)
{
    vector<int>  matrix_size = std::get<0>(tup);
    vector<char> type        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("jobz", type[1]);
    arg.set<char>("uplo", type[2]);

    arg.set<double>("tolerance", 2 * get_epsilon<T>());
    arg.set<rocblas_int>("max_sweeps", 100);
    arg.set<rocblas_int>("sort_eig", 1);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYGVJ_HEGVJ : public ::TestWithParam<sygvj_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygvj_setup_arguments<T>(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'U' && arg.peek<rocblas_int>("n") == -1)
            testing_sygvj_hegvj_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_sygvj_hegvj<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYGVJ : public SYGVJ_HEGVJ<API_NORMAL>
{
};

class HEGVJ : public SYGVJ_HEGVJ<API_NORMAL>
{
};

class SYGVJ_FORTRAN : public SYGVJ_HEGVJ<API_FORTRAN>
{
};

class HEGVJ_FORTRAN : public SYGVJ_HEGVJ<API_FORTRAN>
{
};

class SYGVJ_COMPAT : public SYGVJ_HEGVJ<API_COMPAT>
{
};

class HEGVJ_COMPAT : public SYGVJ_HEGVJ<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYGVJ, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVJ, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVJ, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVJ, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYGVJ_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVJ_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVJ_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVJ_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYGVJ_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVJ_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVJ_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVJ_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVJ,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVJ,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVJ,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVJ,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVJ_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVJ_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVJ_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVJ_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVJ_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVJ_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVJ_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVJ_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
