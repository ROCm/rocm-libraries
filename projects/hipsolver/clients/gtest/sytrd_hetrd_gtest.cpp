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

#include "testing_sytrd_hetrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> sytrd_tuple;

// each matrix_size_range is a {n, lda}

// case when n = -1 and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {152, 152},
//     {640, 640},
//     {1000, 1024},
// };

Arguments sytrd_setup_arguments(sytrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char        uplo        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYTRD_HETRD : public ::TestWithParam<sytrd_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sytrd_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'U' && arg.peek<rocblas_int>("n") == -1)
            testing_sytrd_hetrd_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_sytrd_hetrd<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYTRD : public SYTRD_HETRD<API_NORMAL>
{
};

class HETRD : public SYTRD_HETRD<API_NORMAL>
{
};

class SYTRD_FORTRAN : public SYTRD_HETRD<API_FORTRAN>
{
};

class HETRD_FORTRAN : public SYTRD_HETRD<API_FORTRAN>
{
};

class SYTRD_COMPAT : public SYTRD_HETRD<API_COMPAT>
{
};

class HETRD_COMPAT : public SYTRD_HETRD<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYTRD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HETRD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HETRD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYTRD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HETRD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HETRD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYTRD_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRD_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HETRD_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HETRD_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRD,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HETRD,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRD_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HETRD_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRD_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HETRD_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
