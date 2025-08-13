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

#include "testing_orgtr_ungtr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> orgtr_tuple;

// each size_range vector is a {n, lda}

// case when n = -1 and uplo = 'U' will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<int> uplo_range = {0, 1};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100},
    {100, 150}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{192, 192}, {500, 600}, {640, 640}, {1000, 1024}};

Arguments orgtr_setup_arguments(orgtr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    int         uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("uplo", uplo == 1 ? 'U' : 'L');

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class ORGTR_UNGTR : public ::TestWithParam<orgtr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgtr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<char>("uplo") == 'U')
            testing_orgtr_ungtr_bad_arg<API, T>();

        testing_orgtr_ungtr<API, T>(arg);
    }
};

class ORGTR : public ORGTR_UNGTR<API_NORMAL>
{
};

class UNGTR : public ORGTR_UNGTR<API_NORMAL>
{
};

class ORGTR_FORTRAN : public ORGTR_UNGTR<API_FORTRAN>
{
};

class UNGTR_FORTRAN : public ORGTR_UNGTR<API_FORTRAN>
{
};

class ORGTR_COMPAT : public ORGTR_UNGTR<API_COMPAT>
{
};

class UNGTR_COMPAT : public ORGTR_UNGTR<API_COMPAT>
{
};

// non-batch tests

TEST_P(ORGTR, __float)
{
    run_tests<float>();
}

TEST_P(ORGTR, __double)
{
    run_tests<double>();
}

TEST_P(UNGTR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGTR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGTR_FORTRAN, __float)
{
    run_tests<float>();
}

TEST_P(ORGTR_FORTRAN, __double)
{
    run_tests<double>();
}

TEST_P(UNGTR_FORTRAN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGTR_FORTRAN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGTR_COMPAT, __float)
{
    run_tests<float>();
}

TEST_P(ORGTR_COMPAT, __double)
{
    run_tests<double>();
}

TEST_P(UNGTR_COMPAT, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGTR_COMPAT, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack, ORGTR, Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGTR,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack, UNGTR, Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGTR,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGTR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGTR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGTR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGTR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGTR_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGTR_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGTR_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGTR_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(uplo_range)));
