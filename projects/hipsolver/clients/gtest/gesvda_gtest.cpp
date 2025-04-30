/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesvda.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvda_tuple;

// each size_range vector is a {m, n, lda, ldu, ldv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit

// each opt_range vector is a {vect, rank};
// if vect = 1 then compute singular vectors
// if vect = 0 then no singular vectors are computed

// case when m = n = 0, vect = 0 and rank = 1 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    {20, 20, 0, 0, 0}, {40, 30, 0, 0, 0}, {30, 30, 1, 0, 0}, {60, 40, 0, 1, 0}, {50, 50, 1, 1, 1}};

const vector<vector<int>> opt_range = {
    {0, 5},
    {0, 15},
    {1, 5},
    {1, 20},
};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range
//     = {{100, 100, 1, 0, 0}, {300, 120, 0, 0, 1}, {200, 300, 0, 0, 0}};

// const vector<vector<int>> large_opt_range = {{0, 100}, {1, 10}, {1, 20}};

Arguments gesvda_setup_arguments(gesvda_tuple tup)
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
    arg.set<rocblas_int>("lda", m + size[2] * 10);
    arg.set<rocblas_int>("ldu", m + size[3] * 10);
    arg.set<rocblas_int>("ldv", min(m, n) + size[4] * 10);

    // vector options
    if(opt[0] == 0)
        arg.set<char>("jobz", 'N');
    else
        arg.set<char>("jobz", 'V');

    // ranges
    arg.set<rocblas_int>("rank", opt[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class GESVDA_BASE : public ::TestWithParam<gesvda_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvda_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0
           && arg.peek<char>("jobz") == 'N' && arg.peek<rocblas_int>("rank") == 1)
            testing_gesvda_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gesvda<API, BATCHED, STRIDED, T>(arg);
    }
};

class GESVDA_COMPAT : public GESVDA_BASE<API_COMPAT>
{
};

// strided_batched tests

TEST_P(GESVDA_COMPAT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESVDA_COMPAT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESVDA_COMPAT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESVDA_COMPAT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// // daily_lapack tests normal execution with medium to large sizes
// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GESVDA_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVDA_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));
