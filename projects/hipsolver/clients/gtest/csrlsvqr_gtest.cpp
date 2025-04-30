/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrlsvqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, vector<int>> csrlsvqr_tuple;

// each n_range vector is {n}

// each nnz_range vector is {nnzA, reorder, base1}

// case when n = 20 and nnz = 60 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<int> n_range = {
    20,
    50,
};
const vector<vector<int>> nnz_range = {
    {60, 0, 1},
    {60, 1, 0},
    {100, 2, 0},
    {140, 3, 1},
};

// for daily_lapack tests
const vector<int> large_n_range = {
    // normal (valid) samples
    100,
    250,
};
const vector<vector<int>> large_nnz_range = {
    // normal (valid) samples
    {300, 0, 0},
    {300, 1, 1},
    {500, 2, 1},
    {700, 3, 0},
};

Arguments csrlsvqr_setup_arguments(csrlsvqr_tuple tup)
{
    int         n_v   = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v);
    arg.set<rocblas_int>("nnzA", nnz_v[0]);
    arg.set<rocblas_int>("reorder", nnz_v[1]);
    arg.set<rocblas_int>("base1", nnz_v[2]);

    arg.timing = 0;

    return arg;
}

template <bool HOST>
class CSRLSVQR_BASE : public ::TestWithParam<csrlsvqr_tuple>
{
protected:
    void SetUp() override
    {
        if(hipsolverSpCreate(nullptr) == HIPSOLVER_STATUS_NOT_SUPPORTED)
            GTEST_SKIP() << "Sparse dependencies could not be loaded";
    }
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrlsvqr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 20 && arg.peek<rocblas_int>("nnzA") == 60)
            testing_csrlsvqr_bad_arg<HOST, T>();

        arg.batch_count = 1;
        testing_csrlsvqr<HOST, T>(arg);
    }
};

class CSRLSVQR : public CSRLSVQR_BASE<false>
{
};

class CSRLSVQRHOST : public CSRLSVQR_BASE<true>
{
};

// non-batch tests

TEST_P(CSRLSVQR, __float)
{
    run_tests<float>();
}

TEST_P(CSRLSVQR, __double)
{
    run_tests<double>();
}

// TEST_P(CSRLSVQR, __float_complex)
// {
//     run_tests<rocblas_float_complex>();
// }

// TEST_P(CSRLSVQR, __double_complex)
// {
//     run_tests<rocblas_double_complex>();
// }

// TEST_P(CSRLSVQRHOST, __float)
// {
//     run_tests<float>();
// }

// TEST_P(CSRLSVQRHOST, __double)
// {
//     run_tests<double>();
// }

// TEST_P(CSRLSVQRHOST, __float_complex)
// {
//     run_tests<rocblas_float_complex>();
// }

// TEST_P(CSRLSVQRHOST, __double_complex)
// {
//     run_tests<rocblas_double_complex>();
// }

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRLSVQR,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, CSRLSVQR, Combine(ValuesIn(n_range), ValuesIn(nnz_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          csrlsvqrHOST,
//                          Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

// INSTANTIATE_TEST_SUITE_P(checkin_lapack,
//                          csrlsvqrHOST,
//                          Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
