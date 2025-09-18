/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrrf_solve.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> csrrf_solve_tuple;

// each n_range vector is {n, ldb}

// each nnz_range vector is {nnzT, nrhs}

// case when n = 20 and nnz = 60 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> n_range = {
    {20, 20},
    {50, 60},
};
const vector<vector<int>> nnz_range = {
    {60, 1},
    {100, 1},
    {140, 1},
};

// // for daily_lapack tests
// const vector<vector<int>> large_n_range = {
//     // normal (valid) samples
//     {100, 110},
//     {250, 250},
// };
// const vector<vector<int>> large_nnz_range = {
//     // normal (valid) samples
//     {300, 1},
//     {500, 1},
//     {700, 1},
// };

Arguments csrrf_solve_setup_arguments(csrrf_solve_tuple tup)
{
    vector<int> n_v   = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v[0]);
    arg.set<rocblas_int>("ldb", n_v[1]);
    arg.set<rocblas_int>("nnzA", nnz_v[0]);
    arg.set<rocblas_int>("nrhs", nnz_v[1]);

    arg.timing = 0;

    return arg;
}

class CSRRF_SOLVE : public ::TestWithParam<csrrf_solve_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_solve_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 20 && arg.peek<rocblas_int>("nnzA") == 60)
            testing_csrrf_solve_bad_arg<T>();

        arg.batch_count = 1;
        testing_csrrf_solve<T>(arg);
    }
};

// non-batch tests

/*TEST_P(CSRRF_SOLVE, __float)
{
    run_tests<float>();
}*/

TEST_P(CSRRF_SOLVE, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_SOLVE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_SOLVE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          CSRRF_SOLVE,
//                          Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRRF_SOLVE,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
