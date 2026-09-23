/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_local_matrix.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, std::string> local_matrix_conversions_tuple;
typedef std::tuple<int, int>              local_matrix_allocations_tuple;
typedef std::tuple<int, int>              local_matrix_zero_tuple;

int         local_matrix_conversions_size[]     = {10, 17, 21};
int         local_matrix_conversions_blockdim[] = {4, 7, 11};
std::string local_matrix_type[]                 = {"Laplacian2D", "PermutedIdentity", "Random"};

int local_matrix_allocations_size[]     = {100, 1475, 2524};
int local_matrix_allocations_blockdim[] = {4, 7, 11};

class parameterized_local_matrix_conversions
    : public testing::TestWithParam<local_matrix_conversions_tuple>
{
protected:
    parameterized_local_matrix_conversions() {}
    virtual ~parameterized_local_matrix_conversions() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_local_matrix_conversions_arguments(local_matrix_conversions_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.blockdim    = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    return arg;
}

class parameterized_local_matrix_allocations
    : public testing::TestWithParam<local_matrix_allocations_tuple>
{
protected:
    parameterized_local_matrix_allocations() {}
    virtual ~parameterized_local_matrix_allocations() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_local_matrix_allocations_arguments(local_matrix_allocations_tuple tup)
{
    Arguments arg;
    arg.size     = std::get<0>(tup);
    arg.blockdim = std::get<1>(tup);
    return arg;
}

TEST(local_matrix_bad_args, local_matrix)
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED"}))
    {
        GTEST_SKIP();
    }

    testing_local_matrix_bad_args<float>();
}

TEST_P(parameterized_local_matrix_conversions, local_matrix_conversions_float)
{
    Arguments arg = setup_local_matrix_conversions_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_conversions<float>(arg), true);
}

TEST_P(parameterized_local_matrix_conversions, local_matrix_conversions_double)
{
    Arguments arg = setup_local_matrix_conversions_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_conversions<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_conversions,
                        parameterized_local_matrix_conversions,
                        testing::Combine(testing::ValuesIn(local_matrix_conversions_size),
                                         testing::ValuesIn(local_matrix_conversions_blockdim),
                                         testing::ValuesIn(local_matrix_type)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_allocations_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_allocations<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_allocations_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_allocations<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_allocations,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_zero_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_zero<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_zero_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_zero<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_zero,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_set_data_ptr_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_set_data_ptr<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_set_data_ptr_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_set_data_ptr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_set_data_ptr,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));

// Test fixture for LocalMatrix
class local_matrix_test : public testing::TestWithParam<local_matrix_conversions_tuple>
{
protected:
    void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

class parameterized_local_matrix_test
    : public testing::TestWithParam<local_matrix_conversions_tuple>
{
protected:
    parameterized_local_matrix_test() {}
    virtual ~parameterized_local_matrix_test() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

#define TEST_CASE(function, type, disable_accelerator)   \
    disable_accelerator_rocalution(disable_accelerator); \
    set_device_rocalution(0);                            \
    init_rocalution();                                   \
    function<type>();                                    \
    stop_rocalution();                                   \
    disable_accelerator_rocalution(false);

#define GENERATE_TEST_CASES_DEVICE(test_fixture, function, name) \
    TEST_F(test_fixture, name##_device_float)                    \
    {                                                            \
        TEST_CASE(function, float, false);                       \
    }                                                            \
    TEST_F(test_fixture, name##_device_double)                   \
    {                                                            \
        TEST_CASE(function, double, false);                      \
    }

#define GENERATE_TEST_CASES_HOST(test_fixture, function, name) \
    TEST_F(test_fixture, name##_host_float)                    \
    {                                                          \
        TEST_CASE(function, float, true);                      \
    }                                                          \
    TEST_F(test_fixture, name##_host_double)                   \
    {                                                          \
        TEST_CASE(function, double, true);                     \
    }
#define GENERATE_TEST_CASES(test_fixture, function, name)    \
    GENERATE_TEST_CASES_DEVICE(test_fixture, function, name) \
    GENERATE_TEST_CASES_HOST(test_fixture, function, name)

#define TEST_P_CASE(function, type, disable_accelerator)                  \
    disable_accelerator_rocalution(disable_accelerator);                  \
    set_device_rocalution(0);                                             \
    init_rocalution();                                                    \
    Arguments arg = setup_local_matrix_conversions_arguments(GetParam()); \
    function<type>(arg);                                                  \
    stop_rocalution();                                                    \
    disable_accelerator_rocalution(false);

#define GENERATE_P_TEST_CASES_DEVICE(test_fixture, function, name) \
    TEST_P(test_fixture, name##_device_float)                      \
    {                                                              \
        TEST_P_CASE(function, float, false);                       \
    }                                                              \
    TEST_P(test_fixture, name##_device_double)                     \
    {                                                              \
        TEST_P_CASE(function, double, false);                      \
    }

#define GENERATE_P_TEST_CASES_HOST(test_fixture, function, name) \
    TEST_P(test_fixture, name##_host_float)                      \
    {                                                            \
        TEST_P_CASE(function, float, true);                      \
    }                                                            \
    TEST_P(test_fixture, name##_host_double)                     \
    {                                                            \
        TEST_P_CASE(function, double, true);                     \
    }
#define GENERATE_P_TEST_CASES(test_fixture, function, name)    \
    GENERATE_P_TEST_CASES_DEVICE(test_fixture, function, name) \
    GENERATE_P_TEST_CASES_HOST(test_fixture, function, name)

// Test for matrix allocation
GENERATE_TEST_CASES(local_matrix_test, testing_local_allocate, allocate_matrix)

// Test for matrix check with empty matrix
GENERATE_TEST_CASES(local_matrix_test, testing_check_with_empty_matrix, check_with_empty_matrix)

// Test for matrix clearing
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_clear, clear_matrix)

// Test for zeros matrix
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_zeros, zeros_matrix)

// Test for matrix copy
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_copy, copy_matrix)

// Test for matrix scaling
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_scale, scale_matrix)

// Test for extracting diagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_extract_diagonal,
                      extract_diagonal)

// Test for extracting inverse diagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_extract_inverse_diagonal,
                      extract_inverse_diagonal)

// Test for matrix transpose
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_transpose, transpose_matrix)

// Test for matrix clone
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_clone_matrix, clone_matrix)

// Test for matrix inversion
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_invert_matrix, invert_matrix)

// Test for setting data pointers
GENERATE_TEST_CASES(local_matrix_test, testing_set_data_pointer, set_data_pointer)

// Test for leaving data pointers
GENERATE_TEST_CASES(local_matrix_test, testing_leave_data_pointer, leave_data_pointer)

// Test for MaximalIndependentSet
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_maximal_independent_set,
                      maximal_independent_set)

// Test for ZeroBlockPermutation
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_zero_block_permutation,
                      zero_block_permutation)

// Test for Householder
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_Householder, householder)

// Test for CMK
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_CMK, CMK)

// Test for RCMK
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_RCMK, RCMK)

// Test for ConnectivityOrder
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_connectivity_order,
                      connectivity_order)

// Test for SymbolicPower
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_symbolic_power, symbolic_power)

// Test for ScaleOffDiagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_scale_off_diagonal,
                      scale_off_diagonal)

// Test for AddScalar
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_add_scalar, add_scalar)

// Test for ReplaceColumnVector
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_replace_column_vector,
                      replace_column_vector)

// Test for ExtractColumnVector
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_extract_column_vector,
                      extract_column_vector)

// Test for SetDataPtrDIA
GENERATE_TEST_CASES(local_matrix_test, testing_set_data_ptr_DIA, set_data_ptr_dia)

// Test for LeaveDataPtrDIA
GENERATE_TEST_CASES(local_matrix_test, testing_leave_data_ptr_DIA, leave_data_ptr_dia)

// Test for CopyFromCOO
GENERATE_TEST_CASES(local_matrix_test, testing_copy_from_COO, copy_from_coo)

// Test for CopyToCOO
GENERATE_TEST_CASES(local_matrix_test, testing_copy_to_COO, copy_to_coo)

// Test for CopyFromHostCSR
GENERATE_TEST_CASES(local_matrix_test, testing_copy_from_host_CSR, copy_from_host_csr)

// Test for ReadFileMTX
GENERATE_TEST_CASES(local_matrix_test, testing_read_file_MTX, read_file_mtx_to_csr)

// Test case for LocalMatrix::WriteFileMTX
GENERATE_TEST_CASES(local_matrix_test, testing_write_file_MTX, write_file_mtx_to_csr)

// Test case for LocalMatrix::WriteFileRSIO
GENERATE_TEST_CASES(local_matrix_test, testing_write_file_RSIO, write_file_RSIO)

// Test case for LocalMatrix::CopyFromAsync
GENERATE_TEST_CASES_DEVICE(local_matrix_test, testing_local_copy_from_async, local_copy_from_async)

// Test case for LocalMatrix::CopyToAsync
GENERATE_TEST_CASES(local_matrix_test, testing_local_update_values_csr, local_update_values_csr)

// Test case for LocalMatrix::CopyFromHostAsync
GENERATE_TEST_CASES_DEVICE(local_matrix_test,
                           testing_local_move_to_accelerator,
                           local_move_to_accelerator)

// Test case for LocalMatrix::CopyToHostAsync
GENERATE_TEST_CASES_DEVICE(local_matrix_test,
                           testing_local_move_to_accelerator_async,
                           local_move_to_accelerator_async)

// Test case for LocalMatrix::CopyFromHostAsync
GENERATE_TEST_CASES_DEVICE(local_matrix_test,
                           testing_local_move_to_host_async,
                           local_move_to_host_async)

// Test case for LocalMatrix::Apply
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_apply, local_apply)

// Test case for LocalMatrix::ApplyAdd
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_apply_add, local_apply_add)

// Test case for LocalMatrix::ExtractSubmatrix
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_extract_submatrix,
                      local_extract_submatrix)

// Test case for LocalMatrix::ExtractU
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_extract_u, local_extract_u)

// Test case for LocalMatrix::ExtractL
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_extract_l, local_extract_l)

// Test case for LocalMatrix::MatrixAdd
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_matrix_add, local_matrix_add)

// Test case for LocalMatrix::Gershgorin
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_gershgorin, local_gershgorin)

// Test case for LocalMatrix::ScaleDiagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_scale_diagonal,
                      local_scale_diagonal)

// Test case for LocalMatrix::AddScalarDiagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_add_scalar_diagonal,
                      local_add_scalar_diagonal)

// Test case for LocalMatrix::AddScalarOffDiagonal
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_add_scalar_off_diagonal,
                      local_add_scalar_off_diagonal)

// Test case for LocalMatrix::MatrixMult
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_matrix_mult, local_matrix_mult)

// Test case for LocalMatrix::TripleMatrixProduct
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_triple_matrix_product,
                      local_triple_matrix_product)

// Test case for LocalMatrix::DiagonalMatrixMultR
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_diagonal_matrix_mult_r,
                      local_diagonal_matrix_mult_r)

// Test case for LocalMatrix::DiagonalMatrixMult
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_diagonal_matrix_mult,
                      local_diagonal_matrix_mult)

// Test case for LocalMatrix::DiagonalMatrixMultL
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_diagonal_matrix_mult_l,
                      local_diagonal_matrix_mult_l)

// Test case for LocalMatrix::Compress
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_compress, local_compress)

// Test case for LocalMatrix::ReplaceRowVector
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_replace_row_vector,
                      local_replace_row_vector)

// Test case for LocalMatrix::ExtractRowVector
GENERATE_P_TEST_CASES(parameterized_local_matrix_test,
                      testing_local_extract_row_vector,
                      local_extract_row_vector)

// Test case for LocalMatrix::Key
GENERATE_P_TEST_CASES(parameterized_local_matrix_test, testing_local_key, local_key)

// Test case for LocalMatrix::InitialPairwiseAggregation
GENERATE_TEST_CASES(local_matrix_test,
                    testing_initial_pairwise_aggregation,
                    initial_pairwise_aggregation)

// Test case for LocalMatrix::InitialPairwiseAggregation
GENERATE_TEST_CASES(local_matrix_test,
                    testing_initial_pairwise_aggregation_2,
                    initial_pairwise_aggregation_2)

INSTANTIATE_TEST_CASE_P(local_matrix_test,
                        parameterized_local_matrix_test,
                        testing::Combine(testing::ValuesIn(local_matrix_conversions_size),
                                         testing::ValuesIn(local_matrix_conversions_blockdim),
                                         testing::ValuesIn(local_matrix_type)));