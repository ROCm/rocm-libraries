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

#include "testing_local_vector.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int> local_vector_tuple;

int local_vector_size[] = {10, 17, 21};

/*
typedef std::tuple<int, int, int, int, bool, int, bool> backend_tuple;

int backend_rank[] = {-1, 0, 13};
int backend_dev_node[] = {-1, 0, 13};
int backend_dev[] = {-1, 0, 13};
int backend_omp_threads[] = {-1, 0, 8};
bool backend_affinity[] = {true, false};
int backend_omp_threshold[] = {-1, 0, 20000};
bool backend_disable_acc[] = {true, false};

class parameterized_backend : public testing::TestWithParam<backend_tuple>
{
    protected:
    parameterized_backend() {}
    virtual ~parameterized_backend() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_backend_arguments(backend_tuple tup)
{
    Arguments arg;
    arg.rank          = std::get<0>(tup);
    arg.dev_per_node  = std::get<1>(tup);
    arg.dev           = std::get<2>(tup);
    arg.omp_nthreads  = std::get<3>(tup);
    arg.omp_affinity  = std::get<4>(tup);
    arg.omp_threshold = std::get<5>(tup);
    arg.use_acc       = std::get<6>(tup);
    return arg;
}
*/
TEST(local_vector_bad_args, local_vector)
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED"}))
    {
        GTEST_SKIP();
    }

    testing_local_vector_bad_args<float>();
}
/*
TEST_P(parameterized_backend, backend)
{
    Arguments arg = setup_backend_arguments(GetParam());
    testing_backend(arg);
}

INSTANTIATE_TEST_CASE_P(backend,
                        parameterized_backend,
                        testing::Combine(testing::ValuesIn(backend_rank),
                                         testing::ValuesIn(backend_dev_node),
                                         testing::ValuesIn(backend_dev),
                                         testing::ValuesIn(backend_omp_threads),
                                         testing::ValuesIn(backend_affinity),
                                         testing::ValuesIn(backend_omp_threshold),
                                         testing::ValuesIn(backend_disable_acc)));
*/

// Test fixture for LocalVector
class local_vector_test : public ::testing::Test
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

class parameterized_local_vector_test : public testing::TestWithParam<local_vector_tuple>
{
protected:
    parameterized_local_vector_test() {}
    virtual ~parameterized_local_vector_test() {}
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

Arguments setup_local_vector_arguments(local_vector_tuple tup)
{
    Arguments arg;
    arg.size = std::get<0>(tup);
    return arg;
}

#define TEST_P_CASE(function, type, disable_accelerator)      \
    disable_accelerator_rocalution(disable_accelerator);      \
    set_device_rocalution(0);                                 \
    init_rocalution();                                        \
    Arguments arg = setup_local_vector_arguments(GetParam()); \
    function<type>(arg);                                      \
    stop_rocalution();                                        \
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

// Test for GetInterior
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_get_interior, get_interior)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_const_get_interior,
                      const_get_interior)

// Test for LocalVector::Check
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_check, check)

// Test for LocalVector::SetDataPtr when *ptr == NULL
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_set_data_ptr_null, set_data_ptr_null)

// Test for LocalVector::ScaleAdd2
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_scale_add_2, scale_add_2)

// Test for LocalVector::CopyFrom
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_copy_from, copy_from)

// Test for LocalVector::CloneFrom
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_clone_from, clone_from)

// Test for Binary
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_binary, file_binary)

// Test for LocalVector::AddScale
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_add_scale, add_scale)

// Test for LocalVector::ScaleAdd
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_scale_add, scale_add)

// Test for LocalVector::Scale
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_scale, scale)

// Test for LocalVector::Dot
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_dot, dot)

// Test for LocalVector::DotNonConj
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_dot_non_conj, dot_nonconj)

// Test for LocalVector::Norm
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_norm, norm)

// Test for LocalVector::Reduce
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_reduce, reduce)

// Test for LocalVector::PointWiseMult
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_point_wise_mult, point_wise_mult)
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_point_wise_mult_2, point_wise_mult_2)

// Test for LocalVector::LeaveDataPtr
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_leave_data_ptr, leave_data_ptr)

// Test for LocalVector::Sync
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_sync, sync)

// Test for LocalVector::CopyFromData
GENERATE_P_TEST_CASES_HOST(parameterized_local_vector_test, testing_copy_from_data, copy_from_data)

// Test for LocalVector::CopyToHostData
GENERATE_P_TEST_CASES_HOST(parameterized_local_vector_test,
                           testing_copy_to_host_data,
                           copy_to_host_data)

// Test for LocalVector::Restriction
GENERATE_TEST_CASES(local_vector_test, testing_restriction, restriction)

// Test for LocalVector::CopyFromWithOffsets
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_copy_from_with_offsets,
                      copy_from_with_offsets)

// Test for LocalVector::CopyFromFloat
TEST_F(local_vector_test, copy_from_float)
{
    disable_accelerator_rocalution(false);
    set_device_rocalution(0);
    init_rocalution();
    // Create a LocalVector<float> and allocate some size
    LocalVector<float> src_vec = getTestVector<float>(5);

    // Create a LocalVector<double> to copy the data into
    LocalVector<double> dest_vec;

    // Perform the copy
    EXPECT_NO_THROW(dest_vec.CopyFromFloat(src_vec));

    // Validate the copied values
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_DOUBLE_EQ(dest_vec[i], static_cast<double>(i + 1));
    }

    // Ensure the destination vector has the correct size
    EXPECT_EQ(dest_vec.GetSize(), src_vec.GetSize());
    stop_rocalution();
}

// Test for LocalVector::CopyFromDouble
TEST_F(local_vector_test, copy_from_double)
{
    disable_accelerator_rocalution(false);
    set_device_rocalution(0);
    init_rocalution();
    // Create a LocalVector<double> and allocate some size
    LocalVector<double> src_vec = getTestVector<double>(5);

    // Create a LocalVector<float> to copy the data into
    LocalVector<float> dest_vec;

    // Perform the copy
    EXPECT_NO_THROW(dest_vec.CopyFromDouble(src_vec));

    // Validate the copied values
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(dest_vec[i], static_cast<float>(i + 1));
    }

    // Ensure the destination vector has the correct size
    EXPECT_EQ(dest_vec.GetSize(), src_vec.GetSize());
    stop_rocalution();
}

// Test for LocalVector::Info
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_info, info)

// Test for LocalVector::Zeros
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_zeros, zeros)

// Test for LocalVector::Ones
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_ones, ones)

// Test for LocalVector::SetValue
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_set_values, set_value)

// Test for LocalVector::SetRandom
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_set_random_uniform,
                      set_random_uniform)

// Test for LocalVector::SetUniform
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_set_random_normal, set_random_normal)

// Test for LocalVector::Power
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_power, power)

// Test for LocalVector::Sort
GENERATE_TEST_CASES(local_vector_test, testing_sort, sort)

// Test for LocalVector::PermuteBackward
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_permute_backward, permute_backward)

// Test for LocalVector::InclusiveSum
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_inclusive_sum, inclusive_sum)

// Test for LocalVector::ScaleAddScale
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_scale_add_scale, scale_add_scale)

// Test for LocalVector::operator[] (const version)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_operator_index_const,
                      operator_index_const)

// Test for LocalVector::ReadFileASCII
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_read_file_ascii, read_file_ascii)

// Test for LocalVector::CopyFromAsync
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_copy_from_async, copy_from_async)

// Test for LocalVector::MoveToAcceleratorAsync
GENERATE_P_TEST_CASES_DEVICE(parameterized_local_vector_test,
                             testing_move_to_accelerator_async,
                             move_to_accelerator_async)

// Test for LocalVector::CopyFromHostData
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_copy_from_host_data,
                      copy_from_host_data)

// Test for LocalVector::CopyToData
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_copy_to_data, copy_to_data)

// Test for LocalVector::WriteFileASCII
GENERATE_TEST_CASES(local_vector_test, testing_write_file_ascii, write_file_ascii)

// Test for LocalVector::ScaleAddScale with offsets
GENERATE_TEST_CASES(local_vector_test,
                    testing_scale_add_scale_with_offsets,
                    scale_add_scale_with_offsets)

// Test for LocalVector::InclusiveSum(void)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_inclusive_sum_void,
                      inclusive_sum_void)

// Test for LocalVector::InclusiveSum(const LocalVector<ValueType>& vec)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_inclusive_sum_with_input,
                      inclusive_sum_with_input)

// Test for LocalVector::ExclusiveSum(void)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_exclusive_sum_void,
                      exclusive_sum_void)

// Test for LocalVector::ExclusiveSum(const LocalVector<ValueType>& vec)
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_exclusive_sum_with_input,
                      exclusive_sum_with_input)

// Test for LocalVector::Asum(void) const
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_asum, asum)

// Test for LocalVector::Amax(ValueType& value) const
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_amax, amax)

// Test for LocalVector::Prolongation
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_prolongation, prolongation)

// Test for LocalVector::GetIndexValues
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_get_index_values, get_index_values)

// Test for LocalVector::SetIndexValues
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_set_index_values, set_index_values)

// Test for LocalVector::AddIndexValues
GENERATE_P_TEST_CASES(parameterized_local_vector_test, testing_add_index_values, add_index_values)

// Test for LocalVector::GetContinuousValues
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_get_continuous_values,
                      get_continuous_values)

// Test for LocalVector::SetContinuousValues
GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_set_continuous_values,
                      set_continuous_values)

// Test for LocalVector::ExtractCoarseMapping
TEST_F(local_vector_test, extract_coarse_mapping_int)
{
    disable_accelerator_rocalution(false);
    set_device_rocalution(0);
    init_rocalution();
    testing_extract_coarse_mapping<int>();
    stop_rocalution();
}

GENERATE_P_TEST_CASES(parameterized_local_vector_test,
                      testing_move_to_host_async,
                      move_to_host_async)

INSTANTIATE_TEST_CASE_P(local_vector,
                        parameterized_local_vector_test,
                        testing::Combine(testing::ValuesIn(local_vector_size)));
