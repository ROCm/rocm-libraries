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

#pragma once
#ifndef TESTING_LOCAL_VECTOR_HPP
#define TESTING_LOCAL_VECTOR_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_vector_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // LocalVector object
    LocalVector<T> vec;

    // SetDataPtr
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != (NULL|__null)*");
        ASSERT_DEATH(vec.SetDataPtr(&null_ptr, "", safe_size),
                     ".*Assertion.*ptr != (NULL|__null)*");
    }

    // LeaveDataPtr
    {
        T* vdata = nullptr;
        allocate_host(safe_size, &vdata);
        ASSERT_DEATH(vec.LeaveDataPtr(&vdata), ".*Assertion.*ptr == (NULL|__null)*");
        free_host(&vdata);
    }

    // CopyFromData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyFromData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // CopyToData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyToData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // GetContinuousValues
    {
        vec.Allocate("", safe_size);
        T* null_T = nullptr;
        ASSERT_DEATH(vec.GetContinuousValues(0, safe_size, null_T),
                     ".*Assertion.*values != (NULL|__null)*");
    }

    // ExtractCoarseMapping
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*map != (NULL|__null)*");
        free_host(&vint);
    }

    // ExtractCoarseBoundary
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*boundary != (NULL|__null)*");
        free_host(&vint);
    }

    // Stop rocALUTION
    stop_rocalution();
}

template <typename T>
LocalVector<T> getTestVector(int size = 10)
{
    // Create a LocalVector
    LocalVector<T> vec;
    vec.Allocate("TestVector", size);

    // Fill the source vector with some values
    for(int i = 0; i < size; ++i)
    {
        vec[i] = static_cast<T>(i + 1);
    }

    return vec;
}

template <typename T>
void testing_get_interior(const Arguments& argus)
{
    // Get a LocalVector with size from argus
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Call GetInterior
    LocalVector<T>& interior = vec.GetInterior();

    // Validate that the returned reference is the same as the original object
    EXPECT_EQ(&interior, &vec);

    // Validate that the size and properties of the vector remain unchanged
    EXPECT_EQ(interior.GetSize(), vec.GetSize());

    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], interior[i]);
    }
}

template <typename T>
void testing_const_get_interior(const Arguments& argus)
{
    // Get a LocalVector with size from argus
    const LocalVector<T> vec = getTestVector<T>(argus.size);

    // Call GetInterior
    const LocalVector<T>& interior = vec.GetInterior();

    // Validate that the returned reference is the same as the original object
    EXPECT_EQ(&interior, &vec);

    // Validate that the size and properties of the vector remain unchanged
    EXPECT_EQ(interior.GetSize(), vec.GetSize());

    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], interior[i]);
    }
}

template <typename T>
void testing_check(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Check the vector
    EXPECT_NO_THROW(vec.Check());
}

template <typename T>
void testing_zeros(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    vec.Zeros();
    // Validate that all elements are set to zero
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(0.0));
    }
}

template <typename T>
void testing_ones(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    vec.Ones();
    // Validate that all elements are set to one
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(1.0));
    }
}

template <typename T>
void testing_set_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Set values in the vector
    T value = static_cast<T>(5.0);
    vec.SetValues(value);

    // Validate that all elements are set to the specified value
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], value);
    }
}

template <typename T>
void testing_set_random_uniform(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Set random values in the vector
    T min = static_cast<T>(0.0);
    T max = static_cast<T>(1.0);
    vec.SetRandomUniform(min, max);

    // Validate that all elements are within the specified range
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_GE(vec[i], min);
        EXPECT_LE(vec[i], max);
    }
}

template <typename T>
void testing_set_random_normal(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size > 0 ? argus.size : 1000);

    // Set random values in the vector
    T mean = static_cast<T>(0.0);
    T std  = static_cast<T>(1.0);
    vec.SetRandomNormal(mean, std);

    // Validate that the mean of all elements is close to the specified mean
    T sum = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        sum += vec[i];
    }
    T calculated_mean = sum / vec.GetSize();
    EXPECT_NEAR(calculated_mean, mean, 3 * std);

    // Validate that the standard deviation of all elements is close to the specified std
    T sum_sq = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        sum_sq += (vec[i] - calculated_mean) * (vec[i] - calculated_mean);
    }
    T calculated_std = std::sqrt(sum_sq / vec.GetSize());
    EXPECT_NEAR(calculated_std, std, 3 * std);
}

template <typename T>
void testing_copy_from(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create another LocalVector to copy from
    LocalVector<T> src_vec = getTestVector<T>(argus.size);

    // Copy values from the source vector
    EXPECT_NO_THROW(vec.CopyFrom(src_vec));

    // Validate the copied values
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], src_vec[i]);
    }
}

template <typename T>
void testing_clone_from(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Clone the vector
    LocalVector<T> cloned_vec;
    EXPECT_NO_THROW(cloned_vec.CloneFrom(vec));

    // Validate the cloned values
    for(int i = 0; i < cloned_vec.GetSize(); ++i)
    {
        EXPECT_EQ(cloned_vec[i], vec[i]);
    }
}

template <typename T>
void testing_binary(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Define the filename for the test
    const std::string filename = get_temp_dir() + "test_vector.bin";

    // Write the vector to a binary file
    EXPECT_NO_THROW(vec.WriteFileBinary(filename));

    // Read the vector from the binary file
    LocalVector<T> read_vec;
    EXPECT_NO_THROW(read_vec.ReadFileBinary(filename));

    // Validate the read values
    for(int i = 0; i < read_vec.GetSize(); ++i)
    {
        EXPECT_EQ(read_vec[i], vec[i]);
    }

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_add_scale(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> v = getTestVector<T>(argus.size);
    LocalVector<T> x = getTestVector<T>(argus.size);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // Perform the AddScale operation
    EXPECT_NO_THROW(v.AddScale(x, alpha));

    // Validate the result
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], static_cast<T>((i + 1) * (1 + alpha)));
    }
}
template <typename T>
void testing_scale_add(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> v = getTestVector<T>(argus.size);
    LocalVector<T> x = getTestVector<T>(argus.size);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // v = 1 initially
    v.Ones();

    // Perform the v = alpha * v + x
    EXPECT_NO_THROW(v.ScaleAdd(alpha, x));

    // Validate the result
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], alpha + x[i]);
    }
}

template <typename T>
void testing_scale_add_2(const Arguments& argus)
{
    LocalVector<T> v = getTestVector<T>(argus.size);
    LocalVector<T> x = getTestVector<T>(argus.size);
    LocalVector<T> y = getTestVector<T>(argus.size);

    // Fill x and y with test values
    if(argus.size >= 3)
    {
        x[0] = static_cast<T>(1);
        x[1] = static_cast<T>(2);
        x[2] = static_cast<T>(3);
        y[0] = static_cast<T>(4);
        y[1] = static_cast<T>(5);
        y[2] = static_cast<T>(6);
    }

    // v = 1 initially
    v.Ones();

    // v.ScaleAdd2(alpha, x, beta, y, gamma)
    T alpha = static_cast<T>(2);
    T beta  = static_cast<T>(3);
    T gamma = static_cast<T>(3);
    v.ScaleAdd2(alpha, x, beta, y, gamma);

    // v[i] = alpha * v[i] + beta * x[i] + gamma * y[i]
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], alpha + beta * x[i] + gamma * y[i]);
    }
}

template <typename T>
void testing_scale(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // Perform the Scale operation
    EXPECT_NO_THROW(vec.Scale(alpha));

    // Validate the result
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>((i + 1) * alpha));
    }
}

template <typename T>
void testing_dot(const Arguments& argus)
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(argus.size);
    LocalVector<T> vec2 = getTestVector<T>(argus.size);

    // Perform the Dot product
    T result = 0.0;
    EXPECT_NO_THROW(result = vec1.Dot(vec2));

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        result2 += vec1[i] * vec2[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_dot_non_conj(const Arguments& argus)
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(argus.size);
    LocalVector<T> vec2 = getTestVector<T>(argus.size);

    // Perform the Dot product without conjugation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec1.DotNonConj(vec2));

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        result2 += vec1[i] * vec2[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_norm(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Perform the Norm operation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec.Norm());

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        result2 += vec[i] * vec[i];
    }
    EXPECT_NEAR(result * result, result2, 1e-6 * std::abs(result2));
}

template <typename T>
void testing_reduce(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Perform the Reduce operation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec.Reduce());

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        result2 += vec[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_point_wise_mult(const Arguments& argus)
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(argus.size);
    LocalVector<T> vec2 = getTestVector<T>(argus.size);

    // Perform the PointwiseMult operation
    EXPECT_NO_THROW(vec1.PointWiseMult(vec2));

    // Validate the result
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        EXPECT_EQ(vec1[i], static_cast<T>((i + 1) * (i + 1)));
    }
}

template <typename T>
void testing_point_wise_mult_2(const Arguments& argus)
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(argus.size);
    LocalVector<T> vec2 = getTestVector<T>(argus.size);
    LocalVector<T> vec3 = getTestVector<T>(argus.size);

    // Perform the PointwiseMult operation
    EXPECT_NO_THROW(vec1.PointWiseMult(vec2, vec3));

    // Validate the result
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        EXPECT_EQ(vec1[i], static_cast<T>((i + 1) * (i + 1)));
    }
}

template <typename T>
void testing_leave_data_ptr(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Pointer to data (initialized to NULL)
    T* data_ptr = nullptr;

    // Attempt to leave the data pointer
    EXPECT_NO_THROW(vec.LeaveDataPtr(&data_ptr));

    // Ensure the pointer is not NULL after the call
    EXPECT_NE(data_ptr, nullptr);

    // Ensure the vector is cleared
    EXPECT_EQ(vec.GetSize(), 0);
}

template <typename T>
void testing_sync(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Perform a synchronization operation
    EXPECT_NO_THROW(vec.Sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), argus.size);
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_copy_from_data(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Pointer to data
    T data_ptr[5] = {0, 2, 4, 8, 16};

    // Attempt to copy from the data pointer
    EXPECT_NO_THROW(vec.CopyFromData(data_ptr));

    EXPECT_EQ(vec.GetSize(), argus.size);
    if(argus.size >= 5)
    {
        EXPECT_EQ(vec[0], static_cast<T>(0.0));
        EXPECT_EQ(vec[1], static_cast<T>(2.0));
        EXPECT_EQ(vec[2], static_cast<T>(4.0));
        EXPECT_EQ(vec[3], static_cast<T>(8.0));
        EXPECT_EQ(vec[4], static_cast<T>(16.0));
    }
}

template <typename T>
void testing_copy_to_host_data(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Pointer to host data
    std::vector<T> host_data(argus.size);

    // Attempt to copy to the host data pointer
    EXPECT_NO_THROW(vec.CopyToHostData(host_data.data()));

    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(host_data[i], static_cast<T>(i + 1));
    }
}
template <typename T>
void testing_restriction()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create a mapping vector (map fine indices to coarse indices)
    LocalVector<int> map;
    map.Allocate("MappingVector", 5);
    map[0] = 0;
    map[1] = 0;
    map[2] = 1;
    map[3] = 1;
    map[4] = 2;

    // Create a LocalVector to store the restriction
    LocalVector<T> restricted_vec = getTestVector<T>(5);

    // Perform the Restriction operation
    EXPECT_NO_THROW(restricted_vec.Restriction(vec, map));

    // Validate the result size (should match the number of unique coarse indices, here 3)
    EXPECT_EQ(restricted_vec.GetSize(), 5);

    // Optionally, check the restricted values
    EXPECT_EQ(restricted_vec[0], static_cast<T>(vec[0] + vec[1])); // sum of fine indices 0 and 1
    EXPECT_EQ(restricted_vec[1], static_cast<T>(vec[2] + vec[3])); // sum of fine indices 2 and 3
    EXPECT_EQ(restricted_vec[2], static_cast<T>(vec[4])); // fine index 4
}

template <typename T>
void testing_copy_from_with_offsets(const Arguments& argus)
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> src_vec = getTestVector<T>(argus.size);

    // Create a destination LocalVector and allocate some size
    LocalVector<T> dst_vec;
    dst_vec.Allocate("DstVector", argus.size);

    // Fill destination with zeros
    dst_vec.Zeros();

    // Copy 3 elements from src_vec[1..3] to dst_vec[2..4]
    int64_t src_offset = 1;
    int64_t dst_offset = 2;
    int64_t size       = 3;

    if(argus.size >= 5)
    {
        EXPECT_NO_THROW(dst_vec.CopyFrom(src_vec, src_offset, dst_offset, size));

        // Validate the copied values
        // dst_vec should now be [0, 0, 2, 3, 4]
        EXPECT_EQ(dst_vec[0], static_cast<T>(0));
        EXPECT_EQ(dst_vec[1], static_cast<T>(0));
        EXPECT_EQ(dst_vec[2], static_cast<T>(2));
        EXPECT_EQ(dst_vec[3], static_cast<T>(3));
        EXPECT_EQ(dst_vec[4], static_cast<T>(4));
    }
}

template <typename T>
void testing_set_data_ptr_null(const Arguments& argus)
{
    // Create a LocalVector
    LocalVector<T> vec;

    // Pointer to data (initialized to NULL)
    T* data_ptr = NULL;

    // Attempt to set the data pointer
    EXPECT_NO_THROW(vec.SetDataPtr(&data_ptr, "TestVector", 0));

    // Ensure the pointer is still NULL after the call
    EXPECT_EQ(data_ptr, nullptr);

    // Ensure the vector is cleared
    EXPECT_EQ(vec.GetSize(), 0);
}

template <typename T>
void testing_info(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", argus.size);

    // Redirect standard output to a stringstream to capture the Info output
    std::stringstream output_stream;
    std::streambuf*   original_cout_buffer = std::cout.rdbuf();
    std::cout.rdbuf(output_stream.rdbuf());

    // Call the Info method
    vec.Info();

    // Restore the original standard output buffer
    std::cout.rdbuf(original_cout_buffer);

    // Get the captured output
    std::string output = output_stream.str();

    // Validate the output contains expected information
    EXPECT_NE(output.find("name=TestVector"), std::string::npos); // Check vector name
    EXPECT_NE(output.find("size=" + std::to_string(argus.size)),
              std::string::npos); // Check vector size
    EXPECT_NE(output.find("prec=" + std::to_string(sizeof(T) * 8) + "bit"),
              std::string::npos); // Check data type
}

template <typename T>
void testing_sort()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    LocalVector<T> sorted_vec;
    sorted_vec.Allocate("TestVector", 5);

    // Fill the vector with unsorted values
    vec[0] = static_cast<T>(3.0);
    vec[1] = static_cast<T>(1.0);
    vec[2] = static_cast<T>(5.0);
    vec[3] = static_cast<T>(2.0);
    vec[4] = static_cast<T>(4.0);

    // Call the Sort method
    EXPECT_NO_THROW(vec.Sort(&sorted_vec, nullptr));

    // Validate that the vector is sorted in ascending order
    EXPECT_EQ(sorted_vec[0], static_cast<T>(1.0));
    EXPECT_EQ(sorted_vec[1], static_cast<T>(2.0));
    EXPECT_EQ(sorted_vec[2], static_cast<T>(3.0));
    EXPECT_EQ(sorted_vec[3], static_cast<T>(4.0));
    EXPECT_EQ(sorted_vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_permute_backward(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create a permutation vector
    LocalVector<int> permutation;
    permutation.Allocate("PermutationVector", argus.size);

    if(argus.size >= 5)
    {
        permutation[0] = 4; // Move element at index 4 to index 0
        permutation[1] = 2; // Move element at index 2 to index 1
        permutation[2] = 0; // Move element at index 0 to index 2
        permutation[3] = 3; // Keep element at index 3 in place
        permutation[4] = 1; // Move element at index 1 to index 4

        // Call the PermuteBackward method
        EXPECT_NO_THROW(vec.PermuteBackward(permutation));

        // Validate the permuted vector
        EXPECT_EQ(vec[0], static_cast<T>(5.0)); // Element at index 4
        EXPECT_EQ(vec[1], static_cast<T>(3.0)); // Element at index 2
        EXPECT_EQ(vec[2], static_cast<T>(1.0)); // Element at index 0
        EXPECT_EQ(vec[3], static_cast<T>(4.0)); // Element at index 3
        EXPECT_EQ(vec[4], static_cast<T>(2.0)); // Element at index 1
    }
}

template <typename T>
void testing_inclusive_sum(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> input_vec;
    input_vec.Allocate("InputVector", argus.size);

    // Fill the input vector with values
    for(int i = 0; i < argus.size; ++i)
    {
        input_vec[i] = static_cast<T>(i + 1);
    }

    // Create a LocalVector to store the result
    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", argus.size);

    // Call the InclusiveSum method
    EXPECT_NO_THROW(result_vec.InclusiveSum(input_vec));

    // Validate the result
    T sum = 0;
    for(int i = 0; i < argus.size; ++i)
    {
        sum += static_cast<T>(i + 1);
        EXPECT_EQ(result_vec[i], sum);
    }
}

template <typename T>
void testing_power(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Apply the Power method with power = 2.0
    EXPECT_NO_THROW(vec.Power(2.0));

    // Validate the results
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>((i + 1) * (i + 1))); // Squared values
    }

    // Apply the Power method with power = 0.5 (square root)
    EXPECT_NO_THROW(vec.Power(0.5));

    // Validate the results
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1)); // Original values restored
    }
}

template <typename T>
void testing_scale_add_scale(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create another LocalVector for the operation
    LocalVector<T> x;
    x.Allocate("XVector", argus.size);

    // Fill the second vector with values
    for(int i = 0; i < argus.size; ++i)
    {
        x[i] = static_cast<T>(argus.size - i);
    }

    // Define scaling factors
    T alpha = 2.0f;
    T beta  = 3.0f;

    // Perform the ScaleAddScale operation
    EXPECT_NO_THROW(vec.ScaleAddScale(alpha, x, beta));

    // Validate the result
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(alpha * (i + 1) + beta * (argus.size - i)));
    }
}

template <typename T>
void testing_operator_index_const(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create a const reference to the vector
    const LocalVector<T>& const_vec = vec;

    // Validate access to elements using the const operator[]
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(const_vec[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_read_file_ascii(const Arguments& argus)
{
    // Create a temporary ASCII file with test data
    std::string   filename = get_temp_dir() + "test_vector.txt";
    std::ofstream file(filename);
    ASSERT_TRUE(file.is_open());
    for(int i = 0; i < argus.size; ++i)
    {
        file << static_cast<double>(i + 1) << "\n";
    }
    file.close();

    // Create a LocalVector to read the data into
    LocalVector<T> vec;

    // Call the ReadFileASCII method
    EXPECT_NO_THROW(vec.ReadFileASCII(filename));

    // Validate the vector size
    EXPECT_EQ(vec.GetSize(), argus.size);

    // Validate the values in the vector
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_copy_from_async(const Arguments& argus)
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> src_vec = getTestVector<T>(argus.size);

    // Create a destination LocalVector
    LocalVector<T> dest_vec;

    // Perform the asynchronous copy
    EXPECT_NO_THROW(dest_vec.CopyFromAsync(src_vec));

    // Synchronize to ensure the copy is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate the copied values
    EXPECT_EQ(dest_vec.GetSize(), src_vec.GetSize());
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(dest_vec[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_move_to_accelerator_async(const Arguments& argus)
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Perform the asynchronous move to the accelerator
    EXPECT_NO_THROW(vec.MoveToAcceleratorAsync());

    // Synchronize to ensure the move is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), argus.size);
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_copy_from_host_data(const Arguments& argus)
{
    // clang-format off
    // Create a host array with test data
    std::vector<T> host_data(argus.size);
    for(int i = 0; i < argus.size; ++i)
    {
        host_data[i] = static_cast<T>(i + 1);
    }
    // clang-format on

    // Create a LocalVector and allocate the same size
    LocalVector<T> vec;
    vec.Allocate("TestVector", argus.size);

    // Copy data from the host array to the LocalVector
    EXPECT_NO_THROW(vec.CopyFromHostData(host_data.data()));

    // Validate the vector size
    EXPECT_EQ(vec.GetSize(), argus.size);

    // Validate the copied values
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_copy_to_data(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create a host array to store the copied data
    std::vector<T> host_data(argus.size, static_cast<T>(0.0));

    // Copy data from the LocalVector to the host array
    EXPECT_NO_THROW(vec.CopyToData(host_data.data()));

    // Validate the copied values
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(host_data[i], static_cast<T>(i + 1));
    }
}

// Helper function to read the contents of a file into a string
std::string ReadFileContents(const std::string& filename)
{
    std::ifstream     file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

template <typename T>
void testing_write_file_ascii()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Define the filename for the test
    const std::string filename = get_temp_dir() + "test_vector.txt";

    // Write the vector to the file
    EXPECT_NO_THROW(vec.WriteFileASCII(filename));

    // Read the contents of the file
    std::string file_contents = ReadFileContents(filename);

    // Define the expected output
    std::string expected_output
        = "1.000000e+00\n2.000000e+00\n3.000000e+00\n4.000000e+00\n5.000000e+00\n";

    // Validate the file contents
    EXPECT_EQ(file_contents, expected_output);

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_scale_add_scale_with_offsets()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(10);

    // Create another LocalVector for the operation
    LocalVector<T> x;
    x.Allocate("XVector", 10);

    // Fill the second vector with values
    for(int i = 0; i < 10; ++i)
    {
        x[i] = static_cast<T>(10 - i); // Values: 10.0, 9.0, ..., 1.0
    }

    // Define scaling factors and offsets
    T       alpha      = 2.0f;
    T       beta       = 3.0f;
    int64_t src_offset = 2; // Start from index 2 in x
    int64_t dst_offset = 4; // Start from index 4 in vec
    int64_t size       = 3; // Operate on 3 elements

    // Perform the ScaleAddScale operation
    EXPECT_NO_THROW(vec.ScaleAddScale(alpha, x, beta, src_offset, dst_offset, size));

    // Validate the results
    EXPECT_EQ(vec[4], static_cast<T>(alpha * 5.0 + beta * 8.0)); // 2 * 5 + 3 * 8 = 34
    EXPECT_EQ(vec[5], static_cast<T>(alpha * 6.0 + beta * 7.0)); // 2 * 6 + 3 * 7 = 33
    EXPECT_EQ(vec[6], static_cast<T>(alpha * 7.0 + beta * 6.0)); // 2 * 7 + 3 * 6 = 32

    // Ensure other elements remain unchanged
    EXPECT_EQ(vec[0], static_cast<T>(1.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(3.0));
    EXPECT_EQ(vec[3], static_cast<T>(4.0));
    EXPECT_EQ(vec[7], static_cast<T>(8.0));
    EXPECT_EQ(vec[8], static_cast<T>(9.0));
    EXPECT_EQ(vec[9], static_cast<T>(10.0));
}

template <typename T>
void testing_inclusive_sum_void(const Arguments& argus)
{
    LocalVector<T> vec = getTestVector<T>(argus.size);

    T expected_sum = 0;
    for(int i = 0; i < argus.size; ++i)
    {
        expected_sum += static_cast<T>(i + 1);
    }
    EXPECT_EQ(vec.InclusiveSum(), expected_sum);
}

template <typename T>
void testing_inclusive_sum_with_input(const Arguments& argus)
{
    LocalVector<T> input_vec = getTestVector<T>(argus.size);

    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", argus.size);

    EXPECT_NO_THROW(result_vec.InclusiveSum(input_vec));

    T sum = 0;
    for(int i = 0; i < argus.size; ++i)
    {
        sum += static_cast<T>(i + 1);
        EXPECT_EQ(result_vec[i], sum);
    }
}

template <typename T>
void testing_exclusive_sum_void(const Arguments& argus)
{
    LocalVector<T> vec = getTestVector<T>(argus.size);

    T expected_sum = 0;
    for(int i = 0; i < argus.size - 1; ++i)
    {
        expected_sum += static_cast<T>(i + 1);
    }
    EXPECT_EQ(vec.ExclusiveSum(), expected_sum);
}

template <typename T>
void testing_exclusive_sum_with_input(const Arguments& argus)
{
    LocalVector<T> input_vec = getTestVector<T>(argus.size);

    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", argus.size);

    EXPECT_NO_THROW(result_vec.ExclusiveSum(input_vec));

    T sum = 0;
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(result_vec[i], sum);
        sum += static_cast<T>(i + 1);
    }
}

template <typename T>
void testing_asum(const Arguments& argus)
{
    LocalVector<T> vec = getTestVector<T>(argus.size);

    T expected_sum = 0;
    for(int i = 0; i < argus.size; ++i)
    {
        expected_sum += std::abs(static_cast<T>(i + 1));
    }
    EXPECT_EQ(vec.Asum(), expected_sum);
}

template <typename T>
void testing_amax(const Arguments& argus)
{
    LocalVector<T> vec = getTestVector<T>(argus.size);

    T   max_value = 0.0;
    int max_index = vec.Amax(max_value);

    EXPECT_EQ(max_index, argus.size - 1);
    EXPECT_EQ(max_value, static_cast<T>(argus.size));
}

// Helper function to read binary file contents into a vector
template <typename ValueType>
std::vector<ValueType> ReadBinaryFile(const std::string& filename, size_t& size_out)
{
    std::ifstream file(filename, std::ios::binary);

    // Read the size of the vector
    size_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    size_out = size;

    // Read the vector data
    std::vector<ValueType> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(ValueType));
    file.close();

    return data;
}

template <typename T>
void testing_prolongation(const Arguments& argus)
{
    // Create a coarse-level LocalVector and allocate some size
    int coarse_size = argus.size > 2 ? argus.size / 2 : 2;
    int fine_size   = coarse_size * 2;

    LocalVector<T> vec_coarse;
    vec_coarse.Allocate("CoarseVector", coarse_size);

    // Fill the coarse-level vector with values
    for(int i = 0; i < coarse_size; ++i)
    {
        vec_coarse[i] = static_cast<T>(i + 1);
    }

    // Create a mapping vector
    LocalVector<int> map;
    map.Allocate("MappingVector", fine_size);

    // Define the mapping (fine indices map to coarse indices)
    for(int i = 0; i < fine_size; ++i)
    {
        map[i] = i / 2;
    }

    // Create a fine-level LocalVector to store the result
    LocalVector<T> vec_fine;
    vec_fine.Allocate("FineVector", fine_size);

    // Perform the prolongation operation
    EXPECT_NO_THROW(vec_fine.Prolongation(vec_coarse, map));

    // Validate the prolonged values
    for(int i = 0; i < fine_size; ++i)
    {
        EXPECT_EQ(vec_fine[i], vec_coarse[map[i]]);
    }
}

template <typename T>
void testing_get_index_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Create an index vector
    int              num_indices = std::min(3, argus.size);
    LocalVector<int> index;
    index.Allocate("IndexVector", num_indices);
    for(int i = 0; i < num_indices; ++i)
    {
        index[i] = i * 2;
    }

    // Create a LocalVector to store the values
    LocalVector<T> values;
    values.Allocate("ValuesVector", num_indices);

    // Call the GetIndexValues method
    EXPECT_NO_THROW(vec.GetIndexValues(index, &values));

    // Validate the retrieved values
    for(int i = 0; i < num_indices; ++i)
    {
        EXPECT_EQ(values[i], static_cast<T>(index[i] + 1));
    }
}

template <typename T>
void testing_set_index_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", argus.size);

    // Create an index vector
    int              num_indices = std::min(3, argus.size);
    LocalVector<int> index;
    index.Allocate("IndexVector", num_indices);
    for(int i = 0; i < num_indices; ++i)
    {
        index[i] = i * 2;
    }

    // Create a LocalVector to store the values
    LocalVector<T> values;
    values.Allocate("ValuesVector", num_indices);
    for(int i = 0; i < num_indices; ++i)
    {
        values[i] = static_cast<T>(100.0 * (i + 1));
    }

    // Call the SetIndexValues method
    EXPECT_NO_THROW(vec.SetIndexValues(index, values));

    // Validate the set values
    for(int i = 0; i < num_indices; ++i)
    {
        EXPECT_EQ(vec[index[i]], static_cast<T>(100.0 * (i + 1)));
    }
}

template <typename T>
void testing_add_index_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", argus.size);

    // Fill the vector with initial values
    for(int i = 0; i < argus.size; ++i)
    {
        vec[i] = static_cast<T>(10.0 * (i + 1));
    }

    // Create an index vector
    int              num_indices = std::min(3, argus.size);
    LocalVector<int> index;
    index.Allocate("IndexVector", num_indices);
    for(int i = 0; i < num_indices; ++i)
    {
        index[i] = i * 2;
    }

    // Create a LocalVector to store the values to be added
    LocalVector<T> values;
    values.Allocate("ValuesVector", num_indices);
    for(int i = 0; i < num_indices; ++i)
    {
        values[i] = static_cast<T>(5.0 * (i + 1));
    }

    // Call the AddIndexValues method
    EXPECT_NO_THROW(vec.AddIndexValues(index, values));

    // Validate the updated values
    for(int i = 0; i < num_indices; ++i)
    {
        EXPECT_EQ(vec[index[i]], static_cast<T>(10.0 * (index[i] + 1) + 5.0 * (i + 1)));
    }
}

template <typename T>
void testing_get_continuous_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    int num_values = std::min(3, argus.size - 1);
    if(num_values > 0)
    {
        // Create an array to store the continuous values
        std::vector<T> values(num_values);

        // Call the GetContinuousValues method
        EXPECT_NO_THROW(vec.GetContinuousValues(1, 1 + num_values, values.data()));

        // Validate the retrieved values
        for(int i = 0; i < num_values; ++i)
        {
            EXPECT_EQ(values[i], static_cast<T>(i + 2));
        }
    }
}

template <typename T>
void testing_set_continuous_values(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", argus.size);

    int num_values = std::min(3, argus.size - 1);
    if(num_values > 0)
    {
        // Create an array of values to set
        std::vector<T> values(num_values);
        for(int i = 0; i < num_values; ++i)
        {
            values[i] = static_cast<T>(100.0 * (i + 1));
        }

        // Call the SetContinuousValues method
        EXPECT_NO_THROW(vec.SetContinuousValues(1, 1 + num_values, values.data()));

        // Validate the set values
        for(int i = 0; i < num_values; ++i)
        {
            EXPECT_EQ(vec[1 + i], static_cast<T>(100.0 * (i + 1)));
        }
    }
}

template <typename T>
void testing_extract_coarse_mapping()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Create an index vector
    int index[3] = {0, 2, 4};
    int size     = 0;
    int map[3];

    // Call the ExtractCoarseMapping method
    EXPECT_NO_THROW(vec.ExtractCoarseMapping(0, 5, index, 3, &size, map));

    // Validate the extracted mapping
    EXPECT_EQ(size, 5);
    EXPECT_EQ(map[0], 0);
}

template <typename T>
void testing_move_to_host_async(const Arguments& argus)
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(argus.size);

    // Move the vector to the accelerator
    EXPECT_NO_THROW(vec.MoveToAccelerator());

    // Perform the asynchronous move to the host
    EXPECT_NO_THROW(vec.MoveToHostAsync());

    // Synchronize to ensure the move is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), argus.size);
    for(int i = 0; i < argus.size; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }
}

#endif // TESTING_LOCAL_VECTOR_HPP
