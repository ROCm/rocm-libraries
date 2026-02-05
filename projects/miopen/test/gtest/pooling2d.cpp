// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "../network_data.hpp"
#include "pooling2d_common.hpp"

// Configuration define matching the original ctest behavior
// These can be overridden at compile time via -D flags
// TEST_GET_INPUT_TENSOR: When 0, uses all 18 predefined input shapes (matching ctest with --all).
//                        When 1, uses get_inputs() function to generate input shapes from
//                        network_data.
#ifndef TEST_GET_INPUT_TENSOR
#define TEST_GET_INPUT_TENSOR 0
#endif

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    std::vector<std::vector<int>> dataset0_inputs;
#if TEST_GET_INPUT_TENSOR
    // When TEST_GET_INPUT_TENSOR = 1, use get_inputs() function (matching original ctest behavior)
    int batch_factor                      = 0; // Default batch factor matching original ctest
    std::set<std::vector<int>> in_dim_set = get_inputs<int>(batch_factor);
    dataset0_inputs.assign(in_dim_set.begin(), in_dim_set.end());
#else
    // When TEST_GET_INPUT_TENSOR = 0, use predefined shapes
    // Match ctest's generate_multi_data_limited(..., 9) by only using the first 9 shapes.
    dataset0_inputs = {{1, 19, 1024, 2048},  // Shape 1
                       {10, 3, 32, 32},      // Shape 2
                       {5, 32, 8, 8},        // Shape 3
                       {2, 1024, 12, 12},    // Shape 4
                       {4, 3, 231, 231},     // Shape 5
                       {8, 3, 227, 227},     // Shape 6
                       {1, 384, 13, 13},     // Shape 7
                       {1, 96, 27, 27},      // Shape 8
                       {2, 160, 7, 7}};      // Shape 9
#endif
    std::vector<std::vector<int>> dataset0_lens         = {{2, 2}, {3, 3}};
    std::vector<std::vector<int>> dataset0_strides      = {{2, 2}, {1, 1}};
    std::vector<std::vector<int>> dataset0_pads         = {{0, 0}, {1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 0
    // This matches the original ctest test_pooling2d behavior (default dataset, dataset_id=0)
    // Filter invalid combinations at generation time instead of skipping at runtime
    int num_uint16_case         = 0;
    int num_uint32_case         = 0;
    int num_uint32_case_imgidx  = 0;
    int num_uint64_case         = 0;
    int num_uint64_case_imgidx  = 0;

    for(const auto& input_dims : dataset0_inputs)
    {
        AddTestCasesForInput(input_dims,
                             dataset0_lens,
                             dataset0_strides,
                             dataset0_pads,
                             dataset0_index_types,
                             modes,
                             wsidx_values,
                             test_cases,
                             num_uint16_case,
                             num_uint32_case,
                             num_uint32_case_imgidx,
                             num_uint64_case,
                             num_uint64_case_imgidx,
                             false,  // skip_wide_check=false for Dataset 0
                             true);  // apply_index_type_limits=true for Dataset 0
    }

    // Note: Dataset 1 (asymmetric) and Dataset 2 (wide window) are tested separately
    // via pooling2d_asymmetric.cpp and pooling2d_wide.cpp to maintain the same
    // structure as the original ctest implementation.

    std::cerr << "\n=== Dataset 0 (Standard) Test Case Generation Summary ===\n";
    std::cerr << "Total test cases generated: " << test_cases.size() << "\n";
    std::cerr << "Input shapes: " << dataset0_inputs.size() << "\n";
    std::cerr << "===========================================================\n\n";

    // Log all test configurations to a file for comparison with ctest
    // Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx
    std::ofstream log_file("pooling2d_gtest_configs.txt");
    if(log_file.is_open())
    {
        log_file << "# Total test cases: " << test_cases.size() << "\n";
        log_file << "# Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx\n";
        for(const auto& tc : test_cases)
        {
            log_file << tc.input_dims[0] << " " << tc.input_dims[1] << " " << tc.input_dims[2]
                     << " " << tc.input_dims[3] << " ";
            log_file << tc.lens[0] << " " << tc.lens[1] << " ";
            log_file << tc.pads[0] << " " << tc.pads[1] << " ";
            log_file << tc.strides[0] << " " << tc.strides[1] << " ";
            log_file << static_cast<int>(tc.index_type) << " " << static_cast<int>(tc.mode) << " "
                     << tc.wsidx << "\n";
        }
        log_file.close();
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 0 (standard pooling)
class GPU_Pooling2d_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};

class GPU_Pooling2d_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunTest(); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
