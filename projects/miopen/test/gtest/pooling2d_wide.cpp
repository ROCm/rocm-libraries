// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

namespace {

using namespace pooling2d_gtest;

std::vector<Pooling2dTestCase> GetPooling2dWideTestCases()
{
    // Cache results to avoid duplicate generation when called multiple times
    static std::vector<Pooling2dTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<Pooling2dTestCase> test_cases;

    // Dataset 2: Wide window configurations
    // Input shapes matching ctest behavior with --dataset 2
    // From pooling2d.hpp: get_2d_pooling_input_shapes_wide()
    // NOTE: Even when TEST_GET_INPUT_TENSOR=1, dataset 2 uses the predefined wide window shapes
    //       because dataset_id is determined by lens/strides/pads selection, and dataset 2
    //       is specifically for wide window testing with these predefined shapes
    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};

    // Lens: {{35, 35}, {100, 100}, {255, 255}, {410, 400}} - wide window kernel sizes
    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};

    // Strides: {{1, 1}} - only stride 1 for wide windows
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};

    // Pads: {{0, 0}} - no padding for wide windows
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    // Dataset 2 uses only uint32 (matching ctest behavior)
    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 2
    // This matches the original ctest test_pooling2d behavior with --dataset 2
    // IMPORTANT: Order must match ctest exactly: index_type -> mode -> input_shape -> lens -> strides -> pads -> wsidx
    // This is the order in which test_driver processes test cases (based on add() call order)
    for(const auto& index_type : dataset2_index_types)
    {
        for(const auto& mode : modes)
        {
            for(const auto& input_dims : dataset2_inputs)
            {
                AddTestCasesForInput(input_dims,
                                     dataset2_lens,
                                     dataset2_strides,
                                     dataset2_pads,
                                     {index_type}, // Single index_type for this iteration
                                     {mode},       // Single mode for this iteration
                                     wsidx_values,
                                     test_cases,
                                     false,  // skip_wide_check=false for Dataset 2 (wide window)
                                     false,  // apply_index_type_limits=false for Dataset 2 (matching ctest)
                                     true);  // is_wide_dataset=true for Dataset 2 (wide window)
            }
        }
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

} // namespace

// Derived classes for Dataset 2 (wide window pooling)
using GPU_WidePooling2d_FP32 = Pooling2dCommon<float>;
using GPU_WidePooling2d_FP16 = Pooling2dCommon<half_float::half>;

TEST_P(GPU_WidePooling2d_FP32, FloatTest_pooling2d_wide) { this->RunTest(); }

TEST_P(GPU_WidePooling2d_FP16, HalfTest_pooling2d_wide) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_WidePooling2d_FP32,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         GetPooling2dTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_WidePooling2d_FP16,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         GetPooling2dTestCaseName);
