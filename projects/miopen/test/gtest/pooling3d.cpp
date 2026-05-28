// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "pooling_common.hpp"

namespace {

using PoolingTestCase = pooling_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling3dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    // Match ctest generate_data_limited(..., 4)
    std::vector<std::vector<int>> dataset0_inputs = {
        {16, 64, 3, 4, 4}, {16, 32, 4, 9, 9}, {8, 512, 3, 14, 14}, {8, 512, 4, 28, 28}};

    // Match ctest spatial parameters exactly
    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}, {3, 3, 3}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}, {1, 1, 1}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_pads    = {{0, 0, 0}, {1, 1, 1}};

    // Match ctest index types and modes
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};

    // Match ctest wsidx exactly (only 1 for 3D)
    std::vector<int> wsidx_values = {1};

    // Counters limit non-uint8 index type cases to 5 each (matching original ctest behavior)
    int num_uint16_case        = 0;
    int num_uint32_case        = 0;
    int num_uint32_case_imgidx = 0;
    int num_uint64_case        = 0;
    int num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset0_inputs)
    {
        pooling_gtest::AddTestCasesForInput(in_shape,
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
                                            true,
                                            false,
                                            "NCDHW");
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

template <typename T, typename Index>
void RunPooling3dTestWithIndexType(const PoolingTestCase& test_case)
{
    // Create input tensor (in_shape matches ctest)
    tensor<T> input{test_case.in_shape};
    input.generate(tensor_elem_gen_integer{
        (miopen_type<T>{} == miopenHalf || miopen_type<T>{} == miopenBFloat16) ? 5 : 17});

    // Apply NDHWC layout if requested
    if(test_case.in_layout != "NCDHW")
    {
        const std::vector<std::size_t> dim_lens = input.desc.GetLengths();
        std::vector<std::size_t> dim_strides;
        miopen::tensor_layout_to_strides(dim_lens,
                                         miopen::tensor_layout_get_default(input.desc.GetNumDims()),
                                         test_case.in_layout,
                                         dim_strides);
        input.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
    }

    // Setup pooling descriptor
    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    // Run forward pooling
    std::vector<Index> indices;
    pooling_gtest::verify_forward_pooling<3> forward_verifier;
    auto forward_result     = forward_verifier.cpu(input, filter, indices);
    auto forward_gpu_result = forward_verifier.gpu(input, filter, indices);

    // Compare forward results
    EXPECT_EQ(miopen::range_distance(forward_result), miopen::range_distance(forward_gpu_result));

    using value_type               = T;
    const double tolerance         = 80.0;
    const double threshold         = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double forward_rms_error = miopen::rms_range(forward_result, forward_gpu_result);

    EXPECT_LE(forward_rms_error, threshold)
        << "Forward RMS error: " << forward_rms_error << " exceeds threshold: " << threshold;

    // Run backward pooling
    auto dout = forward_result;
    dout.generate(tensor_elem_gen_integer{2503});

    // Validate indices are populated (required for max pooling backward)
    if(test_case.mode == miopenPoolingMax && indices.empty())
    {
        GTEST_FAIL() << "Indices not populated for max pooling backward";
    }

    pooling_gtest::verify_backward_pooling<3> backward_verifier;
    auto backward_result = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);

    // Compare backward results
    EXPECT_EQ(miopen::range_distance(backward_result), miopen::range_distance(backward_gpu_result));

    const double backward_rms_error = miopen::rms_range(backward_result, backward_gpu_result);

    EXPECT_LE(backward_rms_error, threshold)
        << "Backward RMS error: " << backward_rms_error << " exceeds threshold: " << threshold;
}

template <typename T>
void RunPooling3dTest(const PoolingTestCase& test_case)
{
    try
    {
        // Dispatch to the appropriate index type template
        switch(test_case.index_type)
        {
        case miopenIndexUint8: {
            RunPooling3dTestWithIndexType<T, uint8_t>(test_case);
            break;
        }
        case miopenIndexUint16: {
            RunPooling3dTestWithIndexType<T, uint16_t>(test_case);
            break;
        }
        case miopenIndexUint32: {
            RunPooling3dTestWithIndexType<T, uint32_t>(test_case);
            break;
        }
        case miopenIndexUint64: {
            RunPooling3dTestWithIndexType<T, uint64_t>(test_case);
            break;
        }
        default: {
            GTEST_FAIL() << "Unsupported index type: " << test_case.index_type;
            break;
        }
        }
    }
    catch(const std::exception& e)
    {
        std::string error_msg = e.what();
        // Skip test if no solver is found or hardware limits are exceeded
        if(error_msg.find("No solver found") != std::string::npos ||
           error_msg.find("exceeds the device limit") != std::string::npos)
        {
            GTEST_SKIP() << "Unsupported configuration: " << error_msg;
        }
        GTEST_FAIL() << "Exception thrown with test case: " << test_case << "\n"
                     << "Exception: " << error_msg;
    }
    catch(...)
    {
        GTEST_FAIL() << "Unknown exception thrown with test case: " << test_case;
    }
}

} // namespace

class GPU_Pooling3d_FP32 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

class GPU_Pooling3d_FP16 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

class GPU_Pooling3d_BFP16 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_Pooling3d_FP32, Test) { RunPooling3dTest<float>(GetParam()); }

TEST_P(GPU_Pooling3d_FP16, Test) { RunPooling3dTest<half_float::half>(GetParam()); }

TEST_P(GPU_Pooling3d_BFP16, Test) { RunPooling3dTest<bfloat16>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_FP32,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_FP16,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_BFP16,
                         testing::ValuesIn(GetPooling3dTestCases()),
                         pooling_gtest::GetPoolingTestCaseName);
