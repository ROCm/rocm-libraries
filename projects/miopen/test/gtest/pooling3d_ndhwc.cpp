// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <half/half.hpp>

#include <miopen/logger.hpp>
#include <miopen/tensor_layout.hpp>

#include "gtest_common.hpp"
#include "pooling_gtest_common.hpp"
#include "pooling2d_common.hpp"

namespace {

using PoolingTestCase = pooling2d_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling3dNDHWCTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    std::vector<std::vector<int>> dataset0_inputs = {
        {16, 64, 3, 4, 4}, {16, 32, 4, 9, 9}, {8, 512, 3, 14, 14}, {8, 512, 4, 28, 28}};

    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}, {3, 3, 3}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}, {1, 1, 1}, {1, 2, 2}};
    std::vector<std::vector<int>> dataset0_pads    = {{0, 0, 0}, {1, 1, 1}};

    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};

    std::vector<int> wsidx_values = {1};

    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;

    for(const auto& in_shape : dataset0_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(in_shape,
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
                                              false,
                                              true,
                                              false,
                                              "NDHWC",
                                              "NDHWC");
    }

    cached_test_cases = test_cases;
    cached            = true;
    return test_cases;
}

template <typename T, typename Index>
void RunPooling3dTestWithIndexType(const PoolingTestCase& test_case)
{
    tensor<T> input{test_case.in_shape};
    input.generate(tensor_elem_gen_integer{
        (miopen_type<T>{} == miopenHalf || miopen_type<T>{} == miopenBFloat16) ? 5 : 17});

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

    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    std::vector<Index> indices;
    verify_forward_pooling<3> forward_verifier;
    auto forward_result     = forward_verifier.cpu(input, filter, indices);
    auto forward_gpu_result = forward_verifier.gpu(input, filter, indices);

    EXPECT_EQ(miopen::range_distance(forward_result), miopen::range_distance(forward_gpu_result));

    using value_type               = T;
    const double tolerance         = 80.0;
    const double threshold         = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double forward_rms_error = miopen::rms_range(forward_result, forward_gpu_result);

    EXPECT_LE(forward_rms_error, threshold)
        << "Forward RMS error: " << forward_rms_error << " exceeds threshold: " << threshold;

    auto dout = forward_result;
    dout.generate(tensor_elem_gen_integer{2503});

    if(test_case.mode == miopenPoolingMax && indices.empty())
    {
        GTEST_FAIL() << "Indices not populated for max pooling backward";
    }

    verify_backward_pooling<3> backward_verifier;
    auto backward_result = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);

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
        switch(test_case.index_type)
        {
        case miopenIndexUint8: RunPooling3dTestWithIndexType<T, uint8_t>(test_case); break;
        case miopenIndexUint16: RunPooling3dTestWithIndexType<T, uint16_t>(test_case); break;
        case miopenIndexUint32: RunPooling3dTestWithIndexType<T, uint32_t>(test_case); break;
        case miopenIndexUint64: RunPooling3dTestWithIndexType<T, uint64_t>(test_case); break;
        default: GTEST_FAIL() << "Unsupported index type: " << test_case.index_type; break;
        }
    }
    catch(const std::exception& e)
    {
        std::string error_msg = e.what();
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

class GPU_Pooling3d_NDHWC_FP32 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};
class GPU_Pooling3d_NDHWC_FP16 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};
class GPU_Pooling3d_NDHWC_BFP16 : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_Pooling3d_NDHWC_FP32, FloatTest_pooling3d_ndhwc) { RunPooling3dTest<float>(GetParam()); }
TEST_P(GPU_Pooling3d_NDHWC_FP16, HalfTest_pooling3d_ndhwc)
{
    RunPooling3dTest<half_float::half>(GetParam());
}
TEST_P(GPU_Pooling3d_NDHWC_BFP16, BFloat16Test_pooling3d_ndhwc)
{
    RunPooling3dTest<bfloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP32,
                         testing::ValuesIn(GetPooling3dNDHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_FP16,
                         testing::ValuesIn(GetPooling3dNDHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Pooling3d_NDHWC_BFP16,
                         testing::ValuesIn(GetPooling3dNDHWCTestCases()),
                         pooling2d_gtest::GetPoolingTestCaseName);
