// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include <miopen/logger.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../pooling_common.hpp"

namespace {

struct Pooling2dTestCase
{
    std::vector<int> input_dims; // [N, C, H, W]
    std::vector<int> lens;       // [H, W]
    std::vector<int> pads;       // [H, W]
    std::vector<int> strides;    // [H, W]
    miopenIndexType_t index_type;
    miopenPoolingMode_t mode;
    int wsidx;

    friend std::ostream& operator<<(std::ostream& os, const Pooling2dTestCase& tc)
    {
        os << "input_dims: ";
        miopen::LogRange(os << "[", tc.input_dims, ",") << "] ";
        os << "lens: ";
        miopen::LogRange(os << "[", tc.lens, ",") << "] ";
        os << "pads: ";
        miopen::LogRange(os << "[", tc.pads, ",") << "] ";
        os << "strides: ";
        miopen::LogRange(os << "[", tc.strides, ",") << "] ";
        return os << "index_type: " << tc.index_type << ", mode: " << tc.mode
                  << ", wsidx: " << tc.wsidx;
    }
};

std::vector<Pooling2dTestCase> GetPooling2dTestCases()
{
    std::vector<Pooling2dTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    // Limited to 9 input shapes (matching generate_multi_data_limited with limit_multiplier=9)
    std::vector<std::vector<int>> dataset0_inputs = {
        {1, 19, 1024, 2048},
        {10, 3, 32, 32},
        {5, 32, 8, 8},
        {2, 1024, 12, 12},
        {4, 3, 231, 231},
        {8, 3, 227, 227},
        {1, 384, 13, 13},
        {1, 96, 27, 27},
        {2, 160, 7, 7}}; // First 9 from the original 18
    std::vector<std::vector<int>> dataset0_lens         = {{2, 2}, {3, 3}};
    std::vector<std::vector<int>> dataset0_strides      = {{2, 2}, {1, 1}};
    std::vector<std::vector<int>> dataset0_pads         = {{0, 0}, {1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 0
    for(const auto& input_dims : dataset0_inputs)
    {
        for(const auto& lens : dataset0_lens)
        {
            for(const auto& strides : dataset0_strides)
            {
                for(const auto& pads : dataset0_pads)
                {
                    for(const auto& index_type : dataset0_index_types)
                    {
                        for(const auto& mode : modes)
                        {
                            for(int wsidx : wsidx_values)
                            {
                                test_cases.push_back(
                                    {input_dims, lens, pads, strides, index_type, mode, wsidx});
                            }
                        }
                    }
                }
            }
        }
    }

    // Dataset 1: Minimal dataset (asymmetric configs, small tensors)
    std::vector<std::vector<int>> dataset1_inputs       = {{1, 4, 4, 4}};
    std::vector<std::vector<int>> dataset1_lens         = {{2, 2}, {1, 2}, {2, 1}};
    std::vector<std::vector<int>> dataset1_strides      = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
    std::vector<std::vector<int>> dataset1_pads         = {{0, 0}}; // WORKAROUND_ISSUE_1670
    std::vector<miopenIndexType_t> dataset1_index_types = {miopenIndexUint8, miopenIndexUint32};

    // Generate cartesian product for dataset 1
    for(const auto& input_dims : dataset1_inputs)
    {
        for(const auto& lens : dataset1_lens)
        {
            for(const auto& strides : dataset1_strides)
            {
                for(const auto& pads : dataset1_pads)
                {
                    for(const auto& index_type : dataset1_index_types)
                    {
                        for(const auto& mode : modes)
                        {
                            for(int wsidx : wsidx_values)
                            {
                                test_cases.push_back(
                                    {input_dims, lens, pads, strides, index_type, mode, wsidx});
                            }
                        }
                    }
                }
            }
        }
    }

    // Dataset 2: Wide window dataset
    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};
    std::vector<std::vector<int>> dataset2_lens    = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};
    std::vector<std::vector<int>> dataset2_pads    = {{0, 0}};
    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};

    // Generate cartesian product for dataset 2
    for(const auto& input_dims : dataset2_inputs)
    {
        for(const auto& lens : dataset2_lens)
        {
            for(const auto& strides : dataset2_strides)
            {
                for(const auto& pads : dataset2_pads)
                {
                    for(const auto& index_type : dataset2_index_types)
                    {
                        for(const auto& mode : modes)
                        {
                            for(int wsidx : wsidx_values)
                            {
                                test_cases.push_back(
                                    {input_dims, lens, pads, strides, index_type, mode, wsidx});
                            }
                        }
                    }
                }
            }
        }
    }

    return test_cases;
}

template <typename T, typename Index>
void RunPooling2dTestWithIndexType(const Pooling2dTestCase& test_case)
{
    // Create input tensor
    tensor<T> input{static_cast<size_t>(test_case.input_dims[0]),
                    static_cast<size_t>(test_case.input_dims[1]),
                    static_cast<size_t>(test_case.input_dims[2]),
                    static_cast<size_t>(test_case.input_dims[3])};
    input.generate(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17});

    // Setup pooling descriptor
    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    // Validate dimensions
    auto input_desc = miopen::TensorDescriptor(miopen_type<T>{}, test_case.input_dims);
    int spt_dim     = static_cast<int>(test_case.input_dims.size()) - 2;

    if(spt_dim != 2)
    {
        GTEST_SKIP() << "Only 2D pooling is supported (spt_dim == 2)";
    }

    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] >
           (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * test_case.pads[i]))
        {
            GTEST_SKIP() << "Invalid config: lens[" << i << "] > (input_dims[" << i + 2
                         << "] + 2 * pads[" << i << "])";
        }
    }

    // Skip configurations that would cause "Index range not enough" exception
    // This happens when the index type doesn't have enough range for max pooling backward
    if(test_case.mode == miopenPoolingMax)
    {
        // Calculate index_max based on index type from test_case
        size_t index_max = 0;
        switch(test_case.index_type)
        {
        case miopenIndexUint8:
            index_max = std::numeric_limits<uint8_t>::max();
            break;
        case miopenIndexUint16:
            index_max = std::numeric_limits<uint16_t>::max();
            break;
        case miopenIndexUint32:
            index_max = std::numeric_limits<uint32_t>::max();
            break;
        case miopenIndexUint64:
            index_max = std::numeric_limits<uint64_t>::max();
            break;
        default:
            index_max = SIZE_MAX; // Unknown type, assume it's large enough
            break;
        }

        // For max pooling backward, check if index range is sufficient
        if(test_case.wsidx == 0) // miopenPoolingWorkspaceIndexMask
        {
            // Check if index_max is sufficient for the pooling window
            size_t lens_product = 1;
            for(int len : test_case.lens)
            {
                lens_product *= static_cast<size_t>(len);
            }
            if(index_max <= lens_product)
            {
                int index_bits = 0;
                switch(test_case.index_type)
                {
                case miopenIndexUint8: index_bits = 8; break;
                case miopenIndexUint16: index_bits = 16; break;
                case miopenIndexUint32: index_bits = 32; break;
                case miopenIndexUint64: index_bits = 64; break;
                default: index_bits = 0; break;
                }
                GTEST_SKIP() << "Index range not enough: uint" << index_bits << " index_max ("
                             << index_max << ") <= lens product (" << lens_product
                             << ") for max pooling backward with workspace index mask mode";
            }
        }
        else // miopenPoolingWorkspaceIndexImage (wsidx == 1)
        {
            // Check if index_max is sufficient for output spatial dimensions
            auto output_tensor = get_output_tensor(filter, input);
            size_t output_spatial_product = 1;
            for(size_t i = 2; i < output_tensor.desc.GetLengths().size(); i++)
            {
                output_spatial_product *= static_cast<size_t>(output_tensor.desc.GetLengths()[i]);
            }
            if(index_max <= output_spatial_product)
            {
                int index_bits = 0;
                switch(test_case.index_type)
                {
                case miopenIndexUint8: index_bits = 8; break;
                case miopenIndexUint16: index_bits = 16; break;
                case miopenIndexUint32: index_bits = 32; break;
                case miopenIndexUint64: index_bits = 64; break;
                default: index_bits = 0; break;
                }
                GTEST_SKIP() << "Index range not enough: uint" << index_bits << " index_max ("
                             << index_max << ") <= output spatial product (" << output_spatial_product
                             << ") for max pooling backward with workspace index image mode";
            }
        }
    }

    // Run forward pooling
    std::vector<Index> indices;
    verify_forward_pooling<2> forward_verifier;
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

    verify_backward_pooling<2> backward_verifier;
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
void RunPooling2dTest(const Pooling2dTestCase& test_case)
{
    try
    {
        // Dispatch to the appropriate index type template
        switch(test_case.index_type)
        {
        case miopenIndexUint8: {
            RunPooling2dTestWithIndexType<T, uint8_t>(test_case);
            break;
        }
        case miopenIndexUint16: {
            RunPooling2dTestWithIndexType<T, uint16_t>(test_case);
            break;
        }
        case miopenIndexUint32: {
            RunPooling2dTestWithIndexType<T, uint32_t>(test_case);
            break;
        }
        case miopenIndexUint64: {
            RunPooling2dTestWithIndexType<T, uint64_t>(test_case);
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
        GTEST_FAIL() << "Exception thrown with test case: " << test_case << "\n"
                     << "Exception: " << e.what();
    }
    catch(...)
    {
        GTEST_FAIL() << "Unknown exception thrown with test case: " << test_case;
    }
}

} // namespace

class GPU_Pooling2d_FP32 : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

class GPU_Pooling2d_FP16 : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunPooling2dTest<float>(GetParam()); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunPooling2dTest<half_float::half>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         testing::PrintToStringParamName());
