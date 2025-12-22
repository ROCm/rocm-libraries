// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include <miopen/logger.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../network_data.hpp"
#include "../pooling_common.hpp"

// Configuration define matching the original ctest behavior
// These can be overridden at compile time via -D flags
// TEST_GET_INPUT_TENSOR: When 1, uses get_inputs() function to generate input shapes.
//                        When 0, uses predefined input shapes (first 9 from the original 18).
//                        When 1, uses all shapes from get_inputs(0), matching the original ctest
//                        behavior when TEST_GET_INPUT_TENSOR is enabled.
#ifndef TEST_GET_INPUT_TENSOR
#define TEST_GET_INPUT_TENSOR 0
#endif

namespace {

struct Pooling2dTestCase
{
    std::array<int, 4> input_dims; // [N, C, H, W]
    std::array<int, 2> lens;       // [H, W]
    std::array<int, 2> pads;       // [H, W]
    std::array<int, 2> strides;    // [H, W]
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

// Helper function to calculate output spatial dimensions for 2D pooling
std::array<int, 4> CalculateOutputDims(const std::array<int, 4>& input_dims,
                                       const std::array<int, 2>& lens,
                                       const std::array<int, 2>& strides,
                                       const std::array<int, 2>& pads)
{
    // input_dims is [N, C, H, W]
    // Returns [N, C, H_out, W_out]
    std::array<int, 4> output_dims;
    output_dims[0] = input_dims[0];
    output_dims[1] = input_dims[1];
    for(int i = 0; i < 2; i++)
    {
        int input_size     = input_dims[i + 2];
        int output_size    = (input_size + 2 * pads[i] - lens[i]) / strides[i] + 1;
        output_dims[i + 2] = output_size;
    }
    return output_dims;
}

// Helper function to get index max value
size_t GetIndexMax(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: return std::numeric_limits<uint8_t>::max();
    case miopenIndexUint16: return std::numeric_limits<uint16_t>::max();
    case miopenIndexUint32: return std::numeric_limits<uint32_t>::max();
    case miopenIndexUint64: return std::numeric_limits<uint64_t>::max();
    default: return SIZE_MAX;
    }
}

// Helper function to check if a test case should be included
// This matches the original ctest filtering logic for Dataset 0
bool ShouldIncludeTestCase(const Pooling2dTestCase& test_case)
{
    // Check 1: Validate dimensions (spt_dim == 2 for 2D pooling)
    int spt_dim = static_cast<int>(test_case.input_dims.size()) - 2;
    if(spt_dim != 2)
    {
        return false;
    }

    // Check 2: Validate kernel size doesn't exceed input + padding
    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] >
           (test_case.input_dims[i + 2] + static_cast<int>(2) * test_case.pads[i]))
        {
            return false;
        }
    }

    // Check 3: Skip wide dataset with wsidx=0 and max pooling
    bool is_wide_dataset = false;
    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] >= 35) // Wide window threshold
        {
            is_wide_dataset = true;
            break;
        }
    }
    if(test_case.wsidx == 0 && test_case.mode == miopenPoolingMax && is_wide_dataset)
    {
        return false;
    }

    // Check 4: Skip uint8/uint16 max pooling with wsidx=1 in 2D
    if(test_case.mode == miopenPoolingMax && test_case.wsidx == 1 &&
       (test_case.index_type == miopenIndexUint8 || test_case.index_type == miopenIndexUint16))
    {
        return false;
    }

    // Check 5: Skip average pooling with wsidx=0 (workspace index modes are irrelevant for Average)
    // This matches original ctest behavior: skip to optimize performance, but ensure wsidx=1 is
    // tested
    if(test_case.wsidx == 0 &&
       (test_case.mode == miopenPoolingAverage || test_case.mode == miopenPoolingAverageInclusive))
    {
        return false;
    }

    // Check 5: Index range validation for max pooling
    if(test_case.mode == miopenPoolingMax)
    {
        size_t index_max = GetIndexMax(test_case.index_type);

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
                return false;
            }
        }
        else // miopenPoolingWorkspaceIndexImage (wsidx == 1)
        {
            // Check if index_max is sufficient for output spatial dimensions
            auto output_dims = CalculateOutputDims(
                test_case.input_dims, test_case.lens, test_case.strides, test_case.pads);
            size_t output_spatial_product =
                static_cast<size_t>(output_dims[2]) * static_cast<size_t>(output_dims[3]);
            if(index_max <= output_spatial_product)
            {
                return false;
            }
        }
    }

    return true;
}

std::vector<Pooling2dTestCase> GetPooling2dTestCases()
{
    std::vector<Pooling2dTestCase> test_cases;

    // Counters to limit non-uint8 index types (matching original ctest behavior)
    // The original ctest limits these to speed up testing of the default dataset
    int num_uint16_case        = 0;
    int num_uint32_case        = 0;
    int num_uint32_case_imgidx = 0;
    int num_uint64_case        = 0;
    int num_uint64_case_imgidx = 0;

    // Dataset 0: Default dataset (various tensor sizes)
    std::vector<std::vector<int>> dataset0_inputs;
#if TEST_GET_INPUT_TENSOR
    // When TEST_GET_INPUT_TENSOR = 1, use get_inputs() function (matching original ctest behavior)
    int batch_factor                      = 0; // Default batch factor matching original ctest
    std::set<std::vector<int>> in_dim_set = get_inputs<int>(batch_factor);
    dataset0_inputs.assign(in_dim_set.begin(), in_dim_set.end());
#else
    // When TEST_GET_INPUT_TENSOR = 0, use predefined shapes
    // Limited to 9 input shapes (matching generate_multi_data_limited with limit_multiplier=9)
    dataset0_inputs = {{1, 19, 1024, 2048},
                       {10, 3, 32, 32},
                       {5, 32, 8, 8},
                       {2, 1024, 12, 12},
                       {4, 3, 231, 231},
                       {8, 3, 227, 227},
                       {1, 384, 13, 13},
                       {1, 96, 27, 27},
                       {2, 160, 7, 7}}; // First 9 from the original 18
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
                                Pooling2dTestCase test_case = {
                                    {input_dims[0], input_dims[1], input_dims[2], input_dims[3]},
                                    {lens[0], lens[1]},
                                    {pads[0], pads[1]},
                                    {strides[0], strides[1]},
                                    index_type,
                                    mode,
                                    wsidx};
                                if(ShouldIncludeTestCase(test_case))
                                {
                                    // Apply original ctest limits for non-uint8 index types
                                    // (matching skip_many_configs_with_non_int8_index logic)
                                    bool should_add = true;
                                    switch(index_type)
                                    {
                                    case miopenIndexUint16:
                                        // Only test 5 uint16 cases total
                                        if(num_uint16_case >= 5)
                                        {
                                            should_add = false;
                                        }
                                        else
                                        {
                                            ++num_uint16_case;
                                        }
                                        break;
                                    case miopenIndexUint32:
                                        // Only test 5 uint32 cases for each wsidx mode
                                        if(wsidx == 0)
                                        {
                                            if(num_uint32_case >= 5)
                                            {
                                                should_add = false;
                                            }
                                            else
                                            {
                                                ++num_uint32_case;
                                            }
                                        }
                                        else
                                        {
                                            if(num_uint32_case_imgidx >= 5)
                                            {
                                                should_add = false;
                                            }
                                            else
                                            {
                                                ++num_uint32_case_imgidx;
                                            }
                                        }
                                        break;
                                    case miopenIndexUint64:
                                        // Only test 5 uint64 cases for wsidx=0
                                        // For wsidx=1, limit to 5 cases for 2D (spt_dim == 2)
                                        if(wsidx == 0)
                                        {
                                            if(num_uint64_case >= 5)
                                            {
                                                should_add = false;
                                            }
                                            else
                                            {
                                                ++num_uint64_case;
                                            }
                                        }
                                        else
                                        {
                                            // For 2D pooling (spt_dim == 2), limit to 5 cases
                                            if(num_uint64_case_imgidx >= 5)
                                            {
                                                should_add = false;
                                            }
                                            else
                                            {
                                                ++num_uint64_case_imgidx;
                                            }
                                        }
                                        break;
                                    case miopenIndexUint8:
                                    default:
                                        // No limit for uint8
                                        break;
                                    }

                                    if(should_add)
                                    {
                                        test_cases.push_back(test_case);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Note: Dataset 1 (asymmetric) and Dataset 2 (wide window) are tested separately
    // via pooling2d_asymmetric.cpp and pooling2d_wide.cpp to maintain the same
    // structure as the original ctest implementation.

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

    // Validate indices are populated (required for max pooling backward)
    if(test_case.mode == miopenPoolingMax && indices.empty())
    {
        GTEST_FAIL() << "Indices not populated for max pooling backward";
    }

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

std::string GetPooling2dTestCaseName(const testing::TestParamInfo<Pooling2dTestCase>& info)
{
    const auto& tc = info.param;
    std::ostringstream os;
    os << tc; // Use operator<< to format
    std::string result = os.str();
    // Convert to valid test name format: remove spaces, brackets, colons, commas
    // Replace with underscores and remove separators
    std::string name;
    name.reserve(result.size());
    for(char c : result)
    {
        if(c == '[' || c == ']' || c == ':' || c == ',' || c == ' ')
        {
            if(!name.empty() && name.back() != '_')
                name += '_';
        }
        else
        {
            name += c;
        }
    }
    // Remove trailing underscores and clean up multiple consecutive underscores
    std::string cleaned;
    cleaned.reserve(name.size());
    bool last_was_underscore = false;
    for(char c : name)
    {
        if(c == '_')
        {
            if(!last_was_underscore)
            {
                cleaned += c;
                last_was_underscore = true;
            }
        }
        else
        {
            cleaned += c;
            last_was_underscore = false;
        }
    }
    // Remove trailing underscore if present
    if(!cleaned.empty() && cleaned.back() == '_')
        cleaned.pop_back();
    return cleaned;
}

} // namespace

template <typename T>
struct Pooling2dCommon : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

using GPU_Pooling2d_FP32 = Pooling2dCommon<float>;
using GPU_Pooling2d_FP16 = Pooling2dCommon<half_float::half>;

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunPooling2dTest<float>(GetParam()); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunPooling2dTest<half_float::half>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         GetPooling2dTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         GetPooling2dTestCaseName);
