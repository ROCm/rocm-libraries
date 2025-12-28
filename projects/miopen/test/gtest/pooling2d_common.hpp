// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP
#define GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include <miopen/logger.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../network_data.hpp"
#include "../pooling_common.hpp"

// Configuration defines matching the original ctest behavior
// These can be overridden at compile time via -D flags
#ifndef WORKAROUND_ISSUE_1670
#define WORKAROUND_ISSUE_1670 1
#endif

#ifndef TEST_GET_INPUT_TENSOR
#define TEST_GET_INPUT_TENSOR 0
#endif

namespace pooling2d_gtest {

// Dataset definitions (matching original pooling2d.hpp ctest driver, now removed):
// - Dataset 0: Default dataset with various tensor sizes (tested in pooling2d.cpp)
// - Dataset 1: Intended for testing of asymmetric configs (tested in pooling2d_asymmetric.cpp)
// - Dataset 2: Intended for testing of configs with wide window (tested in pooling2d_wide.cpp)

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
inline std::array<int, 4> CalculateOutputDims(const std::array<int, 4>& input_dims,
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
inline size_t GetIndexMax(miopenIndexType_t index_type)
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
// This matches the original ctest filtering logic
// skip_wide_check: if true, skips the wide dataset check (for Dataset 1 - asymmetric)
inline bool ShouldIncludeTestCase(const Pooling2dTestCase& test_case, bool skip_wide_check = false)
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

    // Check 3: Skip wide dataset with wsidx=0 and max pooling (only for Dataset 0 and Dataset 2)
    if(!skip_wide_check)
    {
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
    }

    // Check 4: Skip uint8/uint16 max pooling with wsidx=1 in 2D
    // The original ctest skips these when full_set is true (with --all flag)
    // because uint8/uint16 index range is insufficient for output spatial dimensions
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

    // Check 6: Index range validation for max pooling
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

// Helper struct to track index type limits (matching original ctest behavior)
struct IndexTypeCounters
{
    int num_uint16_case        = 0;
    int num_uint32_case        = 0;
    int num_uint32_case_imgidx = 0;
    int num_uint64_case        = 0;
    int num_uint64_case_imgidx = 0;

    // Check if we should add a test case based on index type limits
    bool ShouldAddBasedOnIndexType(miopenIndexType_t index_type, int wsidx)
    {
        switch(index_type)
        {
        case miopenIndexUint16:
            // Only test 5 uint16 cases total (but ctest uses > 5, allowing 6 cases)
            // Match ctest behavior exactly: if(num_uint16_case > 5) return false;
            if(num_uint16_case > 5)
                return false;
            ++num_uint16_case;
            return true;
        case miopenIndexUint32:
            // Only test 5 uint32 cases for each wsidx mode (but ctest uses > 5, allowing 6)
            // Match ctest behavior exactly
            if(wsidx == 0)
            {
                if(num_uint32_case > 5)
                    return false;
                ++num_uint32_case;
            }
            else
            {
                if(num_uint32_case_imgidx > 5)
                    return false;
                ++num_uint32_case_imgidx;
            }
            return true;
        case miopenIndexUint64:
            // Only test 5 uint64 cases for wsidx=0 (but ctest uses > 5, allowing 6)
            // For wsidx=1, limit to 5 cases for 2D (spt_dim == 2)
            // Match ctest behavior exactly
            if(wsidx == 0)
            {
                if(num_uint64_case > 5)
                    return false;
                ++num_uint64_case;
            }
            else
            {
                // For 2D pooling (spt_dim == 2), limit to 5 cases (but ctest uses > 5)
                if(num_uint64_case_imgidx > 5)
                    return false;
                ++num_uint64_case_imgidx;
            }
            return true;
        case miopenIndexUint8:
        default:
            // No limit for uint8
            return true;
        }
    }
};

// Helper function to generate test cases for a single input configuration
inline void AddTestCasesForInput(const std::vector<int>& input_dims,
                                 const std::vector<std::vector<int>>& lens_list,
                                 const std::vector<std::vector<int>>& strides_list,
                                 const std::vector<std::vector<int>>& pads_list,
                                 const std::vector<miopenIndexType_t>& index_types,
                                 const std::vector<miopenPoolingMode_t>& modes,
                                 const std::vector<int>& wsidx_values,
                                 IndexTypeCounters& counters,
                                 std::vector<Pooling2dTestCase>& test_cases,
                                 bool skip_wide_check = false)
{
    for(const auto& lens : lens_list)
    {
        for(const auto& strides : strides_list)
        {
            for(const auto& pads : pads_list)
            {
                for(const auto& index_type : index_types)
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
                            if(ShouldIncludeTestCase(test_case, skip_wide_check))
                            {
                                // Apply original ctest limits for non-uint8 index types
                                if(counters.ShouldAddBasedOnIndexType(index_type, wsidx))
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
    // Convert std::array to std::vector for PoolingDescriptor constructor
    std::vector<int> lens_vec(test_case.lens.begin(), test_case.lens.end());
    std::vector<int> strides_vec(test_case.strides.begin(), test_case.strides.end());
    std::vector<int> pads_vec(test_case.pads.begin(), test_case.pads.end());
    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, lens_vec, strides_vec, pads_vec};
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

inline std::string GetPooling2dTestCaseName(const testing::TestParamInfo<Pooling2dTestCase>& info)
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

template <typename T>
struct Pooling2dCommon : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }

protected:
    // Common test execution method for all pooling2d tests
    void RunTest() { RunPooling2dTest<T>(this->GetParam()); }
};

} // namespace pooling2d_gtest

#endif // GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP
