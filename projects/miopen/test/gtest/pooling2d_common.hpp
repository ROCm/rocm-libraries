// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP
#define GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include <limits>
#include <numeric>
#include <miopen/logger.hpp>
#include <numeric>
#include <miopen/tensor_layout.hpp>
#include <sstream>
#include <vector>
// network_data.hpp provides get_inputs() function used when TEST_GET_INPUT_TENSOR = 1
// (currently TEST_GET_INPUT_TENSOR = 0, but include is needed to support both cases)
#include "../network_data.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling_gtest_common.hpp"

// Configuration defines matching the original ctest behavior
#define WORKAROUND_ISSUE_1670 1
#define TEST_GET_INPUT_TENSOR 0

namespace pooling2d_gtest {

// Dataset definitions (matching original pooling2d.hpp ctest harness, now removed):
// - Dataset 0: Default dataset with various tensor sizes (tested in pooling2d.cpp)
// - Dataset 1: Intended for testing of asymmetric configs (tested in pooling2d_asymmetric.cpp)
// - Dataset 2: Intended for testing of configs with wide window (tested in pooling2d_wide.cpp)

// Unified test case structure for both 2D and 3D pooling (matches ctest pooling_driver names)
// For 2D: in_shape is [N, C, H, W], lens/pads/strides are [H, W]
// For 3D: in_shape is [N, C, D, H, W], lens/pads/strides are [D, H, W]
struct PoolingTestCase
{
    std::vector<int> in_shape; // [N, C, ...] - matches ctest in_shape
    std::vector<int> lens;     // Spatial dimensions (2 for 2D, 3 for 3D)
    std::vector<int> pads;     // Spatial dimensions (2 for 2D, 3 for 3D)
    std::vector<int> strides;  // Spatial dimensions (2 for 2D, 3 for 3D)
    miopenIndexType_t index_type;
    miopenPoolingMode_t mode;
    int wsidx;
    std::string in_layout{"NCHW"};  // NCHW for 2D, NCDHW for 3D; use NHWC/NDHWC for channel-last
    std::string out_layout{"NCHW"}; // same as in_layout typically

    friend std::ostream& operator<<(std::ostream& os, const PoolingTestCase& tc)
    {
        os << "in_shape: ";
        miopen::LogRange(os << "[", tc.in_shape, ",") << "] ";
        os << "lens: ";
        miopen::LogRange(os << "[", tc.lens, ",") << "] ";
        os << "pads: ";
        miopen::LogRange(os << "[", tc.pads, ",") << "] ";
        os << "strides: ";
        miopen::LogRange(os << "[", tc.strides, ",") << "] ";
        os << "in_layout: " << tc.in_layout << " ";
        return os << "index_type: " << tc.index_type << ", mode: " << tc.mode
                  << ", wsidx: " << tc.wsidx;
    }
};

// Helper function to calculate output spatial dimensions for pooling
// Works for both 2D and 3D pooling based on in_shape size
inline std::vector<int> CalculateOutputDims(const std::vector<int>& in_shape,
                                            const std::vector<int>& lens,
                                            const std::vector<int>& strides,
                                            const std::vector<int>& pads)
{
    // in_shape is [N, C, ...] where ... are spatial dims
    // Returns [N, C, ...] with calculated output spatial dims
    std::vector<int> output_dims;
    output_dims.reserve(in_shape.size());
    output_dims.push_back(in_shape[0]); // N
    output_dims.push_back(in_shape[1]); // C
    for(size_t i = 0; i < lens.size(); i++)
    {
        int input_size  = in_shape[i + 2];
        int output_size = (input_size + 2 * pads[i] - lens[i]) / strides[i] + 1;
        output_dims.push_back(output_size);
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

// Filtering function matching ctest's run() method exactly
// This copies the exact logic from test/pooling_common.hpp pooling_driver::run()
// Matching variable names: idx_typ, idx_sz, spt_dim, wide_dataset
inline bool ShouldIncludeTestCase(const PoolingTestCase& test_case,
                                  int& num_uint16_case,
                                  int& num_uint32_case,
                                  int& num_uint32_case_imgidx,
                                  int& num_uint64_case,
                                  int& num_uint64_case_imgidx,
                                  bool /*skip_wide_check         = false*/,
                                  bool apply_index_type_limits = true,
                                  bool is_wide_dataset         = false)
{
    // Match ctest variable names exactly
    auto idx_typ = test_case.index_type;
    auto idx_sz  = sizeof(uint8_t);
    int spt_dim  = static_cast<int>(test_case.in_shape.size()) - 2;
    const bool skip_many_configs_with_non_int8_index =
        apply_index_type_limits;               // dataset_id == 0 behavior
    const bool wide_dataset = is_wide_dataset; // dataset_id == 2 behavior

    // Match ctest run() order exactly:
    // 1. wsidx == 0 && spt_dim == 3 && max (not applicable for 2D)
    if(test_case.wsidx == 0 && spt_dim == 3 && test_case.mode == miopenPoolingMax)
    {
        return false;
    }

    // 2. wsidx == 0 && spt_dim == 2 && max && wide_dataset
    if(test_case.wsidx == 0 && spt_dim == 2 && test_case.mode == miopenPoolingMax && wide_dataset)
    {
        return false;
    }

    // 3. wsidx == 0 && average
    if(test_case.wsidx == 0 &&
       (test_case.mode == miopenPoolingAverage || test_case.mode == miopenPoolingAverageInclusive))
    {
        return false;
    }

    // 4. Additional 3D specific skip: uint8/uint16 max pooling in 3D
    if(spt_dim == 3 && test_case.mode == miopenPoolingMax &&
       (test_case.index_type == miopenIndexUint8 || test_case.index_type == miopenIndexUint16))
    {
        return false;
    }

    // 5. switch(idx_typ) - matches ctest exactly
    switch(idx_typ)
    {
    case miopenIndexUint8: {
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) &&
           test_case.mode == miopenPoolingMax)
        {
            return false;
        }
        break;
    }
    case miopenIndexUint16: {
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) &&
           test_case.mode == miopenPoolingMax)
        {
            return false;
        }
        if(skip_many_configs_with_non_int8_index)
        {
            if(num_uint16_case > 5)
            {
                return false;
            }
            ++num_uint16_case;
        }
        idx_sz = sizeof(uint16_t);
        break;
    }
    case miopenIndexUint32: {
        if(skip_many_configs_with_non_int8_index)
        {
            if(test_case.wsidx == 0)
            {
                if(num_uint32_case > 5)
                {
                    return false;
                }
                ++num_uint32_case;
            }
            else
            {
                if(num_uint32_case_imgidx > 5)
                {
                    return false;
                }
                ++num_uint32_case_imgidx;
            }
        }
        idx_sz = sizeof(uint32_t);
        break;
    }
    case miopenIndexUint64: {
        if(skip_many_configs_with_non_int8_index)
        {
            if(test_case.wsidx == 0)
            {
                if(num_uint64_case > 5)
                {
                    return false;
                }
                ++num_uint64_case;
            }
            else
            {
                if(num_uint64_case_imgidx > 5 && spt_dim == 2)
                {
                    return false;
                }
                ++num_uint64_case_imgidx;
            }
        }
        idx_sz = sizeof(uint64_t);
        break;
    }
    }

    // 5. spt_dim != 2 && spt_dim != 3
    if(spt_dim != 2 && spt_dim != 3)
    {
        return false;
    }

    // 6. lens[i] > (input + 2*pads[i])
    miopen::TensorDescriptor input_desc(miopenFloat, test_case.in_shape);
    for(int i = 0; i < spt_dim; i++)
    {
        if(static_cast<size_t>(test_case.lens[i]) >
           (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * test_case.pads[i]))
        {
            return false;
        }
    }

    // 7. Memory check (matching ctest's full_set=true behavior for this gtest path)
    {
        try
        {
            auto pooling_desc = miopen::PoolingDescriptor(test_case.mode,
                                                          miopenPaddingDefault,
                                                          test_case.lens,
                                                          test_case.strides,
                                                          test_case.pads);
            auto output_desc  = pooling_desc.GetForwardOutputTensor(input_desc);

            // 7a. Index range check for max pooling (moved from pooling3d.cpp)
            if(test_case.mode == miopenPoolingMax && test_case.wsidx == 1)
            {
                size_t index_max = GetIndexMax(test_case.index_type);
                if(index_max <= output_desc.GetElementSize())
                {
                    return false;
                }
            }

            size_t total_mem = 3 * input_desc.GetNumBytes() + output_desc.GetNumBytes() +
                               idx_sz * output_desc.GetElementSize();

            size_t device_mem = get_handle().GetGlobalMemorySize();
            if(total_mem >= device_mem)
            {
                return false;
            }
        }
        catch(...)
        {
            // Skip memory check if handle not available
        }
    }

    return true;
}

// Helper function to generate test cases for a single input configuration
// Uses original loops matching ctest generation order
// Note: Global counters (num_uint16_case, etc.) accumulate globally across all test cases
// in ctest, matching the behavior where the test harness processes cases sequentially.
// These counters are NOT reset per input shape - they accumulate to limit the total number
// of non-uint8 index type test cases when apply_index_type_limits is true.
inline void AddTestCasesForInput(const std::vector<int>& in_shape,
                                 const std::vector<std::vector<int>>& lens_list,
                                 const std::vector<std::vector<int>>& strides_list,
                                 const std::vector<std::vector<int>>& pads_list,
                                 const std::vector<miopenIndexType_t>& index_types,
                                 const std::vector<miopenPoolingMode_t>& modes,
                                 const std::vector<int>& wsidx_values,
                                 std::vector<PoolingTestCase>& test_cases,
                                 int& num_uint16_case,
                                 int& num_uint32_case,
                                 int& num_uint32_case_imgidx,
                                 int& num_uint64_case,
                                 int& num_uint64_case_imgidx,
                                 bool skip_wide_check          = false,
                                 bool apply_index_type_limits  = true,
                                 bool is_wide_dataset          = false,
                                 const std::string& in_layout  = "NCHW",
                                 const std::string& out_layout = "NCHW")
{
    // Match ctest order exactly: index_type -> mode -> lens -> strides -> pads -> wsidx
    for(const auto& index_type : index_types)
    {
        for(const auto& mode : modes)
        {
            for(const auto& lens : lens_list)
            {
                for(const auto& strides : strides_list)
                {
                    for(const auto& pads : pads_list)
                    {
                        for(int wsidx : wsidx_values)
                        {
                            PoolingTestCase test_case = {in_shape,
                                                         lens,
                                                         pads,
                                                         strides,
                                                         index_type,
                                                         mode,
                                                         wsidx,
                                                         in_layout,
                                                         out_layout};

                            if(ShouldIncludeTestCase(test_case,
                                                     num_uint16_case,
                                                     num_uint32_case,
                                                     num_uint32_case_imgidx,
                                                     num_uint64_case,
                                                     num_uint64_case_imgidx,
                                                     skip_wide_check,
                                                     apply_index_type_limits,
                                                     is_wide_dataset))
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

// Overload for when apply_index_type_limits is false: counters are not needed.
inline void AddTestCasesForInput(const std::vector<int>& in_shape,
                                 const std::vector<std::vector<int>>& lens_list,
                                 const std::vector<std::vector<int>>& strides_list,
                                 const std::vector<std::vector<int>>& pads_list,
                                 const std::vector<miopenIndexType_t>& index_types,
                                 const std::vector<miopenPoolingMode_t>& modes,
                                 const std::vector<int>& wsidx_values,
                                 std::vector<PoolingTestCase>& test_cases,
                                 bool skip_wide_check,
                                 bool is_wide_dataset,
                                 const std::string& in_layout  = "NCHW",
                                 const std::string& out_layout = "NCHW")
{
    int num_uint16_case = 0, num_uint32_case = 0, num_uint32_case_imgidx = 0;
    int num_uint64_case = 0, num_uint64_case_imgidx = 0;
    AddTestCasesForInput(in_shape,
                         lens_list,
                         strides_list,
                         pads_list,
                         index_types,
                         modes,
                         wsidx_values,
                         test_cases,
                         num_uint16_case,
                         num_uint32_case,
                         num_uint32_case_imgidx,
                         num_uint64_case,
                         num_uint64_case_imgidx,
                         skip_wide_check,
                         false, // apply_index_type_limits
                         is_wide_dataset,
                         in_layout,
                         out_layout);
}

template <typename T, typename Index>
void RunPooling2dTestWithIndexType(const PoolingTestCase& test_case)
{
    // Create input tensor for 2D pooling (in_shape matches ctest)
    tensor<T> input{test_case.in_shape};
    input.generate(tensor_elem_gen_integer{
        (miopen_type<T>{} == miopenHalf || miopen_type<T>{} == miopenBFloat16) ? 5 : 17});

    // Apply NHWC layout if requested
    if(test_case.in_layout != "NCHW")
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
    // For wsidx==0, index verification is known to produce noisy CHECK output ("FAILED:")
    // without causing test failure. Keep verification only for image index mode (wsidx!=0).
    const bool use_global_index = test_case.wsidx != 0;
    const bool verify_index     = use_global_index;
    auto backward_result        = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, use_global_index, verify_index);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, use_global_index, verify_index);

    // Compare backward results
    EXPECT_EQ(miopen::range_distance(backward_result), miopen::range_distance(backward_gpu_result));

    const double backward_rms_error = miopen::rms_range(backward_result, backward_gpu_result);

    EXPECT_LE(backward_rms_error, threshold)
        << "Backward RMS error: " << backward_rms_error << " exceeds threshold: " << threshold;
}

template <typename T>
void RunPooling2dTest(const PoolingTestCase& test_case)
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

// Generate GTest test name from PoolingTestCase
// Uses operator<< for formatting, then sanitizes to create valid GTest test names
// (replaces special characters like [, ], :, ,, spaces with underscores)
inline std::string GetPoolingTestCaseName(const testing::TestParamInfo<PoolingTestCase>& info)
{
    const auto& tc = info.param;
    std::ostringstream os;
    os << tc; // Use operator<< to format
    std::string result = os.str();

    // Helper lambda to check if character is a special character that needs replacement
    auto is_special_char = [](char c) {
        return c == '[' || c == ']' || c == ':' || c == ',' || c == ' ';
    };

    // Sanitize in a single pass: replace special characters with underscores,
    // skip consecutive underscores, and remove trailing underscore
    std::string name;
    name.reserve(result.size());
    bool last_was_underscore = false;
    for(char c : result)
    {
        if(!is_special_char(c))
        {
            // Regular character: add it
            name += c;
            last_was_underscore = false;
        }
        else if(!last_was_underscore || name.empty())
        {
            // Replace special character with underscore
            name += '_';
            last_was_underscore = true;
        }
        // else: special character but already have underscore or name is empty, skip it
    }
    // Remove trailing underscore if present
    if(!name.empty() && name.back() == '_')
        name.pop_back();

    return name;
}

template <typename T>
struct Pooling2dCommon : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        // Reset internal environment values - ensure clean state for each test
        // Note: get_handle() is called inside verify_forward_pooling::gpu() and
        // verify_backward_pooling::gpu() (in pooling_gtest_common.hpp), which creates a fresh
        // handle
        // for each test execution, resetting internal MIOpen state
    }

protected:
    // Common test execution method for all pooling2d tests
    void RunTest() { RunPooling2dTest<T>(this->GetParam()); }
};

} // namespace pooling2d_gtest

#endif // GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP
