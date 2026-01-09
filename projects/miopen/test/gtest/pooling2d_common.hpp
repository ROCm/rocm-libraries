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

// Filtering function matching ctest's run() method exactly
// This copies the exact logic from pooling_common.hpp pooling_driver::run()
// Matching variable names: idx_typ, idx_sz, spt_dim, skip_many_configs_with_non_int8_index,
// wide_dataset, full_set
inline bool ShouldIncludeTestCase(const Pooling2dTestCase& test_case,
                                  bool skip_wide_check         = false,
                                  bool apply_index_type_limits = true,
                                  bool is_wide_dataset         = false)
{
    // Match ctest variable names exactly
    auto idx_typ = test_case.index_type;
    auto idx_sz  = sizeof(uint8_t);
    int spt_dim  = static_cast<int>(test_case.input_dims.size()) - 2;
    const bool skip_many_configs_with_non_int8_index =
        apply_index_type_limits;     // dataset_id == 0 && full_set
    const bool wide_dataset = is_wide_dataset; // dataset_id == 2 && full_set
    const bool full_set     = true;  // Always true for Dataset 0

    // Match ctest run() order exactly:
    // 1. wsidx == 0 && spt_dim == 3 && max && full_set (not applicable for 2D)
    if(test_case.wsidx == 0 && spt_dim == 3 && test_case.mode == miopenPoolingMax && full_set)
    {
        return false;
    }

    // 2. wsidx == 0 && spt_dim == 2 && max && wide_dataset
    // Note: wide_dataset is false for Dataset 0, so this check won't trigger for Dataset 0
    // But we keep it to match ctest structure exactly
    if(test_case.wsidx == 0 && spt_dim == 2 && test_case.mode == miopenPoolingMax && wide_dataset)
    {
        return false;
    }

    // 3. wsidx == 0 && average && full_set
    if(test_case.wsidx == 0 &&
       (test_case.mode == miopenPoolingAverage ||
        test_case.mode == miopenPoolingAverageInclusive) &&
       full_set)
    {
        return false;
    }

    // 4. switch(idx_typ) - matches ctest exactly
    switch(idx_typ)
    {
    case miopenIndexUint8: {
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) && full_set &&
           test_case.mode == miopenPoolingMax)
        {
            return false;
        }
        break;
    }
    case miopenIndexUint16: {
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) && full_set &&
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
    // Convert test_case.input_dims to vector for GetLengths()
    std::vector<int> in_shape_vec(test_case.input_dims.begin(), test_case.input_dims.end());
    std::vector<int> lens_vec(test_case.lens.begin(), test_case.lens.end());
    std::vector<int> pads_vec(test_case.pads.begin(), test_case.pads.end());
    miopen::TensorDescriptor input_desc(miopenFloat, in_shape_vec);
    for(int i = 0; i < spt_dim; i++)
    {
        if(lens_vec[i] > (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * pads_vec[i]))
        {
            return false;
        }
    }

    // 7. Memory check (matching ctest exactly)
    if(full_set)
    {
        try
        {
            auto output_desc = miopen::PoolingDescriptor(test_case.mode,
                                                         miopenPaddingDefault,
                                                         lens_vec,
                                                         std::vector<int>(test_case.strides.begin(),
                                                                          test_case.strides.end()),
                                                         pads_vec)
                                   .GetForwardOutputTensor(input_desc);
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

// Note: Global counters (num_uint16_case, num_uint32_case, etc.) are defined in pooling_common.hpp
// which is included above, so we use those directly

// Helper function to generate test cases for a single input configuration
// Uses original loops matching ctest generation order
inline void AddTestCasesForInput(const std::vector<int>& input_dims,
                                 const std::vector<std::vector<int>>& lens_list,
                                 const std::vector<std::vector<int>>& strides_list,
                                 const std::vector<std::vector<int>>& pads_list,
                                 const std::vector<miopenIndexType_t>& index_types,
                                 const std::vector<miopenPoolingMode_t>& modes,
                                 const std::vector<int>& wsidx_values,
                                 std::vector<Pooling2dTestCase>& test_cases,
                                 bool skip_wide_check         = false,
                                 bool apply_index_type_limits = true,
                                 bool is_wide_dataset          = false)
{
    // Match ctest order exactly: index_type -> mode -> lens -> strides -> pads -> wsidx
    // This matches the order parameters are added in pooling_driver (base class adds index_type,
    // mode first, then derived class adds lens, strides, pads, wsidx)
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
                            Pooling2dTestCase test_case = {
                                {input_dims[0], input_dims[1], input_dims[2], input_dims[3]},
                                {lens[0], lens[1]},
                                {pads[0], pads[1]},
                                {strides[0], strides[1]},
                                index_type,
                                mode,
                                wsidx};

                            if(ShouldIncludeTestCase(
                                   test_case, skip_wide_check, apply_index_type_limits, is_wide_dataset))
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
    void SetUp() override
    {
        prng::reset_seed();
        // Reset internal environment values - ensure clean state for each test
        // Note: get_handle() is called inside verify_forward_pooling::gpu() and
        // verify_backward_pooling::gpu() (in pooling_common.hpp), which creates a fresh handle
        // for each test execution, resetting internal MIOpen state
    }

protected:
    // Common test execution method for all pooling2d tests
    void RunTest() { RunPooling2dTest<T>(this->GetParam()); }
};

} // namespace pooling2d_gtest

#endif // GUARD_MIOPEN_TEST_GTEST_POOLING2D_COMMON_HPP
