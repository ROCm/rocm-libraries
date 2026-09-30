// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <vector>

namespace gpu_layernorm_ref_test
{

struct LayernormTestCase
{
    std::vector<int64_t> dims;
    int64_t normalizedDim;
    bool optionalTensors;
    hipdnn_data_sdk::utilities::TensorLayout layout;

    std::vector<int64_t> computeBatchDims() const
    {
        std::vector<int64_t> out(dims.size(), 1);
        for(size_t i = 0; i < static_cast<size_t>(normalizedDim); ++i)
        {
            out[i] = dims[i];
        }
        return out;
    }

    std::vector<int64_t> computeNormalizedDims() const
    {
        std::vector<int64_t> out(dims.size(), 1);
        for(size_t i = 0; i < dims.size() - static_cast<size_t>(normalizedDim); ++i)
        {
            out[i + static_cast<size_t>(normalizedDim)]
                = dims[i + static_cast<size_t>(normalizedDim)];
        }
        return out;
    }

    int64_t computeNormalizedDimCount() const
    {
        return static_cast<int64_t>(dims.size()) - normalizedDim;
    }

    friend std::ostream& operator<<(std::ostream& ss, const LayernormTestCase& testCase)
    {
        ss << "(dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.dims);
        ss << " normalizedDim:" << testCase.normalizedDim;
        ss << " optionalTensors:" << testCase.optionalTensors;
        ss << " layout:" << testCase.layout;
        ss << ")";

        return ss;
    }
};

template <typename T>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance,
                    const std::string& tensorName = "Tensor")
{
    auto validator = hipdnn_gpu_ref::GpuFpReferenceValidation<T>(tolerance, tolerance);
    ASSERT_TRUE(validator.allClose(expected, actual)) << tensorName << " failed verification";
}

} // namespace gpu_layernorm_ref_test
