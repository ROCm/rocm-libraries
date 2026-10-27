// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn-gpu-ref/GpuIntReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_resample_ref_test
{

enum class ResampleDirection
{
    FORWARD,
    BACKWARD
};

struct ResampleTestCase
{
    std::vector<int64_t> xDims; // Maps to dxDims for backward tests
    hipdnn_data_sdk::utilities::TensorLayout layout;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode;
    ResampleDirection direction = ResampleDirection::FORWARD;

    // Derived member variable
    std::vector<int64_t> yDims; // Maps to dyDims for backward tests

    ResampleTestCase(std::vector<int64_t> xDims,
                     hipdnn_data_sdk::utilities::TensorLayout layout,
                     std::vector<int64_t> prePadding,
                     std::vector<int64_t> postPadding,
                     std::vector<int64_t> stride,
                     std::vector<int64_t> window,
                     hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
                     hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
                     ResampleDirection direction = ResampleDirection::FORWARD)
        : xDims(std::move(xDims))
        , layout(std::move(layout))
        , prePadding(std::move(prePadding))
        , postPadding(std::move(postPadding))
        , stride(std::move(stride))
        , window(std::move(window))
        , resampleMode(resampleMode)
        , paddingMode(paddingMode)
        , direction(direction)
    {
        computeYDims();
    }

    friend std::ostream& operator<<(std::ostream& os, const ResampleTestCase& tc)
    {
        os << "(direction: "
           << (tc.direction == ResampleDirection::FORWARD ? "forward" : "backward");
        if(tc.direction == ResampleDirection::FORWARD)
        {
            os << " x dims: ";
            hipdnn_data_sdk::utilities::vecToStream(os, tc.xDims);
            os << " y dims: ";
            hipdnn_data_sdk::utilities::vecToStream(os, tc.yDims);
        }
        else
        {
            os << " dx dims: ";
            hipdnn_data_sdk::utilities::vecToStream(os, tc.xDims);
            os << " dy dims: ";
            hipdnn_data_sdk::utilities::vecToStream(os, tc.yDims);
        }
        os << " layout:" << tc.layout.name;
        os << " pre-padding: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.prePadding);
        os << " post-padding: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.postPadding);
        os << " stride: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.stride);
        os << " window: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.window);
        os << " resample mode: "
           << hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(tc.resampleMode);
        os << " padding mode: "
           << hipdnn_flatbuffers_sdk::data_objects::EnumNamePaddingMode(tc.paddingMode);
        os << ")";

        return os;
    }

private:
    void computeYDims()
    {
        yDims = xDims;
        const auto spatialRank = prePadding.size();
        const auto spatialOffset = xDims.size() - spatialRank;

        for(size_t i = 0; i < spatialRank; ++i)
        {
            yDims[spatialOffset + i]
                = (xDims[spatialOffset + i] + prePadding[i] + postPadding[i] - window[i])
                      / stride[i]
                  + 1;
        }
    }
};

template <
    typename T,
    typename std::enable_if_t<std::disjunction_v<std::is_same<T, float>,
                                                 std::is_same<T, double>,
                                                 std::is_same<T, hipdnn_data_sdk::types::half>,
                                                 std::is_same<T, hipdnn_data_sdk::types::bfloat16>>,
                              int>
    = 0>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance)
{
    auto validator = hipdnn_gpu_ref::GpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

template <typename T,
          typename std::enable_if_t<std::disjunction_v<std::is_same<T, int8_t>,
                                                       std::is_same<T, uint8_t>,
                                                       std::is_same<T, int32_t>>,
                                    int>
          = 0>
void assertAllExact(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual)
{
    auto validator = hipdnn_gpu_ref::GpuIntReferenceValidation<T>();
    ASSERT_TRUE(validator.allClose(expected, actual));
}

} // namespace gpu_resample_ref_test
