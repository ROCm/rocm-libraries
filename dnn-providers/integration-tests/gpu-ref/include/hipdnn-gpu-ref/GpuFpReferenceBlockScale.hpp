// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

class GpuFpReferenceBlockScale
{
public:
    static constexpr unsigned int BLOCK_SIZE = 256;

    template <class XDataType, class YDataType, class ScaleDataType, class ComputeDataType = float>
    static void quantize(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                         hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                         hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                         int32_t blockSize,
                         std::optional<int64_t> axis = std::nullopt)
    {
        static_assert(IS_SUPPORTED_IO_TYPE<XDataType> && IS_SUPPORTED_IO_TYPE<YDataType>);
        static_assert(IS_SUPPORTED_SCALE_TYPE<ScaleDataType>);
        static_assert(std::is_same_v<ComputeDataType, float>);

        validateQuantize(x.dims(), y.dims(), scale.dims(), blockSize, axis);
        const auto targetAxis = axis.value_or(static_cast<int64_t>(x.dims().size()) - 1);
        auto defines = buildDefines<XDataType, YDataType, ScaleDataType, ComputeDataType>();
        launchQuantize(x.memory().deviceData(),
                       x.dims(),
                       x.strides(),
                       y.memory().deviceData(),
                       y.strides(),
                       scale.memory().deviceData(),
                       scale.dims(),
                       scale.strides(),
                       blockSize,
                       targetAxis,
                       static_cast<double>(std::numeric_limits<YDataType>::lowest()),
                       static_cast<double>(std::numeric_limits<YDataType>::max()),
                       defines);
        y.memory().markDeviceModified();
        scale.memory().markDeviceModified();
    }

    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void dequantize(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                           hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                           hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                           const std::vector<int32_t>& blockSize,
                           bool isNegativeScale = false)
    {
        static_assert(IS_SUPPORTED_IO_TYPE<XDataType> && IS_SUPPORTED_IO_TYPE<YDataType>);
        static_assert(IS_SUPPORTED_SCALE_TYPE<ScaleDataType>);
        static_assert(std::is_same_v<ComputeDataType, float>);

        const auto effectiveBlockSize
            = validateDequantize(x.dims(), y.dims(), scale.dims(), blockSize);
        auto defines = buildDefines<XDataType, YDataType, ScaleDataType, ComputeDataType>();
        launchDequantize(x.memory().deviceData(),
                         x.dims(),
                         x.strides(),
                         scale.memory().deviceData(),
                         scale.strides(),
                         y.memory().deviceData(),
                         y.strides(),
                         effectiveBlockSize,
                         isNegativeScale,
                         defines);
        y.memory().markDeviceModified();
    }

private:
    template <typename T>
    static constexpr bool IS_SUPPORTED_IO_TYPE
        = std::is_same_v<T, float> || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16> || std::is_same_v<T, int8_t>
          || std::is_same_v<T, uint8_t> || std::is_same_v<T, int32_t>
          || std::is_same_v<T, hipdnn_data_sdk::types::fp4_e2m1>
          || std::is_same_v<T, hipdnn_data_sdk::types::fp6_e2m3>
          || std::is_same_v<T, hipdnn_data_sdk::types::fp6_e3m2>
          || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e4m3>
          || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e5m2>;

    template <typename T>
    static constexpr bool IS_SUPPORTED_SCALE_TYPE
        = std::is_same_v<T, float> || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e8m0>;

    template <typename XDataType,
              typename YDataType,
              typename ScaleDataType,
              typename ComputeDataType>
    static std::vector<std::string> buildDefines()
    {
        return {std::string("-DX_TYPE=") + detail::HipRtcTypeName<XDataType>::VALUE,
                std::string("-DY_TYPE=") + detail::HipRtcTypeName<YDataType>::VALUE,
                std::string("-DSCALE_TYPE=") + detail::HipRtcTypeName<ScaleDataType>::VALUE,
                std::string("-DCOMPUTE_TYPE=") + detail::HipRtcTypeName<ComputeDataType>::VALUE};
    }

    static void validateQuantize(const std::vector<int64_t>& xDims,
                                 const std::vector<int64_t>& yDims,
                                 const std::vector<int64_t>& scaleDims,
                                 int32_t blockSize,
                                 std::optional<int64_t> axis);
    static std::vector<int64_t> validateDequantize(const std::vector<int64_t>& xDims,
                                                   const std::vector<int64_t>& yDims,
                                                   const std::vector<int64_t>& scaleDims,
                                                   const std::vector<int32_t>& blockSize);
    static void launchQuantize(const void* x,
                               const std::vector<int64_t>& xDims,
                               const std::vector<int64_t>& xStrides,
                               void* y,
                               const std::vector<int64_t>& yStrides,
                               void* scale,
                               const std::vector<int64_t>& scaleDims,
                               const std::vector<int64_t>& scaleStrides,
                               int32_t blockSize,
                               int64_t axis,
                               double outputLowest,
                               double outputMax,
                               const std::vector<std::string>& defines);
    static void launchDequantize(const void* x,
                                 const std::vector<int64_t>& xDims,
                                 const std::vector<int64_t>& xStrides,
                                 const void* scale,
                                 const std::vector<int64_t>& scaleStrides,
                                 void* y,
                                 const std::vector<int64_t>& yStrides,
                                 const std::vector<int64_t>& effectiveBlockSize,
                                 bool isNegativeScale,
                                 const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
