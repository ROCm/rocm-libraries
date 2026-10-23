// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceBlockScaleQuantize
{
public:
    // ============================================================================
    //  Supported data types for block scale quantization:
    //
    //  Input (X) / Output (Y) types:
    //      FP32, FP16, BF16,
    //      fp4_e2m1, fp6_e2m3, fp6_e3m2,
    //      fp8_e4m3, fp8_e4m3_fnuz, fp8_e5m2, fp8_e5m2_fnuz,
    //      int8_t, uint8_t, int32_t
    //
    //  Scale type:
    //      FP32, FP16, BF16, fp8_e8m0
    //
    //  Compute type:
    //      FP32, FP64
    // ============================================================================
    template <typename T>
    static constexpr auto IS_VALID_IO_TYPE_V
        = std::disjunction_v<std::is_same<T, float>,
                             std::is_same<T, hipdnn_data_sdk::types::half>,
                             std::is_same<T, hipdnn_data_sdk::types::bfloat16>,
                             std::is_same<T, hipdnn_data_sdk::types::fp4_e2m1>,
                             std::is_same<T, hipdnn_data_sdk::types::fp6_e2m3>,
                             std::is_same<T, hipdnn_data_sdk::types::fp6_e3m2>,
                             std::is_same<T, hipdnn_data_sdk::types::fp8_e4m3>,
                             std::is_same<T, hipdnn_data_sdk::types::fp8_e4m3_fnuz>,
                             std::is_same<T, hipdnn_data_sdk::types::fp8_e5m2>,
                             std::is_same<T, hipdnn_data_sdk::types::fp8_e5m2_fnuz>,
                             std::is_same<T, int8_t>,
                             std::is_same<T, uint8_t>,
                             std::is_same<T, int32_t>>;

    template <typename T>
    static constexpr auto IS_VALID_SCALE_TYPE_V
        = std::disjunction_v<std::is_same<T, float>,
                             std::is_same<T, hipdnn_data_sdk::types::half>,
                             std::is_same<T, hipdnn_data_sdk::types::bfloat16>,
                             std::is_same<T, hipdnn_data_sdk::types::fp8_e8m0>>;

    template <typename T>
    static constexpr auto IS_VALID_COMPUTE_TYPE_V
        = std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>;

    /// Block scale quantization: Y[i] = X[i] / scale[block_of(i)]
    /// Computes each block's maximum absolute value, then stores maxAbs / max(Y), rounded upward,
    /// in the scale tensor before quantizing each element with that scale.
    ///
    /// @param x         Input tensor (high-precision data)
    /// @param y         Output tensor (quantized, same shape as x)
    /// @param scale     Per-block scale tensor
    /// @param blockSize Block size for each blocked dimension
    /// @param axis      Axis along which to apply block scaling (default: last dimension)
    template <class XDataType, class YDataType, class ScaleDataType, class ComputeDataType = float>
    static void quantize(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                         hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                         hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                         int32_t blockSize,
                         std::optional<int64_t> axis = std::nullopt)
    {
        // Validate allowed data types
        static_assert(IS_VALID_IO_TYPE_V<XDataType>,
                      "BlockScaleQuantize does not support this input data type.");
        static_assert(IS_VALID_IO_TYPE_V<YDataType>,
                      "BlockScaleQuantize does not support this output data type.");
        static_assert(IS_VALID_SCALE_TYPE_V<ScaleDataType>,
                      "BlockScaleQuantize does not support this scale data type.");
        static_assert(IS_VALID_COMPUTE_TYPE_V<ComputeDataType>,
                      "BlockScaleQuantize only supports float or double compute type.");

        const auto& xDims = x.dims();
        const auto& yDims = y.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.empty() || yDims.empty() || scaleDims.empty())
        {
            throw std::runtime_error("BlockScaleQuantize requires non-empty tensor dimensions.");
        }

        if(xDims != yDims)
        {
            throw std::invalid_argument("BlockScaleQuantize requires input and output tensors to "
                                        "have the same dimensions.");
        }

        if(blockSize <= 0)
        {
            throw std::invalid_argument("BlockScaleQuantize requires a positive block size.");
        }

        // Determine the target axis for block scaling (default to the last dimension)
        const auto targetAxisImpl = axis.value_or(static_cast<int64_t>(xDims.size()) - 1);
        if(targetAxisImpl < 0 || targetAxisImpl >= static_cast<int64_t>(xDims.size()))
        {
            throw std::out_of_range("BlockScaleQuantize: axis " + std::to_string(targetAxisImpl)
                                    + " is out of bounds for input tensor with rank "
                                    + std::to_string(xDims.size()) + ".");
        }
        const auto targetAxis = static_cast<size_t>(targetAxisImpl);

        // Scale dimensions should match the input dimensions, except for the target axis
        // which should be reduced by blockSize
        std::vector<int64_t> expectedScaleDims = xDims;
        expectedScaleDims[targetAxis] = (xDims[targetAxis] + blockSize - 1) / blockSize;
        if(scaleDims != expectedScaleDims)
        {
            throw std::invalid_argument("BlockScaleQuantize: scale tensor dimensions "
                                        + hipdnn_data_sdk::utilities::vecToString(scaleDims)
                                        + " do not match expected dimensions "
                                        + hipdnn_data_sdk::utilities::vecToString(expectedScaleDims)
                                        + " based on input tensor dimensions "
                                        + hipdnn_data_sdk::utilities::vecToString(xDims)
                                        + " and block size " + std::to_string(blockSize) + ".");
        }

        // Limit values for output data type
        const auto maxOutVal = static_cast<double>(std::numeric_limits<YDataType>::max());
        const auto minOutValCompute
            = static_cast<ComputeDataType>(std::numeric_limits<YDataType>::lowest());
        const auto maxOutValCompute = static_cast<ComputeDataType>(maxOutVal);

        // Check for Infs and NaNs in input tensor
        hipdnn_data_sdk::utilities::iterateAlongDimensions(
            xDims, [&](const std::vector<int64_t>& idx) {
                const auto xVal = static_cast<ComputeDataType>(x.getHostValue(idx));
                if(std::isnan(xVal) || std::isinf(xVal))
                {
                    throw std::runtime_error(
                        "BlockScaleQuantize: input tensor contains a NaN or Inf value.");
                }
            });

        auto quantizeFunc = [&](const std::vector<int64_t>& scaleIndices) {
            const auto blockStart = scaleIndices[targetAxis] * blockSize;
            const auto blockEnd = std::min(blockStart + blockSize, xDims[targetAxis]);

            // Find maximum absolute value in the block
            auto maxAbsVal = static_cast<ComputeDataType>(0.0);
            auto elementIndices = scaleIndices;
            for(int64_t i = blockStart; i < blockEnd; ++i)
            {
                elementIndices[targetAxis] = i;
                const auto absVal
                    = std::abs(static_cast<ComputeDataType>(x.getHostValue(elementIndices)));
                maxAbsVal = std::max(maxAbsVal, absVal);
            }

            const auto scaleVal
                = quantizeRoundUp<ScaleDataType>(static_cast<double>(maxAbsVal) / maxOutVal);
            scale.setHostValue(scaleVal, scaleIndices);

            // Quantize each element in the block
            const auto scaleValCompute = static_cast<ComputeDataType>(scaleVal);
            for(int64_t i = blockStart; i < blockEnd; ++i)
            {
                elementIndices[targetAxis] = i;
                const auto xVal = static_cast<ComputeDataType>(x.getHostValue(elementIndices));
                auto yVal = scaleValCompute != static_cast<ComputeDataType>(0.0)
                                ? xVal / scaleValCompute
                                : static_cast<ComputeDataType>(0.0);
                yVal = std::clamp(yVal, minOutValCompute, maxOutValCompute);

                if constexpr(std::is_integral_v<YDataType>)
                {
                    // yVal is already clamped to [minOutValCompute, maxOutValCompute] above, but for
                    // integral YDataType those bounds themselves may not be exactly representable in
                    // ComputeDataType (e.g. INT32_MAX rounds up to 2147483648.0f in float), so casting
                    // a saturated yVal directly to YDataType can still be UB. Here we are routing the
                    // saturated cases explicitly to the exact integer limits instead of directly
                    // casting the saturated yVal to YDataType.
                    YDataType clampedYVal;
                    if(yVal <= minOutValCompute)
                    {
                        clampedYVal = std::numeric_limits<YDataType>::lowest();
                    }
                    else if(yVal >= maxOutValCompute)
                    {
                        clampedYVal = std::numeric_limits<YDataType>::max();
                    }
                    else
                    {
                        clampedYVal = static_cast<YDataType>(std::nearbyint(yVal));
                    }

                    y.setHostValue(clampedYVal, elementIndices);
                }
                else
                {
                    // Casting already does an RNE under the hood for the allowed float IO types
                    y.setHostValue(static_cast<YDataType>(yVal), elementIndices);
                }
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(quantizeFunc, scale.dims());
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
        scale.memory().markHostModified();
    }

private:
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, float>, T> quantizeRoundUp(double val)
    {
        if(val > static_cast<double>(std::numeric_limits<T>::max()))
        {
            return std::numeric_limits<T>::max();
        }
        auto rounded = static_cast<T>(val);
        if(static_cast<double>(rounded) < val)
        {
            rounded = std::nextafter(rounded, std::numeric_limits<T>::max());
        }
        return rounded;
    }

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, hipdnn_data_sdk::types::half>
                                || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>
                                || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e8m0>,
                            T>
        quantizeRoundUp(double val)
    {
        if(val > static_cast<double>(std::numeric_limits<T>::max()))
        {
            return std::numeric_limits<T>::max();
        }

        T rounded(val);

        if(static_cast<double>(rounded) < val)
        {
            // Increment the value by one ULP to round up
            if(rounded.data < std::numeric_limits<T>::max().data)
            {
                rounded = T::from_bits(static_cast<decltype(rounded.data)>(rounded.data + 1));
            }
            else
            {
                rounded = std::numeric_limits<T>::max();
            }
        }

        return rounded;
    }
};

} // namespace hipdnn_test_sdk::utilities
