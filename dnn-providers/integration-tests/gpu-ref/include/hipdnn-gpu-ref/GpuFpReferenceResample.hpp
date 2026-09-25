// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename XDataType, typename YDataType, typename ComputeDataType, typename IndexDataType>
inline std::vector<std::string> buildResampleFwdDefines(bool hasIndex)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DX_TYPE=") + HipRtcTypeName<XDataType>::VALUE);
    defines.emplace_back(std::string("-DY_TYPE=") + HipRtcTypeName<YDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DINDEX_TYPE=") + HipRtcTypeName<IndexDataType>::VALUE);
    defines.emplace_back(std::string("-DHAS_INDEX=") + std::to_string(hasIndex ? 1 : 0));
    return defines;
}

} // namespace detail

class GpuFpReferenceResample
{
private:
    struct ResampleSpatialParams
    {
        std::vector<int64_t> prePadding;
        std::vector<int64_t> stride;
        std::vector<int64_t> window;
    };

    struct ResampleModeParams
    {
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode;
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode;
    };

public:
    static constexpr unsigned int BLOCK_SIZE = 256;

    // --- Forward resample ---

    template <class XDataType,
              class YDataType = XDataType,
              class ComputeDataType = float,
              class IndexDataType = int32_t>
    static void forward(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& window,
                        hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
                        hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
                        hipdnn_data_sdk::utilities::TensorBase<IndexDataType>* index = nullptr)
    {
        // Validate IO tensors, spatial parameters, and resample mode
        validateInput(x, y, {prePadding, stride, window}, resampleMode);

        // Validate consistency of spatial parameters with IO tensor dimensions
        const auto& xDims = x.dims();
        const auto& yDims = y.dims();
        const auto spatialDims = xDims.size() - 2;
        for(size_t i = 0; i < spatialDims; ++i)
        {
            const auto expectedYDim = (xDims[i + 2] + prePadding[i] - window[i]) / stride[i] + 1;
            if(expectedYDim <= 0 || yDims[i + 2] != expectedYDim)
            {
                throw std::runtime_error(
                    "Resample forward output shape mismatch for spatial dimension "
                    + std::to_string(i) + ". Expected: " + std::to_string(expectedYDim)
                    + ", Actual: " + std::to_string(yDims[i + 2]));
            }
        }

        // Validate index tensor
        if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
        {
            if(index == nullptr || index->dims() != y.dims() || index->strides() != y.strides())
            {
                throw std::runtime_error("Resample forward requires an index tensor matching the "
                                         "output tensor for MAXPOOL resample mode.");
            }
        }

        // Validate data types
        static_assert(
            IS_SUPPORTED_DATA_TYPE<XDataType, YDataType, ComputeDataType, IndexDataType>,
            "Resample forward supports only double, float, half, and bfloat16 data types for X, "
            "Y, and compute types, and int32_t for index type.");

        auto defines
            = detail::buildResampleFwdDefines<XDataType, YDataType, ComputeDataType, IndexDataType>(
                index != nullptr);

        launchForward(x.memory().deviceData(),
                      {x.dims().begin() + 2, x.dims().end()},
                      x.strides(),
                      y.memory().deviceData(),
                      {y.dims().begin() + 2, y.dims().end()},
                      y.strides(),
                      {x.dims()[0], x.dims()[1]},
                      {prePadding, stride, window},
                      {resampleMode, paddingMode},
                      defines,
                      index ? index->memory().deviceData() : nullptr);

        y.memory().markDeviceModified();
        if(index != nullptr)
        {
            index->memory().markDeviceModified();
        }
    }

private:
    // --- Validators ---

    template <typename T>
    using isSupportedFpType = std::disjunction<std::is_same<T, double>,
                                               std::is_same<T, float>,
                                               std::is_same<T, hipdnn_data_sdk::types::half>,
                                               std::is_same<T, hipdnn_data_sdk::types::bfloat16>>;

    template <typename InputDataType,
              typename OutputDataType,
              typename ComputeDataType,
              typename IndexDataType>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = isSupportedFpType<InputDataType>::value && isSupportedFpType<OutputDataType>::value
          && isSupportedFpType<ComputeDataType>::value && std::is_same_v<IndexDataType, int32_t>;

    template <typename InputDataType, typename OutputDataType>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                              const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                              const ResampleSpatialParams& spatialParams,
                              hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode)
    {
        const auto& inputDims = input.dims();
        const auto& inputStrides = input.strides();
        const auto& outputDims = output.dims();
        const auto& outputStrides = output.strides();

        // Validate IO tensor dimensions
        if(inputDims.size() < 4 || inputDims.size() > 5 || inputDims.size() != outputDims.size())
        {
            throw std::runtime_error("Resample supports matching 4D or 5D IO tensors.");
        }

        if(inputDims[0] != outputDims[0] || inputDims[1] != outputDims[1])
        {
            throw std::runtime_error(
                "Resample requires matching N and C dimensions for IO tensors.");
        }

        // Validate IO tensor layouts
        using hipdnn_data_sdk::utilities::TensorLayout;
        const std::pair<const std::vector<int64_t>&, const std::vector<int64_t>&> ioTensors[]
            = {{inputDims, inputStrides}, {outputDims, outputStrides}};
        for(const auto& [dims, strides] : ioTensors)
        {
            const auto nDims = dims.size();
            if(!hipdnn_data_sdk::utilities::isLayoutAgnostic(dims))
            {
                const auto [channelFirst, channelLast]
                    = (nDims == 4) ? std::make_pair(TensorLayout::NCHW, TensorLayout::NHWC)
                                   : std::make_pair(TensorLayout::NCDHW, TensorLayout::NDHWC);

                const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(strides);
                if(strideOrder != channelFirst.strideOrder
                   && strideOrder != channelLast.strideOrder)
                {
                    throw std::invalid_argument("Resample requires IO tensors to be in "
                                                + channelFirst.name + " or " + channelLast.name
                                                + " layout.");
                }
            }
        }

        // Validate spatial dimensions
        const auto spatialDims = inputDims.size() - 2;
        const auto validateSpatialParameter = [&spatialDims](const std::vector<int64_t>& param,
                                                             bool allowZero,
                                                             std::string_view name) {
            if(param.size() != spatialDims)
            {
                throw std::runtime_error("Resample spatial parameter " + std::string(name)
                                         + " must have rank equal to the tensor spatial rank.");
            }

            const bool invalid
                = allowZero
                      ? std::any_of(param.begin(), param.end(), [](int64_t v) { return v < 0; })
                      : std::any_of(param.begin(), param.end(), [](int64_t v) { return v <= 0; });

            if(invalid)
            {
                throw std::runtime_error("Resample spatial parameter " + std::string(name)
                                         + " must have " + (allowZero ? "non-negative" : "positive")
                                         + " values.");
            }
        };

        validateSpatialParameter(spatialParams.prePadding, true, "prePadding");
        validateSpatialParameter(spatialParams.stride, false, "stride");
        validateSpatialParameter(spatialParams.window, false, "window");

        // Validate resample and padding modes
        if(resampleMode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
        {
            throw std::runtime_error("Resample received an unsupported resample mode.");
        }
    }

    // --- Kernel launchers (defined in GpuFpReferenceResample.cpp) ---

    static void launchForward(const void* xPtr,
                              const std::vector<int64_t>& xSpatialDims,
                              const std::vector<int64_t>& xStrides,
                              void* yPtr,
                              const std::vector<int64_t>& ySpatialDims,
                              const std::vector<int64_t>& yStrides,
                              const std::array<int64_t, 2>& batchChannelDims,
                              const ResampleSpatialParams& spatialParams,
                              const ResampleModeParams& modeParams,
                              std::vector<std::string>& defines,
                              void* indexPtr = nullptr);
};

} // namespace hipdnn_gpu_ref
