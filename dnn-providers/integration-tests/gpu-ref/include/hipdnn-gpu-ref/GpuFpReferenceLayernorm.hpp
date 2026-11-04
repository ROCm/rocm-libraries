// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn-gpu-ref/detail/HipRtcTypeName.hpp"

#include <cstddef>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_gpu_ref
{

using namespace hipdnn_data_sdk::utilities;

namespace detail
{

inline void getLayernormDimensions(int64_t& outerSize,
                                   int64_t& innerSize,
                                   int64_t& stride,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides,
                                   const int64_t normalizedDimCount)
{
    outerSize = 1;
    innerSize = 1;
    stride = 1;
    auto strideOrder = extractStrideOrder(strides);
    const int64_t normalizedDim = static_cast<int64_t>(dims.size()) - normalizedDimCount;

    // Set stride to C if there is a stride between blocks of inner size (i.e. C not part of normalized dimensions and channel-last layout)
    if(normalizedDim > 1
       && (strideOrder == TensorLayout::NHWC.strideOrder
           || strideOrder == TensorLayout::NDHWC.strideOrder))
    {
        stride = dims[1];
    }

    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i < static_cast<size_t>(normalizedDim))
        {
            // Don't add C to outerSize if there is a stride
            if(stride == 1 || i != 1)
            {
                outerSize *= dims[i];
            }
        }
        else
        {
            innerSize *= dims[i];
        }
    }
}

template <typename XDataType,
          typename YDataType,
          typename ScaleBiasDataType,
          typename MeanRstdDataType,
          typename ComputeDataType,
          unsigned int LocalSize>
inline std::vector<std::string> buildLayernormDefines(const std::vector<int64_t>& dims,
                                                      const std::vector<int64_t>& strides,
                                                      const int64_t normalizedDimCount)
{
    int64_t outerSize;
    int64_t innerSize;
    int64_t stride;
    getLayernormDimensions(outerSize, innerSize, stride, dims, strides, normalizedDimCount);

    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DX_TYPE=") + HipRtcTypeName<XDataType>::VALUE);
    defines.emplace_back(std::string("-DY_TYPE=") + HipRtcTypeName<YDataType>::VALUE);
    defines.emplace_back(std::string("-DSCALE_BIAS_TYPE=")
                         + HipRtcTypeName<ScaleBiasDataType>::VALUE);
    defines.emplace_back(std::string("-DMEAN_RSTD_TYPE=")
                         + HipRtcTypeName<MeanRstdDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(LocalSize));
    defines.emplace_back(std::string("-DOUTER_SIZE=") + std::to_string(outerSize));
    defines.emplace_back(std::string("-DINNER_SIZE=") + std::to_string(innerSize));
    defines.emplace_back(std::string("-DSTRIDE=") + std::to_string(stride));
    return defines;
}

} // namespace detail

class GpuFpReferenceLayernorm
{
public:
    static constexpr unsigned int LOCAL_SIZE = 1024;

    // Layernorm forward propagation
    template <typename XDataType,
              typename ScaleBiasDataType,
              typename YDataType = XDataType,
              typename MeanRstdDataType = ScaleBiasDataType,
              typename ComputeDataType = float>
    static void fprop(TensorBase<XDataType>& x,
                      TensorBase<ScaleBiasDataType>* scale,
                      TensorBase<ScaleBiasDataType>* bias,
                      TensorBase<YDataType>& y,
                      const double epsilon,
                      const int64_t normalizedDimCount,
                      TensorBase<MeanRstdDataType>* mean = nullptr,
                      TensorBase<MeanRstdDataType>* rstd = nullptr)
    {
        validateFwd(x, scale, bias, y, normalizedDimCount, mean, rstd);

        auto defines
            = detail::buildLayernormDefines<XDataType,
                                            YDataType,
                                            ScaleBiasDataType,
                                            MeanRstdDataType,
                                            ComputeDataType,
                                            LOCAL_SIZE>(x.dims(), x.strides(), normalizedDimCount);

        launchFprop(x.rawDeviceData(),
                    x.dims(),
                    x.strides(),
                    scale == nullptr ? nullptr : scale->rawDeviceData(),
                    bias == nullptr ? nullptr : bias->rawDeviceData(),
                    y.rawDeviceData(),
                    mean == nullptr ? nullptr : mean->rawDeviceData(),
                    rstd == nullptr ? nullptr : rstd->rawDeviceData(),
                    normalizedDimCount,
                    defines,
                    LOCAL_SIZE,
                    epsilon);

        y.markDeviceModified();
        if(mean != nullptr)
        {
            mean->markDeviceModified();
        }
        if(rstd != nullptr)
        {
            rstd->markDeviceModified();
        }
    }

    // Layernorm backward propagation
    template <typename YDataType,
              typename ScaleBiasDataType,
              typename XDataType = YDataType,
              typename MeanRstdDataType = ScaleBiasDataType,
              typename ComputeDataType = float>
    static void bprop(TensorBase<YDataType>& dy,
                      TensorBase<XDataType>& x,
                      TensorBase<ScaleBiasDataType>& scale,
                      TensorBase<XDataType>& dx,
                      TensorBase<ScaleBiasDataType>& dscale,
                      TensorBase<ScaleBiasDataType>& dbias,
                      const double epsilon,
                      TensorBase<MeanRstdDataType>* mean,
                      TensorBase<MeanRstdDataType>* rstd,
                      const int64_t normalizedDimCount)
    {
        validateBwd(dy, x, scale, dx, dscale, dbias, mean, rstd, normalizedDimCount);

        auto defines = detail::buildLayernormDefines<XDataType,
                                                     YDataType,
                                                     ScaleBiasDataType,
                                                     MeanRstdDataType,
                                                     ComputeDataType,
                                                     LOCAL_SIZE>(
            dy.dims(), dy.strides(), normalizedDimCount);

        std::shared_ptr<Workspace<>> workspace;
        if(mean == nullptr || rstd == nullptr)
        {
            int64_t outerSize;
            int64_t innerSize;
            int64_t stride;
            detail::getLayernormDimensions(
                outerSize, innerSize, stride, dy.dims(), dy.strides(), normalizedDimCount);
            const int64_t workspaceSize = 2 * outerSize * stride;
            workspace = std::make_shared<Workspace<>>(sizeof(ComputeDataType)
                                                      * static_cast<size_t>(workspaceSize));
        }

        launchBprop(dy.rawDeviceData(),
                    dy.dims(),
                    dy.strides(),
                    x.rawDeviceData(),
                    scale.rawDeviceData(),
                    dx.rawDeviceData(),
                    dscale.rawDeviceData(),
                    dbias.rawDeviceData(),
                    mean == nullptr ? nullptr : mean->rawDeviceData(),
                    rstd == nullptr ? nullptr : rstd->rawDeviceData(),
                    mean == nullptr || rstd == nullptr ? workspace->get() : nullptr,
                    normalizedDimCount,
                    defines,
                    LOCAL_SIZE,
                    epsilon);

        dx.markDeviceModified();
        dscale.markDeviceModified();
        dbias.markDeviceModified();
    }

private:
    // --- Validators ---

    struct TensorProps
    {
        std::string name;
        const std::vector<int64_t>* dims;
        const std::vector<int64_t>* strides;

        TensorProps(std::string n, const std::vector<int64_t>* d, const std::vector<int64_t>* s)
            : name(std::move(n))
            , dims(d)
            , strides(s)
        {
        }
    };

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, float> || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    template <class T>
    static constexpr bool IS_SUPPORTED_COMPUTE_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    static void validateConsistentDimensions(const std::vector<TensorProps>& ioTensorProps,
                                             const std::vector<TensorProps>& affineTensorProps,
                                             const std::vector<TensorProps>& statTensorProps,
                                             const int64_t normalizedDimCount)
    {
        const TensorProps* refTensorProps = nullptr;
        for(const auto& props : ioTensorProps)
        {
            if(props.dims != nullptr)
            {
                refTensorProps = &props;
                break;
            }
        }
        if(refTensorProps == nullptr)
        {
            throw std::runtime_error("At least one input tensor is needed for validation.");
        }

        const auto nDims = refTensorProps->dims->size();
        const auto normalizedDim = static_cast<int64_t>(nDims) - normalizedDimCount;

        // Validate input tensor rank
        if(nDims < 4 || nDims > 5)
        {
            throw std::invalid_argument("Layernorm requires input tensor rank to be 4 or 5.");
        }

        // Validate all I/O tensors have the same rank and shape as the input tensor
        for(const auto& [name, dims, strides] : ioTensorProps)
        {
            if(dims == nullptr)
            {
                continue;
            }
            if(dims->size() != nDims)
            {
                throw std::invalid_argument("Layernorm requires " + name
                                            + " tensor rank to be equal to " + refTensorProps->name
                                            + " tensor rank.");
            }
            if(*dims != *refTensorProps->dims)
            {
                throw std::invalid_argument("Layernorm requires " + name + " and "
                                            + refTensorProps->name
                                            + " tensors to have the same shape.");
            }
        }

        // Validate all affine tensors have the same rank as the input tensor
        for(const auto& [name, dims, strides] : affineTensorProps)
        {
            if(dims == nullptr)
            {
                continue;
            }
            if(dims->size() != nDims)
            {
                throw std::invalid_argument("Layernorm requires " + name
                                            + " tensor rank to be equal to " + refTensorProps->name
                                            + " tensor rank.");
            }
        }

        // Validate all stat tensors have the same rank as the input tensor
        for(const auto& [name, dims, strides] : statTensorProps)
        {
            if(dims == nullptr)
            {
                continue;
            }
            if(dims->size() != nDims)
            {
                throw std::invalid_argument("Layernorm requires " + name
                                            + " tensor rank to be equal to " + refTensorProps->name
                                            + " tensor rank.");
            }
        }

        // Find the first available affine tensor to use as the shape reference for the rest
        const TensorProps* refAffineTensorProps = nullptr;
        for(const auto& props : affineTensorProps)
        {
            if(props.dims != nullptr)
            {
                refAffineTensorProps = &props;
                break;
            }
        }

        // Validate all affine tensors have the same shape as the first affine tensor
        if(refAffineTensorProps != nullptr)
        {
            for(const auto& [name, dims, strides] : affineTensorProps)
            {
                if(dims == nullptr || dims == refAffineTensorProps->dims)
                {
                    continue;
                }
                if(*dims != *refAffineTensorProps->dims)
                {
                    throw std::invalid_argument("Layernorm requires " + name + " and "
                                                + refAffineTensorProps->name
                                                + " tensors to have the same shape.");
                }
            }

            // Validate affine tensor dimensions have 1s in the leading dimensions
            const auto& affineDims = *refAffineTensorProps->dims;
            if(!std::all_of(affineDims.begin(), affineDims.begin() + normalizedDim, [](int64_t d) {
                   return d == 1;
               }))
            {
                throw std::invalid_argument("Layernorm requires affine tensor dimensions to have "
                                            "1s in the leading dimensions.");
            }

            // Validate affine tensor dimensions are equal to the input tensor in the trailing dimensions
            if(std::vector<int64_t>(affineDims.begin() + normalizedDim, affineDims.end())
               != std::vector<int64_t>(refTensorProps->dims->begin() + normalizedDim,
                                       refTensorProps->dims->end()))
            {
                throw std::invalid_argument(
                    "Layernorm requires affine tensor dimensions to be equal to the "
                    + refTensorProps->name + " tensor in the trailing dimensions.");
            }
        }

        // Find the first available stat tensor to use as the shape reference for the rest
        const TensorProps* refStatTensorProps = nullptr;
        for(const auto& props : statTensorProps)
        {
            if(props.dims != nullptr)
            {
                refStatTensorProps = &props;
                break;
            }
        }

        // Validate all stat tensors have the same shape as the first stat tensor
        if(refStatTensorProps != nullptr)
        {
            for(const auto& [name, dims, strides] : statTensorProps)
            {
                if(dims == nullptr || dims == refStatTensorProps->dims)
                {
                    continue;
                }
                if(*dims != *refStatTensorProps->dims)
                {
                    throw std::invalid_argument("Layernorm requires " + name + " and "
                                                + refStatTensorProps->name
                                                + " tensors to have the same shape.");
                }
            }

            // Validate stat tensors have 1s in the trailing dimensions
            const auto& statDims = *refStatTensorProps->dims;
            if(!std::all_of(statDims.begin() + normalizedDim, statDims.end(), [](int64_t d) {
                   return d == 1;
               }))
            {
                throw std::invalid_argument("Layernorm requires stat tensor dimensions to have 1s "
                                            "in the trailing dimensions.");
            }

            // Validate affine tensor dimensions are equal to the input tensor in the leading dimensions
            if(std::vector<int64_t>(statDims.begin(), statDims.begin() + normalizedDim)
               != std::vector<int64_t>(refTensorProps->dims->begin(),
                                       refTensorProps->dims->begin() + normalizedDim))
            {
                throw std::invalid_argument(
                    "Layernorm requires stat tensor dimensions to be equal to the "
                    + refTensorProps->name + " tensor in the leading dimensions.");
            }
        }
    }

    static void validateConsistentLayouts(const std::vector<TensorProps>& ioTensorProps,
                                          const std::vector<TensorProps>& affineTensorProps,
                                          const std::vector<TensorProps>& statTensorProps)
    {
        std::vector<TensorProps> tensorProps(ioTensorProps.begin(), ioTensorProps.end());
        tensorProps.insert(tensorProps.end(), affineTensorProps.begin(), affineTensorProps.end());
        tensorProps.insert(tensorProps.end(), statTensorProps.begin(), statTensorProps.end());

        const TensorProps* refTensorProps = nullptr;
        for(const auto& props : ioTensorProps)
        {
            if(props.dims != nullptr)
            {
                refTensorProps = &props;
                break;
            }
        }
        if(refTensorProps == nullptr)
        {
            throw std::runtime_error("At least one input tensor is needed for validation.");
        }

        const auto nDims = refTensorProps->dims->size();
        const auto inputStrideOrder = extractStrideOrder(*refTensorProps->strides);

        // Validate reference tensor layout
        static const std::unordered_map<size_t, std::pair<TensorLayout, TensorLayout>>
            s_validLayouts = {{4, {TensorLayout::NCHW, TensorLayout::NHWC}},
                              {5, {TensorLayout::NCDHW, TensorLayout::NDHWC}}};

        const auto it = s_validLayouts.find(nDims);
        if(it == s_validLayouts.end())
        {
            throw std::invalid_argument("Layernorm requires input tensor rank to be 4 or 5.");
        }

        const auto& [channelFirst, channelLast] = it->second;
        if(inputStrideOrder != channelFirst.strideOrder
           && inputStrideOrder != channelLast.strideOrder)
        {
            throw std::invalid_argument("Layernorm requires " + std::to_string(nDims)
                                        + "D IO tensor to be in " + channelFirst.name + " or "
                                        + channelLast.name + " layout.");
        }

        // Validate all I/O tensor layouts are consistent with the reference tensor layout
        for(const auto& [name, dims, strides] : ioTensorProps)
        {
            if(dims == nullptr || strides == nullptr)
            {
                continue;
            }

            if(!isLayoutAgnostic(*dims) && extractStrideOrder(*strides) != inputStrideOrder)
            {
                throw std::invalid_argument("Layernorm requires " + name
                                            + " tensor layout to be consistent with "
                                            + refTensorProps->name + " tensor layout.");
            }
        }

        const TensorProps* refAffineTensorProps = nullptr;
        for(const auto& props : affineTensorProps)
        {
            if(props.dims != nullptr)
            {
                refAffineTensorProps = &props;
                break;
            }
        }

        if(refAffineTensorProps != nullptr)
        {
            for(const auto& [name, dims, strides] : affineTensorProps)
            {
                if(*refAffineTensorProps->strides != *strides)
                {
                    throw std::invalid_argument("Layernorm requires " + name
                                                + " tensor strides to be consistent with "
                                                + refAffineTensorProps->name + " tensor strides.");
                }
            }
        }

        const TensorProps* refStatTensorProps = nullptr;
        for(const auto& props : statTensorProps)
        {
            if(props.dims != nullptr)
            {
                refStatTensorProps = &props;
                break;
            }
        }

        if(refStatTensorProps != nullptr)
        {
            for(const auto& [name, dims, strides] : statTensorProps)
            {
                if(*refStatTensorProps->strides != *strides)
                {
                    throw std::invalid_argument("Layernorm requires " + name
                                                + " tensor strides to be consistent with "
                                                + refStatTensorProps->name + " tensor strides.");
                }
            }
        }
    }

    template <typename XDataType,
              typename YDataType,
              typename ScaleBiasDataType,
              typename MeanRstdDataType>
    static void validateFwd(const TensorBase<XDataType>& x,
                            const TensorBase<ScaleBiasDataType>* scale,
                            const TensorBase<ScaleBiasDataType>* bias,
                            TensorBase<YDataType>& y,
                            const int64_t normalizedDimCount,
                            TensorBase<MeanRstdDataType>* mean,
                            TensorBase<MeanRstdDataType>* rstd)
    {
        const auto& xDims = x.dims();
        const auto* scaleDims = scale ? &scale->dims() : nullptr;
        const auto* biasDims = bias ? &bias->dims() : nullptr;
        const auto& yDims = y.dims();
        const auto* meanDims = mean ? &mean->dims() : nullptr;
        const auto* rstdDims = rstd ? &rstd->dims() : nullptr;

        // Validate tensor dimensions
        std::vector<TensorProps> ioTensorProps;
        ioTensorProps.emplace_back("x", &xDims, &x.strides());
        ioTensorProps.emplace_back("y", &yDims, &y.strides());

        std::vector<TensorProps> affineTensorProps;
        if(scaleDims != nullptr)
        {
            affineTensorProps.emplace_back("scale", scaleDims, &scale->strides());
        }
        if(biasDims != nullptr)
        {
            affineTensorProps.emplace_back("bias", biasDims, &bias->strides());
        }

        std::vector<TensorProps> statTensorProps;
        if(meanDims != nullptr)
        {
            statTensorProps.emplace_back("mean", meanDims, &mean->strides());
        }
        if(rstdDims != nullptr)
        {
            statTensorProps.emplace_back("rstd", rstdDims, &rstd->strides());
        }

        validateConsistentDimensions(
            ioTensorProps, affineTensorProps, statTensorProps, normalizedDimCount);
        validateConsistentLayouts(ioTensorProps, affineTensorProps, statTensorProps);

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<XDataType>,
                      "Layernorm forward supports only float, half, and bfloat16 x data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<YDataType>,
                      "Layernorm forward supports only float, half, and bfloat16 y data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ScaleBiasDataType>,
            "Layernorm forward supports only float, half, and bfloat16 scale/bias data types.");
        static_assert(IS_SUPPORTED_COMPUTE_DATA_TYPE<MeanRstdDataType>,
                      "Layernorm forward supports only double, float, half, and bfloat16 mean/rstd "
                      "data types.");
    }

    template <typename XDataType,
              typename YDataType,
              typename ScaleBiasDataType,
              typename MeanRstdDataType>
    static void validateBwd(const TensorBase<YDataType>& dy,
                            const TensorBase<XDataType>& x,
                            const TensorBase<ScaleBiasDataType>& scale,
                            TensorBase<XDataType>& dx,
                            TensorBase<ScaleBiasDataType>& dscale,
                            TensorBase<ScaleBiasDataType>& dbias,
                            TensorBase<MeanRstdDataType>* mean,
                            TensorBase<MeanRstdDataType>* rstd,
                            const int64_t normalizedDimCount)
    {
        const auto& dyDims = dy.dims();
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();
        const auto& dxDims = dx.dims();
        const auto& dscaleDims = dscale.dims();
        const auto& dbiasDims = dbias.dims();
        const auto* meanDims = mean ? &mean->dims() : nullptr;
        const auto* rstdDims = rstd ? &rstd->dims() : nullptr;

        // Validate tensor dimensions
        std::vector<TensorProps> ioTensorProps;
        ioTensorProps.emplace_back("dy", &dyDims, &dy.strides());
        ioTensorProps.emplace_back("x", &xDims, &x.strides());
        ioTensorProps.emplace_back("dx", &dxDims, &dx.strides());

        std::vector<TensorProps> affineTensorProps;
        affineTensorProps.emplace_back("scale", &scaleDims, &scale.strides());
        affineTensorProps.emplace_back("dscale", &dscaleDims, &dscale.strides());
        affineTensorProps.emplace_back("dbias", &dbiasDims, &dbias.strides());

        std::vector<TensorProps> statTensorProps;
        if(meanDims != nullptr)
        {
            statTensorProps.emplace_back("mean", meanDims, &mean->strides());
        }
        if(rstdDims != nullptr)
        {
            statTensorProps.emplace_back("rstd", rstdDims, &rstd->strides());
        }

        validateConsistentDimensions(
            ioTensorProps, affineTensorProps, statTensorProps, normalizedDimCount);
        validateConsistentLayouts(ioTensorProps, affineTensorProps, statTensorProps);

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<XDataType>,
                      "Layernorm backward supports only float, half, and bfloat16 x data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<YDataType>,
                      "Layernorm backward supports only float, half, and bfloat16 y data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ScaleBiasDataType>,
            "Layernorm backward supports only float, half, and bfloat16 scale/bias data types.");
        static_assert(
            IS_SUPPORTED_COMPUTE_DATA_TYPE<MeanRstdDataType>,
            "Layernorm backward supports only double, float, half, and bfloat16 mean/rstd "
            "data types.");
    }

    // --- Kernel launchers (defined in GpuFpReferenceLayernorm.cpp) ---

    static void launchFprop(const void* xPtr,
                            const std::vector<int64_t>& xDims,
                            const std::vector<int64_t>& xStrides,
                            const void* scalePtr,
                            const void* biasPtr,
                            void* yPtr,
                            void* meanPtr,
                            void* rstdPtr,
                            int64_t normalizedDimCount,
                            const std::vector<std::string>& defines,
                            int64_t localSize,
                            double epsilon = 1e-5);

    static void launchBprop(const void* dyPtr,
                            const std::vector<int64_t>& dyDims,
                            const std::vector<int64_t>& dyStrides,
                            const void* xPtr,
                            const void* scalePtr,
                            void* dxPtr,
                            void* dscalePtr,
                            void* dbiasPtr,
                            const void* meanPtr,
                            const void* rstdPtr,
                            void* workspace,
                            int64_t normalizedDimCount,
                            const std::vector<std::string>& defines,
                            int64_t localSize,
                            double epsilon = 1e-5);
};

} // namespace hipdnn_gpu_ref
