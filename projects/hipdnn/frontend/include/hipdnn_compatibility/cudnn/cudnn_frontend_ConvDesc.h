// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_ConvDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_ConvDesc.h
 * @brief cuDNN v8 `ConvDesc_v8` / `ConvDescBuilder_v8`, built as a shim convolution record.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/build_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_ConvDesc.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class ConvDesc_v8 : public BackendDescriptor
{
public:
    friend class ConvDescBuilder_v8;
    template <typename Descriptor, typename MakeRecord>
    friend Descriptor&& detail::v8::buildDescriptor(Descriptor& descriptor,
                                                    cudnnBackendDescriptorType_t type,
                                                    MakeRecord&& makeRecord);
    std::string describe() const override
    {
        std::stringstream ss;
        char sep = ' ';
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR :" << " Datatype: "
           << detail::enumJson(compute_type) << " Mode: " << detail::enumJson(mode)
#else
        ss << "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR :" << " Datatype: "
           << static_cast<int>(compute_type) << " Mode: " << static_cast<int>(mode)
#endif
           << " Num Dimensions: " << nDims;
        ss << " PadLower [";
        for(auto i = 0; i < nDims; i++)
        {
            ss << sep << padLower[i];
            sep = ',';
        }
        ss << " ] PadUpper [";
        for(auto i = 0; i < nDims; i++)
        {
            ss << sep << padUpper[i];
            sep = ',';
        }
        ss << " ] Dilation [";
        for(auto i = 0; i < nDims; i++)
        {
            ss << sep << dilation[i];
            sep = ',';
        }
        ss << " ] Stride [";
        for(auto i = 0; i < nDims; i++)
        {
            ss << sep << stride[i];
            sep = ',';
        }
        ss << "]";
        return ss.str();
    }

    ConvDesc_v8(ConvDesc_v8&& from) = default;
    ConvDesc_v8& operator=(ConvDesc_v8&&) = default;

    ~ConvDesc_v8() = default;

    DataType_t getComputePrecision() const
    {
        return compute_type;
    }

    DataType_t getComputeType() const
    {
        return compute_type;
    }

    int64_t getSpatialDimCount() const
    {
        return nDims;
    }

    const int64_t* getPadding() const
    {
        return padLower;
    }

    const int64_t* getSpatialStride() const
    {
        return stride;
    }

    const int64_t* getDilation() const
    {
        return dilation;
    }

    cudnnConvolutionMode_t getMathMode() const
    {
        return detail::convert_to_cudnn_type(mode);
    }

    int64_t getDimensionCount() const
    {
        return getSpatialDimCount();
    }

    const int64_t* getStride() const
    {
        return getSpatialStride();
    }

private:
    ConvDesc_v8() = default;
    ConvDesc_v8(const ConvDesc_v8&) = delete;
    ConvDesc_v8& operator=(const ConvDesc_v8&) = delete;

    DataType_t compute_type = DataType_t::FLOAT;
    ConvolutionMode_t mode = ConvolutionMode_t::CONVOLUTION;
    int64_t nDims = -1;
    int64_t padLower[CUDNN_DIM_MAX + 1] = {0};
    int64_t padUpper[CUDNN_DIM_MAX + 1] = {0};
    int64_t dilation[CUDNN_DIM_MAX + 1] = {0};
    int64_t stride[CUDNN_DIM_MAX + 1] = {-1};
};

class ConvDescBuilder_v8
{
public:
    auto setComputeType(DataType_t data_type) -> ConvDescBuilder_v8&
    {
        m_convDesc.compute_type = data_type;
        return *this;
    }
    auto setComputeType(cudnnDataType_t data_type_) -> ConvDescBuilder_v8&
    {
        m_convDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }
    auto setPrePadding(int64_t ndims, const int64_t* padding) -> ConvDescBuilder_v8&
    {
        std::copy(padding, padding + ndims, m_convDesc.padLower);
        return *this;
    }
    auto setPostPadding(int64_t ndims, const int64_t* padding) -> ConvDescBuilder_v8&
    {
        std::copy(padding, padding + ndims, m_convDesc.padUpper);
        return *this;
    }
    auto setDilation(int64_t ndims, const int64_t* dilation) -> ConvDescBuilder_v8&
    {
        std::copy(dilation, dilation + ndims, m_convDesc.dilation);
        return *this;
    }
    auto setSpatialStride(int64_t ndims, const int64_t* strides) -> ConvDescBuilder_v8&
    {
        std::copy(strides, strides + ndims, m_convDesc.stride);
        return *this;
    }
    auto setSpatialDimCount(int64_t nDims_) -> ConvDescBuilder_v8&
    {
        m_convDesc.nDims = nDims_;
        return *this;
    }
    auto setMathMode(ConvolutionMode_t mode_) -> ConvDescBuilder_v8&
    {
        m_convDesc.mode = mode_;
        return *this;
    }

    auto setMathMode(cudnnConvolutionMode_t mode_) -> ConvDescBuilder_v8&
    {
        m_convDesc.mode = detail::convert_from_cudnn_type(mode_);
        return *this;
    }

    auto setNDims(int64_t nDims_) -> ConvDescBuilder_v8&
    {
        return setSpatialDimCount(nDims_);
    }
    auto setDataType(cudnnDataType_t data_type_) -> ConvDescBuilder_v8&
    {
        return setComputeType(data_type_);
    }
    auto setComputePrecision(cudnnDataType_t data_type_) -> ConvDescBuilder_v8&
    {
        return setComputeType(data_type_);
    }
    auto setStrides(int64_t ndims, const int64_t* strides) -> ConvDescBuilder_v8&
    {
        return setSpatialStride(ndims, strides);
    }

    ConvDesc_v8&& build()
    {
        static const detail::v8::ConvolutionLabels s_labels{
            "ConvDescBuilder: compute type",
            "ConvDescBuilder: mode",
            "ConvDescBuilder: spatial dimension count",
            "ConvDescBuilder: pre-padding",
            "ConvDescBuilder: post-padding",
            "ConvDescBuilder: dilation",
            "ConvDescBuilder: spatial stride",
        };
        return detail::v8::buildDescriptor(m_convDesc, CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, [&] {
            if(m_convDesc.nDims <= 0)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "ConvDescBuilder: spatial dimension count must be positive");
            }
            if(m_convDesc.stride[0] <= 0)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "ConvDescBuilder: spatial strides must be set and positive");
            }

            cudnnDataType_t cudnn_data_type;
            auto status = detail::convert_to_cudnn_type(m_convDesc.compute_type, cudnn_data_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status, "ConvDescBuilder: compute type is not set or not a valid DataType_t");
            }

            cudnnConvolutionMode_t mode_ = detail::convert_to_cudnn_type(m_convDesc.mode);

            // A count above the arrays' capacity no longer matches nDims, so the core rejects it.
            const auto count
                = static_cast<std::size_t>(std::min<int64_t>(m_convDesc.nDims, CUDNN_DIM_MAX + 1));
            detail::v8::ConvolutionInputs inputs;
            inputs.computeType = cudnn_data_type;
            inputs.mode = mode_;
            inputs.spatialDims = m_convDesc.nDims;
            inputs.prePaddings.assign(m_convDesc.padLower, m_convDesc.padLower + count);
            inputs.postPaddings.assign(m_convDesc.padUpper, m_convDesc.padUpper + count);
            inputs.dilations.assign(m_convDesc.dilation, m_convDesc.dilation + count);
            inputs.filterStrides.assign(m_convDesc.stride, m_convDesc.stride + count);
            return detail::v8::makeConvRecord(std::move(inputs), s_labels);
        });
    }

    explicit ConvDescBuilder_v8() = default;
    ~ConvDescBuilder_v8() = default;
    ConvDescBuilder_v8(ConvDescBuilder_v8&&) = delete;
    ConvDescBuilder_v8(const ConvDescBuilder_v8&) = delete;
    ConvDescBuilder_v8& operator=(const ConvDescBuilder_v8&) = delete;

private:
    ConvDesc_v8 m_convDesc;
};

using ConvDesc = ConvDesc_v8;
using ConvDescBuilder = ConvDescBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
