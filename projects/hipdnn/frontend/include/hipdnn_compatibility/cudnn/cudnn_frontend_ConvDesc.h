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
 * @brief cuDNN v8 `ConvDesc_v8` / `ConvDescBuilder_v8` over the shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

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
        if(m_convDesc.nDims <= 0)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS field");
            return std::move(m_convDesc);
        };
        if(m_convDesc.stride[0] <= 0)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES field");
            return std::move(m_convDesc);
        }

        auto status
            = m_convDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Bad descriptor created");
            return std::move(m_convDesc);
        }

        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_convDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_convDesc.compute_type, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
            return std::move(m_convDesc);
        }
        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
            return std::move(m_convDesc);
        }

        cudnnConvolutionMode_t mode_ = detail::convert_to_cudnn_type(m_convDesc.mode);

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       CUDNN_TYPE_CONVOLUTION_MODE,
                                       1,
                                       &mode_);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_CONV_MODE Failed");
            return std::move(m_convDesc);
        }

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_convDesc.nDims);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS Failed");
            return std::move(m_convDesc);
        }

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       CUDNN_TYPE_INT64,
                                       m_convDesc.nDims,
                                       m_convDesc.padLower);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                       CUDNN_TYPE_INT64,
                                       m_convDesc.nDims,
                                       m_convDesc.padUpper);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_POST_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                       CUDNN_TYPE_INT64,
                                       m_convDesc.nDims,
                                       m_convDesc.dilation);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_DILATIONS Failed");
            return std::move(m_convDesc);
        }

        status = detail::set_attribute(m_convDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                       CUDNN_TYPE_INT64,
                                       m_convDesc.nDims,
                                       m_convDesc.stride);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_convDesc,
                                          status,
                                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES Failed");
            return std::move(m_convDesc);
        }

        status = detail::finalize(m_convDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_convDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_convDesc);
        return std::move(m_convDesc);
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
