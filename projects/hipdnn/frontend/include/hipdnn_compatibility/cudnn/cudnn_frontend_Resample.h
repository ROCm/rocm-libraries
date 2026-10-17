// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Resample.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Resample.h
 * @brief cuDNN FE v0.x `ResampleDesc_v8` and its builder, on the shim's backend C API.
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
              "include/cudnn_frontend_Resample.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,misc-const-correctness,modernize-avoid-c-arrays):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class ResampleDesc_v8 : public BackendDescriptor
{
public:
    friend class ResampleDescBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        char sep = ',';
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: " << "Compute Type: "
           << detail::enumJson(computeType)
           << ", Resample Mode: " << detail::enumJson(resample_mode)
           << ", Spatial Dimensions: " << spatialDim
           << ", Nan Propagation: " << std::to_string(nanOpt)
           << ", Padding Mode: " << detail::enumJson(padding_mode);
#else
        ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: " << "Compute Type: "
           << static_cast<int>(computeType)
           << ", Resample Mode: " << static_cast<int>(resample_mode)
           << ", Spatial Dimensions: " << spatialDim
           << ", Nan Propagation: " << std::to_string(nanOpt)
           << ", Padding Mode: " << static_cast<int>(padding_mode);
#endif

        ss << ", WindowDim: [";
        for(auto i = 0; i < spatialDim; i++)
        {
            ss << '(' << windowDim[i].numerator << sep << windowDim[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", prePadding: [";
        for(auto i = 0; i < spatialDim; i++)
        {
            ss << '(' << prePadding[i].numerator << sep << prePadding[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", postPadding: [";
        for(auto i = 0; i < spatialDim; i++)
        {
            ss << '(' << postPadding[i].numerator << sep << postPadding[i].denominator << ')'
               << sep;
        }
        ss << "]";
        ss << ", stride: [ ";
        for(auto i = 0; i < spatialDim; i++)
        {
            ss << '(' << stride[i].numerator << sep << stride[i].denominator << ')' << sep;
        }
        ss << "]";
        return ss.str();
    }

    ResampleDesc_v8(ResampleDesc_v8&& from) = default;
    ResampleDesc_v8& operator=(ResampleDesc_v8&&) = default;

    ~ResampleDesc_v8() = default;

    DataType_t getComputeType() const
    {
        return computeType;
    }

    int64_t getSpatialDimCount() const
    {
        return spatialDim;
    }

    cudnnNanPropagation_t getNanOpt() const
    {
        return nanOpt;
    }

    ResampleMode_t getMode() const
    {
        return resample_mode;
    }

    PaddingMode_t getPaddingMode() const
    {
        return padding_mode;
    }

    const cudnnFraction_t* getSpatialStride() const
    {
        return stride;
    }

    const cudnnFraction_t* getPrePadding() const
    {
        return prePadding;
    }

    const cudnnFraction_t* getPostPadding() const
    {
        return postPadding;
    }

    const cudnnFraction_t* getWindowDim() const
    {
        return windowDim;
    }

private:
    ResampleDesc_v8() = default;
    ResampleDesc_v8(const ResampleDesc_v8&) = delete;
    ResampleDesc_v8& operator=(const ResampleDesc_v8&) = delete;

    DataType_t computeType = DataType_t::FLOAT;
    cudnnNanPropagation_t nanOpt = CUDNN_PROPAGATE_NAN;
    ResampleMode_t resample_mode = ResampleMode_t::NOT_SET;
    PaddingMode_t padding_mode = PaddingMode_t::NOT_SET;

    int64_t spatialDim = 0;

    cudnnFraction_t windowDim[CUDNN_DIM_MAX] = {{0, 1}, {0, 1}};
    cudnnFraction_t prePadding[CUDNN_DIM_MAX] = {{0, 1}, {0, 1}};
    cudnnFraction_t postPadding[CUDNN_DIM_MAX] = {{0, 1}, {0, 1}};
    cudnnFraction_t stride[CUDNN_DIM_MAX] = {{0, 1}, {0, 1}};
};

class ResampleDescBuilder_v8
{
public:
    auto setComputeType(DataType_t data_type) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.computeType = data_type;
        return *this;
    }
    auto setComputeType(cudnnDataType_t data_type_) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.computeType = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }

    auto setNanPropagation(cudnnNanPropagation_t nanOpt_) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.nanOpt = nanOpt_;
        return *this;
    }

    auto setPostPadding(int64_t count, const cudnnFraction_t* arr) -> ResampleDescBuilder_v8&
    {
        std::copy(arr, arr + count, m_resampleDesc.postPadding);
        return *this;
    }

    auto setPrePadding(int64_t count, const cudnnFraction_t* arr) -> ResampleDescBuilder_v8&
    {
        std::copy(arr, arr + count, m_resampleDesc.prePadding);
        return *this;
    }

    auto setSpatialStride(int64_t count, const cudnnFraction_t* arr) -> ResampleDescBuilder_v8&
    {
        std::copy(arr, arr + count, m_resampleDesc.stride);
        return *this;
    }

    auto setResampleMode(const cudnnResampleMode_t mode_) -> ResampleDescBuilder_v8&
    {
        detail::convert_from_cudnn_type(mode_, m_resampleDesc.resample_mode);
        return *this;
    }

    auto setSpatialDim(int64_t count, const cudnnFraction_t* arr) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.spatialDim = count;
        std::copy(arr, arr + count, m_resampleDesc.windowDim);
        return *this;
    }

    auto setPaddingMode(const cudnnPaddingMode_t padding_mode) -> ResampleDescBuilder_v8&
    {
        detail::convert_from_cudnn_type(padding_mode, m_resampleDesc.padding_mode);
        return *this;
    }

    auto setPaddingMode(const PaddingMode_t padding_mode) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.padding_mode = padding_mode;
        return *this;
    }

    auto setResampleMode(const ResampleMode_t mode) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.resample_mode = mode;
        return *this;
    }

    auto setPostPadding(int64_t count, const int64_t* arr) -> ResampleDescBuilder_v8&
    {
        for(int i = 0; i < count; i++)
        {
            m_resampleDesc.postPadding[i].numerator = arr[i];
            m_resampleDesc.postPadding[i].denominator = 1;
        }
        return *this;
    }

    auto setPrePadding(int64_t count, const int64_t* arr) -> ResampleDescBuilder_v8&
    {
        for(int i = 0; i < count; i++)
        {
            m_resampleDesc.prePadding[i].numerator = arr[i];
            m_resampleDesc.prePadding[i].denominator = 1;
        }
        return *this;
    }

    auto setSpatialStride(int64_t count, const int64_t* arr) -> ResampleDescBuilder_v8&
    {
        for(int i = 0; i < count; i++)
        {
            m_resampleDesc.stride[i].numerator = arr[i];
            m_resampleDesc.stride[i].denominator = 1;
        }
        return *this;
    }

    auto setSpatialDim(int64_t count, const int64_t* arr) -> ResampleDescBuilder_v8&
    {
        m_resampleDesc.spatialDim = count;
        for(int i = 0; i < count; i++)
        {
            m_resampleDesc.windowDim[i].numerator = arr[i];
            m_resampleDesc.windowDim[i].denominator = 1;
        }
        return *this;
    }

    ResampleDesc_v8&& build()
    {
        if(m_resampleDesc.spatialDim < 0)
        {
            set_error_and_throw_exception(
                &m_resampleDesc,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: Check and Set the spatialDim field");
            return std::move(m_resampleDesc);
        };

        auto status
            = m_resampleDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_resampleDesc);
        }

        ::cudnnResampleMode_t cudnn_resample_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.resample_mode, cudnn_resample_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_MODE,
                                       CUDNN_TYPE_RESAMPLE_MODE,
                                       1,
                                       &cudnn_resample_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_resampleDesc.computeType, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                       CUDNN_TYPE_NAN_PROPOGATION,
                                       1,
                                       &(m_resampleDesc.nanOpt));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION Failed");
            return std::move(m_resampleDesc);
        }

        cudnnPaddingMode_t cudnn_padding_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.padding_mode, cudnn_padding_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                                       CUDNN_TYPE_PADDING_MODE,
                                       1,
                                       &cudnn_padding_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &(m_resampleDesc.spatialDim));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.windowDim);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_WINDOW_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.prePadding);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_PRE_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.postPadding);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_POST_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_STRIDES,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.stride);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_resampleDesc,
                                          status,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RESAMPLE_STRIDES Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::finalize(m_resampleDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_resampleDesc);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_resampleDesc);
        return std::move(m_resampleDesc);
    }

    explicit ResampleDescBuilder_v8() = default;
    ~ResampleDescBuilder_v8() = default;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8&&) = delete;
    ResampleDescBuilder_v8(const ResampleDescBuilder_v8&) = delete;
    ResampleDescBuilder_v8& operator=(const ResampleDescBuilder_v8&) = delete;

private:
    ResampleDesc_v8 m_resampleDesc;
};

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,misc-const-correctness,modernize-avoid-c-arrays)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
