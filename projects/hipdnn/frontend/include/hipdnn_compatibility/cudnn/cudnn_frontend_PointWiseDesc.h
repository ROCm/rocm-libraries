// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_PointWiseDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_PointWiseDesc.h
 * @brief cuDNN v8 `PointWiseDesc_v8` / `PointWiseDescBuilder_v8` over the shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
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
              "include/cudnn_frontend_PointWiseDesc.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class PointWiseDesc_v8 : public BackendDescriptor
{
public:
    friend class PointWiseDescBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_POINTWISE_DESCRIPTOR :" << " Mode: " << detail::enumJson(mode)
           << " Math precision " << detail::enumJson(compute_type);
#else
        ss << "CUDNN_BACKEND_POINTWISE_DESCRIPTOR :" << " Mode: " << static_cast<int>(mode)
           << " Math precision " << static_cast<int>(compute_type);
#endif
        return ss.str();
    }

    int64_t getPortCount() const
    {
        return get_pointwise_mode_port_count(mode);
    }

    PointwiseMode_t getPointWiseMode() const
    {
        return mode;
    }

    PointWiseDesc_v8(PointWiseDesc_v8&& from) = default;
    PointWiseDesc_v8& operator=(PointWiseDesc_v8&& from) = default;

    ~PointWiseDesc_v8() = default;

private:
    PointWiseDesc_v8() = default;
    PointWiseDesc_v8(const PointWiseDesc_v8&) = delete;
    PointWiseDesc_v8& operator=(const PointWiseDesc_v8&) = delete;

    DataType_t compute_type = DataType_t::FLOAT;
    PointwiseMode_t mode = PointwiseMode_t::NOT_SET;
    cudnnNanPropagation_t nan_propagation = CUDNN_PROPAGATE_NAN;
    double upper_clip = std::numeric_limits<double>::max();
    double lower_clip = 0.0;
    double lower_clip_slope = 0.0;
    double elu_alpha = 1.0;
    double softplus_beta = 1.0;
    double swish_beta = 1.0;
    int64_t axis = -1;
};

class PointWiseDescBuilder_v8
{
public:
    auto setComputeType(DataType_t data_type_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.compute_type = data_type_;
        return *this;
    }
    auto setComputeType(cudnnDataType_t data_type_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }
    auto setClipping(double l, double u) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.upper_clip = u;
        m_pointWiseDesc.lower_clip = l;
        return *this;
    }
    auto setMode(PointwiseMode_t mode) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.mode = mode;
        return *this;
    }

    auto setMode(cudnnPointwiseMode_t mode) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.mode = detail::convert_from_cudnn_type(mode);
        return *this;
    }

    auto setMode(cudnnNanPropagation_t nan_mode_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.nan_propagation = nan_mode_;
        return *this;
    }

    auto setMathPrecision(cudnnDataType_t data_type_) -> PointWiseDescBuilder_v8&
    {
        return setComputeType(data_type_);
    }

    auto setReluLowerClip(double lower_clip_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.lower_clip = lower_clip_;
        return *this;
    }

    auto setReluUpperClip(double upper_clip_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.upper_clip = upper_clip_;
        return *this;
    }

    auto setReluLowerClipSlope(double lower_clip_slope_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.lower_clip_slope = lower_clip_slope_;
        return *this;
    }

    auto setEluAlpha(double elu_alpha_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.elu_alpha = elu_alpha_;
        return *this;
    }

    auto setSoftplusBeta(double softplus_beta_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.softplus_beta = softplus_beta_;
        return *this;
    }

    auto setSwishBeta(double swish_beta_) -> PointWiseDescBuilder_v8&
    {
        m_pointWiseDesc.swish_beta = swish_beta_;
        return *this;
    }

    auto setAxis(int64_t axis_) -> PointWiseDescBuilder_v8&
    {
        CUDNN_FRONTEND_UNUSED(axis_);
        m_pointWiseDesc.axis = axis_;
        return *this;
    }

    PointWiseDesc_v8 build()
    {
        auto status = m_pointWiseDesc.initialize_managed_backend_pointer(
            CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_pointWiseDesc, status, "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_pointWiseDesc);
        }

        cudnnPointwiseMode_t cudnn_pointwise_mode;
        status = detail::convert_to_cudnn_type(m_pointWiseDesc.mode, cudnn_pointwise_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_pointWiseDesc,
                                          status,
                                          "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_TYPE_POINTWISE_MODE Failed");
            return std::move(m_pointWiseDesc);
        }
        status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_POINTWISE_MODE,
                                       CUDNN_TYPE_POINTWISE_MODE,
                                       1,
                                       &cudnn_pointwise_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_pointWiseDesc,
                                          status,
                                          "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_TYPE_POINTWISE_MODE Failed");
            return std::move(m_pointWiseDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_pointWiseDesc.compute_type, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_pointWiseDesc,
                                          status,
                                          "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_POINTWISE_MATH_PREC Failed");
            return std::move(m_pointWiseDesc);
        }
        status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_POINTWISE_MATH_PREC,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_pointWiseDesc,
                                          status,
                                          "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_POINTWISE_MATH_PREC Failed");
            return std::move(m_pointWiseDesc);
        }

        if(m_pointWiseDesc.mode == PointwiseMode_t::RELU_FWD
           || m_pointWiseDesc.mode == PointwiseMode_t::RELU_BWD)
        {
            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                                           CUDNN_TYPE_NAN_PROPOGATION,
                                           1,
                                           &m_pointWiseDesc.nan_propagation);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_NAN_PROPAGATION Failed");
                return std::move(m_pointWiseDesc);
            }

            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_pointWiseDesc.lower_clip);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, Failed");
                return std::move(m_pointWiseDesc);
            }

            if(m_pointWiseDesc.compute_type == DataType_t::FLOAT)
            {
                double clamped_upper_clip
                    = std::min<double>(m_pointWiseDesc.upper_clip,
                                       static_cast<double>(std::numeric_limits<float>::max()));
                status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                               CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                               CUDNN_TYPE_DOUBLE,
                                               1,
                                               &clamped_upper_clip);
            }
            else
            {
                status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                               CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                               CUDNN_TYPE_DOUBLE,
                                               1,
                                               &m_pointWiseDesc.upper_clip);
            }
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, Failed");
                return std::move(m_pointWiseDesc);
            }

            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_pointWiseDesc.lower_clip_slope);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, Failed");
                return std::move(m_pointWiseDesc);
            }
        }
        else if(m_pointWiseDesc.mode == PointwiseMode_t::ELU_FWD
                || m_pointWiseDesc.mode == PointwiseMode_t::ELU_BWD)
        {
            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_ELU_ALPHA,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_pointWiseDesc.elu_alpha);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_ELU_ALPHA, Failed");
                return std::move(m_pointWiseDesc);
            }
        }
        else if(m_pointWiseDesc.mode == PointwiseMode_t::SOFTPLUS_FWD
                || m_pointWiseDesc.mode == PointwiseMode_t::SOFTPLUS_BWD)
        {
            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_pointWiseDesc.softplus_beta);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, Failed");
                return std::move(m_pointWiseDesc);
            }
        }
        else if(m_pointWiseDesc.mode == PointwiseMode_t::SWISH_FWD
                || m_pointWiseDesc.mode == PointwiseMode_t::SWISH_BWD)
        {
            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_SWISH_BETA,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_pointWiseDesc.swish_beta);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_SWISH_BETA, Failed");
                return std::move(m_pointWiseDesc);
            }
        }
        else if(m_pointWiseDesc.mode == PointwiseMode_t::GEN_INDEX)
        {
            status = detail::set_attribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_POINTWISE_AXIS,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &m_pointWiseDesc.axis);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_AXIS, Failed");
                return std::move(m_pointWiseDesc);
            }
        }

        status = detail::finalize(m_pointWiseDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_pointWiseDesc,
                status,
                "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_pointWiseDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_pointWiseDesc);
        return std::move(m_pointWiseDesc);
    }

    explicit PointWiseDescBuilder_v8() = default;
    ~PointWiseDescBuilder_v8() = default;
    PointWiseDescBuilder_v8(PointWiseDescBuilder_v8&&) = delete;
    PointWiseDescBuilder_v8(const PointWiseDescBuilder_v8&) = delete;
    PointWiseDescBuilder_v8& operator=(const PointWiseDescBuilder_v8&) = delete;

private:
    PointWiseDesc_v8 m_pointWiseDesc;
};
using PointWiseDescBuilder = PointWiseDescBuilder_v8;
using PointWiseDesc = PointWiseDesc_v8;
using PointwiseDescBuilder = PointWiseDescBuilder_v8;
using PointwiseDesc = PointWiseDesc_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
