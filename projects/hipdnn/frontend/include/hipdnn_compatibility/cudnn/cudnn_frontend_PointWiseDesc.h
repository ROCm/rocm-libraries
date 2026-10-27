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
 * @brief cuDNN v8 `PointWiseDesc_v8` / `PointWiseDescBuilder_v8`, built as a shim pointwise record.
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
#include <hipdnn_compatibility/cudnn/detail/v8/build_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

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
    template <typename Descriptor, typename MakeRecord>
    friend Descriptor&& detail::v8::buildDescriptor(Descriptor& descriptor,
                                                    cudnnBackendDescriptorType_t type,
                                                    MakeRecord&& makeRecord);
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
        static const detail::v8::PointwiseLabels s_labels{
            "PointWiseDescBuilder: mode",
            "PointWiseDescBuilder: compute type",
            "PointWiseDescBuilder: ReLU lower clip",
            "PointWiseDescBuilder: ReLU upper clip",
            "PointWiseDescBuilder: ReLU lower clip slope",
            "PointWiseDescBuilder: ELU alpha",
            "PointWiseDescBuilder: softplus beta",
            "PointWiseDescBuilder: swish beta",
        };
        return detail::v8::buildDescriptor(
            m_pointWiseDesc, CUDNN_BACKEND_POINTWISE_DESCRIPTOR, [&] {
                cudnnPointwiseMode_t cudnn_pointwise_mode;
                auto status
                    = detail::convert_to_cudnn_type(m_pointWiseDesc.mode, cudnn_pointwise_mode);
                if(status != CUDNN_STATUS_SUCCESS)
                {
                    throw detail::v8::ShimError(
                        status,
                        "PointWiseDescBuilder: mode is not set or not a valid PointwiseMode_t");
                }

                cudnnDataType_t cudnn_data_type;
                status
                    = detail::convert_to_cudnn_type(m_pointWiseDesc.compute_type, cudnn_data_type);
                if(status != CUDNN_STATUS_SUCCESS)
                {
                    throw detail::v8::ShimError(
                        status, "PointWiseDescBuilder: compute type is not a valid DataType_t");
                }

                detail::v8::PointwiseInputs inputs;
                inputs.mode = cudnn_pointwise_mode;
                inputs.computeType = cudnn_data_type;
                if(m_pointWiseDesc.mode == PointwiseMode_t::RELU_FWD
                   || m_pointWiseDesc.mode == PointwiseMode_t::RELU_BWD)
                {
                    detail::v8::requireEnumerator(m_pointWiseDesc.nan_propagation,
                                                  "PointWiseDescBuilder: NaN propagation");
                    inputs.reluLowerClip = m_pointWiseDesc.lower_clip;
                    if(m_pointWiseDesc.compute_type == DataType_t::FLOAT)
                    {
                        inputs.reluUpperClip = std::min<double>(
                            m_pointWiseDesc.upper_clip,
                            static_cast<double>(std::numeric_limits<float>::max()));
                    }
                    else
                    {
                        inputs.reluUpperClip = m_pointWiseDesc.upper_clip;
                    }
                    inputs.reluLowerClipSlope = m_pointWiseDesc.lower_clip_slope;
                }
                else if(m_pointWiseDesc.mode == PointwiseMode_t::ELU_FWD
                        || m_pointWiseDesc.mode == PointwiseMode_t::ELU_BWD)
                {
                    inputs.eluAlpha = m_pointWiseDesc.elu_alpha;
                }
                else if(m_pointWiseDesc.mode == PointwiseMode_t::SOFTPLUS_FWD
                        || m_pointWiseDesc.mode == PointwiseMode_t::SOFTPLUS_BWD)
                {
                    inputs.softplusBeta = m_pointWiseDesc.softplus_beta;
                }
                else if(m_pointWiseDesc.mode == PointwiseMode_t::SWISH_FWD
                        || m_pointWiseDesc.mode == PointwiseMode_t::SWISH_BWD)
                {
                    inputs.swishBeta = m_pointWiseDesc.swish_beta;
                }
                else if(m_pointWiseDesc.mode == PointwiseMode_t::GEN_INDEX)
                {
                    inputs.axis = m_pointWiseDesc.axis;
                }

                return detail::v8::makePointwiseRecord(inputs, s_labels);
            });
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
