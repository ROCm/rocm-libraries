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
 * @brief cuDNN FE v0.x `ResampleDesc_v8` and its builder, built as a shim resample record.
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
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

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
        static const detail::v8::ResampleLabels s_labels{
            "ResampleDescBuilder: resample mode",
            "ResampleDescBuilder: compute type",
            "ResampleDescBuilder: padding mode",
            "ResampleDescBuilder: spatial dimension count",
            "ResampleDescBuilder: window dimensions",
            "ResampleDescBuilder: pre-padding",
            "ResampleDescBuilder: post-padding",
            "ResampleDescBuilder: spatial stride",
        };
        auto outcome = detail::v8::guarded([&] {
            // Created before upstream's precheck so that a failed descriptor still has a (null)
            // raw descriptor to report.
            m_resampleDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
            if(m_resampleDesc.spatialDim < 0)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "ResampleDescBuilder: spatial dimension count must be non-negative");
            }

            ::cudnnResampleMode_t cudnn_resample_mode;
            auto status
                = detail::convert_to_cudnn_type(m_resampleDesc.resample_mode, cudnn_resample_mode);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status,
                    "ResampleDescBuilder: resample mode is not set or not a valid ResampleMode_t");
            }

            cudnnDataType_t cudnn_data_type;
            status = detail::convert_to_cudnn_type(m_resampleDesc.computeType, cudnn_data_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status, "ResampleDescBuilder: compute type is not a valid DataType_t");
            }

            detail::v8::requireEnumerator(m_resampleDesc.nanOpt,
                                          "ResampleDescBuilder: NaN propagation");

            cudnnPaddingMode_t cudnn_padding_mode;
            status = detail::convert_to_cudnn_type(m_resampleDesc.padding_mode, cudnn_padding_mode);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status,
                    "ResampleDescBuilder: padding mode is not set or not a valid PaddingMode_t");
            }

            // A count above the arrays' capacity no longer matches spatialDim, so the core rejects
            // it. Zero denominators are rejected here, ahead of the core's support checks.
            const auto count = static_cast<std::size_t>(
                std::min<int64_t>(m_resampleDesc.spatialDim, CUDNN_DIM_MAX));
            const auto fractions = [count](const cudnnFraction_t* array, std::string_view label) {
                for(std::size_t i = 0; i < count; ++i)
                {
                    if(array[i].denominator == 0)
                    {
                        throw detail::v8::ShimError(CUDNN_STATUS_BAD_PARAM,
                                                    std::string(label) + "[" + std::to_string(i)
                                                        + "] has a zero denominator");
                    }
                }
                return std::vector<cudnnFraction_t>(array, array + count);
            };
            detail::v8::ResampleInputs inputs;
            inputs.mode = cudnn_resample_mode;
            inputs.computeType = cudnn_data_type;
            inputs.paddingMode = cudnn_padding_mode;
            inputs.spatialDims = m_resampleDesc.spatialDim;
            inputs.windowDims = fractions(m_resampleDesc.windowDim, s_labels.windowDims);
            inputs.prePaddings = fractions(m_resampleDesc.prePadding, s_labels.prePaddings);
            inputs.postPaddings = fractions(m_resampleDesc.postPadding, s_labels.postPaddings);
            inputs.strides = fractions(m_resampleDesc.stride, s_labels.strides);
            m_resampleDesc.pointer
                = detail::v8::makeManaged(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                          detail::v8::makeResampleRecord(inputs, s_labels));
        });
        if(!outcome)
        {
            set_error_and_throw_exception(&m_resampleDesc, outcome.status, outcome.message.c_str());
            return std::move(m_resampleDesc);
        }
        // A retry after a failed build() must not keep that failure's status.
        m_resampleDesc.set_status(CUDNN_STATUS_SUCCESS);
        m_resampleDesc.set_error("");
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
