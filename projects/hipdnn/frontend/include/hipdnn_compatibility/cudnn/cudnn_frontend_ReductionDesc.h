// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_ReductionDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_ReductionDesc.h
 * @brief cuDNN FE v0.x `ReductionDesc_v8` and its builder, on the shim's backend C API.
 */
#pragma once

#include <algorithm>
#include <array>
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
              "include/cudnn_frontend_ReductionDesc.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,readability-else-after-return):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class ReductionDesc_v8 : public BackendDescriptor
{
public:
    friend class ReductionDescBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_REDUCTION_DESCRIPTOR :" << " Math precision "
           << detail::enumJson(compute_type) << " Reduction mode "
           << detail::enumJson(reduction_mode) << " Is deterministic "
           << (is_deterministic ? "true" : "false");
#else
        ss << "CUDNN_BACKEND_REDUCTION_DESCRIPTOR :" << " Math precision "
           << static_cast<int>(compute_type) << " Reduction mode "
           << static_cast<int>(reduction_mode) << " Is deterministic "
           << static_cast<int>(is_deterministic);
#endif
        return ss.str();
    }

    ReductionDesc_v8(ReductionDesc_v8&& from) = default;
    ReductionDesc_v8& operator=(ReductionDesc_v8&& from) = default;

    ~ReductionDesc_v8() = default;

private:
    ReductionDesc_v8() = default;
    ReductionDesc_v8(const ReductionDesc_v8&) = delete;
    ReductionDesc_v8& operator=(const ReductionDesc_v8&) = delete;

    DataType_t compute_type = DataType_t::NOT_SET;
    ReductionMode_t reduction_mode = ReductionMode_t::NOT_SET;
    bool is_deterministic = false;
};

class ReductionDescBuilder_v8
{
public:
    auto setComputeType(DataType_t data_type_) -> ReductionDescBuilder_v8&
    {
        m_reductionDesc.compute_type = data_type_;
        return *this;
    }
    auto setComputeType(cudnnDataType_t data_type_) -> ReductionDescBuilder_v8&
    {
        m_reductionDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }
    auto setReductionOp(ReductionMode_t op_) -> ReductionDescBuilder_v8&
    {
        m_reductionDesc.reduction_mode = op_;
        return *this;
    }
    auto setReductionOp(cudnnReduceTensorOp_t op_) -> ReductionDescBuilder_v8&
    {
        m_reductionDesc.reduction_mode = detail::convert_from_cudnn_type(op_);
        return *this;
    }

    auto setMathPrecision(cudnnDataType_t data_type_) -> ReductionDescBuilder_v8&
    {
        return setComputeType(data_type_);
    }
    auto setIsDeterministic(bool is_deterministic_) -> ReductionDescBuilder_v8&
    {
        m_reductionDesc.is_deterministic = is_deterministic_;
        return *this;
    }

    ReductionDesc_v8&& build()
    {
        auto status = m_reductionDesc.initialize_managed_backend_pointer(
            CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_reductionDesc, status, "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_reductionDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_reductionDesc.compute_type, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_reductionDesc,
                                          status,
                                          "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_REDUCTION_COMP_TYPE Failed");
            return std::move(m_reductionDesc);
        }
        status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_REDUCTION_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_reductionDesc,
                                          status,
                                          "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_REDUCTION_COMP_TYPE Failed");
            return std::move(m_reductionDesc);
        }

        cudnnReduceTensorOp_t cudnn_reduction_mode;
        status
            = detail::convert_to_cudnn_type(m_reductionDesc.reduction_mode, cudnn_reduction_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_reductionDesc,
                                          status,
                                          "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_REDUCTION_OPERATOR Failed");
            return std::move(m_reductionDesc);
        }
        status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_REDUCTION_OPERATOR,
                                       CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                       1,
                                       &cudnn_reduction_mode);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_reductionDesc,
                                          status,
                                          "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_REDUCTION_OPERATOR Failed");
            return std::move(m_reductionDesc);
        }

#if(CUDNN_VERSION >= 91100)
        if(detail::get_backend_version() < 91100)
        {
            if(m_reductionDesc.is_deterministic)
            {
                set_error_and_throw_exception(&m_reductionDesc,
                                              CUDNN_STATUS_NOT_SUPPORTED,
                                              "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: DETERMINISTIC "
                                              "mode is not supported "
                                              "in cudnn version < 9.11.0");
                return std::move(m_reductionDesc);
            }
            else
            {
                // Do nothing.
            }
        }
        else
        {
            status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_reductionDesc.is_deterministic);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_reductionDesc,
                                              status,
                                              "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC Failed");
                return std::move(m_reductionDesc);
            }
        }
#endif

        status = detail::finalize(m_reductionDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_reductionDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_reductionDesc);
        return std::move(m_reductionDesc);
    }

    explicit ReductionDescBuilder_v8() = default;
    ~ReductionDescBuilder_v8() = default;
    ReductionDescBuilder_v8(ReductionDescBuilder_v8&&) = delete;
    ReductionDescBuilder_v8(const ReductionDescBuilder_v8&) = delete;
    ReductionDescBuilder_v8& operator=(const ReductionDescBuilder_v8&) = delete;

private:
    ReductionDesc_v8 m_reductionDesc;
};

using ReductionDesc = ReductionDesc_v8;
using ReductionDescBuilder = ReductionDescBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,readability-else-after-return)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
