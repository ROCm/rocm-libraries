// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_MatMulDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_MatMulDesc.h
 * @brief cuDNN v8 `MatMulDesc_v8` / `MatMulDescBuilder_v8` over the shim's backend C API.
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
              "include/cudnn_frontend_MatMulDesc.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class MatMulDesc_v8 : public BackendDescriptor
{
public:
    friend class MatMulDescBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_MATMUL_DESCRIPTOR :" << " Math precision "
           << detail::enumJson(compute_type);
#else
        ss << "CUDNN_BACKEND_MATMUL_DESCRIPTOR :" << " Math precision "
           << static_cast<int>(compute_type);
#endif
        return ss.str();
    }

    MatMulDesc_v8(MatMulDesc_v8&& from) = default;
    MatMulDesc_v8& operator=(MatMulDesc_v8&& from) = default;

    ~MatMulDesc_v8() = default;

private:
    MatMulDesc_v8() = default;
    MatMulDesc_v8(const MatMulDesc_v8&) = delete;
    MatMulDesc_v8& operator=(const MatMulDesc_v8&) = delete;

    DataType_t compute_type = DataType_t::NOT_SET;
    bool isPadded = false;
    double paddingValue = 0.0;
};

class MatMulDescBuilder_v8
{
public:
    auto setComputeType(DataType_t data_type_) -> MatMulDescBuilder_v8&
    {
        m_matMulDesc.compute_type = data_type_;
        return *this;
    }
    auto setComputeType(cudnnDataType_t data_type_) -> MatMulDescBuilder_v8&
    {
        m_matMulDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }

    auto setMathPrecision(cudnnDataType_t data_type_) -> MatMulDescBuilder_v8&
    {
        return setComputeType(data_type_);
    }

    auto setPaddingValue(double paddingValue) -> MatMulDescBuilder_v8&
    {
        m_matMulDesc.isPadded = true;
        m_matMulDesc.paddingValue = paddingValue;
        return *this;
    }

    MatMulDesc_v8&& build()
    {
        auto status
            = m_matMulDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_matMulDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_matMulDesc.compute_type, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_matMulDesc,
                status,
                "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_COMP_TYPE Failed");
            return std::move(m_matMulDesc);
        }
        status = detail::set_attribute(m_matMulDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_MATMUL_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_matMulDesc,
                status,
                "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_COMP_TYPE Failed");
            return std::move(m_matMulDesc);
        }

#if(CUDNN_VERSION >= 8900)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8900,
            m_matMulDesc,
            "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_PADDING_VALUE "
            "requires cudnn 8.9.0");
        if(m_matMulDesc.isPadded)
        {
            status = detail::set_attribute(m_matMulDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_MATMUL_PADDING_VALUE,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_matMulDesc.paddingValue);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_matMulDesc,
                                              status,
                                              "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_MATMUL_PADDING_VALUE Failed");
                return std::move(m_matMulDesc);
            }
        }
#endif

        status = detail::finalize(m_matMulDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_matMulDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_matMulDesc);
        return std::move(m_matMulDesc);
    }

    explicit MatMulDescBuilder_v8() = default;
    ~MatMulDescBuilder_v8() = default;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8&&) = delete;
    MatMulDescBuilder_v8(const MatMulDescBuilder_v8&) = delete;
    MatMulDescBuilder_v8& operator=(const MatMulDescBuilder_v8&) = delete;

private:
    MatMulDesc_v8 m_matMulDesc;
};
using MatMulDesc = MatMulDesc_v8;
using MatMulDescBuilder = MatMulDescBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
