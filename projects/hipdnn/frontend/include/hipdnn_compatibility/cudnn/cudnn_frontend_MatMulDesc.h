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
 * @brief cuDNN v8 `MatMulDesc_v8` / `MatMulDescBuilder_v8`, built as a shim matmul record.
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
#include <hipdnn_compatibility/cudnn/detail/v8/build_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

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
        static const detail::v8::MatmulLabels s_labels{
            "MatMulDescBuilder: compute type",
            "MatMulDescBuilder: padding value",
        };
        return detail::v8::buildDescriptor(m_matMulDesc, CUDNN_BACKEND_MATMUL_DESCRIPTOR, [&] {
            cudnnDataType_t cudnn_data_type;
            auto status = detail::convert_to_cudnn_type(m_matMulDesc.compute_type, cudnn_data_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status, "MatMulDescBuilder: compute type is not set or not a valid DataType_t");
            }
            return detail::v8::makeMatmulRecord({cudnn_data_type, m_matMulDesc.isPadded}, s_labels);
        });
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
