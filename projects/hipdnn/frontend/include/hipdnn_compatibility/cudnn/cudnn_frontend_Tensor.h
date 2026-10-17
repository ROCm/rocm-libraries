// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Tensor.h
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Tensor.h
 * @brief cuDNN v8 `Tensor_v8` / `TensorBuilder_v8` over the shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Tensor.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override,readability-redundant-smartptr-get,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class Tensor_v8 : public BackendDescriptor
{
public:
    friend class TensorBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        ss << "               CUDNN_BACKEND_TENSOR_DESCRIPTOR :\n"
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
           << "                                   Datatype: " << detail::enumJson(data_type)
#else
           << "                                   Datatype: " << static_cast<int>(data_type)
#endif
           << "\n                                   Id: " << id
           << "\n                                   Dims " << nDims
           << "\n                                   VectorCount: " << vectorCount
           << "\n                                   vectorDimension " << vectorDimension;
        ss << "\n                                   Dim [ ";
        for(auto i = 0; i < nDims; i++)
        {
            if(i != 0)
            {
                ss << ',';
            }
            ss << btensor_dimA[i];
        }
        ss << " ]\n                                   Str [ ";
        for(auto i = 0; i < nDims; i++)
        {
            if(i != 0)
            {
                ss << ',';
            }
            ss << btensor_strA[i];
        }
        ss << " ]";
        ss << "\n                                   isVirtual: " << isVirtual
           << "\n                                   isByValue: " << isByValue
           << "\n                                   Alignment: " << alignment;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "\n                                   reorder_type: "
           << detail::enumJson(reorder_type);
#else
        ss << "\n                                   reorder_type: "
           << static_cast<int>(reorder_type);
#endif
        if(raggedOffset != nullptr)
        {
            ss << "\n                                   raggedOffset: Enabled UID: "
               << raggedOffset->getId();
        }
        return ss.str();
    }

    int64_t getPackedElementCount() const
    {
        int64_t count = vectorCount;
        for(auto i = 0; i < nDims; i++)
        {
            count = count * btensor_dimA[i];
        }
        return count;
    };

    int64_t getDimCount() const
    {
        return nDims;
    }

    const int64_t* getDim() const
    {
        return btensor_dimA;
    }

    const int64_t* getStride() const
    {
        return btensor_strA;
    }

    const int64_t* getDimArray() const
    {
        return getDim();
    }

    const int64_t* getStrideArray() const
    {
        return getStride();
    }

    int64_t getDataType() const
    {
        return static_cast<int64_t>(data_type);
    }

    int64_t getId() const
    {
        return id;
    }

    int64_t getAlignment() const
    {
        return alignment;
    }

    bool isVirtualTensor() const
    {
        return isVirtual;
    }

    int64_t getDimensionCount() const
    {
        return getDimCount();
    }

    Tensor_v8(Tensor_v8&& from) = default;
    Tensor_v8& operator=(Tensor_v8&&) = default;

    ~Tensor_v8() = default;

private:
    Tensor_v8() = default;
    Tensor_v8(const Tensor_v8&) = delete;
    Tensor_v8& operator=(const Tensor_v8&) = delete;

    DataType_t data_type = DataType_t::NOT_SET;
    int64_t btensor_dimA[CUDNN_DIM_MAX + 1] = {-1};
    int64_t btensor_strA[CUDNN_DIM_MAX + 1] = {-1};
    int64_t id = -1;
    int64_t alignment = -1;
    int64_t nDims = -1;
    int64_t vectorDimension = -1;
    int64_t vectorCount = 1;
    bool isVirtual = false;
    bool isByValue = false;

    bool hasConstValue = false;
    std::vector<uint8_t> constValueBytes;

    cudnn_frontend::TensorReordering_t reorder_type = cudnn_frontend::TensorReordering_t::NONE;
    std::shared_ptr<Tensor_v8> raggedOffset;
};

class TensorBuilder_v8
{
public:
    auto setDataType(DataType_t data_type) -> TensorBuilder_v8&
    {
        m_tensor.data_type = data_type;
        return *this;
    }
    auto setDataType(cudnnDataType_t data_type) -> TensorBuilder_v8&
    {
        m_tensor.data_type = detail::convert_from_cudnn_type(data_type);
        return *this;
    }
    auto setDim(int64_t ndim, const int64_t* dim) -> TensorBuilder_v8&
    {
        std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
        m_tensor.nDims = ndim;
        return *this;
    }
    auto setStride(int64_t ndim, const int64_t* strides) -> TensorBuilder_v8&
    {
        std::copy(strides, strides + ndim, m_tensor.btensor_strA);
        return *this;
    }
    auto setId(int64_t id_) -> TensorBuilder_v8&
    {
        m_tensor.id = id_;
        return *this;
    }
    auto setAlignment(int64_t alignment_) -> TensorBuilder_v8&
    {
        m_tensor.alignment = alignment_;
        return *this;
    }
    auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8&
    {
        m_tensor.isVirtual = virtual_;
        return *this;
    }
    auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8&
    {
        m_tensor.isByValue = isByValue_;
        return *this;
    }

    template <typename T>
    auto setConstValue(const T& value) -> TensorBuilder_v8&
    {
        m_tensor.hasConstValue = true;
        m_tensor.constValueBytes.resize(sizeof(T));
        std::memcpy(m_tensor.constValueBytes.data(), &value, sizeof(T));
        m_tensor.isByValue = true;
        return *this;
    }

    auto setVectorCountAndDimension(int64_t vectorCount_,
                                    int64_t vectorDimension_) -> TensorBuilder_v8&
    {
        m_tensor.vectorCount = vectorCount_;
        m_tensor.vectorDimension = vectorDimension_;
        return *this;
    }

    auto setReorderType(cudnn_frontend::TensorReordering_t reordering_type) -> TensorBuilder_v8&
    {
        m_tensor.reorder_type = reordering_type;
        return *this;
    }

    auto setReorderType(cudnnBackendTensorReordering_t reordering_type) -> TensorBuilder_v8&
    {
        detail::convert_from_cudnn_type(reordering_type, m_tensor.reorder_type);
        return *this;
    }

    auto setStrides(int64_t ndim, const int64_t* strides) -> TensorBuilder_v8&
    {
        return setStride(ndim, strides);
    }

    auto setRaggedOffset(std::shared_ptr<Tensor_v8>& raggedOffset) -> TensorBuilder_v8&
    {
        m_tensor.raggedOffset = raggedOffset;
        return *this;
    }

    auto cloneFrom(const Tensor_v8& from, int64_t newID) -> TensorBuilder_v8&
    {
        m_tensor.data_type = from.data_type;
        m_tensor.nDims = from.nDims;
        m_tensor.id = newID;
        std::copy(from.getDimArray(), from.getDimArray() + m_tensor.nDims, m_tensor.btensor_dimA);
        std::copy(
            from.getStrideArray(), from.getStrideArray() + m_tensor.nDims, m_tensor.btensor_strA);
        m_tensor.alignment = from.alignment;
        m_tensor.isVirtual = from.isVirtual;
        m_tensor.isByValue = from.isByValue;
        m_tensor.vectorCount = from.vectorCount;
        m_tensor.vectorDimension = from.vectorDimension;
        m_tensor.reorder_type = from.reorder_type;
        return *this;
    }

    Tensor_v8&& build()
    {
        if(m_tensor.alignment <= 0)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT field");
            return std::move(m_tensor);
        }
        if(m_tensor.id < 0)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_TENSOR_UNIQUE_ID as a valid value");
            return std::move(m_tensor);
        }
        if(m_tensor.btensor_strA[0] < 0)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_TENSOR_STRIDES Correctly");
            return std::move(m_tensor);
        }
        if(m_tensor.btensor_dimA[0] <= 0)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_TENSOR_DIMENSIONS Correctly");
            return std::move(m_tensor);
        }
        if(m_tensor.pointer != nullptr)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Bad tensor created. "
                                          "The tensor already seems to be pointing to something");
            return std::move(m_tensor);
        }

        auto status = m_tensor.initialize_managed_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_tensor);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_tensor.data_type, cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DATA_TYPE Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_DATA_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DATA_TYPE Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64,
                                       m_tensor.nDims,
                                       m_tensor.btensor_dimA);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_tensor,
                                          status,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_TENSOR_DIMENSIONS Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_STRIDES,
                                       CUDNN_TYPE_INT64,
                                       m_tensor.nDims,
                                       m_tensor.btensor_strA);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_STRIDES Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_tensor.id);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_UNIQUE_ID Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_tensor.alignment);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_tensor,
                                          status,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT Failed");
            return std::move(m_tensor);
        }
        if(m_tensor.isVirtual)
        {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_tensor.isVirtual);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT Failed");
                return std::move(m_tensor);
            }
        }
        if(m_tensor.isByValue)
        {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_IS_BY_VALUE,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_tensor.isByValue);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_IS_BY_VALUE Failed");
                return std::move(m_tensor);
            }
        }

        if(m_tensor.vectorCount > 1)
        {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_VECTOR_COUNT,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &m_tensor.vectorCount);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_VECTOR_COUNT Failed");
                return std::move(m_tensor);
            }
        }
        if(m_tensor.vectorDimension >= 0)
        {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &m_tensor.vectorDimension);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION Failed");
                return std::move(m_tensor);
            }
        }

#if(CUDNN_VERSION >= 8900)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8900,
            m_tensor,
            "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
            "CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC requires cudnn version 8.9");
        if(m_tensor.raggedOffset != nullptr)
        {
            std::vector<cudnnBackendDescriptor_t> backendRaggedOffset;
            backendRaggedOffset.push_back(
                m_tensor.raggedOffset.get()->pointer->get_backend_descriptor());
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           static_cast<int64_t>(backendRaggedOffset.size()),
                                           backendRaggedOffset.data());
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC Failed");
                return std::move(m_tensor);
            }
        }
#endif

        if(m_tensor.reorder_type != cudnn_frontend::TensorReordering_t::NONE)
        {
            cudnnBackendTensorReordering_t cudnn_reordering_type;
            status = detail::convert_to_cudnn_type(m_tensor.reorder_type, cudnn_reordering_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_REORDERING_MODE Failed");
                return std::move(m_tensor);
            }
            // Upstream passes the FE enum where the C enum is meant; the shim
            // gives both the same values, so the bytes land as the C value.
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_REORDERING_MODE,
                                           CUDNN_TYPE_TENSOR_REORDERING_MODE,
                                           1,
                                           &m_tensor.reorder_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_REORDERING_MODE Failed");
                return std::move(m_tensor);
            }
        }

#if(CUDNN_VERSION >= 92200)
        if(m_tensor.hasConstValue)
        {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                92200,
                m_tensor,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                "CUDNN_ATTR_TENSOR_CONSTANT_VALUE requires cudnn version 9.22.0");
            void* value_ptr = m_tensor.constValueBytes.data();
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                           CUDNN_TYPE_VOID_PTR,
                                           1,
                                           &value_ptr);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_tensor,
                                              status,
                                              "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_TENSOR_CONSTANT_VALUE Failed");
                return std::move(m_tensor);
            }
        }
#else
        if(m_tensor.hasConstValue)
        {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_INVALID_VALUE,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: "
                                          "CUDNN_ATTR_TENSOR_CONSTANT_VALUE requires cudnn "
                                          "version 9.22.0");
            return std::move(m_tensor);
        }
#endif

        status = detail::finalize(m_tensor.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed");
            return std::move(m_tensor);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_tensor);
        return std::move(m_tensor);
    }

    explicit TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8&&) = delete;
    TensorBuilder_v8(const TensorBuilder_v8&) = delete;
    TensorBuilder_v8& operator=(const TensorBuilder_v8&) = delete;

private:
    Tensor_v8 m_tensor;
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override,readability-redundant-smartptr-get,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
