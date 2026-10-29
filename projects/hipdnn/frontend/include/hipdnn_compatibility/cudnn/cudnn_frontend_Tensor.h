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
 * @brief cuDNN v8 `Tensor_v8` / `TensorBuilder_v8`, built as a shim tensor record.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/build_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Tensor.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override):
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
        m_constValueBytes.resize(sizeof(T));
        std::memcpy(m_constValueBytes.data(), &value, sizeof(T));
        if constexpr(std::is_integral_v<T>)
        {
            m_constantSource = std::is_signed_v<T> ? detail::v8::ConstantSource::SIGNED_INTEGER
                                                   : detail::v8::ConstantSource::UNSIGNED_INTEGER;
        }
        else
        {
            m_constantSource = detail::v8::ConstantSource::OTHER;
        }
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
        m_raggedOffset = raggedOffset;
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
        // The builder owns the ragged offset and the constant: binding a failed build() by value
        // moves m_tensor's copies out, and the fixed builder may build again.
        m_tensor.raggedOffset = m_raggedOffset;
        if(m_tensor.alignment <= 0)
        {
            return failPrecheck("TensorBuilder: byte alignment must be positive");
        }
        if(m_tensor.id < 0)
        {
            return failPrecheck("TensorBuilder: unique id must be non-negative");
        }
        if(m_tensor.btensor_strA[0] < 0)
        {
            return failPrecheck("TensorBuilder: strides must be set and non-negative");
        }
        if(m_tensor.btensor_dimA[0] <= 0)
        {
            return failPrecheck("TensorBuilder: dimensions must be set and positive");
        }
        // A failed build() leaves an empty placeholder, which does not count as built.
        if(detail::v8::recordOf<detail::v8::TensorRecord>(m_tensor.pointer) != nullptr)
        {
            return failPrecheck("TensorBuilder: the tensor is already built");
        }

        static const detail::v8::TensorLabels s_labels{
            "TensorBuilder: data type",
            "TensorBuilder: dimensions",
            "TensorBuilder: strides",
            "TensorBuilder: unique id",
            "TensorBuilder: byte alignment",
            "TensorBuilder: vector count",
            "TensorBuilder: reordering mode",
            "TensorBuilder: constant value",
            "TensorBuilder: virtual tensor",
        };
        return detail::v8::buildDescriptor(m_tensor, CUDNN_BACKEND_TENSOR_DESCRIPTOR, [&] {
            cudnnDataType_t cudnn_data_type;
            auto status = detail::convert_to_cudnn_type(m_tensor.data_type, cudnn_data_type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                throw detail::v8::ShimError(
                    status, "TensorBuilder: data type is not set or not a valid DataType_t");
            }

            detail::v8::TensorInputs inputs;
            if(m_tensor.raggedOffset != nullptr)
            {
                inputs.raggedOffset = detail::v8::recordOf<detail::v8::TensorRecord>(
                    m_tensor.raggedOffset->get_desc());
                if(inputs.raggedOffset == nullptr)
                {
                    throw detail::v8::ShimError(CUDNN_STATUS_BAD_PARAM,
                                                "TensorBuilder: ragged offset tensor is not built");
                }
            }

            cudnnBackendTensorReordering_t cudnn_reordering_type = CUDNN_TENSOR_REORDERING_NONE;
            if(m_tensor.reorder_type != cudnn_frontend::TensorReordering_t::NONE)
            {
                status
                    = detail::convert_to_cudnn_type(m_tensor.reorder_type, cudnn_reordering_type);
                if(status != CUDNN_STATUS_SUCCESS)
                {
                    throw detail::v8::ShimError(
                        status, "TensorBuilder: reordering mode is not a valid TensorReordering_t");
                }
            }

            // A count above the arrays' capacity (already an overrun in setDim) still reaches the
            // core as too many elements.
            const auto values = [&](const int64_t* array) {
                constexpr int64_t capacity = CUDNN_DIM_MAX + 1;
                std::vector<int64_t> copied(
                    array, array + std::clamp<int64_t>(m_tensor.nDims, 0, capacity));
                if(m_tensor.nDims > capacity)
                {
                    copied.resize(capacity + 1);
                }
                return copied;
            };
            inputs.dataType = cudnn_data_type;
            inputs.dimensions = values(m_tensor.btensor_dimA);
            inputs.strides = values(m_tensor.btensor_strA);
            inputs.uniqueId = m_tensor.id;
            inputs.alignment = m_tensor.alignment;
            // Upstream sends the vector count only above 1.
            inputs.vectorCount = m_tensor.vectorCount > 1 ? m_tensor.vectorCount : 1;
            inputs.reorderingMode = cudnn_reordering_type;
            inputs.isVirtual = m_tensor.isVirtual;
            inputs.isByValue = m_tensor.isByValue;
            if(!m_constValueBytes.empty())
            {
                inputs.constantValue.emplace(m_constValueBytes.size());
                std::memcpy(inputs.constantValue->data(),
                            m_constValueBytes.data(),
                            m_constValueBytes.size());
                inputs.constantSource = m_constantSource;
            }
            return detail::v8::makeTensorRecord(std::move(inputs), s_labels);
        });
    }

    explicit TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8&&) = delete;
    TensorBuilder_v8(const TensorBuilder_v8&) = delete;
    TensorBuilder_v8& operator=(const TensorBuilder_v8&) = delete;

private:
    // Upstream's prechecks run before the descriptor exists. Without exceptions the failed tensor
    // is handed back with an empty pointer: an operation's optional slot then reads it as set but
    // not built (BAD_PARAM) rather than as absent. buildDescriptor() replaces the placeholder, so a
    // fixed builder can still build.
    Tensor_v8&& failPrecheck(const char* message)
    {
#ifdef NV_CUDNN_DISABLE_EXCEPTION
        if(m_tensor.pointer == nullptr)
        {
            detail::v8::guarded([&] {
                m_tensor.initialize_managed_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
            });
        }
#endif
        set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, message);
        return std::move(m_tensor);
    }

    Tensor_v8 m_tensor;
    std::shared_ptr<Tensor_v8> m_raggedOffset;
    std::vector<uint8_t> m_constValueBytes;
    detail::v8::ConstantSource m_constantSource = detail::v8::ConstantSource::OTHER;
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-avoid-c-arrays,modernize-use-equals-delete,modernize-use-override)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
