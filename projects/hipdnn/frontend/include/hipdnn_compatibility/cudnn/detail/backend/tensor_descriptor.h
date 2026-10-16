// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// Finalize helpers shared with parameter_descriptors.h.

template <typename T>
T requireAttribute(const std::optional<T>& value, cudnnBackendAttributeName_t name)
{
    if(!value)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, attributeName(name) + " is required");
    }
    return *value;
}

template <typename T>
std::vector<T> requireNonEmpty(std::vector<T> values, cudnnBackendAttributeName_t name)
{
    if(values.empty())
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, attributeName(name) + " is required (non-empty)");
    }
    return values;
}

template <typename CEnum>
[[noreturn]] void throwUnsupportedValue(CEnum value, cudnnBackendAttributeName_t name)
{
    throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                    attributeName(name) + " value " + std::to_string(static_cast<int>(value))
                        + " is not supported by the hipDNN cuDNN shim");
}

// cuDNN value -> shim FE enum, for values that are stored but never lowered.
template <typename CEnum>
auto requireFe(CEnum value, cudnnBackendAttributeName_t name)
{
    const auto mapped = toFe(value);
    if(!mapped)
    {
        throwUnsupportedValue(value, name);
    }
    return *mapped;
}

// cuDNN value -> shim FE enum that hipDNN can also represent; anything else is
// NOT_SUPPORTED.
template <typename CEnum>
auto requireLowerable(CEnum value, cudnnBackendAttributeName_t name)
{
    const auto mapped = requireFe(value, name);
    if(!hasHipdnnCounterpart(mapped))
    {
        throwUnsupportedValue(value, name);
    }
    return mapped;
}

class TensorDescriptor final : public Descriptor
{
public:
    TensorDescriptor()
        : Descriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR)
    {
    }

    // Lowering facts; valid once finalized.
    DataType_t dataType() const noexcept
    {
        return _dataType;
    }

    const std::vector<std::int64_t>& dimensions() const noexcept
    {
        return _dimensions;
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _strides;
    }

    std::int64_t uniqueId() const noexcept
    {
        return _uniqueId;
    }

    bool isVirtual() const noexcept
    {
        return _isVirtual;
    }

    // cuDNN's alignment is a lower bound and 16 is hipDNN's default, so only a
    // smaller value is forwarded.
    std::optional<std::int64_t> forwardedAlignment() const noexcept
    {
        return _alignment < HIPDNN_DEFAULT_ALIGNMENT ? std::optional<std::int64_t>(_alignment)
                                                     : std::nullopt;
    }

    // IS_BY_VALUE without CONSTANT_VALUE; a constant lowers to the tensor value
    // only.
    bool isRuntimePassByValue() const noexcept
    {
        return _isByValue && !_constantValue;
    }

    const std::optional<std::vector<std::byte>>& constantValue() const noexcept
    {
        return _constantValue;
    }

    const std::shared_ptr<TensorDescriptor>& raggedOffset() const noexcept
    {
        return _raggedOffset;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_TENSOR_DATA_TYPE),
                               CUDNN_ATTR_TENSOR_DATA_TYPE);
        std::vector<std::int64_t> dimensions
            = requireNonEmpty(arrayAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_DIMENSIONS),
                              CUDNN_ATTR_TENSOR_DIMENSIONS);
        std::vector<std::int64_t> strides = requireNonEmpty(
            arrayAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_STRIDES), CUDNN_ATTR_TENSOR_STRIDES);
        if(dimensions.size() != strides.size())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_TENSOR_DIMENSIONS has " + std::to_string(dimensions.size())
                                + " elements but CUDNN_ATTR_TENSOR_STRIDES has "
                                + std::to_string(strides.size()));
        }
        for(std::size_t i = 0; i < dimensions.size(); ++i)
        {
            if(dimensions[i] <= 0 || strides[i] < 0)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                "tensor dimension " + std::to_string(i) + " is "
                                    + std::to_string(dimensions[i]) + " with stride "
                                    + std::to_string(strides[i])
                                    + "; dimensions must be positive and strides non-negative");
            }
        }
        const std::int64_t uniqueId
            = requireAttribute(scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_UNIQUE_ID),
                               CUDNN_ATTR_TENSOR_UNIQUE_ID);
        if(uniqueId < 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_TENSOR_UNIQUE_ID must be >= 0");
        }
        const std::int64_t alignment
            = requireAttribute(scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT),
                               CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
        if(alignment <= 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT must be > 0");
        }

        const DataType_t dataType = requireLowerable(rawType, CUDNN_ATTR_TENSOR_DATA_TYPE);
        if(scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_VECTOR_COUNT).value_or(1) != 1)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_TENSOR_VECTOR_COUNT other than 1 is not supported");
        }
        if(scalarAttribute<cudnnBackendTensorReordering_t>(CUDNN_ATTR_TENSOR_REORDERING_MODE)
               .value_or(CUDNN_TENSOR_REORDERING_NONE)
           != CUDNN_TENSOR_REORDERING_NONE)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_TENSOR_REORDERING_MODE other than NONE is not supported");
        }

        std::optional<std::vector<std::byte>> constantValue;
        if(hasAttribute(CUDNN_ATTR_TENSOR_CONSTANT_VALUE))
        {
            constantValue = copyConstant(dataType);
        }

        std::shared_ptr<TensorDescriptor> raggedOffset;
        if(std::shared_ptr<Descriptor> referent
           = descriptorAttribute(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC))
        {
            descriptorCast<TensorDescriptor>(*referent, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
            raggedOffset = std::static_pointer_cast<TensorDescriptor>(std::move(referent));
        }

        _dataType = dataType;
        _dimensions = std::move(dimensions);
        _strides = std::move(strides);
        _uniqueId = uniqueId;
        _alignment = alignment;
        _isVirtual = scalarAttribute<bool>(CUDNN_ATTR_TENSOR_IS_VIRTUAL).value_or(false);
        _isByValue = scalarAttribute<bool>(CUDNN_ATTR_TENSOR_IS_BY_VALUE).value_or(false);
        _constantValue = std::move(constantValue);
        _raggedOffset = std::move(raggedOffset);
        if(_constantValue)
        {
            // GET must not hand back the caller's pointer: it refers to builder
            // storage that is gone once the builder is.
            const void* owned = _constantValue->data();
            storeComputed(CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                          CUDNN_TYPE_VOID_PTR,
                          1,
                          static_cast<const void*>(&owned));
        }
    }

private:
    static constexpr std::int64_t HIPDNN_DEFAULT_ALIGNMENT = 16;

    // CONSTANT_VALUE holds a pointer to builder-owned bytes, so they are copied
    // here; the element size is rounded up to whole bytes (upstream stores a
    // bool constant as one byte).
    std::vector<std::byte> copyConstant(DataType_t dataType) const
    {
        const void* source
            = scalarAttribute<const void*>(CUDNN_ATTR_TENSOR_CONSTANT_VALUE).value_or(nullptr);
        if(source == nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_TENSOR_CONSTANT_VALUE is null");
        }
        const std::size_t size = (get_element_size_in_bits(dataType) + 7) / 8;
        if(size == 0)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_TENSOR_CONSTANT_VALUE: the tensor data type has no size");
        }
        std::vector<std::byte> bytes(size);
        std::memcpy(bytes.data(), source, size);
        return bytes;
    }

    DataType_t _dataType = DataType_t::NOT_SET;
    std::vector<std::int64_t> _dimensions;
    std::vector<std::int64_t> _strides;
    std::int64_t _uniqueId = -1;
    std::int64_t _alignment = 0;
    bool _isVirtual = false;
    bool _isByValue = false;
    std::optional<std::vector<std::byte>> _constantValue;
    std::shared_ptr<TensorDescriptor> _raggedOffset;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
