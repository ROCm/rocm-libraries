// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// Attribute name as a v8 core message label; the table names have static
// storage.
inline std::string_view attributeLabel(cudnnBackendAttributeName_t name)
{
    const std::optional<AttributeInfo> info = attributeInfo(name);
    return info ? std::string_view(info->name) : std::string_view("unknown attribute");
}

template <typename T>
T requireAttribute(const std::optional<T>& value, cudnnBackendAttributeName_t name)
{
    return v8::requireValue(value, attributeLabel(name));
}

class TensorDescriptor final : public Descriptor
{
public:
    TensorDescriptor()
        : Descriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR)
    {
    }

    // Null until finalized.
    const std::shared_ptr<const v8::TensorRecord>& record() const noexcept
    {
        return _record;
    }

    // Lowering facts; valid once finalized.
    DataType_t dataType() const noexcept
    {
        return _record->dataType();
    }

    const std::vector<std::int64_t>& dimensions() const noexcept
    {
        return _record->dimensions();
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _record->strides();
    }

    std::int64_t uniqueId() const noexcept
    {
        return _record->uniqueId();
    }

    bool isVirtual() const noexcept
    {
        return _record->isVirtual();
    }

    std::optional<std::int64_t> forwardedAlignment() const noexcept
    {
        return _record->forwardedAlignment();
    }

    bool isRuntimePassByValue() const noexcept
    {
        return _record->isRuntimePassByValue();
    }

    const std::optional<std::vector<std::byte>>& constantValue() const noexcept
    {
        return _record->constantValue();
    }

    const std::shared_ptr<TensorDescriptor>& raggedOffset() const noexcept
    {
        return _raggedOffset;
    }

protected:
    void finalizeImpl() override
    {
        static const v8::TensorLabels s_labels{
            attributeLabel(CUDNN_ATTR_TENSOR_DATA_TYPE),
            attributeLabel(CUDNN_ATTR_TENSOR_DIMENSIONS),
            attributeLabel(CUDNN_ATTR_TENSOR_STRIDES),
            attributeLabel(CUDNN_ATTR_TENSOR_UNIQUE_ID),
            attributeLabel(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT),
            attributeLabel(CUDNN_ATTR_TENSOR_VECTOR_COUNT),
            attributeLabel(CUDNN_ATTR_TENSOR_REORDERING_MODE),
            attributeLabel(CUDNN_ATTR_TENSOR_CONSTANT_VALUE),
        };

        std::shared_ptr<TensorDescriptor> raggedOffset;
        if(std::shared_ptr<Descriptor> referent
           = descriptorAttribute(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC))
        {
            descriptorCast<TensorDescriptor>(*referent, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
            raggedOffset = std::static_pointer_cast<TensorDescriptor>(std::move(referent));
        }

        v8::TensorInputs inputs;
        inputs.dataType = scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_TENSOR_DATA_TYPE);
        inputs.dimensions = arrayAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_DIMENSIONS);
        inputs.strides = arrayAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_STRIDES);
        inputs.uniqueId = scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_UNIQUE_ID);
        inputs.alignment = scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
        inputs.vectorCount
            = scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_VECTOR_COUNT).value_or(1);
        inputs.reorderingMode
            = scalarAttribute<cudnnBackendTensorReordering_t>(CUDNN_ATTR_TENSOR_REORDERING_MODE)
                  .value_or(CUDNN_TENSOR_REORDERING_NONE);
        inputs.isVirtual = scalarAttribute<bool>(CUDNN_ATTR_TENSOR_IS_VIRTUAL).value_or(false);
        inputs.isByValue = scalarAttribute<bool>(CUDNN_ATTR_TENSOR_IS_BY_VALUE).value_or(false);
        if(hasAttribute(CUDNN_ATTR_TENSOR_CONSTANT_VALUE))
        {
            inputs.constantValue
                = scalarAttribute<const void*>(CUDNN_ATTR_TENSOR_CONSTANT_VALUE).value_or(nullptr);
        }
        if(raggedOffset)
        {
            inputs.raggedOffset = raggedOffset->record();
        }

        std::shared_ptr<const v8::TensorRecord> record
            = v8::makeTensorRecord(std::move(inputs), s_labels);
        if(record->constantValue())
        {
            // GET must not hand back the caller's pointer: it refers to builder
            // storage that is gone once the builder is.
            const void* owned = record->constantValue()->data();
            storeComputed(CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                          CUDNN_TYPE_VOID_PTR,
                          1,
                          static_cast<const void*>(&owned));
        }
        _record = std::move(record);
        _raggedOffset = std::move(raggedOffset);
    }

private:
    std::shared_ptr<const v8::TensorRecord> _record;
    std::shared_ptr<TensorDescriptor> _raggedOffset;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
