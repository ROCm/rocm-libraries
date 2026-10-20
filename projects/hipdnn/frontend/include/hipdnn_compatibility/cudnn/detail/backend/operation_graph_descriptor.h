// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/lowering.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

class OperationGraphDescriptor final : public Descriptor
{
public:
    OperationGraphDescriptor()
        : Descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::GraphCore>& requireCore() const
    {
        if(!_core)
        {
            throw ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                            "the engine index space needs a finalized "
                            "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR");
        }
        return _core;
    }

    cudnnHandle_t handle() const noexcept
    {
        return _core ? _core->handle() : nullptr;
    }

    hipdnnBackendDescriptor_t hipdnnGraph() const noexcept
    {
        return _core ? _core->hipdnnGraph() : nullptr;
    }

    std::int64_t engineCount() const
    {
        return requireCore()->engineCount();
    }

    const std::vector<std::int64_t>& rankedEngineIds() const
    {
        return requireCore()->rankedEngineIds();
    }

    std::optional<std::int64_t> engineIdAt(std::int64_t index) const
    {
        return requireCore()->engineIdAt(index);
    }

    std::optional<std::int64_t> engineIndexOf(std::int64_t id) const
    {
        return requireCore()->engineIndexOf(id);
    }

    hipdnnBackendDescriptor_t defaultEngineConfig(std::int64_t id) const
    {
        return requireCore()->defaultEngineConfig(id);
    }

    hipdnnBackendDescriptor_t defaultEngine(std::int64_t id) const
    {
        return requireCore()->defaultEngine(id);
    }

protected:
    void finalizeImpl() override
    {
        static const v8::LoweringLabels s_labels{
            attributeName(CUDNN_ATTR_OPERATIONGRAPH_HANDLE),
            attributeName(CUDNN_ATTR_OPERATIONGRAPH_OPS),
            attributeName(CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED),
            attributeName(CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED),
            attributeName(CUDNN_ATTR_TENSOR_CONSTANT_VALUE)};
        const v8::GraphFlags flags{
            scalarAttribute<bool>(CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED)
                .value_or(false),
            scalarAttribute<bool>(CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED)
                .value_or(false)};
        const std::vector<std::shared_ptr<Descriptor>> operations
            = descriptorArrayAttribute(CUDNN_ATTR_OPERATIONGRAPH_OPS);
        std::vector<std::shared_ptr<const v8::OperationRecord>> records;
        records.reserve(operations.size());
        for(const std::shared_ptr<Descriptor>& operation : operations)
        {
            records.push_back(operationCast(*operation).operationRecord());
        }
        _core = v8::lowerGraph(
            records,
            scalarAttribute<cudnnHandle_t>(CUDNN_ATTR_OPERATIONGRAPH_HANDLE).value_or(nullptr),
            flags,
            s_labels,
            {});
    }

    std::optional<StoredAttribute> lazyAttribute(cudnnBackendAttributeName_t name,
                                                 bool countOnly) const override
    {
        if(name != CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT)
        {
            return std::nullopt;
        }
        // The element count of this scalar is always 1; only the value needs the
        // engine sweep (and a device).
        const std::int64_t count = countOnly ? 0 : engineCount();
        StoredAttribute value;
        value.type = CUDNN_TYPE_INT64;
        value.count = 1;
        value.bytes.resize(sizeof(count));
        std::memcpy(value.bytes.data(), &count, sizeof(count));
        return value;
    }

private:
    std::shared_ptr<const v8::GraphCore> _core;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
