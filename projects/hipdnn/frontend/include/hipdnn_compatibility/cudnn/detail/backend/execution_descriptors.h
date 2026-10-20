// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/engine_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/v8/plan.h>
#include <hipdnn_compatibility/cudnn/detail/v8/variant_pack.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

inline constexpr v8::PlanLabels PLAN_LABELS{"CUDNN_ATTR_EXECUTION_PLAN_HANDLE",
                                            "CUDNN_ATTR_OPERATIONGRAPH_HANDLE",
                                            "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION"};

inline constexpr v8::VariantPackLabels VARIANT_PACK_LABELS{"CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",
                                                           "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS"};

// The plan JSON format as the C API reads it: failures name the JSON attribute.
namespace plan_json
{
using v8::plan_json::Plan;
using v8::plan_json::write;

inline std::optional<Plan> read(std::string_view text)
{
    return v8::plan_json::read(text, PLAN_LABELS.json);
}
} // namespace plan_json

class ExecutionPlanDescriptor final : public Descriptor
{
public:
    ExecutionPlanDescriptor()
        : Descriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<const v8::PlanRecord>& record() const noexcept
    {
        return _record;
    }

    cudnnHandle_t contextHandle() const noexcept
    {
        return _record ? _record->contextHandle() : nullptr;
    }

    hipdnnBackendDescriptor_t hipdnnPlan() const noexcept
    {
        return _record ? _record->hipdnnPlan() : nullptr;
    }

protected:
    void finalizeImpl() override
    {
        const std::optional<cudnnHandle_t> planHandle
            = scalarAttribute<cudnnHandle_t>(CUDNN_ATTR_EXECUTION_PLAN_HANDLE);
        const cudnnHandle_t handle = planHandle ? *planHandle : nullptr;
        const std::shared_ptr<Descriptor> config
            = descriptorAttribute(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        const std::optional<std::string> json
            = stringAttribute(CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION);
        if((config != nullptr) == json.has_value())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "exactly one of CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG and "
                            "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION must be set");
        }

        std::shared_ptr<const v8::PlanRecord> plan;
        if(config != nullptr)
        {
            const auto& engineConfig = descriptorCast<EngineConfigDescriptor>(
                *config, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            plan = v8::buildPlan(engineConfig.record(), handle, PLAN_LABELS);
        }
        else
        {
            plan = v8::loadPlan(handle, *json, PLAN_LABELS);
            storeComputedDescriptors(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                     {EngineConfigDescriptor::forLoadedPlan(plan->configRecord())});
        }

        const std::int64_t workspaceSize = plan->workspaceSize();
        storeComputed(
            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &workspaceSize);
        _record = std::move(plan);
    }

    std::optional<StoredAttribute> lazyAttribute(cudnnBackendAttributeName_t name,
                                                 bool /*countOnly*/) const override
    {
        if(name != CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION)
        {
            return std::nullopt;
        }
        const std::string& json = _record->json();
        StoredAttribute value;
        value.type = CUDNN_TYPE_CHAR;
        value.count = static_cast<std::int64_t>(json.size() + 1);
        value.bytes.resize(json.size());
        std::memcpy(value.bytes.data(), json.data(), json.size());
        return value;
    }

private:
    std::shared_ptr<const v8::PlanRecord> _record;
};

class VariantPackDescriptor final : public Descriptor
{
public:
    VariantPackDescriptor()
        : Descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<const v8::VariantPackRecord>& record() const noexcept
    {
        return _record;
    }

    hipdnnBackendDescriptor_t hipdnnVariantPack() const noexcept
    {
        return _record ? _record->hipdnnVariantPack() : nullptr;
    }

protected:
    void finalizeImpl() override
    {
        const StoredAttribute* pointers = findAttribute(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
        const StoredAttribute* uids = findAttribute(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
        void* const workspace
            = scalarAttribute<void*>(CUDNN_ATTR_VARIANT_PACK_WORKSPACE).value_or(nullptr);
        // Stored bytes come from operator new, so they are aligned for any element type.
        _record = v8::buildVariantPack(
            pointers != nullptr ? reinterpret_cast<void* const*>(pointers->bytes.data()) : nullptr,
            pointers != nullptr ? pointers->count : 0,
            uids != nullptr ? reinterpret_cast<const std::int64_t*>(uids->bytes.data()) : nullptr,
            uids != nullptr ? uids->count : 0,
            workspace,
            VARIANT_PACK_LABELS);
    }

private:
    std::shared_ptr<const v8::VariantPackRecord> _record;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
