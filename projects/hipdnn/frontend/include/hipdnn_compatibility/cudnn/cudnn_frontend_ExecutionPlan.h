// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_ExecutionPlan.h
//     Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_ExecutionPlan.h
 * @brief cuDNN v8 `ExecutionPlan_v8` / `ExecutionPlanBuilder_v8` over hipDNN execution plans.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/plan.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_ExecutionPlan.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-redundant-void-arg,readability-braces-around-statements,performance-unnecessary-value-param):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class ExecutionPlan_v8 : public BackendDescriptor
{
public:
    friend class ExecutionPlanBuilder_v8;

    ExecutionPlan_v8(ExecutionPlan_v8&& from) = default;
    ExecutionPlan_v8& operator=(ExecutionPlan_v8&&) = default;

    ~ExecutionPlan_v8() = default;

    auto getWorkspaceSize(void) const -> int64_t
    {
        return workSpaceSize;
    }

    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR : ";
        ss << getTag() << ", ";
        ss << "numeric_notes:" << "[";
        for(auto note : numeric_notes_vec)
        {
            ss << cudnn_frontend::to_string(note) << ",";
        }
        ss << "] behavior_notes:" << "[";
        for(auto note : behavior_notes_vec)
        {
            ss << cudnn_frontend::to_string(note) << ",";
        }
        ss << "] workSpaceSize: " << workSpaceSize;
        return ss.str();
    }

    const std::string& getTag() const
    {
        return planTag;
    }

    void setExecutionTime(float time_)
    {
        execution_time_ms = time_;
    }

    float getExecutionTime() const
    {
        return execution_time_ms;
    }

    const std::vector<cudnnBackendNumericalNote_t>& getAllNumericNotes() const
    {
        return numeric_notes_vec;
    }

    const std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT>&
        getNumericNotes() const
    {
        return numeric_notes;
    }

    const std::array<cudnnBackendBehaviorNote_t, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT>&
        getBehaviorNotes() const
    {
        return behavior_notes;
    }
    const std::vector<cudnnBackendBehaviorNote_t>& getAllBehaviorNotes() const
    {
        return behavior_notes_vec;
    }

    // Upstream's GET of the JSON attribute counts the NUL terminator, and callers round-trip it.
    std::string getJsonRepresentation() const
    {
        std::string json_string;
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            const std::shared_ptr<const detail::v8::PlanRecord> plan
                = detail::v8::recordOf<detail::v8::PlanRecord>(pointer);
            if(plan == nullptr)
            {
                throw detail::v8::ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                                            "ExecutionPlan: the plan was not built");
            }
            const std::string& json = plan->json();
            json_string.reserve(json.size() + 1);
            json_string = json;
            json_string += '\0';
        });
        if(!outcome)
        {
            set_error_and_throw_exception(this, outcome.status, outcome.message.c_str());
        }
        return json_string;
    }

    ExecutionPlan_v8(const ExecutionPlan_v8&) = default;
    ExecutionPlan_v8& operator=(const ExecutionPlan_v8&) = default;

private:
    void fetchNotes(const detail::v8::PlanRecord& plan)
    {
        numeric_notes_vec = plan.numericalNotes();
        const auto numericCount = static_cast<int64_t>(numeric_notes_vec.size());
        ptrdiff_t end = static_cast<ptrdiff_t>(
            std::min(numericCount, static_cast<int64_t>(CUDNN_NUMERICAL_NOTE_TYPE_COUNT)));
        std::copy(
            numeric_notes_vec.begin(), numeric_notes_vec.begin() + end, numeric_notes.begin());
        if(static_cast<size_t>(numericCount) < numeric_notes.size())
            std::fill_n(numeric_notes.begin() + static_cast<ptrdiff_t>(numericCount),
                        numeric_notes.size() - static_cast<size_t>(numericCount),
                        CUDNN_NUMERICAL_NOTE_TYPE_COUNT);

        behavior_notes_vec = plan.behaviorNotes();
        const auto behaviorCount = static_cast<int64_t>(behavior_notes_vec.size());
        end = static_cast<ptrdiff_t>(
            std::min(behaviorCount, static_cast<int64_t>(CUDNN_BEHAVIOR_NOTE_TYPE_COUNT)));
        std::copy(
            behavior_notes_vec.begin(), behavior_notes_vec.begin() + end, behavior_notes.begin());
        if(static_cast<size_t>(behaviorCount) < behavior_notes.size())
            std::fill_n(behavior_notes.begin() + static_cast<ptrdiff_t>(behaviorCount),
                        behavior_notes.size() - static_cast<size_t>(behaviorCount),
                        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT);
    }

    void buildTag(const detail::v8::PlanRecord& plan)
    {
        planTag += plan.tag();
    }

    void computeWorkSpaceSize(const detail::v8::PlanRecord& plan)
    {
        workSpaceSize = plan.workspaceSize();
        if(workSpaceSize < 0)
        {
            throw detail::v8::ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                                        "hipDNN reported a negative workspace size "
                                            + std::to_string(workSpaceSize));
        }
    }

    // Tag, notes and workspace come from the record; the plan then views its hipDNN plan.
    void adopt(std::shared_ptr<const detail::v8::PlanRecord> plan)
    {
        buildTag(*plan);
        fetchNotes(*plan);
        computeWorkSpaceSize(*plan);
        const cudnnBackendDescriptor_t view = plan->hipdnnPlan();
        pointer = detail::v8::makeManaged(
            CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, std::move(plan), view);
    }

    ExecutionPlan_v8() = default;
    ManagedOpaqueDescriptor engine_config = nullptr;
    cudnnHandle_t handle = nullptr;
    std::string planTag;

    std::int64_t workSpaceSize = 0;
    std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT> numeric_notes;
    std::vector<cudnnBackendNumericalNote_t> numeric_notes_vec;
    std::array<cudnnBackendBehaviorNote_t, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT> behavior_notes;
    std::vector<cudnnBackendBehaviorNote_t> behavior_notes_vec;

    float execution_time_ms = 0.0f;
    std::shared_ptr<KernelCache> kernel_cache = nullptr;
};

class ExecutionPlanBuilder_v8
{
public:
    auto setHandle(cudnnHandle_t handle_) -> ExecutionPlanBuilder_v8&
    {
        m_execution_plan.handle = handle_;
        return *this;
    }

    auto setEngineConfig(const EngineConfig_v8& engine_config_) -> ExecutionPlanBuilder_v8&
    {
        m_execution_plan.engine_config = engine_config_.get_desc();
        m_execution_plan.planTag = engine_config_.getTag();
        return *this;
    }

    auto setKernelCache(std::shared_ptr<KernelCache> kernel_cache) -> ExecutionPlanBuilder_v8&
    {
        m_execution_plan.kernel_cache = kernel_cache;
        return *this;
    }

    auto setEngineConfig(ManagedOpaqueDescriptor& desc,
                         const std::string& opGraphTag_ = "") -> ExecutionPlanBuilder_v8&
    {
        m_execution_plan.engine_config = desc;
        m_execution_plan.planTag = opGraphTag_;
        return *this;
    }

    auto setEngineConfig(const ManagedOpaqueDescriptor& desc,
                         const std::string& opGraphTag_ = "") -> ExecutionPlanBuilder_v8&
    {
        m_execution_plan.engine_config = desc;
        m_execution_plan.planTag = opGraphTag_;
        return *this;
    }

    ExecutionPlan_v8&& build()
    {
        // Created first so that even a plan failing a precondition has a (null) raw descriptor.
        auto status = m_execution_plan.initialize_managed_backend_pointer(
            CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan, status, "ExecutionPlanBuilder: creating the plan failed");
            return std::move(m_execution_plan);
        }

        if(m_execution_plan.engine_config == nullptr)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "ExecutionPlanBuilder: an engine config is required");
            return std::move(m_execution_plan);
        };

        std::shared_ptr<const detail::v8::ConfigRecord> config
            = detail::v8::recordOf<detail::v8::ConfigRecord>(m_execution_plan.engine_config);
        if(config == nullptr)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                CUDNN_STATUS_BAD_PARAM,
                "ExecutionPlanBuilder: the engine config must be a built engine config");
            return std::move(m_execution_plan);
        }

#if(CUDNN_VERSION >= 90400)
        if(m_execution_plan.kernel_cache)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          CUDNN_STATUS_NOT_SUPPORTED,
                                          "ExecutionPlanBuilder: a kernel cache (setKernelCache) "
                                          "is not supported by the hipDNN cuDNN shim");
            return std::move(m_execution_plan);
        }
#endif
        // Upstream's build() never sends the plan handle; the plan runs on its graph's handle.
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            m_execution_plan.adopt(detail::v8::buildPlan(std::move(config), nullptr, PLAN_LABELS));
        });
        if(!outcome)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          outcome.status,
                                          ("ExecutionPlanBuilder: " + outcome.message).c_str());
            return std::move(m_execution_plan);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_execution_plan);
        return std::move(m_execution_plan);
    }

    ExecutionPlan_v8&& loadFromJson(const std::string& json_plan)
    {
        auto status = m_execution_plan.initialize_managed_backend_pointer(
            CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan, status, "ExecutionPlanBuilder: creating the plan failed");
            return std::move(m_execution_plan);
        }

        if(m_execution_plan.handle == nullptr)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                CUDNN_STATUS_BAD_PARAM,
                "ExecutionPlanBuilder: loadFromJson requires a handle (setHandle)");
            return std::move(m_execution_plan);
        };

        // getJsonRepresentation() ends in a NUL; the JSON is the text before it.
        std::string_view text(json_plan);
        text = text.substr(0, text.find('\0'));
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            std::shared_ptr<const detail::v8::PlanRecord> plan
                = detail::v8::loadPlan(m_execution_plan.handle, text, PLAN_LABELS);
            m_execution_plan.engine_config
                = detail::v8::makeManaged(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, plan->configRecord());
            m_execution_plan.adopt(std::move(plan));
        });
        if(!outcome)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          outcome.status,
                                          ("ExecutionPlanBuilder: " + outcome.message).c_str());
            return std::move(m_execution_plan);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_execution_plan);
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8() = default;
    ~ExecutionPlanBuilder_v8() = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8&&) = delete;
    ExecutionPlanBuilder_v8(const ExecutionPlanBuilder_v8&) = delete;
    ExecutionPlanBuilder_v8& operator=(const ExecutionPlanBuilder_v8&) = delete;

private:
    static constexpr detail::v8::PlanLabels PLAN_LABELS{
        "the handle (setHandle)", "handle", "loadFromJson's plan JSON"};

    ExecutionPlan_v8 m_execution_plan;
};

using ExecutionPlan = ExecutionPlan_v8;
using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-redundant-void-arg,readability-braces-around-statements,performance-unnecessary-value-param)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
