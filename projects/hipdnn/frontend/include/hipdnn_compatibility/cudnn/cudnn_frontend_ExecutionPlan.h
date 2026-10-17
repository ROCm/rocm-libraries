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
 * @brief cuDNN v8 `ExecutionPlan_v8` / `ExecutionPlanBuilder_v8` over the shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_ExecutionPlan.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-redundant-void-arg,readability-braces-around-statements,readability-isolate-declaration,readability-container-data-pointer,bugprone-multi-level-implicit-pointer-conversion,performance-unnecessary-value-param):
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

    std::string getJsonRepresentation() const
    {
        auto status = CUDNN_STATUS_SUCCESS;
        int64_t serializationSize;
        std::vector<char> serialization_buf;
        status = detail::get_attribute(pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       0,
                                       &serializationSize,
                                       nullptr);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
        }
        serialization_buf.resize(static_cast<size_t>(serializationSize));
        status = detail::get_attribute(pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       serializationSize,
                                       &serializationSize,
                                       serialization_buf.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
        }
        std::string json_string(serialization_buf.begin(), serialization_buf.end());
        return json_string;
    }

    ExecutionPlan_v8(const ExecutionPlan_v8&) = default;
    ExecutionPlan_v8& operator=(const ExecutionPlan_v8&) = default;

private:
    void fetchNotes(ManagedOpaqueDescriptor& extractedEngine)
    {
        auto status = CUDNN_STATUS_SUCCESS;
        int64_t elem_count = 0;
        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        status = detail::get_attribute(extractedEngine_,
                                       CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                       CUDNN_TYPE_NUMERICAL_NOTE,
                                       CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                       &elem_count,
                                       nullptr);
        numeric_notes_vec.resize(static_cast<size_t>(elem_count));
        status = detail::get_attribute(extractedEngine_,
                                       CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                       CUDNN_TYPE_NUMERICAL_NOTE,
                                       CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                       &elem_count,
                                       numeric_notes_vec.data());
        ptrdiff_t end = static_cast<ptrdiff_t>(
            std::min(elem_count, static_cast<int64_t>(CUDNN_NUMERICAL_NOTE_TYPE_COUNT)));
        std::copy(
            numeric_notes_vec.begin(), numeric_notes_vec.begin() + end, numeric_notes.begin());
        if(static_cast<size_t>(elem_count) < numeric_notes.size())
            std::fill_n(numeric_notes.begin() + static_cast<ptrdiff_t>(elem_count),
                        numeric_notes.size() - static_cast<size_t>(elem_count),
                        CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_NUMERICAL_NOTE Failed");
        }
        status = detail::get_attribute(extractedEngine_,
                                       CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                       CUDNN_TYPE_BEHAVIOR_NOTE,
                                       CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                       &elem_count,
                                       nullptr);
        behavior_notes_vec.resize(static_cast<size_t>(elem_count));
        status = detail::get_attribute(extractedEngine_,
                                       CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                       CUDNN_TYPE_BEHAVIOR_NOTE,
                                       CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                       &elem_count,
                                       behavior_notes_vec.data());
        end = static_cast<ptrdiff_t>(
            std::min(elem_count, static_cast<int64_t>(CUDNN_BEHAVIOR_NOTE_TYPE_COUNT)));
        std::copy(
            behavior_notes_vec.begin(), behavior_notes_vec.begin() + end, behavior_notes.begin());
        if(static_cast<size_t>(elem_count) < behavior_notes.size())
            std::fill_n(behavior_notes.begin() + static_cast<ptrdiff_t>(elem_count),
                        behavior_notes.size() - static_cast<size_t>(elem_count),
                        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE Failed");
        }
    }

    void buildTag(ManagedOpaqueDescriptor& extractedEngine)
    {
        auto status = CUDNN_STATUS_SUCCESS;
        std::stringstream tag{""};
        int64_t elemCount = 0, engineId = 0, numKnobs = 0;

        std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs{{nullptr}};
        for(auto& knob : extractedKnobs)
        {
            knob = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            status = knob->get_status();
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate "
                    "Failed when compute tag");
            }
        }

        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs_{{nullptr}};
        for(std::uint32_t i = 0; i < extractedKnobs.size(); i++)
        {
            extractedKnobs_[i] = extractedKnobs[i]->get_backend_descriptor();
        }

        status = detail::get_attribute(extractedEngine_,
                                       CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &elemCount,
                                       &engineId);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
        }
        tag << "eng" << engineId;

        status = detail::get_attribute(engine_config->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       CUDNN_KNOB_TYPE_COUNTS,
                                       &numKnobs,
                                       &(extractedKnobs_[0]));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES Failed");
        }
        if(numKnobs > CUDNN_KNOB_TYPE_COUNTS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "numKnobs exceed the CUDNN_KNOB_TYPE_COUNTS");
        }
        for(size_t idx = 0; idx < static_cast<size_t>(numKnobs); ++idx)
        {
            const cudnnBackendDescriptor_t& knob = extractedKnobs_[idx];
            cudnnBackendKnobType_t type = CUDNN_KNOB_TYPE_COUNTS;
            int64_t choice = -2;
            status = detail::get_attribute(
                knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, nullptr, &type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(this,
                                              status,
                                              "computeTag CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: "
                                              "GetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE Failed");
            }
            status = detail::get_attribute(
                knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, nullptr, &choice);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                    "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE Failed");
            }
            tag << "_k" << type << "=" << choice;
        }
        planTag += tag.str();
    }

    void computeWorkSpaceSize()
    {
        auto status = detail::get_attribute(pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                            CUDNN_TYPE_INT64,
                                            1,
                                            nullptr,
                                            &workSpaceSize);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE Failed");
        }
        if(workSpaceSize < 0)
        {
            set_error_and_throw_exception(
                this,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute Workspace Size Invalid");
        }
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
        if(m_execution_plan.engine_config == nullptr)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set "
                                          "the CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");
            return std::move(m_execution_plan);
        };

        auto status = m_execution_plan.initialize_managed_backend_pointer(
            CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        status = detail::set_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_execution_plan.engine_config->get_backend_descriptor()));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }

#if(CUDNN_VERSION >= 90400)
        if(m_execution_plan.kernel_cache)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          CUDNN_STATUS_NOT_SUPPORTED,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE is not supported "
                                          "by the hipDNN cuDNN shim");
            return std::move(m_execution_plan);
        }
#endif
        status = detail::finalize(m_execution_plan.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }

        ManagedOpaqueDescriptor extractedEngine
            = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        status = extractedEngine->get_status();
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR failed when compute tag");
            return std::move(m_execution_plan);
        }
        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        int64_t elemCount = 0;
        status = detail::get_attribute(m_execution_plan.engine_config->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elemCount,
                                       &extractedEngine_);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.buildTag(extractedEngine);
        m_execution_plan.fetchNotes(extractedEngine);
        m_execution_plan.computeWorkSpaceSize();

        CUDNN_FE_LOG_LABEL_ENDL(m_execution_plan);
        return std::move(m_execution_plan);
    }

    ExecutionPlan_v8&& loadFromJson(const std::string& json_plan)
    {
        CUDNN_FRONTEND_UNUSED(json_plan);
        auto status = CUDNN_STATUS_SUCCESS;

        if(m_execution_plan.handle == nullptr)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set "
                                          "the CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
            return std::move(m_execution_plan);
        };

        status = m_execution_plan.initialize_managed_backend_pointer(
            CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        std::vector<char> serialization_buf;
        serialization_buf.assign(json_plan.begin(), json_plan.end());
        status = detail::set_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       static_cast<int64_t>(serialization_buf.size()),
                                       serialization_buf.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
            return std::move(m_execution_plan);
        }

        status = detail::set_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                       CUDNN_TYPE_HANDLE,
                                       1,
                                       &m_execution_plan.handle);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_HANDLE Failed");
            return std::move(m_execution_plan);
        }

        status = detail::finalize(m_execution_plan.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.engine_config
            = make_shared_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        status = m_execution_plan.engine_config->get_status();
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                                          "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR failed when "
                                          "computing tag");
            return std::move(m_execution_plan);
        }

        cudnnBackendDescriptor_t engCfgDesc
            = m_execution_plan.engine_config->get_backend_descriptor();
        int64_t elemCount = 0;
        status = detail::get_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elemCount,
                                       &engCfgDesc);

        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }
        ManagedOpaqueDescriptor extractedEngine
            = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        status = extractedEngine->get_status();
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR failed when computing "
                                          "tag");
            return std::move(m_execution_plan);
        }

        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();

        status = detail::get_attribute(m_execution_plan.engine_config->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elemCount,
                                       &extractedEngine_);

        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.buildTag(extractedEngine);
        m_execution_plan.fetchNotes(extractedEngine);
        m_execution_plan.computeWorkSpaceSize();

        CUDNN_FE_LOG_LABEL_ENDL(m_execution_plan);
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8() = default;
    ~ExecutionPlanBuilder_v8() = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8&&) = delete;
    ExecutionPlanBuilder_v8(const ExecutionPlanBuilder_v8&) = delete;
    ExecutionPlanBuilder_v8& operator=(const ExecutionPlanBuilder_v8&) = delete;

private:
    ExecutionPlan_v8 m_execution_plan;
};

using ExecutionPlan = ExecutionPlan_v8;
using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-redundant-void-arg,readability-braces-around-statements,readability-isolate-declaration,readability-container-data-pointer,bugprone-multi-level-implicit-pointer-conversion,performance-unnecessary-value-param)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
