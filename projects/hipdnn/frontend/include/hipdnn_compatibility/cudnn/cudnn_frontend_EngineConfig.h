// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_EngineConfig.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_EngineConfig.h
 * @brief cuDNN v8 `EngineConfig_v8` / `EngineConfigBuilder_v8` and `EngineConfigList` over the
 *        shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_EngineConfig.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-loop-convert,readability-redundant-member-init,readability-simplify-boolean-expr,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class EngineConfig_v8 : public BackendDescriptor
{
public:
    friend class EngineConfigBuilder_v8;

    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR :";
        ss << " Number of knobs: " << numKnobs;
        return ss.str();
    }

    EngineConfig_v8& operator=(EngineConfig_v8&& from) = default;

    EngineConfig_v8(EngineConfig_v8&& from) = default;

    ~EngineConfig_v8() = default;

    const std::string& getTag() const
    {
        return opGraphTag;
    }

private:
    EngineConfig_v8()
        : BackendDescriptor()
    {
        cudnnStatus_t status;
        for(size_t i = 0; i < bChoices.size(); i++)
        {
            bChoices[i] = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            if(bChoices[i]->is_good() == false)
            {
                status = bChoices[i]->get_status();
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                    "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR cudnnCreate Failed");
                break;
            }
        }
    }
    EngineConfig_v8(const EngineConfig_v8&) = delete;
    EngineConfig_v8& operator=(const EngineConfig_v8&) = delete;

    ManagedOpaqueDescriptor engine = nullptr;
    int64_t numKnobs = 0;
    std::string opGraphTag;
    bool set_knobs_attr = false;
    std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> bChoices = {};
};

class EngineConfigBuilder_v8
{
public:
    auto setEngine(const Engine_v8& engine_) -> EngineConfigBuilder_v8&
    {
        m_engine_config.engine = engine_.get_desc();
        m_engine_config.opGraphTag = engine_.getTag();
        auto& knobs = engine_.getFinalizedKnobs();
        m_engine_config.numKnobs = static_cast<int64_t>(knobs.size());

        m_engine_config.set_knobs_attr = engine_.knobs_set();

        for(std::uint32_t i = 0; i < knobs.size(); i++)
        {
            cudnnStatus_t status;
            cudnnBackendKnobType_t type = knobs[i].getKnobType();
            int64_t value = knobs[i].getChoice();
            status = detail::set_attribute(m_engine_config.bChoices[i]->get_backend_descriptor(),
                                           CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                           CUDNN_TYPE_KNOB_TYPE,
                                           1,
                                           &type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR SetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE Failed");
            }
            status = detail::set_attribute(m_engine_config.bChoices[i]->get_backend_descriptor(),
                                           CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &value);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR SetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE Failed");
            }
            status = detail::finalize(m_engine_config.bChoices[i]->get_backend_descriptor());
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR cudnnFinalize "
                                              "Failed");
            }
        }

        return *this;
    }

    EngineConfig_v8&& build()
    {
        if(m_engine_config.status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_engine_config,
                m_engine_config.status,
                "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: is not created properly");
            return std::move(m_engine_config);
        }
        if(m_engine_config.engine == nullptr)
        {
            set_error_and_throw_exception(&m_engine_config,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE.");
            return std::move(m_engine_config);
        }
        auto status = m_engine_config.initialize_managed_backend_pointer(
            CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_engine_config, status, "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_engine_config);
        }

        status = detail::set_attribute(m_engine_config.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_engine_config.engine->get_backend_descriptor()));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_engine_config,
                                          status,
                                          "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_engine_config);
        }

        if(m_engine_config.set_knobs_attr && m_engine_config.numKnobs > 0)
        {
            std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bChoices_;
            for(auto i = 0; i < m_engine_config.numKnobs; i++)
            {
                bChoices_[static_cast<size_t>(i)]
                    = m_engine_config.bChoices[static_cast<size_t>(i)]->get_backend_descriptor();
            }
            status = detail::set_attribute(m_engine_config.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           m_engine_config.numKnobs,
                                           bChoices_.data());
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES Failed");
                return std::move(m_engine_config);
            }
        }

        status = detail::finalize(m_engine_config.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_engine_config,
                status,
                "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_engine_config);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_engine_config);
        return std::move(m_engine_config);
    }

    explicit EngineConfigBuilder_v8() = default;
    ~EngineConfigBuilder_v8() = default;
    EngineConfigBuilder_v8(EngineConfigBuilder_v8&&) = delete;
    EngineConfigBuilder_v8(const EngineConfigBuilder_v8&) = delete;
    EngineConfigBuilder_v8& operator=(const EngineConfigBuilder_v8&) = delete;

private:
    EngineConfig_v8 m_engine_config;
};

using EngineConfigList = std::vector<ManagedOpaqueDescriptor>;

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-loop-convert,readability-redundant-member-init,readability-simplify-boolean-expr,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
