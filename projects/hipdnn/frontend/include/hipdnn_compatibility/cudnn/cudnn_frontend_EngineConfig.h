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
 *        shim's v8 core.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_EngineConfig.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,readability-simplify-boolean-expr):
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
    EngineConfig_v8() = default;
    EngineConfig_v8(const EngineConfig_v8&) = delete;
    EngineConfig_v8& operator=(const EngineConfig_v8&) = delete;

    ManagedOpaqueDescriptor engine = nullptr;
    int64_t numKnobs = 0;
    std::string opGraphTag;
    bool set_knobs_attr = false;
    detail::v8::KnobChoices knobChoices;
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

        m_engine_config.knobChoices.clear();
        for(const auto& knob : knobs)
        {
            m_engine_config.knobChoices.emplace_back(knob.getKnobType(), knob.getChoice());
        }

        return *this;
    }

    EngineConfig_v8&& build()
    {
        if(m_engine_config.engine == nullptr)
        {
            set_error_and_throw_exception(&m_engine_config,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "EngineConfigBuilder: an engine must be set");
            return std::move(m_engine_config);
        }

        ManagedOpaqueDescriptor built;
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            m_engine_config.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            std::shared_ptr<const detail::v8::EngineRecord> engine
                = detail::v8::recordOf<detail::v8::EngineRecord>(m_engine_config.engine);
            if(!engine)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "EngineConfigBuilder: the engine is not a built Engine");
            }
            // Every knob's type reaches the config, sent or not, as upstream's knob choices do.
            for(const auto& choice : m_engine_config.knobChoices)
            {
                detail::v8::requireEnumerator(choice.first, "EngineConfigBuilder: knob type");
            }
            // Knob choices are sent only once a reported knob is set.
            detail::v8::KnobChoices requested;
            if(m_engine_config.set_knobs_attr && m_engine_config.numKnobs > 0)
            {
                requested = m_engine_config.knobChoices;
            }
            std::shared_ptr<const detail::v8::ConfigRecord> config = detail::v8::makeConfigRecord(
                std::move(engine), std::move(requested), "EngineConfigBuilder");
            // Builds the knob config now, so a knob-config failure surfaces here rather than at
            // plan build.
            hipdnnBackendDescriptor_t view = config->hipdnnEngineConfig();
            built = detail::v8::makeManaged(
                CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, std::move(config), view);
        });
        if(!outcome)
        {
            set_error_and_throw_exception(
                &m_engine_config, outcome.status, outcome.message.c_str());
            return std::move(m_engine_config);
        }

        m_engine_config.pointer = std::move(built);
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

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,readability-simplify-boolean-expr)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
