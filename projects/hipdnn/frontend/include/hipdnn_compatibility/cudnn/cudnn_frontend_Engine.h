// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Engine.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Engine.h
 * @brief cuDNN v8 `Engine_v8` / `EngineBuilder_v8` over the shim's v8 core.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Engine.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,performance-unnecessary-value-param,readability-isolate-declaration,readability-simplify-boolean-expr):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class Engine_v8 : public BackendDescriptor
{
private:
    Engine_v8() = default;
    Engine_v8(const Engine_v8&) = delete;
    Engine_v8& operator=(const Engine_v8&) = delete;

    class Knob
    {
    public:
        Knob(cudnnBackendKnobType_t type_, int64_t max, int64_t min, int64_t stride_)
            : knobType(type_)
            , maxValue(max)
            , minValue(min)
            , stride(stride_)
        {
        }

        std::string describe() const
        {
            std::stringstream ss;
            ss << "Knob:" << knobType;
            ss << " Min: " << minValue;
            ss << " Max: " << maxValue;
            ss << " Stride: " << stride;
            return ss.str();
        }

        void setChoice(uint64_t val_)
        {
            choice = static_cast<int64_t>(val_);
        }

        int64_t getChoice() const
        {
            return choice;
        }

        cudnnBackendKnobType_t getKnobType() const
        {
            return knobType;
        }

        int64_t getMinValue() const
        {
            return minValue;
        }

        int64_t getMaxValue() const
        {
            return maxValue;
        }

        int64_t getStride() const
        {
            return stride;
        }

    private:
        cudnnBackendKnobType_t knobType = CUDNN_KNOB_TYPE_COUNTS;
        int64_t maxValue = 0, minValue = 0, stride = 0;
        int64_t choice = -1;
    };

    ManagedOpaqueDescriptor opGraph = nullptr;
    int64_t idx = -1;
    int64_t numKnobs = 0;
    std::vector<Knob> knobs;
    std::string opGraphTag;

public:
    friend class EngineBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINE_DESCRIPTOR :";
        ss << " ID: " << idx;
        ss << " Has " << numKnobs << " knobs";
        return ss.str();
    }
    Engine_v8(Engine_v8&& from) = default;

    Engine_v8& operator=(Engine_v8&&) = default;
    ~Engine_v8() = default;

    const std::string& getTag() const
    {
        return opGraphTag;
    }

    std::vector<Knob>& getSupportedKnobs()
    {
        return knobs;
    }

    const std::vector<Knob>& getFinalizedKnobs() const
    {
        return knobs;
    }

    bool knobs_set() const
    {
        bool is_knob_set = false;
        for(auto i = 0; i < numKnobs; i++)
        {
            if(knobs[static_cast<size_t>(i)].getChoice() != -1)
            {
                is_knob_set = true;
                break;
            }
        }
        return is_knob_set;
    }
};

class EngineBuilder_v8
{
public:
    auto setOperationGraph(const OperationGraph_v8& opGraph_) -> EngineBuilder_v8&
    {
        m_engine.opGraph = opGraph_.get_desc();
        m_engine.opGraphTag = opGraph_.getTag();
        return *this;
    }

    auto setOperationGraph(ManagedOpaqueDescriptor desc_) -> EngineBuilder_v8&
    {
        m_engine.opGraph = desc_;
        return *this;
    }
    auto setGlobalEngineIdx(int64_t idx_) -> EngineBuilder_v8&
    {
        m_engine.idx = idx_;
        return *this;
    }

    Engine_v8&& build()
    {
        if(m_engine.idx < 0)
        {
            set_error_and_throw_exception(
                &m_engine,
                CUDNN_STATUS_BAD_PARAM,
                "EngineBuilder: the global engine index must be non-negative");
            return std::move(m_engine);
        }
        if(m_engine.opGraph == nullptr)
        {
            set_error_and_throw_exception(
                &m_engine, CUDNN_STATUS_BAD_PARAM, "EngineBuilder: an operation graph must be set");
            return std::move(m_engine);
        }

        ManagedOpaqueDescriptor built;
        std::vector<Engine_v8::Knob> knobs;
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            // Upstream creates the descriptor here, so a failed engine still has a (null) raw
            // view.
            m_engine.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            std::shared_ptr<const detail::v8::GraphCore> graph
                = detail::v8::recordOf<detail::v8::GraphCore>(m_engine.opGraph);
            if(!graph)
            {
                throw detail::v8::ShimError(
                    CUDNN_STATUS_BAD_PARAM,
                    "EngineBuilder: the operation graph is not a built OperationGraph");
            }
            std::shared_ptr<const detail::v8::EngineRecord> record = detail::v8::makeEngineRecord(
                std::move(graph), m_engine.idx, "EngineBuilder: global engine index");
            for(const detail::v8::EngineKnob& knob : record->knobs())
            {
                knobs.emplace_back(
                    knob.type, knob.range.maxValue, knob.range.minValue, knob.range.stride);
            }
            hipdnnBackendDescriptor_t engine = record->hipdnnEngine();
            built = detail::v8::makeManaged(
                CUDNN_BACKEND_ENGINE_DESCRIPTOR, std::move(record), engine);
        });
        if(!outcome)
        {
            set_error_and_throw_exception(&m_engine, outcome.status, outcome.message.c_str());
            return std::move(m_engine);
        }

        m_engine.pointer = std::move(built);
        m_engine.numKnobs = static_cast<int64_t>(knobs.size());
        m_engine.knobs = std::move(knobs);
        CUDNN_FE_LOG_LABEL_ENDL(m_engine);
        return std::move(m_engine);
    }

    explicit EngineBuilder_v8() = default;
    ~EngineBuilder_v8() = default;
    EngineBuilder_v8(EngineBuilder_v8&&) = delete;
    EngineBuilder_v8(const EngineBuilder_v8&) = delete;
    EngineBuilder_v8& operator=(const EngineBuilder_v8&) = delete;

private:
    Engine_v8 m_engine;
};

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,performance-unnecessary-value-param,readability-isolate-declaration,readability-simplify-boolean-expr)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
