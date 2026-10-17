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
 * @brief cuDNN v8 `Engine_v8` / `EngineBuilder_v8` over the shim's backend C API.
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

#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Engine.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-use-emplace,modernize-loop-convert,performance-unnecessary-value-param,readability-isolate-declaration,readability-simplify-boolean-expr,bugprone-multi-level-implicit-pointer-conversion):
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
    std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> bKnobs = {};
    std::vector<Knob> knobs;
    std::string opGraphTag;

    void buildKnobs()
    {
        cudnnStatus_t status;
        for(auto i = 0; i < numKnobs; i++)
        {
            auto bKnob = bKnobs[static_cast<size_t>(i)]->get_backend_descriptor();
            cudnnBackendKnobType_t type;
            int64_t maxValue, minValue, stride, elemCount;
            status = detail::get_attribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &elemCount, &type);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_TYPE failed");
            }
            status = detail::get_attribute(bKnob,
                                           CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &elemCount,
                                           &maxValue);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_ENGINE_DESCRIPTOR: "
                    "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                    "GetAttribute CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE Failed");
            }
            status = detail::get_attribute(bKnob,
                                           CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &elemCount,
                                           &minValue);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_ENGINE_DESCRIPTOR: "
                    "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                    "GetAttribute CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE Failed");
            }
            status = detail::get_attribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_STRIDE Failed");
            }
            knobs.emplace_back(Knob(type, maxValue, minValue, stride));
        }
    }

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
            set_error_and_throw_exception(&m_engine,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINE_GLOBAL_INDEX to valid value");
            return std::move(m_engine);
        }
        if(m_engine.opGraph == nullptr)
        {
            set_error_and_throw_exception(&m_engine,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set "
                                          "CUDNN_ATTR_ENGINE_OPERATION_GRAPH to valid value");
            return std::move(m_engine);
        }

        auto status = m_engine.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_engine,
                status,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_engine);
        }

        status = detail::set_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_engine.opGraph->get_backend_descriptor()));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_engine,
                                          status,
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_ENGINE_OPERATION_GRAPH Failed");
            return std::move(m_engine);
        }

        status = detail::set_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_engine.idx);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_engine,
                                          status,
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
            return std::move(m_engine);
        }

        status = detail::finalize(m_engine.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_engine);
        }

        for(size_t i = 0; i < m_engine.bKnobs.size(); i++)
        {
            m_engine.bKnobs[i] = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
            if(m_engine.bKnobs[i]->is_good() == false)
            {
                status = m_engine.bKnobs[i]->get_status();
                set_error_and_throw_exception(
                    &m_engine,
                    status,
                    "CUDNN_BACKEND_ENGINE_DESCRIPTOR: "
                    "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR cudnnCreate Failed");
            }
        }

        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bKnobs_ = {};
        for(std::uint32_t i = 0; i < m_engine.bKnobs.size(); i++)
        {
            bKnobs_[i] = m_engine.bKnobs[i]->get_backend_descriptor();
        }
        status = detail::get_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       CUDNN_KNOB_TYPE_COUNTS,
                                       &m_engine.numKnobs,
                                       bKnobs_.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_engine,
                                          status,
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_KNOB_INFO Query Failed");
        }
        m_engine.buildKnobs();
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

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-use-emplace,modernize-loop-convert,performance-unnecessary-value-param,readability-isolate-declaration,readability-simplify-boolean-expr,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
