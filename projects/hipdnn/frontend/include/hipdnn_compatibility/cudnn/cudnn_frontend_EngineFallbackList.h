// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_EngineFallbackList.h
//     Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_EngineFallbackList.h
 * @brief cuDNN v8 `EngineFallbackList_v8` / `EngineFallbackListBuilder_v8` and
 *        `get_fallback_engine_list` over the shim's backend C API.
 */

#pragma once

#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_EngineFallbackList.h at the new tag, then update this "
              "assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,performance-move-constructor-init,readability-else-after-return):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

// NVIDIA engine numbers; they do not name hipDNN cuDNN shim engines.
inline auto get_fallback_engine_list(DescriptorType_t mode,
                                     const std::string& opGraphTag) -> std::vector<int>
{
    auto major_version = detail::get_backend_version() / 1000;

    auto minor_version = (detail::get_backend_version() / 100) % 10;
    if(major_version >= 8)
    {
        if(minor_version <= 2)
        {
            if(mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            {
                if(opGraphTag.find("bias") == std::string::npos)
                {
                    std::vector<int> engine_list(50);
                    std::iota(engine_list.begin(), engine_list.end(), 0);
                    return engine_list;
                }
                else
                {
                    return {11, 0};
                }
            }
            else if(mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            {
                std::vector<int> engine_list(61);
                std::iota(engine_list.begin(), engine_list.end(), 0);
                return engine_list;
            }
            else if(mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            {
                return {0, 1, 20};
            }
            else
            {
                return {};
            }
        }
        else
        {
            if(mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            {
                if(opGraphTag.find("bias") == std::string::npos)
                {
                    return {0, 1, 28};
                }
                else
                {
                    return {};
                }
            }
            else if(mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            {
                return {0, 1, 25};
            }
            else if(mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            {
                return {0, 1, 20};
            }
            else
            {
                return {};
            }
        }
    }
    else
    {
        return {};
    }
}

class EngineFallbackList_v8 : public BackendDescriptor
{
public:
    friend class EngineFallbackListBuilder_v8;

    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_FALLBACK ENGINES :";
        return ss.str();
    }

    auto getFallbackList() -> std::vector<ManagedOpaqueDescriptor>&
    {
        return m_engine_configs;
    }

    ~EngineFallbackList_v8() = default;

    EngineFallbackList_v8(EngineFallbackList_v8&& from)
        : BackendDescriptor(from.get_desc(), from.get_status(), from.get_error())
        , opGraph(from.opGraph)
        , mode(from.mode)
        , num_ops(from.num_ops)
        , opGraphTag(from.opGraphTag)
    {
        m_engine_configs.swap(from.m_engine_configs);
    }

private:
    EngineFallbackList_v8() = default;
    EngineFallbackList_v8(const EngineFallbackList_v8&) = delete;
    EngineFallbackList_v8& operator=(const EngineFallbackList_v8&) = delete;

    ManagedOpaqueDescriptor opGraph = nullptr;
    DescriptorType_t mode;
    uint64_t num_ops;
    std::vector<ManagedOpaqueDescriptor> m_engine_configs;
    std::string opGraphTag;
};

class EngineFallbackListBuilder_v8
{
public:
    auto setOperationGraph(OperationGraph_v8& opGraph_) -> EngineFallbackListBuilder_v8&
    {
        m_fallback_list.opGraph = opGraph_.get_desc();
        m_fallback_list.opGraphTag = opGraph_.getTag();
        m_fallback_list.num_ops = opGraph_.getOpCount();
        return *this;
    }

    auto setOperation(DescriptorType_t mode) -> EngineFallbackListBuilder_v8&
    {
        m_fallback_list.mode = mode;
        return *this;
    }

    auto setOperation(cudnnBackendDescriptorType_t mode) -> EngineFallbackListBuilder_v8&
    {
        m_fallback_list.mode = detail::convert_from_cudnn_type(mode);
        return *this;
    }

    EngineFallbackList_v8&& build()
    {
        if(m_fallback_list.opGraph == nullptr)
        {
            set_error_and_throw_exception(
                &m_fallback_list,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_fallback_list);
        };
        auto fallback_heuristics
            = EngineHeuristicsBuilder_v8()
                  .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                  .setOperationGraph(m_fallback_list.opGraph, m_fallback_list.opGraphTag)
                  .build();
        auto count = fallback_heuristics.getEngineConfigCount();
        m_fallback_list.m_engine_configs = fallback_heuristics.getEngineConfig(count);
        CUDNN_FE_LOG_LABEL_ENDL(m_fallback_list);
        return std::move(m_fallback_list);
    }

    explicit EngineFallbackListBuilder_v8() = default;
    ~EngineFallbackListBuilder_v8() = default;
    EngineFallbackListBuilder_v8(EngineFallbackListBuilder_v8&&) = delete;
    EngineFallbackListBuilder_v8(const EngineFallbackListBuilder_v8&) = delete;
    EngineFallbackListBuilder_v8& operator=(const EngineFallbackListBuilder_v8&) = delete;

private:
    EngineFallbackList_v8 m_fallback_list;
};

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,performance-move-constructor-init,readability-else-after-return)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
