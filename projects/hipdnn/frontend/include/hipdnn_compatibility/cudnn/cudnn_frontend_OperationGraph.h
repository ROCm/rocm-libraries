// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_OperationGraph.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_OperationGraph.h
 * @brief cuDNN v8 `OperationGraph_v8` / `OperationGraphBuilder_v8` over the shim's backend C API.
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

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Operation.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

// Global, as upstream: Errata.h and consumers name it unqualified.
constexpr int64_t MAX_OPGRAPH_OPS = 250;

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_OperationGraph.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-redundant-void-arg,performance-avoid-endl,performance-unnecessary-value-param,readability-container-size-empty,readability-else-after-return,readability-redundant-string-init,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class OperationGraph_v8 : public BackendDescriptor
{
public:
    friend class OperationGraphBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR has " << numOps << " operations."
           << std::endl;
        ss << "Tag: " << opGraphTag << std::endl;
        return ss.str();
    }

    OperationGraph_v8(OperationGraph_v8&& from) = default;
    OperationGraph_v8& operator=(OperationGraph_v8&& from) = default;

    ~OperationGraph_v8() = default;

    auto getEngineCount(void) const -> int64_t
    {
        int64_t global_count = -1;
        auto status = detail::get_attribute(pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                            CUDNN_TYPE_INT64,
                                            1,
                                            nullptr,
                                            &global_count);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT Failed");
        }
        return global_count;
    }

    uint64_t getOpCount() const
    {
        return static_cast<uint64_t>(numOps);
    }

    const std::string& getTag() const
    {
        return opGraphTag;
    }

    bool setFeatureVector(feature_vector_t fv)
    {
        feature_vectors.push_back(fv);
        return true;
    }

    feature_vector_t getFeatureVector() const
    {
        if(feature_vectors.size() != 0)
        {
            return feature_vectors[0];
        }
        else
        {
            return {};
        }
    }

    const std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS>& getOps() const
    {
        return ops;
    }

private:
    OperationGraph_v8() = default;
    OperationGraph_v8(const OperationGraph_v8&) = delete;
    OperationGraph_v8& operator=(const OperationGraph_v8&) = delete;

    cudnnHandle_t handle = nullptr;
    std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops{};
    int64_t numOps = -1;
    std::string opGraphTag = "";
    std::vector<feature_vector_t> feature_vectors;
    bool is_dynamic_shape_enabled = false;
    bool is_override_shape_enabled = false;
};

class OperationGraphBuilder_v8
{
public:
    auto setHandle(cudnnHandle_t handle_) -> OperationGraphBuilder_v8&
    {
        m_operationGraph.handle = handle_;
        return *this;
    }

    auto setOperationGraph(int64_t numOps_, const Operation_v8** ops_) -> OperationGraphBuilder_v8&
    {
        if(numOps_ > MAX_OPGRAPH_OPS)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: numOps exceeds MAX_OPGRAPH_OPS");
            return *this;
        }
        m_operationGraph.numOps = numOps_;
        m_operationGraph.feature_vectors.resize(static_cast<size_t>(numOps_));
        for(auto i = 0u; i < numOps_; i++)
        {
            m_operationGraph.ops[i] = ops_[i]->get_desc();
            m_operationGraph.opGraphTag += ops_[i]->getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i]->getFeatureVector();
        }
        return *this;
    }

    auto setOperationGraph(const std::vector<Operation>& ops_) -> OperationGraphBuilder_v8&
    {
        if(ops_.size() > static_cast<size_t>(MAX_OPGRAPH_OPS))
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: numOps exceeds MAX_OPGRAPH_OPS");
            return *this;
        }
        m_operationGraph.numOps = static_cast<int64_t>(ops_.size());
        m_operationGraph.feature_vectors.resize(ops_.size());
        for(auto i = 0u; i < ops_.size(); i++)
        {
            m_operationGraph.ops[i] = ops_[i].get_desc();
            m_operationGraph.opGraphTag += ops_[i].getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i].getFeatureVector();
        }
        return *this;
    }

    auto addOperation(ManagedOpaqueDescriptor desc) -> OperationGraphBuilder_v8&
    {
        // numOps starts at -1; upstream stores a fresh builder's first op at ops[-1].
        m_operationGraph.numOps = std::max<int64_t>(m_operationGraph.numOps, 0);
        if(m_operationGraph.numOps >= MAX_OPGRAPH_OPS)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: numOps exceeds MAX_OPGRAPH_OPS");
            return *this;
        }
        m_operationGraph.ops[static_cast<size_t>(m_operationGraph.numOps)] = desc;
        ++m_operationGraph.numOps;
        return *this;
    }

    auto setIsDynamicShapeEnabled(bool is_enabled) -> OperationGraphBuilder_v8&
    {
        m_operationGraph.is_dynamic_shape_enabled = is_enabled;
        return *this;
    }

    auto setIsOverrideShapeEnabled(bool is_enabled) -> OperationGraphBuilder_v8&
    {
        m_operationGraph.is_override_shape_enabled = is_enabled;
        return *this;
    }

    OperationGraph_v8&& build()
    {
        if(m_operationGraph.numOps <= 0)
        {
            set_error_and_throw_exception(&m_operationGraph,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set "
                                          "the CUDNN_ATTR_OPERATIONGRAPH_OPS Count field");
            return std::move(m_operationGraph);
        }
        if(m_operationGraph.numOps > MAX_OPGRAPH_OPS)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: numOps exceeds MAX_OPGRAPH_OPS");
            return std::move(m_operationGraph);
        }
        if(m_operationGraph.ops[0] == nullptr)
        {
            set_error_and_throw_exception(&m_operationGraph,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and set "
                                          "CUDNN_ATTR_OPERATIONGRAPH_OPS field");
            return std::move(m_operationGraph);
        }
#if(CUDNN_VERSION < 90800)
        if(m_operationGraph.handle == nullptr)
        {
            set_error_and_throw_exception(&m_operationGraph,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set "
                                          "CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
            return std::move(m_operationGraph);
        }
#endif

        auto status = m_operationGraph.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_operationGraph);
        }

        std::array<cudnnBackendDescriptor_t, MAX_OPGRAPH_OPS> ops_raw{nullptr};
        for(auto i = 0u; i < m_operationGraph.numOps; i++)
        {
            ops_raw[i] = m_operationGraph.ops[i]->get_backend_descriptor();
        }

        status = detail::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       m_operationGraph.numOps,
                                       ops_raw.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_operationGraph,
                                          status,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_OPERATIONGRAPH_OPS Failed");
            return std::move(m_operationGraph);
        }

        if(m_operationGraph.handle != nullptr)
        {
            status = detail::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                           CUDNN_TYPE_HANDLE,
                                           1,
                                           &m_operationGraph.handle);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    &m_operationGraph,
                    status,
                    "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute "
                    "CUDNN_ATTR_OPERATIONGRAPH_HANDLE Failed");
                return std::move(m_operationGraph);
            }
        }

#if(CUDNN_VERSION >= 90400)
        if(m_operationGraph.is_dynamic_shape_enabled)
        {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                90400,
                m_operationGraph,
                "CUDNN_BACKEND_OPERATION_GRAPH: Dynamic shape support requires cudnn 9.4.0 and "
                "above");
            status = detail::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_operationGraph.is_dynamic_shape_enabled);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    &m_operationGraph,
                    status,
                    "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute "
                    "CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED "
                    "Failed");
                return std::move(m_operationGraph);
            }
        }
#endif

#if(CUDNN_VERSION >= 92100)
        if(m_operationGraph.is_override_shape_enabled)
        {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                92100,
                m_operationGraph,
                "CUDNN_BACKEND_OPERATION_GRAPH: Override shape support requires cudnn 9.21.0 and "
                "above");
            status = detail::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_operationGraph.is_override_shape_enabled);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                set_error_and_throw_exception(
                    &m_operationGraph,
                    status,
                    "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute "
                    "CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED "
                    "Failed");
                return std::move(m_operationGraph);
            }
        }
#endif

        status = detail::finalize(m_operationGraph.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_operationGraph);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_operationGraph);
        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder_v8() = default;
    ~OperationGraphBuilder_v8() = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8&&) = delete;
    OperationGraphBuilder_v8(const OperationGraphBuilder_v8&) = delete;
    OperationGraphBuilder_v8& operator=(const OperationGraphBuilder_v8&) = delete;

private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph = OperationGraph_v8;
using OperationGraphBuilder = OperationGraphBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-equals-delete,modernize-use-override,modernize-redundant-void-arg,performance-avoid-endl,performance-unnecessary-value-param,readability-container-size-empty,readability-else-after-return,readability-redundant-string-init,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
