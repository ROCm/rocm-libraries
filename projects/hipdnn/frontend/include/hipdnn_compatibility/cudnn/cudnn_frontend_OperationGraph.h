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
 * @brief cuDNN v8 `OperationGraph_v8` / `OperationGraphBuilder_v8` over the shim's v8 core.
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
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/lowering.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>

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
        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            const auto core = detail::v8::recordOf<detail::v8::GraphCore>(pointer);
            if(core == nullptr)
            {
                throw detail::v8::ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                                            "OperationGraph: the operation graph was not built");
            }
            global_count = core->engineCount();
        });
        if(!outcome)
        {
            set_error_and_throw_exception(this, outcome.status, outcome.message.c_str());
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
        // The empty pointer the built graph replaces, created before the prechecks
        // so a failed graph's get_raw_desc() is null, not a crash.
        const detail::v8::Outcome created = detail::v8::guarded([&] {
            m_operationGraph.initialize_managed_backend_pointer(
                CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        });
        if(!created)
        {
            set_error_and_throw_exception(
                &m_operationGraph, created.status, created.message.c_str());
            return std::move(m_operationGraph);
        }

        if(m_operationGraph.numOps <= 0)
        {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "OperationGraphBuilder: operations is required (non-empty)");
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
                                          "OperationGraphBuilder: operation 0 is not set");
            return std::move(m_operationGraph);
        }
#if(CUDNN_VERSION < 90800)
        if(m_operationGraph.handle == nullptr)
        {
            set_error_and_throw_exception(&m_operationGraph,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "OperationGraphBuilder: handle is required");
            return std::move(m_operationGraph);
        }
#endif

        const detail::v8::Outcome outcome = detail::v8::guarded([&] {
            static const detail::v8::LoweringLabels labels{
                "OperationGraphBuilder: handle",
                "OperationGraphBuilder: operations",
                "OperationGraphBuilder: dynamic shape",
                "OperationGraphBuilder: override shape",
                "OperationGraphBuilder: constant tensor"};
            std::vector<std::shared_ptr<const detail::v8::OperationRecord>> records;
            records.reserve(static_cast<size_t>(m_operationGraph.numOps));
            for(auto i = 0u; i < m_operationGraph.numOps; i++)
            {
                records.push_back(
                    detail::v8::recordOf<detail::v8::OperationRecord>(m_operationGraph.ops[i]));
                if(records.back() == nullptr)
                {
                    throw detail::v8::ShimError(CUDNN_STATUS_BAD_PARAM,
                                                "OperationGraphBuilder: operation "
                                                    + std::to_string(i)
                                                    + " is not a built operation");
                }
            }
            std::shared_ptr<const detail::v8::GraphCore> core
                = detail::v8::lowerGraph(records,
                                         m_operationGraph.handle,
                                         {m_operationGraph.is_dynamic_shape_enabled,
                                          m_operationGraph.is_override_shape_enabled},
                                         labels,
                                         m_operationGraph.opGraphTag);
            const cudnnBackendDescriptor_t raw = core->hipdnnGraph();
            m_operationGraph.pointer = detail::v8::makeManaged(
                CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, std::move(core), raw);
        });
        if(!outcome)
        {
            set_error_and_throw_exception(
                &m_operationGraph, outcome.status, outcome.message.c_str());
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
