// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_ExecutionPlanCache.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_ExecutionPlanCache.h
 * @brief cuDNN v8 execution plan caches keyed by operation-graph feature vector.
 */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_ExecutionPlanCache.h at the new tag, then update this "
              "assert.");

// NOLINTBEGIN(readability-identifier-naming,misc-const-correctness,modernize-use-override):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class ExecutionPlanCache_v1
{
protected:
    struct compare
    {
        bool operator()(const feature_vector_t& fv1, const feature_vector_t& fv2) const
        {
            return fv1 < fv2;
        }
    };

    std::string name = "plan_cache_[unnamed]";

    using FeatureVectorToPlanMap = std::map<cudnn_frontend::feature_vector_t,
                                            cudnn_frontend::ExecutionPlan,
                                            cudnn_frontend::ExecutionPlanCache_v1::compare>;
    FeatureVectorToPlanMap cache;

    mutable std::mutex cache_mutex;

public:
    virtual bool is_fastest_plan_stable(const cudnn_frontend::OperationGraph& op_graph,
                                        const std::string& tag)
    {
        CUDNN_FRONTEND_UNUSED(op_graph);
        CUDNN_FRONTEND_UNUSED(tag);
        return true;
    }

    void add_plan_to_cache(const cudnn_frontend::OperationGraph& op_graph,
                           const cudnn_frontend::ExecutionPlan& plan)
    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        cache.insert(std::make_pair(op_graph.getFeatureVector(), plan));
        CUDNN_FE_LOG_LABEL_ENDL("Added to " << name << " " << op_graph.getTag());
    }

    ExecutionPlanCache_v1(const char* name_)
    {
        name = name_;
    }

    const std::string& get_name() const
    {
        return name;
    }

    bool get_plan_from_cache(const cudnn_frontend::OperationGraph& op_graph,
                             const cudnn_frontend::ExecutionPlan*& plan) const
    {
        {
            std::lock_guard<std::mutex> guard(cache_mutex);
            auto it = cache.find(op_graph.getFeatureVector());

            if(it == cache.end())
            {
                CUDNN_FE_LOG_LABEL_ENDL("Cached Plan Not Found in " << name);
                return false;
            }
            plan = &(it->second);
        }
        CUDNN_FE_LOG_LABEL_ENDL("Cached Plan Found in " << name);
        return true;
    }

    virtual ~ExecutionPlanCache_v1() = default;
};

class ExecutionPlanCache_v2 : public ExecutionPlanCache_v1
{
    using SaturationTracker
        = std::map<std::pair<cudnn_frontend::feature_vector_t, std::string>, int32_t>;
    SaturationTracker tracker;

    int32_t saturationCount = 1;

public:
    virtual bool is_fastest_plan_stable(const cudnn_frontend::OperationGraph& op_graph,
                                        const std::string& tag)
    {
        if(saturationCount == 1)
        {
            return true;
        }

        // A graph that already has a cached plan is never re-tuned.
        const cudnn_frontend::ExecutionPlan* plan = nullptr;
        if(get_plan_from_cache(op_graph, plan))
        {
            CUDNN_FE_LOG_LABEL_ENDL("SaturationTracker " << name << " " << op_graph.getTag() << " "
                                                         << tag << " plan already in cache.");
            return false;
        }

        std::lock_guard<std::mutex> guard(cache_mutex);
        auto cnt = tracker[std::make_pair(op_graph.getFeatureVector(), tag)] += 1;
        CUDNN_FE_LOG_LABEL_ENDL("SaturationTracker " << name << " " << op_graph.getTag() << " "
                                                     << tag << " " << cnt);
        return cnt >= saturationCount;
    }

    void set_saturation_count(int32_t count)
    {
        saturationCount = count;
    }

    ExecutionPlanCache_v2(const char* name_)
        : ExecutionPlanCache_v1(name_)
    {
    }

    virtual ~ExecutionPlanCache_v2() = default;
};

using ExecutionPlanCache = ExecutionPlanCache_v2;

// NOLINTEND(readability-identifier-naming,misc-const-correctness,modernize-use-override)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
