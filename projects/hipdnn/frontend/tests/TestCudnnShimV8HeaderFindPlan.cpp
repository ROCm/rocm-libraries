// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only: cudnn_frontend_find_plan.h alone and first in its TU, as a
// consumer outside the umbrella includes it.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>

#include <utility>

namespace hipdnn_shim_test
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// External linkage forces every sampling technique to be instantiated, not
// just parsed.
// NOLINTNEXTLINE(misc-use-internal-linkage)
fe::ExecutionPlan findEveryWayWithFindPlanAlone(cudnnHandle_t handle,
                                                fe::EngineConfigGenerator& generator,
                                                fe::OperationGraph& graph,
                                                const fe::VariantPack& pack,
                                                fe::ExecutionPlanCache& cache)
{
    fe::executionPlans_t once
        = generator.cudnnFindPlan<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            handle, graph, pack);
    fe::executionPlans_t median
        = fe::time_sorted_plan<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            handle, std::move(once), pack);
    static_cast<void>(
        fe::time_sorted_plan<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(
            handle, std::move(median), pack, 1));
    return generator
        .cudnnFindPlanAndCache<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(
            handle, graph, pack, cache);
}

} // namespace hipdnn_shim_test
