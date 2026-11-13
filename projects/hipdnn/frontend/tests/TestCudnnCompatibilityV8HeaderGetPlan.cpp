// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only: cudnn_frontend_get_plan.h alone and first in its TU, as a
// consumer outside the umbrella includes it.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

namespace hipdnn_shim_test
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// External linkage forces both overloads to be emitted, not just parsed.
// NOLINTNEXTLINE(misc-use-internal-linkage)
fe::executionPlans_t getPlansWithGetPlanAlone(cudnnHandle_t handle,
                                              fe::EngineConfigGenerator& generator,
                                              fe::OperationGraph& graph,
                                              const fe::Predicate& predicate)
{
    fe::executionPlans_t plans = generator.cudnnGetPlan(handle, graph, 4);
    fe::executionPlans_t filtered = generator.cudnnGetPlan(handle, graph, predicate);
    plans.insert(plans.end(), filtered.begin(), filtered.end());
    return plans;
}

} // namespace hipdnn_shim_test
