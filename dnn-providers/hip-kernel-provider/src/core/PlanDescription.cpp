// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlanDescription.hpp"

#include "Handle.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#endif

namespace hip_kernel_provider::core
{

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
namespace
{

std::string kernelName(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    return kernel.name.empty() ? hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId)
                               : kernel.name;
}

} // namespace
#endif

PlanDescription describePlan(const Context& context)
{
    PlanDescription description;
    if(!context.hasValidPlan())
    {
        return description;
    }
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    const auto& plan = context.plan();
    if(const auto* generic = plan.saveablePlan())
    {
        description.configName = kernelName(generic->kernel());
        return description;
    }
    if(dynamic_cast<const hipdnn_plugin_sdk::ingestor::BenchmarkPlan<Handle>*>(&plan) != nullptr)
    {
        description.autotunePending = true;
        return description;
    }
    if(const auto* slotted
       = dynamic_cast<const hipdnn_plugin_sdk::ingestor::SlotProgramPlan<Handle>*>(&plan))
    {
        for(const auto& stage : slotted->stages())
        {
            if(!description.configName.empty())
            {
                description.configName += ", ";
            }
            description.configName += stage.slot + "=" + kernelName(stage.plan->kernel());
        }
    }
#endif
    return description;
}

} // namespace hip_kernel_provider::core
