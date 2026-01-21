// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Scheduling/Costs/NoneCost.hpp>
#include <rocRoller/Scheduling/Costs/UniformCost.hpp>
#include <rocRoller/Scheduling/Costs/WaitCntNopCost.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<Scheduling::Cost>::registerImplementations()
    {
        registerComponent<Scheduling::LinearWeightedCost>();
        registerComponent<Scheduling::MinNopsCost>();
        registerComponent<Scheduling::NoneCost>();
        registerComponent<Scheduling::UniformCost>();
        registerComponent<Scheduling::WaitCntNopCost>();
    }
}
