// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/CooperativeScheduler.hpp>
#include <rocRoller/Scheduling/PriorityScheduler.hpp>
#include <rocRoller/Scheduling/RandomScheduler.hpp>
#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Scheduling/SequentialScheduler.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<Scheduling::Scheduler>::registerImplementations()
    {
        registerComponent<Scheduling::CooperativeScheduler>();
        registerComponent<Scheduling::PriorityScheduler>();
        registerComponent<Scheduling::RandomScheduler>();
        registerComponent<Scheduling::RoundRobinScheduler>();
        registerComponent<Scheduling::SequentialScheduler>();
    }
}
