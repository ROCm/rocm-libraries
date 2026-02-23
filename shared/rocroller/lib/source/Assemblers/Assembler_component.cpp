// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/Assemblers/InProcessAssembler.hpp>
#include <rocRoller/Assemblers/SubprocessAssembler.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<Assembler>::registerImplementations()
    {
        registerComponent<InProcessAssembler>();
        registerComponent<SubprocessAssembler>();
    }
}
