// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<
        InstructionGenerators::ScaledMatrixMultiply>::registerImplementations()
    {
        registerComponent<InstructionGenerators::ScaledMatrixMultiplyGenerator>();
    }
}
