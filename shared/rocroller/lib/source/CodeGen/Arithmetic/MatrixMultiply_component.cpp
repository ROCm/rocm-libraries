// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<
        InstructionGenerators::MatrixMultiply>::registerImplementations()
    {
        registerComponent<InstructionGenerators::MatrixMultiplyGenerator>();
    }
}
