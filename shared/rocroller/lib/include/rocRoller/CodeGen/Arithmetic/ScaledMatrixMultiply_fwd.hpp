// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        struct ScaledMatrixMultiply;
        using ScaledMatrixMultiplyPtr = std::shared_ptr<ScaledMatrixMultiply>;
    }
}
