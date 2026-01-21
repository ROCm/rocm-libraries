// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        struct MatrixMultiply;
        using MatrixMultiplyPtr = std::shared_ptr<MatrixMultiply>;

        struct MatrixMultiplySizes
        {
            int m, n, k, b = 1;
        };

        std::string toString(MatrixMultiplySizes mi);
    }
}
