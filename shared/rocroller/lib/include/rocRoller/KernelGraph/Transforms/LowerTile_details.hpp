// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace LowerTileDetails
        {
            using namespace rocRoller::InstructionGenerators;
            bool isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(DataType            type,
                                                                 MatrixMultiplySizes mi);
        }
    }
}
