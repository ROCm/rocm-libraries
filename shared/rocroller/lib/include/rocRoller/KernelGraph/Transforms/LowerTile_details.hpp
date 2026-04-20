// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

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

        namespace LDSSwizzleDetail
        {
            /// Parameters for LDS bank swizzle column permutation.
            struct LDSSwizzleParams
            {
                unsigned int numColumns; ///< dwordx4 chunks per tile row in K
                unsigned int rowsPerBankRow; ///< tile rows sharing one LDS bank row
                unsigned int elementsPerChunk; ///< elements per dwordx4 chunk (128 / elementBits)
                unsigned int columnsPerBankRow; ///< dwordx4 columns per bank row (arch-dependent)

                static LDSSwizzleParams compute(unsigned int tileK,
                                                unsigned int elementBits,
                                                unsigned int colsPerBankRow);

                static LDSSwizzleParams fromColumnCount(unsigned int numCols,
                                                        unsigned int colsPerBankRow);

                bool noConflicts() const;
            };

            /// Insert forward PairSwap + Rotate edges for LoadTiled swizzle.
            /// Returns the swizzled column coordinate tag.
            int addLoadTiledSwizzleEdges(KernelGraph&            graph,
                                         int                     colCoord,
                                         int                     rowCoord,
                                         int                     kDim,
                                         LDSSwizzleParams const& params);

            /// Insert inverse Rotate + PairSwap edges for LoadLDSTile unswizzle.
            /// Returns the unswizzled element-level column coordinate tag.
            int addLoadLDSTileSwizzleEdges(KernelGraph&            graph,
                                           int                     colCoord,
                                           int                     rowCoord,
                                           int                     kDim,
                                           LDSSwizzleParams const& params);
        }
    }
}
