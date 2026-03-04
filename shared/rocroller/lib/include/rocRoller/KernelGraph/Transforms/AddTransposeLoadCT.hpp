// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply_fwd.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /** @brief returns True iff the target architecture has instructions
         * to transpose wavetiles of sizes mi.m x mi.n x mi.k of @typde datatype.
         */
        bool isTransposableTile(GPUArchitecture const&                     arch,
                                InstructionGenerators::MatrixMultiplySizes mi,
                                DataType                                   type);

        /** @brief Add coordinate-transforms for transpose-loading a WaveTile
         * from row/column coordinates `iWaveX` and `iWaveY`.
         *
         * The `lane` and `element` parameters are existing coordinates
         * corresponding to a Lane coordiante and VGPR coordinate (which should
         * be thought of as which element/item is being addressed).
         */
        void addTransposeLoadWaveTileCT(ContextPtr                                 context,
                                        std::vector<DeferredConnection>&           connections,
                                        KernelGraph&                               graph,
                                        int                                        macTileTag,
                                        int                                        iWaveX,
                                        int                                        iWaveY,
                                        int                                        lane,
                                        int                                        element,
                                        InstructionGenerators::MatrixMultiplySizes mi,
                                        uint                                       bitsPerElement,
                                        int                                        wavefrontSize);
    }
}
