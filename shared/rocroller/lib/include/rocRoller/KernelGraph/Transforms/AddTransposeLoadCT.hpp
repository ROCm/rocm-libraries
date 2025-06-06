/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /** @brief returns True iff the target architecture has instructions
         * to transpose wavetiles of MacroTile @macroTile of @typde datatype.
         */
        bool isTransposableTile(GPUArchitecture const&     arch,
                                CoordinateGraph::MacroTile macroTile,
                                DataType                   type);

        /** @brief Add coordinate-transforms for transpose-loading a WaveTile
         * from row/column coordinates `iWaveX` and `iWaveY`.
         *
         * The `lane` and `element` parameters are existing coordinates
         * corresponding to a Lane coordiante and VGPR coordinate (which should
         * be thought of as which element/item is being addressed). Each lane
         * loads 32 elements.
         */
        void addTransposeLoadWaveTileCT(ContextPtr                       context,
                                        std::vector<DeferredConnection>& connections,
                                        KernelGraph&                     graph,
                                        int                              macTileTag,
                                        int                              iWaveX,
                                        int                              iWaveY,
                                        int                              lane,
                                        int                              element,
                                        uint                             M,
                                        uint                             K,
                                        uint                             bitsPerElement,
                                        int                              wavefrontSize);
    }
}
