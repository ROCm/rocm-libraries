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

#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/KernelGraph/Transforms/AddTransposeLoadCT.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace InstructionGenerators;

        bool isTransposableTile(GPUArchitecture const& arch, MatrixMultiplySizes mi, DataType type)
        {
            auto isF8F6F4TransposableTileLayout
                = isUnpackedF8F6F4(type)
                  && (((mi.m == 16) && (mi.n == 16) && (mi.k == 128))
                      || ((mi.m == 32) && (mi.n == 32) && (mi.k == 64)));

            auto isF16TransposableTileLayout
                = isUnpackedF16(type)
                  && (((mi.m == 16) && (mi.n == 16) && (mi.k == 32))
                      || ((mi.m == 32) && (mi.n == 32) && (mi.k == 16)));

            auto hasTransposeInstructionForType = [&arch](DataType type) {
                switch(type)
                {
                case DataType::Half:
                case DataType::BFloat16:
                    return arch.HasCapability(GPUCapability::HasDSReadTransposeB16);
                case DataType::FP8:
                case DataType::BF8:
                    return arch.HasCapability(GPUCapability::HasDSReadTransposeB8);
                case DataType::FP6:
                case DataType::BF6:
                    return arch.HasCapability(GPUCapability::HasDSReadTransposeB6);
                case DataType::FP4:
                    return arch.HasCapability(GPUCapability::HasDSReadTransposeB4);
                default:
                    return false;
                };
            };

            return (isF8F6F4TransposableTileLayout || isF16TransposableTileLayout)
                   && hasTransposeInstructionForType(type);
        };

        /** @brief Sets isTransposedTile field of LoadTiled/LoadLDSTile ops
         * connected to MacroTile @tileTag in Coordinate Graph @graph.
         */
        void setIsTransposedLoad(int tileTag, KernelGraph& graph)
        {
            auto conns = graph.mapper.getCoordinateConnections(tileTag);
            AssertFatal(conns.size() == 1,
                        "Macrotile(",
                        tileTag,
                        ") is connected to more than 1 operation in control graph!");
            auto opTag = conns[0].control;
            auto e     = graph.control.getElement(opTag);
            std::visit(rocRoller::overloaded{[&](LoadTiled& op) {
                                                 auto newLoadTiled(op);
                                                 newLoadTiled.isTransposedTile = true;
                                                 graph.control.setElement(opTag, newLoadTiled);
                                             },
                                             [&](LoadLDSTile& op) {
                                                 auto newLoadLDSTile(op);
                                                 newLoadLDSTile.isTransposedTile = true;
                                                 graph.control.setElement(opTag, newLoadLDSTile);
                                             },
                                             [&](auto& op) {
                                                 Throw<FatalError>("Unexpected control node ",
                                                                   op.name(),
                                                                   "(",
                                                                   opTag,
                                                                   ") connected to MacroTile(",
                                                                   tileTag,
                                                                   ")");
                                             }},
                       std::get<Operation>(e));
        }

        template <GPUArchitectureGFX target>
        void addTransposeLoadWaveTileCTImpl(ContextPtr                       context,
                                            std::vector<DeferredConnection>& connections,
                                            KernelGraph&                     graph,
                                            int                              macTileTag,
                                            int                              iWaveX,
                                            int                              iWaveY,
                                            int                              lane,
                                            int                              element,
                                            MatrixMultiplySizes              mi,
                                            uint                             bitsPerElement,
                                            int                              wavefrontSize);

        template <>
        void addTransposeLoadWaveTileCTImpl<GPUArchitectureGFX::GFX950>(
            ContextPtr                       context,
            std::vector<DeferredConnection>& connections,
            KernelGraph&                     graph,
            int                              macTileTag,
            int                              iWaveX,
            int                              iWaveY,
            int                              lane,
            int                              element,
            MatrixMultiplySizes              mi,
            uint                             bitsPerElement,
            int                              wavefrontSize)
        {
            const auto simdsInWave    = 4;
            const auto lanesInSIMD    = 16;
            const auto simdsPerSGroup = mi.m / lanesInSIMD;

            const auto& arch                    = context->targetArchitecture();
            const auto  bitsPerTrLoad           = bitsPerTransposeLoad(arch, bitsPerElement);
            const auto  elementsTrLoadedPerLoad = bitsPerTrLoad / bitsPerElement;
            const auto  numTrLoadsPerWave       = 2;
            const auto  numTrLoads = (mi.m * mi.k) / wavefrontSize / elementsTrLoadedPerLoad;

            auto simdsPerWave = graph.coordinates.addElement(
                Adhoc("transpose.simdsPerWave", literal(simdsInWave), nullptr));
            auto lanesPerSIMD = graph.coordinates.addElement(Lane(literal(lanesInSIMD), nullptr));

            auto simdBlockNumber = graph.coordinates.addElement(
                Adhoc("transpose.simdBlockNumber", literal(simdsInWave / simdsPerSGroup), nullptr));
            auto simdBlockIndex = graph.coordinates.addElement(
                Adhoc("transpose.simdBlockIndex", literal(simdsPerSGroup), nullptr));

            auto lanesPerSIMDInM = graph.coordinates.addElement(
                Lane(literal(lanesInSIMD / elementsTrLoadedPerLoad), nullptr));
            auto lanesPerSIMDInK
                = graph.coordinates.addElement(Lane(literal(elementsTrLoadedPerLoad), nullptr));

            auto trLoadBlockNumber = graph.coordinates.addElement(Adhoc(
                "transpose.LoadBlockNumber", literal(numTrLoads / numTrLoadsPerWave), nullptr));
            auto trLoadBlockIndex  = graph.coordinates.addElement(
                Adhoc("transpose.LoadBlockIndex", literal(numTrLoadsPerWave), nullptr));

            auto elementBlockNumber
                = graph.coordinates.addElement(VGPRBlockNumber(literal(numTrLoads), nullptr));
            auto elementBlockIndex = graph.coordinates.addElement(
                VGPRBlockIndex(literal(elementsTrLoadedPerLoad), nullptr));

            connections.push_back(DC<VGPRBlockNumber>(elementBlockNumber));
            connections.push_back(DC<VGPRBlockIndex>(elementBlockIndex));

            graph.coordinates.addElement(
                Flatten(), {trLoadBlockNumber, trLoadBlockIndex}, {elementBlockNumber});

            graph.coordinates.addElement(
                Tile(), {iWaveX}, {simdBlockIndex, lanesPerSIMDInM, elementBlockIndex});

            graph.coordinates.addElement(
                Tile(),
                {iWaveY},
                {trLoadBlockNumber, simdBlockNumber, trLoadBlockIndex, lanesPerSIMDInK});

            graph.coordinates.addElement(
                Flatten(), {lanesPerSIMDInK, lanesPerSIMDInM}, {lanesPerSIMD});
            graph.coordinates.addElement(
                Flatten(), {simdBlockNumber, simdBlockIndex}, {simdsPerWave});
            graph.coordinates.addElement(Flatten(), {simdsPerWave, lanesPerSIMD}, {lane});
            graph.coordinates.addElement(
                Flatten(), {elementBlockNumber, elementBlockIndex}, {element});
        }

        void addTransposeLoadWaveTileCT(ContextPtr                       context,
                                        std::vector<DeferredConnection>& connections,
                                        KernelGraph&                     graph,
                                        int                              macTileTag,
                                        int                              iWaveX,
                                        int                              iWaveY,
                                        int                              lane,
                                        int                              element,
                                        MatrixMultiplySizes              mi,
                                        uint                             bitsPerElement,
                                        int                              wavefrontSize)
        {
            TIMER(t, "KernelGraph::AddTransposeLoadCT");

            switch(context->targetArchitecture().target().gfx)
            {
            case GPUArchitectureGFX::GFX950:
            {
                addTransposeLoadWaveTileCTImpl<GPUArchitectureGFX::GFX950>(context,
                                                                           connections,
                                                                           graph,
                                                                           macTileTag,
                                                                           iWaveX,
                                                                           iWaveY,
                                                                           lane,
                                                                           element,
                                                                           mi,
                                                                           bitsPerElement,
                                                                           wavefrontSize);
                setIsTransposedLoad(macTileTag, graph);
            }
            break;
            default:
                Throw<FatalError>("addTransposeLoadWaveTileCT is not implemented for ",
                                  context->targetArchitecture().target().toString());
            }
        }
    }
}
