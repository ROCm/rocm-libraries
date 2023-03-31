/**
 * @copyright Copyright 2023 Advanced Micro Devices, Inc.
 */

#pragma once

#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Class for generating instructions related to loading and storing tiles
         *        to and from memory.
         *
         */
        class LoadStoreTileGenerator
        {
        public:
            LoadStoreTileGenerator(std::shared_ptr<KernelGraph>, ContextPtr, unsigned int);

            /**
                 * @brief Generate instructions needed to load a tile from global memory
                 *
                 * @param tag The tag of the node in the control graph
                 * @param load The node in the control graph
                 * @param coords Known coordinates
                 * @return Generator<Instruction>
                 */
            Generator<Instruction> genLoadTile(int                            tag,
                                               ControlGraph::LoadTiled const& load,
                                               CoordinateGraph::Transformer   coords);

            /**
                 * @brief Generate instructions needed to load a tile from LDS
                 *
                 * @param tag The tag of the node in the control graph
                 * @param load The node in the control graph
                 * @param coords Known coordinates
                 * @return Generator<Instruction>
                 */
            Generator<Instruction> genLoadLDSTile(int                              tag,
                                                  ControlGraph::LoadLDSTile const& load,
                                                  CoordinateGraph::Transformer     coords);

            /**
                 * @brief Generate instructions needed to store a tile to global memory
                 *
                 * @param tag The tag of the node in the control graph
                 * @param load The node in the control graph
                 * @param coords Known coordinates
                 * @return Generator<Instruction>
                 */
            Generator<Instruction> genStoreTile(int                             tag,
                                                ControlGraph::StoreTiled const& store,
                                                CoordinateGraph::Transformer    coords);

            /**
                 * @brief Generate instructions needed to store a tile to LDS
                 *
                 * @param tag The tag of the node in the control graph
                 * @param load The node in the control graph
                 * @param coords Known coordinates
                 * @return Generator<Instruction>
                 */
            Generator<Instruction> genStoreLDSTile(int                               tag,
                                                   ControlGraph::StoreLDSTile const& store,
                                                   CoordinateGraph::Transformer      coords);

            /**
                 * @brief Generate instructions needed to calculate offset and stride information
                 *
                 * @param tag The tag of the node in the control graph
                 * @param load The node in the control graph
                 * @param coords Known coordinates
                 * @return Generator<Instruction>
                 */
            Generator<Instruction> genComputeIndex(int                               tag,
                                                   ControlGraph::ComputeIndex const& ci,
                                                   CoordinateGraph::Transformer      coords);

        private:
            std::map<int, int>           m_baseOffsets;
            ContextPtr                   m_context;
            std::shared_ptr<KernelGraph> m_graph;
            Expression::FastArithmetic   m_fastArith;
            unsigned int                 m_workgroupSizeTotal;

            inline Generator<Instruction> generate(auto&                     dest,
                                                   Expression::ExpressionPtr expr) const;

            // Index calculation Helpers
            Register::ValuePtr        getBufferSrd(int tag);
            Expression::ExpressionPtr getOffsetExpr(int                                 offsetTag,
                                                    CoordinateGraph::Transformer const& coords);
            Generator<Instruction>    getOffset(Register::ValuePtr&          dst,
                                                Expression::ExpressionPtr&   expr,
                                                CoordinateGraph::Transformer coords,
                                                int                          tag,
                                                int                          dimension);
            Generator<Instruction>
                generateStride(Register::ValuePtr& stride, int tag, int dimension);

            // Load Tile Helpers
            Generator<Instruction> loadTile(MemoryInstructions::MemoryKind kind,
                                            uint64_t                       m,
                                            uint64_t                       n,
                                            VariableType                   dataType,
                                            int                            tag,
                                            Register::ValuePtr             offset,
                                            CoordinateGraph::Transformer&  coords);
            Generator<Instruction> loadMacroTileVGPRCI(int                            tag,
                                                       ControlGraph::LoadTiled const& load,
                                                       CoordinateGraph::Transformer   coords,
                                                       int                            sdim);
            Generator<Instruction> loadMacroTileVGPR(int                            tag,
                                                     ControlGraph::LoadTiled const& load,
                                                     CoordinateGraph::Transformer   coords);
            Generator<Instruction> loadMacroTileLDS(int                              tag,
                                                    ControlGraph::LoadLDSTile const& load,
                                                    CoordinateGraph::Transformer     coords);
            Generator<Instruction> loadMacroTileWAVELDSCI(int                              tag,
                                                          ControlGraph::LoadLDSTile const& load,
                                                          CoordinateGraph::Transformer     coords,
                                                          int                              sdim);
            Generator<Instruction> loadMacroTileWAVECI(int                            tag,
                                                       ControlGraph::LoadTiled const& load,
                                                       CoordinateGraph::Transformer   coords,
                                                       int                            sdim);
            Generator<Instruction> loadMacroTileWAVECIACCUM(int                            tag,
                                                            ControlGraph::LoadTiled const& load,
                                                            CoordinateGraph::Transformer   coords);

            // Store Tile Helpers
            Generator<Instruction> storeTile(MemoryInstructions::MemoryKind kind,
                                             uint64_t                       m,
                                             uint64_t                       n,
                                             VariableType                   dataType,
                                             int                            tag,
                                             Register::ValuePtr             vgpr,
                                             Register::ValuePtr             offset,
                                             CoordinateGraph::Transformer&  coords);
            Generator<Instruction> storeMacroTileLDS(int                               tag,
                                                     ControlGraph::StoreLDSTile const& store,
                                                     CoordinateGraph::Transformer      coords);
            Generator<Instruction> storeMacroTileVGPR(int                             tag,
                                                      ControlGraph::StoreTiled const& store,
                                                      CoordinateGraph::Transformer    coords);
            Generator<Instruction> storeMacroTileWAVELDS(int                               tag,
                                                         ControlGraph::StoreLDSTile const& store,
                                                         CoordinateGraph::Transformer      coords);
            Generator<Instruction> storeMacroTileWAVECI(int                             tag,
                                                        ControlGraph::StoreTiled const& store,
                                                        CoordinateGraph::Transformer    coords);
        };
    }
}
