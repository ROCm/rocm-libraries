// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

#include <rocRoller/Expression_fwd.hpp>
#include <string>
#include <vector>

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
            LoadStoreTileGenerator(KernelGraphPtr, ContextPtr, unsigned int);

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
             * @brief Generate instructions needed to load a tile from global memory direct to lds
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction>
                genLoadTileDirect2LDS(int                                     tag,
                                      ControlGraph::LoadTileDirect2LDS const& load,
                                      CoordinateGraph::Transformer            coords);

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
             * @brief Generate instructions needed to load a tile from global into LDS with TDM
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genLoadTiledTDMToLDS(int                                    tag,
                                                        ControlGraph::LoadTiledTDMToLDS const& load,
                                                        CoordinateGraph::Transformer coords);

            /**
             * @brief Information needed in order to load or store a tile.
             *
             * @field tag The tag of the control graph node generating the load or store
             * @field kind The kind of memory instruction to use
             * @field m Number of rows in the tile
             * @field n Number of columns in the tile
             * @field dataType The type of the data being loaded
             * @field isTransposedTile if tile needs to be transposed
             * @field vgpr The registers to store the data in (null is loading)
             * @field offset Offset from the starting index
             */
            struct LoadStoreTileInfo
            {
                int                            tag          = -1;
                MemoryInstructions::MemoryKind kind         = MemoryInstructions::MemoryKind::Count;
                uint64_t                       m            = 0;
                uint64_t                       n            = 0;
                uint32_t                       elementBits  = 0;
                uint32_t                       packedAmount = 0;
                uint32_t                       ldsWriteStride = 0;
                Register::ValuePtr             data           = nullptr;
                VariableType                   varType        = VariableType{DataType::Count};
                Register::ValuePtr             rowOffsetReg   = nullptr;
                Register::ValuePtr             rowStrideReg   = nullptr;
                RegisterExpressionAttributes   rowStrideAttributes;
                Register::ValuePtr             colStrideReg = nullptr;
                RegisterExpressionAttributes   colStrideAttributes;
                Register::ValuePtr             offset               = nullptr;
                Register::ValuePtr             bufDesc              = nullptr;
                BufferInstructionOptions       bufOpts              = {};
                bool                           isTransposedTile     = false;
                bool                           isPadded             = false;
                bool                           isMacroTileRowStride = false;
                Register::ValuePtr             tdmDesc              = nullptr;
                std::vector<std::string>       comments;
            };

            LoadStoreTileInfo getLoadLDSTileInfo(int tag, ControlGraph::LoadLDSTile const& load);
            LoadStoreTileInfo getStoreLDSTileInfo(int tag, ControlGraph::StoreLDSTile const& store);

        private:
            ContextPtr                       m_context;
            KernelGraphPtr                   m_graph;
            Expression::ExpressionTransducer m_fastArith;
            unsigned int                     m_workgroupSizeTotal;

            inline Generator<Instruction> generate(auto&                     dest,
                                                   Expression::ExpressionPtr expr) const;

            // Index calculation Helpers
            Register::ValuePtr        getBufferDesc(int tag);
            Expression::ExpressionPtr getOffsetExpr(int  opTag,
                                                    bool isStorePartOfGlobalToLDS,
                                                    CoordinateGraph::Transformer const& coords);
            Generator<Instruction>    getOffset(LoadStoreTileInfo&           info,
                                                CoordinateGraph::Transformer coords,
                                                bool                         preserveOffset,
                                                bool isStorePartOfGlobalToLDS = false);

            /**
             * @brief Generate stride (in bytes).
             *
             * The `unitStride` flag is set if the generated
             * byte-stride corresponds to a unit element-stride.  A
             * unit element-stride is a unitary (=1) stride with
             * respect to the element of the underlying data type.
             *
             * The generated stride is in bytes.  This facilitates,
             * eg, advancing offset registers to the next macro tile
             * by simply adding the stride in the increment of a for
             * loop.
             *
             * However, determining whether a byte-stride in a
             * stride-expression is a unit-stride is tricky for
             * sub-byte datatypes.  To make this more robust,
             * stride-expressions have meta-data attached to the
             * expression to make this explicit.
             *
             * For example, if we only knew the byte-stride:
             *
             * | data type | byte-stride | unit element-stride |
             * |-----------|-------------|---------------------|
             * | FP64      | 8           | true                |
             * | FP32      | 4           | true                |
             * | FP32      | 8           | false               |
             * | FP16      | 2           | true                |
             * | FP8       | 1           | true                |
             * | Sub-byte  | 1           | maybe!              |
             */
            Generator<Instruction> generateStride(Register::ValuePtr&           stride,
                                                  RegisterExpressionAttributes& attrs,
                                                  int                           tag,
                                                  int                           dimension);

            // Move Tile Helpers
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTile(LoadStoreTileInfo&            info,
                                            CoordinateGraph::Transformer& coords);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileLiteralStrides(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileColStrideOne(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileRuntimeStrides(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileDirect2LDS(LoadStoreTileInfo& info,
                                                      int                numBytes,
                                                      bool               setM0,
                                                      Register::ValuePtr readAddr);
            Generator<Instruction> loadTileLiteralStridesPack(LoadStoreTileInfo& info);
            Generator<Instruction> loadTileRuntimeStridesPack(LoadStoreTileInfo& info);

            // Load Tile Helpers
            Generator<Instruction> loadMacroTileVGPR(int                            tag,
                                                     ControlGraph::LoadTiled const& load,
                                                     CoordinateGraph::Transformer   coords);
            Generator<Instruction> loadMacroTileWAVE(int                            tag,
                                                     ControlGraph::LoadTiled const& load,
                                                     CoordinateGraph::Transformer   coords);
            Generator<Instruction> loadMacroTileWAVECIACCUM(int                            tag,
                                                            ControlGraph::LoadTiled const& load,
                                                            CoordinateGraph::Transformer   coords);
            Generator<Instruction>
                              loadMacroTileDirect2LDS(int                                     tag,
                                                      ControlGraph::LoadTileDirect2LDS const& load,
                                                      CoordinateGraph::Transformer            coords);
            LoadStoreTileInfo loadMacroTileLDSInfo(int tag, ControlGraph::LoadLDSTile const& load);
            LoadStoreTileInfo loadMacroTileWAVELDSInfo(int                              tag,
                                                       ControlGraph::LoadLDSTile const& load);
            LoadStoreTileInfo storeMacroTileLDSInfo(int                               tag,
                                                    ControlGraph::StoreLDSTile const& store);
            LoadStoreTileInfo storeMacroTileWAVELDSInfo(int                               tag,
                                                        ControlGraph::StoreLDSTile const& store);

            // Store Tile Helpers
            Generator<Instruction> storeMacroTileVGPR(int                             tag,
                                                      ControlGraph::StoreTiled const& store,
                                                      CoordinateGraph::Transformer    coords);
            Generator<Instruction> storeMacroTileWAVE(int                             tag,
                                                      ControlGraph::StoreTiled const& store,
                                                      CoordinateGraph::Transformer    coords);
        };

        std::string toString(LoadStoreTileGenerator::LoadStoreTileInfo const& info);

    }
}
