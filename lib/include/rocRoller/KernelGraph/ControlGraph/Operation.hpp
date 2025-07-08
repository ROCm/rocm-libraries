/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/StructUtils.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        /*
         * Control flow graph nodes.
         * Represent operations done on the input.
         */

        /**
         * Kernel - represents the start of a kernel.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Kernel);

        /**
         * Scope - represents a register scope.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Scope);

        /**
         * SetCoordinate - Sets the value of a Coordinate
         */
        struct SetCoordinate
        {
            SetCoordinate();
            explicit SetCoordinate(Expression::ExpressionPtr value);

            Expression::ExpressionPtr value;

            std::string name() const;
        };

        /**
         * DoWhileLoopOp - Represents a do-while loop.
         *
         * Must have nodes connected via the following outgoing edges:
         *
         * - Body: The loop body. The loop body must cause a change in the condition, this body will also be emitted at least once.
         *
         * There may be multiple outgoing edges for any of these.  Code that follows the for loop should be connected via a Sequence edge.
         *
         * condition is a scalar or vector condition and is executed before each iteration to determine if we must exit the loop.
         *
         * Currently generates code that behaves like:
         *
         * while_top:
         * <Body>
         * if(condition) goto while_top
         * <Sequence>
         */
        struct DoWhileOp
        {
            Expression::ExpressionPtr condition;

            std::string loopName;

            std::string name() const;
            std::string toString() const;
        };

        /**
         * ForLoopOp - Represents a for loop.
         *
         * Must have nodes connected via the following outgoing edges:
         *
         * - Initialize: Always executed once, when entering the for loop
         * - Body: The loop body.
         * - ForLoopIncrement: Executed after each iteration.
         *
         * There may be multiple outgoing edges for any of these.  Code that follows the for loop should be connected via a Sequence edge.
         *
         * condition is a scalar condition and is executed before each iteration to determine if we must exit the for loop.
         *
         * Currently generates code that behaves like:
         *
         * <Initialize>
         * if(!condition) goto for_bottom
         * for_top:
         * <Body>
         * <ForLoopIncrement>
         * if(condition) goto for_top
         * for_bottom:
         * <Sequence>
         */
        struct ForLoopOp
        {
            Expression::ExpressionPtr condition;

            std::string loopName;

            std::string name() const;
            std::string toString() const;
        };

        /**
         * ConditionalOp - Represents a conditional.
         *
         * Must have nodes connected via the following outgoing edges:
         *
         * - True  body:
         * - False body:
         *
         * Code that follows the Conditional Op regardless of the validity of the condition should be connected via a Sequence edge.
         *
         * Currently generates code that behaves like:
         *
         * if(condition)
         * <True Body>
         * else
         * <False Body>
         * <Sequence>
         *
        */
        struct ConditionalOp
        {
            Expression::ExpressionPtr condition;

            std::string conditionName;

            std::string name() const;
            std::string toString() const;
        };

        /**
         * AssertOp - Represents an assert.
         *
         * Must have nodes connected via the following outgoing edges:
         *
         * - True  Sequence:
         * - False <terminates kernel>:
         *
         * Currently generates code that behaves like:
         *
         * if(not condition)
         *   <terminates kernel>
         * <Sequence>
         *
         * Where <terminates kernel> is a instruction sequence that causes a trap or exception in kernel code.
         *
        */
        struct AssertOp
        {
            std::string assertName;

            Expression::ExpressionPtr condition;

            std::string name() const;
            std::string toString() const;
        };

        /**
         * UnrollOp - a kernel unroll.
         */
        struct UnrollOp
        {
            Expression::ExpressionPtr size;

            std::string name() const;
            std::string toString() const;
        };

        /*
         * Computes the value of `expression` and stores it into the associated register.
         *
         * If the register already exists, it must be of type 'regType'.  If not, `regType`
         * specifies which type of register will be allocated.
         */
        struct Assign
        {
            Register::Type            regType = Register::Type::Count;
            Expression::ExpressionPtr expression;

            size_t valueCount = 1;

            // If variableType is a packed type then
            // (valueCount / variableType.packing) registers will be allocated.
            std::optional<VariableType> variableType = std::nullopt;

            std::string name() const;
            std::string toString() const;
        };

        /**
         * @brief Represents a memory barrier
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Barrier);

        /**
         * @brief Computes offsets and strides between coordinates.
         *
         * Offsets and strides into the `target` dimension, based on
         * incrementing the `increment` dimension.
         *
         * Introduced to prevent recomputation (e.g. of an address)
         *
         * @param target Target dimension.
         * @param increment Increment dimension
         * @param base
         */
        struct ComputeIndex
        {
            // TODO: might be nicer to have UInt32 for strides; need
            // to allow user to specify stride types instead of
            // forcing size_t.
            ComputeIndex();
            ComputeIndex(bool     forward,
                         DataType valueType,
                         DataType offsetType = DataType::UInt64,
                         DataType strideType = DataType::UInt64);

            bool     forward    = false;
            DataType valueType  = DataType::Count;
            DataType offsetType = DataType::Count;
            DataType strideType = DataType::Count;

            std::string name() const;
        };

        /**
         * @brief Deallocates a register.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Deallocate);

        /**
         * LoadLinear - Load linear dimension.
         */
        struct LoadLinear
        {
            LoadLinear();
            explicit LoadLinear(rocRoller::VariableType const varType);

            rocRoller::VariableType varType;

            std::string name() const;
        };

        /**
         * LoadTiled.  Loads a tile (typically a MacroTile or
         * WaveTile).
         *
         * Storage location (LDS, VGPR, etc) is specified by the
         * `MemoryType` member of the MacroTile node.
         *
         * When loading a WaveTile, the storage layout (for MFMA
         * instructions) is specified by the `LayoutType` member of
         * the the WaveTile node.
         */
        struct LoadTiled
        {
            LoadTiled();
            explicit LoadTiled(VariableType const varType,
                               bool const         isTransposedTile = false,
                               bool const         isDirect2LDS     = false);

            VariableType varType;
            bool         isTransposedTile;
            bool         isDirect2LDS;

            std::string name() const;
        };

        /**
         * LoadVGPR - replaces LoadLinear.
         */
        struct LoadVGPR
        {
            LoadVGPR();
            explicit LoadVGPR(VariableType const varType, bool const scalar = false);

            VariableType varType;
            bool         scalar;

            std::string name() const;
        };

        /**
         * LoadSGPR - load scalar value from memory.
         */
        struct LoadSGPR
        {
            LoadSGPR();
            LoadSGPR(VariableType const varType, BufferInstructionOptions const bio);

            VariableType             varType;
            BufferInstructionOptions bufOpts;

            std::string name() const;
        };

        /**
         * LoadLDSTile - loads a tile from LDS
         */
        struct LoadLDSTile
        {
            LoadLDSTile();
            explicit LoadLDSTile(VariableType const varType, bool const isTransposedTile = false);

            VariableType varType;
            bool         isTransposedTile;

            std::string name() const;
            std::string toString() const;
        };

        struct LoadTileDirect2LDS
        {
            LoadTileDirect2LDS();
            explicit LoadTileDirect2LDS(VariableType const varType);

            VariableType varType;

            std::string name() const;
        };

        /**
         * Multiply - Multiply two MacroTiles
         */
        struct Multiply
        {
            Multiply();
            Multiply(Operations::ScaleMode scaleA, Operations::ScaleMode scaleB);

            Operations::ScaleMode scaleA;
            Operations::ScaleMode scaleB;

            std::string name() const;
        };

        /**
         * NOP - Do nothing.
         */
        RR_EMPTY_STRUCT_WITH_NAME(NOP);

        /**
         * Block - similar to NOP but Block locks the scheduler
         */
        RR_EMPTY_STRUCT_WITH_NAME(Block);

        /**
         * StoreLinear - Store linear dimension.
         */
        RR_EMPTY_STRUCT_WITH_NAME(StoreLinear);

        /**
         * StoreTiled.  Stores a tile.
         *
         * Storage location and affinity is specified by the MacroTile
         * node.
         */
        struct StoreTiled
        {
            StoreTiled();
            explicit StoreTiled(VariableType const dtype);

            VariableType             varType = DataType::Count;
            BufferInstructionOptions bufOpts;

            std::string name() const;
        };

        /**
         * StoreVGPR - replaces StoreLinear.
         */
        RR_EMPTY_STRUCT_WITH_NAME(StoreVGPR);

        /**
         * StoreSGPR - stores a scalar value to memory.
         */
        struct StoreSGPR
        {
            StoreSGPR();
            StoreSGPR(VariableType const varType, BufferInstructionOptions const bio);

            VariableType             varType;
            BufferInstructionOptions bufOpts;

            std::string name() const;
        };

        /**
         * StoreLDSTile - store a tile into LDS
         */
        struct StoreLDSTile
        {
            StoreLDSTile();
            explicit StoreLDSTile(VariableType const varType);

            VariableType varType;

            std::string name() const;
        };

        /**
         * TensorContraction - Tensor contraction operation.
         */
        struct TensorContraction
        {
            TensorContraction();
            TensorContraction(std::vector<int> const& aContractedDimensions,
                              std::vector<int> const& bContractedDimensions,
                              VariableType const      accType = DataType::Float);

            std::vector<int>      aDims, bDims; // contracted dimensions
            Operations::ScaleMode scaleModeA = Operations::ScaleMode::None;
            Operations::ScaleMode scaleModeB = Operations::ScaleMode::None;
            std::vector<size_t>   scaleStridesA;
            std::vector<size_t>   scaleStridesB;
            VariableType          accType = DataType::Float;

            std::string name() const;
        };

        /**
         * WaitZero - Emit a Wait Count of zero on all wait queues.
         *
         * This is important in preventing certain race conditions.
         * It forces the wait queues to be emptied before proceeding
         * to the next graph nodes (connected by Sequence edges).
         *
         * Example:
         * Store tile -> WaitZero -> Store sync flags
         */
        RR_EMPTY_STRUCT_WITH_NAME(WaitZero);

        /**
         * Exchange - permute the lanes data within a wave.
         */
        struct Exchange
        {
            Exchange();
            explicit Exchange(VariableType const varType);

            VariableType varType;

            std::string name() const;
        };

        /**
         * SeedPRNG - Set the initial seed value of a random number generator
         */
        struct SeedPRNG
        {
            SeedPRNG();
            explicit SeedPRNG(bool addTID);
            std::string toString() const;

            std::string name() const;

            // Add workitem ID to the seed if this flag is true
            bool addTID = false;
        };

        template <CConcreteOperation Op>
        std::string name(const Op& x);

        /*
         * Helpers
         */
        std::string name(const Operation& x);

        std::string toString(const Operation& x);

        /**
         * @brief Return the datatype associated with the Operation.
         */
        DataType getDataType(const Operation& x);

        VariableType getVariableType(Operation const& op);
        void         setVariableType(Operation& op, VariableType varType);
    }
}

#include <rocRoller/KernelGraph/ControlGraph/Operation_impl.hpp>
