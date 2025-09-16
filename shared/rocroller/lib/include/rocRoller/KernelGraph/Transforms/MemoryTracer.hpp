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

/**
 * Memory tracer for the rocRoller kernel graph.
 *
 * This file implements a memory tracer that simulates memory accesses
 * in a kernel graph.
 *
 * The general idea is:
 *
 * 1. Instantiate a `MemoryTracer()` object with the kernel graph.
 *
 * 2. Call `trace()` to walk the control graph and generate a list of
 *    memory events.  Each memory event roughly corresponds to a
 *    memory instruction that the code-generator will emit.
 *
 *    This step is done once.
 *
 * 3. For each memory effect that you want to simulate, instantiate a
 *    "model".
 *
 *    For example, the `LDSBankModel()` focuses on LDS read/writes,
 *    and tries to predict LDS bank conflicts.
 *
 *    a. Call the tracer's `simulateLaunch()` and provide your model.
 *
 *    b. The `simulateLaunch()` method will "blow up" all memory
 *       events by evaluating the indexing expression for a collection
 *       of `Workgroup` and `Workitem` values into a large collection
 *       of `MemoryEventSimulated` objects.
 *
 *    c. Each of these simulated memory events will be passed to your
 *       model through the `simulate()` method.
 */

#pragma once
#include <array>
#include <list>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CoordinateGraph
        {
            class Transformer;
        }

        namespace MemoryTracer
        {
            namespace Expression = rocRoller::Expression;
            using ExpressionPtr  = Expression::ExpressionPtr;

            enum class LdsDirection
            {
                Read,
                Write
            };

            struct MemoryOpLDS
            {
                LdsDirection direction;
            };

            using MemoryOp = std::variant<MemoryOpLDS>;

            /**
             * @brief Memory event expression.
             *
             * This structure roughly corresponds to memory instruction
             * that the code-generator will emit.
             */
            struct MemoryEventExpression
            {
                int           operationTag; //< Operation tag
                int           sourceTag; //< Source coordinate tag
                int           destinationTag; //< Destination coordinate tag
                MemoryOp      memoryOp; //< Memory operation type
                ExpressionPtr index; //< Index expression
                uint          bytesRequested; //< Number of bytes requested
            };

            /**
             * @brief Memory event simulated.
             *
             * This is a "blown up" version of `MemoryEventExpression`.
             *
             * Note that each `MemoryEventExpression` has an index
             * expression that may contain `Workgroup` and/or `Workitem`
             * coordinates.
             *
             * The `MemoryTracer` will evaluate the index expression in
             * `MemoryEventSimulated` for a collection of `Workgroup` and
             * `Workitem` values and create a "blown up" version of the
             * memory event that contains the actual byte offset.
             * 
             * Each event represents a single instruction with max 4 dwords (128 bits).
             */
            struct MemoryEventSimulated
            {
                int      operationTag; //< Operation tag
                int      sourceTag; //< Source coordinate tag
                int      destinationTag; //< Destination coordinate tag
                MemoryOp memoryOp; //< Memory operation type
                uint     byteOffset; //< Buffer offset in bytes
                uint     bytesRequested; //< Number of bytes requested (max 16 bytes/4 dwords)
                uint     workGroup; //< Workgroup index
                uint     workItem; //<Workitem index
                uint     instructionIndex; //< Instruction index within the operation
                uint     instructionCount; //< Total number of instructions for this operation

                // For future: consider adding SMEM vs VMEM, ie, if VMEM, this has a Workitem dependency
                // If VMEM, possibly remove workItem and just keep a stride?
            };

            /**
             * @brief Memory tracer for the kernel graph.
             *
             * This class walks the control graph and builds a list of
             * MemoryEventExpression objects. These objects represent
             * instructions that the code-generator will emit.
             *
             * Note that the base LDS allocation address is assumed to be
             * zero.  If you are comparing the bank indexes reported here
             * vs those computed by, eg, inspecting register values, you
             * may see a discrepancy.  However, the number of bank
             * conflicts should be the same.
             * 
             * Note that not all visit operations are correctly implemented.
             */
            struct MemoryTracer
            {
                MemoryTracer(KernelGraph const& graph);

                /**
                 * @brief Walk the control graph and generate memory events
                 */
                void trace();

                /**
                 * @brief Simulate memory launches with given model
                 */
                template <typename Model>
                void simulateLaunch(Model& model, uint numWorkgroups, uint numWorkitems);

            private:
                KernelGraph   m_graph;
                std::set<int> m_completedControlNodes;

                std::list<MemoryEventExpression> m_events;

                KernelArguments m_arguments;

                std::array<uint, 3>          m_workgroupOffset, m_workitemOffset;
                std::array<ExpressionPtr, 3> m_kernelWorkgroupIndexes, m_kernelWorkitemIndexes;

                bool hasGeneratedInputs(int const& tag);
                void generate(std::set<int> candidates, CoordinateGraph::Transformer coords);
                void call(int                            tag,
                          ControlGraph::Operation const& op,
                          CoordinateGraph::Transformer   coords);

                friend class OperationVisitor;
            };

        }
    }
}

#include <rocRoller/KernelGraph/Transforms/MemoryTracer_impl.hpp>