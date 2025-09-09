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

#pragma once
#include <array>
#include <list>
#include <memory>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
#include <set>
#include <variant>

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

            enum class Direction
            {
                Load,
                Store
            };

            struct MemoryOpGlobal
            {
                Direction direction;
            };

            struct MemoryOpLDS
            {
                Direction direction;
            };

            using MemoryOp = std::variant<MemoryOpGlobal, MemoryOpLDS>;

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
             */
            struct MemoryEventSimulated
            {
                int      operationTag; //< Operation tag
                int      sourceTag; //< Source coordinate tag
                int      destinationTag; //< Destination coordinate tag
                MemoryOp memoryOp; //< Memory operation type
                uint     byteOffset; //< Buffer offset in bytes
                uint     bytesRequested; //< Number of bytes requested
                uint     workGroup; //< Workgroup index
                uint     workItem; //<Workitem index

                // XXX Consider adding SMEM vs VMEM, ie, if VMEM, this has a Workitem dependency
                //
                // If VMEM, possibly remove workItem and just keep a stride?
            };

            /**
             * @brief Memory tracer for the kernel graph.
             *
             * This class walks the control graph and builds a list of
             * MemoryEventExpression objects.  These objects represent
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
                void simulateLaunch(Model& model, uint numWorkgroups, uint numWorkitems)
                {
                    // TODO: revisit -> have the memory model generate a list of MemoryEventSimulated
                    // that the model can consume?
                    // TODO: move this to _impl.hpp
                    auto rawArguments = m_arguments.dataVector();
                    auto runtimeArguments
                        = RuntimeArguments(rawArguments.data(), rawArguments.size());

                    auto setWorkgroup = [&](uint i, uint v) {
                        *((uint*)(rawArguments.data() + m_workgroupOffset[i])) = v;
                    };
                    auto setWorkitem = [&](uint i, uint v) {
                        *((uint*)(rawArguments.data() + m_workitemOffset[i])) = v;
                    };

                    for(auto const& event : m_events)
                    {
                        if(not model.filter(event))
                            continue;

                        for(uint wg = 0; wg < numWorkgroups; ++wg)
                        {
                            setWorkgroup(0, wg);
                            for(uint wi = 0; wi < numWorkitems; ++wi)
                            {
                                setWorkitem(0, wi);

                                // Might want to cache these
                                auto offsetValue
                                    = Expression::evaluate(event.index, runtimeArguments);
                                auto offset
                                    = std::visit([](auto x) { return (size_t)x; }, offsetValue);
                                auto simulated = MemoryEventSimulated{event.operationTag,
                                                                      event.sourceTag,
                                                                      event.destinationTag,
                                                                      event.memoryOp,
                                                                      static_cast<uint>(offset),
                                                                      event.bytesRequested,
                                                                      wg,
                                                                      wi};
                                model.simulate(simulated);
                            }
                        }
                    }
                }

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
