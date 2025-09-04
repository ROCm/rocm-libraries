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
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace MemoryTracer
        {
            namespace Expression = rocRoller::Expression;
            using ExpressionPtr  = Expression::ExpressionPtr;

            enum Direction
            {
                GlobalLoad,
                GlobalStore,
                LDSLoad,
                LDSStore
            };

            struct MemoryInstruction
            {
                Direction             direction;
                int                   dwords; // 1 for b32, 2 for b64, 3 for b96, 4 for b128
                DataType              dataType;
                std::vector<uint32_t> addresses; // LDS/Global addresses accessed
            };

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
                Direction     direction; //< Memory access type
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
                int operationTag; //< Operation tag
                int sourceTag; //< Source coordinate tag
                int destinationTag; //< Destination coordinate tag
                Direction
                     direction; //< Memory access type: GlobalLoad, GlobalStore, LDSLoad, LDSStore
                uint byteOffset; //< Buffer offset in bytes
                uint bytesRequested; //< Number of bytes requested
                uint workGroup; //< Workgroup index
                uint workItem; //<Workitem index

                // XXX Consider adding SMEM vs VMEM, ie, if VMEM, this has a Workitem dependency
                //
                // If VMEM, possibly remove workItem and just keep a stride?
            };

            struct Summary
            {
                static constexpr bool echoBanks = false;

                struct Banks
                {
                    uint   bankIndex;
                    size_t workitemsAccessed;
                    bool   imbalanced;
                };
                struct Access
                {
                    int                           ldsTag;
                    std::vector<Banks>            accessedBanks;
                    std::vector<std::vector<int>> banksToWorkitems;
                };

                std::map<int, Access> accesses;
                std::set<int>         imbalancedTags;

                std::string toString() const;
            };

            std::ostream& operator<<(std::ostream& stream, Summary const& summary);
        }

        /**
	 * @brief Memory tracer for the rocRoller kernel graph.
	 *
	 * This is a work-in-progress implementation of a memory
	 * access analysis tool that simulates memory accesses.
	 */

        MemoryTracer::Summary memoryTrace(KernelGraph const&      original,
                                          KernelInvocation const& invocation);
    }
}
