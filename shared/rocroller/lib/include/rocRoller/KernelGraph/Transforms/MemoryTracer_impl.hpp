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
#include <set>
#include <variant>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>

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

            template <typename Model>
            void MemoryTracer::simulateLaunch(Model& model, uint numWorkgroups, uint numWorkitems)
            {
                auto rawArguments     = m_arguments.dataVector();
                auto runtimeArguments = RuntimeArguments(rawArguments.data(), rawArguments.size());

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
                            auto offsetValue = Expression::evaluate(event.index, runtimeArguments);
                            auto offset = std::visit([](auto x) { return (size_t)x; }, offsetValue);

                            // Break down the event into instruction-level events
                            // Each instruction can handle max 4 dwords (16 bytes)
                            uint remainingBytes   = event.bytesRequested;
                            uint currentOffset    = static_cast<uint>(offset);
                            uint instructionIndex = 0;

                            // Calculate total number of instructions needed
                            uint instructionCount = 0;
                            uint tempBytes        = event.bytesRequested;
                            while(tempBytes > 0)
                            {
                                uint instructionBytes = (tempBytes >= 16)  ? 16
                                                        : (tempBytes >= 8) ? 8
                                                        : (tempBytes >= 4) ? 4
                                                                           : 4;
                                instructionCount++;
                                tempBytes -= std::min(tempBytes, instructionBytes);
                            }

                            // Generate instruction-level events
                            while(remainingBytes > 0)
                            {
                                // Determine instruction size (try to maximize width)
                                uint instructionBytes;
                                if(remainingBytes >= 16)
                                {
                                    instructionBytes = 16; // 4 dwords (b128)
                                }
                                else if(remainingBytes >= 8)
                                {
                                    instructionBytes = 8; // 2 dwords (b64)
                                }
                                else if(remainingBytes >= 4)
                                {
                                    instructionBytes = 4; // 1 dword (b32)
                                }
                                else
                                {
                                    // Round up to 1 dword for sub-dword accesses
                                    instructionBytes = 4;
                                }

                                auto simulated = MemoryEventSimulated{event.operationTag,
                                                                      event.sourceTag,
                                                                      event.destinationTag,
                                                                      event.memoryOp,
                                                                      currentOffset,
                                                                      instructionBytes,
                                                                      wg,
                                                                      wi,
                                                                      instructionIndex,
                                                                      instructionCount};
                                model.simulate(simulated);

                                // Move to next instruction
                                remainingBytes -= instructionBytes;
                                currentOffset += instructionBytes;
                                instructionIndex++;
                            }
                        }
                    }
                }
            }

        }
    }
}
