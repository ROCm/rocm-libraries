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
 * @file MemoryTracer.cpp
 * @author rocRoller Developers
 * @brief Memory tracer for the rocRoller kernel graph.
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

#include <map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    namespace MemoryTracer
    {
        namespace Expression = rocRoller::Expression;
        using ExpressionPtr  = Expression::ExpressionPtr;

        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using namespace CoordinateGraph;
        using namespace ControlGraph;

        MemoryTracer::MemoryTracer(KernelGraph const& graph)
            : m_graph(graph)
        {
            for(int i = 0; i < 3; ++i)
            {
                m_workgroupOffset[i] = m_arguments.size();
                auto wg_name         = concatenate("WG", i);
                auto wg_carg         = CommandArgument(nullptr,
                                               DataType::UInt32,
                                               m_workgroupOffset[i],
                                               DataDirection::ReadOnly,
                                               wg_name);
                auto wg              = std::make_shared<CommandArgument>(wg_carg);
                m_arguments.appendUnbound<uint>(wg_name);

                m_workitemOffset[i] = m_arguments.size();
                auto wi_name        = concatenate("WI", i);
                auto wi_carg        = CommandArgument(nullptr,
                                               DataType::UInt32,
                                               m_workitemOffset[i],
                                               DataDirection::ReadOnly,
                                               wi_name);
                auto wi             = std::make_shared<CommandArgument>(wi_carg);
                m_arguments.appendUnbound<uint>(wi_name);

                m_kernelWorkgroupIndexes[i] = std::make_shared<Expression::Expression>(wg);
                m_kernelWorkitemIndexes[i]  = std::make_shared<Expression::Expression>(wi);
            }
        }

        void MemoryTracer::trace()
        {
            Log::debug("MemoryTracer::trace()");
            auto coordinateGraph
                = std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                    m_graph.coordinates);

            auto coords = Transformer(coordinateGraph.get());
            coords.fillExecutionCoordinates(
                nullptr, m_kernelWorkgroupIndexes, m_kernelWorkitemIndexes);

            auto candidates = m_graph.control.roots().to<std::set>();
            generate(candidates, coords);
        }

        bool MemoryTracer::hasGeneratedInputs(int const& tag)
        {
            auto inputs = m_graph.control.getInputNodeIndices<Sequence>(tag);
            for(auto const& input : inputs)
            {
                if(m_completedControlNodes.find(input) == m_completedControlNodes.end())
                    return false;
            }
            return true;
        }

        void MemoryTracer::generate(std::set<int> candidates, Transformer coords)
        {
            while(!candidates.empty())
            {
                std::set<int> nodes;

                // Find all candidate nodes whose inputs have been satisfied
                for(auto const& tag : candidates)
                    if(hasGeneratedInputs(tag))
                        nodes.insert(tag);

                // If there are none, we have a problem.
                AssertFatal(!nodes.empty(),
                            "Invalid control graph!",
                            ShowValue(m_graph.control),
                            ShowValue(candidates));

                // Visit all the nodes we found.
                for(auto const& tag : nodes)
                {
                    auto op = std::get<Operation>(m_graph.control.getElement(tag));
                    call(tag, op, coords);
                }

                // Add output nodes to candidates.
                for(auto const& tag : nodes)
                {
                    auto outTags = m_graph.control.getOutputNodeIndices<Sequence>(tag);
                    candidates.insert(outTags.begin(), outTags.end());
                }

                // Delete generated nodes from candidates.
                for(auto const& node : nodes)
                    candidates.erase(node);
            }
        }

        // Local visitor struct to handle operation dispatch
        struct OperationVisitor
        {
            MemoryTracer& tracer;
            int           tag;
            Transformer   coords;

            void operator()(AssertOp const& op) {}

            void operator()(Assign const& op) {}

            void operator()(Barrier const& op) {}

            void operator()(Block const& op)
            {
                auto body = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(body, coords);
            }

            void operator()(ComputeIndex const& op) {}

            void operator()(ConditionalOp const& op)
            {
                auto trueBody
                    = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(trueBody, coords);
                auto elseBody
                    = tracer.m_graph.control.getOutputNodeIndices<Else>(tag).to<std::set>();
                if(!elseBody.empty())
                {
                    tracer.generate(elseBody, coords);
                }
            }

            void operator()(Deallocate const& op) {}

            void operator()(DoWhileOp const& op) {}

            void operator()(Exchange const& op) {}

            void operator()(ForLoopOp const& op)
            {
                auto loopIncrTag = tracer.m_graph.mapper.get(tag, NaryArgument::DEST);
                auto loopDims
                    = tracer.m_graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loopIncrTag);
                for(auto const& dim : loopDims)
                {
                    // XXX this is a hack, we should have a way to set the coordinate
                    Log::warn("Setting coordinate {} to 0 for ForLoop", dim);
                    coords.setCoordinate(dim, Expression::literal(0));
                }

                auto body = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(body, coords);
            }

            void operator()(Kernel const& op)
            {
                auto body = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(body, coords);
            }

            void operator()(LoadLDSTile const& load)
            {
                auto [ldsTag, lds]   = tracer.m_graph.getDimension<LDS>(tag);
                auto [tileTag, tile] = tracer.m_graph.getDimension<MacroTile>(tag);

                auto maybeParentLDS = only(
                    tracer.m_graph.coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    ldsTag = *maybeParentLDS;

                if(tile.memoryType == MemoryType::WAVE)
                {
                    auto [waveTileTag, waveTile] = tracer.m_graph.getDimension<WaveTile>(tag);
                    auto [vgprTag, vgpr]         = tracer.m_graph.getDimension<VGPR>(tag);

                    auto dataTypeInfo = DataTypeInfo::Get(load.varType);
                    auto numBits
                        = static_cast<uint>(dataTypeInfo.elementBits / dataTypeInfo.packing);
                    auto numElements = getUnsignedInt(evaluate(vgpr.size));
                    auto numBytes    = (numBits * numElements) / 8u;

                    coords.setCoordinate(vgprTag, Expression::literal(0));
                    auto index = coords.reverse({ldsTag})[0];

                    Log::info("LDS WAVE LOAD: tag {}, numBits {}, numElements {}, numBytes {}",
                              tag,
                              numBits,
                              numElements,
                              numBytes);

                    tracer.m_events.push_back({tag,
                                               ldsTag,
                                               tileTag,
                                               MemoryOpLDS{LdsDirection::Read},
                                               index * Expression::literal(numBits),
                                               numBytes});
                }
            }

            void operator()(LoadTileDirect2LDS const& op) {}

            void operator()(LoadLinear const& op) {}

            void operator()(LoadTiled const& load) {}

            void operator()(LoadVGPR const& load) {}

            void operator()(LoadSGPR const& load) {}

            void operator()(Multiply const& op) {}

            void operator()(NOP const& op) {}

            void operator()(Scope const& op)
            {
                auto body = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(body, coords);
            }

            void operator()(SeedPRNG const& op) {}

            void operator()(SetCoordinate const& setCoordinate)
            {
                auto connections = tracer.m_graph.mapper.getConnections(tag);
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto init
                    = tracer.m_graph.control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                tracer.generate(init, coords);

                auto body = tracer.m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                tracer.generate(body, coords);
            }

            void operator()(StoreLDSTile const& op) {}

            void operator()(StoreLinear const& op) {}

            void operator()(StoreTiled const& op) {}

            void operator()(StoreVGPR const& op) {}

            void operator()(StoreSGPR const& op) {}

            void operator()(TensorContraction const& op) {}

            void operator()(UnrollOp const& op) {}

            void operator()(WaitZero const& op) {}
        };

        void MemoryTracer::call(int tag, Operation const& op, Transformer coords)
        {
            auto opName = toString(op);
            Log::debug("MemoryTracer::{}({})", opName, tag);
            OperationVisitor visitor{*this, tag, coords};
            std::visit(visitor, op);
            m_completedControlNodes.insert(tag);
        }
    }
}
