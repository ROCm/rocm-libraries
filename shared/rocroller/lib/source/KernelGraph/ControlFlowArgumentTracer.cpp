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

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>

#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    namespace CG = ControlGraph;
    namespace CT = CoordinateGraph;

    void mergeSets(std::unordered_set<std::string>& dest, std::unordered_set<std::string> src)
    {
        if(dest.empty())
            dest = std::move(src);
        else
            dest.insert(src.begin(), src.end());
    }

    struct CoordinateArgumentTracer
    {
        std::unordered_set<std::string> const& trace(int idx, bool stride)
        {
            auto iter = m_cache.find({idx, stride});
            if(iter == m_cache.end())
            {
                if(m_graph.coordinates.getElementType(idx) == Graph::ElementType::Node)
                {
                    auto dimension = m_graph.coordinates.getNode(idx);
                    auto expr      = stride ? getStride(dimension) : getSize(dimension);

                    m_cache[{idx, stride}] = referencedKernelArguments(expr);
                }
                else
                {
                    m_cache[{idx, stride}] = {};
                }
                iter = m_cache.find({idx, stride});
            }

            return iter->second;
        }

        std::optional<std::string> call(int idx)
        {
            auto dim = m_graph.coordinates.getNode(idx);

            return call(idx, dim);
        }

        std::optional<std::string> call(int idx, CT::Dimension const& dim)
        {
            return std::visit(*this, singleVariant(idx), dim);
        }

        std::optional<std::string> operator()(int idx, CT::User const& dim)
        {
            return dim.argumentName;
        }

        std::optional<std::string> operator()(int idx, auto const& dim)
        {
            return std::nullopt;
        }

        KernelGraph const& m_graph;

        std::unordered_map<std::tuple<int, bool>, std::unordered_set<std::string>> m_cache;
    };

    struct ControlFlowArgumentVisitor : public TopoControlGraphVisitor<ControlFlowArgumentVisitor>
    {
        void call(int node)
        {
            auto op = m_graph.control.getNode(node);
            call(node, op);
        }

        void call(int node, CG::Operation const& op)
        {
            std::visit(*this, singleVariant(node), op);
        }

        void incorporate(int node, Expression::ExpressionPtr const& expr)
        {
            incorporate(node, referencedKernelArguments(expr, m_tagManager));
        }

        void incorporate(int node, std::unordered_set<std::string> const& args)
        {
            auto& dest = m_referencedArgs[node];

            for(auto const& arg : args)
            {
                incorporate(node, arg);
            }
        }

        void incorporate(int node, std::string const& arg)
        {
            auto& dest = m_referencedArgs[node];

            dest.insert(m_kernel->findArgument(arg).name);
        }

        void operator()(int node, CG::SetCoordinate const& op)
        {
            incorporate(node, op.value);
        }

        void operator()(
            int node,
            CIsAnyOf<CG::DoWhileOp, CG::ForLoopOp, CG::ConditionalOp, CG::AssertOp> auto const& op)
        {
            incorporate(node, op.condition);
        }

        void operator()(int node, CG::UnrollOp const& op)
        {
            incorporate(node, op.size);
        }

        void operator()(int node, CG::Assign const& op)
        {
            incorporate(node, op.expression);
        }

        void operator()(int node, CG::ComputeIndex const& op)
        {
            for(int argIdx = 0; argIdx < static_cast<int>(Connections::ComputeIndexArgument::Count);
                argIdx++)
            {
                auto arg = static_cast<Connections::ComputeIndexArgument>(argIdx);

                auto dim = m_graph.mapper.get(node, Connections::ComputeIndex{arg});

                if(dim > 0)
                {
                    incorporate(node, m_tracer.trace(dim, true));

                    if(arg == Connections::ComputeIndexArgument::TARGET)
                    {
                        auto dir = op.forward ? Graph::Direction::Upstream
                                              : Graph::Direction::Downstream;

                        auto [coords, path] = findRequiredCoordinates(dim, dir, m_graph);

                        path = includeEdgeNeighbours(m_graph.coordinates, opposite(dir), path);

                        std::unordered_set<std::string> targetCoords;
                        for(auto coord : coords)
                        {
                            mergeSets(targetCoords, m_tracer.trace(coord, false));
                            mergeSets(targetCoords, m_tracer.trace(coord, true));
                        }
                        for(auto coord : path)
                        {
                            mergeSets(targetCoords, m_tracer.trace(coord, false));
                            mergeSets(targetCoords, m_tracer.trace(coord, true));
                        }

                        auto stride = m_graph.mapper.get(
                            node,
                            Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});

                        if(stride > 0)
                        {
                            Expression::ExpressionPtr expr;
                            for(auto const& argName : targetCoords)
                            {
                                auto arg = std::make_shared<AssemblyKernelArgument>(
                                    argName, DataType::Int32);

                                auto argExpr
                                    = std::make_shared<Expression::Expression>(std::move(arg));

                                if(expr)
                                    expr = expr + argExpr;
                                else
                                    expr = argExpr;
                            }

                            if(expr)
                                m_tagManager.addExpression(stride, expr, {});
                        }

                        incorporate(node, std::move(targetCoords));

                        auto buffer = m_graph.mapper.get(
                            node,
                            Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});
                        if(buffer > 0)
                        {
                            incorporate(node, m_tracer.trace(dim, false));
                            auto pointer = m_tracer.call(dim);
                            if(pointer)
                                incorporate(node, std::move(*pointer));
                        }
                    }
                }
            }
        }

        void operator()(int                                 node,
                        CIsAnyOf<CG::LoadLDSTile, //
                                 CG::LoadLinear,
                                 CG::LoadSGPR,
                                 CG::StoreSGPR> auto const& op)
        {
            auto [target, dir] = getOperationTarget(node, m_graph);

            incorporate(node, m_tracer.trace(target, false));

            auto [coords, path] = findAllRequiredCoordinates(node, m_graph);

            for(auto coord : coords)
            {
                incorporate(node, m_tracer.trace(coord, true));
            }
            for(auto coord : path)
            {
                incorporate(node, m_tracer.trace(coord, false));
            }

            auto pointer = m_tracer.call(target);
            if(pointer)
                incorporate(node, std::move(*pointer));
        }

        void operator()(int                                  node,
                        CIsAnyOf<CG::LoadTileDirect2LDS, //
                                 CG::LoadTiled,
                                 CG::StoreTiled> auto const& op)
        {
            auto [target, dir] = getOperationTarget(node, m_graph);

            auto [coords, path] = findAllRequiredCoordinates(node, m_graph);

            for(auto coord : path)
                incorporate(node, m_tracer.trace(coord, true));
        }

        void operator()(int node, CIsAnyOf<CG::LoadVGPR, CG::StoreVGPR> auto const& op)
        {
            auto [target, dir] = getOperationTarget(node, m_graph);

            auto [coords, path] = findAllRequiredCoordinates(node, m_graph);

            for(auto coord : path)
                incorporate(node, m_tracer.trace(coord, true));

            auto pointer = m_tracer.call(target);
            if(pointer)
                incorporate(node, std::move(*pointer));
        }

        void operator()(int node, CG::Deallocate const& op)
        {
            incorporate(node, std::unordered_set<std::string>{});

            auto dimTag = m_graph.mapper.get<CT::Dimension>(node);

            if(m_tagManager.hasExpression(dimTag))
            {
                m_tagManager.deleteTag(dimTag);
            }
        }

        void operator()(int                                node,
                        CIsAnyOf<CG::Barrier,
                                 CG::Block,
                                 CG::Exchange,
                                 CG::Kernel,
                                 CG::StoreLDSTile,
                                 CG::StoreLinear,
                                 CG::SeedPRNG,
                                 CG::Multiply,
                                 CG::NOP,
                                 CG::Scope,
                                 CG::TensorContraction,
                                 CG::WaitZero> auto const& op)
        {
            incorporate(node, std::unordered_set<std::string>{});
        }

        ControlFlowArgumentVisitor(KernelGraph const& graph, AssemblyKernelPtr kernel)
            : TopoControlGraphVisitor(graph)
            , m_tracer{graph}
            , m_tagManager(nullptr)
            , m_kernel(std::move(kernel))
        {
        }

        void fixup()
        {
            for(auto const& arg : m_kernel->arguments())
            {
                auto argArgs = referencedKernelArguments(arg.expression);
                for(auto const& argArg : argArgs)
                {
                    m_subReferencedArgs[argArg].insert(arg.name);
                }
            }

            bool any = false;
            do
            {
                any = false;

                for(auto& [arg, subArgs] : m_subReferencedArgs)
                {
                    std::unordered_set<std::string> additions;

                    for(auto const& subArg : subArgs)
                    {
                        auto iter = m_subReferencedArgs.find(subArg);
                        if(iter != m_subReferencedArgs.end())
                        {
                            additions.insert(iter->second.begin(), iter->second.end());
                        }
                    }

                    auto beforeSize = subArgs.size();
                    subArgs.insert(additions.begin(), additions.end());
                    auto afterSize = subArgs.size();

                    if(afterSize > beforeSize)
                        any = true;
                }

            } while(any);

            for(auto const& [arg, subArgs] : m_subReferencedArgs)
                for(auto const& subArg : subArgs)
                    Log::debug("SubArg: {} -> {}", arg, subArg);

            Log::debug("-=-=-=-=-=-=-=-=-=-=");

            do
            {
                any = false;

                for(auto& [node, referencedArgs] : m_referencedArgs)
                {
                    std::unordered_set<std::string> additions;

                    for(auto const& arg : referencedArgs)
                    {
                        auto iter = m_subReferencedArgs.find(arg);
                        if(iter != m_subReferencedArgs.end())
                        {
                            additions.insert(iter->second.begin(), iter->second.end());
                        }
                    }

                    auto beforeSize = referencedArgs.size();
                    referencedArgs.insert(additions.begin(), additions.end());
                    auto afterSize = referencedArgs.size();

                    if(afterSize > beforeSize)
                        any = true;
                }
            } while(any);
        }

        CoordinateArgumentTracer m_tracer;

        std::unordered_map<int, std::unordered_set<std::string>> m_referencedArgs;

        // If we reference a given kernel argument, we might also or instead
        // need to reference a kernel argument derived from that original
        // argument, e.g. if we reference "x", we might in practice reference
        // MagicMultiple(x) instead.
        // subReferencedArgs is a map from an arg to all the args that
        // reference that arg.
        std::unordered_map<std::string, std::unordered_set<std::string>> m_subReferencedArgs;

        RegisterTagManager m_tagManager;

        AssemblyKernelPtr m_kernel;
    };

    ControlFlowArgumentTracer::ControlFlowArgumentTracer(KernelGraph const&       kgraph,
                                                         AssemblyKernelPtr const& kernel)
    {
        TIMER(t, "ControlFlowArgumentTracer");

        ControlFlowArgumentVisitor visitor(kgraph, kernel);

        visitor.walk();
        visitor.fixup();

        m_referencedArguments = std::move(visitor.m_referencedArgs);

        for(auto arg : kernel->arguments())
            m_neverReferencedArguments.insert(arg.name);

        for(auto const& [node, args] : m_referencedArguments)
        {
            for(auto const& arg : args)
                m_neverReferencedArguments.erase(arg);
        }

        if(m_neverReferencedArguments.size() > 0)
        {
            std::ostringstream msg;
            msg << "Argument(s) ";
            streamJoin(msg, m_neverReferencedArguments, ", ");
            msg << " are never referenced!";

            Log::warn(msg.str());
        }
    }

    std::unordered_set<std::string> const&
        ControlFlowArgumentTracer::referencedArguments(int controlNode) const
    {
        auto iter = m_referencedArguments.find(controlNode);
        AssertFatal(iter != m_referencedArguments.end(), ShowValue(controlNode));

        return iter->second;
    }

    std::unordered_map<int, std::unordered_set<std::string>> const&
        ControlFlowArgumentTracer::referencedArguments() const
    {
        return m_referencedArguments;
    }

    std::set<std::string> const& ControlFlowArgumentTracer::neverReferencedArguments() const
    {
        return m_neverReferencedArguments;
    }
}
