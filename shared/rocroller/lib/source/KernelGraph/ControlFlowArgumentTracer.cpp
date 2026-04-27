// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <variant>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>

#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
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

            dest.insert(m_kernel->findArgument(arg).getName());
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
            auto dest = m_graph.mapper.get(node, NaryArgument::DEST);
            if(dest > 0)
            {
                incorporate(node, m_tracer.trace(dest, true));
                incorporate(node, m_tracer.trace(dest, false));
                if(op.strideExpressionAttributes)
                {
                    incorporate(node, op.strideExpressionAttributes->elementBlockStride);
                    incorporate(node, op.strideExpressionAttributes->trLoadPairStride);

                    // Register the stride expression in the TagManager so that other nodes
                    // referencing this stride coordinate (via DataFlowTag) can trace dependencies.
                    m_tagManager.addExpression(dest, op.expression, {});
                }
            }

            incorporate(node, op.expression);
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

            auto subDimensionCoordinates
                = filterCoordinates<CoordinateGraph::SubDimension>(path, m_graph);
            auto isPretiled = subDimensionCoordinates.size() > 2;

            for(auto coord : path)
            {
                incorporate(node, m_tracer.trace(coord, true));

                auto isSubDimension
                    = m_graph.coordinates.get<CoordinateGraph::SubDimension>(coord).has_value();
                if(isSubDimension && isPretiled)
                {
                    Log::debug("Pretiled coordinate: {}", coord);
                    incorporate(node, m_tracer.trace(coord, false));
                }
            }
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
                auto argArgs = referencedKernelArguments(arg.getExpression());
                for(auto const& argArg : argArgs)
                {
                    m_subReferencedArgs[argArg].insert(arg.getName());
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

            // Collect directly referenced args before propagation
            // These are args actually used in control flow operations
            for(auto const& [node, args] : m_referencedArgs)
            {
                for(auto const& arg : args)
                {
                    m_directlyReferencedArgs.insert(arg);
                }
            }
        }

        // Arguments directly used in control flow (before propagation)
        std::set<std::string> m_directlyReferencedArgs;

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
            m_neverReferencedArguments.insert(arg.getName());

        // Collect all referenced args and compute launch-time-only args
        std::set<std::string> allReferenced;
        for(auto const& [node, args] : m_referencedArguments)
        {
            for(auto const& arg : args)
            {
                m_neverReferencedArguments.erase(arg);
                allReferenced.insert(arg);
            }
        }

        if(m_neverReferencedArguments.size() > 0)
        {
            std::ostringstream msg;
            msg << "Argument(s) ";
            streamJoin(msg, m_neverReferencedArguments, ", ");
            msg << " are never referenced!";
            Log::debug(msg.str());
        }

        if(visitor.m_directlyReferencedArgs.size() > 0)
        {
            std::ostringstream msg;
            msg << "Directly referenced args (" << visitor.m_directlyReferencedArgs.size() << "): ";
            streamJoin(msg, visitor.m_directlyReferencedArgs, ", ");
            Log::debug(msg.str());
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
