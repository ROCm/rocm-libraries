
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    /**
     * @brief Collect all coordinate tags referenced in an Expression.
     */
    struct CollectDataFlowExpressionVisitor
    {
        std::vector<int> tags;

        template <Expression::CUnary Expr>
        void operator()(Expr const& expr)
        {
            if(expr.arg)
            {
                call(expr.arg);
            }
        }

        template <Expression::CBinary Expr>
        void operator()(Expr const& expr)
        {
            if(expr.lhs)
            {
                call(expr.lhs);
            }
            if(expr.rhs)
            {
                call(expr.rhs);
            }
        }

        template <Expression::CTernary Expr>
        void operator()(Expr const& expr)
        {
            if(expr.lhs)
            {
                call(expr.lhs);
            }
            if(expr.r1hs)
            {
                call(expr.r1hs);
            }
            if(expr.r2hs)
            {
                call(expr.r2hs);
            }
        }

        void operator()(Expression::DataFlowTag const& expr)
        {
            tags.push_back(expr.tag);
        }

        template <Expression::CValue Value>
        void operator()(Value const& expr)
        {
        }

        void call(Expression::ExpressionPtr expr)
        {
            if(expr)
            {
                std::visit(*this, *expr);
            }
        }
    };

    /**
     * @brief Register read/write tracer.
     *
     * The tracer walks the control flow graph and records when
     * registers are modified.
     *
     * The main entry point is the `trace` method.  This walks the
     * control graph and records when registers are modified.  After a
     * trace is complete, the `deallocateLocations` method analyses
     * the trace and returns a list of coordinate+control pairs: a
     * Deallocate operation (to deallocate the register associated
     * with the coordinate) can be added after the control node.
     */
    struct ControlFlowRWTracer
    {
        struct EventRecord
        {
            int depth, loop, control, coordinate;
        };

        ControlFlowRWTracer(KernelGraph const& graph)
            : m_graph(graph)
        {
        }

        /**
         * @brief Walk the control graph and record register
         * read/write locations.
         */
        void trace()
        {
            auto candidates = m_graph.control.roots().to<std::set>();
            generate(candidates);
        }

        /**
         * @brief Analyse the trace and return locations where
         * Deallocate operations can be added.
         */
        std::map<int, int> deallocateLocations() const
        {
            //
            // We need to avoid situations like: we can't deallocate a
            // register in the middle of a for-loop if it was created
            // out-side the for-loop.
            //
            // Note: Using `insert_or_assign` means we'll be left with
            // the last one...
            //

            // track operation that touches coordinate last
            std::map<int, int> rv;

            // track event that touches coordinate last, at lowest
            // depth (closest to kernel)
            std::map<int, EventRecord> ev;

            for(auto x : m_trace)
            {
                // have we seen this coordinate before?
                if(ev.count(x.coordinate) > 0)
                {
                    // if last recorded event was not at the same
                    // depth, deallocate when we reach last loop at
                    // the same depth
                    if(ev.at(x.coordinate).depth < x.depth)
                    {
                        rv.insert_or_assign(x.coordinate, x.loop);
                        continue;
                    }
                }

                rv.insert_or_assign(x.coordinate, x.control);
                ev.insert_or_assign(x.coordinate, x);
            }
            return rv;
        }

        /*
         * tracing...
         */

        void trackRegister(int control, int coordinate)
        {
            m_trace.push_back({static_cast<int>(m_loop.size()),
                               m_loop.empty() ? -1 : m_loop.back(),
                               control,
                               coordinate});
        }

        bool hasGeneratedInputs(int const& tag)
        {
            auto inputs = m_graph.control.getInputNodeIndices<Sequence>(tag);
            for(auto const& input : inputs)
            {
                if(m_completedControlNodes.find(input) == m_completedControlNodes.end())
                    return false;
            }
            return true;
        }

        void generate(std::set<int> candidates)
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
                    call(op, tag);
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

        void call(Operation const& op, int tag)
        {
            std::visit(*this, op, std::variant<int>(tag));
            m_completedControlNodes.insert(tag);
        }

        void operator()(Assign const& op, int tag)
        {
            auto dst = m_graph.mapper.getConnections(tag)[0].coordinate;
            trackRegister(tag, dst);

            CollectDataFlowExpressionVisitor visitor;
            visitor.call(op.expression);
            for(auto src : visitor.tags)
            {
                trackRegister(tag, src);
            }
        }

        void operator()(Barrier const& op, int tag) {}

        void operator()(ComputeIndex const& op, int tag)
        {
            // already in a scope
        }

        void operator()(Deallocate const& op, int tag) {}

        void operator()(ForLoopOp const& op, int tag)
        {
            m_loop.push_back(tag);

            //
            // Don't examine for loop intialize or increment operations.
            //
            // Assign operations within loop initialisation operations
            // are scoped already.
            //
            // Assign operations within loop increment operations
            // typically involve: incrementing loop counters and
            // offsets.  Loop counters are scoped already.
            //
            // Offsets are created "inside" ComputeIndex nodes and are
            // used in other nodes like LoadTiled.  These "inside"
            // references do not explicitly appear in the graph.
            //
            // If we examine loop increment operations and "track" an
            // offset increment, but don't track it during loads, then
            // a Deallocate node would be mis-placed.
            //
            // A few solutions:
            //
            // 1. Don't examine loop increment operations.  They
            // already appear in Scopes so are deallocated regardless.
            // Fairly easy but perhaps we miss an opporunity to free
            // up registers early.
            //
            // 2. Teach the tracker how to dig into all nodes.  Very
            // tedious and not future-proof.
            //
            // 3. Expose all references in the grpah.  Ideal but we
            // aren't there yet.
            //

            // auto init = m_graph.control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
            // generate(init);

            // auto incr = m_graph.control.getOutputNodeIndices<ForLoopIncrement>(tag).to<std::set>();
            // generate(incr);

            auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
            generate(body);

            m_loop.pop_back();
        }

        void operator()(Kernel const& op, int tag)
        {
            auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
            generate(body);
        }

        void operator()(LoadLDSTile const& op, int tag)
        {
            auto dst = m_graph.mapper.get<MacroTile>(tag);
            trackRegister(tag, dst);
        }

        void operator()(LoadLinear const& op, int tag)
        {
            auto dst = m_graph.mapper.get<Linear>(tag);
            trackRegister(tag, dst);
        }

        void operator()(LoadTiled const& op, int tag)
        {
            auto dst = m_graph.mapper.get<MacroTile>(tag);
            trackRegister(tag, dst);
        }

        void operator()(LoadVGPR const& op, int tag)
        {
            auto dst = m_graph.mapper.get<VGPR>(tag);
            trackRegister(tag, dst);
        }

        void operator()(Multiply const& op, int tag)
        {
            auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
            generate(body);

            auto a   = m_graph.mapper.get<MacroTile>(tag, 0);
            auto b   = m_graph.mapper.get<MacroTile>(tag, 1);
            auto dst = m_graph.mapper.get<MacroTile>(tag, 2);

            trackRegister(tag, a);
            trackRegister(tag, b);
            trackRegister(tag, dst);
        }

        void operator()(Scope const& op, int tag)
        {
            auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
            generate(body);
        }

        void operator()(SetCoordinate const& op, int tag)
        {
            auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
            generate(body);
        }

        void operator()(StoreLDSTile const& op, int tag)
        {
            auto dst = m_graph.mapper.get<MacroTile>(tag);
            trackRegister(tag, dst);
        }

        void operator()(StoreLinear const& op, int tag)
        {
            auto dst = m_graph.mapper.get<Linear>(tag);
            trackRegister(tag, dst);
        }

        void operator()(StoreTiled const& op, int tag)
        {
            auto dst = m_graph.mapper.get<MacroTile>(tag);
            trackRegister(tag, dst);
        }

        void operator()(StoreVGPR const& op, int tag)
        {
            auto dst = m_graph.mapper.get<VGPR>(tag);
            trackRegister(tag, dst);
        }

        void operator()(TensorContraction const& op, int tag) {}

        void operator()(UnrollOp const& op, int tag)
        {
            Throw<FatalError>("Not implemented yet.");
        }

    private:
        KernelGraph              m_graph;
        std::set<int>            m_completedControlNodes;
        std::vector<EventRecord> m_trace;
        std::vector<int>         m_loop;
    };

    KernelGraph addDeallocate(KernelGraph const& original)
    {
        auto graph  = original;
        auto tracer = ControlFlowRWTracer(graph);

        tracer.trace();
        for(auto kv : tracer.deallocateLocations())
        {
            auto coordinate = kv.first;
            auto control    = kv.second;
            auto deallocate = graph.control.addElement(Deallocate());

            std::vector<int> children;
            for(auto elem : graph.control.getNeighbours<Graph::Direction::Downstream>(control))
            {
                auto sequence = graph.control.get<Sequence>(elem);
                if(sequence)
                {
                    for(auto child :
                        graph.control.getNeighbours<Graph::Direction::Downstream>(elem))
                    {
                        children.push_back(child);
                    }
                    graph.control.deleteElement(elem);
                }
            }

            graph.control.addElement(Sequence(), {control}, {deallocate});

            for(auto child : children)
            {
                graph.control.addElement(Sequence(), {deallocate}, {child});
            }

            graph.mapper.connect<Dimension>(deallocate, coordinate);
        }

        return graph;
    }
}
