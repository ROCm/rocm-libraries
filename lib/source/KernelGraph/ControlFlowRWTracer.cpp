
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

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

    void ControlFlowRWTracer::trace()
    {
        auto candidates = m_graph.control.roots().to<std::set>();
        generate(candidates);
    }

    void ControlFlowRWTracer::trace(int start)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(start).to<std::set>();
        generate(body);
    }

    std::map<int, int> ControlFlowRWTracer::deallocateLocations() const
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

    std::vector<ControlFlowRWTracer::ReadWriteRecord>
        ControlFlowRWTracer::coordinatesReadWrite() const
    {
        std::vector<ControlFlowRWTracer::ReadWriteRecord> rv;
        for(auto x : m_trace)
        {
            rv.push_back({x.control, x.coordinate, x.rw});
        }
        return rv;
    }

    void ControlFlowRWTracer::trackRegister(int control, int coordinate, ReadWrite rw)
    {
        // Ignore invalid tags
        if(control >= 0 && coordinate >= 0)
        {
            m_trace.push_back({static_cast<int>(m_loop.size()),
                               m_loop.empty() ? -1 : m_loop.back(),
                               control,
                               coordinate,
                               rw});
        }
    }

    bool ControlFlowRWTracer::hasGeneratedInputs(int const& tag)
    {
        auto inputs = m_graph.control.getInputNodeIndices<Sequence>(tag);
        for(auto const& input : inputs)
        {
            if(m_completedControlNodes.find(input) == m_completedControlNodes.end())
                return false;
        }
        return true;
    }

    void ControlFlowRWTracer::generate(std::set<int> candidates)
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

    void ControlFlowRWTracer::call(Operation const& op, int tag)
    {
        std::visit(*this, op, std::variant<int>(tag));
        m_completedControlNodes.insert(tag);
    }

    void ControlFlowRWTracer::operator()(Assign const& op, int tag)
    {
        auto dst = m_graph.mapper.getConnections(tag)[0].coordinate;
        trackRegister(tag, dst, ReadWrite::WRITE);

        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.expression);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }
    }

    void ControlFlowRWTracer::operator()(Barrier const& op, int tag) {}

    void ControlFlowRWTracer::operator()(ComputeIndex const& op, int tag)
    {
        // already in a scope
    }

    void ControlFlowRWTracer::operator()(Deallocate const& op, int tag) {}

    void ControlFlowRWTracer::operator()(ForLoopOp const& op, int tag)
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
        // 3. Expose all references in the graph.  Ideal but we
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

    void ControlFlowRWTracer::operator()(Kernel const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(LoadLDSTile const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        trackRegister(tag, dst, ReadWrite::WRITE);
        trackRegister(tag, m_graph.mapper.get<LDS>(tag), ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(LoadLinear const& op, int tag)
    {
        auto dst = m_graph.mapper.get<Linear>(tag);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadTiled const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadVGPR const& op, int tag)
    {
        auto dst = m_graph.mapper.get<VGPR>(tag);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(Multiply const& op, int tag)
    {
        auto a = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
        auto b = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
        auto dst
            = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

        trackRegister(tag, a, ReadWrite::READ);
        trackRegister(tag, b, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::READWRITE);
    }

    void ControlFlowRWTracer::operator()(Scope const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(SetCoordinate const& op, int tag)
    {
        if(m_setCoordinatesAsLoop)
            m_loop.push_back(tag);

        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);

        if(m_setCoordinatesAsLoop)
            m_loop.pop_back();
    }

    void ControlFlowRWTracer::operator()(StoreLDSTile const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        trackRegister(tag, dst, ReadWrite::READ);
        trackRegister(tag, m_graph.mapper.get<LDS>(tag), ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(StoreLinear const& op, int tag)
    {
        auto dst = m_graph.mapper.get<Linear>(tag);
        trackRegister(tag, dst, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreTiled const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        trackRegister(tag, dst, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreVGPR const& op, int tag)
    {
        auto dst = m_graph.mapper.get<VGPR>(tag);
        trackRegister(tag, dst, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(TensorContraction const& op, int tag) {}

    void ControlFlowRWTracer::operator()(UnrollOp const& op, int tag)
    {
        Throw<FatalError>("Not implemented yet.");
    }
}
