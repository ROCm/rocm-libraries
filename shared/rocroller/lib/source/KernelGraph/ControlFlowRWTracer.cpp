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

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    std::string toString(ControlFlowRWTracer::ReadWriteRecord const& record)
    {
        return fmt::format(
            "ctrl {} {} coord {}", record.control, toString(record.rw), record.coordinate);
    }

    /**
     * @brief Collect all coordinate tags referenced in an Expression.
     */
    struct CollectDataFlowExpressionVisitor
    {
        std::set<int> tags;

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

        void operator()(Expression::ScaledMatrixMultiply const& expr)
        {
            if(expr.matA)
            {
                call(expr.matA);
            }
            if(expr.matB)
            {
                call(expr.matB);
            }
            if(expr.matC)
            {
                call(expr.matC);
            }
            if(expr.scaleA)
            {
                call(expr.scaleA);
            }
            if(expr.scaleB)
            {
                call(expr.scaleB);
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
            tags.insert(expr.tag);
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

    ControlFlowRWTracer::ControlFlowRWTracer(KernelGraph const& graph,
                                             int                start,
                                             bool               trackConnections)
        : m_graph(graph)
        , m_trackConnections(trackConnections)
    {
        TIMER(t, "ControlFlowRWTracer")
        if(start == -1)
            trace();
        else
            trace(start);
    }

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

    std::vector<ControlFlowRWTracer::ReadWriteRecord>
        ControlFlowRWTracer::coordinatesReadWrite() const
    {
        return m_trace;
    }

    std::vector<ControlFlowRWTracer::ReadWriteRecord>
        ControlFlowRWTracer::coordinatesReadWrite(int coordinate) const
    {
        std::vector<ControlFlowRWTracer::ReadWriteRecord> rv;
        std::copy_if(m_trace.begin(),
                     m_trace.end(),
                     std::back_inserter(rv),
                     [coordinate](ControlFlowRWTracer::ReadWriteRecord x) {
                         return coordinate == x.coordinate;
                     });
        return rv;
    }

    void ControlFlowRWTracer::trackRegister(int control, int coordinate, ReadWrite rw)
    {
        if(control < 0 || coordinate < 0)
            return;
        m_trace.push_back({control, coordinate, rw});
    }

    void ControlFlowRWTracer::trackConnections(int                            control,
                                               std::unordered_set<int> const& except,
                                               ReadWrite                      rw)
    {
        if(!m_trackConnections)
            return;

        if(control < 0)
            return;

        for(auto const& c : m_graph.mapper.getConnections(control))
        {
            if(except.contains(c.coordinate))
                continue;
            if(m_graph.coordinates.exists(c.coordinate))
                m_trace.push_back({control, c.coordinate, rw});
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
            AssertRecoverable(!nodes.empty(),
                              "ControlFlowRWTracer:Invalid control graph!",
                              ShowValue(m_graph.control),
                              ShowValue(candidates),
                              ShowValue(m_completedControlNodes));

            // Visit all the nodes we found.
            for(auto const& tag : nodes)
            {
                auto op = m_graph.control.getNode(tag);
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
        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.expression);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }

        auto dst = m_graph.mapper.getConnections(tag)[0].coordinate;
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(Barrier const& op, int tag) {}

    void ControlFlowRWTracer::operator()(ComputeIndex const& op, int tag)
    {
        // Already in a Scope
    }

    void ControlFlowRWTracer::operator()(ConditionalOp const& op, int tag)
    {
        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.condition);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }

        auto trueBody = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(trueBody);

        auto falseBody = m_graph.control.getOutputNodeIndices<Else>(tag).to<std::set>();
        generate(falseBody);
    }

    void ControlFlowRWTracer::operator()(AssertOp const& op, int tag)
    {
        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.condition);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }
    }

    void ControlFlowRWTracer::operator()(Deallocate const& op, int tag) {}

    void ControlFlowRWTracer::operator()(DoWhileOp const& op, int tag)
    {
        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.condition);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }

        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(ForLoopOp const& op, int tag)
    {
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

        CollectDataFlowExpressionVisitor visitor;
        visitor.call(op.condition);
        for(auto src : visitor.tags)
        {
            trackRegister(tag, src, ReadWrite::READ);
        }

        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(Kernel const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(LoadLDSTile const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        auto lds = m_graph.mapper.get<LDS>(tag);

        dst = only(m_graph.coordinates.getInputNodeIndices(dst, CT::isEdge<CT::View>))
                  .value_or(dst);
        lds = only(m_graph.coordinates.getOutputNodeIndices(lds, CT::isEdge<CT::View>))
                  .value_or(lds);

        trackRegister(tag, lds, ReadWrite::READ);
        trackConnections(tag, {dst, lds}, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadLinear const& op, int tag)
    {
        auto dst = m_graph.mapper.get<Linear>(tag);
        trackConnections(tag, {dst}, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(SeedPRNG const& op, int tag)
    {
        // Tracking read/write from/to the VGPR that stores the seed of random number generator
        // This VGPR should be deallocated by tracer when it is no longer being used.
        auto seedVGPR = m_graph.mapper.get(tag, NaryArgument::DEST);
        trackRegister(tag, seedVGPR, ReadWrite::WRITE);
        trackConnections(tag, {seedVGPR}, ReadWrite::READ);

        auto rhs = m_graph.mapper.get(tag, NaryArgument::RHS);
        trackRegister(tag, rhs, ReadWrite::READ);
        trackConnections(tag, {rhs}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(LoadTiled const& op, int tag)
    {

        auto dst = m_graph.mapper.get<MacroTile>(tag);

        dst = only(m_graph.coordinates.getInputNodeIndices(dst, CT::isEdge<CT::View>))
                  .value_or(dst);

        trackConnections(tag, {dst}, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadVGPR const& op, int tag)
    {
        auto dst = m_graph.mapper.get<VGPR>(tag);
        trackConnections(tag, {dst}, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadSGPR const& op, int tag)
    {
        auto dst = m_graph.mapper.get<VGPR>(tag);
        trackConnections(tag, {dst}, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(Multiply const& op, int tag)
    {
        auto a = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
        trackRegister(tag, a, ReadWrite::READ);

        if(op.scaleA == Operations::ScaleMode::Separate)
        {
            auto aScale = m_graph.mapper.get(
                tag, Connections::typeArgument<MacroTile>(NaryArgument::LHS_SCALE));
            aScale = only(m_graph.coordinates.getOutputNodeIndices(aScale, CT::isEdge<CT::Index>))
                         .value_or(aScale);
            trackRegister(tag, aScale, ReadWrite::READ);
        }
        else if(op.scaleA == Operations::ScaleMode::SingleScale)
        {
            auto aScale = m_graph.mapper.get(tag, NaryArgument::LHS_SCALE);
            AssertFatal(aScale != -1);
            trackRegister(tag, aScale, ReadWrite::READ);
        }

        auto b = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
        trackRegister(tag, b, ReadWrite::READ);

        if(op.scaleB == Operations::ScaleMode::Separate)
        {
            auto bScale = m_graph.mapper.get(
                tag, Connections::typeArgument<MacroTile>(NaryArgument::RHS_SCALE));
            bScale = only(m_graph.coordinates.getOutputNodeIndices(bScale, CT::isEdge<CT::Index>))
                         .value_or(bScale);
            trackRegister(tag, bScale, ReadWrite::READ);
        }
        else if(op.scaleB == Operations::ScaleMode::SingleScale)
        {
            auto bScale = m_graph.mapper.get(tag, NaryArgument::RHS_SCALE);
            AssertFatal(bScale != -1);
            trackRegister(tag, bScale, ReadWrite::READ);
        }

        auto dst
            = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::DEST));
        trackRegister(tag, dst, ReadWrite::READWRITE);
    }

    void ControlFlowRWTracer::operator()(NOP const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(Block const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(Scope const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(SetCoordinate const& op, int tag)
    {
        auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
        generate(body);
    }

    void ControlFlowRWTracer::operator()(StoreLDSTile const& op, int tag)
    {
        auto dst = m_graph.mapper.get<MacroTile>(tag);
        auto lds = m_graph.mapper.get<LDS>(tag);
        trackRegister(tag, dst, ReadWrite::READ);
        trackConnections(tag, {dst, lds}, ReadWrite::READ);
        trackRegister(tag, lds, ReadWrite::WRITE);
    }

    void ControlFlowRWTracer::operator()(LoadTileDirect2LDS const& op, int tag)
    {
        auto source = m_graph.mapper.get<MacroTile>(tag);
        auto dst    = m_graph.mapper.get<LDS>(tag);
        trackRegister(tag, source, ReadWrite::READ);
        trackRegister(tag, dst, ReadWrite::WRITE);
        trackConnections(tag, {source, dst}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreLinear const& op, int tag)
    {
        auto src = m_graph.mapper.get<Linear>(tag);
        trackRegister(tag, src, ReadWrite::READ);
        trackConnections(tag, {src}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreTiled const& op, int tag)
    {
        auto src = m_graph.mapper.get<MacroTile>(tag);

        src = only(m_graph.coordinates.getOutputNodeIndices(src, CT::isEdge<CT::View>))
                  .value_or(src);

        trackRegister(tag, src, ReadWrite::READ);
        trackConnections(tag, {src}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreVGPR const& op, int tag)
    {
        auto src = m_graph.mapper.get<VGPR>(tag);
        trackRegister(tag, src, ReadWrite::READ);
        trackConnections(tag, {src}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(StoreSGPR const& op, int tag)
    {
        auto src = m_graph.mapper.get<VGPR>(tag);
        trackRegister(tag, src, ReadWrite::READ);
        trackConnections(tag, {src}, ReadWrite::READ);
    }

    void ControlFlowRWTracer::operator()(TensorContraction const& op, int tag) {}

    void ControlFlowRWTracer::operator()(UnrollOp const& op, int tag)
    {
        Throw<FatalError>("ControlFlowRWTracer UnrollOp not implemented yet.");
    }

    void ControlFlowRWTracer::operator()(WaitZero const& op, int tag) {}

    void ControlFlowRWTracer::operator()(Exchange const& op, int tag)
    {
        auto src = m_graph.mapper.get<MacroTile>(tag);
        src      = only(m_graph.coordinates.getOutputNodeIndices(src, CT::isEdge<CT::Index>))
                  .value_or(src);
        trackRegister(tag, src, ReadWrite::READ);

        auto dst
            = m_graph.mapper.get(tag, Connections::typeArgument<MacroTile>(NaryArgument::DEST));
        trackRegister(tag, dst, ReadWrite::READWRITE);
    }

    std::string toString(ControlFlowRWTracer::ReadWrite rw)
    {
        switch(rw)
        {
        case ControlFlowRWTracer::READ:
            return "READ";
        case ControlFlowRWTracer::WRITE:
            return "WRITE";
        case ControlFlowRWTracer::READWRITE:
            return "READWRITE";
        case ControlFlowRWTracer::Count:
            break;
        }

        Throw<FatalError>("Invalid ReadWrite.");
    }

    std::ostream& operator<<(std::ostream& stream, ControlFlowRWTracer::ReadWrite rw)
    {
        return stream << toString(rw);
    }

}
