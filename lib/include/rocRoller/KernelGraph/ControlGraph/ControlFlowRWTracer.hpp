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

#pragma once

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    /**
     * @brief Register read/write tracer.
     *
     * The tracer walks the control flow graph and records when
     * coordinates are accessed/modified.
     *
     * The `coordinatesReadWrite` methods can be used to query the
     * recorded trace for all operations in the control graph that
     * access or modify a coordinate.
     */
    class ControlFlowRWTracer
    {
    public:
        enum ReadWrite
        {
            READ,
            WRITE,
            READWRITE,

            Count
        };

        struct ReadWriteRecord
        {
            int       control, coordinate;
            ReadWrite rw;
        };

        ControlFlowRWTracer(KernelGraph const& graph, int start = -1, bool trackConnections = false)
            : m_graph(graph)
            , m_trackConnections(trackConnections)
        {
            if(start == -1)
                trace();
            else
                trace(start);
        }

        /**
         * @brief Get all trace records.
         */
        std::vector<ReadWriteRecord> coordinatesReadWrite() const;

        /**
         * @brief Get trace records for a specific coordinate.
         */
        std::vector<ReadWriteRecord> coordinatesReadWrite(int coordinate) const;

        /**
         * @brief Get map of body-parents.
         */
        std::unordered_map<int, int> getBodyParents() const;

        void operator()(ControlGraph::AssertOp const& op, int tag);
        void operator()(ControlGraph::Assign const& op, int tag);
        void operator()(ControlGraph::Barrier const& op, int tag);
        void operator()(ControlGraph::Block const& op, int tag);
        void operator()(ControlGraph::ComputeIndex const& op, int tag);
        void operator()(ControlGraph::ConditionalOp const& op, int tag);
        void operator()(ControlGraph::Deallocate const& op, int tag);
        void operator()(ControlGraph::DoWhileOp const& op, int tag);
        void operator()(ControlGraph::Exchange const& op, int tag);
        void operator()(ControlGraph::ForLoopOp const& op, int tag);
        void operator()(ControlGraph::Kernel const& op, int tag);
        void operator()(ControlGraph::LoadLDSTile const& op, int tag);
        void operator()(ControlGraph::LoadLinear const& op, int tag);
        void operator()(ControlGraph::LoadSGPR const& op, int tag);
        void operator()(ControlGraph::LoadTileDirect2LDS const& op, int tag);
        void operator()(ControlGraph::LoadTiled const& op, int tag);
        void operator()(ControlGraph::LoadVGPR const& op, int tag);
        void operator()(ControlGraph::Multiply const& op, int tag);
        void operator()(ControlGraph::NOP const& op, int tag);
        void operator()(ControlGraph::Scope const& op, int tag);
        void operator()(ControlGraph::SeedPRNG const& op, int tag);
        void operator()(ControlGraph::SetCoordinate const& op, int tag);
        void operator()(ControlGraph::StoreLDSTile const& op, int tag);
        void operator()(ControlGraph::StoreLinear const& op, int tag);
        void operator()(ControlGraph::StoreSGPR const& op, int tag);
        void operator()(ControlGraph::StoreTiled const& op, int tag);
        void operator()(ControlGraph::StoreVGPR const& op, int tag);
        void operator()(ControlGraph::TensorContraction const& op, int tag);
        void operator()(ControlGraph::UnrollOp const& op, int tag);
        void operator()(ControlGraph::WaitZero const& op, int tag);

    protected:
        void trackRegister(int control, int coordinate, ReadWrite rw);
        void trackConnections(int control, std::unordered_set<int> const& exclude, ReadWrite rw);

        bool hasGeneratedInputs(int const& tag);
        void generate(std::set<int> candidates);
        void call(ControlGraph::Operation const& op, int tag);

        KernelGraph                  m_graph;
        std::set<int>                m_completedControlNodes;
        std::vector<ReadWriteRecord> m_trace;
        std::unordered_map<int, int> m_bodyParent;
        bool                         m_trackConnections;

    private:
        /**
         * @brief Walk the control graph starting with the roots
         * and record register read/write locations.
         */
        void trace();

        /**
         * @brief Walk the control graph starting with the `start`
         * node and record register read/write locations in its body.
         */
        void trace(int start);
    };

    std::string toString(ControlFlowRWTracer::ReadWrite rw);

    std::string toString(ControlFlowRWTracer::ReadWriteRecord const& record);

    std::ostream& operator<<(std::ostream& stream, ControlFlowRWTracer::ReadWrite rw);

}
