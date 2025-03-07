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
    using namespace CoordinateGraph;
    using namespace ControlGraph;

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

        void operator()(Assign const& op, int tag);
        void operator()(Barrier const& op, int tag);
        void operator()(ComputeIndex const& op, int tag);
        void operator()(ConditionalOp const& op, int tag);
        void operator()(AssertOp const& op, int tag);
        void operator()(Deallocate const& op, int tag);
        void operator()(DoWhileOp const& op, int tag);
        void operator()(Exchange const& op, int tag);
        void operator()(ForLoopOp const& op, int tag);
        void operator()(Kernel const& op, int tag);
        void operator()(LoadLDSTile const& op, int tag);
        void operator()(LoadLinear const& op, int tag);
        void operator()(LoadTiled const& op, int tag);
        void operator()(LoadVGPR const& op, int tag);
        void operator()(LoadSGPR const& op, int tag);
        void operator()(Multiply const& op, int tag);
        void operator()(NOP const& op, int tag);
        void operator()(Block const& op, int tag);
        void operator()(Scope const& op, int tag);
        void operator()(SetCoordinate const& op, int tag);
        void operator()(StoreLDSTile const& op, int tag);
        void operator()(LoadTileDirect2LDS const& op, int tag);
        void operator()(StoreLinear const& op, int tag);
        void operator()(StoreTiled const& op, int tag);
        void operator()(StoreVGPR const& op, int tag);
        void operator()(StoreSGPR const& op, int tag);
        void operator()(TensorContraction const& op, int tag);
        void operator()(UnrollOp const& op, int tag);
        void operator()(WaitZero const& op, int tag);
        void operator()(SeedPRNG const& op, int tag);

    protected:
        void trackRegister(int control, int coordinate, ReadWrite rw);
        void trackConnections(int control, std::unordered_set<int> const& exclude, ReadWrite rw);

        bool hasGeneratedInputs(int const& tag);
        void generate(std::set<int> candidates);
        void call(Operation const& op, int tag);

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

    std::string   toString(ControlFlowRWTracer::ReadWrite const& rw);
    std::ostream& operator<<(std::ostream& stream, ControlFlowRWTracer::ReadWrite rw);

}
