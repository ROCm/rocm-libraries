// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

            auto operator<=>(ReadWriteRecord const& other) const = default;
        };

        ControlFlowRWTracer(KernelGraph const& graph,
                            int                start            = -1,
                            bool               trackConnections = false);

        /**
         * @brief Get all trace records.
         */
        std::vector<ReadWriteRecord> coordinatesReadWrite() const;

        /**
         * @brief Get trace records for a specific coordinate.
         */
        std::vector<ReadWriteRecord> coordinatesReadWrite(int coordinate) const;

        std::vector<ReadWriteRecord> opReadWrite(int op) const;

        /**
         * @brief Build backward dependencies for all coordinates.
         *
         * Computes a map showing which coordinates each coordinate depends on.
         * This is computed once and can be queried multiple times efficiently
         * via getCoordinateDependencies().
         *
         * Must be called before using getCoordinateDependencies().
         */
        void buildDependencies();

        /**
         * @brief Get all coordinate dependencies for a given coordinate.
         *
         * Returns the set of coordinates that the given coordinate depends on
         * (its provenance). Must call buildDependencies() first.
         *
         * @param coordinate The coordinate tag to trace dependencies for
         * @return Set of all coordinate tags that the given coordinate depends on
         */
        std::set<int> getCoordinateDependencies(int coordinate) const;

        void operator()(ControlGraph::AssertOp const& op, int tag);
        void operator()(ControlGraph::Assign const& op, int tag);
        void operator()(ControlGraph::Barrier const& op, int tag);
        void operator()(ControlGraph::Block const& op, int tag);
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
        void trackOffsetAndStride(int control, ReadWrite rw);
        void trackBuffer(int control, ReadWrite rw);

        bool hasGeneratedInputs(int const& tag);
        void generate(std::set<int> candidates);
        void call(ControlGraph::Operation const& op, int tag);

        KernelGraph const&           m_graph;
        std::set<int>                m_completedControlNodes;
        std::vector<ReadWriteRecord> m_trace;
        bool                         m_trackConnections;

        // Backward dependency cache: coordinate -> set of coordinates it depends on
        mutable std::map<int, std::set<int>> m_dependencies;
        mutable bool                         m_dependenciesBuilt = false;

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

    inline constexpr ControlFlowRWTracer::ReadWrite combine(ControlFlowRWTracer::ReadWrite a,
                                                            ControlFlowRWTracer::ReadWrite b)
    {
        if(a == b)
            return a;

        if(a == ControlFlowRWTracer::Count)
            return b;

        if(b == ControlFlowRWTracer::Count)
            return a;

        return ControlFlowRWTracer::READWRITE;
    }

    std::string toString(ControlFlowRWTracer::ReadWrite rw);

    std::string toString(ControlFlowRWTracer::ReadWriteRecord const& record);

    std::ostream& operator<<(std::ostream& stream, ControlFlowRWTracer::ReadWrite rw);

}
