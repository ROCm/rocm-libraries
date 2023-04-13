
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
     * registers are modified.
     *
     * The main entry point is the `trace` method.  This walks the
     * control graph and records when registers are modified.  After a
     * trace is complete, the `deallocateLocations` method analyses
     * the trace and returns a list of coordinate+control pairs: a
     * Deallocate operation (to deallocate the register associated
     * with the coordinate) can be added after the control node.
     */
    class ControlFlowRWTracer
    {
    public:
        enum ReadWrite
        {
            READ,
            WRITE,
            READWRITE
        };

        struct EventRecord
        {
            int       depth, control, coordinate;
            ReadWrite rw;
        };

        struct ReadWriteRecord
        {
            int       control, coordinate;
            ReadWrite rw;
        };

        ControlFlowRWTracer(KernelGraph const& graph)
            : m_graph(graph)
        {
        }

        /**
         * @brief Walk the control graph and record register
         * read/write locations.
         */
        void trace();
        void trace(int start);

        std::vector<ReadWriteRecord> coordinatesReadWrite() const;

        void operator()(Assign const& op, int tag);
        void operator()(Barrier const& op, int tag);
        void operator()(ComputeIndex const& op, int tag);
        void operator()(Deallocate const& op, int tag);
        void operator()(ForLoopOp const& op, int tag);
        void operator()(Kernel const& op, int tag);
        void operator()(LoadLDSTile const& op, int tag);
        void operator()(LoadLinear const& op, int tag);
        void operator()(LoadTiled const& op, int tag);
        void operator()(LoadVGPR const& op, int tag);
        void operator()(Multiply const& op, int tag);
        void operator()(Scope const& op, int tag);
        void operator()(SetCoordinate const& op, int tag);
        void operator()(StoreLDSTile const& op, int tag);
        void operator()(StoreLinear const& op, int tag);
        void operator()(StoreTiled const& op, int tag);
        void operator()(StoreVGPR const& op, int tag);
        void operator()(TensorContraction const& op, int tag);
        void operator()(UnrollOp const& op, int tag);

    protected:
        void trackRegister(int control, int coordinate, ReadWrite rw);

        bool hasGeneratedInputs(int const& tag);
        void generate(std::set<int> candidates);
        void call(Operation const& op, int tag);

        KernelGraph                  m_graph;
        std::set<int>                m_completedControlNodes;
        std::vector<EventRecord>     m_trace;
        std::unordered_map<int, int> m_bodyParent;
        int                          m_depth = 0;
    };

}
