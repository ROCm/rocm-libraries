#pragma once

#include <string>
#include <vector>

#include "ControlEdge.hpp"
#include "Operation.hpp"

#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/TagType.hpp>

namespace rocRoller
{
    /**
     * Control flow routines.
     *
     * Control flow is represented as a graph.  Nodes in the
     * control flow graph represent operations (like load/store or a
     * for loop).  Edges in the graph encode depencies between nodes.
     *
     * Nodes may have negative tag values, to indicate unallocated tags.
     * When the node is added to a graph (control or coordinate), it will
     * be replaced with a positive value that is otherwise unused in that
     * graph.
     *
     */
    namespace KernelGraph::ControlGraph
    {
        /**
         * Control flow graph.
         *
         * Nodes in the graph represent operations.  Edges describe
         * dependencies.
         */

        using EdgeKey = std::pair<TagType, TagType>;
        using Edge    = std::pair<EdgeKey, ControlEdge>;

        class ControlGraph
        {
        public:
            ControlGraph() {}
            ControlGraph(int nextTag)
                : m_nextTag(nextTag)
            {
            }

            void addEdge(std::vector<Operation> const& srcs,
                         std::vector<Operation> const& dsts,
                         ControlEdge const&            edge);

            void addEdge(std::vector<CoordinateTransform::Dimension> const& srcs,
                         std::vector<Operation> const&                      dsts);
            void addEdge(std::vector<CoordinateTransform::Dimension> const& srcs,
                         std::vector<Operation> const&                      dsts,
                         ControlEdge const&                                 edge);

            /**
             * @brief Remove edge within the graph from src to dst.
             *
             * Returns the edge that was deleted.
             *
             * @param src
             * @param dst
             * @return Edge
             */
            Edge removeEdge(Operation const& src, Operation const& dst);

            /**
             * Assigns a new, unused tag value to `dim`.
             */
            TagType allocateTag(Operation& dim);

            /**
             * Returns a new, unused tag.
             */
            int allocateTag();

            /**
             * Returns a tag value that is greater than all tag values currently in this graph.
             */
            int nextTag() const;

            /**
             * Return Operation given tag.
             */
            Operation getOperation(TagType tag) const;

            /**
             * @brief Return all operations of class T.
             */
            template <typename T>
            std::vector<T> findOperations() const;

            /**
             * Return operations in the graph, in topological order.
             */
            std::vector<Edge> getOperationEdges() const;

            ControlEdge getEdge(EdgeKey const& key) const;

            /**
             * Return operations that immediately preceed `dst`.
             */
            std::vector<Operation> getInputs(TagType const& dst) const;
            std::vector<TagType>   getInputTags(TagType const& src) const;

            /**
             * Return operations that immediately preceed `dst` and are of type T.
             */
            template <CControlEdge T>
            std::vector<Operation> getInputs(TagType const& dst) const;
            std::vector<Operation> getInputs(TagType const& dst, ControlEdge edge) const;

            template <CControlEdge T>
            std::vector<TagType> getInputTags(TagType const& dst) const;
            std::vector<TagType> getInputTags(TagType const& dst, ControlEdge edge) const;

            /**
             * Return operations that immediately follow `src`.
             */
            std::vector<Operation> getOutputs(TagType const& src) const;
            std::vector<TagType>   getOutputTags(TagType const& src) const;

            /**
             * Return operations that immediately follow `src` and are of type T.
             */
            template <CControlEdge T>
            std::vector<Operation> getOutputs(TagType const& src) const;
            std::vector<Operation> getOutputs(TagType const& dst, ControlEdge edge) const;

            template <CControlEdge T>
            std::vector<TagType> getOutputTags(TagType const& src) const;
            std::vector<TagType> getOutputTags(TagType const& src, ControlEdge edge) const;

            /**
             * Return all operations.
             */
            std::vector<Operation> getOperations() const;
            std::vector<Operation> getOperations(std::vector<TagType> const& tags) const;

            Operation getRootOperation() const;

            /**
             * Reset/overwrite an operation.
             */
            void resetOperation(Operation const& op);

            /**
             * @brief Remove the operation from the graph with the provided tag.
             *
             * @param tag Operation to remove.
             * @param fullyCompletely When true, remove all incoming and outgoing edges too.
             */
            void removeOperation(TagType const& tag, bool fullyCompletely = false);

            /**
             * Render control flow graph into stream `ss`.
             */
            void toDOT(std::ostream& stream, std::string prefix) const;

        private:
            std::map<TagType, Operation>   m_nodes;
            std::map<EdgeKey, ControlEdge> m_edges;

            int m_nextTag = -1;

            /**
             * Internally recognize that the tag value of `tag` has been used.
             */
            void recognizeTag(TagType tag);

            friend struct rocRoller::KernelGraph::KernelUnrollVisitor;
        };

        std::ostream& operator<<(std::ostream& stream, ControlGraph const& graph);
    }
}

#include "ControlGraph_impl.hpp"
