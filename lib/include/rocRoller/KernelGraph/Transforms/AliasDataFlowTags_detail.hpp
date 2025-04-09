/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AliasDataFlowTagsDetail
        {
            using Record = ControlFlowRWTracer::ReadWriteRecord;
            using Edge   = ControlGraph::ControlEdge;

            class TagRWGraph : public Graph::Hypergraph<Record, Edge, false>
            {
                using Base = Graph::Hypergraph<Record, Edge, false>;
                using Base::Hypergraph;
            };

            /**
             * Represents a span of time in the kernel.
             *
             * There could be some ambiguity still left to the scheduler as to
             * the beginning or end of the span as those nodes may not be
             * ordered relative to each other.
             *
             * - For something to come before this extent, it would have to be
             *   before all the nodes in the `begin` set.
             * - For something to come after this extent, it would have to be
             *   after all the nodes in the `end` set.
             *
             * This could represent the lifetime of a register, or a gap in
             * its liveness.
             *
             * For something to be definitely 'within' this span, it would
             * have to be after all nodes in the `begin` set and before all
             * nodes in the `end` set.
             */
            struct GraphExtent
            {
                std::set<int> begin;
                std::set<int> end;

                std::string toString() const;

                /**
                 * Returns true if `this` is entirely within `gap`.
                 */
                bool isWithin(KernelGraph const& kgraph, GraphExtent const& gap) const;
            };

            std::ostream& operator<<(std::ostream& stream, GraphExtent const& extent);

            /**
             * Represents the liveness of a given tag, including any gaps
             * where it would be acceptable for the registers to be
             * modified.
             */
            struct TagExtent
            {
                using CategoryKey = std::tuple<MemoryType, LayoutType, DataType, int>;

                int              baseTag = -1;
                std::set<int>    tags;
                MemoryType       memoryType = MemoryType::None;
                LayoutType       layoutType = LayoutType::None;
                DataType         dataType   = DataType::None;
                std::vector<int> sizes;
                GraphExtent      extent;

                std::vector<GraphExtent> gaps;

                CategoryKey typeKey() const;

                std::string toString() const;
                std::string orderInfo(KernelGraph const& kgraph) const;
                void        validate(KernelGraph const& kgraph) const;

                bool empty() const;

                /**
                 * `inner` must fit within one of `this`'s gaps. Modifies `gaps`
                 * to represent the new set of gaps if `inner` will borrow
                 * `this`'s registers.
                 */
                void merge(KernelGraph const& kgraph, TagExtent const& inner);

                /**
                 * Returns true if `this` fits within a gap within `outer`.
                 */
                bool fitsWithin(KernelGraph const& kgraph, TagExtent const& outer);
            };

            /**
             * Returns a graph describing the relative ordering of each of the
             * control nodes in `records`.
             */
            TagRWGraph getOrdering(KernelGraph const& kgraph, std::vector<Record> const& records);

            /**
             * Returns a TagExtent with the metadata (but not the extent or
             * gaps) filled in.
             */
            TagExtent getInfo(KernelGraph const& kgraph, std::vector<Record> const& records);

            /**
             * Returns a complete TagExtent representing the usage pattern
             * recorded in `records`.
             */
            TagExtent getExtent(KernelGraph const& kgraph, std::vector<Record> const& records);

            /**
             * Returns a set of aliases inner -> outer where `inner` can borrow
             * the registers of `outer` without causing a correctness problem
             * for the kernel.
             */
            std::map<int, int> findAliasCandidates(KernelGraph const& kgraph);

        }
    }
}
