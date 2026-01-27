// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace AddLDSBarriersDetail
    {
        using namespace KernelGraph::ControlGraph;
        using namespace KernelGraph::CoordinateGraph;
        using RWTracer       = KernelGraph::ControlFlowRWTracer;
        using RWTraceRecords = std::vector<KernelGraph::ControlFlowRWTracer::ReadWriteRecord>;
        using KernelGraph    = KernelGraph::KernelGraph;

        /**
          * @brief Check if a Barrier is connected to an LDS coordinate via mapper connections.
          *
          * @param graph The kernel graph
          * @param barrierTag Tag of the Barrier operation
          * @return true if the barrier is connected to an LDS coordinate
          */
        inline bool IsBarrierForLDS(KernelGraph const& graph, int barrierTag)
        {
            return graph.mapper.get<LDS>(barrierTag) != -1;
        }

        /**
          * @brief Collect all LDS coordinates that appear in the RW trace records.
          *
          * Iterates through all trace records and identifies coordinates that correspond
          * to LDS memory accesses.
          *
          * @param graph The kernel graph
          * @param allRecords All control flow RW tracer records
          * @return A set of coordinate tags for all LDS coordinates found in the trace
          */
        inline std::set<int> CollectAllLDSCoordinatesInRWTrace(KernelGraph const&    graph,
                                                               RWTraceRecords const& allRecords)
        {

            std::set<int> ldsCoordinates;
            for(auto recordIndex = 0; recordIndex < allRecords.size(); ++recordIndex)
            {
                auto const& record = allRecords[recordIndex];
                // Check if this coordinate is an LDS coordinate
                if(graph.coordinates.get<LDS>(record.coordinate))
                {
                    ldsCoordinates.insert(record.coordinate);
                }
            }
            return ldsCoordinates;
        }

        /**
          * @brief Find the index of a control operation in the trace records.
          *
          * Searches through the trace records to find the position of the specified
          * control operation. This position is used to determine execution order.
          *
          * @param controlTag The tag of the control operation to find
          * @param allRecords All control flow RW tracer records
          * @return The index of the control operation in allRecords
          * @throws AssertFatal if the control tag is not found in the records
          */
        inline size_t GetIndexOfControlOpInAllRecords(int                   controlTag,
                                                      RWTraceRecords const& allRecords)
        {
            std::optional<size_t> controlOpIndex;
            for(size_t i = 0; i < allRecords.size(); ++i)
            {
                if(allRecords[i].control == controlTag)
                {
                    controlOpIndex = i;
                    break;
                }
            }
            AssertFatal(
                controlOpIndex, "Control tag not found in allRecords", ShowValue(controlTag));
            return *controlOpIndex;
        };

        /**
          * @brief Collect all read and write operations for a specific coordinate.
          *
          * Iterates through the trace records for a coordinate and separates them into
          * read operations and write operations. Barrier nodes are excluded from the
          * results since they are used to synchronize other operations.
          *
          * @param graph The kernel graph
          * @param recordsForCoord Trace records filtered for a specific coordinate
          * @return A pair of vectors: first contains write operations,
          *         second contains read operations
          */
        inline std::pair<std::vector<int>, std::vector<int>>
            CollectReadAndWritesToCoordinate(KernelGraph const&    graph,
                                             RWTraceRecords const& recordsForCoord)
        {
            std::vector<int> reads;
            std::vector<int> writes;
            for(auto recordIndex = 0; recordIndex < recordsForCoord.size(); ++recordIndex)
            {
                auto const& record = recordsForCoord[recordIndex];

                if(graph.control.get<Barrier>(record.control))
                {
                    // Do not consider Barrier nodes are readers because the
                    // point is to determine if there are barriers in-between
                    // other readers/writers operations.
                    continue;
                }

                if(record.rw == RWTracer::ReadWrite::WRITE
                   || record.rw == RWTracer::ReadWrite::READWRITE)
                {
                    writes.push_back(record.control);
                }
                else if(record.rw == RWTracer::ReadWrite::READ)
                {
                    reads.push_back(record.control);
                }
            }

            return {writes, reads};
        }

        /**
          * @brief Find the immediate parent loop (ForLoopOp or DoWhileOp) containing an operation.
          *
          * Collects all loops containing the operation and returns the one that does not
          * contain any other ForLoopOp or DoWhileOp in its body (i.e., the deepest loop containing
          * the operation).
          *
          * @param graph The kernel graph
          * @param opTag Tag of the operation
          * @return The tag of the immediate parent loop of opTag, or std::nullopt if not in any loop
          */
        inline std::optional<int> FindImmediateParentLoop(KernelGraph const& graph, int opTag)
        {
            // Collect all loops containing the operation
            std::vector<int> containingLoops;
            for(const auto node : graph.control.nodesContaining(opTag))
            {
                if(graph.control.get<ForLoopOp>(node) || graph.control.get<DoWhileOp>(node))
                {
                    containingLoops.push_back(node);
                }
            }

            if(containingLoops.empty())
            {
                return std::nullopt;
            }

            auto isInnermostLoop = [&](int candidateLoop) {
                for(const auto otherLoop : containingLoops)
                {
                    if(otherLoop == candidateLoop)
                    {
                        continue;
                    }

                    // If candidateLoop is in the set of nodes containing otherLoop,
                    // then candidateLoop contains otherLoop, so it's not innermost.
                    auto nodesContainingOther
                        = graph.control.nodesContaining(otherLoop).to<std::set>();
                    if(nodesContainingOther.contains(candidateLoop))
                    {
                        return false;
                    }
                }
                return true;
            };

            auto it = std::find_if(containingLoops.begin(), containingLoops.end(), isInnermostLoop);

            AssertFatal(it != containingLoops.end(),
                        "Operation is contained by loop(s) but innermost loop could "
                        "be found.",
                        ShowValue(opTag),
                        ShowValue(containingLoops));

            return *it;
        }

        /**
          * @brief Check if a barrier and an operation are in the body of the same loop.
          *
          * This function determines if both the barrier and the operation are
          * immediately contained within the same loop body (ForLoopOp or DoWhileOp).
          * If neither is in any loop, they are considered to be in the same body
          * (the kernel body).
          *
          * @param graph The kernel graph
          * @param barrierTag Tag of the barrier operation
          * @param opTag Tag of the operation to check
          * @return true if both are in the body of the same loop (or both outside any loop)
          */
        inline bool AreInSameLoopBody(KernelGraph const& graph, int barrierTag, int opTag)
        {
            auto barrierLoop = FindImmediateParentLoop(graph, barrierTag);
            auto opLoop      = FindImmediateParentLoop(graph, opTag);

            // Both must be in the same loop (or both not in any loop)
            if(barrierLoop.has_value() && opLoop.has_value())
            {
                return *barrierLoop == *opLoop;
            }

            // If neither is in a loop, they're in the same kernel body
            return !barrierLoop.has_value() && !opLoop.has_value();
        }

        /**
          * @brief Find the closest common ancestor loop (ForLoopOp or DoWhileOp) for two operations.
          * If a common ancestor loop exists, then it is the immediate parent of the operation that
          * executes first (opA) and also contains the second operation (opB).
          *
          * @param graph The kernel graph
          * @param opA First operation tag
          * @param opB Second operation tag
          * @return The tag of the closest common ancestor loop, or std::nullopt if none exists
          */
        inline std::optional<int> FindCommonAncestorLoop(KernelGraph const& graph, int opA, int opB)
        {
            // Get all nodes containing opA (ancestors)
            const auto tagOfimmediateParentLoopOfA = FindImmediateParentLoop(graph, opA);

            if(not tagOfimmediateParentLoopOfA.has_value())
            {
                // opA is not in any loop, so no common ancestor loop exists
                return std::nullopt;
            }

            // Iterate through ancestors of opB to find a common loop ancestor
            for(const auto node : graph.control.nodesContaining(opB))
            {
                // Check if it's a loop operation and is also one of the
                // loops that contains opA and opB, then it is the
                // closest common ancestor loop.
                const auto isLoop
                    = graph.control.get<ForLoopOp>(node) || graph.control.get<DoWhileOp>(node);
                if(isLoop && node == tagOfimmediateParentLoopOfA.value())
                {
                    return {node};
                }
            }

            return {};
        }

        /**
          * @brief Find a LDS Barrier operation between firstOp and secondOp.
          *
          * A barrier is considered "between" if it executes after firstOp and before secondOp.
          *
          * @param graph The kernel graph
          * @param allRecords All control flow RW tracer records
          * @param firstOpRecordIndex Position in tracer order of the operation that executes first
          * @param secondOpRecordIndex Position in tracer order of the operation that executes second
          * @return The tag of a barrier between the operations, or std::nullopt if none exists
          */
        inline std::optional<int> FindBarrierBetween(KernelGraph const&    graph,
                                                     RWTraceRecords const& allRecords,
                                                     int                   ldsCoord,
                                                     size_t                firstOpRecordIndex,
                                                     size_t                secondOpRecordIndex)
        {
            const auto startPos = firstOpRecordIndex + 1;
            const auto endPos   = secondOpRecordIndex - 1;

            AssertFatal(startPos >= 0 && endPos < allRecords.size() && startPos <= endPos,
                        "Invalid positions for firstOp and secondOp in trace.",
                        ShowValue(startPos),
                        ShowValue(endPos),
                        ShowValue(firstOpRecordIndex),
                        ShowValue(secondOpRecordIndex));

            // Look for Barrier nodes between firstOp and secondOp in trace order
            for(auto i = startPos; i <= endPos; ++i)
            {
                int ctrl            = allRecords[i].control;
                int barrierLdsCoord = allRecords[i].coordinate;
                if(graph.control.get<Barrier>(ctrl) and IsBarrierForLDS(graph, ctrl)
                   and AreInSameLoopBody(graph, ctrl, allRecords[secondOpRecordIndex].control)
                   and barrierLdsCoord == ldsCoord)
                {
                    auto foundWritesAfterBarrier
                        = std::any_of(allRecords.begin() + i,
                                      allRecords.begin() + endPos,
                                      [ldsCoord, &graph](const RWTracer::ReadWriteRecord& record) {
                                          return (record.rw == RWTracer::READWRITE
                                                  or record.rw == RWTracer::WRITE)
                                                 and record.coordinate == ldsCoord
                                                 and not graph.control.get<Barrier>(record.control);
                                      });
                    // Found a Barrier connected to a LDS coordinate in same loop as secondOp
                    if(not foundWritesAfterBarrier)
                    {
                        return {ctrl};
                    }
                }
            }

            return std::nullopt;
        }

        /**
          * @brief Check if there is a LDS Barrier operation between firstOp and secondOp.
          *
          * A barrier is considered "between" if it executes after firstOp and before secondOp.
          *
          * @param graph The kernel graph
          * @param allRecords All control flow RW tracer records
          * @param firstOpTag Tag of the operation that executes first (used for debug logging)
          * @param secondOpTag Tag of the operation that executes second (used for debug logging)
          * @param firstOpRecordIndex Position in tracer order of the operation that executes first
          * @param secondOpRecordIndex Position in tracer order of the operation that executes second
          * @return true if a barrier exists between the operations
          */
        inline bool HasBarrierBetween(KernelGraph const&    graph,
                                      RWTraceRecords const& allRecords,
                                      int                   ldsCoord,
                                      int                   firstOpTag,
                                      int                   secondOpTag,
                                      size_t                firstOpRecordIndex,
                                      size_t                secondOpRecordIndex)
        {
            auto barrier = FindBarrierBetween(
                graph, allRecords, ldsCoord, firstOpRecordIndex, secondOpRecordIndex);
            if(barrier.has_value())
            {
                Log::debug(fmt::format("FORWARD: Found LDS Barrier({}) between index "
                                       "{} (tag: {}) and index {} (tag: {})",
                                       *barrier,
                                       firstOpRecordIndex,
                                       firstOpTag,
                                       secondOpRecordIndex,
                                       secondOpTag));
                return true;
            }
            return false;
        }

        /**
          * @brief Find a LDS barrier handling loop-carried dependencies.
          *
          * For loop-carried dependencies, we need a barrier that executes either:
          * - After secondOp (before the next iteration's firstOp), OR
          * - Before firstOp (after the previous iteration's secondOp)
          *
          * @param graph The kernel graph
          * @param allRecords All control flow RW tracer records
          * @param commonAncestorLoopTag The common ancestor loop tag
          * @param firstLoopInstructionIndex The index of the first instruction in the common ancestor loop
          * @param firstOpRecordIndex Position in tracer order of the operation that executes first
          * @param secondOpRecordIndex Position in tracer order of the operation that executes second
          * @return The tag of a barrier for loop-carried dependencies, or std::nullopt if none exists
          */
        inline std::optional<int> FindBarrierForLoopCarried(KernelGraph const&    graph,
                                                            RWTraceRecords const& allRecords,
                                                            int                   ldsCoord,
                                                            int    commonAncestorLoopTag,
                                                            size_t firstOpRecordIndex,
                                                            size_t secondOpRecordIndex)
        {
            const auto afterSecondOpPos = secondOpRecordIndex + 1;

            AssertFatal(afterSecondOpPos < allRecords.size(),
                        "Invalid position for secondOp in trace.",
                        ShowValue(secondOpRecordIndex));

            // For loop-carried dependencies, we need to check if there's a barrier
            // that breaks the dependency from iteration N's secondOp to iteration N+1's firstOp.
            //
            // This means we need a barrier either:
            // 1. After secondOp but still within the loop body (before loop end)
            // 2. Before firstOp but after the loop body start
            //
            // In trace order, the loop body executes once. A barrier anywhere
            // after secondOp or before firstOp (but within the loop) suffices.

            // Check for Barrier nodes before firstOp (in common loop body)
            for(size_t i = 0; i < firstOpRecordIndex; ++i)
            {
                int ctrl            = allRecords[i].control;
                int barrierLdsCoord = allRecords[i].coordinate;
                if(graph.control.get<Barrier>(ctrl) && IsBarrierForLDS(graph, ctrl)
                   && barrierLdsCoord == ldsCoord)
                {
                    auto foundWritesAfterBarrier
                        = std::any_of(allRecords.begin() + i,
                                      allRecords.begin() + firstOpRecordIndex,
                                      [ldsCoord, &graph](const RWTracer::ReadWriteRecord& record) {
                                          return (record.rw == RWTracer::READWRITE
                                                  or record.rw == RWTracer::WRITE)
                                                 and record.coordinate == ldsCoord
                                                 and not graph.control.get<Barrier>(record.control);
                                      });
                    // Verify the barrier is inside the common ancestor loop
                    auto containingNodes = graph.control.nodesContaining(ctrl).to<std::set>();
                    if(not foundWritesAfterBarrier
                       and containingNodes.contains(commonAncestorLoopTag))
                    {
                        return {ctrl};
                    }
                }
            }

            // Check for Barrier nodes after secondOp (in common loop body)
            for(auto i = afterSecondOpPos; i < allRecords.size(); ++i)
            {
                int ctrl            = allRecords[i].control;
                int barrierLdsCoord = allRecords[i].coordinate;
                if(graph.control.get<Barrier>(ctrl) && IsBarrierForLDS(graph, ctrl)
                   && barrierLdsCoord == ldsCoord)
                {
                    auto foundWritesAfterBarrier
                        = std::any_of(allRecords.begin() + i,
                                      allRecords.end(),
                                      [ldsCoord, &graph](const RWTracer::ReadWriteRecord& record) {
                                          return (record.rw == RWTracer::READWRITE
                                                  or record.rw == RWTracer::WRITE)
                                                 and record.coordinate == ldsCoord
                                                 and not graph.control.get<Barrier>(record.control);
                                      });
                    // Verify the barrier is inside the common ancestor loop
                    auto containingNodes = graph.control.nodesContaining(ctrl).to<std::set>();
                    if(not foundWritesAfterBarrier
                       and containingNodes.contains(commonAncestorLoopTag))
                    {
                        return {ctrl};
                    }
                }
            }

            return std::nullopt;
        }

        /**
          * @brief Check if there is a LDS barrier handling loop-carried dependencies.
          *
          * For loop-carried dependencies, we need a barrier that executes either:
          * - After secondOp (before the next iteration's firstOp), OR
          * - Before firstOp (after the previous iteration's secondOp)
          *
          * @param graph The kernel graph
          * @param allRecords All control flow RW tracer records
          * @param commonAncestorLoopTag The common ancestor loop tag
          * @param firstLoopInstructionIndex The index of the first instruction in the common ancestor loop
          * @param firstOpTag Tag of the operation that executes first (used for debug logging)
          * @param secondOpTag Tag of the operation that executes second (used for debug logging)
          * @param firstOpRecordIndex Position in tracer order of the operation that executes first
          * @param secondOpRecordIndex Position in tracer order of the operation that executes second
          * @return true if a barrier exists to handle loop-carried dependencies
          */
        inline bool HasBarrierBetweenSecondAndFirstOpsInLoop(KernelGraph const&    graph,
                                                             RWTraceRecords const& allRecords,
                                                             int                   ldsCoord,
                                                             int    commonAncestorLoopTag,
                                                             int    firstOpTag,
                                                             int    secondOpTag,
                                                             size_t firstOpRecordIndex,
                                                             size_t secondOpRecordIndex)
        {
            auto barrier = FindBarrierForLoopCarried(graph,
                                                     allRecords,
                                                     ldsCoord,
                                                     commonAncestorLoopTag,
                                                     firstOpRecordIndex,
                                                     secondOpRecordIndex);
            if(barrier.has_value())
            {
                Log::debug(fmt::format("LOOP-CARRIED: Found LDS Barrier({}) in loop {} "
                                       "between index {} (tag: {}) and index {} (tag: {})",
                                       *barrier,
                                       commonAncestorLoopTag,
                                       firstOpRecordIndex,
                                       firstOpTag,
                                       secondOpRecordIndex,
                                       secondOpTag));
                return true;
            }
            return false;
        }
    } // namespace AddLDSBarriersDetail
} // namespace rocRoller
