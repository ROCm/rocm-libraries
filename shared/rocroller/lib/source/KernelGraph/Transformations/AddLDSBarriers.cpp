/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include "rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp"
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSBarriers.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using RWTraceRecords = std::vector<ControlFlowRWTracer::ReadWriteRecord>;

        namespace
        {
            /**
             * @brief Check if a Barrier is connected to an LDS coordinate via mapper connections.
             *
             * @param graph The kernel graph
             * @param barrierTag Tag of the Barrier operation
             * @return true if the barrier is connected to an LDS coordinate
             */
            inline bool isBarrierForLDS(KernelGraph const& graph, int barrierTag)
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
            inline std::set<int> collectAllLDSCoordinatesInRWTrace(KernelGraph const&    graph,
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
            inline size_t getCrontrolOpIndexInAllRecords(int                   controlTag,
                                                         RWTraceRecords const& allRecords)
            {
                for(size_t i = 0; i < allRecords.size(); ++i)
                {
                    if(allRecords[i].control == controlTag)
                    {
                        return i;
                    }
                }
                AssertFatal(false, "Control tag not found in allRecords", ShowValue(controlTag));
                return 0;
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
                collectReadAndWritesToCoordinate(KernelGraph const&    graph,
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

                    if(record.rw == ControlFlowRWTracer::ReadWrite::WRITE
                       || record.rw == ControlFlowRWTracer::ReadWrite::READWRITE)
                    {
                        writes.push_back(record.control);
                    }
                    else if(record.rw == ControlFlowRWTracer::ReadWrite::READ)
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
            std::optional<int> findImmediateParentLoop(KernelGraph const& graph, int opTag)
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

                // Find the innermost loop: one that doesn't contain any other
                // loop from loops that contain opTag.
                for(auto i = 0; i < containingLoops.size(); ++i)
                {
                    const auto loopA = containingLoops[i];

                    bool isInnermost = true;
                    for(auto j = i + 1; j < containingLoops.size(); ++j)
                    {
                        const auto loopB = containingLoops[j];

                        auto nodesContainingOther
                            = graph.control.nodesContaining(loopB).to<std::set>();
                        if(nodesContainingOther.contains(loopA))
                        {
                            // If loopA contains loopB then loopA  is not the innermost
                            // loop that contains opTag
                            isInnermost = false;
                            break;
                        }
                    }

                    if(isInnermost)
                    {
                        return {loopA};
                    }
                }

                AssertFatal(false,
                            "Operation is contained by loop(s) but no immediate parent loop could "
                            "be found.",
                            ShowValue(opTag),
                            ShowValue(containingLoops));
                return std::nullopt;
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
            inline bool areInSameLoopBody(KernelGraph const& graph, int barrierTag, int opTag)
            {
                auto barrierLoop = findImmediateParentLoop(graph, barrierTag);
                auto opLoop      = findImmediateParentLoop(graph, opTag);

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
             *
             * @param graph The kernel graph
             * @param opA First operation tag
             * @param opB Second operation tag
             * @return The tag of the closest common ancestor loop, or std::nullopt if none exists
             */
            std::optional<int> findCommonAncestorLoop(KernelGraph const& graph, int opA, int opB)
            {
                // Get all nodes containing opA (ancestors)
                // const auto ancestorsA = graph.control.nodesContaining(opA).to<std::set>();
                const auto tagOfimmediateParentLoopOfA = findImmediateParentLoop(graph, opA);

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
            std::optional<int> findBarrierBetween(KernelGraph const&    graph,
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
                    if(graph.control.get<Barrier>(ctrl) and isBarrierForLDS(graph, ctrl)
                       and areInSameLoopBody(graph, ctrl, allRecords[secondOpRecordIndex].control)
                       and barrierLdsCoord == ldsCoord)
                    {
                        auto foundWritesAfterBarrier = std::any_of(
                            allRecords.begin() + i,
                            allRecords.begin() + endPos,
                            [ldsCoord, &graph](const ControlFlowRWTracer::ReadWriteRecord& record) {
                                return (record.rw == ControlFlowRWTracer::READWRITE
                                        or record.rw == ControlFlowRWTracer::WRITE)
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
            bool hasBarrierBetween(KernelGraph const&    graph,
                                   RWTraceRecords const& allRecords,
                                   int                   ldsCoord,
                                   int                   firstOpTag,
                                   int                   secondOpTag,
                                   size_t                firstOpRecordIndex,
                                   size_t                secondOpRecordIndex)
            {
                auto barrier = findBarrierBetween(
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
            std::optional<int> findBarrierForLoopCarried(KernelGraph const&    graph,
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
                    if(graph.control.get<Barrier>(ctrl) && isBarrierForLDS(graph, ctrl)
                       && barrierLdsCoord == ldsCoord)
                    {
                        auto foundWritesAfterBarrier = std::any_of(
                            allRecords.begin() + i,
                            allRecords.begin() + firstOpRecordIndex,
                            [ldsCoord, &graph](const ControlFlowRWTracer::ReadWriteRecord& record) {
                                return (record.rw == ControlFlowRWTracer::READWRITE
                                        or record.rw == ControlFlowRWTracer::WRITE)
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
                    if(graph.control.get<Barrier>(ctrl) && isBarrierForLDS(graph, ctrl)
                       && barrierLdsCoord == ldsCoord)
                    {
                        auto foundWritesAfterBarrier = std::any_of(
                            allRecords.begin() + i,
                            allRecords.end(),
                            [ldsCoord, &graph](const ControlFlowRWTracer::ReadWriteRecord& record) {
                                return (record.rw == ControlFlowRWTracer::READWRITE
                                        or record.rw == ControlFlowRWTracer::WRITE)
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
            bool hasBarrierBetweenSecondAndFirstOpsInLoop(KernelGraph const&    graph,
                                                          RWTraceRecords const& allRecords,
                                                          int                   ldsCoord,
                                                          int    commonAncestorLoopTag,
                                                          int    firstOpTag,
                                                          int    secondOpTag,
                                                          size_t firstOpRecordIndex,
                                                          size_t secondOpRecordIndex)
            {
                auto barrier = findBarrierForLoopCarried(graph,
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

            /**
             * @brief Verify that barriers exist between LDS writes and reads.
             *
             * This function traces through all control flow operations to identify LDS
             * read/write accesses, then verifies that appropriate barriers exist between
             * write and read operations to prevent data races.
             *
             * @param graph The kernel graph to verify
             * @return ConstraintStatus indicating success or containing error messages for
             *         missing barriers
             */
            ConstraintStatus VerifyLDSBarriers(KernelGraph const& graph)
            {
                TIMER(t, "Constraint::VerifyLDSBarriers");
                ConstraintStatus retval;

                ControlFlowRWTracer tracer(graph);

                // Get all trace records from the tracer
                auto allRecords = tracer.coordinatesReadWrite();

                // Collect all LDS coordinates that are accessed
                const auto ldsCoordinates = collectAllLDSCoordinatesInRWTrace(graph, allRecords);

                // This loop is here for debugging purposes: log all barriers found in the trace
                for(auto recordIndex = 0; recordIndex < allRecords.size(); ++recordIndex)
                {
                    auto const& record = allRecords[recordIndex];
                    if(graph.control.get<Barrier>(record.control))
                    {
                        Log::debug(fmt::format("TRACE: Barrier({}) at index {} for coordinate {}",
                                               record.control,
                                               recordIndex,
                                               record.coordinate));
                    }
                }

                auto getOpName = [&graph](int tag) {
                    return std::visit([](auto op) { return op.name(); },
                                      std::get<Operation>(graph.control.getElement(tag)));
                };

                // For each LDS coordinate, find dependent operations and check if barriers exist
                for(int ldsCoord : ldsCoordinates)
                {
                    auto recordsForCoord = tracer.coordinatesReadWrite(ldsCoord);

                    const auto [readOpTags, writeOpTags]
                        = collectReadAndWritesToCoordinate(graph, recordsForCoord);

                    for(const auto writeTag : writeOpTags)
                    {
                        for(const auto readTag : readOpTags)
                        {
                            Log::debug("Found {}({}) that writes LDS({}) and {}({}) that reads it.",
                                       getOpName(writeTag),
                                       writeTag,
                                       ldsCoord,
                                       getOpName(readTag),
                                       readTag);

                            const auto writeRecordIndex
                                = getCrontrolOpIndexInAllRecords(writeTag, allRecords);
                            const auto readRecordIndex
                                = getCrontrolOpIndexInAllRecords(readTag, allRecords);

                            // Determine which operation executes first and second based on
                            // their order in trace
                            const auto [firstOpTag, secondOpTag]
                                = (writeRecordIndex < readRecordIndex)
                                      ? std::make_pair(writeTag, readTag)
                                      : std::make_pair(readTag, writeTag);

                            const auto [firstOpIndex, secondOpIndex]
                                = (writeRecordIndex < readRecordIndex)
                                      ? std::make_pair(writeRecordIndex, readRecordIndex)
                                      : std::make_pair(readRecordIndex, writeRecordIndex);

                            // Find common ancestor loop (if any)
                            const auto commonAncestorLoop
                                = findCommonAncestorLoop(graph, firstOpTag, secondOpTag);

                            // Possible cases:
                            //   1. The operations have a common ancestor loop. This case
                            //      covers the scenarios where: (i) both operations are immediately
                            //      in the body of the same loop, (ii) each operation belongs to
                            //      a different loop and such loops are nested in the common
                            //      ancestor loop, (iii) one of the operations runs before or after
                            //      the inner loop containing the other operation. In all these cases
                            //      dependencies flow forward and can be loop-carried by the common
                            //      ancestor loop, thus two barriers are needed: one between firstOp
                            //      and secondOp (for forward dependency), and one either before
                            //      firstOp or after secondOp within the loop (for loop-carried
                            //      dependency from iteration N's secondOp to iteration N+1's firstOp).
                            //
                            //   2. The operations do not have a common ancestor loop. This case
                            //      covers the scenarios where: (i) both operations are in different
                            //      loops that are sequenced and the loops are not nested in another
                            //      ancestor loop, (ii) only one operation is in a loop, (iii) neither
                            //      operation is inside of a loop. In all these cases dependencies
                            //      only flow forward, thus a single barrier is needed between the
                            //      first and second operations.

                            // Check for barrier between firstOp & secondOp (forward dependency)
                            bool hasBarrierForForwardDependency = hasBarrierBetween(graph,
                                                                                    allRecords,
                                                                                    ldsCoord,
                                                                                    firstOpTag,
                                                                                    secondOpTag,
                                                                                    firstOpIndex,
                                                                                    secondOpIndex);

                            if(not hasBarrierForForwardDependency)
                            {
                                const auto message
                                    = concatenate("Missing LDS barrier between first operation ",
                                                  firstOpTag,
                                                  " (",
                                                  getOpName(firstOpTag),
                                                  ") and second operation ",
                                                  secondOpTag,
                                                  " (",
                                                  getOpName(secondOpTag),
                                                  ") for LDS coordinate ",
                                                  ldsCoord,
                                                  ".");
                                Log::debug(message);
                                retval.combine(false, message);
                            }

                            if(commonAncestorLoop.has_value())
                            {
                                // Check if there is a barrier that executes after secondOp and/or
                                // before firstOp (loop-carried dependency by common ancestor loop)
                                bool hasBarrierForLoopCarriedDependency
                                    = hasBarrierBetweenSecondAndFirstOpsInLoop(
                                        graph,
                                        allRecords,
                                        ldsCoord,
                                        commonAncestorLoop.value(),
                                        firstOpTag,
                                        secondOpTag,
                                        firstOpIndex,
                                        secondOpIndex);

                                if(not hasBarrierForLoopCarriedDependency)
                                {
                                    const auto message = concatenate(
                                        "Missing LDS barrier between second operation ",
                                        secondOpTag,
                                        " (",
                                        getOpName(secondOpTag),
                                        ") and first operation ",
                                        firstOpTag,
                                        " (",
                                        getOpName(firstOpTag),
                                        ") in loop ",
                                        commonAncestorLoop.value(),
                                        " for LDS coordinate ",
                                        ldsCoord,
                                        ". A barrier is required to handle "
                                        "loop-carried dependencies.");
                                    Log::debug(message);
                                    retval.combine(false, message);
                                }
                            }
                        }
                    }
                }

                return retval;
            }

        } // anonymous namespace

        KernelGraph AddLDSBarriers::apply(KernelGraph const& original)
        {
            TIMER(t, "AddLDSBarriers::apply");
            Log::debug("  AddLDSBarriers control graph transform.");

            auto graph = original;

            ControlFlowRWTracer tracer{graph};
            auto                allRecords = tracer.coordinatesReadWrite();

            // Collect all LDS coordinates that are accessed
            const auto ldsCoordinates = collectAllLDSCoordinatesInRWTrace(graph, allRecords);
            if(ldsCoordinates.empty())
            {
                Log::debug("  No Read/Write to LDS found, skipping barrier insertion.");
                return graph;
            }

            // For each LDS coordinate, find dependent operations and ensure barriers exist
            for(int ldsCoord : ldsCoordinates)
            {
                const auto [readOpTags, writeOpTags] = collectReadAndWritesToCoordinate(
                    graph, tracer.coordinatesReadWrite(ldsCoord));

                for(const auto& writeTag : writeOpTags)
                {
                    for(const auto& readTag : readOpTags)
                    {
                        const auto writeRecordIndex
                            = getCrontrolOpIndexInAllRecords(writeTag, allRecords);
                        const auto readRecordIndex
                            = getCrontrolOpIndexInAllRecords(readTag, allRecords);

                        // Determine which operation executes first and second
                        const auto [firstOpTag, secondOpTag]
                            = (writeRecordIndex < readRecordIndex)
                                  ? std::make_pair(writeTag, readTag)
                                  : std::make_pair(readTag, writeTag);

                        auto [firstOpIndex, secondOpIndex]
                            = (writeRecordIndex < readRecordIndex)
                                  ? std::make_pair(writeRecordIndex, readRecordIndex)
                                  : std::make_pair(readRecordIndex, writeRecordIndex);

                        // Find common ancestor loop (if any)
                        const auto commonAncestorLoop
                            = findCommonAncestorLoop(graph, firstOpTag, secondOpTag);

                        // === Handle forward dependency ===
                        const auto existingBarrier = findBarrierBetween(
                            graph, allRecords, ldsCoord, firstOpIndex, secondOpIndex);
                        if(not existingBarrier.has_value())
                        {
                            // Insert new barrier before secondOp
                            auto newBarrier = graph.control.addElement(Barrier());
                            // Either the op itself or its top containing SetCoordinate
                            const auto insertPosition = getTopSetCoordinate(graph, secondOpTag);
                            insertBefore(graph, insertPosition, newBarrier, newBarrier);
                            graph.mapper.connect<LDS>(newBarrier, ldsCoord);
                            auto it = std::find_if(
                                allRecords.begin(),
                                allRecords.end(),
                                [secondOpTag](const ControlFlowRWTracer::ReadWriteRecord& record) {
                                    return record.control == secondOpTag;
                                });
                            AssertFatal(it != allRecords.end(),
                                        "Could not find secondOpTag in allRecords.",
                                        ShowValue(secondOpTag));
                            allRecords.insert(it,
                                              ControlFlowRWTracer::ReadWriteRecord{
                                                  newBarrier, ldsCoord, ControlFlowRWTracer::READ});
                            firstOpIndex  = getCrontrolOpIndexInAllRecords(firstOpTag, allRecords);
                            secondOpIndex = getCrontrolOpIndexInAllRecords(secondOpTag, allRecords);
                            const auto message
                                = fmt::format("  Inserted new Barrier({}) before {} for forward "
                                              "dependency between {} & {} and LDS({})",
                                              newBarrier,
                                              insertPosition,
                                              firstOpTag,
                                              secondOpTag,
                                              ldsCoord);
                            Log::debug(message);
                        }
                        else
                        {
                            Log::debug(fmt::format(
                                "  Omitting insertion of new barrier for forward dependency "
                                "between {} and {} for LDS({}) since existing/previously inserted "
                                "barrier {} was found.",
                                firstOpTag,
                                secondOpTag,
                                ldsCoord,
                                existingBarrier.value()));
                        }

                        // === Handle loop-carried dependency ===
                        if(commonAncestorLoop.has_value())
                        {
                            auto existingBarrier
                                = findBarrierForLoopCarried(graph,
                                                            allRecords,
                                                            ldsCoord,
                                                            commonAncestorLoop.value(),
                                                            firstOpIndex,
                                                            secondOpIndex);

                            if(not existingBarrier.has_value())
                            {
                                // Insert new barrier before firstOp
                                auto newBarrier = graph.control.addElement(Barrier());
                                // Either the op itself or its top containing SetCoordinate
                                const auto insertPosition = getTopSetCoordinate(graph, firstOpTag);
                                insertBefore(graph, insertPosition, newBarrier, newBarrier);
                                graph.mapper.connect<LDS>(newBarrier, ldsCoord);
                                auto it = std::find_if(
                                    allRecords.begin(),
                                    allRecords.end(),
                                    [firstOpTag](
                                        const ControlFlowRWTracer::ReadWriteRecord& record) {
                                        return record.control == firstOpTag;
                                    });
                                AssertFatal(it != allRecords.end(),
                                            "Could not find firstOpTag in allRecords.",
                                            ShowValue(firstOpTag));
                                allRecords.insert(
                                    it,
                                    ControlFlowRWTracer::ReadWriteRecord{
                                        newBarrier, ldsCoord, ControlFlowRWTracer::READ});
                                const auto message = fmt::format(
                                    "  Inserted new Barrier({}) before {} for "
                                    "loop-carried dependency from {} to {} in loop {} for "
                                    "LDS({})",
                                    newBarrier,
                                    insertPosition,
                                    secondOpTag,
                                    firstOpTag,
                                    commonAncestorLoop.value(),
                                    ldsCoord);
                                Log::debug(message);
                            }
                            else
                            {
                                Log::debug(fmt::format("  Omitting insertion of new barrier for "
                                                       "loop-carried dependency "
                                                       "from {} to {} in loop {} for LDS({}) since "
                                                       "existing/previously inserted "
                                                       "barrier {} was found.",
                                                       secondOpTag,
                                                       firstOpTag,
                                                       commonAncestorLoop.value(),
                                                       ldsCoord,
                                                       existingBarrier.value()));
                            }
                        }
                    }
                }
            }

            return graph;
        }

        std::vector<GraphConstraint> AddLDSBarriers::postConstraints() const
        {
            return {&VerifyLDSBarriers};
        }
    }
}
