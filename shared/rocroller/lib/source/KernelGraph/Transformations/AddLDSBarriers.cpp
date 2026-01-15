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

            inline std::pair<std::vector<std::pair<int, size_t>>,
                             std::vector<std::pair<int, size_t>>>
                collectReadAndWritesToCoordinate(KernelGraph const&    graph,
                                                 RWTraceRecords const& recordsForCoord)
            {
                std::vector<std::pair<int, size_t>> reads;
                std::vector<std::pair<int, size_t>> writes;
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
                        writes.push_back({record.control, recordIndex});
                    }
                    else if(record.rw == ControlFlowRWTracer::ReadWrite::READ)
                    {
                        reads.push_back({record.control, recordIndex});
                    }
                }
                return {writes, reads};
            }

            /**
             * @brief Find the closest common ancestor loop (ForLoopOp or DoWhileOp) for two operations.
             *
             * @param graph The kernel graph
             * @param opA First operation tag
             * @param opB Second operation tag
             * @return The tag of the common ancestor loop, or std::nullopt if none exists
             */
            std::optional<int> findCommonAncestorLoop(KernelGraph const& graph, int opA, int opB)
            {
                // Get all nodes containing opA (ancestors)
                const auto ancestorsA = graph.control.nodesContaining(opA).to<std::set>();

                // Iterate through ancestors of opB to find a common loop ancestor
                for(const auto node : graph.control.nodesContaining(opB))
                {
                    if(ancestorsA.contains(node))
                    {
                        // Check if it's a loop operation
                        if(graph.control.get<ForLoopOp>(node) || graph.control.get<DoWhileOp>(node))
                        {
                            return {node};
                        }
                    }
                }

                return {};
            }

            /**
             * @brief Find a Barrier operation between firstOp and secondOp.
             *
             * A barrier is considered "between" if it executes after firstOp and before secondOp.
             *
             * @param graph The kernel graph
             * @param allRecords All control flow RW tracer records
             * @param firstOpRecordIndex Position in tracer order of the operation that executes first
             * @param secondOpRecordIndex Position in tracer order of the operation that executes second
             * @param requireLDSConnection If true, only return barriers connected to LDS
             * @return The tag of a barrier between the operations, or std::nullopt if none exists
             */
            std::optional<int> findBarrierBetween(KernelGraph const&    graph,
                                                  RWTraceRecords const& allRecords,
                                                  size_t                firstOpRecordIndex,
                                                  size_t                secondOpRecordIndex,
                                                  bool                  requireLDSConnection = true)
            {
                const auto startPos = firstOpRecordIndex + 1;
                const auto endPos   = secondOpRecordIndex - 1;

                AssertFatal(startPos >= 0 && endPos < allRecords.size(),
                            "Invalid positions for firstOp and secondOp in trace.",
                            ShowValue(firstOpRecordIndex),
                            ShowValue(secondOpRecordIndex));

                // Look for Barrier nodes between firstOp and secondOp in trace order
                std::set<int> checkedControls;
                for(auto i = startPos; i <= endPos; ++i)
                {
                    int ctrl = allRecords[i].control;
                    if(!checkedControls.contains(ctrl))
                    {
                        checkedControls.insert(ctrl);
                        // Check if this control node is a Barrier
                        if(graph.control.get<Barrier>(ctrl))
                        {
                            if(!requireLDSConnection || isBarrierForLDS(graph, ctrl))
                            {
                                return ctrl;
                            }
                        }
                    }
                }

                return std::nullopt;
            }

            /**
             * @brief Check if there is a Barrier operation between firstOp and secondOp.
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
                                   int                   firstOpTag,
                                   int                   secondOpTag,
                                   size_t                firstOpRecordIndex,
                                   size_t                secondOpRecordIndex)
            {
                auto barrier = findBarrierBetween(
                    graph, allRecords, firstOpRecordIndex, secondOpRecordIndex);
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
             * @brief Find a barrier handling loop-carried dependencies.
             *
             * For loop-carried dependencies, we need a barrier that executes either:
             * - After secondOp (before the next iteration's firstOp), OR
             * - Before firstOp (after the previous iteration's secondOp)
             *
             * @param graph The kernel graph
             * @param allRecords All control flow RW tracer records
             * @param commonAncestorLoopTag The common ancestor loop tag
             * @param firstOpRecordIndex Position in tracer order of the operation that executes first
             * @param secondOpRecordIndex Position in tracer order of the operation that executes second
             * @param requireLDSConnection If true, only return barriers connected to LDS
             * @return The tag of a barrier for loop-carried dependencies, or std::nullopt if none exists
             */
            std::optional<int> findBarrierForLoopCarried(KernelGraph const&    graph,
                                                         RWTraceRecords const& allRecords,
                                                         int    commonAncestorLoopTag,
                                                         size_t firstOpRecordIndex,
                                                         size_t secondOpRecordIndex,
                                                         bool   requireLDSConnection = true)
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
                    int ctrl = allRecords[i].control;
                    if(graph.control.get<Barrier>(ctrl))
                    {
                        if(!requireLDSConnection || isBarrierForLDS(graph, ctrl))
                        {
                            // Verify the barrier is inside the common ancestor loop
                            auto containingNodes
                                = graph.control.nodesContaining(ctrl).to<std::set>();
                            if(containingNodes.contains(commonAncestorLoopTag))
                            {
                                return ctrl;
                            }
                        }
                    }
                }

                // Check for Barrier nodes after secondOp (in common loop body)
                for(auto i = afterSecondOpPos; i < allRecords.size(); ++i)
                {
                    int ctrl = allRecords[i].control;
                    if(graph.control.get<Barrier>(ctrl))
                    {
                        if(!requireLDSConnection || isBarrierForLDS(graph, ctrl))
                        {
                            // Verify the barrier is inside the common ancestor loop
                            auto containingNodes
                                = graph.control.nodesContaining(ctrl).to<std::set>();
                            if(containingNodes.contains(commonAncestorLoopTag))
                            {
                                return ctrl;
                            }
                        }
                    }
                }

                return std::nullopt;
            }

            /**
             * @brief Check if there is a barrier handling loop-carried dependencies.
             *
             * For loop-carried dependencies, we need a barrier that executes either:
             * - After secondOp (before the next iteration's firstOp), OR
             * - Before firstOp (after the previous iteration's secondOp)
             *
             * @param graph The kernel graph
             * @param allRecords All control flow RW tracer records
             * @param commonAncestorLoopTag The common ancestor loop tag
             * @param firstOpTag Tag of the operation that executes first (used for debug logging)
             * @param secondOpTag Tag of the operation that executes second (used for debug logging)
             * @param firstOpRecordIndex Position in tracer order of the operation that executes first
             * @param secondOpRecordIndex Position in tracer order of the operation that executes second
             * @return true if a barrier exists to handle loop-carried dependencies
             */
            bool hasBarrierBetweenSecondAndFirstOpsInLoop(KernelGraph const&    graph,
                                                          RWTraceRecords const& allRecords,
                                                          int    commonAncestorLoopTag,
                                                          int    firstOpTag,
                                                          int    secondOpTag,
                                                          size_t firstOpRecordIndex,
                                                          size_t secondOpRecordIndex)
            {
                auto barrier = findBarrierForLoopCarried(graph,
                                                         allRecords,
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

                // For each LDS coordinate, find dependent operations and check if barriers exist
                for(int ldsCoord : ldsCoordinates)
                {
                    auto recordsForCoord = tracer.coordinatesReadWrite(ldsCoord);

                    const auto [readOpTagsAndRecordIndices, writeOpTagsAndRecordIndices]
                        = collectReadAndWritesToCoordinate(graph, recordsForCoord);

                    // Check each write-read pair
                    for(const auto [writeTag, writeRecordIndex] : writeOpTagsAndRecordIndices)
                    {
                        for(const auto [readTag, readRecordIndex] : readOpTagsAndRecordIndices)
                        {
                            auto getOpName = [graph](int tag) {
                                return std::visit(
                                    [](auto op) { return op.name(); },
                                    std::get<Operation>(graph.control.getElement(tag)));
                            };
                            Log::debug("Found {}({}) that writes LDS({}) and {}({}) that reads it.",
                                       getOpName(writeTag),
                                       writeTag,
                                       ldsCoord,
                                       getOpName(readTag),
                                       readTag);

                            // Determine which operation executes first and second based on
                            // their order in recordsForCoord (which preserves trace order)
                            const auto [firstOpTag, secondOpTag]
                                = (writeRecordIndex < readRecordIndex)
                                      ? std::make_pair(writeTag, readTag)
                                      : std::make_pair(readTag, writeTag);

                            const auto firstOpIndexInAllRecords
                                = getCrontrolOpIndexInAllRecords(firstOpTag, allRecords);
                            const auto secondOpIndexInAllRecords
                                = getCrontrolOpIndexInAllRecords(secondOpTag, allRecords);

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
                            bool hasBarrierForForwardDependency
                                = hasBarrierBetween(graph,
                                                    allRecords,
                                                    firstOpTag,
                                                    secondOpTag,
                                                    firstOpIndexInAllRecords,
                                                    secondOpIndexInAllRecords);

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
                                        commonAncestorLoop.value(),
                                        firstOpTag,
                                        secondOpTag,
                                        firstOpIndexInAllRecords,
                                        secondOpIndexInAllRecords);

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

            ControlFlowRWTracer tracer(graph);
            auto                allRecords = tracer.coordinatesReadWrite();

            // Collect all LDS coordinates that are accessed
            const auto ldsCoordinates = collectAllLDSCoordinatesInRWTrace(graph, allRecords);
            if(ldsCoordinates.empty())
            {
                Log::debug("  No Read/Write to LDS found, skipping barrier insertion.");
                return graph;
            }

            // Track barriers we've already connected to LDS to avoid duplicate connections
            std::set<int> barriersConnectedToLDS;

            // Collect all existing LDS-connected barriers
            for(const auto& record : allRecords)
            {
                if(graph.control.get<Barrier>(record.control)
                   && isBarrierForLDS(graph, record.control))
                {
                    barriersConnectedToLDS.insert(record.control);
                }
            }

            // Track inserted barriers to avoid inserting multiple barriers at the same location
            // Maps (firstOpTag, secondOpTag) -> barrierTag for forward dependencies
            std::map<std::pair<int, int>, int> forwardBarriers;
            // Maps (loopTag, secondOpTag) -> barrierTag for loop-carried dependencies
            std::map<std::pair<int, int>, int> loopCarriedBarriers;

            // For each LDS coordinate, find dependent operations and ensure barriers exist
            for(int ldsCoord : ldsCoordinates)
            {
                auto recordsForCoord = tracer.coordinatesReadWrite(ldsCoord);
                const auto [readOpTagsAndRecordIndices, writeOpTagsAndRecordIndices]
                    = collectReadAndWritesToCoordinate(graph, recordsForCoord);

                // Process each write-read pair
                for(const auto& [writeTag, writeRecordIndex] : writeOpTagsAndRecordIndices)
                {
                    for(const auto& [readTag, readRecordIndex] : readOpTagsAndRecordIndices)
                    {
                        // Determine which operation executes first and second
                        const auto [firstOpTag, secondOpTag]
                            = (writeRecordIndex < readRecordIndex)
                                  ? std::make_pair(writeTag, readTag)
                                  : std::make_pair(readTag, writeTag);

                        const auto firstOpIndexInAllRecords
                            = getCrontrolOpIndexInAllRecords(firstOpTag, allRecords);
                        const auto secondOpIndexInAllRecords
                            = getCrontrolOpIndexInAllRecords(secondOpTag, allRecords);

                        // Find common ancestor loop (if any)
                        const auto commonAncestorLoop
                            = findCommonAncestorLoop(graph, firstOpTag, secondOpTag);

                        // === Handle forward dependency ===
                        // First, check if there's already an LDS-connected barrier
                        auto existingLDSBarrier = findBarrierBetween(graph,
                                                                     allRecords,
                                                                     firstOpIndexInAllRecords,
                                                                     secondOpIndexInAllRecords,
                                                                     true);

                        if(!existingLDSBarrier.has_value())
                        {
                            // Check if we already inserted a barrier for this first-second ops pair
                            auto pairKey = std::make_pair(firstOpTag, secondOpTag);
                            if(not forwardBarriers.contains(pairKey))
                            {
                                // Check if there's a non-LDS barrier we can reuse
                                auto existingBarrier = findBarrierBetween(graph,
                                                                          allRecords,
                                                                          firstOpIndexInAllRecords,
                                                                          secondOpIndexInAllRecords,
                                                                          false);

                                if(existingBarrier.has_value())
                                {
                                    // Reuse existing barrier by connecting to LDS
                                    Log::debug(
                                        "  Reusing existing Barrier({}) for forward dependency "
                                        "between {} and {} for LDS({})",
                                        *existingBarrier,
                                        firstOpTag,
                                        secondOpTag,
                                        ldsCoord);
                                    graph.mapper.connect<LDS>(*existingBarrier, ldsCoord);
                                    barriersConnectedToLDS.insert(*existingBarrier);
                                    forwardBarriers[pairKey] = *existingBarrier;
                                }
                                else
                                {
                                    // Insert new barrier before secondOp
                                    auto newBarrier = graph.control.addElement(Barrier());
                                    // Either the op itself or its top containing SetCoordinate
                                    const auto insertPosition
                                        = getTopSetCoordinate(graph, secondOpTag);
                                    insertBefore(graph, insertPosition, newBarrier, newBarrier);
                                    graph.mapper.connect<LDS>(newBarrier, ldsCoord);
                                    barriersConnectedToLDS.insert(newBarrier);
                                    forwardBarriers[pairKey] = newBarrier;
                                    Log::debug("  Inserted new Barrier({}) before {} for forward "
                                               "dependency between {} & {} and LDS({})",
                                               newBarrier,
                                               insertPosition,
                                               firstOpTag,
                                               secondOpTag,
                                               ldsCoord);
                                }
                            }
                        }

                        // === Handle loop-carried dependency ===
                        if(commonAncestorLoop.has_value())
                        {
                            // Check if there's already an LDS-connected barrier for loop-carried
                            auto existingLDSLoopBarrier
                                = findBarrierForLoopCarried(graph,
                                                            allRecords,
                                                            commonAncestorLoop.value(),
                                                            firstOpIndexInAllRecords,
                                                            secondOpIndexInAllRecords,
                                                            true);

                            if(!existingLDSLoopBarrier.has_value())
                            {
                                // Check if we already inserted a barrier for this loop-firstOp pair
                                auto loopKey
                                    = std::make_pair(commonAncestorLoop.value(), secondOpTag);
                                if(not loopCarriedBarriers.contains(loopKey))
                                {
                                    // Check if there's a non-LDS barrier we can reuse
                                    auto existingLoopBarrier
                                        = findBarrierForLoopCarried(graph,
                                                                    allRecords,
                                                                    commonAncestorLoop.value(),
                                                                    firstOpIndexInAllRecords,
                                                                    secondOpIndexInAllRecords,
                                                                    false);

                                    if(existingLoopBarrier.has_value())
                                    {
                                        // Reuse existing barrier
                                        Log::debug("  Reusing existing Barrier({}) for "
                                                   "loop-carried dependency in loop {} for LDS({})",
                                                   *existingLoopBarrier,
                                                   commonAncestorLoop.value(),
                                                   ldsCoord);
                                        graph.mapper.connect<LDS>(*existingLoopBarrier, ldsCoord);
                                        barriersConnectedToLDS.insert(*existingLoopBarrier);
                                        loopCarriedBarriers[loopKey] = *existingLoopBarrier;
                                    }
                                    else
                                    {
                                        // Insert new barrier before firstOp
                                        auto newBarrier = graph.control.addElement(Barrier());
                                        // Either the op itself or its top containing SetCoordinate
                                        const auto insertPosition
                                            = getTopSetCoordinate(graph, firstOpTag);
                                        insertBefore(graph, insertPosition, newBarrier, newBarrier);
                                        graph.mapper.connect<LDS>(newBarrier, ldsCoord);
                                        barriersConnectedToLDS.insert(newBarrier);
                                        loopCarriedBarriers[loopKey] = newBarrier;
                                        Log::debug(
                                            "  Inserted new Barrier({}) before {} for "
                                            "loop-carried dependency from {} to {} in loop {} for "
                                            "LDS({})",
                                            newBarrier,
                                            insertPosition,
                                            secondOpTag,
                                            firstOpTag,
                                            commonAncestorLoop.value(),
                                            ldsCoord);
                                    }
                                }
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
