// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/AddLDSBarriers.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSBarriers_detail.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace
        {
            using namespace ControlGraph;
            using namespace CoordinateGraph;
            using namespace AddLDSBarriersDetail;

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
                const auto ldsCoordinates = CollectAllLDSCoordinatesInRWTrace(graph, allRecords);

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
                        = CollectReadAndWritesToCoordinate(graph, recordsForCoord);

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
                                = GetIndexOfControlOpInAllRecords(writeTag, allRecords);
                            const auto readRecordIndex
                                = GetIndexOfControlOpInAllRecords(readTag, allRecords);

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
                                = FindCommonAncestorLoop(graph, firstOpTag, secondOpTag);

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
                            bool hasBarrierForForwardDependency = HasBarrierBetween(graph,
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
                                    = HasBarrierBetweenSecondAndFirstOpsInLoop(
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
            const auto ldsCoordinates = CollectAllLDSCoordinatesInRWTrace(graph, allRecords);
            if(ldsCoordinates.empty())
            {
                Log::debug("  No Read/Write to LDS found, skipping barrier insertion.");
                return graph;
            }

            // For each LDS coordinate, find dependent operations and ensure barriers exist
            for(int ldsCoord : ldsCoordinates)
            {
                const auto [readOpTags, writeOpTags] = CollectReadAndWritesToCoordinate(
                    graph, tracer.coordinatesReadWrite(ldsCoord));

                for(const auto& writeTag : writeOpTags)
                {
                    for(const auto& readTag : readOpTags)
                    {
                        const auto writeRecordIndex
                            = GetIndexOfControlOpInAllRecords(writeTag, allRecords);
                        const auto readRecordIndex
                            = GetIndexOfControlOpInAllRecords(readTag, allRecords);

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
                            = FindCommonAncestorLoop(graph, firstOpTag, secondOpTag);

                        // === Handle forward dependency ===
                        const auto existingBarrier = FindBarrierBetween(
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
                            firstOpIndex = GetIndexOfControlOpInAllRecords(firstOpTag, allRecords);
                            secondOpIndex
                                = GetIndexOfControlOpInAllRecords(secondOpTag, allRecords);
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
                                = FindBarrierForLoopCarried(graph,
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
