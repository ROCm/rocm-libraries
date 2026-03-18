// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveDuplicates.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace LoadStoreSpecification
        {
            /**
             * @brief Load/store specification.
             *
             * This uniquely identifies an load/store operation.
             */
            struct Spec
            {
                /// Target coordinate (source for loads, destination
                /// for stores).  This is a storage node: User, LDS
                /// etc
                int targetCoord;

                /// Parent operation.  If the load is within a loop,
                /// this is the ForLoopOp.  If not, this is -1.
                int parentOp;

                /// Operation variant.  Differentiates between the
                /// operation types.
                int opType;

                /// Unroll colour: unroll coordinate to unroll value
                /// mapping.
                std::map<int, int> unrollColour;

                auto operator<=>(Spec const&) const = default;
            };

            std::string toString(Spec const& x)
            {
                std::stringstream os;
                os << "{";
                os << " target: " << x.targetCoord;
                os << ", parent: " << x.parentOp;
                os << ", type: " << x.opType;
                os << ", colour: [ ";
                for(auto [coord, value] : x.unrollColour)
                {
                    os << "(" << coord << ", " << value << "), ";
                }
                os << "] }";
                return os.str();
            }

            /**
             * @brief Return load/store specifier for the load/store operation.
             */
            Spec getLoadStoreSpecifier(KernelGraph const&     k,
                                       UnrollColouring const& colouring,
                                       int                    opTag)
            {
                auto [target, direction] = getOperationTarget(opTag, k);

                auto maybeForLoop = findContainingOperation<ControlGraph::ForLoopOp>(opTag, k);

                auto parentOp = -1;
                if(maybeForLoop)
                    parentOp = *maybeForLoop;

                auto unrollColour = std::map<int, int>{};
                if(maybeForLoop && colouring.operationColour.contains(opTag))
                {
                    unrollColour = colouring.operationColour.at(opTag);
                }

                int opType = k.control.getNode<ControlGraph::Operation>(opTag).index();

                return {target, parentOp, opType, unrollColour};
            }
        }

        namespace RD
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;
            bool loadForLDS(int loadTag, KernelGraph const& graph)
            {

                auto dst = graph.mapper.get<CT::MacroTile>(loadTag);
                dst = only(graph.coordinates.getOutputNodeIndices(dst, CT::isEdge<CT::Duplicate>))
                          .value_or(dst);

                bool hasLDS = false;
                for(auto output :
                    graph.coordinates.getInputNodeIndices(dst, CT::isEdge<CT::DataFlow>))
                {
                    if(graph.coordinates.get<CT::LDS>(output).has_value())
                        return true;
                }
                return false;
            }

            /**
             * @brief Return set of workgroup operations to consider
             * for deduplication.
             *
             * This looks for global loads and LDS stores that operate
             * on workgroup tiles.
             */
            std::unordered_set<int> getWGCandidates(KernelGraph const& graph)
            {
                std::unordered_set<int> candidates;
                for(auto op : graph.control.getNodes())
                {
                    auto isLoadTiled = graph.control.get<ControlGraph::LoadTiled>(op).has_value();
                    auto isStoreLDSTile
                        = graph.control.get<ControlGraph::StoreLDSTile>(op).has_value();
                    if(!(isLoadTiled || isStoreLDSTile))
                        continue;

                    // Only consider within the KLoop
                    auto maybeForLoop = findContainingOperation<ControlGraph::ForLoopOp>(op, graph);
                    if(!maybeForLoop)
                        continue;
                    auto forLoop = *graph.control.get<ControlGraph::ForLoopOp>(*maybeForLoop);
                    if(not forLoop.loopName.starts_with(rocRoller::KLOOP))
                        continue;

                    // If LoadTiled, don't consider when loading directly
                    if(isLoadTiled && !loadForLDS(op, graph))
                        continue;

                    // If we get this far, the operation is a
                    // "workgroup" operation.  This implies that
                    // jammed colours should be squashed.
                    candidates.insert(op);
                }
                return candidates;
            }

            /**
             * @brief Return set of wave operations to consider for
             * deduplication.
             *
             * This looks for LDS loads that operate on wave tiles.
             */
            std::unordered_set<int> getWaveCandidates(KernelGraph const& graph)
            {
                std::unordered_set<int> candidates;
                for(auto op : graph.control.getNodes())
                {
                    auto isLoadTiled = graph.control.get<ControlGraph::LoadTiled>(op).has_value();
                    auto isLoadLDSTile
                        = graph.control.get<ControlGraph::LoadLDSTile>(op).has_value();

                    if(!(isLoadTiled || isLoadLDSTile))
                        continue;

                    // Only consider within the KLoop
                    auto maybeForLoop = findContainingOperation<ControlGraph::ForLoopOp>(op, graph);
                    if(!maybeForLoop)
                        continue;
                    auto forLoop = *graph.control.get<ControlGraph::ForLoopOp>(*maybeForLoop);
                    if(not forLoop.loopName.starts_with(rocRoller::KLOOP))
                        continue;

                    // If LoadTiled, only consider when loading directly
                    if(isLoadTiled && loadForLDS(op, graph))
                        continue;

                    candidates.insert(op);
                }
                return candidates;
            }

            /**
             * @brief Return set of Unroll coordinates that are
             * neighbours of ForLoops.
             *
             * These are "jammed" coordinates.
             */
            std::unordered_set<int> jammedColours(KernelGraph const& graph)
            {
                std::unordered_set<int> rv;
                for(auto forLoopOpTag : graph.control.getNodes<ControlGraph::ForLoopOp>())
                {
                    auto forLoopOp = *graph.control.get<ControlGraph::ForLoopOp>(forLoopOpTag);

                    if(not forLoopOp.loopName.starts_with(rocRoller::KLOOP))
                    {
                        auto [forLoopCoord, _ignore] = getForLoopCoords(forLoopOpTag, graph);
                        auto maybeUnroll             = findUnrollNeighbour(graph, forLoopCoord);
                        if(maybeUnroll)
                            rv.insert(*maybeUnroll);
                    }
                }
                return rv;
            }

            /**
             * @brief Return mapping of "existing" operation to set of
             * "duplicate" operations (subset of `candidates`) that
             * should be removed from the control graph.
             *
             * Note that the "existing" operation is
             * NodeOrdering::LeftFirst of all members of the
             * "duplicate" set.
             *
             * An operation is a duplicate of another if:
             * - they are the same operations (LoadTiled, StoreLDSTile, LoadLDSTile)
             * - they have the same body-parent (typically ForLoopOp)
             * - they have the same target (User, LDS)
             * - they have the same unroll colouring
             *
             * If `squashJammedColours` is enabled: any colouring
             * associated with a JammedWaveTileNumber is squashed (set
             * to zero) when comparing unroll colouring.
             */
            std::map<int, std::unordered_set<int>>
                findDuplicates(KernelGraph const&                  graph,
                               UnrollColouring const&              colouring,
                               auto const&                         candidates,
                               bool                                squashJammedColours,
                               std::unordered_map<int, int> const& nodeOrders)
            {
                using namespace LoadStoreSpecification;

                std::map<int, std::unordered_set<int>> duplicates;
                std::map<Spec, int>                    opSpecs;

                // First pass, create specification to earliest-operation mapping
                for(auto op : candidates)
                {
                    auto spec = getLoadStoreSpecifier(graph, colouring, op);
                    if(opSpecs.contains(spec))
                    {
                        auto originalOp = opSpecs[spec];

                        if(nodeOrders.at(originalOp) < nodeOrders.at(op))
                            opSpecs[spec] = op;
                    }
                    else
                    {
                        opSpecs[spec] = op;
                    }
                }

                std::unordered_set<int> squashColours;
                if(squashJammedColours)
                    squashColours = jammedColours(graph);

                // Second pass, mark duplicates
                for(auto op : candidates)
                {
                    auto spec = getLoadStoreSpecifier(graph, colouring, op);

                    auto originalSpec = spec;
                    if(!squashColours.empty())
                    {
                        for(auto kv : originalSpec.unrollColour)
                        {
                            if(squashColours.contains(kv.first))
                                originalSpec.unrollColour[kv.first] = 0;
                        }
                    }

                    if(opSpecs.contains(originalSpec))
                    {
                        auto originalOp = opSpecs[originalSpec];
                        if(op != originalOp)
                        {
                            duplicates[originalOp].insert(op);
                            Log::debug("Marking as duplicate: {} (orignal {}) {}",
                                       op,
                                       originalOp,
                                       toString(spec));
                        }
                    }
                }

                return duplicates;
            }

            /**
             * @brief Remove duplicates from the graph.
             *
             * The operation (either the original operation, or its
             * top-most SetCoordinate) is replaced by a NOP.
             *
             * References to duplicate tiles are updated to the
             * non-duplicate tiles.
             */
            KernelGraph removeDuplicates(KernelGraph const& original, auto const& duplicates)
            {
                auto graph = original;

                GraphReindexer expressionReindexer;

                for(auto [originalOp, duplicateOps] : duplicates)
                {
                    auto topOriginalOp = getTopSetCoordinate(graph, originalOp);
                    for(auto duplicateOp : duplicateOps)
                    {
                        auto topDuplicateOp = getTopSetCoordinate(graph, duplicateOp);

                        // Replace top-duplicate with a NOP, then purge the top-duplicate.
                        //
                        // Recall that (see findDuplicates) any
                        // operation we replace with a NOP happens
                        // *after* the existing operation that it
                        // duplicates.
                        auto nop = graph.control.addElement(ControlGraph::NOP());
                        replaceWith(graph, topDuplicateOp, nop, false);
                        purgeNodeAndChildren(graph, topDuplicateOp);

                        Log::debug("RemoveDuplicates: NOPing {} (top of {}) because it duplicates "
                                   "{} (top of {}).  NOP is {}.",
                                   topDuplicateOp,
                                   duplicateOp,
                                   topOriginalOp,
                                   originalOp,
                                   nop);

                        // Mark for update: references to the
                        // duplicate tile need to be updated to the
                        // original tile.
                        auto duplicateTileTag = original.mapper.get<CT::MacroTile>(duplicateOp);
                        if(duplicateTileTag != -1)
                        {
                            auto originalTileTag = original.mapper.get<CT::MacroTile>(originalOp);
                            expressionReindexer.coordinates.emplace(duplicateTileTag,
                                                                    originalTileTag);
                        }
                    }
                }

                // Collapse reindexing chains in expressionReindexer.coordinates.
                // If there are mappings like A -> B and B -> C, ensure A maps directly to C.
                // This prevents intermediate mappings from leaving dangling references.
                for(auto& [tag, mappedTag] : expressionReindexer.coordinates)
                {
                    while(expressionReindexer.coordinates.contains(mappedTag) and mappedTag != tag)
                    {
                        auto nextTag = expressionReindexer.coordinates.at(mappedTag);
                        if(nextTag == mappedTag)
                            break;
                        mappedTag = nextTag;
                    }
                    // Update the mapping to point directly to the final target.
                    expressionReindexer.coordinates[tag] = mappedTag;
                }

                // Update tile references
                auto kernel = *graph.control.roots().begin();
                reindexExpressions(graph, kernel, expressionReindexer);

                // Update tile connections and remove old tiles
                for(auto [tag, mappedTag] : expressionReindexer.coordinates)
                {
                    auto connections = graph.mapper.getCoordinateConnections(tag);
                    graph.mapper.purgeMappingsTo(tag);
                    for(auto conn : connections)
                    {
                        conn.coordinate = mappedTag;
                        graph.mapper.connect(conn);
                    }

                    AssertFatal(mappedTag == tag
                                || graph.mapper.getCoordinateConnections(tag).empty());

                    if(graph.mapper.getCoordinateConnections(tag).empty())
                    {
                        for(auto const& child : graph.coordinates.getLocation(tag).incoming)
                            graph.coordinates.deleteElement(child);
                        for(auto const& child : graph.coordinates.getLocation(tag).outgoing)
                            graph.coordinates.deleteElement(child);
                        graph.coordinates.deleteElement(tag);
                    }
                }
                return graph;
            }
        }

        template <typename... OperationTypes>
        static void collectOrderedNodes(KernelGraph const&       graph,
                                        std::unordered_set<int>& visited,
                                        int                      tag,
                                        std::vector<int>&        result)
        {
            // cppcheck-suppress internalAstError
            auto traverse = [&]<typename EdgeType>() {
                for(auto child : graph.control.getOutputNodeIndices<EdgeType>(tag))
                {
                    if(not visited.contains(child))
                    {
                        visited.insert(child);
                        collectOrderedNodes<OperationTypes...>(graph, visited, child, result);
                    }
                }
            };
            traverse.template operator()<ControlGraph::Sequence>();
            traverse.template operator()<ControlGraph::ForLoopIncrement>();
            traverse.template operator()<ControlGraph::Else>();
            traverse.template operator()<ControlGraph::Body>();
            traverse.template operator()<ControlGraph::Initialize>();

            bool const isTargetOperation
                = (graph.control.get<OperationTypes>(tag).has_value() || ...);

            if(isTargetOperation)
            {
                result.push_back(tag);
            }
        }

        template <typename... OperationTypes>
        std::vector<int> getOrderedNodes(KernelGraph const& graph)
        {
            auto roots = graph.control.roots().to<std::vector>();

            std::vector<int> result;
            if(roots.empty())
                return result;

            std::unordered_set<int> visited;
            for(auto const& node : roots)
            {
                visited.insert(node);
                collectOrderedNodes<OperationTypes...>(graph, visited, node, result);
            }
            return result;
        }

        KernelGraph RemoveDuplicates::apply(KernelGraph const& original)
        {
            auto colouring = colourByUnrollValue(original);
            auto graph     = original;
            removeRedundantSequenceEdges(graph);

            auto                         orderedNodes = getOrderedNodes<ControlGraph::LoadTiled,
                                                ControlGraph::StoreLDSTile,
                                                ControlGraph::LoadLDSTile>(graph);
            std::unordered_map<int, int> nodeOrders;
            for(int i = 0; i < orderedNodes.size(); i++)
                nodeOrders.emplace(orderedNodes[i], i);

            // Two passes:
            //   1. Workgroup (LoadTiled and StoreLDSTile) and
            //   2. Wave (LoadLDSTile)
            //
            // We assume that these passes are independent of each
            // other.  In particular, the first pass doesn't change
            // the colouring etc that is used for the second pass.

            // LoadTile and StoreLDSTile operate on Workgroup tiles
            Log::debug("RemoveDuplicates Workgroup pass");
            {
                auto candidates = RD::getWGCandidates(original);
                auto duplicates
                    = RD::findDuplicates(original, colouring, candidates, true, nodeOrders);
                graph = RD::removeDuplicates(graph, duplicates);
            }

            // LoadLDSTile operation on Wave tiles
            Log::debug("RemoveDuplicates Wave pass");
            {
                auto candidates = RD::getWaveCandidates(original);
                auto duplicates
                    = RD::findDuplicates(original, colouring, candidates, false, nodeOrders);
                graph = RD::removeDuplicates(graph, duplicates);
            }

            removeRedundantNOPs(graph);
            return graph;
        }
    }
}
