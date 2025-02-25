
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddConvert.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class AddConvertOperations
        {
        public:
            AddConvertOperations() {}

            void addConverts(KernelGraph& graph);

        private:
            struct ConvertLocation
            {
                int                                       storage;
                std::vector<std::pair<int, NaryArgument>> uses;
                int                                       parentNode;
                ControlEdge                               edgeType;
            };

            struct SharedParents
            {
                // Map from an operation to its parent and argument type
                std::map<int, std::pair<int, NaryArgument>> sharedParent;
                // Map from a parent to its edge type
                std::map<int, ControlEdge> edgeType;
            };

            void          stageMultiplyConverts(KernelGraph const& graph);
            void          commit(KernelGraph& graph);
            SharedParents findSharedParents(KernelGraph const& graph, int coord);

            std::map<int, std::vector<std::pair<int, NaryArgument>>> m_multiplyArgs;
            std::map<int, int>                                       m_loadMap;
            std::map<int, DataType>                                  m_storageDataType;
            std::vector<ConvertLocation>                             m_locations;
        };

        /**
         * @brief Find where converts should be added for a given storage tag.
         *
         * Returns the location of the node whose result should be converted, as well as the
         * edge type that the node is connected with.
         *
         * @param graph
         * @param coord
         * @return std::pair<int, ControlEdge>
         */
        AddConvertOperations::SharedParents
            AddConvertOperations::findSharedParents(KernelGraph const& graph, int coord)
        {
            SharedParents rv;

            for(auto [opTag, arg] : m_multiplyArgs[coord])
            {
                auto meetsCriteriaBefore = [&](int node) {
                    auto const& elem = graph.control.getElement(node);

                    if(isOperation<LoadTiled>(elem) || isOperation<LoadLDSTile>(elem)
                       || isOperation<SetCoordinate>(elem))
                    {
                        if(m_loadMap[node] == coord)
                        {
                            return true;
                        }
                    }
                    else if(isOperation<NOP>(elem))
                    {
                        // Sometimes Multiply could be connected to a NOP (Due to prefetching)
                        return true;
                    }

                    return false;
                };

                auto antecedents
                    = graph.control.nodesBefore(opTag).filter(meetsCriteriaBefore).to<std::set>();

                auto isForLoop = [&](int node) {
                    return isOperation<ForLoopOp>(graph.control.getElement(node));
                };

                auto parents
                    = graph.control.nodesContaining(opTag).filter(isForLoop).to<std::set>();

                std::vector<int> candidates{antecedents.begin(), antecedents.end()};
                candidates.insert(candidates.end(), parents.begin(), parents.end());

                auto nodeOrderCompare = [&](int nodeA, int nodeB) {
                    auto order = graph.control.compareNodes(nodeA, nodeB);
                    // Return true if nodeA should appear before nodeB.
                    return order == NodeOrdering::LeftFirst
                           || order == NodeOrdering::RightInBodyOfLeft;
                };

                std::sort(candidates.begin(), candidates.end(), nodeOrderCompare);

                if(candidates.size() >= 1)
                {
                    auto node              = candidates.back();
                    rv.sharedParent[opTag] = {node, arg};
                    if(antecedents.contains(node))
                    {
                        rv.edgeType[node] = Sequence();
                    }
                    else
                    {
                        rv.edgeType[node] = Body();
                    }
                    continue;
                }
            }

            return rv;
        }

        void AddConvertOperations::stageMultiplyConverts(KernelGraph const& graph)
        {
            // Populate data structures with information about multiplies
            // and loads
            auto root = graph.control.roots().only();
            AssertFatal(root.has_value());

            auto allNodes = graph.control.depthFirstVisit(*root).filter(
                graph.control.isElemType<Operation>());

            for(const auto node : allNodes)
            {
                auto visitor = rocRoller::overloaded{
                    [&](Multiply op) {
                        auto [macATag, macA] = graph.getDimension<MacroTile>(
                            node, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                        m_multiplyArgs[macATag].push_back({node, NaryArgument::LHS});

                        if(op.scaleA == Operations::ScaleMode::Separate)
                        {
                            auto [macATag, macA] = graph.getDimension<MacroTile>(
                                node,
                                Connections::typeArgument<MacroTile>(NaryArgument::LHS_SCALE));
                            m_multiplyArgs[macATag].push_back({node, NaryArgument::LHS_SCALE});
                        }

                        auto [macBTag, macB] = graph.getDimension<MacroTile>(
                            node, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
                        m_multiplyArgs[macBTag].push_back({node, NaryArgument::RHS});

                        if(op.scaleB == Operations::ScaleMode::Separate)
                        {
                            auto [macBTag, macB] = graph.getDimension<MacroTile>(
                                node,
                                Connections::typeArgument<MacroTile>(NaryArgument::RHS_SCALE));
                            m_multiplyArgs[macBTag].push_back({node, NaryArgument::RHS_SCALE});
                        }
                    },
                    [&](CIsAnyOf<LoadTiled, LoadLDSTile> auto op) {
                        m_loadMap[getTopSetCoordinate(graph, node)]
                            = graph.mapper.get<MacroTile>(node);
                        m_storageDataType[graph.mapper.get<MacroTile>(node)]
                            = getDataType(std::get<Operation>(graph.control.getElement(node)));
                    },
                    [&](auto op) {}};

                std::visit(visitor, graph.control.getNode(node));
            }

            for(auto& [storage, multiplies] : m_multiplyArgs)
            {
                auto unsegmented
                    = DataTypeInfo::Get(m_storageDataType[storage]).unsegmentedVariableType();
                if(!unsegmented)
                    continue;
                if(m_storageDataType[storage] == unsegmented->dataType)
                    continue;

                auto sharedParents = findSharedParents(graph, storage);
                for(auto [parent, edgeType] : sharedParents.edgeType)
                {
                    std::vector<std::pair<int, NaryArgument>> uses;
                    for(auto const& kv : sharedParents.sharedParent)
                    {
                        auto opTag              = kv.first;
                        auto [selectedTag, arg] = kv.second;
                        if(selectedTag == parent)
                            uses.push_back({opTag, arg});
                    }
                    m_locations.emplace_back(ConvertLocation{storage, uses, parent, edgeType});
                }
            }
        }

        void AddConvertOperations::commit(KernelGraph& graph)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            std::map<int, int> newStorageDuplicate;

            for(auto& location : m_locations)
            {
                auto edgeTypeIndex = location.edgeType.index();

                // Create new node for convert in control graph and storage in coordinate graph
                auto newStorage
                    = graph.coordinates.addElement(graph.coordinates.getElement(location.storage));
                auto dataFlow    = std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                    location.storage, Register::Type::Vector, DataType::None});
                auto unsegmented = DataTypeInfo::Get(m_storageDataType[location.storage])
                                       .unsegmentedVariableType()
                                       ->dataType;
                auto convertNode = graph.control.addElement(
                    Assign{Register::Type::Vector, Expression::convert(unsegmented, dataFlow)});
                graph.mapper.connect(convertNode, newStorage, NaryArgument::DEST);

                // Create Edge from shared node to convert node
                graph.control.addElement(location.edgeType, {location.parentNode}, {convertNode});

                // Delete edges from shared node to Multiply
                for(auto const& [node, arg] : location.uses)
                {
                    // Make sure the edge still exists in the graph before attempting to delete
                    auto parentNodes
                        = graph.control
                              .getInputNodeIndices(node,
                                                   [&edgeTypeIndex](ControlEdge const& a) {
                                                       return a.index() == edgeTypeIndex;
                                                   })
                              .to<std::set>();

                    if(parentNodes.count(location.parentNode))
                        graph.control.deleteElement(std::vector<int>{location.parentNode},
                                                    std::vector<int>{node},
                                                    [&edgeTypeIndex](ControlEdge const& a) {
                                                        return a.index() == edgeTypeIndex;
                                                    });

                    // Create edges from convert node to multiplies
                    graph.control.addElement(Sequence(), {convertNode}, {node});

                    // Change mappings for all multiplies
                    graph.mapper.disconnect(
                        node, location.storage, Connections::typeArgument<MacroTile>(arg));
                    graph.mapper.connect(
                        node, newStorage, Connections::typeArgument<MacroTile>(arg));
                }

                // Add newStorage to coordinate graph, keeping track of duplicates
                // If duplicate, find original.
                auto duplicateStorage = only(graph.coordinates.getOutputNodeIndices(
                    location.storage, CT::isEdge<Duplicate>));

                auto originalStorage = duplicateStorage ? *duplicateStorage : location.storage;

                // If original hasn't been seen, insert newStorage after original with DataFlow edge
                if(newStorageDuplicate.count(originalStorage) == 0)
                {
                    auto childEdges = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(
                        originalStorage);
                    auto loc = graph.coordinates.getLocation(originalStorage);

                    for(auto const& edge : loc.outgoing)
                    {
                        if(!CT::isEdge<DataFlow>(
                               std::get<Edge>(graph.coordinates.getElement(edge))))
                            continue;

                        auto             edgeLoc = graph.coordinates.getLocation(edge);
                        std::vector<int> newInputs;
                        for(auto const& input : edgeLoc.incoming)
                        {
                            if(input != originalStorage)
                                newInputs.push_back(input);
                            else
                                newInputs.push_back(newStorage);
                        }

                        graph.coordinates.addElement(DataFlow(), newInputs, edgeLoc.outgoing);
                        graph.coordinates.deleteElement(edge);
                    }
                    graph.coordinates.addElement(DataFlow(), {originalStorage}, {newStorage});
                    newStorageDuplicate[originalStorage] = newStorage;
                }
                // if original has been seen, add Duplicate edge from newStorage to saved node
                else
                {
                    graph.coordinates.addElement(
                        Duplicate(), {newStorage}, {newStorageDuplicate[originalStorage]});
                }
            }
        }

        void AddConvertOperations::addConverts(KernelGraph& graph)
        {
            stageMultiplyConverts(graph);
            commit(graph);
        }

        KernelGraph AddConvert::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::addConvert");

            auto graph = k;

            AddConvertOperations adder;

            adder.addConverts(graph);

            return graph;
        }
    }
}
