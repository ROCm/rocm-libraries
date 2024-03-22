#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddConvert.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

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
                // Map from an operation to it's parent and argument type
                std::map<int, std::pair<int, NaryArgument>> sharedParent;
                // Map from a parent to it's edge type
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
                int selectedTag = 0;

                auto parentNodes
                    = graph.control.getInputNodeIndices<Sequence>(opTag).to<std::vector>();

                for(auto const& parent : parentNodes)
                {
                    // Look for loads of coord
                    if(isOperation<LoadTiled>(graph.control.getElement(parent))
                       || isOperation<LoadLDSTile>(graph.control.getElement(parent))
                       || isOperation<SetCoordinate>(graph.control.getElement(parent)))
                    {
                        if(m_loadMap[parent] == coord)
                        {
                            selectedTag = parent;
                            break;
                        }
                    }
                    // Sometimes Multiply could be connected to a NOP (Due to prefetching)
                    else if(isOperation<NOP>(graph.control.getElement(parent)))
                    {
                        selectedTag = parent;
                    }
                }

                if(selectedTag)
                    rv.edgeType[selectedTag] = Sequence();
                else
                {
                    auto parentNodes
                        = graph.control.getInputNodeIndices<Body>(opTag).to<std::vector>();

                    for(auto const& parent : parentNodes)
                    {
                        // Sometimes Multiply could be connected to a ForLoop (Due to prefetching)
                        if(isOperation<ForLoopOp>(graph.control.getElement(parent)))
                        {
                            selectedTag = parent;
                        }
                    }

                    if(selectedTag)
                        rv.edgeType[selectedTag] = Body();
                }

                AssertFatal(selectedTag > 0, "No valid parent found.");

                rv.sharedParent[opTag] = {selectedTag, arg};
            }

            return rv;
        }

        void AddConvertOperations::stageMultiplyConverts(KernelGraph const& graph)
        {
            // Populate data structures with information about multiplies
            // and loads
            for(const auto node : graph.control.depthFirstVisit(*graph.control.roots().begin()))
            {
                if(isOperation<Multiply>(graph.control.getElement(node)))
                {
                    auto [macATag, macA] = graph.getDimension<MacroTile>(
                        node, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                    auto [macBTag, macB] = graph.getDimension<MacroTile>(
                        node, Connections::typeArgument<MacroTile>(NaryArgument::RHS));

                    m_multiplyArgs[macATag].push_back({node, NaryArgument::LHS});
                    m_multiplyArgs[macBTag].push_back({node, NaryArgument::RHS});
                }
                else if(isOperation<LoadTiled>(graph.control.getElement(node))
                        || isOperation<LoadLDSTile>(graph.control.getElement(node)))
                {
                    m_loadMap[getTopSetCoordinate(graph, node)] = graph.mapper.get<MacroTile>(node);
                    m_storageDataType[graph.mapper.get<MacroTile>(node)]
                        = getDataType(std::get<Operation>(graph.control.getElement(node)));
                }
            }

            for(auto& [storage, multiplies] : m_multiplyArgs)
            {
                if(m_storageDataType[storage] != DataType::Half)
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
            std::map<int, int> newStorageDuplicate;

            for(auto& location : m_locations)
            {
                auto edgeTypeIndex = location.edgeType.index();

                // Create new node for convert in control graph and storage in coordinate graph
                auto newStorage
                    = graph.coordinates.addElement(graph.coordinates.getElement(location.storage));
                auto dataFlow    = std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                    location.storage, Register::Type::Vector, DataType::None});
                auto convertNode = graph.control.addElement(Assign{
                    Register::Type::Vector, Expression::convert<DataType::Halfx2>(dataFlow)});
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
