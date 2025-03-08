/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
                /// Storage coordinate (segmented) that needs to be converted
                int storageCoord;

                /// Load operations that write to storageCoord
                std::unordered_set<int> loadOps;

                /// Multiply operations that read from storageCoord
                std::vector<std::pair<int, NaryArgument>> uses;
            };

            void stageMultiplyConverts(KernelGraph const& graph);
            void commit(KernelGraph& graph);

            // Map from storage coordinate to vector of {Multiply operation tag, RHS/LHS} pairs.
            std::map<int, std::vector<std::pair<int, NaryArgument>>> m_multiplyArgs;
            // Map from storage coordinate to set of LoadTiled/LoadLDSTile operations
            std::map<int, std::unordered_set<int>> m_loadMap;
            // Map from storage coordinate to DataType
            std::map<int, DataType> m_storageDataType;

            std::vector<ConvertLocation> m_locations;
        };

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

                        auto [macBTag, macB] = graph.getDimension<MacroTile>(
                            node, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
                        m_multiplyArgs[macBTag].push_back({node, NaryArgument::RHS});
                    },
                    [&](CIsAnyOf<LoadTiled, LoadLDSTile> auto op) {
                        auto coord = graph.mapper.get<MacroTile>(node);
                        m_loadMap[coord].insert(node);
                        m_storageDataType[graph.mapper.get<MacroTile>(node)]
                            = getDataType(graph.control.getNode(node));
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

                m_locations.emplace_back(ConvertLocation{storage, m_loadMap[storage], multiplies});
            }
        }

        void AddConvertOperations::commit(KernelGraph& graph)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            std::map<int, int> newStorageDuplicate;

            for(auto& location : m_locations)
            {
                // Create new node for convert in control graph and storage in coordinate graph
                auto newStorage = graph.coordinates.addElement(
                    graph.coordinates.getElement(location.storageCoord));
                auto dataFlow    = std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                    location.storageCoord, Register::Type::Vector, DataType::None});
                auto unsegmented = DataTypeInfo::Get(m_storageDataType[location.storageCoord])
                                       .unsegmentedVariableType()
                                       ->dataType;

                for(auto loadOp : location.loadOps)
                {
                    auto convertNode = graph.control.addElement(Assign{
                        Register::Type::Vector, Expression::convert(unsegmented, dataFlow), 0});
                    graph.mapper.connect(convertNode, newStorage, NaryArgument::DEST);
                    insertAfter(graph, loadOp, convertNode, convertNode);
                }

                // Change mappings for all multiplies
                for(auto const& [node, arg] : location.uses)
                {
                    graph.mapper.disconnect(
                        node, location.storageCoord, Connections::typeArgument<MacroTile>(arg));
                    graph.mapper.connect(
                        node, newStorage, Connections::typeArgument<MacroTile>(arg));
                }

                // Add newStorage to coordinate graph, keeping track of duplicates
                // If duplicate, find original.
                auto duplicateStorage = only(graph.coordinates.getOutputNodeIndices(
                    location.storageCoord, CT::isEdge<Duplicate>));

                auto originalStorage = duplicateStorage ? *duplicateStorage : location.storageCoord;

                // If original hasn't been seen, insert newStorage after original with DataFlow edge
                if(newStorageDuplicate.count(originalStorage) == 0)
                {
                    auto childEdges = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(
                        originalStorage);
                    auto loc = graph.coordinates.getLocation(originalStorage);

                    for(auto const& edge : loc.outgoing)
                    {
                        if(!CT::isEdge<DataFlow>(graph.coordinates.getEdge(edge)))
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
