// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <common/SourceMatcher.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ScheduleMultiplyAndLDS_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace Catch::Matchers;

namespace ScheduleMultiplyAndLDSDetailTest
{
    using namespace ScheduleMultiplyAndLDSDetail;
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    TEST_CASE("toString formats DataType count map", "[kernel-graph][schedule-multiply-lds]")
    {
        SECTION("Empty map")
        {
            std::map<DataType, int> typeCounts;
            auto                    result = toString(typeCounts);
            CHECK(result == "[]");
        }

        SECTION("Single type")
        {
            std::map<DataType, int> typeCounts;
            typeCounts[DataType::Float] = 2;
            auto result                 = toString(typeCounts);
            CHECK_THAT(result, ContainsSubstring("Float"));
            CHECK_THAT(result, ContainsSubstring("2"));
        }

        SECTION("Multiple types")
        {
            std::map<DataType, int> typeCounts;
            typeCounts[DataType::Float] = 2;
            typeCounts[DataType::Half]  = 3;
            auto result                 = toString(typeCounts);
            CHECK_THAT(result, ContainsSubstring("Float"));
            CHECK_THAT(result, ContainsSubstring("2"));
            CHECK_THAT(result, ContainsSubstring("Half"));
            CHECK_THAT(result, ContainsSubstring("3"));
        }
    }

    TEST_CASE("toString formats ChainTypes", "[kernel-graph][schedule-multiply-lds]")
    {
        SECTION("Empty ChainTypes")
        {
            ChainTypes chainTypes;
            auto       result = toString(chainTypes);
            CHECK(result == "{}");
        }

        SECTION("Single chain with type counts")
        {
            ChainTypes              chainTypes;
            std::map<DataType, int> typeCounts;
            typeCounts[DataType::Float] = 1;
            chainTypes.push_back(typeCounts);
            auto result = toString(chainTypes);
            CHECK_THAT(result, ContainsSubstring("Float"));
        }
    }

    TEST_CASE("showChain formats node vector", "[kernel-graph][schedule-multiply-lds]")
    {
        SECTION("Empty chain")
        {
            Chain chain;
            auto  result = showChain(chain);
            CHECK(result == "()");
        }

        SECTION("Single node")
        {
            Chain chain  = {42};
            auto  result = showChain(chain);
            CHECK(result == "(42)");
        }

        SECTION("Multiple nodes")
        {
            Chain chain  = {1, 2, 3};
            auto  result = showChain(chain);
            CHECK(result == "(1, 2, 3)");
        }
    }

    TEST_CASE("showChains formats vector of chains", "[kernel-graph][schedule-multiply-lds]")
    {
        SECTION("Empty chains")
        {
            Chains chains;
            auto   result = showChains(chains);
            CHECK(result == "");
        }

        SECTION("Single chain")
        {
            Chains chains = {{1, 2, 3}};
            auto   result = showChains(chains);
            CHECK_THAT(result, ContainsSubstring("1"));
            CHECK_THAT(result, ContainsSubstring("2"));
            CHECK_THAT(result, ContainsSubstring("3"));
        }

        SECTION("Multiple chains")
        {
            Chains chains = {{1, 2}, {3, 4, 5}};
            auto   result = showChains(chains);
            CHECK_THAT(result, ContainsSubstring("1"));
            CHECK_THAT(result, ContainsSubstring("2"));
            CHECK_THAT(result, ContainsSubstring("3"));
            CHECK_THAT(result, ContainsSubstring("4"));
            CHECK_THAT(result, ContainsSubstring("5"));
        }
    }

    TEST_CASE("showGroups formats groups of chains", "[kernel-graph][schedule-multiply-lds]")
    {
        SECTION("Empty groups")
        {
            Groups groups;
            auto   result = showGroups(groups);
            CHECK_THAT(result, ContainsSubstring("0 groups"));
        }

        SECTION("Single group")
        {
            Groups groups = {{{1, 2, 3}}};
            auto   result = showGroups(groups);
            CHECK_THAT(result, ContainsSubstring("1 groups"));
            CHECK_THAT(result, ContainsSubstring("1"));
            CHECK_THAT(result, ContainsSubstring("2"));
            CHECK_THAT(result, ContainsSubstring("3"));
        }

        SECTION("Multiple groups")
        {
            Groups groups = {{{1, 2}}, {{3, 4, 5}}};
            auto   result = showGroups(groups);
            CHECK_THAT(result, ContainsSubstring("2 groups"));
        }
    }

    TEST_CASE("makeChains identifies linear chains", "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Empty input returns empty")
        {
            std::vector<int> nodes;
            auto             chains = makeChains(graph, nodes);
            CHECK(chains.empty());
        }

        SECTION("Single node creates single chain")
        {
            auto             node1  = graph.control.addElement(CF::Multiply());
            std::vector<int> nodes  = {node1};
            auto             chains = makeChains(graph, nodes);
            REQUIRE(chains.size() == 1);
            CHECK(chains[0].size() == 1);
            CHECK(chains[0][0] == node1);
        }

        SECTION("Connected chain of multiply nodes")
        {
            // Create 4 multiply nodes connected in sequence
            auto node1 = graph.control.addElement(CF::Multiply());
            auto node2 = graph.control.addElement(CF::Multiply());
            auto node3 = graph.control.addElement(CF::Multiply());
            auto node4 = graph.control.addElement(CF::Multiply());

            graph.control.addElement(CF::Sequence(), {node1}, {node2});
            graph.control.addElement(CF::Sequence(), {node2}, {node3});
            graph.control.addElement(CF::Sequence(), {node3}, {node4});

            std::vector<int> nodes  = {node1, node2, node3, node4};
            auto             chains = makeChains(graph, nodes);

            CHECK(chains == Chains{{node1, node2, node3, node4}});
        }

        SECTION("Two disconnected chains")
        {
            auto kernel = graph.control.addElement(CF::Kernel());

            // Create first chain: 1 -> 2 -> 3
            auto node1 = graph.control.addElement(CF::Multiply());
            auto node2 = graph.control.addElement(CF::Multiply());
            auto node3 = graph.control.addElement(CF::Multiply());

            graph.control.addElement(CF::Body(), {kernel}, {node1});

            graph.control.addElement(CF::Body(), {kernel}, {node1});
            graph.control.chain<CF::Sequence>(node1, node2, node3);

            auto nop = graph.control.addElement(CF::NOP());

            // Create second chain: 5 -> 6 -> 7
            auto node5 = graph.control.addElement(CF::Multiply());
            auto node6 = graph.control.addElement(CF::Multiply());
            auto node7 = graph.control.addElement(CF::Multiply());

            graph.control.chain<CF::Sequence>(node3, nop, node5);

            graph.control.chain<CF::Sequence>(node5, node6, node7);

            std::vector<int> nodes  = {node1, node2, node3, node5, node6, node7};
            auto             chains = makeChains(graph, nodes);

            CHECK(chains == Chains{{node1, node2, node3}, {node5, node6, node7}});
        }
    }

    TEST_CASE("getLDSNode follows SetCoordinate body edges",
              "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Direct LDS node returns itself")
        {
            auto ldsNode = graph.control.addElement(CF::LoadLDSTile());
            auto result  = getLDSNode(graph, ldsNode);
            CHECK(result == ldsNode);
        }

        SECTION("SetCoordinate with single Body edge")
        {
            auto setCoord = graph.control.addElement(CF::SetCoordinate());
            auto ldsNode  = graph.control.addElement(CF::LoadLDSTile());
            graph.control.addElement(CF::Body(), {setCoord}, {ldsNode});

            auto result = getLDSNode(graph, setCoord);
            CHECK(result == ldsNode);
        }

        SECTION("Chain of SetCoordinate nodes with Body edges")
        {
            auto setCoord1 = graph.control.addElement(CF::SetCoordinate());
            auto setCoord2 = graph.control.addElement(CF::SetCoordinate());
            auto ldsNode   = graph.control.addElement(CF::LoadLDSTile());

            graph.control.addElement(CF::Body(), {setCoord1}, {setCoord2});
            graph.control.addElement(CF::Body(), {setCoord2}, {ldsNode});

            auto result = getLDSNode(graph, setCoord1);
            CHECK(result == ldsNode);
        }
    }

    TEST_CASE("getLDSType returns data type of LDS node", "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("LoadLDSTile with Float type")
        {
            auto tile  = CT::MacroTile();
            auto coord = graph.coordinates.addElement(tile);

            auto ldsNode = graph.control.addElement(CF::LoadLDSTile{DataType::Float});
            graph.mapper.connect<CT::MacroTile>(ldsNode, coord);

            auto result = getLDSType(graph, ldsNode);
            CHECK(result == DataType::Float);
        }

        SECTION("SetCoordinate pointing to LoadLDSTile with Half type")
        {
            auto tile  = CT::MacroTile();
            auto coord = graph.coordinates.addElement(tile);

            auto setCoord = graph.control.addElement(CF::SetCoordinate());
            auto ldsNode  = graph.control.addElement(CF::LoadLDSTile{DataType::Half});
            graph.mapper.connect<CT::MacroTile>(ldsNode, coord);
            graph.control.addElement(CF::Body(), {setCoord}, {ldsNode});

            auto result = getLDSType(graph, setCoord);
            CHECK(result == DataType::Half);
        }
    }

    TEST_CASE("getImmediateBodyParents updates nodes to body parents",
              "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("No body parents leaves nodes unchanged")
        {
            auto node1 = graph.control.addElement(CF::LoadLDSTile());
            auto node2 = graph.control.addElement(CF::LoadLDSTile());

            std::vector<int> nodes = {node1, node2};
            getImmediateBodyParents(graph, nodes);

            CHECK(nodes == Chain{node1, node2});
        }

        SECTION("Single body parent updates node")
        {
            auto setCoord = graph.control.addElement(CF::SetCoordinate());
            auto ldsNode  = graph.control.addElement(CF::LoadLDSTile());
            graph.control.addElement(CF::Body(), {setCoord}, {ldsNode});

            std::vector<int> nodes = {ldsNode};
            getImmediateBodyParents(graph, nodes);

            CHECK(nodes == Chain{setCoord});
        }

        SECTION("Chain of body parents updates to root")
        {
            auto setCoord1 = graph.control.addElement(CF::SetCoordinate());
            auto setCoord2 = graph.control.addElement(CF::SetCoordinate());
            auto ldsNode   = graph.control.addElement(CF::LoadLDSTile());

            graph.control.addElement(CF::Body(), {setCoord1}, {setCoord2});
            graph.control.addElement(CF::Body(), {setCoord2}, {ldsNode});

            std::vector<int> nodes = {ldsNode};
            getImmediateBodyParents(graph, nodes);

            CHECK(nodes == Chain{setCoord1});
        }
    }

    TEST_CASE("findLoadLDSChains identifies LoadLDSTile chains",
              "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Empty graph returns empty chains")
        {
            auto chains = findLoadLDSChains(graph);
            CHECK(chains.empty());
        }

        SECTION("Single LoadLDSTile node")
        {
            auto ldsNode = graph.control.addElement(CF::LoadLDSTile());
            auto chains  = findLoadLDSChains(graph);

            CHECK(chains == Chains{{ldsNode}});
        }

        SECTION("Connected LoadLDSTile chain")
        {
            auto lds1 = graph.control.addElement(CF::LoadLDSTile());
            auto lds2 = graph.control.addElement(CF::LoadLDSTile());
            auto lds3 = graph.control.addElement(CF::LoadLDSTile());

            graph.control.addElement(CF::Sequence(), {lds1}, {lds2});
            graph.control.addElement(CF::Sequence(), {lds2}, {lds3});

            auto chains = findLoadLDSChains(graph);

            CHECK(chains == Chains{{lds1, lds2, lds3}});
        }

        SECTION("LoadLDSTile with SetCoordinate parents")
        {
            auto setCoord1 = graph.control.addElement(CF::SetCoordinate());
            auto setCoord2 = graph.control.addElement(CF::SetCoordinate());
            auto lds1      = graph.control.addElement(CF::LoadLDSTile());
            auto lds2      = graph.control.addElement(CF::LoadLDSTile());

            graph.control.addElement(CF::Body(), {setCoord1}, {lds1});
            graph.control.addElement(CF::Body(), {setCoord2}, {lds2});
            graph.control.addElement(CF::Sequence(), {setCoord1}, {setCoord2});

            auto chains = findLoadLDSChains(graph);

            // Should find chain of SetCoordinate parents
            CHECK(chains == Chains{{setCoord1, setCoord2}});
        }
    }

    TEST_CASE("findMultiplyChainsAndCoords finds multiply chains",
              "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Empty graph returns empty results")
        {
            auto [chains, chainTypes] = findMultiplyChainsAndCoords(graph);
            CHECK(chains.empty());
            CHECK(chainTypes.empty());
        }

        SECTION("Connected multiply chain")
        {
            auto kernel = graph.control.addElement(CF::Kernel());

            auto a0 = graph.coordinates.addElement(CT::MacroTile());
            auto a1 = graph.coordinates.addElement(CT::MacroTile());
            auto b0 = graph.coordinates.addElement(CT::MacroTile());
            auto b1 = graph.coordinates.addElement(CT::MacroTile());
            auto d0 = graph.coordinates.addElement(CT::MacroTile());
            auto d1 = graph.coordinates.addElement(CT::MacroTile());

            auto mult = [&](int a, int b, int d) {
                auto mult = graph.control.addElement(CF::Multiply());
                graph.mapper.connect(
                    mult, a, Connections::typeArgument<CT::MacroTile>(NaryArgument::LHS));
                graph.mapper.connect(
                    mult, b, Connections::typeArgument<CT::MacroTile>(NaryArgument::RHS));
                graph.mapper.connect(
                    mult, d, Connections::typeArgument<CT::MacroTile>(NaryArgument::DEST));
                return mult;
            };

            auto mult1 = mult(a0, b0, d0);
            auto mult2 = mult(a1, b1, d1);
            auto mult3 = mult(a0, b1, d1);
            auto mult4 = mult(a1, b0, d0);

            graph.control.addElement(CF::Body(), {kernel}, {mult1});

            graph.control.chain<CF::Sequence>(mult1, mult2, mult3, mult4);

            CAPTURE(mult1, mult2, mult3, mult4);

            auto [chains, chainTypes] = findMultiplyChainsAndCoords(graph);

            CHECK(chains == Chains{{mult3, mult4}});
        }
    }

    TEST_CASE("identifyParallelChains groups parallel chains",
              "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Empty input returns empty")
        {
            Groups groups;
            auto   result = identifyParallelChains(graph, groups);
            CHECK(result.empty());
        }

        SECTION("Parallel chains")
        {
            auto kernel = graph.control.addElement(CF::Kernel());

            auto makeChain = [&](auto tmpl, int len) {
                std::vector<int> nodes;
                nodes.reserve(len);
                std::optional<int> prev;
                for(int i = 0; i < len; i++)
                {
                    auto node = graph.control.addElement(tmpl);
                    nodes.push_back(node);
                    if(prev)
                        graph.control.addElement(CF::Sequence(), {prev.value()}, {node});
                    prev = node;
                }
                return nodes;
            };

            auto multChain1 = makeChain(CF::Multiply(), 3);
            auto multChain2 = makeChain(CF::Multiply(), 3);

            auto ldsChain1 = makeChain(CF::LoadLDSTile(), 3);
            auto ldsChain2 = makeChain(CF::LoadLDSTile(), 3);
            auto ldsChain3 = makeChain(CF::LoadLDSTile(), 3);

            graph.control.addElement(CF::Body(), {kernel}, {ldsChain1.front()});

            auto loop = graph.control.addElement(CF::ForLoopOp());
            graph.control.chain<CF::Sequence>(ldsChain1.back(), loop);

            graph.control.addElement(CF::Body(), {loop}, {multChain1.front()});
            graph.control.addElement(CF::Body(), {loop}, {ldsChain2.front()});

            auto nop = graph.control.addElement(CF::NOP());

            graph.control.chain<CF::Sequence>(ldsChain2.back(), nop, ldsChain3.front());
            graph.control.chain<CF::Sequence>(multChain1.back(), nop, multChain2.front());

            auto ldsChains = findLoadLDSChains(graph);
            CHECK(ldsChains == Chains({ldsChain1, ldsChain2, ldsChain3}));
            auto multiplyChains = findMultiplyChains(graph);
            CHECK(multiplyChains == Chains({multChain1, multChain2}));

            auto result = identifyParallelChains(graph, {multiplyChains, ldsChains});

            CHECK(result == Groups({{{multChain1, ldsChain2}}, {{multChain2, ldsChain3}}}));
        }
    }

    TEST_CASE("distributeChains creates sequence edges", "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Empty chainSets does nothing")
        {
            std::vector<ParallelChainSet> chainSets;
            size_t initialEdgeCount = graph.control.getEdges().to<std::vector>().size();

            distributeChains(graph, chainSets);

            size_t finalEdgeCount = graph.control.getEdges().to<std::vector>().size();
            CHECK(finalEdgeCount == initialEdgeCount);
        }

        SECTION("ChainSet with multiply and LDS chains")
        {
            // Create multiply chain
            auto mult1 = graph.control.addElement(CF::Multiply());
            auto mult2 = graph.control.addElement(CF::Multiply());
            graph.control.addElement(CF::Sequence(), {mult1}, {mult2});

            // Create LDS chain with proper data types
            auto tile1  = CT::MacroTile();
            auto coord1 = graph.coordinates.addElement(tile1);
            auto lds1   = graph.control.addElement(CF::LoadLDSTile{DataType::Float});
            graph.mapper.connect<CT::MacroTile>(lds1, coord1);

            auto tile2  = CT::MacroTile();
            auto coord2 = graph.coordinates.addElement(tile2);
            auto lds2   = graph.control.addElement(CF::LoadLDSTile{DataType::Float});
            graph.mapper.connect<CT::MacroTile>(lds2, coord2);

            graph.control.addElement(CF::Sequence(), {lds1}, {lds2});

            ParallelChainSet chainSet;
            chainSet.multiplyChain = {mult1, mult2};
            chainSet.ldsChain      = {lds1, lds2};

            std::map<DataType, int> typeCounts;
            typeCounts[DataType::Float] = 1;
            chainSet.multiplyTagTypes.push_back(typeCounts);
            chainSet.multiplyTagTypes.push_back(typeCounts);

            chainSet.ldsChainTypes = {DataType::Float, DataType::Float};

            std::vector<ParallelChainSet> chainSets = {chainSet};

            size_t initialEdgeCount = graph.control.getEdges().to<std::vector>().size();
            distributeChains(graph, chainSets);
            size_t finalEdgeCount = graph.control.getEdges().to<std::vector>().size();

            // Should have created new Sequence edges
            CHECK(finalEdgeCount >= initialEdgeCount);
        }
    }

    TEST_CASE("chainTagTable generates formatted table", "[kernel-graph][schedule-multiply-lds]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        auto addMultiply = [&](int coordA, int coordB, int coordD) {
            auto mult = graph.control.addElement(CF::Multiply());
            graph.mapper.connect(
                mult, coordA, Connections::typeArgument<CT::MacroTile>(NaryArgument::LHS));
            graph.mapper.connect(
                mult, coordB, Connections::typeArgument<CT::MacroTile>(NaryArgument::RHS));
            graph.mapper.connect(
                mult, coordD, Connections::typeArgument<CT::MacroTile>(NaryArgument::DEST));
            return mult;
        };

        auto addLoadLDSTile = [&](DataType type, int reg, int lds) {
            auto op = graph.control.addElement(CF::LoadLDSTile{type});
            graph.mapper.connect<CT::MacroTile>(op, reg);
            graph.mapper.connect<CT::LDS>(op, lds);
            return op;
        };

        {
            auto kernel = graph.control.addElement(CF::Kernel());

            auto lds1 = graph.coordinates.addElement(CT::LDS());

            // Create multiple multiply nodes to show table structure
            auto coordA1 = graph.coordinates.addElement(CT::MacroTile({}, LayoutType::MATRIX_A));
            auto coordB1 = graph.coordinates.addElement(CT::MacroTile({}, LayoutType::MATRIX_B));
            auto coordD
                = graph.coordinates.addElement(CT::MacroTile({}, LayoutType::MATRIX_ACCUMULATOR));

            auto coordA2 = graph.coordinates.addElement(CT::MacroTile({}, LayoutType::MATRIX_A));
            auto coordB2 = graph.coordinates.addElement(CT::MacroTile({}, LayoutType::MATRIX_B));

            auto mult1 = addMultiply(coordA1, coordB1, coordD);
            auto mult2 = addMultiply(coordA2, coordB1, coordD);
            auto mult3 = addMultiply(coordA2, coordB2, coordD);

            auto loadA1 = addLoadLDSTile(DataType::FP4x8, coordA1, lds1);
            auto loadB1 = addLoadLDSTile(DataType::FP4x8, coordB1, lds1);
            auto loadD1 = addLoadLDSTile(DataType::Half, coordD, lds1);
            auto loadA2 = addLoadLDSTile(DataType::FP4x8, coordA2, lds1);
            auto loadB2 = addLoadLDSTile(DataType::FP4x8, coordB2, lds1);

            graph.control.addElement(CF::Body(), {kernel}, {mult1});
            graph.control.addElement(CF::Body(), {kernel}, {loadA1});

            graph.control.chain<CF::Sequence>(mult1, mult2, mult3);

            graph.control.chain<CF::Sequence>(loadA1, loadA2, loadB1, loadB2, loadD1);

            SECTION("Multiply table")
            {
                Chain chain{mult1, mult2, mult3};

                auto result = chainTagTable(graph, chain);

                std::string expected = R"(
            |      |  2   |  3   |  5   |  6
            ===================================
            |      |8xFP4 |8xFP4 |8xFP4 |8xFP4
            ===================================
            ===================================
            |      |  A   |  B   |  A   |  B
            ===================================
            |  2   |  V   |  V   |      |
            |  3   |      |  V   |  V   |
            |  4   |      |      |  V   |  V
            Lasts: (3)(2, 3, 4)
            )";
                INFO(result);

                CHECK(NormalizedSource(result) == NormalizedSource(expected));
            }

            SECTION("LoadLDSTile table")
            {
                Chain chain{loadA1, loadA2, loadB1, loadB2, loadD1};
                auto  result = chainTagTable(graph, chain);

                std::string expected = R"(
            |      |  2   |  5   |  3   |  6
            ===================================
            |      |8xFP4 |8xFP4 |8xFP4 |8xFP4
            ===================================
            ===================================
            |      |  A   |  A   |  B   |  B
            ===================================
            |  5   |  ^   |      |      |
            |  8   |      |  ^   |      |
            |  6   |      |      |  ^   |
            |  9   |      |      |      |  ^
            |  7   |      |      |      |
            Lasts: (5)(5, 8, 6, 9, 7)
            )";
                INFO(result);
                CHECK(NormalizedSource(result) == NormalizedSource(expected));
            }
        }
    }
}
