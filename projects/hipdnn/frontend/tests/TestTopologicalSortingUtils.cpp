// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn_frontend/node/TopologicalSortingUtils.hpp>

using namespace hipdnn_frontend;

class TestTopologicalSortingUtils : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code if needed
    }

    static bool isValidTopologicalOrder(const std::vector<size_t>& order,
                                        const GraphStructure& structure)
    {
        if(order.size() != structure.adjacencyList.size())
        {
            return false;
        }

        std::vector<size_t> position(structure.adjacencyList.size());
        for(size_t i = 0; i < order.size(); ++i)
        {
            position[order[i]] = i;
        }

        for(size_t node = 0; node < structure.adjacencyList.size(); ++node)
        {
            for(auto neighbor : structure.adjacencyList[node])
            {
                if(position[node] >= position[neighbor])
                {
                    return false;
                }
            }
        }

        return true;
    }
};

TEST_F(TestTopologicalSortingUtils, ComputesInDegreesCorrectly)
{
    {
        GraphStructure structure;
        structure.adjacencyList = {};
        std::vector<int> inDegreesExpected = {};
        EXPECT_EQ(computeInDegrees(structure), inDegreesExpected);
    }
    {
        GraphStructure structure;
        structure.adjacencyList = {{}};
        std::vector<int> inDegreesExpected = {0};
        EXPECT_EQ(computeInDegrees(structure), inDegreesExpected);
    }
    {
        GraphStructure structure;
        structure.adjacencyList = {{1}, {2}, {}};
        std::vector<int> inDegreesExpected = {0, 1, 1};
        EXPECT_EQ(computeInDegrees(structure), inDegreesExpected);
    }
    {
        GraphStructure structure;
        structure.adjacencyList = {
            {1, 2}, // 0
            {3, 4}, // 1
            {4, 5}, // 2
            {6}, // 3
            {6, 7}, // 4
            {8}, // 5
            {9}, // 6
            {9}, // 7
            {9}, // 8
            {} // 9
        };
        std::vector<int> inDegreesExpected = {0, 1, 1, 1, 2, 1, 2, 1, 1, 3};
        // Graph structure:
        //      0
        //     / \
        //    1   2
        //   / \ / \
        //  3  4   5
        //  | / \   \
        //  6    7   8
        //   \   |   /
        //      9

        EXPECT_EQ(computeInDegrees(structure), inDegreesExpected);
    }
}

TEST_F(TestTopologicalSortingUtils, EmptyGraph)
{
    GraphStructure structure;
    structure.adjacencyList = {};

    auto result = performTopologicalSort(structure);

    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, SingleNode)
{
    GraphStructure structure;
    structure.adjacencyList = {{}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, LinearChain)
{
    // Graph: 0 -> 1 -> 2 -> 3
    GraphStructure structure;
    structure.adjacencyList = {{1}, {2}, {3}, {}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], 3);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, DiamondGraph)
{
    // Graph:     0
    //           / \
    //          1   2
    //           \ /
    //            3
    GraphStructure structure;
    structure.adjacencyList = {{1, 2}, {3}, {3}, {}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[3], 3);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, MultipleSourcesAndSinks)
{
    // Graph: 0 -> 2    1 -> 2
    //        0 -> 3    1 -> 3
    GraphStructure structure;
    structure.adjacencyList = {{2, 3}, {2, 3}, {}, {}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 4);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, ComplexDag)
{
    // Graph:  0 -> 1 -> 3 -> 4
    //         0 -> 2 -> 3
    //         2 -> 4
    GraphStructure structure;
    structure.adjacencyList = {{1, 2}, {3}, {3, 4}, {4}, {}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 5);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[4], 4);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, DisconnectedNodes)
{
    // Graph: 0 -> 1    2 (disconnected)
    GraphStructure structure;
    structure.adjacencyList = {{1}, {}, {}};

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 3);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, LargerGraph)
{
    // Graph with 10 nodes forming a complex DAG

    GraphStructure structure;
    structure.adjacencyList = {
        {1, 2}, // 0
        {3, 4}, // 1
        {4, 5}, // 2
        {6}, // 3
        {6, 7}, // 4
        {8}, // 5
        {9}, // 6
        {9}, // 7
        {9}, // 8
        {} // 9
    };
    // Graph structure:
    //      0
    //     / \
    //    1   2
    //   / \ / \
    //  3  4   5
    //  | / \   \
    //  6    7   8
    //   \   |   /
    //      9

    auto result = performTopologicalSort(structure);

    ASSERT_EQ(result.size(), 10);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[9], 9);
    EXPECT_TRUE(isValidTopologicalOrder(result, structure));
    EXPECT_FALSE(detectCycle(result, structure));
}

TEST_F(TestTopologicalSortingUtils, CyclicGraph)
{
    // Graph: 0 -> 1 -> 2 -> 1 (cycle)
    GraphStructure structure;
    structure.adjacencyList = {{1}, {2}, {1}};
    auto sortResult = performTopologicalSort(structure);

    auto cycleDetected = detectCycle(sortResult, structure);

    EXPECT_TRUE(cycleDetected);
}
