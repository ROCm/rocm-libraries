// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/node/Node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace ::testing;

namespace hipdnn_frontend
{

// Mock node class for testing INode::visit()
class FakeNode : public INode
{
public:
    int value;

    explicit FakeNode(int val, GraphAttributes attrs = GraphAttributes())
        : INode(std::move(attrs))
        , value(val)
    {
    }

    void addChild(const std::shared_ptr<FakeNode>& child)
    {
        _sub_nodes.push_back(child);
    }
};

} // namespace hipdnn_frontend

TEST(TestINode, VisitGraphSingleNode)
{
    auto root = std::make_shared<FakeNode>(1);
    std::vector<int> visitedValues;

    root->visit([&visitedValues](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    EXPECT_EQ(visitedValues.size(), 1);
    EXPECT_EQ(visitedValues[0], 1);
}

TEST(TestINode, VisitGraphWithChildren)
{
    // Create a tree:
    //       1
    //      / \
    //     2   3
    auto root = std::make_shared<FakeNode>(1);
    auto child1 = std::make_shared<FakeNode>(2);
    auto child2 = std::make_shared<FakeNode>(3);

    root->addChild(child1);
    root->addChild(child2);

    std::vector<int> visitedValues;
    root->visit([&visitedValues](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Pre-order traversal: root, then children left to right
    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestINode, VisitGraphDeepHierarchy)
{
    // Create a tree:
    //       1
    //       |
    //       2
    //      / \
    //     3   4
    //     |
    //     5
    auto root = std::make_shared<FakeNode>(1);
    auto child1 = std::make_shared<FakeNode>(2);
    auto child2 = std::make_shared<FakeNode>(3);
    auto child3 = std::make_shared<FakeNode>(4);
    auto child4 = std::make_shared<FakeNode>(5);

    root->addChild(child1);
    child1->addChild(child2);
    child1->addChild(child3);
    child2->addChild(child4);

    std::vector<int> visitedValues;
    root->visit([&visitedValues](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Pre-order traversal: 1, 2, 3, 5, 4
    EXPECT_EQ(visitedValues.size(), 5);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
    EXPECT_EQ(visitedValues[3], 5);
    EXPECT_EQ(visitedValues[4], 4);
}

TEST(TestINode, VisitConstGraphDeepHierarchy)
{
    // Create a tree:
    //       1
    //       |
    //       2
    //      / \
    //     3   4
    //     |
    //     5
    auto root = std::make_shared<FakeNode>(1);
    auto child1 = std::make_shared<FakeNode>(2);
    auto child2 = std::make_shared<FakeNode>(3);
    auto child3 = std::make_shared<FakeNode>(4);
    auto child4 = std::make_shared<FakeNode>(5);

    root->addChild(child1);
    child1->addChild(child2);
    child1->addChild(child3);
    child2->addChild(child4);

    const auto& constRoot = *root;

    std::vector<int> visitedValues;
    constRoot.visit([&visitedValues](const INode& node) {
        auto& mockNode = static_cast<const FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Pre-order traversal: 1, 2, 3, 5, 4
    EXPECT_EQ(visitedValues.size(), 5);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
    EXPECT_EQ(visitedValues[3], 5);
    EXPECT_EQ(visitedValues[4], 4);
}

TEST(TestINode, VisitGraphWithNullChildren)
{
    auto root = std::make_shared<FakeNode>(1);
    auto child1 = std::make_shared<FakeNode>(2);

    root->addChild(child1);
    root->addChild(nullptr); // Add a null child
    root->addChild(std::make_shared<FakeNode>(3));

    std::vector<int> visitedValues;
    root->visit([&visitedValues](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Should skip null children
    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestINode, VisitGraphReferenceOverload)
{
    FakeNode root(1);
    auto child1 = std::make_shared<FakeNode>(2);
    auto child2 = std::make_shared<FakeNode>(3);

    root.addChild(child1);
    root.addChild(child2);

    std::vector<int> visitedValues;
    root.visit([&visitedValues](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestINode, VisitGraphModifyNodes)
{
    auto root = std::make_shared<FakeNode>(1);
    auto child1 = std::make_shared<FakeNode>(2);
    auto child2 = std::make_shared<FakeNode>(3);

    root->addChild(child1);
    root->addChild(child2);

    // Modify all node values during traversal
    root->visit([](INode& node) {
        auto& mockNode = static_cast<FakeNode&>(node);
        mockNode.value *= 10;
    });

    EXPECT_EQ(root->value, 10);
    EXPECT_EQ(child1->value, 20);
    EXPECT_EQ(child2->value, 30);
}
