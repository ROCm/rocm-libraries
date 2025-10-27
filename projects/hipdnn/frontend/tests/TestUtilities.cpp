// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::test_utilities;

// Mock node class for testing visitGraph
class MockNode : public INode
{
public:
    int value;

    explicit MockNode(int val, GraphAttributes attrs = GraphAttributes())
        : INode(std::move(attrs))
        , value(val)
    {
    }

    void addChild(const std::shared_ptr<MockNode>& child)
    {
        _sub_nodes.push_back(child);
    }
};

TEST(TestUtilities, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, InitializeFrontendLoggingReturnsCorrectly)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "info");

    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(nullptr), -1);
    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(), 0);
}

TEST(TestUtilities, VisitGraphSingleNode)
{
    auto root = std::make_shared<MockNode>(1);
    std::vector<int> visitedValues;

    visitGraph(root, [&visitedValues](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    EXPECT_EQ(visitedValues.size(), 1);
    EXPECT_EQ(visitedValues[0], 1);
}

TEST(TestUtilities, VisitGraphWithChildren)
{
    // Create a tree:
    //       1
    //      / \
    //     2   3
    auto root = std::make_shared<MockNode>(1);
    auto child1 = std::make_shared<MockNode>(2);
    auto child2 = std::make_shared<MockNode>(3);

    root->addChild(child1);
    root->addChild(child2);

    std::vector<int> visitedValues;
    visitGraph(root, [&visitedValues](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Pre-order traversal: root, then children left to right
    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestUtilities, VisitGraphDeepHierarchy)
{
    // Create a tree:
    //       1
    //       |
    //       2
    //      / \
    //     3   4
    //     |
    //     5
    auto root = std::make_shared<MockNode>(1);
    auto child1 = std::make_shared<MockNode>(2);
    auto child2 = std::make_shared<MockNode>(3);
    auto child3 = std::make_shared<MockNode>(4);
    auto child4 = std::make_shared<MockNode>(5);

    root->addChild(child1);
    child1->addChild(child2);
    child1->addChild(child3);
    child2->addChild(child4);

    std::vector<int> visitedValues;
    visitGraph(root, [&visitedValues](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
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

TEST(TestUtilities, VisitGraphNullSharedPtr)
{
    std::shared_ptr<MockNode> nullRoot = nullptr;
    int visitCount = 0;

    // Should handle null gracefully without crashing
    visitGraph(nullRoot, [&visitCount]([[maybe_unused]] INode& node) { visitCount++; });

    EXPECT_EQ(visitCount, 0);
}

TEST(TestUtilities, VisitGraphWithNullChildren)
{
    auto root = std::make_shared<MockNode>(1);
    auto child1 = std::make_shared<MockNode>(2);

    root->addChild(child1);
    root->addChild(nullptr); // Add a null child
    root->addChild(std::make_shared<MockNode>(3));

    std::vector<int> visitedValues;
    visitGraph(root, [&visitedValues](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    // Should skip null children
    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestUtilities, VisitGraphReferenceOverload)
{
    MockNode root(1);
    auto child1 = std::make_shared<MockNode>(2);
    auto child2 = std::make_shared<MockNode>(3);

    root.addChild(child1);
    root.addChild(child2);

    std::vector<int> visitedValues;
    visitGraph(root, [&visitedValues](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
        visitedValues.push_back(mockNode.value);
    });

    EXPECT_EQ(visitedValues.size(), 3);
    EXPECT_EQ(visitedValues[0], 1);
    EXPECT_EQ(visitedValues[1], 2);
    EXPECT_EQ(visitedValues[2], 3);
}

TEST(TestUtilities, VisitGraphModifyNodes)
{
    auto root = std::make_shared<MockNode>(1);
    auto child1 = std::make_shared<MockNode>(2);
    auto child2 = std::make_shared<MockNode>(3);

    root->addChild(child1);
    root->addChild(child2);

    // Modify all node values during traversal
    visitGraph(root, [](INode& node) {
        auto& mockNode = static_cast<MockNode&>(node);
        mockNode.value *= 10;
    });

    EXPECT_EQ(root->value, 10);
    EXPECT_EQ(child1->value, 20);
    EXPECT_EQ(child2->value, 30);
}
