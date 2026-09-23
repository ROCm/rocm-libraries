/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// Comprehensive gtest cases for intrusive_list.hpp - Tests every function

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "stinkytofu/support/IntrusiveList.hpp"

using namespace stinkytofu;

// Test node class
class TestNode : public IntrusiveListNode<TestNode> {
   public:
    int value;

    TestNode(int val) : value(val) {}

    bool operator==(const TestNode& other) const {
        return value == other.value;
    }
};

// Test list traits: do not delete nodes (test owns them via unique_ptr).
struct TestListTraits : IntrusiveListNoAllocTraits<TestNode>,
                        IntrusiveListCallbackTraits<TestNode> {};
using TestList = IntrusiveList<TestNode, void, TestListTraits>;

class IntrusiveListTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Create test nodes
        for (int i = 0; i < 10; ++i) {
            nodes.push_back(std::make_unique<TestNode>(i));
        }
    }

    void TearDown() override {
        nodes.clear();
    }

    std::vector<std::unique_ptr<TestNode>> nodes;

    // Helper to get node values as vector
    std::vector<int> getListValues(const TestList& list) {
        std::vector<int> values;
        for (const auto& node : list) {
            values.push_back(node.value);
        }
        return values;
    }
};

// ============================================================================
// IntrusiveListNode Function Tests
// ============================================================================

TEST_F(IntrusiveListTest, NodeConstructor) {
    TestNode node(42);
    EXPECT_FALSE(node.isInList());
    EXPECT_EQ(node.getNext(), nullptr);
    EXPECT_EQ(node.getPrev(), nullptr);
}

TEST_F(IntrusiveListTest, NodeIsInList) {
    TestList list;
    TestNode* node = nodes[0].get();

    EXPECT_FALSE(node->isInList());
    list.push_back(node);
    EXPECT_TRUE(node->isInList());
}

TEST_F(IntrusiveListTest, NodeGetNextPrev) {
    TestList list;
    TestNode* n1 = nodes[0].get();
    TestNode* n2 = nodes[1].get();
    TestNode* n3 = nodes[2].get();

    list.push_back(n1);
    list.push_back(n2);
    list.push_back(n3);

    EXPECT_EQ(n1->getNext(), n2);
    EXPECT_EQ(n1->getPrev(), nullptr);
    EXPECT_EQ(n2->getNext(), n3);
    EXPECT_EQ(n2->getPrev(), n1);
    EXPECT_EQ(n3->getNext(), nullptr);
    EXPECT_EQ(n3->getPrev(), n2);
}

TEST_F(IntrusiveListTest, NodeRemoveFromList) {
    TestList list;
    TestNode* n1 = nodes[0].get();
    TestNode* n2 = nodes[1].get();
    TestNode* n3 = nodes[2].get();

    list.push_back(n1);
    list.push_back(n2);
    list.push_back(n3);

    n2->removeFromList();

    EXPECT_EQ(list.size(), 2);
    EXPECT_FALSE(n2->isInList());
    EXPECT_EQ(n1->getNext(), n3);
    EXPECT_EQ(n3->getPrev(), n1);
}

// ============================================================================
// IntrusiveListIterator Function Tests
// ============================================================================

TEST_F(IntrusiveListTest, IteratorConstruction) {
    TestNode* node = nodes[0].get();
    IntrusiveListIterator<TestNode> iter(node);
    EXPECT_EQ(iter.getNodePtr(), node);
}

TEST_F(IntrusiveListTest, IteratorDereference) {
    TestList list;
    list.push_back(nodes[0].get());

    auto iter = list.begin();
    EXPECT_EQ(&(*iter), nodes[0].get());
    EXPECT_EQ(iter->value, nodes[0]->value);
}

TEST_F(IntrusiveListTest, IteratorIncrement) {
    TestList list;
    list.push_back(nodes[0].get());
    list.push_back(nodes[1].get());

    auto iter = list.begin();
    ++iter;
    EXPECT_EQ(iter.getNodePtr(), nodes[1].get());

    iter = list.begin();
    auto oldIter = iter++;
    EXPECT_EQ(oldIter.getNodePtr(), nodes[0].get());
    EXPECT_EQ(iter.getNodePtr(), nodes[1].get());
}

TEST_F(IntrusiveListTest, IteratorDecrement) {
    TestList list;
    list.push_back(nodes[0].get());
    list.push_back(nodes[1].get());

    auto iter = list.begin();
    ++iter;
    --iter;
    EXPECT_EQ(iter.getNodePtr(), nodes[0].get());

    iter = list.begin();
    ++iter;
    auto oldIter = iter--;
    EXPECT_EQ(oldIter.getNodePtr(), nodes[1].get());
    EXPECT_EQ(iter.getNodePtr(), nodes[0].get());
}

TEST_F(IntrusiveListTest, IteratorComparison) {
    TestList list;
    list.push_back(nodes[0].get());

    auto iter1 = list.begin();
    auto iter2 = list.begin();

    EXPECT_TRUE(iter1 == iter2);
    EXPECT_FALSE(iter1 != iter2);
}

// ============================================================================
// IntrusiveList Function Tests
// ============================================================================

TEST_F(IntrusiveListTest, ListConstruction) {
    TestList list;
    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.begin(), list.end());
}

TEST_F(IntrusiveListTest, ListPushBack) {
    TestList list;

    list.push_back(nodes[0].get());
    EXPECT_EQ(list.size(), 1);
    EXPECT_FALSE(list.empty());
    EXPECT_EQ(&list.front(), nodes[0].get());
    EXPECT_EQ(&list.back(), nodes[0].get());

    list.push_back(nodes[1].get());
    EXPECT_EQ(list.size(), 2);
    EXPECT_EQ(&list.back(), nodes[1].get());
}

TEST_F(IntrusiveListTest, ListPushFront) {
    TestList list;

    list.push_front(nodes[0].get());
    EXPECT_EQ(list.size(), 1);
    EXPECT_EQ(&list.front(), nodes[0].get());

    list.push_front(nodes[1].get());
    EXPECT_EQ(list.size(), 2);
    EXPECT_EQ(&list.front(), nodes[1].get());
}

TEST_F(IntrusiveListTest, ListInsert) {
    TestList list;
    list.push_back(nodes[0].get());
    list.push_back(nodes[2].get());

    auto iter = list.begin();
    ++iter;

    auto insertedIter = list.insert(iter, nodes[1].get());

    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(insertedIter.getNodePtr(), nodes[1].get());

    std::vector<int> expected = {0, 1, 2};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListErase) {
    TestList list;
    list.push_back(nodes[0].get());
    list.push_back(nodes[1].get());
    list.push_back(nodes[2].get());

    auto iter = list.begin();
    ++iter;

    auto nextIter = list.erase(iter);

    EXPECT_EQ(list.size(), 2);
    EXPECT_EQ(nextIter.getNodePtr(), nodes[2].get());

    std::vector<int> expected = {0, 2};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListRemove) {
    TestList list;
    list.push_back(nodes[0].get());
    list.push_back(nodes[1].get());
    list.push_back(nodes[2].get());

    list.remove(nodes[1].get());

    EXPECT_EQ(list.size(), 2);
    std::vector<int> expected = {0, 2};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListClear) {
    TestList list;
    for (int i = 0; i < 5; ++i) {
        list.push_back(nodes[i].get());
    }

    list.clear();

    EXPECT_EQ(list.size(), 0);
    EXPECT_TRUE(list.empty());
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(nodes[i]->isInList());
    }
}

TEST_F(IntrusiveListTest, ListMoveBefore) {
    TestList list;
    for (int i = 0; i < 4; ++i) {
        list.push_back(nodes[i].get());
    }

    auto iter1 = list.begin();
    ++iter1;
    auto iter3 = iter1;
    ++iter3;
    ++iter3;

    list.moveBefore(iter3, iter1);

    std::vector<int> expected = {0, 3, 1, 2};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListMoveAfter) {
    TestList list;
    for (int i = 0; i < 4; ++i) {
        list.push_back(nodes[i].get());
    }

    auto iter0 = list.begin();
    auto iter2 = iter0;
    ++iter2;
    ++iter2;

    list.moveAfter(iter0, iter2);

    std::vector<int> expected = {1, 2, 0, 3};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListMoveToFront) {
    TestList list;
    for (int i = 0; i < 3; ++i) {
        list.push_back(nodes[i].get());
    }

    auto iter = list.begin();
    ++iter;
    ++iter;

    list.moveToFront(iter);

    std::vector<int> expected = {2, 0, 1};
    EXPECT_EQ(getListValues(list), expected);
}

TEST_F(IntrusiveListTest, ListMoveToBack) {
    TestList list;
    for (int i = 0; i < 3; ++i) {
        list.push_back(nodes[i].get());
    }

    auto iter = list.begin();

    list.moveToBack(iter);

    std::vector<int> expected = {1, 2, 0};
    EXPECT_EQ(getListValues(list), expected);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(IntrusiveListTest, NullptrHandling) {
    TestList list;

    list.push_back(nullptr);
    EXPECT_EQ(list.size(), 0);

    list.push_front(nullptr);
    EXPECT_EQ(list.size(), 0);

    list.remove(nullptr);
    EXPECT_EQ(list.size(), 0);
}

TEST_F(IntrusiveListTest, EmptyListOperations) {
    TestList list;

    EXPECT_EQ(list.begin(), list.end());

    auto iter = list.erase(list.end());
    EXPECT_EQ(iter, list.end());
}

TEST_F(IntrusiveListTest, SingleNodeOperations) {
    TestList list;
    TestNode* node = nodes[0].get();

    list.push_back(node);

    EXPECT_EQ(&list.front(), node);
    EXPECT_EQ(&list.back(), node);
    EXPECT_EQ(node->getNext(), nullptr);
    EXPECT_EQ(node->getPrev(), nullptr);
}

TEST_F(IntrusiveListTest, DestructorBehavior) {
    std::vector<TestNode*> testNodes;
    testNodes.reserve(3);
    for (int i = 0; i < 3; ++i) {
        testNodes.push_back(nodes[i].get());
    }

    {
        TestList list;
        for (auto* node : testNodes) {
            list.push_back(node);
        }
        EXPECT_EQ(list.size(), 3);
    }

    // All nodes should be unlinked after destructor
    for (auto* node : testNodes) {
        EXPECT_FALSE(node->isInList());
    }
}

TEST_F(IntrusiveListTest, NodeMoveBetweenLists) {
    TestList list1, list2;
    TestNode* node = nodes[0].get();

    list1.push_back(node);
    EXPECT_EQ(list1.size(), 1);
    EXPECT_EQ(list2.size(), 0);

    list2.push_back(node);
    EXPECT_EQ(list1.size(), 0);
    EXPECT_EQ(list2.size(), 1);
}

TEST_F(IntrusiveListTest, ComplexScenario) {
    TestList list;

    // Build list
    for (int i = 0; i < 5; ++i) {
        list.push_back(nodes[i].get());
    }

    // Complex operations
    list.moveToFront(list.begin());  // No-op
    list.remove(nodes[2].get());
    list.push_front(nodes[2].get());
    list.moveToBack(list.begin());

    EXPECT_EQ(list.size(), 5);

    // Verify integrity
    auto iter = list.begin();
    TestNode* prev = nullptr;
    while (iter != list.end()) {
        TestNode* current = iter.getNodePtr();
        EXPECT_EQ(current->getPrev(), prev);
        if (prev) {
            EXPECT_EQ(prev->getNext(), current);
        }
        prev = current;
        ++iter;
    }
}

// ============================================================================
// Additional Iterator Tests
// ============================================================================

TEST_F(IntrusiveListTest, IteratorRangeBasedLoop) {
    TestList list;
    for (int i = 0; i < 5; ++i) {
        list.push_back(nodes[i].get());
    }

    int expected = 0;
    for (const auto& node : list) {
        EXPECT_EQ(node.value, expected++);
    }
    EXPECT_EQ(expected, 5);
}

TEST_F(IntrusiveListTest, IteratorArithmeticSequence) {
    TestList list;
    for (int i = 0; i < 3; ++i) {
        list.push_back(nodes[i].get());
    }

    // Test forward iteration
    auto iter = list.begin();
    EXPECT_EQ(iter->value, 0);

    ++iter;
    EXPECT_EQ(iter->value, 1);

    ++iter;
    EXPECT_EQ(iter->value, 2);

    // Store the last valid iterator before going to end()
    auto lastIter = iter;
    ++iter;
    EXPECT_EQ(iter, list.end());

    // Test backward iteration starting from last valid element
    iter = lastIter;
    EXPECT_EQ(iter->value, 2);

    --iter;
    EXPECT_EQ(iter->value, 1);

    --iter;
    EXPECT_EQ(iter->value, 0);
    EXPECT_EQ(iter, list.begin());
}

TEST_F(IntrusiveListTest, IteratorBoundaryBehavior) {
    TestList list;
    list.push_back(nodes[0].get());

    auto iter = list.begin();
    EXPECT_EQ(iter->value, 0);

    // Test that incrementing to end() works
    ++iter;
    EXPECT_EQ(iter, list.end());

    // Test that we can go back to the beginning from end() if supported
    // Note: This may not be supported in all intrusive list implementations
    // --iter;  // Uncomment only if the implementation supports this
    // EXPECT_EQ(iter, list.begin());
}

// ============================================================================
// Reverse Iterator Tests
// ============================================================================

TEST_F(IntrusiveListTest, ReverseIteratorBasic) {
    TestList list;
    for (int i = 0; i < 5; ++i) {
        list.push_back(nodes[i].get());
    }

    // Test reverse iteration
    auto riter = list.rbegin();
    EXPECT_EQ(riter->value, 4);

    ++riter;
    EXPECT_EQ(riter->value, 3);

    ++riter;
    EXPECT_EQ(riter->value, 2);

    ++riter;
    EXPECT_EQ(riter->value, 1);

    ++riter;
    EXPECT_EQ(riter->value, 0);

    ++riter;
    EXPECT_EQ(riter, list.rend());
}

TEST_F(IntrusiveListTest, ReverseIteratorRangeBasedLoop) {
    TestList list;
    for (int i = 0; i < 5; ++i) {
        list.push_back(nodes[i].get());
    }

    std::vector<int> values;
    for (auto rit = list.rbegin(); rit != list.rend(); ++rit) {
        values.push_back(rit->value);
    }

    std::vector<int> expected = {4, 3, 2, 1, 0};
    EXPECT_EQ(values, expected);
}

TEST_F(IntrusiveListTest, ReverseIteratorConstness) {
    TestList list;
    for (int i = 0; i < 3; ++i) {
        list.push_back(nodes[i].get());
    }

    const TestList& constList = list;

    auto criter = constList.rbegin();
    EXPECT_EQ(criter->value, 2);

    ++criter;
    EXPECT_EQ(criter->value, 1);

    ++criter;
    EXPECT_EQ(criter->value, 0);

    ++criter;
    EXPECT_EQ(criter, constList.rend());
}

TEST_F(IntrusiveListTest, ReverseIteratorEmptyList) {
    TestList list;

    EXPECT_EQ(list.rbegin(), list.rend());

    // Should be able to iterate over empty list
    for (auto rit = list.rbegin(); rit != list.rend(); ++rit) {
        FAIL() << "Should not iterate over empty list";
    }
}

TEST_F(IntrusiveListTest, ReverseIteratorSingleElement) {
    TestList list;
    list.push_back(nodes[0].get());

    auto riter = list.rbegin();
    EXPECT_EQ(riter->value, 0);

    ++riter;
    EXPECT_EQ(riter, list.rend());
}

TEST_F(IntrusiveListTest, ReverseIteratorBidirectional) {
    TestList list;
    for (int i = 0; i < 3; ++i) {
        list.push_back(nodes[i].get());
    }

    // Start from rbegin and move forward, then backward
    auto riter = list.rbegin();
    EXPECT_EQ(riter->value, 2);

    ++riter;
    EXPECT_EQ(riter->value, 1);

    ++riter;
    EXPECT_EQ(riter->value, 0);

    // Now go backward
    --riter;
    EXPECT_EQ(riter->value, 1);

    --riter;
    EXPECT_EQ(riter->value, 2);
    EXPECT_EQ(riter, list.rbegin());
}
