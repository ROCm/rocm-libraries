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

#include <algorithm>
#include <rocRoller/Operations/OperationTag.hpp>
#include <unordered_set>
#include <vector>

#include "SimpleFixture.hpp"

class OperationTagTest : public SimpleFixture
{
};

TEST_F(OperationTagTest, Basic)
{
    using namespace rocRoller;

    // Verify default value of tag
    Operations::OperationTag tagDefault;
    EXPECT_EQ(static_cast<int32_t>(tagDefault), -1);

    // Verify tags are compared based on values
    std::vector<Operations::OperationTag> tagVec;
    tagVec.emplace_back(4);
    tagVec.emplace_back(3);
    tagVec.emplace_back(1);
    tagVec.emplace_back(2);

    std::sort(tagVec.begin(), tagVec.end());
    EXPECT_GT(tagVec[3].value, tagVec[2].value);
    EXPECT_GT(tagVec[2].value, tagVec[1].value);
    EXPECT_GT(tagVec[1].value, tagVec[0].value);
    EXPECT_GT(tagVec[3], tagVec[2]);
    EXPECT_GT(tagVec[2], tagVec[1]);
    EXPECT_GT(tagVec[1], tagVec[0]);

    // Verify OperationTag can hash
    std::unordered_set<Operations::OperationTag> tagSet;
    Operations::OperationTag                     tagA(1);
    Operations::OperationTag                     tagB(2);
    Operations::OperationTag                     tagC(3);
    tagSet.emplace(tagA);
    tagSet.emplace(tagB);
    tagSet.emplace(tagC);

    EXPECT_EQ(tagSet.count(tagA), 1);
    EXPECT_EQ(tagSet.count(tagB), 1);
    EXPECT_EQ(tagSet.count(tagC), 1);

    Operations::OperationTag tagD(4);
    EXPECT_EQ(tagSet.count(tagD), 0);

    // Cast operator
    EXPECT_EQ(static_cast<int32_t>(tagA), 1);
    EXPECT_EQ(static_cast<int32_t>(tagB), 2);
    EXPECT_EQ(static_cast<int32_t>(tagC), 3);

    // Postfix increment operator
    auto tagE = tagA++;
    EXPECT_EQ(static_cast<int32_t>(tagA), 2);
    EXPECT_EQ(static_cast<int32_t>(tagE), 1);

    // Prefix increment operator
    auto tagF = ++tagA;
    EXPECT_EQ(static_cast<int32_t>(tagA), 3);
    EXPECT_EQ(static_cast<int32_t>(tagF), 3);

    // Equal operator
    EXPECT_EQ(tagA, tagF);

    // Less than operator
    EXPECT_LT(tagB, tagC);
}
