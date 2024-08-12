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
