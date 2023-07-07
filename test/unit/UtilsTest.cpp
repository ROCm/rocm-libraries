

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "SourceMatcher.hpp"
#include "Utilities.hpp"

TEST(UtilsTest, StreamTuple)
{
    std::string        str  = "foo";
    auto               test = std::make_tuple(4, 5.6, str);
    std::ostringstream msg;
    msg << test;

    EXPECT_EQ("[4, 5.6, foo]", msg.str());
}

TEST(UtilsTest, StreamTuple2)
{
    std::string        str  = "foo";
    auto               test = std::make_tuple(str);
    std::ostringstream msg;
    msg << test;

    EXPECT_EQ("[foo]", msg.str());
}

enum class TestEnum : int
{
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    A16,
    A17,
    A18,
    A19,
    A20,
    A21,
    A22,
    A23,
    A24,
    A25,
    A26,
    A27,
    A28,
    A29,
    A30,
    A31,
    A32,
    A33,
    Count
};
std::string toString(TestEnum e)
{
    using namespace rocRoller;
    if(e == TestEnum::Count)
        Throw<FatalError>("Invalid TestEnum");
    auto idx = static_cast<int>(e) + 1;
    return concatenate("A", idx);
}

TEST(EnumBitsetTest, LargeEnum)
{

    using LargeBitset = rocRoller::EnumBitset<TestEnum>;

    LargeBitset a1{TestEnum::A1};
    LargeBitset a33{TestEnum::A33};
    LargeBitset combined{TestEnum::A1, TestEnum::A33};

    EXPECT_NE(a1, a33);
    EXPECT_TRUE(a1[TestEnum::A1]);
    EXPECT_FALSE(a1[TestEnum::A33]);
    EXPECT_TRUE(a33[TestEnum::A33]);
    EXPECT_EQ(combined, a1 | a33);
    EXPECT_TRUE(combined[TestEnum::A1]);
    EXPECT_TRUE(combined[TestEnum::A33]);

    a1[TestEnum::A33] = true;
    a1[TestEnum::A1]  = false;
    EXPECT_FALSE(a1[TestEnum::A1]);
    EXPECT_TRUE(a1[TestEnum::A33]);

    auto expected = R"(
        A1: false
        A2: false
        A3: false
        A4: false
        A5: false
        A6: false
        A7: false
        A8: false
        A9: false
        A10: false
        A11: false
        A12: false
        A13: false
        A14: false
        A15: false
        A16: false
        A17: false
        A18: false
        A19: false
        A20: false
        A21: false
        A22: false
        A23: false
        A24: false
        A25: false
        A26: false
        A27: false
        A28: false
        A29: false
        A30: false
        A31: false
        A32: false
        A33: true
        )";

    EXPECT_EQ(NormalizedSource(expected), NormalizedSource(rocRoller::concatenate(a1)));
}

TEST(UtilsTest, SetIdentityMatrix)
{
    using namespace rocRoller;

    std::vector<float> mat(3 * 5);
    SetIdentityMatrix(mat, 3, 5);

    // clang-format off
    std::vector<float> expected = { 1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1,
                                    0, 0, 0,
                                    0, 0, 0,
                                  };
    // clang-format on

    EXPECT_EQ(mat, expected);

    SetIdentityMatrix(mat, 5, 3);
    // clang-format off
     expected = { 1, 0, 0, 0, 0,
                  0, 1, 0, 0, 0,
                  0, 0, 1, 0, 0 };
    // clang-format on

    EXPECT_EQ(mat, expected);
}
