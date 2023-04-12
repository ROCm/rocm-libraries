

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/Utils.hpp>

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

TEST(EnumBitsetTest, LargeEnum)
{

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
