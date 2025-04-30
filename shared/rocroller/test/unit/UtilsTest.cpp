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

#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

class UtilsTest : public SimpleFixture
{
};

class EnumBitsetTest : public SimpleFixture
{
};

TEST_F(UtilsTest, StreamTuple)
{
    std::string        str  = "foo";
    auto               test = std::make_tuple(4, 5.6, str);
    std::ostringstream msg;
    msg << test;

    EXPECT_EQ("[4, 5.6, foo]", msg.str());
}

TEST_F(UtilsTest, StreamTuple2)
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

TEST_F(EnumBitsetTest, LargeEnum)
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

TEST_F(UtilsTest, SetIdentityMatrix)
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

TEST_F(UtilsTest, SetIdentityMatrixFP4)
{
    using namespace rocRoller;

    size_t const rows = 8;
    size_t const cols = 8;

    auto rng = RandomGenerator(12345);
    auto mat = rng.vector<FP4>(rows * cols, -6.0f, 6.0f);

    SetIdentityMatrix(mat, rows, cols);

    // clang-format off
    auto const expected = std::to_array<uint8_t> ({
       0b00100000, 0b00000000, 0b00000000, 0b00000000,

       0b00000010, 0b00000000, 0b00000000, 0b00000000,

       0b00000000, 0b00100000, 0b00000000, 0b00000000,

       0b00000000, 0b00000010, 0b00000000, 0b00000000,

       0b00000000, 0b00000000, 0b00100000, 0b00000000,

       0b00000000, 0b00000000, 0b00000010, 0b00000000,

       0b00000000, 0b00000000, 0b00000000, 0b00100000,

       0b00000000, 0b00000000, 0b00000000, 0b00000010,
    });
    // clang-format on

    EXPECT_EQ(std::memcmp(mat.data(), expected.data(), expected.size()), 0);
}

template <typename F6Type>
std::string createF6IdentityMatrix(float min, float max)
{
    using namespace rocRoller;

    size_t constexpr rows    = 16;
    size_t constexpr cols    = 16;
    size_t constexpr numBits = rows * cols * 6;

    auto rng = RandomGenerator(12345);
    auto mat = rng.vector<F6Type>(rows * cols, min, max);

    SetIdentityMatrix(mat, rows, cols);

    auto        ptr = reinterpret_cast<uint8_t*>(mat.data());
    std::string bits;
    bits.reserve(numBits);
    for(int i = 0; i < (numBits / 8); i++, ptr++)
    {
        uint8_t        value = *ptr;
        std::bitset<8> valueBits(value);
        for(int j = 7; j >= 0; j--)
        {
            if(valueBits.test(j))
                bits.push_back('1');
            else
                bits.push_back('0');
        }
    }
    return bits;
}

TEST_F(UtilsTest, SetIdentityMatrixF6)
{
    auto fp6_bits = createF6IdentityMatrix<rocRoller::FP6>(-7.5f, 7.5f);

#define ZERO "000000"
    // clang-format off
    std::string_view expected_fp6_bits(
       "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000" ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001000"
    );
    // clang-format on
#undef ZERO

    EXPECT_EQ(fp6_bits, expected_fp6_bits);

    auto bf6_bits = createF6IdentityMatrix<rocRoller::BF6>(-28.0f, 28.0f);

#define ZERO "000000"
    // clang-format off
    std::string_view const expected_bf6_bits(
       "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO  ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100" ZERO
        ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO  ZERO "001100"
    );
    // clang-format on
#undef ZERO

    EXPECT_EQ(bf6_bits, expected_bf6_bits);
}

TEST_F(UtilsTest, SingleVariant)
{
    using namespace rocRoller;

    EXPECT_EQ(std::variant<int>(5), singleVariant(5));

    std::string value = "five";
    auto        var   = singleVariant(value);

    EXPECT_EQ(std::variant<std::string>(value), var);

    auto visitor = [&](int value) { EXPECT_EQ(value, 15); };

    std::visit(visitor, singleVariant(15));
}
