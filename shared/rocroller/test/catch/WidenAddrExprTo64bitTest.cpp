/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/Expression.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/TestValues.hpp>
#include <common/WidenAddrExprTo64bit.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("WidenAddrExprTo64bit ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using namespace Expression;

    auto context = TestContext::ForDefaultTarget();

    // Top-level expression should be in 64bit as the expression is for address calculation.
    SECTION("addr expression only")
    {
        auto one = literal(1);
        CHECK_THROWS_AS(rocRollerTest::widenAddrExprTo64bit(one), FatalError);
    }

    // When the datatype is in already 64-bit, no extra convert is added.
    SECTION("No Extra Convert")
    {
        int64_t two = 2;

        auto r64bit = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int64, 1);
        r64bit->allocateNow();
        auto v64bit = r64bit->expression();

        CHECK_THAT(rocRollerTest::widenAddrExprTo64bit(-literal(two)), IdenticalTo(-literal(two)));
        CHECK(toString(
                  rocRollerTest::widenAddrExprTo64bit(rocRollerTest::widenAddrExprTo64bit(v64bit)))
              == toString(v64bit));
    }

    SECTION("Widen")
    {
        auto a = literal(33);

        auto r0 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        r0->allocateNow();
        auto v0 = r0->expression();

        auto r1 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        r1->allocateNow();
        auto v1 = r1->expression();

        auto     mul = a * v0;
        uint64_t x   = 100;
        auto     add = mul + literal(x);

        CHECK(toString(add) == "Add(Multiply(33:I, v0:U32)U32, 100:U64)U64");
        CHECK(toString(rocRollerTest::widenAddrExprTo64bit(add))
              == "Add(Multiply(Convert(33:I)I64, Convert(v0:U32)U64)U64, 100:U64)U64");

        uint64_t             u64val = 0xFF;
        CommandArgumentValue ca     = u64val; //static_cast<uint64_t>()

        auto d1    = v0 / literal(16);
        auto mod   = d1 % literal(uint32_t(4));
        auto mul32 = v1 * literal(4);
        auto expr  = (mod + mul32) * literal(u64val);

        CHECK(toString(expr)
              == "Multiply(Add(Modulo(Divide(v0:U32, 16:I)U32, 4:U32)U32, Multiply(v1:U32, "
                 "4:I)U32)U32, 255:U64)U64");
        CHECK(toString(rocRollerTest::widenAddrExprTo64bit(expr))
              == "Multiply(Add(Modulo(Divide(Convert(v0:U32)U64, 16:I)U64, 4:U32)U64, "
                 "Multiply(Convert(v1:U32)U64, Convert(4:I)I64)U64)U64, 255:U64)U64");
    }
}
