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

#include <cmath>
#include <memory>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    TEST_CASE("Create expressions and convert to string", "[expression][toString]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto a = Expression::literal(1);
            auto b = Expression::literal(2);

            auto rc = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            rc->allocateNow();

            bool wave32 = GPUArchitectureLibrary::getInstance()->HasCapability(
                arch, GPUCapability::HasWave32);
            std::string waveBits = wave32 ? "32" : "64";

            auto expr1  = a + b;
            auto expr2  = b * expr1;
            auto expr3  = b * expr1 - rc->expression();
            auto expr4  = expr1 > expr2;
            auto expr5  = expr3 < expr2;
            auto expr6  = expr4 >= expr5;
            auto expr7  = expr2 / expr3;
            auto expr8  = expr2 == expr7;
            auto expr9  = -expr2;
            auto expr10 = Expression::fuseTernary(expr1 << b);
            auto expr11 = Expression::fuseTernary((a << b) + b);
            auto expr12 = expr3 != expr7;

            SECTION("toString()")
            {
                CHECK(toString(expr1) == "Add(1:I, 2:I)I");
                CHECK(toString(expr2) == "Multiply(2:I, Add(1:I, 2:I)I)I");
                CHECK(toString(expr3) == "Subtract(Multiply(2:I, Add(1:I, 2:I)I)I, v0:I)I");
                CHECK(toString(expr4)
                      == "GreaterThan(Add(1:I, 2:I)I, Multiply(2:I, Add(1:I, 2:I)I)I)BL");
                CHECK(toString(expr5)
                      == "LessThan(" + toString(expr3) + ", " + toString(expr2) + ")BL" + waveBits);
                CHECK(toString(expr6)
                      == "GreaterThanEqual(" + toString(expr4) + ", " + toString(expr5) + ")BL");
                CHECK(toString(expr7)
                      == "Divide(" + toString(expr2) + ", " + toString(expr3) + ")I");
                CHECK(toString(expr8)
                      == "Equal(" + toString(expr2) + ", " + toString(expr7) + ")BL" + waveBits);
                CHECK(toString(expr9) == "Negate(" + toString(expr2) + ")I");
                CHECK(toString(expr10) == "AddShiftL(1:I, 2:I, 2:I)I");
                CHECK(toString(expr11) == "ShiftLAdd(1:I, 2:I, 2:I)I");
                CHECK(toString(expr12)
                      == "NotEqual(" + toString(expr3) + ", " + toString(expr7) + ")BL" + waveBits);
            }

            SECTION("evaluationTimes()")
            {
                Expression::EvaluationTimes expectedTimes{
                    Expression::EvaluationTime::KernelExecute};
                CHECK(expectedTimes == Expression::evaluationTimes(expr8));
                CHECK(expectedTimes == Expression::evaluationTimes(expr10));
            }
        }
    }
}
