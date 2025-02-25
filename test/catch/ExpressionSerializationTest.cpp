#include <cmath>
#include <memory>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

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

    TEST_CASE("Expression serialization", "[expression][serialization]")
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);
        SECTION("Serializable expressions")
        {

            auto c = Register::Value::Literal(4.2f);
            auto d = Register::Value::Literal(Half(4.2f));

            Expression::DataFlowTag dataFlow;
            dataFlow.tag              = 50;
            dataFlow.regType          = Register::Type::Vector;
            dataFlow.varType.dataType = DataType::Float;

            auto expr1  = a + b;
            auto expr2  = b * expr1;
            auto expr3  = b * expr1 - convert<DataType::Int32>(c->expression());
            auto expr4  = expr1 > (expr2 + convert<DataType::Int32>(d->expression()));
            auto expr5  = expr3 < expr4;
            auto expr6  = expr4 >= expr5;
            auto expr7  = expr5 <= expr6;
            auto expr8  = expr6 == expr7;
            auto expr9  = -expr2;
            auto expr10 = Expression::fuseTernary(expr1 << b);
            auto expr11 = Expression::fuseTernary((a << b) + b);
            auto expr12
                = std::make_shared<Expression::Expression>(dataFlow) / convert<DataType::Float>(a);

            auto expr = GENERATE_COPY(expr1,
                                      expr2,
                                      expr3,
                                      expr4,
                                      expr5,
                                      expr6,
                                      expr7,
                                      expr8,
                                      expr9,
                                      expr10,
                                      expr11,
                                      expr12);

            CAPTURE(expr);

            auto yamlText = Expression::toYAML(expr);
            INFO(yamlText);

            CHECK(yamlText != "");

            auto deserialized = Expression::fromYAML(yamlText);
            REQUIRE(deserialized.get() != nullptr);

            CHECK(Expression::toString(deserialized) == Expression::toString(expr));
            CHECK(Expression::identical(deserialized, expr));

            SECTION("Kernel arg")
            {
                auto kernelArg                   = std::make_shared<AssemblyKernelArgument>();
                kernelArg->name                  = "KernelArg1";
                kernelArg->variableType.dataType = DataType::Int32;
                kernelArg->expression            = Expression::literal(10);
                kernelArg->offset                = 1;
                kernelArg->size                  = 5;

                auto expr = b >> std::make_shared<Expression::Expression>(kernelArg);

                auto yamlText = Expression::toYAML(expr);
                INFO(yamlText);

                CHECK(yamlText != "");

                auto deserialized = Expression::fromYAML(yamlText);
                REQUIRE(deserialized.get() != nullptr);

                CHECK(Expression::toString(deserialized) == Expression::toString(expr));
                CHECK(Expression::identical(deserialized, expr));
            }
        }

        SECTION("Unserializable expressions")
        {
            SECTION("WaveTile")
            {
                auto waveTile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
                auto expr     = std::make_shared<Expression::Expression>(waveTile) + b;
                CHECK_THROWS(Expression::fromYAML(Expression::toYAML(expr)));
            }

            SUPPORTED_ARCH_SECTION(arch)
            {
                auto context = TestContext::ForTarget(arch);

                auto reg = std::make_shared<Register::Value>(
                    context.get(), Register::Type::Vector, DataType::Int32, 1);
                reg->allocateNow();

                CHECK_THROWS(Expression::toYAML(reg->expression()));
            }
        }
    }

}
