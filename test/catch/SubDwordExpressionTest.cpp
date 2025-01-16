#include <rocRoller/Expression.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <common/SourceMatcher.hpp>

#include "TestContext.hpp"

using namespace rocRoller;

namespace SubDwordExpressionTest
{
    TEST_CASE("code generation of basic sub-dword expressions", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        SECTION("cannot get bitfield from a literal")
        {
            auto literal = Register::Value::Literal(42);
            CHECK_THROWS(literal->bitfield(8, 8));
        }

        SECTION("bitOffset cannot be greater than the number of bits in a Value")
        {
            auto r = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            CHECK_THROWS(r->bitfield(32, 8));
        }

        SECTION("bitwidth cannot be zero")
        {
            auto r = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            CHECK_THROWS(r->bitfield(8, 0));
        }

        SECTION("bitwidth cannot be greater than the number of bits in a register")
        {
            auto r = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            CHECK_THROWS(r->bitfield(8, 32));
        }

        SECTION("indices must refer to adjacent elements")
        {
            auto r = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt8x4, 1);
            CHECK_THROWS(r->segment({0, 2}));
        }

        SECTION("generate from expression with bitfields")
        {
            Register::ValuePtr dest;

            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            ra->setName("ra");
            ra->allocateNow();

            auto rb = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 4);
            rb->setName("rb");
            rb->allocateNow();

            auto rc = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            rc->setName("rc");
            rc->allocateNow();

            auto raByte1     = ra->bitfield(8, 8);
            auto rbReg2Byte1 = rb->bitfield(72, 16);
            auto rcByte2     = rc->bitfield(16, 8);

            auto a = raByte1->expression();
            auto b = rbReg2Byte1->expression();
            auto c = rcByte2->expression();

            auto expr1 = a + b;
            auto expr2 = expr1 * c;

            context.get()->schedule(Expression::generate(dest, expr2, context.get()));

            std::string expected = R"(
                v_bfe_u32 v6, v0, 8, 8
                v_bfe_u32 v7, v2, 8, 16
                v_bfe_u32 v8, v5, 16, 8
                v_add_u32 v9, v6, v7
                v_mul_lo_u32 v7, v9, v8
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
        }
    }

    TEST_CASE("code generation of sub-dword expressions of packed types", "[expression][codegen]")
    {

        auto dataType = GENERATE(DataType::Halfx2,
                                 DataType::BFloat16x2,
                                 DataType::FP8x4,
                                 DataType::BF8x4,
                                 DataType::Int8x4,
                                 DataType::UInt8x4);
        SECTION("for each packed datatype")
        {
            auto       info      = DataTypeInfo::Get(dataType);
            auto const bitOffset = 16;
            auto const bitWidth  = info.elementBits / info.packing;
            auto const index     = bitOffset / bitWidth;

            std::string codeForA, codeForB;

            {
                auto ctx = TestContext::ForDefaultTarget().get();

                auto ra
                    = std::make_shared<Register::Value>(ctx, Register::Type::Vector, dataType, 1);
                ra->setName("ra");
                ra->allocateNow();

                auto A = ra->bitfield(bitOffset, bitWidth)->expression();

                Register::ValuePtr dest = std::make_shared<Register::Value>(
                    ctx, Register::Type::Vector, info.segmentVariableType.dataType, 1);
                dest->allocateNow();
                ctx->schedule(Expression::generate(dest, A, ctx));

                codeForA = ctx->instructions()->toString();
            }

            {
                auto ctx = TestContext::ForDefaultTarget().get();

                auto rb
                    = std::make_shared<Register::Value>(ctx, Register::Type::Vector, dataType, 1);
                rb->setName("rb");
                rb->allocateNow();

                auto B = rb->segment({index})->expression();

                Register::ValuePtr dest = std::make_shared<Register::Value>(
                    ctx, Register::Type::Vector, info.segmentVariableType.dataType, 1);
                dest->allocateNow();
                ctx->schedule(Expression::generate(dest, B, ctx));

                codeForB = ctx->instructions()->toString();
            }

            CHECK(NormalizedSource(codeForA) == NormalizedSource(codeForB));
        }
    }
}
