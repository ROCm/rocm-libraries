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
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace ExpressionTest
{

    TEST_CASE("Basic expression code generation", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        Register::ValuePtr dest;
        context.get()->schedule(Expression::generate(dest, expr2, context.get()));

        // Explicitly copy the result into another register.
        auto dest2 = dest->placeholder();
        dest2->allocateNow();
        auto regIndexBefore = Generated(dest2->registerIndices())[0];

        context.get()->schedule(Expression::generate(dest2, dest->expression(), context.get()));
        auto regIndexAfter = Generated(dest2->registerIndices())[0];
        CHECK(regIndexBefore == regIndexAfter);

        context.get()->schedule(Expression::generate(dest2, expr2, context.get()));
        regIndexAfter = Generated(dest2->registerIndices())[0];
        CHECK(regIndexBefore == regIndexAfter);

        std::string expected = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v2

            // Note that v2 is reused
            v_mov_b32 v2, v3

            // Still storing into v2
            v_add_i32 v4, v0, v1
            v_mul_lo_u32 v2, v1, v4
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
    }

    TEST_CASE("FMA Expressions", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rc->setName("rc");
        rc->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();
        auto c = rc->expression();

        auto expr1 = multiplyAdd(a, b, c);

        auto raf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        raf->allocateNow();

        auto rbf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        rbf->allocateNow();

        auto rcf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        rcf->allocateNow();

        auto af = raf->expression();
        auto bf = rbf->expression();
        auto cf = rcf->expression();

        auto expr2 = multiplyAdd(af, bf, cf);

        Register::ValuePtr dest1, dest2;
        context.get()->schedule(Expression::generate(dest1, expr1, context.get()));
        context.get()->schedule(Expression::generate(dest2, expr2, context.get()));

        std::string expected = R"(
            // Int32: a * x + y doesn't have FMA, so should see multiply then add
            v_mul_lo_u32 v6, v0, v1
            v_add_i32 v6, v6, v2

            // Float: a * x + y has FMA
            v_fma_f32 v7, v3, v4, v5
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
    }

    TEST_CASE("BFE Expressions", "[expression][codegen][bfe]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto a = ra->expression();

        // Unsigned (no sign extension)
        auto expr1 = bfe(DataType::UInt8, a, 0, 8); // full width
        auto expr2 = bfe(DataType::UInt8, a, 1, 5); // partial

        // Signed (maybe sign extension)
        auto expr3
            = bfe(DataType::Int8, a, 2, 8); // full width, bfe as unsigned with no sign extension
        auto expr4 = bfe(DataType::Int8, a, 3, 7); // partial, sign extension logic
        auto expr5
            = bfe(DataType::Int8, a, 4, 7); // second partial, hopefully avoid additional load
        auto expr6 = bfe(DataType::Int32, a, 5, 17); // partial but full register,
        auto expr7 = bfe(DataType::Int64, a, 6, 23); // extract to wider output

#define create_output_reg(width, sign_char) \
    std::make_shared<Register::Value>(      \
        context.get(), Register::Type::Vector, DataType::sign_char##Int##width, 1)

        auto dest1 = create_output_reg(8, U);
        auto dest2 = create_output_reg(8, U);
        auto dest3 = create_output_reg(8, );
        auto dest4 = create_output_reg(8, );
        auto dest5 = create_output_reg(8, );
        auto dest6 = create_output_reg(32, );
        auto dest7 = create_output_reg(64, );

#undef create_output_reg

        context.get()->schedule(Expression::generate(dest1, expr1, context.get()));
        context.get()->schedule(Expression::generate(dest2, expr2, context.get()));
        context.get()->schedule(Expression::generate(dest3, expr3, context.get()));
        context.get()->schedule(Expression::generate(dest4, expr4, context.get()));
        context.get()->schedule(Expression::generate(dest5, expr5, context.get()));
        context.get()->schedule(Expression::generate(dest6, expr6, context.get()));
        context.get()->schedule(Expression::generate(dest7, expr7, context.get()));

        std::string expected = R"(
            // Unsigned (no sign extension)
            // expr1
            v_bfe_u32 v1, v0, 0, 8

            // expr2
            v_bfe_u32 v2, v0, 1, 5

            // Signed (maybe sign extension)
            // expr3
            v_bfe_u32 v3, v0, 2, 8

            // expr4
            v_bfe_i32 v4, v0, 3, 7
            v_mov_b32 v5, 255
            v_and_b32 v4, v5, v4

            // expr5
            v_bfe_i32 v5, v0, 4, 7
            v_mov_b32 v6, 255
            v_and_b32 v5, v6, v5

            // expr6
            v_bfe_i32 v6, v0, 5, 17

            // expr7
            v_bfe_i32 v9, v0, 6, 23
            v_ashrrev_i64 v[8:9], 32, v[8:9]
        )";

        // Test extraction of packed data types
        for(auto const& packedDT : {DataType::Halfx2,
                                    DataType::BFloat16x2,
                                    DataType::FP8x4,
                                    DataType::BF8x4,
                                    DataType::FP4x8})
        {
            auto packedDTInfo   = rocRoller::DataTypeInfo::Get(packedDT);
            auto unpackedDT     = packedDTInfo.segmentVariableType.dataType;
            auto unpackedDTInfo = rocRoller::DataTypeInfo::Get(unpackedDT);
            ;

            auto rArg = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, packedDT, 1);
            rArg->setName("rArg");
            rArg->allocateNow();
            auto argExpr = rArg->expression();
            auto expr8   = bfe(unpackedDT, argExpr, 0, unpackedDTInfo.elementBits);
            auto dest8   = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, unpackedDT, packedDTInfo.packing);

            context.get()->schedule(Expression::generate(dest8, expr8, context.get()));
        }

        expected += R"(
             // BitFieldExtract<0,16>(rb: VGPR Value: Halfx2 x 1: v1)
             // Allocated : 2 VGPRs (Value: Half): v3, v2
              v_bfe_u32 v11, v7, 0, 16
              v_bfe_u32 v10, v7, 16, 16

             // BitFieldExtract<0,16>(rb: VGPR Value: BFloat16x2 x 1: v1)
             // Allocated : 2 VGPRs (Value: BFloat16): v3, v2
             v_bfe_u32 v11, v7, 0, 16
             v_bfe_u32 v10, v7, 16, 16

             // BitFieldExtract<0,8>(rb: VGPR Value: FP8x4 x 1: v1)
             // Allocated : 4 VGPRs (Value: FP8): v5, v4, v3, v2
             v_bfe_u32 v13, v7, 0, 8
             v_bfe_u32 v12, v7, 8, 8
             v_bfe_u32 v11, v7, 16, 8
             v_bfe_u32 v10, v7, 24, 8

             // BitFieldExtract<0,8>(rb: VGPR Value: BF8x4 x 1: v1)
             // Allocated : 4 VGPRs (Value: BF8): v5, v4, v3, v2
             v_bfe_u32 v13, v7, 0, 8
             v_bfe_u32 v12, v7, 8, 8
             v_bfe_u32 v11, v7, 16, 8
             v_bfe_u32 v10, v7, 24, 8

             // BitFieldExtract<0,4>(rb: VGPR Value: FP4x8 x 1: v1)
             // Allocated : 8 VGPRs (Value: FP4): v9, v8, v7, v6, v5, v4, v3, v2
             v_bfe_u32 v17, v7, 0, 4
             v_bfe_u32 v16, v7, 4, 4
             v_bfe_u32 v15, v7, 8, 4
             v_bfe_u32 v14, v7, 12, 4
             v_bfe_u32 v13, v7, 16, 4
             v_bfe_u32 v12, v7, 20, 4
             v_bfe_u32 v11, v7, 24, 4
             v_bfe_u32 v10, v7, 28, 4
        )";

        {
            // Extract to a SGPR
            auto rArg = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::Halfx2, 1);
            rArg->setName("rArg");
            rArg->allocateNow();
            auto argExpr = rArg->expression();
            auto expr9   = bfe(DataType::Half, argExpr, 0, 16);
            auto dest9   = std::make_shared<Register::Value>(
                context.get(), Register::Type::Scalar, DataType::Half, 1);
            context.get()->schedule(Expression::generate(dest9, expr9, context.get()));
        }

        expected += R"(
         // BitFieldExtract<0,16>(rArg: VGPR Value: Halfx2 x 1: v7)
         // Allocated : 1 SGPR (Value: Half): s0
         s_bfe_u32 s0, v7, 1048576 //    expr.offset = 0
         )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
    }

    TEST_CASE("Expression comments", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        setComment(expr1, "The Addition");
        appendComment(expr1, " extra comment");
        setComment(expr2, "The Multiplication");

        CHECK(getComment(expr1) == "The Addition extra comment");

        auto expr3 = simplify(expr2);
        CHECK(getComment(expr3) == "The Multiplication");

        Register::ValuePtr dest;
        context.get()->schedule(Expression::generate(dest, expr2, context.get()));

        auto normalized = NormalizedSource(context.output(), true);

        using namespace Catch::Matchers;

        CHECK_THAT(normalized,
                   ContainsSubstring("// Generate {The Multiplication: Multiply(v1:I, {The "
                                     "Addition extra comment: Add(v0:I, v1:I)I})I} into nullptr"));

        CHECK_THAT(normalized, ContainsSubstring("// BEGIN: The Addition extra comment"));
        CHECK_THAT(normalized,
                   ContainsSubstring("// {The Addition extra comment: Add(v0:I, v1:I)I}"));
        CHECK_THAT(normalized, ContainsSubstring("// Allocated : 1 VGPR (Value: Int32): v2"));
        CHECK_THAT(normalized, ContainsSubstring("v_add_i32 v2, v0, v1"));
        CHECK_THAT(normalized, ContainsSubstring("// END: The Addition extra comment"));
        CHECK_THAT(normalized, ContainsSubstring("// BEGIN: The Multiplication"));
        CHECK_THAT(
            normalized,
            ContainsSubstring(
                "// {The Multiplication: Multiply(v1:I, {The Addition extra comment: v2:I})I}"));
        CHECK_THAT(normalized, ContainsSubstring("// Allocated : 1 VGPR (Value: Int32): v3"));
        CHECK_THAT(normalized, ContainsSubstring("v_mul_lo_u32 v3, v1, v2"));
        CHECK_THAT(normalized, ContainsSubstring("// END: The Multiplication"));
        CHECK_THAT(
            normalized,
            ContainsSubstring("// Freeing The Addition extra comment: 1 VGPR (Value: Int32): v2"));

        BENCHMARK("Generate expression")
        {
            context.get()->schedule(Expression::generate(dest, expr2, context.get()));
        };
    }

    TEST_CASE("Expression comment exceptions", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto a = ra->expression();
        CHECK_THROWS_AS(setComment(a, "The a input"), FatalError);
        CHECK_THROWS_AS(appendComment(a, "extra comment"), FatalError);
        CHECK(getComment(a) == "ra");

        Expression::ExpressionPtr expr1;
        CHECK_THROWS_AS(setComment(expr1, "The first expression"), FatalError);
        CHECK_THROWS_AS(appendComment(expr1, "extra"), FatalError);
        CHECK(getComment(expr1) == "");
    }

    TEST_CASE("Expression generation exceptions", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        Register::ValuePtr result;

        context.get()->schedule(context.get()->kernel()->preamble());
        context.get()->schedule(context.get()->kernel()->prolog());

        SECTION("Magic Numbers need generate time operands")
        {
            auto reg = context.get()->kernel()->workitemIndex()[0]->expression();

            auto exp = Expression::magicMultiple(reg);
            CHECK_THROWS(context.get()->schedule(Expression::generate(result, exp, context.get())));

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, Expression::magicShifts(reg), context.get())));

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, Expression::magicShiftAndSign(reg), context.get())));
        }

        SECTION("CommandArgument needs the user args")
        {
            CommandArgumentPtr arg;
            auto               argExp = std::make_shared<Expression::Expression>(arg);
            CHECK_THROWS(
                context.get()->schedule(Expression::generate(result, argExp, context.get())));
        }

        SECTION("More complex expressions")
        {
            Register::ValuePtr nullResult;
            auto               unallocResult = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);
            auto allocResult = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);
            allocResult->allocateNow();

            auto result = GENERATE_COPY(nullResult, unallocResult, allocResult);
            CAPTURE(result);

            auto unallocated = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, unallocated->expression(), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(Expression::generate(
                result, unallocated->expression() + Expression::literal(5), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(Expression::generate(
                result,
                Expression::multiplyHigh(unallocated->expression(), Expression::literal(5)),
                context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(unallocated, unallocated->expression(), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);
        }
    }

    TEST_CASE("Matrix Multiply Expressions", "[expression][codegen][mfma]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 32;
        int N       = 32;
        int K       = 2;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto ic = std::make_shared<Register::Value>(context.get(),
                                                    Register::Type::Accumulator,
                                                    DataType::Float,
                                                    M * N * batches / 64,
                                                    Register::AllocationOptions::FullyContiguous());
        ic->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto C = ic->expression();

        auto expr = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, C));

        context.get()->schedule(
            Expression::generate(ic, expr, context.get())); //Test using input C as dest.

        Register::ValuePtr rc;
        context.get()->schedule(
            Expression::generate(rc, expr, context.get())); //Test using a nullptr as dest.

        CHECK(ic->regType() == Register::Type::Accumulator);
        CHECK(ic->valueCount() == 16);

        CHECK(rc->regType() == Register::Type::Accumulator);
        CHECK(rc->valueCount() == 16);

        auto result = R"(
            v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15] //is matmul
            v_mfma_f32_32x32x2f32 a[16:31], v0, v1, a[0:15] //rc matmul
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expressions reuse input vgprs as output vgprs in arithmetic",
              "[expression][codegen][optimization][fp32]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 16;
        int N       = 16;
        int K       = 4;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Accumulator,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        accumD->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto D = accumD->expression();

        auto mulABExpr
            = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, D));

        context.get()->schedule(
            Expression::generate(accumD, mulABExpr, context.get())); //Test using input D as dest.

        CHECK(accumD->regType() == Register::Type::Accumulator);
        CHECK(accumD->valueCount() == 4);

        auto vecD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        context.get()->schedule(Expression::generate(vecD, D, context.get()));

        auto scaleDExpr = Expression::literal(2.0f) * vecD->expression();
        context.get()->schedule(Expression::generate(vecD, scaleDExpr, context.get()));

        auto vecC
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto addCDExpr = vecC->expression() + vecD->expression();
        context.get()->schedule(Expression::generate(vecD, addCDExpr, context.get()));

        auto result = R"(
            v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[0:3]

            s_nop 10
            v_accvgpr_read v2, a0
            v_accvgpr_read v3, a1
            v_accvgpr_read v4, a2
            v_accvgpr_read v5, a3

            v_mul_f32 v2, 2.00000, v2
            v_mul_f32 v3, 2.00000, v3
            v_mul_f32 v4, 2.00000, v4
            v_mul_f32 v5, 2.00000, v5

            v_add_f32 v2, v6, v2
            v_add_f32 v3, v7, v3
            v_add_f32 v4, v8, v4
            v_add_f32 v5, v9, v5
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expressions reuse input vgprs as output vgprs in arithmetic f16",
              "[expression][codegen][optimization][fp16]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 32;
        int N       = 32;
        int K       = 8;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                M * K / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                K * N / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Accumulator,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        accumD->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto D = accumD->expression();

        auto mulABExpr
            = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, D));

        context.get()->schedule(
            Expression::generate(accumD, mulABExpr, context.get())); //Test using input D as dest.

        CHECK(accumD->regType() == Register::Type::Accumulator);
        CHECK(accumD->valueCount() == 16);

        auto vecD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());

        auto vecC
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Half,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto scaleDExpr = Expression::literal(2.0, DataType::Half) * vecD->expression();
        auto addCDExpr  = vecC->expression() + vecD->expression();

        context.get()->schedule(Expression::generate(vecD, D, context.get()));
        context.get()->schedule(Expression::generate(vecD, scaleDExpr, context.get()));
        context.get()->schedule(Expression::generate(vecD, addCDExpr, context.get()));

        auto X = std::make_shared<Register::Value>(context.get(),
                                                   Register::Type::Vector,
                                                   DataType::Halfx2,
                                                   M * K / 64 / 2,
                                                   Register::AllocationOptions::FullyContiguous());
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(context.get(),
                                                   Register::Type::Vector,
                                                   DataType::Half,
                                                   M * K / 64,
                                                   Register::AllocationOptions::FullyContiguous());
        Y->allocateNow();

        auto addXYExpr = X->expression() + Y->expression();
        context.get()->schedule(Expression::generate(Y, addXYExpr, context.get()));

        // TODO If operand being converted is a literal, do one conversion only.
        auto result = R"(
            // A is in v[0:1], B is in v[2:3], C is in v[4:19], D is in a[0:15]

            // Result R will end up in v[20:35].  Steps are:
            // R <- D
            // R <- alpha * R
            // R <- R + C

            v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]

            s_nop 0xf
            s_nop 2
            v_accvgpr_read v20, a0
            v_accvgpr_read v21, a1
            v_accvgpr_read v22, a2
            v_accvgpr_read v23, a3
            v_accvgpr_read v24, a4
            v_accvgpr_read v25, a5
            v_accvgpr_read v26, a6
            v_accvgpr_read v27, a7
            v_accvgpr_read v28, a8
            v_accvgpr_read v29, a9
            v_accvgpr_read v30, a10
            v_accvgpr_read v31, a11
            v_accvgpr_read v32, a12
            v_accvgpr_read v33, a13
            v_accvgpr_read v34, a14
            v_accvgpr_read v35, a15

            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v20, v36, v20
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v21, v36, v21
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v22, v36, v22
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v23, v36, v23
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v24, v36, v24
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v25, v36, v25
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v26, v36, v26
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v27, v36, v27
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v28, v36, v28
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v29, v36, v29
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v30, v36, v30
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v31, v36, v31
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v32, v36, v32
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v33, v36, v33
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v34, v36, v34
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v35, v36, v35

            v_cvt_f32_f16 v36, v4
            v_add_f32 v20, v36, v20
            v_cvt_f32_f16 v36, v5
            v_add_f32 v21, v36, v21
            v_cvt_f32_f16 v36, v6
            v_add_f32 v22, v36, v22
            v_cvt_f32_f16 v36, v7
            v_add_f32 v23, v36, v23
            v_cvt_f32_f16 v36, v8
            v_add_f32 v24, v36, v24
            v_cvt_f32_f16 v36, v9
            v_add_f32 v25, v36, v25
            v_cvt_f32_f16 v36, v10
            v_add_f32 v26, v36, v26
            v_cvt_f32_f16 v36, v11
            v_add_f32 v27, v36, v27
            v_cvt_f32_f16 v36, v12
            v_add_f32 v28, v36, v28
            v_cvt_f32_f16 v36, v13
            v_add_f32 v29, v36, v29
            v_cvt_f32_f16 v36, v14
            v_add_f32 v30, v36, v30
            v_cvt_f32_f16 v36, v15
            v_add_f32 v31, v36, v31
            v_cvt_f32_f16 v36, v16
            v_add_f32 v32, v36, v32
            v_cvt_f32_f16 v36, v17
            v_add_f32 v33, v36, v33
            v_cvt_f32_f16 v36, v18
            v_add_f32 v34, v36, v34
            v_cvt_f32_f16 v36, v19
            v_add_f32 v35, v36, v35

            // X is v[36:37]:2xH and Y is v[38:41]:H (and Z is same as Y)
            // Then Y <- X + Y will be: Add(v[36:37]:2xH, v[38:41]:H)
            v_mov_b32 v42, 65535
            v_and_b32 v44, v42, v36
            v_lshrrev_b32 v45, 16, v36
            v_add_f16 v38, v44, v38
            v_add_f16 v39, v45, v39
            v_mov_b32 v43, 65535
            v_and_b32 v44, v43, v37
            v_lshrrev_b32 v45, 16, v37
            v_add_f16 v40, v44, v40
            v_add_f16 v41, v45, v41
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE(
        "Expressions reuse input vgprs as output vgprs in arithmetic f16 with smaller packing",
        "[expression][codegen][optimization][future][fp16]")
    {
        auto context = TestContext::ForDefaultTarget();
        int  M       = 32;
        int  K       = 8;

        auto X = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Halfx2, M * K / 64 / 2);
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Half, M * K / 64);
        Y->allocateNow();

        // Since we are asking the result to be stored into X, we
        // currently get a failure.

        // TODO See the "Destination/result packing mismatch" assertion
        // in Expression_generate.cpp.
        auto addXYExpr = X->expression() + Y->expression();
        CHECK_THROWS_AS(context.get()->schedule(Expression::generate(X, addXYExpr, context.get())),
                        FatalError);

        // The above should be possible: Y should be packed, and then
        // the v_pk_add_f16 instructions called.
    }

    TEST_CASE("Expression result types", "[expression]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto vgprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Float, 1)
                             ->expression();
        auto vgprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Double, 1)
                              ->expression();
        auto vgprInt32 = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Int32, 1)
                             ->expression();
        auto vgprInt64 = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Int64, 1)
                             ->expression();
        auto vgprUInt32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::UInt32, 1)
                              ->expression();
        auto vgprUInt64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::UInt64, 1)
                              ->expression();
        auto vgprHalf
            = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Half, 1)
                  ->expression();
        auto vgprHalfx2 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Halfx2, 1)
                              ->expression();
        auto vgprBool32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Bool32, 1)
                              ->expression();
        auto vgprBool
            = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Bool, 1)
                  ->expression();

        auto sgprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Float, 1)
                             ->expression();
        auto sgprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Double, 1)
                              ->expression();
        auto sgprInt32 = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Int32, 1)
                             ->expression();
        auto sgprInt64 = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Int64, 1)
                             ->expression();
        auto sgprUInt32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::UInt32, 1)
                              ->expression();
        auto sgprUInt64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::UInt64, 1)
                              ->expression();
        auto sgprHalf
            = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::Half, 1)
                  ->expression();
        auto sgprHalfx2 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Halfx2, 1)
                              ->expression();
        auto sgprBool64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Bool64, 1)
                              ->expression();
        auto sgprBool32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Bool32, 1)
                              ->expression();
        auto sgprBool
            = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::Bool, 1)
                  ->expression();
        auto sgprWavefrontSized
            = Register::Value::Placeholder(context.get(),
                                           Register::Type::Scalar,
                                           context.get()->kernel()->wavefront_size() == 64
                                               ? DataType::Bool64
                                               : DataType::Bool32,
                                           1)
                  ->expression();

        auto agprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Accumulator, DataType::Float, 1)
                             ->expression();
        auto agprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Accumulator, DataType::Double, 1)
                              ->expression();

        auto litInt32  = Expression::literal<int32_t>(5);
        auto litInt64  = Expression::literal<int64_t>(5);
        auto litFloat  = Expression::literal(5.0f);
        auto litDouble = Expression::literal(5.0);

        Expression::ResultType rVgprFloat{Register::Type::Vector, DataType::Float};
        Expression::ResultType rVgprDouble{Register::Type::Vector, DataType::Double};
        Expression::ResultType rVgprInt32{Register::Type::Vector, DataType::Int32};
        Expression::ResultType rVgprInt64{Register::Type::Vector, DataType::Int64};
        Expression::ResultType rVgprUInt32{Register::Type::Vector, DataType::UInt32};
        Expression::ResultType rVgprUInt64{Register::Type::Vector, DataType::UInt64};
        Expression::ResultType rVgprHalf{Register::Type::Vector, DataType::Half};
        Expression::ResultType rVgprHalfx2{Register::Type::Vector, DataType::Halfx2};
        Expression::ResultType rVgprBool32{Register::Type::Vector, DataType::Bool32};

        Expression::ResultType rSgprFloat{Register::Type::Scalar, DataType::Float};
        Expression::ResultType rSgprDouble{Register::Type::Scalar, DataType::Double};
        Expression::ResultType rSgprInt32{Register::Type::Scalar, DataType::Int32};
        Expression::ResultType rSgprInt64{Register::Type::Scalar, DataType::Int64};
        Expression::ResultType rSgprUInt32{Register::Type::Scalar, DataType::UInt32};
        Expression::ResultType rSgprUInt64{Register::Type::Scalar, DataType::UInt64};
        Expression::ResultType rSgprHalf{Register::Type::Scalar, DataType::Half};
        Expression::ResultType rSgprHalfx2{Register::Type::Scalar, DataType::Halfx2};
        Expression::ResultType rSgprBool32{Register::Type::Scalar, DataType::Bool32};
        Expression::ResultType rSgprBool64{Register::Type::Scalar, DataType::Bool64};
        Expression::ResultType rSgprBool{Register::Type::Scalar, DataType::Bool};
        Expression::ResultType rSgprWavefrontSized{
            Register::Type::Scalar,
            context.get()->kernel()->wavefront_size() == 64 ? DataType::Bool64 : DataType::Bool32};

        Expression::ResultType rVCC{Register::Type::VCC, DataType::Bool32};
        Expression::ResultType rSCC{Register::Type::SCC, DataType::Bool};

        Expression::ResultType rAgprFloat{Register::Type::Accumulator, DataType::Float};
        Expression::ResultType rAgprDouble{Register::Type::Accumulator, DataType::Double};

        SECTION("Value expressions")
        {
            CHECK(rSgprInt64 == resultType(sgprInt64));

            CHECK(rVgprInt32 == resultType(vgprInt32));
            CHECK(rVgprInt64 == resultType(vgprInt64));
            CHECK(rVgprFloat == resultType(vgprFloat));
            CHECK(rSgprFloat == resultType(sgprFloat));
            CHECK(rVgprDouble == resultType(vgprDouble));
            CHECK(rSgprDouble == resultType(sgprDouble));
            CHECK(rAgprDouble == resultType(agprDouble));
            CHECK(rAgprDouble == resultType(agprDouble));
        }

        SECTION("Binary expressions")
        {
            CHECK(rVgprInt32 == resultType(vgprInt32 + vgprInt32));
            CHECK(rVgprInt32 == resultType(vgprInt32 + sgprInt32));
            CHECK(rVgprInt32 == resultType(sgprInt32 - vgprInt32));
            CHECK(rSgprInt32 == resultType(sgprInt32 * sgprInt32));

            CHECK(rVgprInt64 == resultType(vgprInt64 + vgprInt32));
            CHECK(rVgprInt64 == resultType(vgprInt32 + vgprInt64));
            CHECK(rVgprInt64 == resultType(vgprInt64 + vgprInt64));

            CHECK(rVgprFloat == resultType(vgprFloat + vgprFloat));
            CHECK(rVgprFloat == resultType(vgprFloat - sgprFloat));
            CHECK(rVgprFloat == resultType(litFloat * vgprFloat));
            CHECK(rVgprFloat == resultType(vgprFloat * litFloat));

            CHECK(rSgprInt32 == resultType(sgprInt32 + sgprInt32));
            CHECK(rSgprInt32 == resultType(sgprInt32 + litInt32));
            CHECK(rSgprInt32 == resultType(litInt32 + sgprInt32));
            CHECK(rSgprInt64 == resultType(litInt32 + sgprInt64));
            CHECK(rSgprInt64 == resultType(sgprInt64 + litInt32));
            CHECK(rSgprInt64 == resultType(sgprInt64 + sgprInt32));

            CHECK(rSgprWavefrontSized == resultType(vgprFloat > vgprFloat));
            CHECK_THROWS_AS(resultType((vgprFloat > vgprFloat) && vgprBool), FatalError);
            CHECK(rSgprWavefrontSized == resultType(sgprFloat < vgprFloat));
            CHECK(rSgprWavefrontSized == resultType(sgprFloat < vgprFloat && sgprBool));
            CHECK(rSgprWavefrontSized == resultType(sgprDouble <= vgprDouble));
            CHECK(rSgprWavefrontSized == resultType(sgprInt32 <= vgprInt32));
            CHECK(rSgprWavefrontSized == resultType(sgprInt32 <= vgprInt32 || sgprBool));
            CHECK(rSgprWavefrontSized == resultType(litInt32 > vgprInt64));
            CHECK(rSgprWavefrontSized == resultType(litInt32 > vgprInt64 || sgprBool));
            CHECK(rSgprWavefrontSized
                  == resultType(litInt32 > vgprInt64 || Expression::literal(false)));
            CHECK(rSgprBool == resultType(litInt32 <= sgprInt64));
            CHECK(rSgprBool == resultType(sgprInt32 >= litInt32));
        }

        CHECK_THROWS(resultType(sgprDouble <= vgprFloat));
        CHECK_THROWS(resultType(vgprInt32 > vgprFloat));

        SECTION("Arithmetic unary ops")
        {
            // auto ops = ;
            auto op = GENERATE_COPY(
                from_range(std::to_array({Expression::operator-, // cppcheck-suppress syntaxError
                                          Expression::operator~,
                                          Expression::magicMultiple})));

            CAPTURE(op(vgprFloat));
            CHECK(rVgprFloat == resultType(op(vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32)));
            CHECK(rVgprUInt64 == resultType(op(vgprUInt64)));
            CHECK(rVgprHalf == resultType(op(vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32)));
            CHECK(rSgprUInt64 == resultType(op(sgprUInt64)));
            CHECK(rSgprHalf == resultType(op(sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32)));
        }

        SECTION("Magic shifts")
        {
            auto op = Expression::magicShifts;
            CHECK(rVgprInt32 == resultType(op(vgprFloat)));
            CHECK(rVgprInt32 == resultType(op(vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32)));
            CHECK(rVgprInt32 == resultType(op(vgprInt64)));
            CHECK(rVgprInt32 == resultType(op(vgprUInt32)));
            CHECK(rVgprInt32 == resultType(op(vgprUInt64)));
            CHECK(rVgprInt32 == resultType(op(vgprHalf)));
            CHECK(rVgprInt32 == resultType(op(vgprHalfx2)));
            CHECK(rVgprInt32 == resultType(op(vgprBool32)));

            CHECK(rSgprInt32 == resultType(op(sgprFloat)));
            CHECK(rSgprInt32 == resultType(op(sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32)));
            CHECK(rSgprInt32 == resultType(op(sgprInt64)));
            CHECK(rSgprInt32 == resultType(op(sgprUInt32)));
            CHECK(rSgprInt32 == resultType(op(sgprUInt64)));
            CHECK(rSgprInt32 == resultType(op(sgprHalf)));
            CHECK(rSgprInt32 == resultType(op(sgprHalfx2)));
            CHECK(rSgprInt32 == resultType(op(sgprBool32)));
        }

        SECTION("Magic shiftAndSign")
        {
            auto op = Expression::magicShiftAndSign;
            CHECK(rVgprUInt32 == resultType(op(vgprFloat)));
            CHECK(rVgprUInt32 == resultType(op(vgprDouble)));
            CHECK(rVgprUInt32 == resultType(op(vgprInt32)));
            CHECK(rVgprUInt32 == resultType(op(vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprHalf)));
            CHECK(rVgprUInt32 == resultType(op(vgprHalfx2)));
            CHECK(rVgprUInt32 == resultType(op(vgprBool32)));

            CHECK(rSgprUInt32 == resultType(op(sgprFloat)));
            CHECK(rSgprUInt32 == resultType(op(sgprDouble)));
            CHECK(rSgprUInt32 == resultType(op(sgprInt32)));
            CHECK(rSgprUInt32 == resultType(op(sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprHalf)));
            CHECK(rSgprUInt32 == resultType(op(sgprHalfx2)));
            CHECK(rSgprUInt32 == resultType(op(sgprBool32)));
        }

        SECTION("Comparisons")
        {
            auto op = GENERATE(Expression::operator>,
                               Expression::operator>=,
                               Expression::operator<,
                               Expression::operator<=,
                               Expression::operator==);

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rSgprWavefrontSized == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprUInt64, vgprUInt64)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprBool32, vgprBool32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprBool, vgprBool)));

            CHECK(rSgprBool == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprBool == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprBool == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprBool == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprBool == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprBool == resultType(op(sgprUInt64, sgprUInt64)));
            CHECK(rSgprBool == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprBool == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool == resultType(op(sgprBool32, sgprBool32)));
            CHECK(rSgprBool == resultType(op(sgprBool, sgprBool)));
        }

        SECTION("Arithmetic binary")
        {
            auto op = GENERATE(from_range(std::to_array({Expression::operator+,
                                                         Expression::operator-,
                                                         Expression::operator*,
                                                         Expression::operator/,
                                                         Expression::operator%,
                                                         Expression::operator<<,
                                                         Expression::operator>>,
                                                         Expression::operator&,
                                                         Expression::arithmeticShiftR})));

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rVgprUInt64 == resultType(op(vgprUInt64, vgprUInt64)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprUInt64 == resultType(op(sgprUInt64, sgprUInt64)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32)));
        }

        SECTION("Logical")
        {
            auto op = GENERATE(Expression::operator&&, Expression::operator||);

            CAPTURE(op(sgprBool64, sgprBool64));

            CHECK_THROWS(resultType(op(vgprFloat, vgprFloat)));
            CHECK_THROWS(resultType(op(vgprDouble, vgprDouble)));
            CHECK_THROWS(resultType(op(vgprInt32, vgprInt32)));
            CHECK_THROWS(resultType(op(vgprInt64, vgprInt64)));
            CHECK_THROWS(resultType(op(vgprUInt32, vgprUInt32)));
            CHECK_THROWS(resultType(op(vgprUInt64, vgprUInt64)));
            CHECK_THROWS(resultType(op(vgprHalf, vgprHalf)));
            CHECK_THROWS(resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK_THROWS(resultType(op(vgprBool32, vgprBool32)));
            CHECK_THROWS(resultType(op(vgprBool, vgprBool)));

            CHECK_THROWS(resultType(op(sgprFloat, sgprFloat)));
            CHECK_THROWS(resultType(op(sgprDouble, sgprDouble)));
            CHECK_THROWS(resultType(op(sgprInt32, sgprInt32)));
            CHECK_THROWS(resultType(op(sgprInt64, sgprInt64)));
            CHECK_THROWS(resultType(op(sgprUInt32, sgprUInt32)));

            CHECK(rSgprBool64 == resultType(op(sgprBool64, sgprBool64)));
            CHECK(rSgprBool64 == resultType(op(sgprBool64, sgprBool)));
            CHECK_THROWS(resultType(op(sgprHalf, sgprHalf)));
            CHECK_THROWS(resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool)));
            CHECK(rSgprBool32 == resultType(op(sgprBool, sgprBool32)));
            CHECK(rSgprBool == resultType(op(sgprBool, sgprBool)));
        }

        SECTION("Bitwise binary")
        {

            auto op = GENERATE(from_range(std::to_array({Expression::operator<<,
                                    Expression::logicalShiftR,
                                    Expression::operator&,
                                    Expression::operator^,
                                    Expression::operator|})));

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32)));
        }

        SECTION("Arithmetic ternary")
        {
            auto op = GENERATE_COPY(
                Expression::multiplyAdd, Expression::addShiftL, Expression::shiftLAdd);

            CAPTURE(op(vgprFloat, vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32, vgprBool32)));
            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32, sgprBool32)));
        }

        SECTION("Conditional")
        {
            auto op = Expression::conditional;
            CHECK(rVgprFloat == resultType(op(sgprBool, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(sgprBool, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(sgprBool, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(sgprBool, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(sgprBool, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(sgprBool, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(sgprBool, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(sgprBool, vgprBool32, vgprBool32)));

            CHECK(rVgprFloat == resultType(op(vgprBool, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprBool, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprBool, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprBool, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprBool, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprBool, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprBool, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool, vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprBool, sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprBool, sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprBool, sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprBool, sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprBool, sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprBool, sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprBool, sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool, sgprBool32, sgprBool32)));

            CHECK(rVgprFloat == resultType(op(vgprBool, sgprFloat, sgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprBool, sgprDouble, sgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprBool, sgprInt32, sgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprBool, sgprInt64, sgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprBool, sgprUInt32, sgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprBool, sgprHalf, sgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprBool, sgprHalfx2, sgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool, sgprBool32, sgprBool32)));
        }
    }

    TEST_CASE("Expression evaluate", "[expression]")
    {
        SECTION("No arguments")
        {
            auto a = std::make_shared<Expression::Expression>(1.0);
            auto b = std::make_shared<Expression::Expression>(2.0);

            auto expr1 = a + b;
            auto expr2 = b * expr1;

            auto expectedTimes = Expression::EvaluationTimes::All();
            CHECK(expectedTimes == Expression::evaluationTimes(expr2));

            CHECK(Expression::canEvaluateTo(3.0, expr1));
            CHECK(Expression::canEvaluateTo(6.0, expr2));
            CHECK(3.0 == std::get<double>(evaluate(expr1)));
            CHECK(6.0 == std::get<double>(evaluate(expr2)));
        }

        SECTION("Arguments")
        {
            VariableType doubleVal{DataType::Double, PointerType::Value};
            auto         ca = std::make_shared<CommandArgument>(nullptr, doubleVal, 0);
            auto         cb = std::make_shared<CommandArgument>(nullptr, doubleVal, 8);

            auto a = std::make_shared<Expression::Expression>(ca);
            auto b = std::make_shared<Expression::Expression>(cb);

            auto expr1 = a + b;
            auto expr2 = b * expr1;
            auto expr3 = -expr2;

            struct
            {
                double a = 1.0;
                double b = 2.0;
            } args;
            RuntimeArguments runtimeArgs((uint8_t*)&args, sizeof(args));

            Expression::ResultType expected{Register::Type::Literal, DataType::Double};
            CHECK(expected == resultType(expr2));
            CHECK(6.0 == std::get<double>(evaluate(expr2, runtimeArgs)));

            args.a = 2.0;
            CHECK(8.0 == std::get<double>(evaluate(expr2, runtimeArgs)));
            CHECK(-8.0 == std::get<double>(evaluate(expr3, runtimeArgs)));

            args.b = 1.5;
            CHECK(5.25 == std::get<double>(evaluate(expr2, runtimeArgs)));

            CHECK_FALSE(Expression::canEvaluateTo(5.25, expr2));
            // Don't send in the runtimeArgs, can't evaluate the arguments.
            CHECK_THROWS_AS(evaluate(expr2), std::runtime_error);

            Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelLaunch};
            CHECK(expectedTimes == Expression::evaluationTimes(expr2));
        }
    }

    TEST_CASE("Expression test evaluate mixed types", "[expression]")
    {
        using Expression::literal;
        auto one          = literal(1.0);
        auto two          = literal(2.0f);
        auto twoPoint5    = literal(2.5f);
        auto five         = literal(5);
        auto seven        = literal(7.0);
        auto eightPoint75 = literal(8.75);

        auto ptrNull = std::make_shared<Expression::Expression>((float*)nullptr);

        float x        = 3.0f;
        auto  ptrValid = std::make_shared<Expression::Expression>(&x);

        double y              = 9.0;
        auto   ptrDoubleValid = std::make_shared<Expression::Expression>(&y);

        // double + float -> double
        auto expr1 = one + two;
        // float * double -> double
        auto exprSix = two * expr1;

        // double - int -> double
        auto exprOneBad = exprSix - five;
        auto exprOne    = exprSix - convert(DataType::Double, five);

        // float + int -> float
        auto exprSeven = two + convert(DataType::Float, five);

        CHECK(6.0 == std::get<double>(evaluate(exprSix)));
        CHECK_THROWS_AS(evaluate(exprOneBad), FatalError);
        CHECK(1.0 == std::get<double>(evaluate(exprOne)));
        CHECK(7.0f == std::get<float>(evaluate(exprSeven)));

        auto twoDouble = convert(DataType::Double, two);
        CHECK(2.0 == std::get<double>(evaluate(twoDouble)));

        auto twoInt = convert(DataType::Int32, twoPoint5);
        CHECK(2 == std::get<int>(evaluate(twoInt)));

        auto fiveDoubleBad = seven - twoInt;
        auto fiveDouble    = seven - convert(DataType::Double, twoInt);
        CHECK_THROWS_AS(std::get<double>(evaluate(fiveDoubleBad)), FatalError);
        CHECK(5.0 == std::get<double>(evaluate(fiveDouble)));

        auto minusThree64 = convert(DataType::Int64, twoInt - five);
        CHECK(-3l == std::get<int64_t>(evaluate(minusThree64)));

        auto minusThreeU64 = convert(DataType::UInt64, twoInt - five);
        CHECK(18446744073709551613ul == std::get<uint64_t>(evaluate(minusThreeU64)));

        auto eight75Half = convert(DataType::Half, eightPoint75);
        CHECK(Half(8.75) == std::get<Half>(evaluate(eight75Half)));

        Expression::ResultType litDouble{Register::Type::Literal, DataType::Double};
        Expression::ResultType litFloat{Register::Type::Literal, DataType::Float};
        Expression::ResultType litBool{Register::Type::Literal, DataType::Bool};

        CHECK(litDouble == resultType(exprSix));
        // Result type not (yet?) defined for mixed integral/floating point types.

        CHECK(true == std::get<bool>(evaluate(exprSix > exprOne)));
        CHECK(true == std::get<bool>(evaluate(exprSix >= exprOne)));
        CHECK(false == std::get<bool>(evaluate(exprSix < exprOne)));
        CHECK(false == std::get<bool>(evaluate(exprSix <= exprOne)));
        CHECK(true == std::get<bool>(evaluate(exprSix != exprOne)));

        CHECK(litBool == resultType(one > seven));

        CHECK(true == std::get<bool>(evaluate(exprSix < exprSeven)));
        CHECK(true == std::get<bool>(evaluate(exprSix <= exprSeven)));
        CHECK(false == std::get<bool>(evaluate(exprSix > exprSeven)));
        CHECK(false == std::get<bool>(evaluate(exprSix >= exprSeven)));

        CHECK(true == std::get<bool>(evaluate(one <= exprOne)));
        CHECK(true == std::get<bool>(evaluate(one == exprOne)));
        CHECK(true == std::get<bool>(evaluate(one >= exprOne)));
        CHECK(false == std::get<bool>(evaluate(one != exprOne)));

        auto trueExp = std::make_shared<Expression::Expression>(true);
        CHECK(true == std::get<bool>(evaluate(trueExp == (one >= exprOne))));
        CHECK(false == std::get<bool>(evaluate(trueExp == (one < exprOne))));

        // Pointer + double -> error.
        {
            auto exprThrow = ptrValid + exprOne;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        // Pointer * int -> error.
        {
            auto exprThrow = ptrValid * five;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
        }

        // Pointer + pointer -> error
        {
            auto exprThrow = ptrValid + ptrDoubleValid;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        // (float *) -  (double *) -> error
        {
            auto exprThrow = ptrValid - ptrDoubleValid;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        {
            auto exprThrow = ptrNull + five;
            // nullptr + int -> error;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = -ptrNull;
            // -pointer -> error;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = five + ptrNull;
            // Nullptr + int -> error;
            CHECK_THROWS_AS(evaluate(exprThrow), std::runtime_error);
        }

        auto   exprXPlus5          = ptrValid + five;
        float* dontDereferenceThis = std::get<float*>(evaluate(exprXPlus5));
        auto   ptrDifference       = dontDereferenceThis - (&x);
        CHECK(5 == ptrDifference);

        auto expr10PlusX    = five + exprXPlus5;
        dontDereferenceThis = std::get<float*>(evaluate(expr10PlusX));
        ptrDifference       = dontDereferenceThis - (&x);
        CHECK(10 == ptrDifference);

        auto expr5PtrDiff = expr10PlusX - exprXPlus5;
        CHECK(5 == std::get<int64_t>(evaluate(expr5PtrDiff)));

        CHECK(true == std::get<bool>(evaluate(expr10PlusX > ptrValid)));
        CHECK(false == std::get<bool>(evaluate(expr10PlusX < ptrValid)));
    }

    TEST_CASE("Expression equality", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * a;
        auto expr3 = expr1 == expr2;

        Register::ValuePtr destReg;
        context.get()->schedule(Expression::generate(destReg, expr3, context.get()));

        auto result = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v0
            v_cmp_eq_i32 s[0:1], v2, v3
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("64bit Addition should use SGPR carry when possible", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        SECTION("64bit Addition can use SGPR carry for literal 64 bit input that has "
                "inline-constant lsb and msb",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto b = Expression::literal(4294967297UL); // UInt32 max + 2

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_add_co_u32 v2, s[0:1], 1, v0
                v_addc_co_u32 v3, s[0:1], 1, v1, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should use SGPR carry for literal 64 bit input that has large msb",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto b = Expression::literal(549755813888UL); // 2^(32+7)

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_mov_b32 v2, 128
                v_add_co_u32 v4, s[0:1], 0, v0
                v_addc_co_u32 v5, s[0:1], v2, v1, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should use VCC carry for literal 64 bit input that is not an "
                "inline-constant",
                "[expression][codegen]")
        {

            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto b = Expression::literal(100); // > 64

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_add_co_u32 v2, vcc, 100, v0
                v_addc_co_u32 v3, vcc, 0, v1, vcc
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should not add two literals for 32bit register input",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto b = Expression::literal(137438953473UL); // 2^(32+5) + 1

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_mov_b32 v1, 0
                v_add_co_u32 v2, s[0:1], 1, v0
                v_addc_co_u32 v3, s[0:1], 32, v1, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should use VCC carry for literal 32 bit input that is an "
                "inline-constant",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto b = Expression::literal(5);

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_add_co_u32 v2, s[0:1], 5, v0
                v_addc_co_u32 v3, s[0:1], 0, v1, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should use SGPR carry for non-literal input",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto rb = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            rb->setName("b");
            rb->allocateNow();
            auto b = rb->expression();

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_add_co_u32 v4, s[0:1], v0, v2
                v_addc_co_u32 v5, s[0:1], v1, v3, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }

        SECTION("64bit Addition should use SGPR carry for non-literal 32bit input",
                "[expression][codegen]")
        {
            auto ra = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt64, 1);
            ra->setName("a");
            ra->allocateNow();
            auto a = ra->expression();

            auto rb = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::UInt32, 1);
            rb->setName("b");
            rb->allocateNow();
            auto b = rb->expression();

            auto expr1 = a + b;

            Register::ValuePtr destReg;
            context.get()->schedule(Expression::generate(destReg, expr1, context.get()));

            auto result = R"(
                v_add_co_u32 v4, s[0:1], v0, v2
                v_addc_co_u32 v5, s[0:1], 0, v1, s[0:1]
            )";

            CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
        }
    }

    TEST_CASE("Expression evaluate comparisons", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Double, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Double, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_gt = a > b;
        auto vals_lt = a < b;
        auto vals_ge = a >= b;
        auto vals_le = a <= b;
        auto vals_eq = a == b;

        auto expr_gt = a > (a + b);
        auto expr_lt = a < (a + b);
        auto expr_ge = a >= (a + b);
        auto expr_le = a <= (a + b);
        auto expr_eq = a == (a + b);

        auto aVal = GENERATE(from_range(TestValues::doubleValues));
        auto bVal = GENERATE(from_range(TestValues::doubleValues));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<bool>(evaluate(vals_gt, args)) == (aVal > bVal));
        CHECK(std::get<bool>(evaluate(vals_lt, args)) == (aVal < bVal));
        CHECK(std::get<bool>(evaluate(vals_ge, args)) == (aVal >= bVal));
        CHECK(std::get<bool>(evaluate(vals_le, args)) == (aVal <= bVal));
        CHECK(std::get<bool>(evaluate(vals_eq, args)) == (aVal == bVal));

        CHECK(std::get<bool>(evaluate(expr_gt, args)) == (aVal > (aVal + bVal)));
        CHECK(std::get<bool>(evaluate(expr_lt, args)) == (aVal < (aVal + bVal)));
        CHECK(std::get<bool>(evaluate(expr_ge, args)) == (aVal >= (aVal + bVal)));
        CHECK(std::get<bool>(evaluate(expr_le, args)) == (aVal <= (aVal + bVal)));
        CHECK(std::get<bool>(evaluate(expr_eq, args)) == (aVal == (aVal + bVal)));
    }

    TEST_CASE("Expression evaluate logical", "[expression]")
    {
        using Expression::literal;
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = ca->expression();
        auto b = cb->expression();

        auto zero = literal(0);

        auto vals_negate        = logicalNot(a != zero);
        auto vals_double_negate = logicalNot(logicalNot(a != zero));
        auto vals_and           = (a != zero) && (b != zero);
        auto vals_or            = (a != zero) || (b != zero);

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        DYNAMIC_SECTION(concatenate(aVal, ", ", bVal))
        {
            CAPTURE(aVal, bVal);
            KernelArguments runtimeArgs;
            runtimeArgs.append("a", aVal);
            runtimeArgs.append("b", bVal);
            auto args = runtimeArgs.runtimeArguments();

            CHECK(std::get<bool>(evaluate(vals_negate, args)) == (!aVal));
            CHECK(std::get<bool>(evaluate(vals_double_negate, args)) == (!!aVal));
            CHECK(std::get<bool>(evaluate(vals_and, args)) == (aVal && bVal));
            CHECK(std::get<bool>(evaluate(vals_or, args)) == (aVal || bVal));
        }
    }

    TEST_CASE("Expression evaluate shifts", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_shiftL       = a << b;
        auto vals_shiftR       = logicalShiftR(a, b);
        auto vals_signedShiftR = a >> b;

        auto expr_shiftL       = (a + b) << b;
        auto expr_shiftR       = logicalShiftR(a + b, b);
        auto expr_signedShiftR = (a + b) >> b;

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::shiftValues));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(evaluate(vals_shiftL, args)) == (aVal << bVal));

        CHECK(std::get<int>(evaluate(vals_shiftR, args))
              == (static_cast<unsigned int>(aVal) >> bVal));
        CHECK(std::get<int>(evaluate(vals_signedShiftR, args)) == (aVal >> bVal));

        CHECK(std::get<int>(evaluate(expr_shiftL, args)) == ((aVal + bVal) << bVal));
        CHECK(std::get<int>(evaluate(expr_shiftR, args))
              == (static_cast<unsigned int>(aVal + bVal) >> bVal));
        CHECK(std::get<int>(evaluate(expr_signedShiftR, args)) == ((aVal + bVal) >> bVal));
    }

    TEST_CASE("Expression evaluate conditional operator", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_shiftL = conditional(a >= b, a, b);

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        // At kernel launch time
        CHECK(std::get<int>(evaluate(vals_shiftL, args)) == (aVal >= bVal ? aVal : bVal));

        // At translate time
        auto a_static = std::make_shared<Expression::Expression>(aVal);
        auto b_static = std::make_shared<Expression::Expression>(bVal);
        CHECK(std::get<int>(evaluate(conditional(a_static >= b_static, a_static, b_static)))
              == (aVal >= bVal ? aVal : bVal));
    }

    TEST_CASE("Expression evaluate bitwise ops", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_and    = a & b;
        auto vals_or     = a | b;
        auto vals_negate = ~a;

        auto expr_and = (a + b) & b;
        auto expr_or  = (a + b) | b;

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(evaluate(vals_and, args)) == (aVal & bVal));
        CHECK(std::get<int>(evaluate(vals_or, args)) == (aVal | bVal));
        CHECK(std::get<int>(evaluate(vals_negate, args)) == (~aVal));

        CHECK(std::get<int>(evaluate(expr_and, args)) == ((aVal + bVal) & bVal));
        CHECK(std::get<int>(evaluate(expr_or, args)) == ((aVal + bVal) | bVal));
    }

    TEST_CASE("Expression evaluate multiplyHigh signed", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = multiplyHigh(a, b);

        auto expr2 = multiplyHigh(a + b, b);

        std::vector<int> a_values = {-21474836,
                                     -146000,
                                     -1,
                                     0,
                                     1,
                                     2,
                                     4,
                                     5,
                                     7,
                                     12,
                                     19,
                                     33,
                                     63,
                                     906,
                                     3017123,
                                     800000,
                                     1234456,
                                     4022112};

        auto aVal = GENERATE_COPY(from_range(a_values));
        auto bVal = GENERATE_COPY(from_range(a_values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(evaluate(expr1, args)) == ((aVal * (int64_t)bVal) >> 32));

        CHECK(std::get<int>(evaluate(expr2, args)) == (((aVal + bVal) * (int64_t)bVal) >> 32));
    }

    TEST_CASE("Expression evaluate multiplyHigh unsigned", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = multiplyHigh(a, b);

        auto expr2 = multiplyHigh(a + b, b);

        std::vector<unsigned int> a_values = {
            0, 1, 2, 4, 5, 7, 12, 19, 33, 63, 906, 3017123, 800000, 1234456, 4022112,
            //2863311531u // Can cause overflow
        };
        auto aVal = GENERATE_COPY(from_range(a_values));
        auto bVal = GENERATE_COPY(from_range(a_values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<unsigned int>(evaluate(expr1, args)) == ((aVal * (uint64_t)bVal) >> 32));

        CHECK(std::get<unsigned int>(evaluate(expr2, args))
              == (((aVal + (uint64_t)bVal) * (uint64_t)bVal) >> 32));
    }

    TEST_CASE("Expression evaluate exp2", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);

        auto expr = exp2(a);

        auto aVal = GENERATE(from_range(TestValues::floatValues));
        CAPTURE(aVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::exp2(aVal) == std::get<float>(evaluate(expr, args)));
    }

    TEST_CASE("Expression evaluate convert expressions", "[expression]")
    {
        using namespace Expression;

        float    a = 1.25f;
        Half     b = 1.1111;
        double   c = 5.2619;
        BFloat16 d(1.0f);

        auto a_exp = literal(a);
        auto b_exp = literal(b);
        auto c_exp = literal(c);
        auto d_exp = literal(d);

        auto exp1 = convert<DataType::Half>(a_exp);
        auto exp2 = convert<DataType::Half>(b_exp);
        auto exp3 = convert<DataType::Half>(c_exp);

        CHECK(resultVariableType(exp1).dataType == DataType::Half);
        CHECK(resultVariableType(exp2).dataType == DataType::Half);
        CHECK(resultVariableType(exp3).dataType == DataType::Half);

        CHECK(std::get<Half>(evaluate(exp1)) == static_cast<Half>(a));
        CHECK(std::get<Half>(evaluate(exp2)) == b);
        CHECK(std::get<Half>(evaluate(exp3)) == static_cast<Half>(c));

        auto exp4 = convert<DataType::Float>(a_exp);
        auto exp5 = convert<DataType::Float>(b_exp);
        auto exp6 = convert<DataType::Float>(c_exp);
        auto exp7 = convert<DataType::Float>(d_exp);

        CHECK(resultVariableType(exp4).dataType == DataType::Float);
        CHECK(resultVariableType(exp5).dataType == DataType::Float);
        CHECK(resultVariableType(exp6).dataType == DataType::Float);
        CHECK(resultVariableType(exp7).dataType == DataType::Float);

        CHECK(std::get<float>(evaluate(exp4)) == a);
        CHECK(std::get<float>(evaluate(exp5)) == static_cast<float>(b));
        CHECK(std::get<float>(evaluate(exp6)) == static_cast<float>(c));
        CHECK(std::get<float>(evaluate(exp7)) == static_cast<float>(d));
    }

    TEST_CASE("Expression generate dataflow tags", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        Register::AllocationOptions allocOptions{.contiguousChunkWidth
                                                 = Register::FULLY_CONTIGUOUS};

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        ra->allocateNow();
        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{1, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(1, ra);

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        rb->allocateNow();
        auto dfb = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{2, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(2, rb);

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        rc->allocateNow();
        auto dfc = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{3, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(3, rc);

        Register::ValuePtr rr1;
        context.get()->schedule(Expression::generate(rr1, dfa * dfb, context.get()));

        Register::ValuePtr rr2;
        context.get()->schedule(
            Expression::generate(rr2, Expression::fuseTernary(dfa * dfb + dfc), context.get()));

        auto result = R"(
            v_mul_f32 v12, v0, v4
            v_mul_f32 v13, v1, v5
            v_mul_f32 v14, v2, v6
            v_mul_f32 v15, v3, v7

            v_fma_f32 v16, v0, v4, v8
            v_fma_f32 v17, v1, v5, v9
            v_fma_f32 v18, v2, v6, v10
            v_fma_f32 v19, v3, v7, v11
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expression literal datatypes", "[expression]")
    {
        std::vector<VariableType> dataTypes = {{DataType::Int32},
                                               {DataType::UInt32},
                                               {DataType::Int64},
                                               {DataType::UInt64},
                                               {DataType::Float},
                                               {DataType::Half},
                                               {DataType::Double},
                                               {DataType::Bool}};

        auto dataType = GENERATE_COPY(from_range(dataTypes));

        CAPTURE(dataType);
        CHECK(dataType == Expression::resultVariableType(Expression::literal(1, dataType)));
    }

    TEST_CASE("Expression codegen literal int swap", "[expression][codegen][optimization]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto expr1 = ra->expression();
        auto expr2 = Expression::literal(-5);

        Register::ValuePtr destReg;

        context.get()->schedule(Expression::generate(destReg, expr1 + expr2, context.get()));

        context.get()->schedule(Expression::generate(destReg, expr1 & expr2, context.get()));
        context.get()->schedule(Expression::generate(destReg, expr1 | expr2, context.get()));
        context.get()->schedule(Expression::generate(destReg, expr1 ^ expr2, context.get()));

        auto result = R"(
            v_add_i32 v1, -5, v0
            v_and_b32 v1, -5, v0
            v_or_b32 v1, -5, v0
            v_xor_b32 v1, -5, v0
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expression variant test", "[expression]")
    {
        using Expression::literal;

        auto context = TestContext::ForDefaultTarget();

        int32_t  x1          = 3;
        auto     intPtr      = Expression::literal(&x1);
        int64_t  x2          = 3L;
        auto     intLongPtr  = Expression::literal(&x2);
        uint32_t x3          = 3u;
        auto     uintPtr     = Expression::literal(&x3);
        uint64_t x4          = 3UL;
        auto     uintLongPtr = Expression::literal(&x4);
        float    x5          = 3.0f;
        auto     floatPtr    = Expression::literal(&x5);
        double   x6          = 3.0;
        auto     doublePtr   = Expression::literal(&x6);

        auto intExpr    = Expression::literal(1);
        auto uintExpr   = Expression::literal(1u);
        auto floatExpr  = Expression::literal(1.0f);
        auto doubleExpr = Expression::literal(1.0);
        auto boolExpr   = Expression::literal(true);

        auto v_a = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Double, 1);
        v_a->allocateNow();

        Expression::Expression    value    = Register::Value::Literal(1);
        Expression::ExpressionPtr valuePtr = std::make_shared<Expression::Expression>(value);

        Expression::Expression    tag    = Expression::DataFlowTag();
        Expression::ExpressionPtr tagPtr = std::make_shared<Expression::Expression>(tag);
        Expression::Expression    waveTile
            = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        Expression::ExpressionPtr waveTilePtr = std::make_shared<Expression::Expression>(waveTile);

        SECTION("evaluate, fastDivision, toString, evaluationTimes throw")
        {
            std::vector<Expression::ExpressionPtr> exprs = {
                intExpr,
                uintExpr,
                floatExpr,
                doubleExpr,
                boolExpr,
                intExpr + intExpr,
                intExpr - intExpr,
                intExpr * intExpr,
                intExpr / intExpr,
                intExpr % intExpr,
                intExpr << intExpr,
                intExpr >> intExpr,
                logicalShiftR(intExpr, intExpr),
                intExpr & intExpr,
                intExpr ^ intExpr,
                intExpr > intExpr,
                intExpr < intExpr,
                intExpr >= intExpr,
                intExpr <= intExpr,
                intExpr == intExpr,
                -intExpr,
                intPtr,
                intLongPtr,
                uintPtr,
                uintLongPtr,
                floatPtr,
                doublePtr,
                valuePtr,
            };

            auto testFunc = [](auto const& expr) {
                CAPTURE(expr);
                CHECK_NOTHROW(Expression::toString(expr));
                CHECK_NOTHROW(Expression::evaluationTimes(expr));
            };

            for(auto const& expr : exprs)
            {
                testFunc(expr);
                CHECK_NOTHROW(evaluate(expr));
                CHECK_NOTHROW(Expression::fastDivision(expr, context.get()));
            }

            testFunc(v_a);
            CHECK_THROWS_AS(Expression::evaluate(v_a), FatalError);

            testFunc(tag);
            CHECK_THROWS_AS(evaluate(tag), FatalError);

            testFunc(tagPtr);
            CHECK_THROWS_AS(evaluate(tagPtr), FatalError);
            CHECK_NOTHROW(Expression::fastDivision(tagPtr, context.get()));

            testFunc(value);
            CHECK_NOTHROW(evaluate(value));

            testFunc(waveTile);
            CHECK_THROWS_AS(evaluate(waveTile), FatalError);

            testFunc(waveTilePtr);

            CHECK_THROWS_AS(evaluate(waveTilePtr), FatalError);
            CHECK_NOTHROW(Expression::fastDivision(waveTilePtr, context.get()));
        }

        SECTION("convert")
        {
            // This just checks that the runtime version of `convert()` will throw an exception
            // for a conversion that doesn't exist instead of returning something incorrect.
            CHECK_NOTHROW(convert(DataType::Float, intExpr));
            CHECK_NOTHROW(convert(DataType::Double, intExpr));
            CHECK_THROWS_AS(convert(DataType::ComplexFloat, intExpr), FatalError);
            CHECK_THROWS_AS(convert(DataType::ComplexDouble, intExpr), FatalError);
            CHECK_NOTHROW(convert(DataType::Half, intExpr));
            CHECK_NOTHROW(convert(DataType::Halfx2, intExpr));
            CHECK_THROWS_AS(convert(DataType::Int8x4, intExpr), FatalError);
            CHECK_NOTHROW(convert(DataType::Int32, intExpr));
            CHECK_NOTHROW(convert(DataType::Int64, intExpr));
            CHECK_NOTHROW(convert(DataType::BFloat16, intExpr));
            CHECK_NOTHROW(convert(DataType::BFloat16x2, intExpr));
            CHECK_THROWS_AS(convert(DataType::Int8, intExpr), FatalError);
            CHECK_THROWS_AS(convert(DataType::Raw32, intExpr), FatalError);
            CHECK_NOTHROW(convert(DataType::UInt32, intExpr));
            CHECK_NOTHROW(convert(DataType::UInt64, intExpr));
            CHECK_NOTHROW(convert(DataType::Bool, intExpr));
            CHECK_NOTHROW(convert(DataType::Bool32, intExpr));
            CHECK_THROWS_AS(convert(DataType::Count, intExpr), FatalError);
            CHECK_THROWS_AS(convert(static_cast<DataType>(200), intExpr), FatalError);
        }
    }

    TEST_CASE("Expression complexity values", "[expression][optimization]")
    {
        auto intExpr = Expression::literal(1);

        CHECK(Expression::complexity(intExpr) == 0);
        CHECK(Expression::complexity(intExpr + intExpr) > Expression::complexity(intExpr));
        CHECK(Expression::complexity(intExpr + intExpr + intExpr)
              > Expression::complexity(intExpr + intExpr));

        CHECK(Expression::complexity(intExpr / intExpr)
              > Expression::complexity(intExpr + intExpr));
    }

    TEST_CASE("Expression kernel arguments", "[expression][utility]")
    {
        namespace XP  = Expression;
        using strings = std::unordered_set<std::string>;

        auto karg0 = std::make_shared<AssemblyKernelArgument>("KernelArg0", DataType::Int32);
        auto karg1 = std::make_shared<AssemblyKernelArgument>("KernelArg1", DataType::Int32);
        auto karg2 = std::make_shared<AssemblyKernelArgument>("KernelArg2", DataType::Int32);

        auto kargExp0 = std::make_shared<XP::Expression>(karg0);
        auto kargExp1 = std::make_shared<XP::Expression>(karg1);
        auto kargExp2 = std::make_shared<XP::Expression>(karg2);

        CHECK(referencedKernelArguments(kargExp0) == strings({"KernelArg0"}));

        CHECK(referencedKernelArguments((kargExp0 + kargExp0) + XP::literal(5))
              == strings({"KernelArg0"}));

        CHECK(referencedKernelArguments((kargExp0 + kargExp1) + XP::literal(5))
              == strings({"KernelArg0", "KernelArg1"}));

        CHECK(referencedKernelArguments(
                  Expression::conditional(kargExp0, kargExp1, (kargExp2 + XP::literal(5))))
              == strings({"KernelArg0", "KernelArg1", "KernelArg2"}));

        SECTION("a")
        {
            auto scaledMM = std::make_shared<XP::Expression>(Expression::ScaledMatrixMultiply{
                kargExp0, XP::literal(3), XP::literal(2), XP::literal(3), XP::literal(9)});

            CHECK(referencedKernelArguments(scaledMM) == strings({"KernelArg0"}));
        }

        SECTION("b")
        {
            auto scaledMM = std::make_shared<XP::Expression>(Expression::ScaledMatrixMultiply{
                XP::literal(1), kargExp1, XP::literal(3), XP::literal(2), XP::literal(3)});

            CHECK(referencedKernelArguments(scaledMM) == strings({"KernelArg1"}));
        }

        SECTION("c")
        {
            auto scaledMM = std::make_shared<XP::Expression>(Expression::ScaledMatrixMultiply{
                XP::literal(3), XP::literal(1), kargExp2, XP::literal(2), XP::literal(3)});

            CHECK(referencedKernelArguments(scaledMM) == strings({"KernelArg2"}));
        }

        SECTION("sa")
        {
            auto scaledMM = std::make_shared<XP::Expression>(Expression::ScaledMatrixMultiply{
                XP::literal(2), XP::literal(3), XP::literal(1), kargExp0, XP::literal(3)});

            CHECK(referencedKernelArguments(scaledMM) == strings({"KernelArg0"}));
        }

        SECTION("sb")
        {
            auto scaledMM = std::make_shared<XP::Expression>(Expression::ScaledMatrixMultiply{
                XP::literal(3), XP::literal(2), XP::literal(3), XP::literal(1), kargExp1});

            CHECK(referencedKernelArguments(scaledMM) == strings({"KernelArg1"}));
        }
    }

}
