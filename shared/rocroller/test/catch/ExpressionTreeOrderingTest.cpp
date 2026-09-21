// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include "TestContext.hpp"

using namespace rocRoller;

namespace ExpressionTreeOrderingTest
{
    /**
     * Helper to verify that priorities form a valid ordering.
     * Checks that all values are unique, in range [0, size), and dependencies are respected.
     */
    bool isValidPriorityOrdering(Expression::ExpressionTree const& tree)
    {
        // All priority values should be unique and in range [0, tree.size())
        std::set<int> seen;
        for(auto const& node : tree)
        {
            if(node.priorityOrder < 0 || node.priorityOrder >= static_cast<int>(tree.size()))
                return false;
            if(seen.contains(node.priorityOrder))
                return false;
            seen.insert(node.priorityOrder);
        }
        if(seen.size() != tree.size())
            return false;

        // Verify dependencies are respected
        for(size_t i = 0; i < tree.size(); ++i)
        {
            for(int dep : tree[i].deps)
            {
                if(tree[dep].priorityOrder >= tree[i].priorityOrder)
                    return false;
            }
        }
        return true;
    }

    TEST_CASE("updatePriorityOrder produces valid ordering", "[expression][codegen]")
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

        SECTION("Simple binary expression")
        {
            auto expr = a + b;
            auto tree = Expression::consolidateSubExpressions(expr, context.get());

            CHECK(isValidPriorityOrdering(tree));
        }

        SECTION("Nested expression 1")
        {
            auto expr = (a + b) * (a - b);
            auto tree = Expression::consolidateSubExpressions(expr, context.get());

            CHECK(isValidPriorityOrdering(tree));
        }

        SECTION("Nested expression 2")
        {
            // ((a + b) * (a + b)) - ((a - b) * (a - b))
            auto sum  = a + b;
            auto diff = a - b;
            auto expr = (sum * sum) - (diff * diff);
            auto tree = Expression::consolidateSubExpressions(expr, context.get());

            CHECK(isValidPriorityOrdering(tree));
        }

        SECTION("Multiple operations on same base with FastArithmetic")
        {
            auto dReg = std::make_shared<Register::Value>(
                context.get(), Register::Type::Scalar, DataType::UInt32, 1);
            dReg->allocateNow();
            auto d = dReg->expression();

            auto xReg = std::make_shared<Register::Value>(
                context.get(), Register::Type::Scalar, DataType::UInt32, 1);
            xReg->allocateNow();
            auto x = xReg->expression();

            // x / d, x % d, x * 2
            auto quot     = x / d;
            auto rem      = x % d;
            auto mul      = x * Expression::literal(2u);
            auto expr     = quot + rem + mul;
            auto fast     = Expression::FastArithmetic(context.get());
            auto expanded = fast(expr);

            auto tree = Expression::consolidateSubExpressions(expanded, context.get());

            CHECK(isValidPriorityOrdering(tree));
        }
    }

    TEST_CASE("Sethi-Ullman ordering prioritizes heavier subtrees", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rd->allocateNow();

        auto re = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        re->allocateNow();

        auto rf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rf->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();
        auto c = rc->expression();
        auto d = rd->expression();
        auto e = re->expression();
        auto f = rf->expression();

        SECTION("Unbalanced tree - left-heavy with weight 3")
        {
            // Tree structure (unique operations):
            //
            //              root(|)
            //             /       \
            //          (*)       (<<)
            //         /   \     /    \
            //       (+)  (-)   d    neg(e)
            //      /  \  / \          |
            //     a   b c   f         e
            //
            // Sethi-Ullman weights:
            //
            //                 3
            //              /     \
            //            3         2
            //          /   \     /   \
            //        2      2   1     1
            //      /  \   /  \        |
            //     1   1  1    1       1
            //

            auto inner1    = a + b; // Add
            auto inner2    = c - f; // Subtract
            auto heavy     = inner1 * inner2; // Multiply
            auto negE      = -e; // Negate (unary)
            auto lightSide = d << negE; // ShiftL
            auto expr      = heavy | lightSide; // BitwiseOr
            auto tree      = Expression::consolidateSubExpressions(expr, context.get());

            CHECK(isValidPriorityOrdering(tree));

            // Helper to find priority by operation type in the tree
            auto priorityOf = [&](auto const& opType) -> int {
                for(auto const& node : tree)
                    if(std::holds_alternative<std::decay_t<decltype(opType)>>(*node.expr))
                        return node.priorityOrder;
                return -1;
            };

            // Get priorities by operation type
            int pAdd = priorityOf(Expression::Add{});
            int pSub = priorityOf(Expression::Subtract{});
            int pMul = priorityOf(Expression::Multiply{});
            int pNeg = priorityOf(Expression::Negate{});
            int pShl = priorityOf(Expression::ShiftL{});
            int pOr  = priorityOf(Expression::BitwiseOr{});

            // Dependency checks within heavy subtree
            CHECK(pAdd < pMul); // Add before Multiply
            CHECK(pSub < pMul); // Subtract before Multiply

            // Dependency checks within light subtree
            CHECK(pNeg < pShl); // Negate before ShiftL

            // Heavy subtree (weight 3) must complete before light subtree (weight 2) starts
            CHECK(pMul < pNeg); // Multiply before Negate
            CHECK(pMul < pShl); // Multiply before ShiftL

            // Root comes last
            CHECK(pShl < pOr); // ShiftL before BitwiseOr
            CHECK(pMul < pOr); // Multiply before BitwiseOr

            // Within heavy subtree, one inner op completes before the other starts
            bool addFirst = (pAdd < pSub);
            bool subFirst = (pSub < pAdd);
            CHECK((addFirst || subFirst));
        }
    }

    TEST_CASE("maxConcurrentSubExpressions affects code generation order", "[expression][codegen]")
    {
        // Tree structure:
        //
        //              root(+)
        //             /       \
        //          (*)       (*)
        //         /   \     /   \
        //       (+)  (-)   (<<)  (>>)
        //      /  \  / \   /  \  / \
        //     a   b c   f a   b c   f
        //
        // priorityOrder is computed once by updatePriorityOrder and stays the same.
        // maxConcurrentSubExpressions affects the scheduling order during code generation
        // by limiting how many nodes can be "in-flight" at once (categoryLimit in generateNodes).

        auto generateCode = [](int maxConcurrent) -> std::string {
            KernelOptions opts;
            opts->maxConcurrentSubExpressions = maxConcurrent;

            auto context = TestContext::ForDefaultTarget(opts);

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
            auto rf = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            rf->setName("rf");
            rf->allocateNow();

            auto a = ra->expression();
            auto b = rb->expression();
            auto c = rc->expression();
            auto f = rf->expression();

            auto add      = a + b;
            auto sub      = c - f;
            auto shl      = a << b;
            auto shr      = c >> f;
            auto leftMul  = add * sub;
            auto rightMul = shl * shr;
            auto root     = leftMul + rightMul;

            Register::ValuePtr dest;
            context.get()->schedule(Expression::generate(dest, root, context.get()));

            return context.output();
        };

        SECTION("maxConcurrentSubExpressions = 1 processes one subtree at a time")
        {
            auto code = generateCode(1);

            // Find positions of key operations in generated code
            auto addPos    = code.find("v_add");
            auto subPos    = code.find("v_sub");
            auto lshiftPos = code.find("v_lshl");
            auto rshiftPos = code.find("v_ashr");
            auto mulPos    = code.find("v_mul");

            // With max=1, one subtree should be fully completed (including its multiply)
            // before the other subtree starts.
            // Either: add, sub, mul all before lshift, rshift
            // Or:     lshift, rshift, mul all before add, sub
            if(addPos != std::string::npos && subPos != std::string::npos
               && lshiftPos != std::string::npos && rshiftPos != std::string::npos
               && mulPos != std::string::npos)
            {
                bool leftFirst = (addPos < lshiftPos) && (addPos < rshiftPos)
                                 && (subPos < lshiftPos) && (subPos < rshiftPos);
                bool rightFirst = (lshiftPos < addPos) && (lshiftPos < subPos)
                                  && (rshiftPos < addPos) && (rshiftPos < subPos);
                CHECK((leftFirst || rightFirst));

                // There should be a multiply before the second side starts
                if(leftFirst)
                {
                    // mul should come before lshift and rshift
                    CHECK(mulPos < lshiftPos);
                    CHECK(mulPos < rshiftPos);
                }
                else
                {
                    // mul should come before add and sub
                    CHECK(mulPos < addPos);
                    CHECK(mulPos < subPos);
                }
            }
        }

        SECTION("maxConcurrentSubExpressions = 4 processes subtrees in parallel")
        {
            auto code = generateCode(4);

            // Find positions of key operations in generated code
            auto addPos    = code.find("v_add");
            auto subPos    = code.find("v_sub");
            auto lshiftPos = code.find("v_lshl");
            auto rshiftPos = code.find("v_ashr");
            auto mulPos    = code.find("v_mul");

            CHECK(!code.empty());

            // With max=4, all leaf operations can be processed before any multiply.
            // There should be no multiply in between the leaf operations.
            if(addPos != std::string::npos && subPos != std::string::npos
               && lshiftPos != std::string::npos && rshiftPos != std::string::npos
               && mulPos != std::string::npos)
            {
                // Find the range of leaf operations
                auto maxLeafPos = std::max({addPos, subPos, lshiftPos, rshiftPos});

                // Multiply should come after all leaf operations
                CHECK(mulPos > maxLeafPos);
            }
        }
    }
}
