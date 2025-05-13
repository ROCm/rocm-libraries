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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomSections.hpp"
#include "TestContext.hpp"

#include <common/SourceMatcher.hpp>

using namespace rocRoller;

TEST_CASE("Register::Value class works", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);

    CHECK_FALSE(r->canUseAsOperand());
    CHECK_THROWS(r->assertCanUseAsOperand());
    r->allocateNow();
    CHECK(r->canUseAsOperand());
    CHECK_NOTHROW(r->assertCanUseAsOperand());
}

TEST_CASE("Register::Value::PromoteType works", "[codegen][register]")
{
    auto ExpectEqual = [&](Register::Type result, Register::Type lhs, Register::Type rhs) {
        CHECK(result == Register::PromoteType(lhs, rhs));
        CHECK(result == Register::PromoteType(rhs, lhs));
    };

    ExpectEqual(Register::Type::Literal, Register::Type::Literal, Register::Type::Literal);
    ExpectEqual(Register::Type::Scalar, Register::Type::Scalar, Register::Type::Scalar);
    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Vector);
    ExpectEqual(
        Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Accumulator);

    ExpectEqual(Register::Type::Scalar, Register::Type::Scalar, Register::Type::Literal);
    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Literal);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Literal);

    ExpectEqual(Register::Type::Vector, Register::Type::Vector, Register::Type::Scalar);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Scalar);
    ExpectEqual(Register::Type::Accumulator, Register::Type::Accumulator, Register::Type::Vector);

    CHECK_THROWS(Register::PromoteType(Register::Type::Literal, Register::Type::Label));
}

TEST_CASE("Register::Value class string rendering works", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto allocOptions = Register::AllocationOptions::FullyContiguous();

    SECTION("Single register")
    {
        auto r = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        r->allocateNow();
        CHECK(r->toString() == "v0");
    }

    SECTION("Multi-valued type")
    {
        auto r = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Double, 1);
        r->allocateNow();
        CHECK(r->toString() == "v[0:1]");
        CHECK(r->negate()->toString() == "neg(v[0:1])");
        CHECK(r->subset({0})->toString() == "v0");
        CHECK(r->subset({1})->toString() == "v1");
    }

    SECTION("Multiple values")
    {
        auto r = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Double, 4, allocOptions);
        r->allocateNow();
        CHECK(r->toString() == "v[0:7]");
        CHECK(r->subset({4})->toString() == "v4");
        CHECK(r->subset({1, 2, 3, 4})->toString() == "v[1:4]");
        CHECK(r->subset({1, 2, 3, 4})->subset({1})->toString() == "v2");
        CHECK(r->element({1})->toString() == "v[2:3]");
        CHECK(r->element({1, 2})->element({1})->toString() == "v[4:5]");
        CHECK_THROWS(r->subset({22}));
        CHECK_THROWS(r->element({22}));
    }

    SECTION("Multiple values, scalar")
    {
        auto r = std::make_shared<Register::Value>(
            context.get(), Register::Type::Scalar, DataType::Float, 2, allocOptions);
        r->allocateNow();

        CHECK(r->toString() == "s[0:1]");
    }

    SECTION("Multiple values, accumulator, non-contiguous alues")
    {
        auto a = std::make_shared<Register::Allocation>(
            context.get(), Register::Type::Accumulator, DataType::Float, 2);

        a->allocateNow();

        auto r = **a;
        CHECK(r->toString() == "[a1, a0]");
    }
}

TEST_CASE("Register::Value class allocation state works", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    CHECK(r->canAllocateNow());
    auto r_large = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 10000);
    CHECK_FALSE(r_large->canAllocateNow());
    auto r_placeholder
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    auto r_literal = Register::Value::Literal(3.5);

    r->allocateNow();
    CHECK(r->allocationState() == Register::AllocationState::Allocated);

    CHECK_FALSE(r->isPlaceholder());
    CHECK(r_placeholder->isPlaceholder());
    CHECK_FALSE(r_literal->isPlaceholder());
}

TEST_CASE("Register::Value class works with literal values", "[codegen][register]")
{
    auto r = Register::Value::Literal(5);
    CHECK(r->toString() == "5");
    CHECK(r->canUseAsOperand());

    r = Register::Value::Literal(100.0f);
    CHECK(r->toString() == "100.000");
    CHECK(r->canUseAsOperand());

    r = Register::Value::Literal(3.14159f);
    CHECK(r->toString() == "3.14159");

    r = Register::Value::Literal(2.7182818f);
    CHECK(r->toString() == "2.71828");
}

TEST_CASE("Register::Value class works with literal values from CommandArgumentValue",
          "[codegen][register]")
{
    CommandArgumentValue v1 = 5;

    auto r1 = Register::Value::Literal(v1);

    CHECK(r1->toString() == "5");

    CommandArgumentValue v2 = 7.5;

    auto r2 = Register::Value::Literal(v2);

    CHECK(r2->toString() == "7.50000");

    int intVal;

    CommandArgumentValue v3 = &intVal;

    auto r3 = Register::Value::Literal(v3);

    VariableType expected{DataType::Int32, PointerType::PointerGlobal};

    CHECK(expected == r3->variableType());
}

TEST_CASE("Register::Values can be named", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    r->setName("D0");
    r->allocateNow();

    CHECK(r->name() == "D0");
    CHECK(r->allocation()->name() == "D0");

    r->allocation()->setName("D2");
    CHECK(r->allocation()->name() == "D2");
    CHECK(r->name() == "D0");

    r->setName("D3");
    CHECK(r->allocation()->name() == "D2");
}

TEST_CASE("Emitting a label instruction from a label register works.", "[codegen][register]")
{
    auto context    = TestContext::ForDefaultTarget();
    auto kernelName = context->kernel()->kernelName();

    {
        auto label = context->labelAllocator()->label("main loop");
        CHECK(label->canUseAsOperand());
        auto inst = Instruction::Label(label);
        context->schedule(inst);
    }

    auto expected0 = kernelName + "_main_loop:\n";

    CHECK(NormalizedSource(context.output()) == NormalizedSource(expected0));

    {
        auto label = context->labelAllocator()->label("main_loop");
        auto inst  = Instruction::Label(label);
        context->schedule(inst);
    }

    auto expected1 = expected0 + '\n' + kernelName + "_0_main_loop:\n";

    CHECK(NormalizedSource(context.output()) == NormalizedSource(expected1));

    {
        auto label = context->labelAllocator()->label("condition a == b");
        auto inst  = Instruction::Label(label);
        context->schedule(inst);
    }

    auto expected2 = expected1 + '\n' + kernelName + "_condition_a_b:\n";

    CHECK(NormalizedSource(context.output()) == NormalizedSource(expected2));
}

TEST_CASE("AllocationState toString works.", "[codegen][register]")
{
#define CHECK_ALLOCATION_STATE(state) CHECK(#state == toString(Register::AllocationState::state))

    CHECK_ALLOCATION_STATE(Unallocated);
    CHECK_ALLOCATION_STATE(Allocated);
    CHECK_ALLOCATION_STATE(Freed);
    CHECK_ALLOCATION_STATE(NoAllocation);
    CHECK_ALLOCATION_STATE(Error);
    CHECK_ALLOCATION_STATE(Count);
#undef CHECK_ALLOCATION_STATE

    std::stringstream stream;
    stream << Register::AllocationState::Allocated;
    CHECK(stream.str() == "Allocated");
}

TEST_CASE("Register::Value subset() and element() work on unallocated registers.",
          "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Int64, 4);

    auto r2 = r->subset({0});
    CHECK_THROWS_AS(r2->getRegisterIds().to<std::vector>(), FatalError);

    auto r3 = r->element({1});
    CHECK_THROWS_AS(r3->getRegisterIds().to<std::vector>(), FatalError);

    auto r4 = r->subset({2});

    CHECK(r2->sameAs(r->subset({0})));
    CHECK(r2->sameAs(r2));
    CHECK_FALSE(r2->sameAs(r));
    CHECK_FALSE(r2->sameAs(r3));
    CHECK_FALSE(r2->sameAs(r4));

    r->allocateNow();

    auto rIDs = r->getRegisterIds().to<std::vector>();

    REQUIRE(rIDs.size() == 8);

    CHECK(r2->getRegisterIds().to<std::vector>() == std::vector{rIDs[0]});

    CHECK(r3->getRegisterIds().to<std::vector>() == (std::vector{rIDs[2], rIDs[3]}));

    CHECK(r2->sameAs(r->subset({0})));
    CHECK(r2->sameAs(r2));
    CHECK_FALSE(r2->sameAs(r));
    CHECK_FALSE(r2->sameAs(r3));
    CHECK_FALSE(r2->sameAs(r4));
}

TEST_CASE("Register::Value allocation works across different objects that share the same "
          "underlying Allocation object.",
          "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::UInt64, 4);

    auto r2 = r->subset({0, 2});
    CHECK_THROWS_AS(r2->getRegisterIds().to<std::vector>(), FatalError);

    auto r3 = r2->element({1});
    CHECK_THROWS_AS(r3->getRegisterIds().to<std::vector>(), FatalError);

    CHECK_FALSE(r->sameAs(r2));
    CHECK_FALSE(r3->sameAs(r2));
    CHECK(r3->sameAs(r->subset({2})));
    CHECK_FALSE(r3->sameAs(r->subset({0})));

    r3->allocateNow(); //Allocating r3 should allocate the underlying allocation, even though it's 2 levels down.

    auto rIDs = r->getRegisterIds().to<std::vector>();
    REQUIRE(rIDs.size() == 8);

    auto r2IDs = r2->getRegisterIds().to<std::vector>();
    REQUIRE(r2IDs.size() == 2);

    auto r3IDs = r3->getRegisterIds().to<std::vector>();
    REQUIRE(r3IDs.size() == 1);

    CHECK(r2->getRegisterIds().to<std::vector>() == (std::vector{rIDs[0], rIDs[2]}));

    CHECK(r3->getRegisterIds().to<std::vector>() == (std::vector{rIDs[2]}));

    CHECK_FALSE(r->sameAs(r2));
    CHECK_FALSE(r3->sameAs(r2));
    CHECK(r3->sameAs(r->subset({2})));
}

TEST_CASE("Register::Value subset() and element() will throw if given an index out of bounds.",
          "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Int32, 4);
    r->allocateNow();
    CHECK_THROWS_AS(r->subset({5}), FatalError);
    CHECK_THROWS_AS(r->element({5}), FatalError);
}

TEST_CASE("Register::Value objects using the same allocation coords will alias the registers.",
          "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto r = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 8);
    r->allocateNow();
    CHECK(r->allocationState() == Register::AllocationState::Allocated);

    auto val1 = r->subset({3, 4});

    auto val2 = std::make_shared<Register::Value>(
        r->allocation(), Register::Type::Vector, DataType::Half, val1->allocationCoord());

    CHECK(val1->toString() == val2->toString());
}

TEST_CASE("Register::Value::split() works to split up the allocations.", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    std::vector<Register::ValuePtr> individualValues;

    std::vector<int> blockIndices;

    {
        auto block = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Raw32, 8);
        block->allocateNow();
        CHECK(block->allocationState() == Register::AllocationState::Allocated);
        blockIndices = block->registerIndices().to<std::vector>();
        REQUIRE(blockIndices.size() == 8);

        std::vector<std::vector<int>> indices = {{0}, {2, 3}, {4, 5, 6, 7}};

        individualValues = block->split(indices);

        REQUIRE(individualValues.size() == 3);
    }

    CHECK(individualValues[0]->registerIndices().to<std::vector>()
          == std::vector<int>{blockIndices[0]});

    CHECK(individualValues[1]->registerIndices().to<std::vector>()
          == (std::vector<int>{blockIndices[2], blockIndices[3]}));

    CHECK(
        individualValues[2]->registerIndices().to<std::vector>()
        == (std::vector<int>{blockIndices[4], blockIndices[5], blockIndices[6], blockIndices[7]}));

    auto allocator = context->allocator(Register::Type::Vector);
    CHECK(allocator->isFree(blockIndices[1]));
}

TEST_CASE("Register::Value::intersects() works", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

#define CHECK_INTERSECTS(a, b) \
    CHECK(a->intersects(b));   \
    CHECK(b->intersects(a))

#define CHECK_DOESNT_INTERSECT(a, b) \
    CHECK_FALSE(a->intersects(b));   \
    CHECK_FALSE(b->intersects(a))

    SECTION("GPRs and special registers")
    {
        auto a = Register::Value::Placeholder(
            context.get(), Register::Type::Scalar, DataType::Float, 8);
        CHECK_DOESNT_INTERSECT(a, a); // Unallocated registers don't intersect with anything.
        a->allocateNow();
        CHECK_INTERSECTS(a, a);

        auto b = Register::Value::Placeholder(
            context.get(), Register::Type::Scalar, DataType::Int32, 2);
        b->allocateNow();

        CHECK_DOESNT_INTERSECT(a, b);

        auto a_subset_1  = a->subset({1});
        auto a_element_1 = a->element({1});

        CHECK_INTERSECTS(a, a_subset_1);
        CHECK_DOESNT_INTERSECT(b, a_subset_1);

        CHECK_INTERSECTS(a, a_element_1);
        CHECK_INTERSECTS(a_subset_1, a_element_1);
        CHECK_DOESNT_INTERSECT(b, a_element_1);

        auto c = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Float, 8);
        c->allocateNow();

        {
            // Double-check that a and c have register indices in common even
            // though they're different types of registers.
            auto          aIndices = a->registerIndices().to<std::set>();
            auto          cIndices = c->registerIndices().to<std::set>();
            std::set<int> common;
            std::set_intersection(aIndices.begin(),
                                  aIndices.end(),
                                  cIndices.begin(),
                                  cIndices.end(),
                                  std::inserter(common, common.begin()));
            CHECK(common != std::set<int>{});
        }

        CHECK_DOESNT_INTERSECT(a, c);
        CHECK_DOESNT_INTERSECT(a_subset_1, c);
        CHECK_DOESNT_INTERSECT(a_element_1, c);
        CHECK_DOESNT_INTERSECT(b, c);

        auto vcc  = context->getVCC();
        auto scc  = context->getSCC();
        auto exec = context->getExec();

        CHECK_DOESNT_INTERSECT(a, vcc);
        CHECK_DOESNT_INTERSECT(b, vcc);
        CHECK_DOESNT_INTERSECT(c, vcc);
        CHECK_DOESNT_INTERSECT(a, scc);
        CHECK_DOESNT_INTERSECT(b, scc);
        CHECK_DOESNT_INTERSECT(c, scc);
        CHECK_DOESNT_INTERSECT(a, exec);
        CHECK_DOESNT_INTERSECT(b, exec);
        CHECK_DOESNT_INTERSECT(c, exec);

        CHECK_DOESNT_INTERSECT(vcc, context->getSCC());

        CHECK_INTERSECTS(vcc, context->getVCC());
        CHECK_INTERSECTS(scc, context->getSCC());
        CHECK_INTERSECTS(exec, context->getExec());

        CHECK_INTERSECTS(vcc, context->getVCC_LO());
        CHECK_INTERSECTS(vcc, context->getVCC_HI());
        CHECK_DOESNT_INTERSECT(context->getVCC_LO(), context->getVCC_HI());
    }

    SECTION("LDS")
    {
        auto a = Register::Value::Placeholder(
            context.get(), Register::Type::Scalar, DataType::Float, 8);
        a->allocateNow();

        auto lds1 = Register::Value::AllocateLDS(context.get(), DataType::BF6x16, 4);
        auto lds2 = Register::Value::AllocateLDS(context.get(), DataType::BF6x16, 4);

        // Currently can't subdivide LDS allocations.  If this changes,
        // intersects() and this test will need to be updated.
        CHECK_THROWS(lds1->subset({0}));
        CHECK_THROWS(lds1->element({0}));

        CHECK_DOESNT_INTERSECT(a, lds1);
        CHECK_DOESNT_INTERSECT(a, lds2);
        CHECK_DOESNT_INTERSECT(lds1, lds2);

        CHECK_INTERSECTS(lds1, lds1);
        CHECK_INTERSECTS(lds2, lds2);
    }

#undef CHECK_INTERSECTS
#undef CHECK_DOESNT_INTERSECT
}

TEST_CASE("Register::RegisterId and Register::RegisterIdHash() work.", "[codegen][register]")
{
    auto context = TestContext::ForDefaultTarget();

    auto rs = std::make_shared<Register::Value>(
        context.get(), Register::Type::Scalar, DataType::Float, 8);
    rs->allocateNow();

    for(auto& id1 : rs->getRegisterIds())
    {
        for(auto& id2 : rs->getRegisterIds())
        {
            if(id1 != id2)
            {
                CHECK(Register::RegisterIdHash()(id1) != Register::RegisterIdHash()(id2));
            }
            else
            {
                CHECK(Register::RegisterIdHash()(id1) == Register::RegisterIdHash()(id2));
            }
        }
    }

    auto rv = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 8);
    rv->allocateNow();

    for(auto& id1 : rv->getRegisterIds())
    {
        for(auto& id2 : rv->getRegisterIds())
        {
            if(id1 != id2)
            {
                CHECK(Register::RegisterIdHash()(id1) != Register::RegisterIdHash()(id2));
            }
            else
            {
                CHECK(Register::RegisterIdHash()(id1) == Register::RegisterIdHash()(id2));
            }
        }
    }

    for(auto& id1 : rv->getRegisterIds())
    {
        for(auto& id2 : rs->getRegisterIds())
        {
            CHECK(Register::RegisterIdHash()(id1) != Register::RegisterIdHash()(id2));
        }
    }
}

TEST_CASE("Register::RegisterId and Register::RegisterIdHash() work for special registers.",
          "[codegen][register]")
{
    SUPPORTED_ARCH_SECTION(arch)
    {
        auto context = TestContext::ForTarget(arch);

        std::vector<std::shared_ptr<rocRoller::Register::Value>> specialRegisters(
            {context->getVCC_LO(), context->getVCC_HI(), context->getSCC(), context->getExec()});
        std::vector<Register::RegisterId> specialRegisterIDs;
        for(auto& specialRegister : specialRegisters)
        {
            for(auto& id : specialRegister->getRegisterIds())
            {
                specialRegisterIDs.push_back(id);
            }
        }
        CHECK(specialRegisterIDs.size() == specialRegisters.size());

        for(int i = 0; i < specialRegisters.size(); i++)
        {
            for(int j = 0; j < specialRegisters.size(); j++)
            {
                if(specialRegisters[i]->toString() != specialRegisters[j]->toString())
                {
                    CHECK(specialRegisterIDs[i] != specialRegisterIDs[j]);
                    CHECK(Register::RegisterIdHash()(specialRegisterIDs[i])
                          != Register::RegisterIdHash()(specialRegisterIDs[j]));
                }
                else
                {
                    CHECK(specialRegisterIDs[i] == specialRegisterIDs[j]);
                    CHECK(Register::RegisterIdHash()(specialRegisterIDs[i])
                          == Register::RegisterIdHash()(specialRegisterIDs[j]));
                }
            }
        }

        std::vector<Register::RegisterId> vccRegisterIDs;
        for(auto& id : context->getVCC()->getRegisterIds())
        {
            vccRegisterIDs.push_back(id);
        }
        if(context->kernel()->wavefront_size() == 64)
        {
            CHECK(vccRegisterIDs.size() == 2);
            CHECK(vccRegisterIDs[0] != vccRegisterIDs[1]);
            CHECK(Register::RegisterIdHash()(vccRegisterIDs[0])
                  != Register::RegisterIdHash()(vccRegisterIDs[1]));
            CHECK(vccRegisterIDs[1] == specialRegisterIDs[1]);
        }
        else
        {
            CHECK(vccRegisterIDs.size() == 1);
        }
        CHECK(vccRegisterIDs[0] == specialRegisterIDs[0]);
    }
}
