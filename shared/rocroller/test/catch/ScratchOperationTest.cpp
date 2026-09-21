// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Operations/Scratch.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "TestContext.hpp"

#include <set>
#include <sstream>

using namespace rocRoller;
using namespace rocRoller::Operations;

namespace ScratchOperationTest
{
    TEST_CASE("ScratchPolicy enum", "[scratch][serialization]")
    {
        SECTION("toString for all values")
        {
            CHECK(toString(ScratchPolicy::None) == "None");
            CHECK(toString(ScratchPolicy::ZeroedBeforeAndAfter) == "ZeroedBeforeAndAfter");
        }

        SECTION("operator<< for ScratchPolicy")
        {
            std::ostringstream oss;
            oss << ScratchPolicy::None;
            CHECK(oss.str() == "None");

            oss.str("");
            oss << ScratchPolicy::ZeroedBeforeAndAfter;
            CHECK(oss.str() == "ZeroedBeforeAndAfter");
        }

        SECTION("toString for None policy")
        {
            Scratch scratch(OperationTag(), ScratchPolicy::None);
            CHECK(scratch.toString() == "Scratch(None)");
        }

        SECTION("toString for ZeroedBeforeAndAfter policy")
        {
            Scratch scratch(OperationTag(), ScratchPolicy::ZeroedBeforeAndAfter);
            CHECK(scratch.toString() == "Scratch(ZeroedBeforeAndAfter)");
        }

        SECTION("Verify format is Scratch(PolicyName)")
        {
            Scratch     scratch(OperationTag(), ScratchPolicy::None);
            std::string str = scratch.toString();
            CHECK(str.substr(0, 8) == "Scratch(");
            CHECK(str.back() == ')');
        }
    }

    TEST_CASE("Scratch construction", "[operation][scratch]")
    {
        SECTION("Count value for range checking")
        {
            // Verify Count is at the end for range-based operations
            CHECK(static_cast<int>(ScratchPolicy::Count) == 2);
        }

        SECTION("Invalid policy returns Invalid")
        {
            auto invalidPolicy = static_cast<ScratchPolicy>(99);
            CHECK(toString(invalidPolicy) == "Invalid");
        }

        SECTION("Create Scratch with None policy")
        {
            Scratch scratch(OperationTag(), ScratchPolicy::None);
            CHECK(scratch.policy() == ScratchPolicy::None);
        }

        SECTION("Create Scratch with ZeroedBeforeAndAfter policy")
        {
            Scratch scratch(OperationTag(), ScratchPolicy::ZeroedBeforeAndAfter);
            CHECK(scratch.policy() == ScratchPolicy::ZeroedBeforeAndAfter);
        }

        SECTION("Constructor with only tag defaults policy to None")
        {
            auto    tag = OperationTag();
            Scratch scratch(tag);
            CHECK(scratch.policy() == ScratchPolicy::None);
            CHECK(scratch.getTag() == tag);
        }

        SECTION("Create Scratch with None policy and tag")
        {
            auto    tag = OperationTag();
            Scratch scratch(tag, ScratchPolicy::None);
            CHECK(scratch.policy() == ScratchPolicy::None);
            CHECK(scratch.getTag() == tag);
        }

        SECTION("policy() accessor returns correct value")
        {
            Scratch scratch1(OperationTag(), ScratchPolicy::None);
            Scratch scratch2(OperationTag(), ScratchPolicy::ZeroedBeforeAndAfter);

            CHECK(scratch1.policy() == ScratchPolicy::None);
            CHECK(scratch2.policy() == ScratchPolicy::ZeroedBeforeAndAfter);
        }

        SECTION("Same policy should be equal")
        {
            auto    tag1 = OperationTag();
            auto    tag2 = OperationTag();
            Scratch scratch1(tag1, ScratchPolicy::None);
            Scratch scratch2(tag2, ScratchPolicy::None);
            CHECK(tag1 == tag2);
            CHECK(scratch1 == scratch2);

            auto    tag3 = OperationTag();
            auto    tag4 = OperationTag();
            Scratch scratch3(tag3, ScratchPolicy::ZeroedBeforeAndAfter);
            Scratch scratch4(tag4, ScratchPolicy::ZeroedBeforeAndAfter);
            CHECK(tag3 == tag4);
            CHECK(scratch3 == scratch4);
        }

        SECTION("Different policies should not be equal")
        {
            Scratch scratch1(OperationTag(), ScratchPolicy::None);
            Scratch scratch2(OperationTag(), ScratchPolicy::ZeroedBeforeAndAfter);
            CHECK(!(scratch1 == scratch2));
        }

        SECTION("operator== is reflexive")
        {
            Scratch scratch(OperationTag(), ScratchPolicy::None);
            CHECK(scratch == scratch);
        }

        SECTION("ToString visitor")
        {
            Scratch         scratch(OperationTag(), ScratchPolicy::ZeroedBeforeAndAfter);
            ToStringVisitor toStringVisitor;

            std::string str = toStringVisitor(scratch);
            CHECK(str == "Scratch(ZeroedBeforeAndAfter)");
        }

        SECTION("VariableTypeVisitor returns DataType::None")
        {
            Scratch             scratch(OperationTag(), ScratchPolicy::None);
            VariableTypeVisitor typeVisitor;

            auto varType = typeVisitor(scratch);
            CHECK(varType.dataType == DataType::None);
        }

        SECTION("SetCommand visitor")
        {
            auto    command = std::make_shared<Command>();
            auto    tag     = command->allocateTag();
            Scratch scratch(tag, ScratchPolicy::None);

            SetCommand setCommandVisitor(command);
            setCommandVisitor(scratch);

            // Verify command was set (indirectly by checking tag can be assigned)
            auto scratchTag = command->addOperation(scratch);
            CHECK(!scratchTag.uninitialized());
        }
    }

    TEST_CASE("Scratch in Command", "[scratch][command]")
    {
        SECTION("Add Scratch operation to Command")
        {
            auto command = std::make_shared<Command>();
            auto tag     = command->allocateTag();

            auto scratchTag
                = command->addOperation(Scratch(tag, ScratchPolicy::ZeroedBeforeAndAfter));

            CHECK(tag.uninitialized() == false);
            CHECK(scratchTag.uninitialized() == false);
        }

        SECTION("Tag assignment works correctly")
        {
            auto command = std::make_shared<Command>();

            auto scratchTag1 = command->allocateTag();
            auto scratchTag2 = command->allocateTag();

            Scratch scratch1(scratchTag1, ScratchPolicy::None);
            Scratch scratch2(scratchTag2, ScratchPolicy::ZeroedBeforeAndAfter);
            auto    scratchOpTag1 = command->addOperation(scratch1);
            auto    scratchOpTag2 = command->addOperation(scratch2);

            CHECK(scratch1.getTag() == scratchTag1);
            CHECK(scratch2.getTag() == scratchTag2);
            CHECK(scratch1 != scratch2);
            CHECK(scratchOpTag1.uninitialized() == false);
            CHECK(scratchOpTag2.uninitialized() == false);
        }
    }

    TEST_CASE("Scratch allocator per policy", "[scratch][context]")
    {
        auto context = TestContext::ForDefaultTarget();

        SECTION("Context initializes all policies to zero")
        {
            for(size_t i = 0; i < static_cast<size_t>(ScratchPolicy::Count); ++i)
            {
                auto policy = static_cast<ScratchPolicy>(i);
                auto amount = context->getScratchAmount(policy);
                auto value  = rocRoller::Expression::evaluate(amount);
                CHECK(rocRoller::getUnsignedInt(value) == 0);
            }
        }

        SECTION("allocateScratch returns offset before allocation and accumulates size")
        {
            auto size1 = rocRoller::Expression::literal(100u);
            auto size2 = rocRoller::Expression::literal(200u);

            auto offset1 = context->allocateScratch(ScratchPolicy::None, size1);
            auto offset2 = context->allocateScratch(ScratchPolicy::None, size2);

            // First allocation should return offset 0
            CHECK(rocRoller::getUnsignedInt(rocRoller::Expression::evaluate(offset1)) == 0);
            // Second allocation should return offset 100 (after first allocation)
            CHECK(rocRoller::getUnsignedInt(rocRoller::Expression::evaluate(offset2)) == 100);

            // Total should now be 300
            auto amount = context->getScratchAmount(ScratchPolicy::None);
            auto value  = rocRoller::Expression::evaluate(amount);
            CHECK(rocRoller::getUnsignedInt(value) == 300);
        }

        SECTION("allocateScratch returns correct offset per policy")
        {
            auto size   = rocRoller::Expression::literal(500u);
            auto offset = context->allocateScratch(ScratchPolicy::ZeroedBeforeAndAfter, size);

            // First allocation should return offset 0
            CHECK(rocRoller::getUnsignedInt(rocRoller::Expression::evaluate(offset)) == 0);

            // Query total should now be 500
            auto amount = context->getScratchAmount(ScratchPolicy::ZeroedBeforeAndAfter);
            auto value  = rocRoller::Expression::evaluate(amount);
            CHECK(rocRoller::getUnsignedInt(value) == 500);
        }

        SECTION("Different policies have independent allocations")
        {
            auto sizeNone   = rocRoller::Expression::literal(100u);
            auto sizeZeroed = rocRoller::Expression::literal(200u);

            context->allocateScratch(ScratchPolicy::None, sizeNone);
            context->allocateScratch(ScratchPolicy::ZeroedBeforeAndAfter, sizeZeroed);

            auto amountNone   = context->getScratchAmount(ScratchPolicy::None);
            auto amountZeroed = context->getScratchAmount(ScratchPolicy::ZeroedBeforeAndAfter);

            auto valueNone   = rocRoller::Expression::evaluate(amountNone);
            auto valueZeroed = rocRoller::Expression::evaluate(amountZeroed);

            CHECK(rocRoller::getUnsignedInt(valueNone) == 100);
            CHECK(rocRoller::getUnsignedInt(valueZeroed) == 200);
        }
    }

    TEST_CASE("Utilities to get scratch policy name", "[scratch][utils]")
    {
        SECTION("Returns correct name for None policy")
        {
            auto name = rocRoller::getScratchName(ScratchPolicy::None);
            CHECK(name == "SCRATCH_None");
        }

        SECTION("Returns correct name for ZeroedBeforeAndAfter policy")
        {
            auto name = rocRoller::getScratchName(ScratchPolicy::ZeroedBeforeAndAfter);
            CHECK(name == "SCRATCH_ZeroedBeforeAndAfter");
        }

        SECTION("All policies have unique names")
        {
            std::set<std::string> names;
            for(size_t i = 0; i < static_cast<size_t>(ScratchPolicy::Count); ++i)
            {
                auto policy = static_cast<ScratchPolicy>(i);
                auto name   = rocRoller::getScratchName(policy);
                CHECK(names.find(name) == names.end());
                names.insert(name);
            }
            CHECK(names.size() == static_cast<size_t>(ScratchPolicy::Count));
        }
    }
}
