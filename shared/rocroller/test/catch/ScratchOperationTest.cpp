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

#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Operations/Scratch.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

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
}
