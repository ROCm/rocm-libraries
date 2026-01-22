// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Annotate.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

TEST_CASE("AddLocation works", "[codegen][utility]")
{
    using namespace rocRoller;
    using namespace Catch::Matchers;

    // Here is a comment that can be expanded
    // or contracted to make the line numbers
    // below line up.

    CHECK(AddLocation().comment() == "26");
    CHECK(AddLocation({SourceLocationPart::File}).comment()
          == "shared/rocroller/test/catch/AnnotateInstructionsTest.cpp");

    CHECK_THAT(
        AddLocation(EnumBitset<SourceLocationPart>::All()).comment(),
        ContainsSubstring("void CATCH2")
            && ContainsSubstring("shared/rocroller/test/catch/AnnotateInstructionsTest.cpp:33:99"));

    auto generatorOne = []() -> Generator<Instruction> {
        co_yield_(Instruction("v_add_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_sub_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_mul_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_add_f32", {}, {}, {}, ""));
    };

    for(auto inst : generatorOne().map(AddLocation()))
        CHECK_THAT(inst.comments(), Contains("42"));

    for(auto inst :
        generatorOne().map(AddLocation({SourceLocationPart::File, SourceLocationPart::Line})))
        CHECK_THAT(inst.comments(),
                   Contains("shared/rocroller/test/catch/AnnotateInstructionsTest.cpp:46"));
}

TEST_CASE("AddComment works", "[codegen][utility]")
{
    using namespace rocRoller;
    using namespace Catch::Matchers;

    auto generatorOne = []() -> Generator<Instruction> {
        co_yield_(Instruction("v_add_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_sub_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_mul_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_add_f32", {}, {}, {}, ""));
    };

    for(auto inst : generatorOne().map(AddComment("foo")))
        CHECK_THAT(inst.comments(), Contains("foo"));

    auto generatorTwo
        = [&]() -> Generator<Instruction> { co_yield generatorOne().map(AddComment("bar")); };

    for(auto inst : generatorTwo())
        CHECK_THAT(inst.comments(), Contains("bar"));

    for(auto inst : generatorTwo().map(AddComment("baz")))
        CHECK_THAT(inst.comments(), Contains("bar") && Contains("baz"));
}

TEST_CASE("AddControlOp works", "[codegen][utility]")
{
    using namespace rocRoller;
    using namespace Catch::Matchers;

    auto generatorOne = []() -> Generator<Instruction> {
        co_yield_(Instruction("v_add_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_sub_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_mul_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_add_f32", {}, {}, {}, ""));
    };

    for(auto inst : generatorOne().map(AddControlOp(3)))
        CHECK(inst.controlOps() == std::vector{3});

    auto generatorTwo
        = [&]() -> Generator<Instruction> { co_yield generatorOne().map(AddControlOp(9)); };

    for(auto inst : generatorTwo())
        CHECK(inst.controlOps() == std::vector{9});

    for(auto inst : generatorTwo().map(AddControlOp(4)))
        CHECK(inst.controlOps() == std::vector({9, 4}));
}

TEST_CASE("combineCoexec function works", "[scheduling]")
{
    using namespace rocRoller;
    using namespace rocRoller::Scheduling;

    DisallowedCycles a = {{2, {CoexecCategory::Scalar}}};
    DisallowedCycles b = {{3, {CoexecCategory::VMEM}}};

    auto newB = b;

    SECTION("a")
    {
        combineCoexec(newB, a, 0);
        DisallowedCycles expectedB = {{2, {CoexecCategory::Scalar}}, {3, {CoexecCategory::VMEM}}};

        CHECK(newB == expectedB);
    }

    SECTION("b")
    {
        a[3] = {CoexecCategory::LDS};

        combineCoexec(newB, a, 0);
        DisallowedCycles expectedB
            = {{2, {CoexecCategory::Scalar}}, {3, {CoexecCategory::VMEM, CoexecCategory::LDS}}};

        CHECK(newB == expectedB);
    }

    SECTION("c")
    {
        a = {{4, {CoexecCategory::LDS}}};

        combineCoexec(newB, a, 0);
        DisallowedCycles expectedB = {{3, {CoexecCategory::VMEM}}, {4, {CoexecCategory::LDS}}};

        CHECK(newB == expectedB);
    }
}
