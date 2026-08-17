// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("getOpselModifiers2xByte works", "[codegen][utils]")
{
    using namespace rocRoller::Arithmetic;

    CHECK(getOpselModifiers2xByte(0, 0) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(0, 1) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(0, 2) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[0,1]"));
    CHECK(getOpselModifiers2xByte(0, 3) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[0,1]"));

    CHECK(getOpselModifiers2xByte(1, 0) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(1, 1) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(1, 2) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[0,1]"));
    CHECK(getOpselModifiers2xByte(1, 3) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[0,1]"));

    CHECK(getOpselModifiers2xByte(2, 0) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(2, 1) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(2, 2) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[1,1]"));
    CHECK(getOpselModifiers2xByte(2, 3) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[1,1]"));

    CHECK(getOpselModifiers2xByte(3, 0) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(3, 1) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(3, 2) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[1,1]"));
    CHECK(getOpselModifiers2xByte(3, 3) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[1,1]"));

    CHECK_THROWS(getOpselModifiers2xByte(0, 4));
    CHECK_THROWS(getOpselModifiers2xByte(4, 0));
}
