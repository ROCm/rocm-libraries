// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomSections.hpp"
#include "SimpleTest.hpp"

using namespace rocRoller;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("ConvertScales", "[datatypes]", E8M0)
{
    float prevValue = 0;
    for(int i = 0; i < 255; i++)
    {
        uint8_t scale    = i;
        auto    curValue = scaleToFloat<TestType>(static_cast<TestType>(scale));
        if(prevValue != 0)
            CHECK(curValue == prevValue * 2.0f);
        auto convertBack = static_cast<uint8_t>(floatToScale<TestType>(curValue));
        CHECK(scale == convertBack);

        prevValue = curValue;
    }
}
