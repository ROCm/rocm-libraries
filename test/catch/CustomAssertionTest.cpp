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

#include "CustomAssertions.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("CurrentTestHasTag fails test on duplicate test name", "[meta-test][!shouldfail]")
{
    // This would return true if it didn't fail an assertion due to duplicate test names.
    REQUIRE(rocRollerTests::CurrentTestHasTag("meta-test"));
}

TEST_CASE("CurrentTestHasTag fails test on duplicate test name", "[meta-test][asdf][!shouldfail]")
{

    // This would return true if it didn't fail an assertion due to duplicate test names.
    REQUIRE_FALSE(rocRollerTests::CurrentTestHasTag("fdsa"));
}

TEST_CASE("CurrentTestHasTag", "[meta-test][asdf]")
{
    REQUIRE(rocRollerTests::CurrentTestHasTag("meta-test"));
    REQUIRE(rocRollerTests::CurrentTestHasTag("asdf"));
    REQUIRE_FALSE(rocRollerTests::CurrentTestHasTag("fdsa"));
}

TEST_CASE("REQUIRE_TEST_TAG works", "[meta-test][asdf]")
{
    REQUIRE_TEST_TAG("asdf");
    REQUIRE_TEST_TAG("meta-test");
}

TEST_CASE("REQUIRE_TEST_TAG fails the test", "[meta-test][asdf][!shouldfail]")
{
    REQUIRE_TEST_TAG("asdf");
    REQUIRE_TEST_TAG("fdsa");
}
