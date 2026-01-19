/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "GenericContextFixture.hpp"
#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class ErrorTest : public SimpleFixture
    {
    };

    class ErrorFixtureTest : public GenericContextFixture
    {
    };

    using ErrorFixtureDeathTest = ErrorFixtureTest;

    TEST_F(ErrorFixtureDeathTest, BreakOnAssertFatal)
    {
        (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

        Settings::getInstance()->set(Settings::BreakOnThrow, true);

        EXPECT_DEATH({ AssertFatal(0 == 1); }, "");
    }

    TEST_F(ErrorFixtureDeathTest, BreakOnThrow)
    {
        (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

        Settings::getInstance()->set(Settings::BreakOnThrow, true);

        EXPECT_DEATH({ Throw<FatalError>("Error"); }, "");
    }
}
