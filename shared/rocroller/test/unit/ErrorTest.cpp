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

    TEST_F(ErrorTest, BaseErrorTest)
    {
        EXPECT_THROW({ throw Error("Base rocRoller Error"); }, Error);
    }

    TEST_F(ErrorTest, BaseFatalErrorTest)
    {
        EXPECT_THROW({ throw FatalError("Fatal rocRoller Error"); }, FatalError);
    }

    TEST_F(ErrorTest, BaseRecoverableErrorTest)
    {
        EXPECT_THROW({ throw RecoverableError("Recoverable rocRoller Error"); }, RecoverableError);
    }

    TEST_F(ErrorTest, BaseFileNameTest)
    {
        EXPECT_STREQ("/absolute/path/to/file.txt", GetBaseFileName("/absolute/path/to/file.txt"));
        EXPECT_STREQ("relative/local/path/to/file.txt",
                     GetBaseFileName("./relative/local/path/to/file.txt"));
        EXPECT_STREQ("relative/path/to/file.txt", GetBaseFileName("../relative/path/to/file.txt"));
        EXPECT_STREQ("long/relative/path/to/file.txt",
                     GetBaseFileName("../../../long/relative/path/to/file.txt"));
        EXPECT_STREQ("long/local/relative/path/to/file.txt",
                     GetBaseFileName("./../../../long/local/relative/path/to/file.txt"));
        EXPECT_STREQ("", GetBaseFileName("./"));
        EXPECT_STREQ("", GetBaseFileName("../"));
    }

    TEST_F(ErrorTest, FatalErrorTest)
    {
        int         IntA    = 5;
        int         IntB    = 3;
        std::string message = "FatalError Test";

        EXPECT_NO_THROW({ AssertFatal(IntA > IntB, ShowValue(IntA), message); });
        EXPECT_THROW({ AssertFatal(IntA < IntB, ShowValue(IntB), message); }, FatalError);

        std::string expected = R"(
            shared/rocroller/test/unit/ErrorTest.cpp:92: FatalError(IntA < IntB)
                IntA = 5
            FatalError Test)";

        try
        {
            AssertFatal(IntA < IntB, ShowValue(IntA), message);
            FAIL() << "Expected FatalError to be thrown";
        }
        catch(FatalError& e)
        {
            std::string output = e.what();
            EXPECT_EQ(NormalizedSource(output), NormalizedSource(expected))
                << std::source_location::current().file_name();
        }
        catch(...)
        {
            FAIL() << "Caught unexpected error, expected FatalError";
        }
    }

    TEST_F(ErrorTest, RecoverableErrorTest)
    {
        std::string StrA    = "StrA";
        std::string StrB    = "StrB";
        std::string message = "RecoverableError Test";

        EXPECT_NO_THROW({ AssertRecoverable(StrA != StrB, ShowValue(StrA), message); });
        EXPECT_THROW({ AssertRecoverable(StrA == StrB, ShowValue(StrB), message); },
                     RecoverableError);

        std::string expected = R"(
            shared/rocroller/test/unit/ErrorTest.cpp:125: RecoverableError(StrA == StrB)
                StrA = StrA
                StrB = StrB
            RecoverableError Test)";

        try
        {
            AssertRecoverable(StrA == StrB, ShowValue(StrA), ShowValue(StrB), message);
            FAIL() << "Expected RecoverableError to be thrown";
        }
        catch(RecoverableError& e)
        {
            std::string output = e.what();
            EXPECT_EQ(NormalizedSource(output), NormalizedSource(expected))
                << std::source_location::current().file_name();
        }
        catch(...)
        {
            FAIL() << "Caught unexpected error, expected RecoverableError";
        }
    }

    TEST_F(ErrorFixtureTest, DontBreakOnThrow)
    {
        (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        EXPECT_ANY_THROW({ Throw<FatalError>("Error"); });
    }

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

    TEST_F(ErrorTest, ThrowIncludesSourceLocation)
    {
        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        try
        {
            Throw<FatalError>("Throw location test");
            FAIL() << "Expected FatalError to be thrown";
        }
        catch(const FatalError& e)
        {
            std::string output = e.what();

            auto firstColon  = output.find(':');
            auto secondColon = (firstColon == std::string::npos) ? std::string::npos
                                                                 : output.find(':', firstColon + 1);

            EXPECT_NE(firstColon, std::string::npos)
                << "Expected a file:line prefix in error message, but got:\n"
                << output;
            EXPECT_NE(secondColon, std::string::npos)
                << "Expected a file:line prefix with a line number in error message, but got:\n"
                << output;

            std::string filePortion = output.substr(0, firstColon);
            EXPECT_NE(filePortion.rfind("ErrorTest.cpp"), std::string::npos)
                << "Expected error message to originate from ErrorTest.cpp, but got:\n"
                << output;

            EXPECT_NE(output.find("Throw location test"), std::string::npos)
                << "Expected user message in error output, but got:\n"
                << output;
        }
        catch(...)
        {
            FAIL() << "Caught unexpected exception type, expected FatalError";
        }
    }

    TEST_F(ErrorTest, ThrowRecoverableIncludesSourceLocation)
    {
        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        try
        {
            Throw<RecoverableError>("Recoverable throw test");
            FAIL() << "Expected RecoverableError to be thrown";
        }
        catch(const RecoverableError& e)
        {
            std::string output = e.what();

            EXPECT_NE(output.find("ErrorTest.cpp"), std::string::npos)
                << "Expected error message to originate from ErrorTest.cpp, but got:\n"
                << output;

            EXPECT_NE(output.find("Recoverable throw test"), std::string::npos)
                << "Expected user message in error output, but got:\n"
                << output;
        }
        catch(...)
        {
            FAIL() << "Caught unexpected exception type, expected RecoverableError";
        }
    }

    TEST_F(ErrorTest, ThrowMultiPieceMessageIncludesSourceLocation)
    {
        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        int x = 7;

        try
        {
            Throw<FatalError>("Multi piece: ", ShowValue(x));
            FAIL() << "Expected FatalError to be thrown";
        }
        catch(const FatalError& e)
        {
            std::string output = e.what();

            EXPECT_NE(output.find("ErrorTest.cpp"), std::string::npos)
                << "Expected call-site file name in error output, but got:\n"
                << output;

            EXPECT_NE(output.find("Multi piece: "), std::string::npos)
                << "Expected prefix message, but got:\n"
                << output;

            EXPECT_NE(output.find("x = 7"), std::string::npos)
                << "Expected ShowValue expansion, but got:\n"
                << output;
        }
    }
    TEST_F(ErrorTest, ThrowDoesNotReportErrorHppLocation)
    {
        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        try
        {
            Throw<FatalError>("Location sanity check");
            FAIL() << "Expected FatalError to be thrown";
        }
        catch(const FatalError& e)
        {
            std::string output = e.what();

            EXPECT_EQ(output.find("Error.hpp"), std::string::npos)
                << "Direct Throw<> should not report Error.hpp, but got:\n"
                << output;

            EXPECT_NE(output.find("ErrorTest.cpp"), std::string::npos)
                << "Expected call-site file name, but got:\n"
                << output;
        }
    }
    namespace
    {
        [[noreturn]] void HelperThatThrows(std::source_location loc
                                           = std::source_location::current())
        {
            (void)loc;
            Throw<rocRoller::FatalError>("Helper throw test");
        }
    }

    TEST_F(ErrorTest, ThrowReportsHelperCallSiteWhenWrappedInHelper)
    {
        Settings::getInstance()->set(Settings::BreakOnThrow, false);

        try
        {
            HelperThatThrows();
            FAIL() << "Expected FatalError to be thrown";
        }
        catch(const FatalError& e)
        {
            std::string output = e.what();

            EXPECT_NE(output.find("ErrorTest.cpp"), std::string::npos)
                << "Expected file name in error output, but got:\n"
                << output;

            auto firstColon = output.find(':');
            ASSERT_NE(firstColon, std::string::npos) << output;

            auto secondColon = output.find(':', firstColon + 1);
            ASSERT_NE(secondColon, std::string::npos) << output;

            std::string lineStr = output.substr(firstColon + 1, secondColon - (firstColon + 1));
            ASSERT_FALSE(lineStr.empty()) << output;

            int line = 0;
            try
            {
                line = std::stoi(lineStr);
            }
            catch(...)
            {
                FAIL() << "Failed to parse line number from error output. Got:\n" << output;
            }

            EXPECT_GT(line, 0) << "Expected positive line number, got " << line << " from:\n"
                               << output;

            EXPECT_NE(output.find("Helper throw test"), std::string::npos)
                << "Expected user message in error output, but got:\n"
                << output;
        }
        catch(...)
        {
            FAIL() << "Caught unexpected exception type, expected FatalError";
        }
    }
}
