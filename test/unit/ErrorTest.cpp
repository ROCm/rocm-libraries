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
            test/unit/ErrorTest.cpp:66: FatalError(IntA < IntB)
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
            test/unit/ErrorTest.cpp:99: RecoverableError(StrA == StrB)
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
}
