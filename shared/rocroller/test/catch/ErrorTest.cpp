// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <common/SourceMatcher.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cctype>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using namespace rocRoller;

namespace
{
    inline bool HasFileLinePrefix(std::string const& output, std::string const& requiredFileToken)
    {
        auto firstColon = output.find(':');
        if(firstColon == std::string::npos)
            return false;

        std::string filePortion = output.substr(0, firstColon);
        if(filePortion.find(requiredFileToken) == std::string::npos)
            return false;

        auto secondColon = output.find(':', firstColon + 1);
        if(secondColon == std::string::npos)
            return false;

        if(secondColon <= firstColon + 1)
            return false;

        for(size_t i = firstColon + 1; i < secondColon; ++i)
        {
            if(!std::isdigit(static_cast<unsigned char>(output[i])))
                return false;
        }

        return true;
    }

    [[noreturn]] void HelperThatThrows(std::source_location loc = std::source_location::current())
    {
        (void)loc;
        Throw<rocRoller::FatalError>("Helper throw test");
    }
}

TEST_CASE("ErrorTest: BaseErrorTest", "[utils][error]")
{
    CHECK_THROWS_AS(([] { throw Error("Base rocRoller Error"); }()), Error);
}

TEST_CASE("ErrorTest: BaseFatalErrorTest", "[utils][error]")
{
    CHECK_THROWS_AS(([] { throw FatalError("Fatal rocRoller Error"); }()), FatalError);
}

TEST_CASE("ErrorTest: BaseRecoverableErrorTest", "[utils][error]")
{
    CHECK_THROWS_AS(([] { throw RecoverableError("Recoverable rocRoller Error"); }()),
                    RecoverableError);
}

TEST_CASE("ErrorTest: BaseFileNameTest", "[utils][error]")
{
    CHECK(std::string(GetBaseFileName("/absolute/path/to/file.txt"))
          == "/absolute/path/to/file.txt");
    CHECK(std::string(GetBaseFileName("./relative/local/path/to/file.txt"))
          == "relative/local/path/to/file.txt");
    CHECK(std::string(GetBaseFileName("../relative/path/to/file.txt"))
          == "relative/path/to/file.txt");
    CHECK(std::string(GetBaseFileName("../../../long/relative/path/to/file.txt"))
          == "long/relative/path/to/file.txt");
    CHECK(std::string(GetBaseFileName("./../../../long/local/relative/path/to/file.txt"))
          == "long/local/relative/path/to/file.txt");
    CHECK(std::string(GetBaseFileName("./")) == "");
    CHECK(std::string(GetBaseFileName("../")) == "");
}

TEST_CASE("ErrorTest: FatalErrorTest", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    int         IntA    = 5;
    int         IntB    = 3;
    std::string message = "FatalError Test";

    CHECK_NOTHROW(([&] { AssertFatal(IntA > IntB, ShowValue(IntA), message); }()));
    CHECK_THROWS_AS(([&] { AssertFatal(IntA < IntB, ShowValue(IntB), message); }()), FatalError);

    int  expectedLine = 0;
    auto throwFatal   = [&] {
        expectedLine = __LINE__ + 1;
        AssertFatal(IntA < IntB, ShowValue(IntA), message);
    };

    CHECK_THROWS_MATCHES(
        throwFatal(),
        FatalError,
        Predicate<FatalError>(
            [&](FatalError const& ex) -> bool {
                std::string out = ex.what();

                std::string prefix
                    = rocRoller::concatenate(GetBaseFileName(__FILE__), ":", expectedLine, ":");
                if(out.rfind(prefix, 0) != 0)
                    return false;

                if(out.find("FatalError") == std::string::npos)
                    return false;
                if(out.find("IntA < IntB") == std::string::npos)
                    return false;
                if(out.find("IntA = 5") == std::string::npos)
                    return false;
                if(out.find("FatalError Test") == std::string::npos)
                    return false;

                return true;
            },
            "what() begins with call-site file:line and contains condition + ShowValue + message"));
}

TEST_CASE("ErrorTest: RecoverableErrorTest", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    std::string StrA    = "StrA";
    std::string StrB    = "StrB";
    std::string message = "RecoverableError Test";

    CHECK_NOTHROW(([&] { AssertRecoverable(StrA != StrB, ShowValue(StrA), message); }()));
    CHECK_THROWS_AS(([&] { AssertRecoverable(StrA == StrB, ShowValue(StrB), message); }()),
                    RecoverableError);

    int  expectedLine = 0;
    auto throwRecov   = [&] {
        expectedLine = __LINE__ + 1;
        AssertRecoverable(StrA == StrB, ShowValue(StrA), ShowValue(StrB), message);
    };

    CHECK_THROWS_MATCHES(
        throwRecov(),
        RecoverableError,
        Predicate<RecoverableError>(
            [&](RecoverableError const& ex) -> bool {
                std::string out = ex.what();

                std::string prefix
                    = rocRoller::concatenate(GetBaseFileName(__FILE__), ":", expectedLine, ":");
                if(out.rfind(prefix, 0) != 0)
                    return false;

                if(out.find("RecoverableError") == std::string::npos)
                    return false;
                if(out.find("StrA == StrB") == std::string::npos)
                    return false;
                if(out.find("StrA = StrA") == std::string::npos)
                    return false;
                if(out.find("StrB = StrB") == std::string::npos)
                    return false;
                if(out.find("RecoverableError Test") == std::string::npos)
                    return false;

                return true;
            },
            "what() begins with call-site file:line and contains condition + ShowValue + message"));
}

TEST_CASE("ErrorTest: DontBreakOnThrow", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    CHECK_THROWS_AS(([&] { Throw<FatalError>("Error"); }()), FatalError);

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowIncludesSourceLocation", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    CHECK_THROWS_MATCHES(([&] { Throw<FatalError>("Throw location test"); }()),
                         FatalError,
                         Predicate<FatalError>(
                             [&](FatalError const& ex) -> bool {
                                 std::string out = ex.what();
                                 if(!HasFileLinePrefix(out, "ErrorTest.cpp"))
                                     return false;
                                 if(out.find("Throw location test") == std::string::npos)
                                     return false;
                                 return true;
                             },
                             "what() has ErrorTest.cpp:<line>: prefix and contains message"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowRecoverableIncludesSourceLocation", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    CHECK_THROWS_MATCHES(([&] { Throw<RecoverableError>("Recoverable throw test"); }()),
                         RecoverableError,
                         Predicate<RecoverableError>(
                             [&](RecoverableError const& ex) -> bool {
                                 std::string out = ex.what();
                                 if(!HasFileLinePrefix(out, "ErrorTest.cpp"))
                                     return false;
                                 if(out.find("Recoverable throw test") == std::string::npos)
                                     return false;
                                 return true;
                             },
                             "what() has ErrorTest.cpp:<line>: prefix and contains message"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowMultiPieceMessageIncludesSourceLocation", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    int x = 7;

    CHECK_THROWS_MATCHES(([&] { Throw<FatalError>("Multi piece: ", ShowValue(x)); }()),
                         FatalError,
                         Predicate<FatalError>(
                             [&](FatalError const& ex) -> bool {
                                 std::string out = ex.what();
                                 if(!HasFileLinePrefix(out, "ErrorTest.cpp"))
                                     return false;
                                 if(out.find("Multi piece: ") == std::string::npos)
                                     return false;
                                 if(out.find("x = 7") == std::string::npos)
                                     return false;
                                 return true;
                             },
                             "what() has ErrorTest.cpp:<line>: prefix and contains multi-piece "
                             "message + ShowValue"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowDoesNotReportErrorHppLocation", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    CHECK_THROWS_MATCHES(
        ([&] { Throw<FatalError>("Location sanity check"); }()),
        FatalError,
        Predicate<FatalError>(
            [&](FatalError const& ex) -> bool {
                std::string out = ex.what();
                if(out.find("Error.hpp") != std::string::npos)
                    return false;
                if(!HasFileLinePrefix(out, "ErrorTest.cpp"))
                    return false;
                if(out.find("Location sanity check") == std::string::npos)
                    return false;
                return true;
            },
            "what() does not mention Error.hpp and has ErrorTest.cpp:<line>: prefix"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowReportsHelperCallSiteWhenWrappedInHelper", "[utils][error]")
{
    using Catch::Matchers::Predicate;

    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    CHECK_THROWS_MATCHES(
        ([&] { HelperThatThrows(); }()),
        FatalError,
        Predicate<FatalError>(
            [&](FatalError const& ex) -> bool {
                std::string out = ex.what();
                if(!HasFileLinePrefix(out, "ErrorTest.cpp"))
                    return false;
                if(out.find("Helper throw test") == std::string::npos)
                    return false;
                return true;
            },
            "what() has ErrorTest.cpp:<line>: prefix and contains helper message"));

    Settings::reset();
}
