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

#include <common/SourceMatcher.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "SourceMatcher.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cctype>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using namespace rocRoller;

namespace rocRollerTest
{
    template <typename T_Exception>
    class WhatNormalizedEqualsMatcher : public Catch::Matchers::MatcherBase<T_Exception>
    {
    public:
        explicit WhatNormalizedEqualsMatcher(std::string expected)
            : m_expected(std::move(expected))
        {
        }

        bool match(T_Exception const& ex) const override
        {
            return NormalizedSource(std::string(ex.what()), false)
                   == NormalizedSource(m_expected, false);
        }

        std::string describe() const override
        {
            return "exception what() matches expected after NormalizedSource()";
        }

    private:
        std::string m_expected;
    };

    template <typename T_Exception>
    inline WhatNormalizedEqualsMatcher<T_Exception> WhatNormalizedEquals(std::string expected)
    {
        return WhatNormalizedEqualsMatcher<T_Exception>(std::move(expected));
    }

    template <typename T_Exception>
    class WhatContainsMatcher : public Catch::Matchers::MatcherBase<T_Exception>
    {
    public:
        explicit WhatContainsMatcher(std::string needle)
            : m_needle(std::move(needle))
        {
        }

        bool match(T_Exception const& ex) const override
        {
            return std::string(ex.what()).find(m_needle) != std::string::npos;
        }

        std::string describe() const override
        {
            return "exception what() contains substring: " + m_needle;
        }

    private:
        std::string m_needle;
    };

    template <typename T_Exception>
    inline WhatContainsMatcher<T_Exception> WhatContains(std::string needle)
    {
        return WhatContainsMatcher<T_Exception>(std::move(needle));
    }

    template <typename T_Exception, typename Pred>
    class WhatSatisfiesMatcher : public Catch::Matchers::MatcherBase<T_Exception>
    {
    public:
        WhatSatisfiesMatcher(Pred pred, std::string description)
            : m_pred(std::move(pred))
            , m_description(std::move(description))
        {
        }

        bool match(T_Exception const& ex) const override
        {
            return m_pred(std::string(ex.what()));
        }

        std::string describe() const override
        {
            return m_description;
        }

    private:
        Pred        m_pred;
        std::string m_description;
    };

    template <typename T_Exception, typename Pred>
    inline auto WhatSatisfies(Pred pred, std::string description)
    {
        return WhatSatisfiesMatcher<T_Exception, Pred>(std::move(pred), std::move(description));
    }
}

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

    template <typename T_Exception>
    struct WhatNormalizedEqualsMatcher : Catch::Matchers::MatcherBase<T_Exception>
    {
        explicit WhatNormalizedEqualsMatcher(std::string expected)
            : m_expected(std::move(expected))
        {
        }

        bool match(T_Exception const& ex) const override
        {
            return NormalizedSource(ex.what()) == NormalizedSource(m_expected);
        }

        std::string describe() const override
        {
            return "exception what() matches expected output after NormalizedSource()";
        }

    private:
        std::string m_expected;
    };

    template <typename T_Exception>
    inline WhatNormalizedEqualsMatcher<T_Exception> WhatNormalizedEquals(std::string expected)
    {
        return WhatNormalizedEqualsMatcher<T_Exception>(std::move(expected));
    }

    template <typename T_Exception>
    struct WhatHasPrefixAndContainsMatcher : Catch::Matchers::MatcherBase<T_Exception>
    {
        WhatHasPrefixAndContainsMatcher(std::string requiredFileToken,
                                        std::string requiredSubstring,
                                        bool        mustNotContainErrorHpp)
            : m_requiredFileToken(std::move(requiredFileToken))
            , m_requiredSubstring(std::move(requiredSubstring))
            , m_mustNotContainErrorHpp(mustNotContainErrorHpp)
        {
        }

        bool match(T_Exception const& ex) const override
        {
            std::string output = ex.what();

            if(!HasFileLinePrefix(output, m_requiredFileToken))
                return false;

            if(output.find(m_requiredSubstring) == std::string::npos)
                return false;

            if(m_mustNotContainErrorHpp && output.find("Error.hpp") != std::string::npos)
                return false;

            return true;
        }

        std::string describe() const override
        {
            return "exception what() has <file>:<line>: prefix and contains required substring";
        }

    private:
        std::string m_requiredFileToken;
        std::string m_requiredSubstring;
        bool        m_mustNotContainErrorHpp;
    };

    template <typename T_Exception>
    inline WhatHasPrefixAndContainsMatcher<T_Exception>
        WhatHasPrefixAndContains(std::string requiredFileToken,
                                 std::string requiredSubstring,
                                 bool        mustNotContainErrorHpp = false)
    {
        return WhatHasPrefixAndContainsMatcher<T_Exception>(
            std::move(requiredFileToken), std::move(requiredSubstring), mustNotContainErrorHpp);
    }

    [[noreturn]] void HelperThatThrows(std::source_location loc = std::source_location::current())
    {
        (void)loc;
        Throw<rocRoller::FatalError>("Helper throw test");
    }

    bool HelperThrowMessageOk(std::string const& output)
    {
        if(output.find("ErrorTest.cpp") == std::string::npos)
            return false;
        if(output.find("Helper throw test") == std::string::npos)
            return false;
        return HasFileLinePrefix(output, "ErrorTest.cpp");
    }

    bool WhatHasXEquals7(std::string const& output)
    {
        return output.find("x = 7") != std::string::npos;
    }

    static void requireDeath(void (*fn)())
    {
        pid_t pid = ::fork();
        REQUIRE(pid >= 0);

        if(pid == 0)
        {
            fn();
            _exit(0);
        }

        int status = 0;
        REQUIRE(::waitpid(pid, &status, 0) == pid);

        if(WIFSIGNALED(status))
            return;

        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) != 0);
    }
}

TEST_CASE("ErrorTest: BaseErrorTest", "[utils][error]")
{
    REQUIRE_THROWS_AS(throw Error("Base rocRoller Error"), Error);
}

TEST_CASE("ErrorTest: BaseFatalErrorTest", "[utils][error]")
{
    REQUIRE_THROWS_AS(throw FatalError("Fatal rocRoller Error"), FatalError);
}

TEST_CASE("ErrorTest: BaseRecoverableErrorTest", "[utils][error]")
{
    REQUIRE_THROWS_AS(throw RecoverableError("Recoverable rocRoller Error"), RecoverableError);
}

TEST_CASE("ErrorTest: BaseFileNameTest", "[utils][error]")
{
    REQUIRE(std::string(GetBaseFileName("/absolute/path/to/file.txt"))
            == "/absolute/path/to/file.txt");

    REQUIRE(std::string(GetBaseFileName("./relative/local/path/to/file.txt"))
            == "relative/local/path/to/file.txt");

    REQUIRE(std::string(GetBaseFileName("../relative/path/to/file.txt"))
            == "relative/path/to/file.txt");

    REQUIRE(std::string(GetBaseFileName("../../../long/relative/path/to/file.txt"))
            == "long/relative/path/to/file.txt");

    REQUIRE(std::string(GetBaseFileName("./../../../long/local/relative/path/to/file.txt"))
            == "long/local/relative/path/to/file.txt");

    REQUIRE(std::string(GetBaseFileName("./")) == "");
    REQUIRE(std::string(GetBaseFileName("../")) == "");
}

TEST_CASE("ErrorTest: FatalErrorTest", "[utils][error]")
{
    int         IntA    = 5;
    int         IntB    = 3;
    std::string message = "FatalError Test";

    REQUIRE_NOTHROW([&] { AssertFatal(IntA > IntB, ShowValue(IntA), message); }());
    REQUIRE_THROWS_AS([&] { AssertFatal(IntA < IntB, ShowValue(IntB), message); }(), FatalError);

    int expectedLine = 0;

    auto throwFatal = [&] {
        expectedLine = __LINE__ + 1;
        AssertFatal(IntA < IntB, ShowValue(IntA), message);
    };

    REQUIRE_THROWS_MATCHES(
        throwFatal(),
        FatalError,
        rocRollerTest::WhatSatisfies<FatalError>(
            [&](std::string const& out) {
                std::string prefix = rocRoller::concatenate(
                    rocRoller::GetBaseFileName(__FILE__), ":", expectedLine, ":");

                if(out.rfind(prefix, 0) != 0)
                    return false;

                if(out.find("FatalError(IntA < IntB)") == std::string::npos)
                    return false;
                if(out.find("IntA = 5") == std::string::npos)
                    return false;
                if(out.find("FatalError Test") == std::string::npos)
                    return false;

                return true;
            },
            "what() begins with call-site file:line and includes condition, ShowValue, and "
            "message"));
}

TEST_CASE("ErrorTest: RecoverableErrorTest", "[utils][error]")
{
    std::string StrA    = "StrA";
    std::string StrB    = "StrB";
    std::string message = "RecoverableError Test";

    REQUIRE_NOTHROW([&] { AssertRecoverable(StrA != StrB, ShowValue(StrA), message); }());
    REQUIRE_THROWS_AS([&] { AssertRecoverable(StrA == StrB, ShowValue(StrB), message); }(),
                      RecoverableError);

    int  expectedLine = __LINE__ + 2;
    auto throwRecov
        = [&] { AssertRecoverable(StrA == StrB, ShowValue(StrA), ShowValue(StrB), message); };

    std::string expected = rocRoller::concatenate("\n",
                                                  rocRoller::GetBaseFileName(__FILE__),
                                                  ":",
                                                  expectedLine,
                                                  ": RecoverableError(StrA == StrB)\n",
                                                  ShowValue(StrA),
                                                  ShowValue(StrB),
                                                  message);

    REQUIRE_THROWS_MATCHES(throwRecov(),
                           RecoverableError,
                           rocRollerTest::WhatNormalizedEquals<RecoverableError>(expected));
}

TEST_CASE("ErrorTest: DontBreakOnThrow", "[utils][error]")
{
    rocRoller::Settings::getInstance()->set(rocRoller::Settings::BreakOnThrow, false);

    REQUIRE_THROWS_AS([&] { rocRoller::Throw<rocRoller::FatalError>("Error"); }(),
                      rocRoller::FatalError);

    Settings::reset();
}

TEST_CASE("ErrorTest: BreakOnThrow", "[utils][error][death]")
{
    requireDeath([] {
        rocRoller::Settings::getInstance()->set(rocRoller::Settings::BreakOnThrow, true);
        rocRoller::Throw<rocRoller::FatalError>("Error");
    });
}

TEST_CASE("ErrorTest: BreakOnAssertFatal", "[utils][error][death]")
{
    requireDeath([] {
        rocRoller::Settings::getInstance()->set(rocRoller::Settings::BreakOnThrow, true);
        AssertFatal(0 == 1);
    });
}

TEST_CASE("ErrorTest: ThrowIncludesSourceLocation", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    REQUIRE_THROWS_MATCHES(
        Throw<FatalError>("Throw location test"),
        FatalError,
        WhatHasPrefixAndContains<FatalError>("ErrorTest.cpp", "Throw location test"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowRecoverableIncludesSourceLocation", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    REQUIRE_THROWS_MATCHES(
        Throw<RecoverableError>("Recoverable throw test"),
        RecoverableError,
        WhatHasPrefixAndContains<RecoverableError>("ErrorTest.cpp", "Recoverable throw test"));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowMultiPieceMessageIncludesSourceLocation", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    int x = 7;

    REQUIRE_THROWS_MATCHES(Throw<FatalError>("Multi piece: ", ShowValue(x)),
                           FatalError,
                           WhatHasPrefixAndContains<FatalError>("ErrorTest.cpp", "Multi piece: "));

    REQUIRE_THROWS_MATCHES(
        [&] { Throw<FatalError>("Multi piece: ", ShowValue(x)); }(),
        FatalError,
        rocRollerTest::WhatSatisfies<FatalError>(
            &WhatHasXEquals7, "exception what() contains ShowValue expansion 'x = 7'"));
}

TEST_CASE("ErrorTest: ThrowDoesNotReportErrorHppLocation", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    REQUIRE_THROWS_MATCHES(Throw<FatalError>("Location sanity check"),
                           FatalError,
                           WhatHasPrefixAndContains<FatalError>("ErrorTest.cpp",
                                                                "Location sanity check",
                                                                /*mustNotContainErrorHpp=*/true));

    Settings::reset();
}

TEST_CASE("ErrorTest: ThrowReportsHelperCallSiteWhenWrappedInHelper", "[utils][error]")
{
    Settings::getInstance()->set(Settings::BreakOnThrow, false);

    REQUIRE_THROWS_MATCHES(
        HelperThatThrows(),
        FatalError,
        rocRollerTest::WhatSatisfies<FatalError>(
            &HelperThrowMessageOk,
            "exception what() has ErrorTest.cpp:<line>: and contains 'Helper throw test'"));

    Settings::reset();
}