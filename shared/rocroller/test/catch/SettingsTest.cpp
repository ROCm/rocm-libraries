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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <omp.h>
#include <thread>

#include "SimpleTest.hpp"
#include "common/SourceMatcher.hpp"
#include <rocRoller/Utilities/Settings_fwd.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace SettingsTest
{
    class EnvSettingsTest : public SimpleTest
    {
    public:
        EnvSettingsTest()
            : SimpleTest()
        {
            for(auto const& setting : SettingsOptionBase::instances())
            {
                std::optional<std::string> val;
                if(auto ptr = getenv(setting->name.c_str()))
                {
                    val = ptr;
                }
                m_envVars[setting->name] = val;
            }

            // This is not a setting option, but affects the env
            if(auto ptr = getenv(Settings::BitfieldName.c_str()))
            {
                m_envVars[Settings::BitfieldName] = ptr;
            }
            else
            {
                m_envVars[Settings::BitfieldName] = std::nullopt;
            }

            setenv(Settings::BitfieldName.c_str(), "0xFFFFFFFF", 1);
            setenv(Settings::LogConsole.name.c_str(), "0", 1);
            setenv(Settings::AssemblyFile.name.c_str(), "assemblyFileTest.s", 1);
            setenv(Settings::RandomSeed.name.c_str(), "31415", 1);
            setenv(Settings::Scheduler.name.c_str(), "invalidScheduler", 1);
        }

        ~EnvSettingsTest() override
        {
            for(auto entry : m_envVars)
            {
                if(entry.second.has_value())
                {
                    setenv(entry.first.c_str(), entry.second.value().c_str(), 1);
                }
                else
                {
                    unsetenv(entry.first.c_str());
                }
            }

            Settings::reset();
        }

    private:
        std::map<std::string, std::optional<std::string>> m_envVars;
    };

    TEST_CASE("Basic settings behavior", "[settings]")
    {
        SimpleTest t;

        SECTION("Default values")
        {

            auto settings = Settings::getInstance();

            CHECK(settings->get(Settings::LogConsole) == Settings::LogConsole.defaultValue);
            CHECK(settings->get(Settings::SaveAssembly) == Settings::SaveAssembly.defaultValue);
            CHECK_THAT(settings->get(Settings::AssemblyFile),
                       Equals(Settings::AssemblyFile.defaultValue));
            CHECK(settings->get(Settings::BreakOnThrow) == Settings::BreakOnThrow.defaultValue);
            CHECK_THAT(settings->get(Settings::LogFile), Equals(Settings::LogFile.defaultValue));
            CHECK(settings->get(Settings::LogLvl) == Settings::LogLvl.defaultValue);
            CHECK(settings->get(Settings::RandomSeed) == Settings::RandomSeed.defaultValue);
            CHECK(settings->get(Settings::Scheduler) == Settings::Scheduler.defaultValue);
        }

        SECTION("Log levels")
        {
            std::ostringstream out;
            out << LogLevel::None << std::endl;
            out << LogLevel::Error << std::endl;
            out << LogLevel::Warning << std::endl;
            out << LogLevel::Terse << std::endl;
            out << LogLevel::Verbose << std::endl;
            out << LogLevel::Debug << std::endl;
            out << LogLevel::Count << std::endl;

            std::string stringify = "";
            stringify += toString(LogLevel::None) + '\n';
            stringify += toString(LogLevel::Error) + '\n';
            stringify += toString(LogLevel::Warning) + '\n';
            stringify += toString(LogLevel::Terse) + '\n';
            stringify += toString(LogLevel::Verbose) + '\n';
            stringify += toString(LogLevel::Debug) + '\n';
            stringify += toString(LogLevel::Count) + '\n';

            std::string expected = R"(
            None
            Error
            Warning
            Terse
            Verbose
            Debug
            Count
            )";

            CHECK_THAT(NormalizedSource(expected), Equals(NormalizedSource(out.str())));
            CHECK_THAT(NormalizedSource(expected), Equals(NormalizedSource(stringify)));

            CHECK(fromString<LogLevel>("None") == LogLevel::None);
            CHECK(fromString<LogLevel>("Error") == LogLevel::Error);
            CHECK(fromString<LogLevel>("Warning") == LogLevel::Warning);
            CHECK(fromString<LogLevel>("Terse") == LogLevel::Terse);
            CHECK(fromString<LogLevel>("Verbose") == LogLevel::Verbose);
            CHECK(fromString<LogLevel>("Debug") == LogLevel::Debug);
            CHECK_THROWS(fromString<LogLevel>("Count"));
        }

        SECTION("Invalid settings are not allowed")
        {
            auto settings = Settings::getInstance();

            settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
            CHECK_THROWS_AS(settings->set(Settings::Scheduler, "invalidValue"), FatalError);
            CHECK(settings->get(Settings::Scheduler)
                  == Scheduling::SchedulerProcedure::Cooperative);

            CHECK_THROWS_AS(settings->set(Settings::LogConsole, "invalidValue"), FatalError);

            Settings::reset();
        }

        SECTION("Settings should be helpful")
        {
            SimpleTest  t;
            auto        settings = Settings::getInstance();
            std::string help     = settings->help();
            CHECK_THAT(help, ContainsSubstring("default"));
            CHECK_THAT(help, ContainsSubstring("bit"));
        }

        SECTION("Settings should be thread-safe")
        {
            auto settings = Settings::getInstance();

            unsigned int numCores = std::thread::hardware_concurrency();
            CHECK(numCores > 2);
            const unsigned int numTestThreads = (numCores > 8) ? 8 : numCores;

            auto minIters        = 1000;
            auto durationSeconds = 2;
            auto end
                = std::chrono::high_resolution_clock::now() + std::chrono::seconds(durationSeconds);

            size_t numUnexpectedLogLevels = 0;
            size_t numIters               = 0;

#pragma omp parallel num_threads(numTestThreads) reduction(+ : numUnexpectedLogLevels) \
    reduction(+ : numIters)
            {
                size_t iters = 0;
                int    tid   = omp_get_thread_num();
                while(std::chrono::high_resolution_clock::now() < end || iters <= minIters)
                {
                    iters++;
                    if(tid % 2 == 0)
                    {
                        auto logLevel = settings->get(Settings::LogLvl);
                        auto isNoneOrError
                            = (logLevel == LogLevel::None) || (logLevel == LogLevel::Error);
                        if(!isNoneOrError)
                            ++numUnexpectedLogLevels;
                    }
                    else
                    {
                        auto setError = (rand() % 2) == 0;
                        settings->set(Settings::LogLvl,
                                      setError ? LogLevel::Error : LogLevel::None);
                    }
                    usleep(rand() % 100);
                }
                numIters += iters;
            }
            CHECK(numUnexpectedLogLevels == 0);
            CHECK(numIters >= numTestThreads * minIters);
        }

        Settings::reset();
    }

    TEST_CASE("Settings with associated envvars", "[settings]")
    {
        EnvSettingsTest t;

        SECTION("No infinite recursion")
        {
            // Settings ctor should not get from env,
            // otherwise it may throw without a prior settings instance
            // and infinitely recurse
            Settings::reset();
            // unsetenv bitfield revert BreakOnThrow to false (default value)
            unsetenv(Settings::BitfieldName.c_str());
            auto settings = Settings::getInstance();
            CHECK_THROWS_AS(settings->get(Settings::Scheduler), FatalError);

            Settings::reset();
        }

        SECTION("Environment variables take precedence")
        {
            Settings::reset();
            auto settings = Settings::getInstance();

            // Env Var takes precedence over bitfield
            CHECK_FALSE(settings->get(Settings::LogConsole));

            // bitfield takes precedence over default value
            CHECK(settings->get(Settings::SaveAssembly));

            Settings::reset();
        }

        SECTION("Set and get from envvars-backed settings")
        {
            Settings::reset();
            auto settings = Settings::getInstance();
            CHECK_FALSE(settings->get(Settings::LogConsole));
            CHECK(Settings::Get(Settings::SaveAssembly));
            CHECK_THAT(settings->get(Settings::AssemblyFile), Equals("assemblyFileTest.s"));
            CHECK(Settings::Get(Settings::RandomSeed) == 31415);

            // Set settings in memory
            settings->set(Settings::LogConsole, true);
            settings->set(Settings::AssemblyFile, "differentFile.s");
            CHECK(settings->get(Settings::LogConsole));
            CHECK_THAT(settings->get(Settings::AssemblyFile), Equals("differentFile.s"));

            // Values set in memory should not persist
            Settings::reset();
            settings = Settings::getInstance();

            CHECK_FALSE(settings->get(Settings::LogConsole));
            CHECK(Settings::Get(Settings::SaveAssembly));
            CHECK_THAT(settings->get(Settings::AssemblyFile), Equals("assemblyFileTest.s"));
            CHECK(Settings::Get(Settings::RandomSeed) == 31415);

            // set BreakOnThrow to false (previously true via bitfield)
            settings->set(Settings::BreakOnThrow, false);
            // Fatal error reading unparseable env var
            CHECK_THROWS_AS(settings->get(Settings::Scheduler), FatalError);

            Settings::reset();
        }
    }
}
