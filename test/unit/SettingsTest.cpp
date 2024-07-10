
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <any>
#include <bitset>
#include <map>
#include <sstream>
#include <stdlib.h>

#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class GenericSettings : public GenericContextFixture
    {
    };

    class EnvSettings : public GenericContextFixture
    {
        void SetUp()
        {
            for(auto const& setting : SettingsOptionBase::instances())
            {
                std::optional<std::string> val;
                if(auto ptr = getenv(setting->name.c_str()))
                {
                    val = ptr;
                }
                m_envVars.emplace_back(setting->name, std::move(val));
            }

            setenv(Settings::BitfieldName.c_str(), "0xFFFFFFFF", 1);
            setenv(Settings::LogConsole.name.c_str(), "0", 1);
            setenv(Settings::AssemblyFile.name.c_str(), "assemblyFileTest.s", 1);
            setenv(Settings::RandomSeed.name.c_str(), "31415", 1);
            setenv(Settings::Scheduler.name.c_str(), "invalidScheduler", 1);

            GenericContextFixture::SetUp();
        }

        void TearDown()
        {
            for(auto& env : m_envVars)
            {
                if(!env.second)
                {
                    unsetenv(env.first.c_str());
                }
                else
                {
                    setenv(env.first.c_str(), env.second->c_str(), 1);
                }
            }

            GenericContextFixture::TearDown();
        }

    private:
        std::vector<std::pair<std::string, std::optional<std::string>>> m_envVars;
    };

    TEST_F(GenericSettings, DefaultValueTest)
    {
        auto settings = Settings::getInstance();

        EXPECT_EQ(settings->get(Settings::LogConsole), Settings::LogConsole.defaultValue);
        EXPECT_EQ(settings->get(Settings::SaveAssembly), Settings::SaveAssembly.defaultValue);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), Settings::AssemblyFile.defaultValue);
        EXPECT_EQ(settings->get(Settings::BreakOnThrow), Settings::BreakOnThrow.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogFile), Settings::LogFile.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogLvl), Settings::LogLvl.defaultValue);
        EXPECT_EQ(settings->get(Settings::RandomSeed), Settings::RandomSeed.defaultValue);
        EXPECT_EQ(settings->get(Settings::Scheduler), Settings::Scheduler.defaultValue);
        EXPECT_EQ(settings->get(Settings::F8ModeOption), Settings::F8ModeOption.defaultValue());
    }

    TEST_F(GenericSettings, GetDefaultValueTest)
    {
        auto settings = Settings::getInstance();

        EXPECT_EQ(settings->getDefault(Settings::LogConsole), Settings::LogConsole.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::SaveAssembly),
                  Settings::SaveAssembly.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::AssemblyFile),
                  Settings::AssemblyFile.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::BreakOnThrow),
                  Settings::BreakOnThrow.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::LogFile), Settings::LogFile.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::LogLvl), Settings::LogLvl.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::RandomSeed), Settings::RandomSeed.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::Scheduler), Settings::Scheduler.defaultValue);
        EXPECT_EQ(settings->getDefault(Settings::F8ModeOption),
                  Settings::F8ModeOption.defaultValue());
    }

    TEST_F(GenericSettings, LogLevelTest)
    {
        auto settings = Settings::getInstance();

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

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(out.str()));
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(stringify));

        EXPECT_EQ(fromString<LogLevel>("None"), LogLevel::None);
        EXPECT_EQ(fromString<LogLevel>("Error"), LogLevel::Error);
        EXPECT_EQ(fromString<LogLevel>("Warning"), LogLevel::Warning);
        EXPECT_EQ(fromString<LogLevel>("Terse"), LogLevel::Terse);
        EXPECT_EQ(fromString<LogLevel>("Verbose"), LogLevel::Verbose);
        EXPECT_EQ(fromString<LogLevel>("Debug"), LogLevel::Debug);
        EXPECT_ANY_THROW(fromString<LogLevel>("Count"));
    }

    TEST_F(GenericSettings, F8ModeTest)
    {
        auto settings = Settings::getInstance();

        std::ostringstream out;
        out << F8Mode::NaNoo << std::endl;
        out << F8Mode::OCP << std::endl;
        out << F8Mode::Count << std::endl;

        std::string stringify = "";
        stringify += toString(F8Mode::NaNoo) + '\n';
        stringify += toString(F8Mode::OCP) + '\n';
        stringify += toString(F8Mode::Count) + '\n';

        std::string expected = R"(
            NaNoo
            OCP
            Count
            )";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(out.str()));
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(stringify));

        EXPECT_EQ(fromString<F8Mode>("NaNoo"), F8Mode::NaNoo);
        EXPECT_EQ(fromString<F8Mode>("OCP"), F8Mode::OCP);
        EXPECT_ANY_THROW(fromString<F8Mode>("Count"));

        settings->set(Settings::F8ModeOption, F8Mode::NaNoo);
        EXPECT_THROW(settings->set(Settings::F8ModeOption, "invalidValue"), FatalError);
        EXPECT_EQ(settings->get(Settings::F8ModeOption), F8Mode::NaNoo);
    }

    TEST_F(GenericSettings, InvalidValueTest)
    {
        auto settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
        EXPECT_THROW(settings->set(Settings::Scheduler, "invalidValue"), FatalError);
        EXPECT_EQ(settings->get(Settings::Scheduler), Scheduling::SchedulerProcedure::Cooperative);

        EXPECT_THROW(settings->set(Settings::LogConsole, "invalidValue"), FatalError);
    }

    TEST_F(EnvSettings, PrecedenceTest)
    {
        Settings::reset();
        auto settings = Settings::getInstance();

        // Env Var takes precedence over bitfield
        EXPECT_EQ(settings->get(Settings::LogConsole), false);

        // bitfield takes precedence over default value
        EXPECT_EQ(settings->get(Settings::SaveAssembly), true);
    }

    TEST_F(EnvSettings, EnvVarsTest)
    {
        Settings::reset();
        auto settings = Settings::getInstance();
        EXPECT_EQ(settings->get(Settings::LogConsole), false);
        EXPECT_EQ(Settings::Get(Settings::SaveAssembly), true);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), "assemblyFileTest.s");
        EXPECT_EQ(Settings::Get(Settings::RandomSeed), 31415);

        // Set settings in memory
        settings->set(Settings::LogConsole, true);
        settings->set(Settings::AssemblyFile, "differentFile.s");
        EXPECT_EQ(settings->get(Settings::LogConsole), true);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), "differentFile.s");

        // Values set in memory should not persist
        Settings::reset();
        settings = Settings::getInstance();

        EXPECT_EQ(settings->get(Settings::LogConsole), false);
        EXPECT_EQ(Settings::Get(Settings::SaveAssembly), true);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), "assemblyFileTest.s");
        EXPECT_EQ(Settings::Get(Settings::RandomSeed), 31415);

        // set BreakOnThrow to false (previously true via bitfield)
        settings->set(Settings::BreakOnThrow, false);
        // Fatal error reading unparseable env var
        EXPECT_THROW(settings->get(Settings::Scheduler), FatalError);
    }

    TEST_F(EnvSettings, InfiniteRecursionTest)
    {
        // Settings ctor should not get from env,
        // otherwise it may throw without a prior settings instance
        // and infinitely recurse
        Settings::reset();
        // unsetenv bitfield revert BreakOnThrow to false (default value)
        unsetenv(Settings::BitfieldName.c_str());
        auto settings = Settings::getInstance();
        EXPECT_THROW(settings->get(Settings::Scheduler), FatalError);
    }

    TEST_F(GenericSettings, HelpString)
    {
        auto        settings = Settings::getInstance();
        std::string help     = settings->help();
        EXPECT_THAT(help, testing::HasSubstr("default"));
        EXPECT_THAT(help, testing::HasSubstr("bit"));
    }
}
