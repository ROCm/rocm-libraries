
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
        std::shared_ptr<Settings> settings = Settings::getInstance();

        std::vector<std::pair<std::string, char*>> envVars;

        void SetUp()
        {
            Settings::bitFieldType bf(0x0001);

            settings->set(Settings::SettingsBitField, bf);

            envVars.push_back(std::pair<std::string, char*>(
                Settings::LogConsole.name, getenv(Settings::LogConsole.name.c_str())));
            envVars.push_back(std::pair<std::string, char*>(
                Settings::AssemblyFile.name, getenv(Settings::AssemblyFile.name.c_str())));
            envVars.push_back(std::pair<std::string, char*>(
                Settings::RandomSeed.name, getenv(Settings::RandomSeed.name.c_str())));
            envVars.push_back(std::pair<std::string, char*>(
                Settings::SaveAssembly.name, getenv(Settings::SaveAssembly.name.c_str())));
            setenv(Settings::LogConsole.name.c_str(), "0", 1);
            setenv(Settings::SaveAssembly.name.c_str(), "1", 1);
            setenv(Settings::AssemblyFile.name.c_str(), "assemblyFileTest.s", 1);
            setenv(Settings::RandomSeed.name.c_str(), "31415", 1);

            GenericContextFixture::SetUp();
        }

        void TearDown()
        {
            for(auto& env : envVars)
            {
                if(!env.second)
                {
                    unsetenv(env.first.c_str());
                }
                else
                {
                    setenv(env.first.c_str(), env.second, 1);
                }
            }

            GenericContextFixture::TearDown();
        }
    };

    TEST_F(GenericSettings, DefaultValueTest)
    {
        auto settings = Settings::getInstance();

        EXPECT_EQ(settings->get(Settings::SettingsBitField),
                  Settings::SettingsBitField.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogConsole), Settings::LogConsole.defaultValue);
        EXPECT_EQ(settings->get(Settings::SaveAssembly), Settings::SaveAssembly.defaultValue);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), Settings::AssemblyFile.defaultValue);
        EXPECT_EQ(settings->get(Settings::BreakOnThrow), Settings::BreakOnThrow.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogFile), Settings::LogFile.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogLvl), Settings::LogLvl.defaultValue);
        EXPECT_EQ(settings->get(Settings::RandomSeed), Settings::RandomSeed.defaultValue);
    }

    TEST_F(GenericSettings, LogLevelTest)
    {
        auto settings = Settings::getInstance();

        std::ostringstream out;
        out << Settings::LogLevel::None << std::endl;
        out << Settings::LogLevel::Error << std::endl;
        out << Settings::LogLevel::Warning << std::endl;
        out << Settings::LogLevel::Terse << std::endl;
        out << Settings::LogLevel::Verbose << std::endl;
        out << Settings::LogLevel::Debug << std::endl;
        out << Settings::LogLevel::Count;

        std::string stringify = "";
        stringify += settings->toString(Settings::LogLevel::None) + '\n';
        stringify += settings->toString(Settings::LogLevel::Error) + '\n';
        stringify += settings->toString(Settings::LogLevel::Warning) + '\n';
        stringify += settings->toString(Settings::LogLevel::Terse) + '\n';
        stringify += settings->toString(Settings::LogLevel::Verbose) + '\n';
        stringify += settings->toString(Settings::LogLevel::Debug) + '\n';
        stringify += settings->toString(Settings::LogLevel::Count) + '\n';

        std::string expected = R"(
            None
            Error
            Warning
            Terse
            Verbose
            Debug
            LogLevel Count (6)
            )";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(out.str()));
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(stringify));
    }

    TEST_F(GenericSettings, InvalidValueTest)
    {
        auto settings = Settings::getInstance();

        EXPECT_THROW(settings->set(Settings::SettingsBitField, "invalidValue"), FatalError);
    }

    TEST_F(GenericSettings, BitFieldTest)
    {
        auto settings = Settings::getInstance();

        Settings::bitFieldType bf(0x0000);

        // All bit field options are set to false with 0x0000
        settings->set(Settings::SettingsBitField, bf);
        EXPECT_EQ(settings->get(Settings::LogConsole), false);
        EXPECT_EQ(settings->get(Settings::SaveAssembly), false);
        EXPECT_EQ(settings->get(Settings::BreakOnThrow), false);

        // Turn on LogConsole bit
        bf.reset().set(Settings::LogConsole.bit);
        settings->set(Settings::SettingsBitField, bf);
        EXPECT_EQ(settings->get(Settings::LogConsole), true); //failing
    }

    TEST_F(EnvSettings, PrecedenceTest)
    {
        auto                   settings = Settings::getInstance();
        Settings::bitFieldType bf{0x0003};

        // Overwrite exisitng 0x0001 value in m_values map to 0x0011
        settings->set(Settings::SettingsBitField, bf);

        EXPECT_EQ(settings->get(Settings::SaveAssembly), true);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), "assemblyFileTest.s");
        EXPECT_EQ(settings->get(Settings::RandomSeed), 31415);
        EXPECT_EQ(settings->get(Settings::SettingsBitField), bf);

        // Bitfield has this as true but env variable explicitly set it to false
        EXPECT_EQ(settings->get(Settings::LogConsole), false);

        // Create a default Settings object
        settings->reset();
        settings = Settings::getInstance();

        EXPECT_EQ(settings->get(Settings::SettingsBitField),
                  Settings::SettingsBitField.defaultValue);
        EXPECT_EQ(settings->get(Settings::BreakOnThrow), Settings::BreakOnThrow.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogFile), Settings::LogFile.defaultValue);
        EXPECT_EQ(settings->get(Settings::LogLvl), Settings::LogLvl.defaultValue);

        // Environment variables persist after reset
        EXPECT_EQ(settings->get(Settings::LogConsole), false);
        EXPECT_EQ(settings->get(Settings::AssemblyFile), "assemblyFileTest.s");
        EXPECT_EQ(settings->get(Settings::RandomSeed), 31415);
        EXPECT_EQ(settings->get(Settings::SaveAssembly), true);

        // Calling set will have no effect against the environment variables
        std::string assemblyName = "newAssemblyFileTest.s";
        settings->set(Settings::AssemblyFile, assemblyName);
        settings->set(Settings::RandomSeed, 271828);
        settings->set(Settings::SaveAssembly, false);

        EXPECT_EQ(settings->get(Settings::AssemblyFile), "assemblyFileTest.s");
        EXPECT_EQ(settings->get(Settings::RandomSeed), 31415);
        EXPECT_EQ(settings->get(Settings::SaveAssembly), true);
    }
}
