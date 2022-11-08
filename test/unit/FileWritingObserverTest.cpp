
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class FileWritingObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx90a";
        }

        void SetUp()
        {
            Settings::getInstance()->set(Settings::SaveAssembly, true);
            Settings::getInstance()->set(Settings::AssemblyFile,
                                         "FileWritingObserverTestSaveAssemblyFile_kernel_gfx90a.s");
            GenericContextFixture::SetUp();
        }
    };

    TEST_F(FileWritingObserverTest, SaveAssemblyFile)
    {
        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile);

        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());

        EXPECT_EQ(std::filesystem::exists(expected_file), false)
            << "The assembly file shouldn't exist before any instructions are emitted.";

        for(int i = 1; i <= 10; i++)
        {
            auto inst = Instruction("", {}, {}, {}, concatenate("Testing", i));
            m_context->schedule(inst);

            EXPECT_EQ(std::filesystem::exists(expected_file), true)
                << "The assembly file should exist after the test outputs at least one "
                   "instruction.";

            std::ifstream ifs(expected_file);
            std::string   stage(std::istreambuf_iterator<char>{ifs}, {});
            ifs.close();

            EXPECT_EQ(output(), stage) << "The assembly file should ONLY contain the instructions "
                                          "that have been output so far.";
        }

        // Delete the file that was created
        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());
    }
    class FileWritingObserverNegativeTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx90a";
        }

        void SetUp()
        {
            Settings::getInstance()->set(Settings::SaveAssembly, false);
            Settings::getInstance()->set(Settings::AssemblyFile,
                                         "FileWritingObserverTestSaveAssemblyFile_kernel_gfx90a.s");
            GenericContextFixture::SetUp();
        }
    };

    TEST_F(FileWritingObserverNegativeTest, DontSaveAssemblyFile)
    {
        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile);

        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());

        EXPECT_EQ(std::filesystem::exists(expected_file), false)
            << "The assembly file should not exist.";

        for(int i = 1; i <= 10; i++)
        {
            auto inst = Instruction("", {}, {}, {}, concatenate("Testing", i));
            m_context->schedule(inst);

            EXPECT_EQ(std::filesystem::exists(expected_file), false)
                << "The assembly file should not exist.";
        }

        // Delete the file that was created
        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());
    }
}
