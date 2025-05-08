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
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::SaveAssembly, true);
            Settings::getInstance()->set(Settings::AssemblyFile, testKernelName());
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
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::SaveAssembly, false);
            Settings::getInstance()->set(Settings::AssemblyFile, testKernelName());
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
