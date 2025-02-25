#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class SupportedInstructionObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }
    };

    TEST_F(SupportedInstructionObserverTest, KnownInstruction)
    {
        auto inst = Instruction("s_barrier", {}, {}, {}, "");
        m_context->schedule(inst);
    }

    TEST_F(SupportedInstructionObserverTest, Comments)
    {
        auto inst = Instruction::Comment("Test Comment");
        m_context->schedule(inst);
    }

    TEST_F(SupportedInstructionObserverTest, UnknownInstruction)
    {
        auto inst = Instruction("unknown_instruction", {}, {}, {}, "");
        EXPECT_THROW(m_context->schedule(inst), FatalError);
    }

    class SupportedInstructionObserverOffTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);
            GenericContextFixture::SetUp();
        }
    };

    TEST_F(SupportedInstructionObserverOffTest, KnownInstruction)
    {
        auto inst = Instruction("s_barrier", {}, {}, {}, "");
        m_context->schedule(inst);
    }

    TEST_F(SupportedInstructionObserverOffTest, Comments)
    {
        auto inst = Instruction::Comment("Test Comment");
        m_context->schedule(inst);
    }

    TEST_F(SupportedInstructionObserverOffTest, UnknownInstruction)
    {
        auto inst = Instruction("unknown_instruction", {}, {}, {}, "");
        m_context->schedule(inst);
    }
}
