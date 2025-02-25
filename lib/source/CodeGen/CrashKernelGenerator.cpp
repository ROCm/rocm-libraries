#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>

namespace rocRoller
{
    CrashKernelGenerator::CrashKernelGenerator(ContextPtr context)
        : m_context(context)
    {
    }

    CrashKernelGenerator::~CrashKernelGenerator() = default;

    Generator<Instruction> CrashKernelGenerator::generateCrashSequence(AssertOpKind assertOpKind)
    {
        switch(assertOpKind)
        {
        case AssertOpKind::MemoryViolation:
        {
            auto context = m_context.lock();
            co_yield writeToNullPtr();
            co_yield Instruction::Wait(
                WaitCount::Zero(context->targetArchitecture(), "DEBUG: Wait after memory write"));
        }
        break;
        case AssertOpKind::STrap:
            co_yield sTrap();
            break;
        case AssertOpKind::NoOp:
            Throw<FatalError>("Unexpected AssertOpKind::NoOp");
        default:
            Throw<FatalError>("Unknown AssertOpKind");
        }
    }

    Generator<Instruction> rocRoller::CrashKernelGenerator::writeToNullPtr()
    {
        auto context     = m_context.lock();
        auto invalidAddr = std::make_shared<Register::Value>(
            context, Register::Type::Scalar, DataType::Int64, 1);
        co_yield context->copier()->copy(invalidAddr, Register::Value::Literal(0L));

        auto dummyData = std::make_shared<Register::Value>(
            context, Register::Type::Scalar, DataType::Int32, 1);
        co_yield context->copier()->copy(dummyData, Register::Value::Literal(42));

        co_yield context->mem()->storeScalar(invalidAddr, dummyData, /*offset*/ 0, /*numBytes*/ 4);
    }

    Generator<Instruction> CrashKernelGenerator::sTrap()
    {
        auto context = m_context.lock();
        auto trapID  = Register::Value::Literal(0x02);
        co_yield_(Instruction("s_trap", {}, {trapID}, {}, ""));
    }
}
