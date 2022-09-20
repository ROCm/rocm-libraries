#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    Generator<Instruction> MemoryInstructions::loadAndPack(MemoryKind                       kind,
                                                           std::shared_ptr<Register::Value> dest,
                                                           std::shared_ptr<Register::Value> addr1,
                                                           std::shared_ptr<Register::Value> offset1,
                                                           std::shared_ptr<Register::Value> addr2,
                                                           std::shared_ptr<Register::Value> offset2,
                                                           std::string                      comment)
    {
        AssertFatal(dest && dest->regType() == Register::Type::Vector
                        && dest->variableType() == DataType::Halfx2,
                    "loadAndPack destination must be a vector register of type Halfx2");

        // TODO: Reuse the same register for destination
        //co_yield Register::AllocateIfNeeded(dest);

        // Use the same register for the destination and the temporary val1
        //auto val1 = std::make_shared<Register::Value>(
        //    dest->allocation(), Register::Type::Vector, DataType::Half, 1);
        auto val1 = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Half, 1);
        auto val2 = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Half, 1);

        co_yield load(kind, val1, addr1, offset1, 2, comment, false);
        co_yield load(kind, val2, addr2, offset2, 2, comment, true);

        co_yield generateOp<Expression::BitwiseOr>(dest, val1, val2);
    }

    Generator<Instruction> MemoryInstructions::packAndStore(MemoryKind                       kind,
                                                            std::shared_ptr<Register::Value> addr,
                                                            std::shared_ptr<Register::Value> data1,
                                                            std::shared_ptr<Register::Value> data2,
                                                            std::shared_ptr<Register::Value> offset,
                                                            std::string comment)
    {
        auto val = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Halfx2, 1);

        co_yield m_context.lock()->copier()->pack(val, data1, data2);

        co_yield store(kind, addr, val, offset, 4, comment);
    }
}
