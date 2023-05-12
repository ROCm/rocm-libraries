#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    std::string ToString(MemoryInstructions::MemoryDirection const& d)
    {
        switch(d)
        {
        case MemoryInstructions::MemoryDirection::Load:
            return "Load";
        case MemoryInstructions::MemoryDirection::Store:
            return "Store";
        }

        Throw<FatalError>("Invalid MemoryDirection");
    }

    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryDirection d)
    {
        return stream << ToString(d);
    }

    Generator<Instruction>
        MemoryInstructions::loadAndPack(MemoryKind                        kind,
                                        std::shared_ptr<Register::Value>  dest,
                                        std::shared_ptr<Register::Value>  addr1,
                                        std::shared_ptr<Register::Value>  offset1,
                                        std::shared_ptr<Register::Value>  addr2,
                                        std::shared_ptr<Register::Value>  offset2,
                                        std::string const                 comment,
                                        std::shared_ptr<BufferDescriptor> buffDesc,
                                        BufferInstructionOptions          buffOpts)
    {
        AssertFatal(dest && dest->regType() == Register::Type::Vector
                        && dest->variableType() == DataType::Halfx2,
                    "loadAndPack destination must be a vector register of type Halfx2");

        co_yield Register::AllocateIfNeeded(dest);

        // Use the same register for the destination and the temporary val1
        auto val1 = std::make_shared<Register::Value>(
            dest->allocation(), Register::Type::Vector, DataType::Half, dest->allocationCoord());
        auto val2 = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Half, 1);

        co_yield load(kind, val1, addr1, offset1, 2, comment, false, buffDesc, buffOpts);
        co_yield load(kind, val2, addr2, offset2, 2, comment, true, buffDesc, buffOpts);

        co_yield generateOp<Expression::BitwiseOr>(dest, val1, val2);
    }

    Generator<Instruction> MemoryInstructions::packAndStore(MemoryKind                       kind,
                                                            std::shared_ptr<Register::Value> addr,
                                                            std::shared_ptr<Register::Value> data1,
                                                            std::shared_ptr<Register::Value> data2,
                                                            std::shared_ptr<Register::Value> offset,
                                                            std::string const comment)
    {
        auto val = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Halfx2, 1);

        co_yield m_context.lock()->copier()->pack(val, data1, data2);

        co_yield store(kind, addr, val, offset, 4, comment);
    }

    /**
     * @brief Pack values from toPack into result. There may be multiple values within toPack.
     *
     * Currently only works for going from Half to Halfx2
     *
     * @param result
     * @param toPack
     * @return Generator<Instruction>
     */
    Generator<Instruction> MemoryInstructions::packForStore(Register::ValuePtr& result,
                                                            Register::ValuePtr  toPack) const
    {
        auto valuesPerWord = wordSize / toPack->variableType().getElementSize();
        AssertFatal(valuesPerWord == 2);
        result = Register::Value::Placeholder(toPack->context(),
                                              toPack->regType(),
                                              DataType::Halfx2,
                                              toPack->valueCount() / valuesPerWord);
        co_yield Register::AllocateIfNeeded(result);

        for(int i = 0; i < result->registerCount(); i++)
        {
            co_yield m_context.lock()->copier()->pack(result->element({i}),
                                                      toPack->element({i * valuesPerWord}),
                                                      toPack->element({i * valuesPerWord + 1}));
        }
    }
}
