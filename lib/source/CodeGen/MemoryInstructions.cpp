#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    std::string toString(MemoryInstructions::MemoryDirection const& d)
    {
        switch(d)
        {
        case MemoryInstructions::MemoryDirection::Load:
            return "Load";
        case MemoryInstructions::MemoryDirection::Store:
            return "Store";
        default:
            break;
        }

        Throw<FatalError>("Invalid MemoryDirection");
    }

    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryDirection d)
    {
        return stream << toString(d);
    }

    std::string toString(MemoryInstructions::MemoryKind const& k)
    {
        switch(k)
        {
        case MemoryInstructions::MemoryKind::Global:
            return "Global";
        case MemoryInstructions::MemoryKind::Scalar:
            return "Scalar";
        case MemoryInstructions::MemoryKind::Local:
            return "Local";
        case MemoryInstructions::MemoryKind::Buffer:
            return "Buffer";
        case MemoryInstructions::MemoryKind::Buffer2LDS:
            return "Buffer2LDS";
        default:
            break;
        }

        Throw<FatalError>("Invalid MemoryKind");
    }

    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryKind k)
    {
        return stream << toString(k);
    }

    Generator<Instruction>
        MemoryInstructions::loadAndPack(MemoryKind                        kind,
                                        Register::ValuePtr                dest,
                                        Register::ValuePtr                addr1,
                                        Register::ValuePtr                offset1,
                                        Register::ValuePtr                addr2,
                                        Register::ValuePtr                offset2,
                                        std::string const                 comment,
                                        std::shared_ptr<BufferDescriptor> buffDesc,
                                        BufferInstructionOptions          buffOpts)
    {
        AssertFatal(dest && dest->regType() == Register::Type::Vector
                        && dest->variableType() == DataType::Halfx2,
                    "loadAndPack destination must be a vector register of type Halfx2");

        // Use the same register for the destination and the temporary val1
        auto val1 = std::make_shared<Register::Value>(
            dest->allocation(), Register::Type::Vector, DataType::Half, dest->allocationCoord());
        auto val2 = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Half, 1);

        co_yield load(kind, val1, addr1, offset1, 2, comment, false, buffDesc, buffOpts);
        co_yield load(kind, val2, addr2, offset2, 2, comment, true, buffDesc, buffOpts);

        co_yield generateOp<Expression::BitwiseOr>(dest, val1, val2);
    }

    Generator<Instruction> MemoryInstructions::packAndStore(MemoryKind         kind,
                                                            Register::ValuePtr addr,
                                                            Register::ValuePtr data1,
                                                            Register::ValuePtr data2,
                                                            Register::ValuePtr offset,
                                                            std::string const  comment)
    {
        auto val = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Halfx2, 1);

        co_yield m_context.lock()->copier()->packHalf(val, data1, data2);

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
        auto valuesPerWord = m_wordSize / toPack->variableType().getElementSize();
        auto unsegmented   = DataTypeInfo::Get(toPack->variableType()).unsegmentedVariableType();
        if(!unsegmented)
        {
            Throw<FatalError>("Segmented variable type not found for ",
                              ShowValue(toPack->variableType()));
        }

        result = Register::Value::Placeholder(toPack->context(),
                                              toPack->regType(),
                                              *unsegmented,
                                              toPack->valueCount() / valuesPerWord,
                                              Register::AllocationOptions::FullyContiguous());
        for(int i = 0; i < result->registerCount(); i++)
        {
            std::vector<Register::ValuePtr> values;
            for(int j = 0; j < valuesPerWord; j++)
            {
                values.push_back(toPack->element({i * valuesPerWord + j}));
            }
            co_yield m_context.lock()->copier()->pack(result->element({i}), values);
        }
    }
}
