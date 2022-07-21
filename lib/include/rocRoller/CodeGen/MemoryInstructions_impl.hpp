/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

namespace rocRoller
{
    inline MemoryInstructions::MemoryInstructions(std::shared_ptr<Context> context)
        : m_context(context)
    {
    }

    inline Generator<Instruction> MemoryInstructions::load(MemoryKind                       kind,
                                                           std::shared_ptr<Register::Value> dest,
                                                           std::shared_ptr<Register::Value> addr,
                                                           std::shared_ptr<Register::Value> offset,
                                                           int         numBytes,
                                                           std::string comment)
    {
        auto                             context    = m_context.lock();
        std::string                      offset_str = "";
        std::shared_ptr<Register::Value> newAddr    = addr;

        if(offset)
            offset_str = offset->getLiteral();

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadFlat
            if(offset && offset_str == "")
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                auto arith = Component::Get<Arithmetic>(context, addr->regType(), DataType::Int64);
                co_yield newAddr->allocate();
                co_yield arith->add(newAddr, addr, offset);
            }

            co_yield loadFlat(dest, newAddr, offset_str, numBytes);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadLocal
            if(offset && offset_str == "")
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                auto arith = Component::Get<Arithmetic>(context, addr->regType(), DataType::Int32);
                co_yield newAddr->allocate();
                co_yield arith->add(newAddr, addr, offset);
            }

            co_yield loadLocal(dest, newAddr, offset_str, numBytes, comment);
            break;

        case Scalar:
            co_yield loadScalar(dest, newAddr, offset, numBytes);
            break;

        default:
            throw std::runtime_error("Load not supported for provided Memorykind");
        }
    }

    inline Generator<Instruction> MemoryInstructions::store(MemoryKind                       kind,
                                                            std::shared_ptr<Register::Value> addr,
                                                            std::shared_ptr<Register::Value> data,
                                                            std::shared_ptr<Register::Value> offset,
                                                            int         numBytes,
                                                            std::string comment)
    {
        auto                             context    = m_context.lock();
        std::string                      offset_str = "";
        std::shared_ptr<Register::Value> newAddr    = addr;

        if(offset)
            offset_str = offset->getLiteral();

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeFlat
            if(offset && offset_str == "")
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                auto arith = Component::Get<Arithmetic>(context, addr->regType(), DataType::Int64);
                co_yield newAddr->allocate();
                co_yield arith->add(newAddr, addr, offset);
            }

            co_yield storeFlat(newAddr, data, offset_str, numBytes);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeLocal
            if(offset && offset_str == "")
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                auto arith = Component::Get<Arithmetic>(context, addr->regType(), DataType::Int32);
                co_yield newAddr->allocate();
                co_yield arith->add(newAddr, addr, offset);
            }

            co_yield storeLocal(newAddr, data, offset_str, numBytes, comment);
            break;

        default:
            throw std::runtime_error("Store not supported for provided Memorykind");
        }
    }

    inline std::string MemoryInstructions::genOffsetModifier(std::string offset) const
    {
        std::string offset_modifier = "";
        if(offset.size() > 0)
            offset_modifier += "offset:" + offset;

        return offset_modifier;
    }

    inline Generator<Instruction>
        MemoryInstructions::loadFlat(std::shared_ptr<Register::Value> dest,
                                     std::shared_ptr<Register::Value> addr,
                                     std::string                      offset,
                                     int                              numBytes)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto offset_modifier = genOffsetModifier(offset);

        switch(numBytes)
        {
        case 1:
            co_yield_(
                Instruction("flat_load_ubyte", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        case 2:
            co_yield_(
                Instruction("flat_load_ushort", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        case 4:
            co_yield_(
                Instruction("flat_load_dword", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        case 8:
            co_yield_(
                Instruction("flat_load_dwordx2", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        case 12:
            co_yield_(
                Instruction("flat_load_dwordx3", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        case 16:
            co_yield_(
                Instruction("flat_load_dwordx4", {dest}, {addr}, {offset_modifier}, "Load value"));
            break;
        default:
            throw std::runtime_error("Unsupported number of bytes for load.");
        }

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeFlat(std::shared_ptr<Register::Value> addr,
                                      std::shared_ptr<Register::Value> data,
                                      std::string                      offset,
                                      int                              numBytes)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        auto offset_modifier = genOffsetModifier(offset);

        switch(numBytes)
        {
        case 1:
            co_yield_(
                Instruction("flat_store_byte", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        case 2:
            co_yield_(Instruction(
                "flat_store_short", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        case 4:
            co_yield_(Instruction(
                "flat_store_dword", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        case 8:
            co_yield_(Instruction(
                "flat_store_dwordx2", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        case 12:
            co_yield_(Instruction(
                "flat_store_dwordx3", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        case 16:
            co_yield_(Instruction(
                "flat_store_dwordx4", {}, {addr, data}, {offset_modifier}, "Store value"));
            break;
        default:
            throw std::runtime_error("Unsupported number of bytes for store.");
        }

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after store", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::loadScalar(std::shared_ptr<Register::Value> dest,
                                       std::shared_ptr<Register::Value> base,
                                       std::shared_ptr<Register::Value> offset,
                                       int                              numBytes)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(base != nullptr);

        AssertFatal(contains({4, 8, 16, 32, 64}, numBytes),
                    "Unsupported number of bytes for load.: " + std::to_string(numBytes));

        std::string instruction_string
            = concatenate("s_load_dword",
                          (numBytes > 4 ? "x" : ""),
                          (numBytes > 4 ? std::to_string(numBytes / 4) : ""));

        co_yield_(Instruction(instruction_string, {dest}, {base, offset}, {}, "Load value"));

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::genLocalAddr(std::shared_ptr<Register::Value>& addr) const
    {
        // If an allocation of LocalData is passed in, allocate a new register that contains
        // the offset.
        if(addr->regType() == Register::Type::LocalData)
        {
            auto offset = addr->getLDSAllocation()->offset();
            addr        = Register::Value::Placeholder(
                m_context.lock(), Register::Type::Vector, DataType::Int32, 1);
            co_yield m_context.lock()->copier()->copy(addr, Register::Value::Literal(offset));
        }
    }

    inline Generator<Instruction>
        MemoryInstructions::loadLocal(std::shared_ptr<Register::Value> dest,
                                      std::shared_ptr<Register::Value> addr,
                                      std::string                      offset,
                                      int                              numBytes,
                                      std::string                      comment)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        auto offset_modifier = genOffsetModifier(offset);

        AssertFatal(contains({1, 2, 4, 8, 12, 16}, numBytes),
                    "Unsupported number of bytes for load.: " + std::to_string(numBytes));

        std::string instruction_string
            = concatenate("ds_read_", (numBytes <= 2 ? "u" : "b"), std::to_string(numBytes * 8));

        co_yield_(Instruction(instruction_string,
                              {dest},
                              {newAddr},
                              {offset_modifier},
                              concatenate("Load local data ", comment)));

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeLocal(std::shared_ptr<Register::Value> addr,
                                       std::shared_ptr<Register::Value> data,
                                       std::string                      offset,
                                       int                              numBytes,
                                       std::string                      comment)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        auto offset_modifier = genOffsetModifier(offset);

        AssertFatal(contains({1, 2, 4, 8, 12, 16}, numBytes),
                    "Unsupported number of bytes for store.: " + std::to_string(numBytes));

        std::string instruction_string = "ds_write_b" + std::to_string(numBytes * 8);
        co_yield_(Instruction(instruction_string,
                              {},
                              {newAddr, data},
                              {offset_modifier},
                              concatenate("Store local data ", comment)));

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after store", ctx->targetArchitecture()));
    }

    inline Generator<Instruction> MemoryInstructions::barrier()
    {
        co_yield Instruction("s_barrier", {}, {}, {}, "Memory barrier");
    }

}
