/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#include "Arithmetic/ArithmeticGenerator.hpp"
#include <algorithm>
#include <ranges>

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
                                                           int                numBytes,
                                                           std::string const& comment,
                                                           bool               high)
    {
        auto                             context    = m_context.lock();
        int                              offset_val = 0;
        std::shared_ptr<Register::Value> newAddr    = addr;

        if(offset && offset->regType() == Register::Type::Literal)
            offset_val = std::get<int>(offset->getLiteralValue());

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadFlat
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                co_yield newAddr->allocate();
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield loadFlat(dest, newAddr, offset_val, numBytes, high);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadLocal
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield newAddr->allocate();
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield loadLocal(dest, newAddr, offset_val, numBytes, comment, high);
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
                                                            int                numBytes,
                                                            std::string const& comment)
    {
        auto                             context    = m_context.lock();
        int                              offset_val = 0;
        std::shared_ptr<Register::Value> newAddr    = addr;

        if(offset && offset->regType() == Register::Type::Literal)
            offset_val = std::get<int>(offset->getLiteralValue());

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeFlat
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                co_yield newAddr->allocate();
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield storeFlat(newAddr, data, offset_val, numBytes);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeLocal
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield newAddr->allocate();
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield storeLocal(newAddr, data, offset_val, numBytes, comment);
            break;

        default:
            throw std::runtime_error("Store not supported for provided Memorykind");
        }
    }

    inline std::string MemoryInstructions::genOffsetModifier(int offset) const
    {
        std::string offset_modifier = "";
        if(offset > 0)
            offset_modifier += "offset:" + std::to_string(offset);

        return offset_modifier;
    }

    inline int MemoryInstructions::chooseWidth(int                     numWords,
                                               const std::vector<int>& potentialWidths,
                                               int                     maxWidth) const
    {
        // Find the largest load instruction that can be used
        auto width
            = std::find_if(potentialWidths.begin(),
                           potentialWidths.end(),
                           [numWords, maxWidth](int b) { return b <= maxWidth && b <= numWords; });
        AssertFatal(width != potentialWidths.end());
        return *width;
    }

    inline Generator<Instruction>
        MemoryInstructions::loadFlat(std::shared_ptr<Register::Value> dest,
                                     std::shared_ptr<Register::Value> addr,
                                     int                              offset,
                                     int                              numBytes,
                                     bool                             high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(dest);

        if(numBytes < wordSize)
        {
            AssertFatal(numBytes < wordSize && dest->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offset_modifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(Instruction(
                    "flat_load_ubyte", {dest}, {addr}, {offset_modifier}, "Load value"));
            }
            else if(numBytes == 2)
            {
                if(high)
                {
                    co_yield_(Instruction(
                        "flat_load_short_d16_hi", {dest}, {addr}, {offset_modifier}, "Load value"));
                }
                else
                {
                    co_yield_(Instruction(
                        "flat_load_ushort", {dest}, {addr}, {offset_modifier}, "Load value"));
                }
            }
        }
        else
        {
            // Generate enough load instructions to load numBytes
            int numWords = numBytes / wordSize;
            AssertFatal(dest->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadGlobalWidth);
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(Instruction(
                    concatenate("flat_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr},
                    {offset_modifier},
                    "Load value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeFlat(std::shared_ptr<Register::Value> addr,
                                      std::shared_ptr<Register::Value> data,
                                      int                              offset,
                                      int                              numBytes)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(data);

        if(numBytes < wordSize)
        {
            AssertFatal(numBytes < wordSize && data->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offset_modifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(Instruction(
                    "flat_store_byte", {}, {addr, data}, {offset_modifier}, "Store value"));
            }
            else if(numBytes == 2)
            {
                co_yield_(Instruction(
                    "flat_store_short", {}, {addr, data}, {offset_modifier}, "Store value"));
            }
        }
        else
        {
            // Generate enough store instructions to store numBytes
            int numWords = numBytes / wordSize;
            AssertFatal(data->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeGlobalWidth);
                // Find the largest store instruction that can be used
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(Instruction(
                    concatenate("flat_store_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {},
                    {addr, data->subset(Generated(iota(count, count + width)))},
                    {offset_modifier},
                    "Store value"));
                count += width;
            }
        }

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
                                      int                              offset,
                                      int                              numBytes,
                                      std::string const&               comment,
                                      bool                             high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");

        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(dest);

        if(numBytes < wordSize)
        {
            auto offset_modifier = genOffsetModifier(offset);
            co_yield_(Instruction(
                concatenate("ds_read_u", std::to_string(numBytes * 8), (high ? "_d16_hi" : "")),
                {dest},
                {newAddr},
                {offset_modifier},
                concatenate("Load local data ", comment)));
        }
        else
        {
            // Generate enough load instructions to load numBytes
            int numWords = numBytes / wordSize;
            AssertFatal(dest->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadLocalWidth);
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(
                    Instruction(concatenate("ds_read_b", std::to_string(width * wordSize * 8)),
                                {dest->subset(Generated(iota(count, count + width)))},
                                {newAddr},
                                {offset_modifier},
                                concatenate("Load local data ", comment)));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeLocal(std::shared_ptr<Register::Value> addr,
                                       std::shared_ptr<Register::Value> data,
                                       int                              offset,
                                       int                              numBytes,
                                       std::string const&               comment)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(data);

        if(numBytes < wordSize)
        {
            auto offset_modifier = genOffsetModifier(offset);
            co_yield_(Instruction(concatenate("ds_write_b", std::to_string(numBytes * 8)),
                                  {},
                                  {newAddr, data},
                                  {offset_modifier},
                                  concatenate("Store local data ", comment)));
        }
        else
        {

            // Generate enough store instructions to store numBytes
            int numWords = numBytes / wordSize;
            AssertFatal(data->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                // Find the largest store instruction that can be used
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeLocalWidth);
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(
                    Instruction(concatenate("ds_write_b", std::to_string(width * wordSize * 8)),
                                {},
                                {newAddr, data->subset(Generated(iota(count, count + width)))},
                                {offset_modifier},
                                concatenate("Store local data ", comment)));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after store", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::loadBuffer(std::shared_ptr<Register::Value> dest,
                                       std::shared_ptr<Register::Value> addr,
                                       int                              offset,
                                       BufferDescriptor                 buffDesc,
                                       BufferInstructionOptions         buffOpts,
                                       int                              numBytes,
                                       bool                             high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);
        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");
        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        std::string offset_modifier = "", glc = "", slc = "", lds = "";
        if(buffOpts.getOffen() || offset == 0)
        {
            offset_modifier += "offset: 0";
        }
        else
        {
            offset_modifier += genOffsetModifier(offset);
        }
        if(buffOpts.getGlc())
        {
            glc += "glc";
        }
        if(buffOpts.getSlc())
        {
            slc += "slc";
        }
        if(buffOpts.getLds())
        {
            lds += "lds";
        }
        auto sgprSrd = buffDesc.allRegisters();
        auto ctx     = m_context.lock();

        if(numBytes < wordSize)
        {
            std::string opEnd = "";
            if(numBytes == 1)
            {
                opEnd += "ubyte";
            }
            else if(numBytes == 2)
            {
                if(high)
                    opEnd += "short_d16_hi";
                else
                    opEnd += "ushort";
            }
            co_yield_(Instruction("buffer_load_" + opEnd,
                                  {dest},
                                  {addr, sgprSrd, Register::Value::Literal(0)},
                                  {"offen", offset_modifier, glc, slc, lds},
                                  "Load value"));
        }
        else
        {
            int              numWords       = numBytes / wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadGlobalWidth);
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(Instruction(
                    concatenate("buffer_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr, sgprSrd, Register::Value::Literal(0)},
                    {"offen", offset_modifier, glc, slc, lds},
                    "Load value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeBuffer(std::shared_ptr<Register::Value> addr,
                                        std::shared_ptr<Register::Value> data,
                                        int                              offset,
                                        BufferDescriptor                 buffDesc,
                                        BufferInstructionOptions         buffOpts,
                                        int                              numBytes)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);
        AssertFatal(numBytes > 0 && (numBytes < wordSize || numBytes % wordSize == 0),
                    "Invalid number of bytes");

        std::string offset_modifier = "", glc = "", slc = "", lds = "";
        if(buffOpts.getOffen() || offset == 0)
        {
            offset_modifier += "offset: 0";
        }
        else
        {
            offset_modifier += genOffsetModifier(offset);
        }
        if(buffOpts.getGlc())
        {
            glc += "glc";
        }
        if(buffOpts.getSlc())
        {
            slc += "slc";
        }
        if(buffOpts.getLds())
        {
            lds += "lds";
        }
        auto sgprSrd = buffDesc.allRegisters();
        auto ctx     = m_context.lock();

        if(numBytes < wordSize)
        {
            std::string opEnd = "";
            AssertFatal(numBytes <= 2);
            if(numBytes == 1)
            {
                opEnd += "byte";
            }
            else if(numBytes == 2)
            {
                opEnd += "short";
            }
            co_yield_(Instruction("buffer_store_" + opEnd,
                                  {addr},
                                  {data, sgprSrd, Register::Value::Literal(0)},
                                  {"offen", offset_modifier, glc, slc, lds},
                                  "Store value"));
        }
        else
        {
            int              numWords       = numBytes / wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeGlobalWidth);
                auto offset_modifier = genOffsetModifier(offset + count * wordSize);
                co_yield_(Instruction(concatenate("buffer_store_dword",
                                                  width == 1 ? "" : "x" + std::to_string(width)),
                                      {addr->subset(Generated(iota(count, count + width)))},
                                      {data, sgprSrd, Register::Value::Literal(0)},
                                      {"offen", offset_modifier, glc, slc, lds},
                                      "Store value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after store", ctx->targetArchitecture()));
    }

    inline Generator<Instruction> MemoryInstructions::barrier()
    {
        co_yield Instruction("s_barrier", {}, {}, {}, "Memory barrier");
    }

}
