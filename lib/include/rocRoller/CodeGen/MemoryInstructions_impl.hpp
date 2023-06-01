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

    inline Generator<Instruction>
        MemoryInstructions::load(MemoryKind                        kind,
                                 std::shared_ptr<Register::Value>  dest,
                                 std::shared_ptr<Register::Value>  addr,
                                 std::shared_ptr<Register::Value>  offset,
                                 int                               numBytes,
                                 std::string const                 comment,
                                 bool                              high,
                                 std::shared_ptr<BufferDescriptor> bufDesc,
                                 BufferInstructionOptions          buffOpts)
    {
        auto                             context   = m_context.lock();
        int                              offsetVal = 0;
        std::shared_ptr<Register::Value> newAddr   = addr;

        if(offset && offset->regType() == Register::Type::Literal)
            offsetVal = getUnsignedInt(offset->getLiteralValue());

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadFlat
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield loadFlat(dest, newAddr, offsetVal, numBytes, high);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadLocal
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield loadLocal(dest, newAddr, offsetVal, numBytes, comment, high);
            break;

        case Scalar:
            co_yield loadScalar(dest, newAddr, offset, numBytes);
            break;

        case Buffer:
            AssertFatal(bufDesc);
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to loadLocal
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield loadBuffer(
                dest, newAddr->subset({0}), offsetVal, bufDesc, buffOpts, numBytes, high);
            break;

        default:
            throw std::runtime_error("Load not supported for provided Memorykind");
        }
    }

    inline Generator<Instruction>
        MemoryInstructions::store(MemoryKind                        kind,
                                  std::shared_ptr<Register::Value>  addr,
                                  std::shared_ptr<Register::Value>  data,
                                  std::shared_ptr<Register::Value>  offset,
                                  int                               numBytes,
                                  std::string const                 comment,
                                  bool                              high,
                                  std::shared_ptr<BufferDescriptor> bufDesc,
                                  BufferInstructionOptions          buffOpts)
    {
        auto                             context   = m_context.lock();
        int                              offsetVal = 0;
        std::shared_ptr<Register::Value> newAddr   = addr;

        if(offset && offset->regType() == Register::Type::Literal)
            offsetVal = getUnsignedInt(offset->getLiteralValue());

        switch(kind)
        {
        case Flat:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeFlat
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int64, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield storeFlat(newAddr, data, offsetVal, numBytes, high);
            break;

        case Local:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeLocal
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield storeLocal(newAddr, data, offsetVal, numBytes, comment, high);
            break;

        case Buffer:
            // If the provided offset is not a literal, create a new register that will store the value
            // of addr + offset and pass it to storeBuffer
            if(offset && offset->regType() != Register::Type::Literal)
            {
                newAddr
                    = Register::Value::Placeholder(context, addr->regType(), DataType::Int32, 1);
                co_yield generateOp<Expression::Add>(newAddr, addr, offset);
            }

            co_yield storeBuffer(data, newAddr, offsetVal, bufDesc, buffOpts, numBytes, high);
            break;

        default:
            throw std::runtime_error("Store not supported for provided Memorykind");
        }
    }

    template <MemoryInstructions::MemoryDirection Dir>
    inline Generator<Instruction>
        MemoryInstructions::moveData(MemoryKind                        kind,
                                     std::shared_ptr<Register::Value>  addr,
                                     std::shared_ptr<Register::Value>  data,
                                     std::shared_ptr<Register::Value>  offset,
                                     int                               numBytes,
                                     std::string const                 comment,
                                     bool                              high,
                                     std::shared_ptr<BufferDescriptor> buffDesc,
                                     BufferInstructionOptions          buffOpts)
    {
        if constexpr(Dir == MemoryDirection::Load)
            co_yield load(kind, data, addr, offset, numBytes, comment, high, buffDesc, buffOpts);
        else if constexpr(Dir == MemoryDirection::Store)
            co_yield store(kind, addr, data, offset, numBytes, comment, high, buffDesc, buffOpts);
        else
        {
            Throw<FatalError>("Unsupported MemoryDirection");
        }
    }

    inline std::string MemoryInstructions::genOffsetModifier(int offset) const
    {
        std::string offsetModifier = "";
        if(offset > 0)
            offsetModifier += "offset:" + std::to_string(offset);

        return offsetModifier;
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

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(dest);

        if(numBytes < m_wordSize)
        {
            AssertFatal(numBytes < m_wordSize && dest->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offsetModifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(
                    Instruction("flat_load_ubyte", {dest}, {addr}, {offsetModifier}, "Load value"));
            }
            else if(numBytes == 2)
            {
                if(high)
                {
                    co_yield_(Instruction(
                        "flat_load_short_d16_hi", {dest}, {addr}, {offsetModifier}, "Load value"));
                }
                else
                {
                    co_yield_(Instruction(
                        "flat_load_ushort", {dest}, {addr}, {offsetModifier}, "Load value"));
                }
            }
        }
        else
        {
            // Generate enough load instructions to load numBytes
            int numWords = numBytes / m_wordSize;
            AssertFatal(dest->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadGlobalWidth);
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                co_yield_(Instruction(
                    concatenate("flat_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr},
                    {offsetModifier},
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
                                      int                              numBytes,
                                      bool                             high)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(data);

        if(numBytes < m_wordSize)
        {
            AssertFatal(numBytes < m_wordSize && data->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offsetModifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(Instruction(
                    "flat_store_byte", {}, {addr, data}, {offsetModifier}, "Store value"));
            }
            else if(numBytes == 2)
            {
                if(high)
                {
                    co_yield_(Instruction("flat_store_short_d16_hi",
                                          {},
                                          {addr, data},
                                          {offsetModifier},
                                          "Store value"));
                }
                else
                {
                    co_yield_(Instruction(
                        "flat_store_short", {}, {addr, data}, {offsetModifier}, "Store value"));
                }
            }
        }
        else
        {
            // Generate enough store instructions to store numBytes
            int numWords = numBytes / m_wordSize;
            AssertFatal(data->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeGlobalWidth);
                // Find the largest store instruction that can be used
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                co_yield_(Instruction(
                    concatenate("flat_store_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {},
                    {addr, data->subset(Generated(iota(count, count + width)))},
                    {offsetModifier},
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
                                      std::string const                comment,
                                      bool                             high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(dest);

        if(numBytes < m_wordSize)
        {
            auto offsetModifier = genOffsetModifier(offset);
            co_yield_(Instruction(
                concatenate("ds_read_u", std::to_string(numBytes * 8), (high ? "_d16_hi" : "")),
                {dest},
                {newAddr},
                {offsetModifier},
                concatenate("Load local data ", comment)));
        }
        else
        {
            // Generate enough load instructions to load numBytes
            int numWords = numBytes / m_wordSize;
            AssertFatal(dest->registerCount() == numWords);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadLocalWidth);
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                co_yield_(
                    Instruction(concatenate("ds_read_b", std::to_string(width * m_wordSize * 8)),
                                {dest->subset(Generated(iota(count, count + width)))},
                                {newAddr},
                                {offsetModifier},
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
                                       std::string const                comment,
                                       bool                             high)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield Register::AllocateIfNeeded(data);

        if(numBytes < m_wordSize)
        {
            auto offsetModifier = genOffsetModifier(offset);
            co_yield_(Instruction(
                concatenate("ds_write_b", std::to_string(numBytes * 8), high ? "_d16_hi" : ""),
                {},
                {newAddr, data},
                {offsetModifier},
                concatenate("Store local data ", comment)));
        }
        else
        {

            // Generate enough store instructions to store numBytes
            int  numWords      = numBytes / m_wordSize;
            auto valuesPerWord = m_wordSize / data->variableType().getElementSize();
            AssertFatal(data->registerCount() == numWords * valuesPerWord);
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;

            while(count < numWords)
            {
                // Find the largest store instruction that can be used
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeLocalWidth);
                auto               offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                Register::ValuePtr vgprs;
                if(valuesPerWord > 1)
                {
                    co_yield packForStore(vgprs,
                                          data->element(Generated(iota(
                                              count * valuesPerWord,
                                              count * valuesPerWord + width * valuesPerWord))));
                }
                else
                {
                    vgprs = data->subset(Generated(iota(count, count + width)));
                }
                co_yield_(
                    Instruction(concatenate("ds_write_b", std::to_string(width * m_wordSize * 8)),
                                {},
                                {newAddr, vgprs},
                                {offsetModifier},
                                concatenate("Store local data ", comment)));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after store", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::loadBuffer(std::shared_ptr<Register::Value>  dest,
                                       std::shared_ptr<Register::Value>  addr,
                                       int                               offset,
                                       std::shared_ptr<BufferDescriptor> buffDesc,
                                       BufferInstructionOptions          buffOpts,
                                       int                               numBytes,
                                       bool                              high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        // TODO : add support for buffer loads where numBytes == 3 || numBytes % m_wordSize != 0
        AssertFatal(numBytes > 0
                        && ((numBytes < m_wordSize && numBytes != 3) || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        std::string offsetModifier = "", glc = "", slc = "", lds = "";
        if(buffOpts.getOffen() || offset == 0)
        {
            offsetModifier += "offset: 0";
        }
        else
        {
            offsetModifier += genOffsetModifier(offset);
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
        auto sgprSrd = buffDesc->allRegisters();
        auto ctx     = m_context.lock();

        if(numBytes < m_wordSize)
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
                                  {"offen", offsetModifier, glc, slc, lds},
                                  "Load value"));
        }
        else
        {
            int              numWords       = numBytes / m_wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadGlobalWidth);
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                co_yield_(Instruction(
                    concatenate("buffer_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr, sgprSrd, Register::Value::Literal(0)},
                    {"offen", offsetModifier, glc, slc, lds},
                    "Load value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero("DEBUG: Wait after load", ctx->targetArchitecture()));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeBuffer(std::shared_ptr<Register::Value>  data,
                                        std::shared_ptr<Register::Value>  addr,
                                        int                               offset,
                                        std::shared_ptr<BufferDescriptor> buffDesc,
                                        BufferInstructionOptions          buffOpts,
                                        int                               numBytes,
                                        bool                              high)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        // TODO : add support for buffer stores where numBytes == 3 || numBytes % m_wordSize != 0
        AssertFatal(numBytes > 0
                        && ((numBytes < m_wordSize && numBytes != 3) || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        std::string offsetModifier = "", glc = "", slc = "", lds = "";
        if(buffOpts.getOffen() || offset == 0)
        {
            offsetModifier += "offset: 0";
        }
        else
        {
            offsetModifier += genOffsetModifier(offset);
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
        auto sgprSrd = buffDesc->allRegisters();
        auto ctx     = m_context.lock();

        if(numBytes < m_wordSize)
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
                if(high)
                    opEnd += "_d16_hi";
            }
            co_yield_(Instruction("buffer_store_" + opEnd,
                                  {},
                                  {data, addr, sgprSrd, Register::Value::Literal(0)},
                                  {"offen", offsetModifier, glc, slc, lds},
                                  "Store value"));
        }
        else
        {
            int              numWords       = numBytes / m_wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeGlobalWidth);
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);

                auto dataSubset = data->subset(Generated(iota(count, count + width)));
                co_yield_(Instruction(concatenate("buffer_store_dword",
                                                  width == 1 ? "" : "x" + std::to_string(width)),
                                      {},
                                      {dataSubset, addr, sgprSrd, Register::Value::Literal(0)},
                                      {"offen", offsetModifier, glc, slc, lds},
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
