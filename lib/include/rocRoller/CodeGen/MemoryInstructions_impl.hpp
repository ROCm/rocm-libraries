/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
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

#include <algorithm>
#include <ranges>

namespace rocRoller
{
    inline MemoryInstructions::MemoryInstructions(ContextPtr context)
        : m_context(context)
    {
    }

    inline Generator<Instruction>
        MemoryInstructions::load(MemoryKind                        kind,
                                 Register::ValuePtr                dest,
                                 Register::ValuePtr                addr,
                                 Register::ValuePtr                offset,
                                 int                               numBytes,
                                 std::string const                 comment,
                                 bool                              high,
                                 std::shared_ptr<BufferDescriptor> bufDesc,
                                 BufferInstructionOptions          buffOpts)
    {
        auto               context   = m_context.lock();
        int                offsetVal = 0;
        Register::ValuePtr newAddr   = addr;

        if(offset && kind != MemoryKind::Buffer2LDS)
        {
            if(offset->regType() == Register::Type::Literal)
            {
                offsetVal = getUnsignedInt(offset->getLiteralValue());
            }
            else
            {
                // If the provided offset is not a literal, create a new
                // register that will store the value of addr + offset and pass
                // it to the load function.

                auto addrType = DataType::Int32;
                if(kind == MemoryKind::Global || kind == MemoryKind::Scalar)
                    addrType = DataType::Int64;
                newAddr = Register::Value::Placeholder(context, addr->regType(), addrType, 1);

                auto expr = addr->expression() + offset->expression();
                co_yield generate(newAddr, expr, context);
            }
        }

        switch(kind)
        {
        case MemoryKind::Global:
            co_yield loadGlobal(dest, newAddr, offsetVal, numBytes, high);
            break;

        case MemoryKind::Local:
            co_yield loadLocal(dest, newAddr, offsetVal, numBytes, comment, high);
            break;

        case MemoryKind::Scalar:
            co_yield loadScalar(dest, newAddr, offsetVal, numBytes, buffOpts.glc);
            break;

        case MemoryKind::Buffer:
            AssertFatal(bufDesc);
            co_yield loadBuffer(
                dest, newAddr->subset({0}), offsetVal, bufDesc, buffOpts, numBytes, high);
            break;

        case MemoryKind::Buffer2LDS:
            AssertFatal(bufDesc);
            AssertFatal(offset->regType() == Register::Type::Literal
                        || offset->regType() == Register::Type::Scalar);
            co_yield bufferLoad2LDS(newAddr->subset({0}), bufDesc, buffOpts, numBytes, offset);

            break;
        default:
            throw std::runtime_error("Load not supported for provided Memorykind");
        }
    }

    inline Generator<Instruction>
        MemoryInstructions::store(MemoryKind                        kind,
                                  Register::ValuePtr                addr,
                                  Register::ValuePtr                data,
                                  Register::ValuePtr                offset,
                                  int                               numBytes,
                                  std::string const                 comment,
                                  bool                              high,
                                  std::shared_ptr<BufferDescriptor> bufDesc,
                                  BufferInstructionOptions          buffOpts)
    {
        auto               context   = m_context.lock();
        int                offsetVal = 0;
        Register::ValuePtr newAddr   = addr;

        if(offset)
        {
            if(offset->regType() == Register::Type::Literal)
            {
                offsetVal = getUnsignedInt(offset->getLiteralValue());
            }
            else
            {
                // If the provided offset is not a literal, create a new
                // register that will store the value of addr + offset and pass
                // it to the store function.

                auto addrType = DataType::Int32;
                if(kind == MemoryKind::Global || kind == MemoryKind::Scalar)
                    addrType = DataType::Int64;
                newAddr = Register::Value::Placeholder(context, addr->regType(), addrType, 1);

                auto expr = addr->expression() + offset->expression();
                co_yield generate(newAddr, expr, context);
            }
        }

        switch(kind)
        {
        case MemoryKind::Global:
            co_yield storeGlobal(newAddr, data, offsetVal, numBytes, high);
            break;

        case MemoryKind::Local:
            co_yield storeLocal(newAddr, data, offsetVal, numBytes, comment, high);
            break;

        case MemoryKind::Buffer:
            co_yield storeBuffer(data, newAddr, offsetVal, bufDesc, buffOpts, numBytes, high);
            break;

        case MemoryKind::Scalar:
            co_yield storeScalar(newAddr, data, offsetVal, numBytes, buffOpts.glc);
            break;

        default:
            throw std::runtime_error("Store not supported for provided Memorykind");
        }
    }

    template <MemoryInstructions::MemoryDirection Dir>
    inline Generator<Instruction>
        MemoryInstructions::moveData(MemoryKind                        kind,
                                     Register::ValuePtr                addr,
                                     Register::ValuePtr                data,
                                     Register::ValuePtr                offset,
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

    inline Generator<Instruction> MemoryInstructions::addLargerOffset2Addr(int& offset,
                                                                           Register::ValuePtr& addr,
                                                                           std::string instruction)
    {
        AssertFatal(!instruction.empty());
        auto maxOffset = m_context.lock()
                             ->targetArchitecture()
                             .GetInstructionInfo(instruction)
                             .maxOffsetValue();

        if(maxOffset != 0 && (offset > maxOffset || offset < 0))
        {
            auto currentAddr = addr;
            addr             = nullptr;
            co_yield generate(
                addr, currentAddr->expression() + Expression::literal(offset), m_context.lock());
            offset = 0;
        }
    }

    inline Generator<Instruction> MemoryInstructions::loadGlobal(
        Register::ValuePtr dest, Register::ValuePtr addr, int offset, int numBytes, bool high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        AssertFatal(!high || (high && numBytes == 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();
        co_yield addLargerOffset2Addr(offset, addr, "global_load_dword");

        co_yield addLargerOffset2Addr(offset, addr, "flat_load_dword");

        if(numBytes < m_wordSize)
        {
            AssertFatal(numBytes < m_wordSize && dest->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offsetModifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(Instruction(
                    "global_load_ubyte", {dest}, {addr}, {"off " + offsetModifier}, "Load value"));
            }
            else if(numBytes == 2)
            {
                if(high)
                {
                    co_yield_(Instruction("global_load_short_d16_hi",
                                          {dest},
                                          {addr},
                                          {"off", offsetModifier},
                                          "Load value"));
                }
                else
                {
                    co_yield_(Instruction("global_load_ushort",
                                          {dest},
                                          {addr},
                                          {"off", offsetModifier},
                                          "Load value"));
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
                    concatenate("global_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr},
                    {"off", offsetModifier},
                    "Load value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after load"));
    }

    inline Generator<Instruction> MemoryInstructions::storeGlobal(
        Register::ValuePtr addr, Register::ValuePtr data, int offset, int numBytes, bool high)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();
        co_yield addLargerOffset2Addr(offset, addr, "global_store_dword");

        co_yield addLargerOffset2Addr(offset, addr, "flat_store_dword");

        if(numBytes < m_wordSize)
        {
            AssertFatal(numBytes < m_wordSize && data->registerCount() == 1);
            AssertFatal(numBytes <= 2);
            auto offsetModifier = genOffsetModifier(offset);
            if(numBytes == 1)
            {
                co_yield_(Instruction(
                    "global_store_byte", {}, {addr, data}, {"off", offsetModifier}, "Store value"));
            }
            else if(numBytes == 2)
            {
                if(high)
                {
                    co_yield_(Instruction("global_store_short_d16_hi",
                                          {},
                                          {addr, data},
                                          {"off", offsetModifier},
                                          "Store value"));
                }
                else
                {
                    co_yield_(Instruction("global_store_short",
                                          {},
                                          {addr, data},
                                          {"off", offsetModifier},
                                          "Store value"));
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
                co_yield_(Instruction(concatenate("global_store_dword",
                                                  width == 1 ? "" : "x" + std::to_string(width)),
                                      {},
                                      {addr, data->subset(Generated(iota(count, count + width)))},
                                      {"off", offsetModifier},
                                      "Store value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after store"));
    }

    inline Generator<Instruction> MemoryInstructions::loadScalar(
        Register::ValuePtr dest, Register::ValuePtr base, int offset, int numBytes, bool glc)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(base != nullptr);

        AssertFatal(contains({4, 8, 16, 32, 64}, numBytes),
                    "Unsupported number of bytes for load.: " + std::to_string(numBytes));

        auto offsetLiteral = Register::Value::Literal(offset);

        std::string instruction_string
            = concatenate("s_load_dword",
                          (numBytes > 4 ? "x" : ""),
                          (numBytes > 4 ? std::to_string(numBytes / 4) : ""));

        std::string modifier = glc ? "glc" : "";

        co_yield_(Instruction(
            instruction_string, {dest}, {base, offsetLiteral}, {modifier}, "Load scalar value"));

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after load"));
    }

    inline Generator<Instruction> MemoryInstructions::storeScalar(
        Register::ValuePtr addr, Register::ValuePtr data, int offset, int numBytes, bool glc)
    {
        AssertFatal(data != nullptr);
        AssertFatal(addr != nullptr);

        AssertFatal(contains({4, 8, 16, 32, 64}, numBytes),
                    "Unsupported number of bytes for load.: " + std::to_string(numBytes));

        auto offsetLiteral = Register::Value::Literal(offset);

        std::string instruction_string
            = concatenate("s_store_dword",
                          (numBytes > 4 ? "x" : ""),
                          (numBytes > 4 ? std::to_string(numBytes / 4) : ""));

        std::string modifier = glc ? "glc" : "";

        co_yield_(Instruction(
            instruction_string, {}, {data, addr, offsetLiteral}, {modifier}, "Store scalar value"));

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after store"));
    }

    inline Generator<Instruction> MemoryInstructions::genLocalAddr(Register::ValuePtr& addr) const
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

    inline Generator<Instruction> MemoryInstructions::loadLocal(Register::ValuePtr dest,
                                                                Register::ValuePtr addr,
                                                                int                offset,
                                                                int                numBytes,
                                                                std::string const  comment,
                                                                bool               high)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        AssertFatal(!high || (high && numBytes <= 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        auto ctx = m_context.lock();

        co_yield addLargerOffset2Addr(offset, newAddr, "ds_read_b32");

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
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after load"));
    }

    inline Generator<Instruction> MemoryInstructions::storeLocal(Register::ValuePtr addr,
                                                                 Register::ValuePtr data,
                                                                 int                offset,
                                                                 int                numBytes,
                                                                 std::string const  comment,
                                                                 bool               high)
    {
        AssertFatal(addr != nullptr);
        AssertFatal(data != nullptr);

        auto newAddr = addr;

        co_yield genLocalAddr(newAddr);

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");

        auto ctx = m_context.lock();

        co_yield addLargerOffset2Addr(offset, newAddr, "ds_write_b32");

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
            int              numWords       = numBytes / m_wordSize;
            auto             valuesPerWord  = m_wordSize / data->variableType().getElementSize();
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
                AssertFatal((width <= vgprs->allocation()->options().contiguousChunkWidth),
                            "Write needs more contiguity",
                            ShowValue(width),
                            ShowValue(m_wordSize),
                            ShowValue(vgprs->allocation()->options().contiguousChunkWidth));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after store"));
    }

    inline Generator<Instruction>
        MemoryInstructions::loadBuffer(Register::ValuePtr                dest,
                                       Register::ValuePtr                addr,
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
                    "Invalid number of bytes",
                    ShowValue(numBytes),
                    ShowValue(m_wordSize));

        AssertFatal(!high || (high && numBytes <= 2),
                    "Operation doesn't support hi argument for sizes of "
                        + std::to_string(numBytes));

        auto ctx = m_context.lock();
        co_yield addLargerOffset2Addr(offset, addr, "buffer_load_dword");

        std::string offsetModifier = "", glc = "", slc = "", lds = "";
        if(buffOpts.offen || offset == 0)
        {
            offsetModifier += "offset: 0";
        }
        else
        {
            offsetModifier += genOffsetModifier(offset);
        }
        if(buffOpts.glc)
        {
            glc += "glc";
        }
        if(buffOpts.slc)
        {
            slc += "slc";
        }
        if(buffOpts.lds)
        {
            lds += "lds";
        }
        auto sgprSrd = buffDesc->allRegisters();

        if(numBytes < m_wordSize)
        {
            std::string opEnd = "";
            if(numBytes == 1)
            {
                if(high)
                {
                    AssertFatal(m_context.lock()->targetArchitecture().HasCapability(
                        GPUCapability::HasMFMA_fp8));
                    opEnd += "ubyte_d16_hi";
                }
                else
                    opEnd += "ubyte";
            }
            else if(numBytes == 2)
            {
                if(high)
                    opEnd += "short_d16_hi";
                else
                    opEnd += "ushort";
            }
            const auto& gpu = ctx->targetArchitecture().target();
            const auto  soffset
                = gpu.isGFX12GPU() ? Register::Value::NullLiteral() : Register::Value::Literal(0);
            co_yield_(Instruction("buffer_load_" + opEnd,
                                  {dest},
                                  {addr, sgprSrd, soffset},
                                  {"offen", offsetModifier, glc, slc, lds},
                                  "Load value"));
        }
        else
        {
            int              numWords       = numBytes / m_wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            const auto&      gpu            = ctx->targetArchitecture().target();
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().loadGlobalWidth);
                auto       offsetModifier = genOffsetModifier(offset + count * m_wordSize);
                const auto soffset        = gpu.isGFX12GPU() ? Register::Value::NullLiteral()
                                                             : Register::Value::Literal(0);
                co_yield_(Instruction(
                    concatenate("buffer_load_dword", width == 1 ? "" : "x" + std::to_string(width)),
                    {dest->subset(Generated(iota(count, count + width)))},
                    {addr, sgprSrd, soffset},
                    {"offen", offsetModifier, glc, slc, lds},
                    "Load value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after load"));
    }

    inline Generator<Instruction>
        MemoryInstructions::bufferLoad2LDS(Register::ValuePtr                data,
                                           std::shared_ptr<BufferDescriptor> buffDesc,
                                           BufferInstructionOptions          buffOpts,
                                           int                               numBytes,
                                           Register::ValuePtr                soffset)
    {
        AssertFatal(data != nullptr);
        AssertFatal(buffOpts.lds);

        auto ctx = m_context.lock();

        if(ctx->targetArchitecture().HasCapability(GPUCapability::HasWiderDirectToLds))
        {
            AssertFatal(numBytes == 1 || numBytes == 2 || numBytes == 4 || numBytes == 12
                            || numBytes == 16,
                        "Invalid number of bytes",
                        ShowValue(numBytes));
        }
        else
        {
            AssertFatal(numBytes == 1 || numBytes == 2 || numBytes == 4,
                        "Invalid number of bytes",
                        ShowValue(numBytes));
        }

        std::string offsetModifier = "offset: 0", glc = "", slc = "", lds = "lds";
        if(buffOpts.glc)
        {
            glc += "glc";
        }
        if(buffOpts.slc)
        {
            slc += "slc";
        }

        auto sgprSrd = buffDesc->allRegisters();

        std::string opEnd = "";
        if(numBytes == 1)
        {
            opEnd += "ubyte";
        }
        else if(numBytes == 2)
        {
            opEnd += "ushort";
        }
        else if(numBytes == 4)
        {
            opEnd += "dword";
        }
        else if(numBytes == 12)
        {
            opEnd += "dwordx3";
        }
        else if(numBytes == 16)
        {
            opEnd += "dwordx4";
        }
        else
        {
            Throw<FatalError>("Invalid number of bytes for buffer load direct to LDS.");
        }

        const auto& gpu = ctx->targetArchitecture().target();
        if(gpu.isGFX12GPU())
            soffset = Register::Value::NullLiteral();
        co_yield_(Instruction("buffer_load_" + opEnd,
                              {},
                              {data, sgprSrd, soffset},
                              {"offen", offsetModifier, glc, slc, lds},
                              "Load value direct to lds"));

        if(ctx->kernelOptions().alwaysWaitAfterLoad)
            co_yield Instruction::Wait(WaitCount::Zero(
                ctx->targetArchitecture(), "DEBUG: Wait after direct buffer load to lds"));
    }

    inline Generator<Instruction>
        MemoryInstructions::storeBuffer(Register::ValuePtr                data,
                                        Register::ValuePtr                addr,
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

        auto ctx = m_context.lock();
        co_yield addLargerOffset2Addr(offset, addr, "buffer_store_dword");

        std::string offsetModifier = "", glc = "", slc = "", sc1 = "", lds = "";
        if(buffOpts.offen || offset == 0)
        {
            offsetModifier += "offset: 0";
        }
        else
        {
            offsetModifier += genOffsetModifier(offset);
        }
        if(buffOpts.glc)
        {
            glc += "glc";
        }
        if(buffOpts.slc)
        {
            slc += "slc";
        }
        if(buffOpts.sc1)
        {
            sc1 += "sc1";
        }
        if(buffOpts.lds)
        {
            lds += "lds";
        }
        auto sgprSrd = buffDesc->allRegisters();

        // TODO Use UInt32 offset register for StoreTiled operations
        // that use an offset modifier.
        //
        // If an offset modifier is present, we can't use a 64bit
        // vaddr.  This can happen when: the offset register created
        // by a ComputeIndex operation for a StoreTiled operation is
        // 64bit.
        if(!offsetModifier.empty())
            addr = addr->subset({0});

        if(numBytes < m_wordSize)
        {
            std::string opEnd = "";
            AssertFatal(numBytes <= 2);
            if(numBytes == 1)
            {
                opEnd += "byte";
                if(high)
                {
                    AssertFatal(m_context.lock()->targetArchitecture().HasCapability(
                        GPUCapability::HasMFMA_fp8));
                    opEnd += "_d16_hi";
                }
            }
            else if(numBytes == 2)
            {
                opEnd += "short";
                if(high)
                    opEnd += "_d16_hi";
            }
            const auto& gpu = ctx->targetArchitecture().target();
            const auto  soffset
                = gpu.isGFX12GPU() ? Register::Value::NullLiteral() : Register::Value::Literal(0);
            co_yield_(Instruction("buffer_store_" + opEnd,
                                  {},
                                  {data, addr, sgprSrd, soffset},
                                  {"offen", offsetModifier, glc, slc, sc1, lds},
                                  "Store value"));
        }
        else
        {
            int              numWords       = numBytes / m_wordSize;
            std::vector<int> potentialWords = {4, 3, 2, 1};
            int              count          = 0;
            const auto&      gpu            = ctx->targetArchitecture().target();
            while(count < numWords)
            {
                auto width = chooseWidth(
                    numWords - count, potentialWords, ctx->kernelOptions().storeGlobalWidth);
                auto offsetModifier = genOffsetModifier(offset + count * m_wordSize);

                auto valuesPerWord = m_wordSize / data->variableType().getElementSize();
                Register::ValuePtr dataSubset;
                if(valuesPerWord > 1)
                {
                    AssertFatal(data->registerCount() == numWords * valuesPerWord);
                    co_yield packForStore(dataSubset,
                                          data->element(Generated(iota(
                                              count * valuesPerWord,
                                              count * valuesPerWord + width * valuesPerWord))));
                }
                else
                {
                    dataSubset = data->subset(Generated(iota(count, count + width)));
                }
                const auto soffset = gpu.isGFX12GPU() ? Register::Value::NullLiteral()
                                                      : Register::Value::Literal(0);
                co_yield_(Instruction(concatenate("buffer_store_dword",
                                                  width == 1 ? "" : "x" + std::to_string(width)),
                                      {},
                                      {dataSubset, addr, sgprSrd, soffset},
                                      {"offen", offsetModifier, glc, slc, sc1, lds},
                                      "Store value"));
                count += width;
            }
        }

        if(ctx->kernelOptions().alwaysWaitAfterStore)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after store"));
    }

    inline Generator<Instruction>
        MemoryInstructions::barrier(CForwardRangeOf<Register::ValuePtr> auto srcs,
                                    std::string                              comment)
    {
        co_yield barrierImpl(std::move(srcs), std::move(comment));
    }
    inline Generator<Instruction>
        MemoryInstructions::barrier(std::initializer_list<Register::ValuePtr> srcs,
                                    std::string                               comment)
    {
        co_yield barrierImpl(std::move(srcs), std::move(comment));
    }

    inline Generator<Instruction>
        MemoryInstructions::barrierImpl(CForwardRangeOf<Register::ValuePtr> auto srcs,
                                        std::string                              comment)
    {
        const auto& arch = m_context.lock()->targetArchitecture();
        if(arch.HasCapability(GPUCapability::s_barrier))
        {
            Instruction inst("s_barrier", {}, {}, {}, std::move(comment));
            for(auto reg : srcs)
                inst.addExtraSrc(reg);
            co_yield inst;
        }
        else if(arch.HasCapability(GPUCapability::s_barrier_signal))
        {
            const auto normalBarrierID = -1;
            {
                Instruction inst("s_barrier_signal",
                                 {},
                                 {Register::Value::Literal(normalBarrierID)},
                                 {},
                                 comment);
                for(auto reg : srcs)
                    inst.addExtraSrc(reg);
                co_yield inst;
            }

            co_yield_(Instruction("s_barrier_wait",
                                  {},
                                  {Register::Value::Literal(normalBarrierID)},
                                  {},
                                  std::move(comment)));
        }
        else
        {
            Throw<FatalError>(
                fmt::format("Barriers are not implemented for {}.\n", arch.target().toString()));
        }
    }

    inline auto MemoryInstructions::addExtraDst(Register::ValuePtr dst)
    {
        return [dst](Instruction inst) -> Instruction {
            if(GPUInstructionInfo::isVMEM(inst.getOpCode())
               || GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                inst.addExtraDst(dst);
            }

            return inst;
        };
    }

    inline auto MemoryInstructions::addExtraSrc(Register::ValuePtr src)
    {
        return [src](Instruction inst) -> Instruction {
            if(GPUInstructionInfo::isVMEM(inst.getOpCode())
               || GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                inst.addExtraSrc(src);
            }

            return inst;
        };
    }
}
