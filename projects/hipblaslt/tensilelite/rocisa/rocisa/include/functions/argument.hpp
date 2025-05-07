/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include "code.hpp"
#include "instruction/mem.hpp"

#include <memory>

namespace rocisa
{
    class ArgumentLoader
    {
    public:
        ArgumentLoader()
            : kernArgOffset(0)
        {
        }

        void resetOffset()
        {
            kernArgOffset = 0;
        }

        void setOffset(int offset)
        {
            kernArgOffset = offset;
        }

        int getOffset() const
        {
            return kernArgOffset;
        }

        //////////////////////////////////////////////////////////////////////////////
        // loadKernArg
        // Write an argument to specified SGPR and move the kernArgOffset
        // if writeSgpr==0, just move the kernArgOffset - this is used to skip
        // unused parms
        //////////////////////////////////////////////////////////////////////////////
        template <typename DST, typename SRC>
        std::shared_ptr<Item> loadKernArg(DST                             dst,
                                          SRC                             srcAddr,
                                          std::optional<InstructionInput> sgprOffset = std::nullopt,
                                          int                             dword      = 1,
                                          bool                            writeSgpr  = true)
        {
            std::shared_ptr<Item> item = nullptr;
            int                   size = dword * 4;

            if(writeSgpr)
            {
                auto comment
                    = sgprOffset ? std::visit(
                          [](const auto& value) -> std::string {
                              if constexpr(std::is_same_v<decltype(value), int>)
                                  return std::to_string(value);
                              else if constexpr(std::is_same_v<decltype(value),
                                                               std::shared_ptr<RegisterContainer>>)
                                  return value->toString();
                              else
                                  return "";
                          },
                          *sgprOffset)
                                 : std::to_string(kernArgOffset);
                auto dstSgpr = sgpr(dst, dword);
                auto srcSgpr = sgpr(srcAddr, 2);
                // Determine the appropriate SLoadBX based on dword size
                item = [this, dword, &dstSgpr, &srcSgpr, &sgprOffset, &comment]()
                    -> std::shared_ptr<Item> {
                    auto offset = sgprOffset ? *sgprOffset : this->kernArgOffset;
                    switch(dword * 32)
                    {
                    case 512:
                        return std::make_shared<SLoadB512>(
                            dstSgpr, srcSgpr, offset, std::nullopt, comment);
                    case 256:
                        return std::make_shared<SLoadB256>(
                            dstSgpr, srcSgpr, offset, std::nullopt, comment);
                    case 128:
                        return std::make_shared<SLoadB128>(
                            dstSgpr, srcSgpr, offset, std::nullopt, comment);
                    case 64:
                        return std::make_shared<SLoadB64>(
                            dstSgpr, srcSgpr, offset, std::nullopt, comment);
                    case 32:
                        return std::make_shared<SLoadB32>(
                            dstSgpr, srcSgpr, offset, std::nullopt, comment);
                    default:
                        throw std::invalid_argument("Invalid dword size");
                        return nullptr; // This line is unreachable but added to satisfy the compiler
                    }
                }();
            }
            else
            {
                item = std::make_shared<TextBlock>("Move offset by " + std::to_string(size) + "\n");
            }

            kernArgOffset += sgprOffset ? 0 : size;
            return item;
        }

        // currently align sgpr to kernel argument memory, and use s_load_bxxx to load argument as large as possible in one instruction
        // however, in order to match sgpr to kernel argument memory, some unnecessarily sgpr will also be defined, and caused wasting of sgpr.
        // TODO: more efficient way is to organize both sgpr and kernel argument memory in API
        template <typename SRC>
        std::shared_ptr<Module> loadAllKernArg(int sgprStartIndex,
                                               SRC srcAddr,
                                               int numSgprToLoad,
                                               int numSgprPreload = 0)
        {
            auto module     = std::make_shared<Module>("LoadAllKernArg");
            int  actualLoad = numSgprToLoad - numSgprPreload;
            sgprStartIndex += numSgprPreload;
            kernArgOffset += numSgprPreload * 4;

            while(actualLoad > 0)
            {
                int i = 16; // 16, 8, 4, 2, 1
                while(i >= 1)
                {
                    bool isSgprAligned = false;
                    if((i >= 4) && (sgprStartIndex % 4 == 0))
                    {
                        isSgprAligned = true;
                    }
                    else if((i == 2) && (sgprStartIndex % 2 == 0))
                    {
                        isSgprAligned = true;
                    }
                    else if(i == 1)
                    {
                        isSgprAligned = true;
                    }

                    if(isSgprAligned && actualLoad >= i)
                    {
                        actualLoad -= i;
                        std::shared_ptr<Item> SLoadBX;
                        auto                  dstSgpr          = sgpr(sgprStartIndex, i);
                        auto                  srcSgpr          = sgpr(srcAddr, 2);
                        auto                  kernArgOffsetStr = std::to_string(kernArgOffset);
                        switch(i * 32)
                        {
                        case 512:
                            SLoadBX = std::make_shared<SLoadB512>(
                                dstSgpr, srcSgpr, kernArgOffset, std::nullopt, kernArgOffsetStr);
                            break;
                        case 256:
                            SLoadBX = std::make_shared<SLoadB256>(
                                dstSgpr, srcSgpr, kernArgOffset, std::nullopt, kernArgOffsetStr);
                            break;
                        case 128:
                            SLoadBX = std::make_shared<SLoadB128>(
                                dstSgpr, srcSgpr, kernArgOffset, std::nullopt, kernArgOffsetStr);
                            break;
                        case 64:
                            SLoadBX = std::make_shared<SLoadB64>(
                                dstSgpr, srcSgpr, kernArgOffset, std::nullopt, kernArgOffsetStr);
                            break;
                        case 32:
                            SLoadBX = std::make_shared<SLoadB32>(
                                dstSgpr, srcSgpr, kernArgOffset, std::nullopt, kernArgOffsetStr);
                            break;
                        default:
                            throw std::invalid_argument("Invalid SGPR size");
                            SLoadBX
                                = nullptr; // This line is unreachable but added to satisfy the compiler
                        }
                        module->add(SLoadBX);
                        sgprStartIndex += i;
                        kernArgOffset += i * 4;
                        break;
                    }
                    i /= 2;
                }
            }
            return module;
        }

    private:
        int kernArgOffset;
    };
}
