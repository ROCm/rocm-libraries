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
#include "instruction/instruction.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace rocisa
{
    struct ReadWriteInstruction : public Instruction
    {
        enum class RWType
        {
            RW_TYPE0 = 1,
            RW_TYPE1 = 2
        };

        RWType rwType;

        ReadWriteInstruction(InstType instType, RWType rwType, const std::string& comment = "")
            : Instruction(instType, comment)
            , rwType(rwType)
        {
        }

        ReadWriteInstruction(const ReadWriteInstruction& other)
            : Instruction(other)
            , rwType(other.rwType)
        {
        }

        virtual std::string typeConvert() const
        {
            std::string kStr;
            auto        isa = kernel().isaVersion;
            if(rwType == RWType::RW_TYPE0)
            {
                switch(instType)
                {
                case InstType::INST_U16:
                    kStr = isa[0] < 11 ? "ushort" : "u16";
                    break;
                case InstType::INST_B8:
                    kStr = isa[0] < 11 ? "byte" : "b8";
                    break;
                case InstType::INST_U8:
                    kStr = isa[0] < 11 ? "ubyte" : "u8";
                    break;
                case InstType::INST_B16:
                    kStr = isa[0] < 11 ? "short" : "b16";
                    break;
                case InstType::INST_B32:
                    kStr = isa[0] < 11 ? "dword" : "b32";
                    break;
                case InstType::INST_B64:
                    kStr = isa[0] < 11 ? "dwordx2" : "b64";
                    break;
                case InstType::INST_B128:
                    kStr = isa[0] < 11 ? "dwordx4" : "b128";
                    break;
                case InstType::INST_B256:
                    kStr = isa[0] < 11 ? "dwordx8" : "b256";
                    break;
                case InstType::INST_B512:
                    kStr = isa[0] < 11 ? "dwordx16" : "b512";
                    break;
                case InstType::INST_D16_U8:
                    kStr = isa[0] < 11 ? "ubyte_d16" : "d16_u8";
                    break;
                case InstType::INST_D16_HI_U8:
                    kStr = isa[0] < 11 ? "ubyte_d16_hi" : "d16_hi_u8";
                    break;
                case InstType::INST_D16_HI_B8:
                    kStr = isa[0] < 11 ? "byte_d16_hi" : "d16_hi_b8";
                    break;
                case InstType::INST_D16_B16:
                    kStr = isa[0] < 11 ? "short_d16" : "d16_b16";
                    break;
                case InstType::INST_D16_HI_B16:
                    kStr = isa[0] < 11 ? "short_d16_hi" : "d16_hi_b16";
                    break;
                case InstType::INST_TR8_B64:
                    kStr = isa <= std::array<int, 3>{12, 0, 1} ? "tr_b64" : "";
                    break;
                case InstType::INST_TR16_B128:
                    kStr = isa <= std::array<int, 3>{12, 0, 1} ? "tr_b128" : "";
                    break;
                default:
                    break;
                }
            }
            return kStr;
        }

        std::string preStr() const override
        {
            return instStr + typeConvert();
        }

        static int issueLatency()
        {
            // In Quad-Cycle
            return 1;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct GlobalReadInstruction : public ReadWriteInstruction
    {
        std::shared_ptr<Container> dst;

        GlobalReadInstruction(InstType                          instType,
                              const std::shared_ptr<Container>& dst,
                              const std::string&                comment = "")
            : ReadWriteInstruction(instType, RWType::RW_TYPE0, comment)
            , dst(dst)
        {
        }

        GlobalReadInstruction(const GlobalReadInstruction& other)
            : ReadWriteInstruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
        {
        }
    };

    struct FLATReadInstruction : public GlobalReadInstruction
    {
        std::shared_ptr<Container>   vaddr;
        std::optional<FLATModifiers> flat;

        FLATReadInstruction(InstType                          instType,
                            const std::shared_ptr<Container>& dst,
                            const std::shared_ptr<Container>& vaddr,
                            std::optional<FLATModifiers>      flat    = std::nullopt,
                            const std::string&                comment = "")
            : GlobalReadInstruction(instType, dst, comment)
            , vaddr(vaddr)
            , flat(flat)
        {
            instStr = "flat_load_";
        }

        FLATReadInstruction(const FLATReadInstruction& other)
            : GlobalReadInstruction(other)
            , vaddr(other.vaddr ? other.vaddr->clone() : nullptr)
            , flat(other.flat)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, vaddr};
        }

        virtual std::string getArgStr() const
        {
            return dst->toString() + ", " + vaddr->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(flat)
            {
                kStr += flat->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct GLOBALLoadInstruction : public GlobalReadInstruction
    {
        std::shared_ptr<Container>     vaddr;
        std::shared_ptr<Container>     saddr;
        std::optional<GLOBALModifiers> modifier;

        GLOBALLoadInstruction(InstType                          instType,
                              const std::shared_ptr<Container>& dst,
                              const std::shared_ptr<Container>& vaddr,
                              const std::shared_ptr<Container>& saddr,
                              std::optional<GLOBALModifiers>    modifier = std::nullopt,
                              const std::string&                comment  = "")
            : GlobalReadInstruction(instType, dst, comment)
            , vaddr(vaddr)
            , saddr(saddr)
            , modifier(modifier)
        {
            instStr = "global_load_";
        }

        GLOBALLoadInstruction(const GLOBALLoadInstruction& other)
            : GlobalReadInstruction(other)
            , vaddr(other.vaddr ? other.vaddr->clone() : nullptr)
            , saddr(other.saddr ? other.saddr->clone() : nullptr)
            , modifier(other.modifier)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, vaddr, saddr};
        }

        virtual std::string getArgStr() const
        {
            return dst->toString() + ", " + vaddr->toString() + ", " + saddr->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(modifier)
            {
                kStr += modifier->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct MUBUFReadInstruction : public GlobalReadInstruction
    {
        std::shared_ptr<Container>    vaddr;
        std::shared_ptr<Container>    saddr;
        InstructionInput              soffset;
        std::optional<MUBUFModifiers> mubuf;

        MUBUFReadInstruction(InstType                          instType,
                             const std::shared_ptr<Container>& dst,
                             const std::shared_ptr<Container>& vaddr,
                             const std::shared_ptr<Container>& saddr,
                             const InstructionInput&           soffset,
                             std::optional<MUBUFModifiers>     mubuf   = std::nullopt,
                             const std::string&                comment = "")
            : GlobalReadInstruction(instType, dst, comment)
            , vaddr(vaddr)
            , saddr(saddr)
            , soffset(soffset)
            , mubuf(mubuf)
        {
            instStr = "buffer_load_";
        }

        MUBUFReadInstruction(const MUBUFReadInstruction& other)
            : GlobalReadInstruction(other)
            , vaddr(other.vaddr ? other.vaddr->clone() : nullptr)
            , saddr(other.saddr ? other.saddr->clone() : nullptr)
            , soffset(copyInstructionInput(other.soffset))
            , mubuf(other.mubuf)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, vaddr, saddr, soffset};
        }

        std::string getArgStr() const
        {
            std::string dstStr     = dst ? dst->toString() + ", " : "";
            auto        soffsetStr = InstructionInputToString(soffset);
            if(getAsmCaps()["HasMUBUFConst"])
            {
                return dstStr + vaddr->toString() + ", " + saddr->toString() + ", " + soffsetStr;
            }
            else
            {
                if(soffsetStr == "0")
                {
                    return dstStr + vaddr->toString() + ", " + saddr->toString() + ", null";
                }
                else
                {
                    return dstStr + vaddr->toString() + ", " + saddr->toString() + ", "
                           + soffsetStr;
                }
            }
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(mubuf)
            {
                kStr += mubuf->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct AtomicReadWriteInstruction : public ReadWriteInstruction
    {
        std::shared_ptr<Container> dst;
        std::shared_ptr<Container> srcs;

        AtomicReadWriteInstruction(InstType                          instType,
                                   const std::shared_ptr<Container>& dst,
                                   const std::shared_ptr<Container>& srcs,
                                   const std::string&                comment = "")
            : ReadWriteInstruction(instType, RWType::RW_TYPE1, comment)
            , dst(dst)
            , srcs(srcs)
        {
        }

        AtomicReadWriteInstruction(const AtomicReadWriteInstruction& other)
            : ReadWriteInstruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
            , srcs(other.srcs ? other.srcs->clone() : nullptr)
        {
        }
    };

    struct SMemAtomicDecInstruction : public AtomicReadWriteInstruction
    {
        std::shared_ptr<Container>   base;
        std::optional<SMEMModifiers> smem;

        SMemAtomicDecInstruction(InstType                          instType,
                                 const std::shared_ptr<Container>& dst,
                                 const std::shared_ptr<Container>& base,
                                 std::optional<SMEMModifiers>      smem    = std::nullopt,
                                 const std::string&                comment = "")
            : AtomicReadWriteInstruction(instType, dst, nullptr, comment)
            , base(base)
            , smem(smem)
        {
            instStr = "s_atomic_dec";
        }

        SMemAtomicDecInstruction(const SMemAtomicDecInstruction& other)
            : AtomicReadWriteInstruction(other)
            , base(other.base ? other.base->clone() : nullptr)
            , smem(other.smem)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, base};
        }

        std::string getArgStr() const
        {
            return dst->toString() + ", " + base->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(smem)
            {
                kStr += smem->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct SMemLoadInstruction : public GlobalReadInstruction
    {
        std::shared_ptr<Container>   base;
        InstructionInput             soffset;
        std::optional<SMEMModifiers> smem;

        SMemLoadInstruction(InstType                          instType,
                            const std::shared_ptr<Container>& dst,
                            const std::shared_ptr<Container>& base,
                            const InstructionInput&           soffset,
                            std::optional<SMEMModifiers>      smem    = std::nullopt,
                            const std::string&                comment = "")
            : GlobalReadInstruction(instType, dst, comment)
            , base(base)
            , soffset(soffset)
            , smem(smem)
        {
            instStr = "s_load_";
        }

        SMemLoadInstruction(const SMemLoadInstruction& other)
            : GlobalReadInstruction(other)
            , base(other.base ? other.base->clone() : nullptr)
            , soffset(copyInstructionInput(other.soffset))
            , smem(other.smem)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, base, soffset};
        }

        std::string getArgStr() const
        {
            return dst->toString() + ", " + base->toString() + ", "
                   + InstructionInputToString(soffset);
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(smem)
            {
                kStr += smem->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct GlobalWriteInstruction : public ReadWriteInstruction
    {
        std::shared_ptr<Container> srcData;

        GlobalWriteInstruction(InstType                          instType,
                               const std::shared_ptr<Container>& srcData,
                               const std::string&                comment = "")
            : ReadWriteInstruction(instType, RWType::RW_TYPE0, comment)
            , srcData(srcData)
        {
        }

        GlobalWriteInstruction(const GlobalWriteInstruction& other)
            : ReadWriteInstruction(other)
            , srcData(other.srcData ? other.srcData->clone() : nullptr)
        {
        }
    };

    struct SMemStoreInstruction : public GlobalWriteInstruction
    {
        std::shared_ptr<Container>   base;
        InstructionInput             soffset;
        std::optional<SMEMModifiers> smem;

        SMemStoreInstruction(InstType                          instType,
                             const std::shared_ptr<Container>& srcData,
                             const std::shared_ptr<Container>& base,
                             const InstructionInput&           soffset,
                             std::optional<SMEMModifiers>      smem    = std::nullopt,
                             const std::string&                comment = "")
            : GlobalWriteInstruction(instType, srcData, comment)
            , base(base)
            , soffset(soffset)
            , smem(smem)
        {
            instStr = "s_store_";
        }

        SMemStoreInstruction(const SMemStoreInstruction& other)
            : GlobalWriteInstruction(other)
            , base(other.base ? other.base->clone() : nullptr)
            , soffset(copyInstructionInput(other.soffset))
            , smem(other.smem)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {srcData, base, soffset};
        }

        std::string getArgStr() const
        {
            return srcData->toString() + ", " + base->toString() + ", "
                   + InstructionInputToString(soffset);
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(smem)
            {
                kStr += smem->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct FLATStoreInstruction : public GlobalWriteInstruction
    {
        std::shared_ptr<Container>   vaddr;
        std::optional<FLATModifiers> flat;

        FLATStoreInstruction(InstType                          instType,
                             const std::shared_ptr<Container>& vaddr,
                             const std::shared_ptr<Container>& srcData,
                             std::optional<FLATModifiers>      flat    = std::nullopt,
                             const std::string&                comment = "")
            : GlobalWriteInstruction(instType, srcData, comment)
            , vaddr(vaddr)
            , flat(flat)
        {
            instStr = "flat_store_";
        }

        FLATStoreInstruction(const FLATStoreInstruction& other)
            : GlobalWriteInstruction(other)
            , vaddr(other.vaddr ? other.vaddr->clone() : nullptr)
            , flat(other.flat)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {vaddr, srcData};
        }

        virtual std::string getArgStr() const
        {
            return vaddr->toString() + ", " + srcData->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(flat)
            {
                kStr += flat->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct MUBUFStoreInstruction : public GlobalWriteInstruction
    {
        std::shared_ptr<Container>    vaddr;
        std::shared_ptr<Container>    saddr;
        InstructionInput              soffset;
        std::optional<MUBUFModifiers> mubuf;

        MUBUFStoreInstruction(InstType                          instType,
                              const std::shared_ptr<Container>& srcData,
                              const std::shared_ptr<Container>& vaddr,
                              const std::shared_ptr<Container>& saddr,
                              const InstructionInput&           soffset,
                              std::optional<MUBUFModifiers>     mubuf   = std::nullopt,
                              const std::string&                comment = "")
            : GlobalWriteInstruction(instType, srcData, comment)
            , vaddr(vaddr)
            , saddr(saddr)
            , soffset(soffset)
            , mubuf(mubuf)
        {
            instStr = "buffer_store_";
        }

        MUBUFStoreInstruction(const MUBUFStoreInstruction& other)
            : GlobalWriteInstruction(other)
            , vaddr(other.vaddr ? other.vaddr->clone() : nullptr)
            , saddr(other.saddr ? other.saddr->clone() : nullptr)
            , soffset(copyInstructionInput(other.soffset))
            , mubuf(other.mubuf)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {srcData, vaddr, saddr, soffset};
        }

        std::string getArgStr() const
        {
            auto soffsetStr = InstructionInputToString(soffset);
            if(getAsmCaps()["HasMUBUFConst"])
            {
                return srcData->toString() + ", " + vaddr->toString() + ", " + saddr->toString()
                       + ", " + soffsetStr;
            }
            else
            {
                if(soffsetStr == "0")
                {
                    return srcData->toString() + ", " + vaddr->toString() + ", " + saddr->toString()
                           + ", null";
                }
                else
                {
                    return srcData->toString() + ", " + vaddr->toString() + ", " + saddr->toString()
                           + ", " + soffsetStr;
                }
            }
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(mubuf)
            {
                kStr += mubuf->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct LocalReadInstruction : public ReadWriteInstruction
    {
        std::shared_ptr<Container> dst;
        std::shared_ptr<Container> srcs;

        LocalReadInstruction(InstType                          instType,
                             const std::shared_ptr<Container>& dst,
                             const std::shared_ptr<Container>& srcs,
                             const std::string&                comment = "")
            : ReadWriteInstruction(instType, RWType::RW_TYPE1, comment)
            , dst(dst)
            , srcs(srcs)
        {
        }

        LocalReadInstruction(const LocalReadInstruction& other)
            : ReadWriteInstruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
            , srcs(other.srcs ? other.srcs->clone() : nullptr)
        {
        }
    };

    struct DSLoadInstruction : public LocalReadInstruction
    {
        std::optional<DSModifiers> ds;

        DSLoadInstruction(InstType                          instType,
                          const std::shared_ptr<Container>& dst,
                          const std::shared_ptr<Container>& srcs,
                          std::optional<DSModifiers>        ds      = std::nullopt,
                          const std::string&                comment = "")
            : LocalReadInstruction(instType, dst, srcs, comment)
            , ds(ds)
        {
        }

        DSLoadInstruction(const DSLoadInstruction& other)
            : LocalReadInstruction(other)
            , ds(other.ds)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, srcs};
        }

        std::string preStr() const override
        {
            if(kernel().isaVersion[0] < 11)
            {
                std::string copy = instStr;
                auto        pos  = copy.find("load");
                if(pos != std::string::npos)
                {
                    copy.replace(pos, 4, "read");
                }
                return copy;
            }
            return instStr;
        }

        std::string getArgStr() const
        {
            return dst->toString() + ", " + srcs->toString();
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(ds)
            {
                kStr += ds->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct LocalWriteInstruction : public ReadWriteInstruction
    {
        std::shared_ptr<Container> dstAddr;
        std::shared_ptr<Container> src0;
        std::shared_ptr<Container> src1;

        LocalWriteInstruction(InstType                          instType,
                              const std::shared_ptr<Container>& dstAddr,
                              const std::shared_ptr<Container>& src0,
                              const std::shared_ptr<Container>& src1,
                              const std::string&                comment = "")
            : ReadWriteInstruction(instType, RWType::RW_TYPE1, comment)
            , dstAddr(dstAddr)
            , src0(src0)
            , src1(src1)
        {
        }

        LocalWriteInstruction(const LocalWriteInstruction& other)
            : ReadWriteInstruction(other)
            , dstAddr(other.dstAddr ? other.dstAddr->clone() : nullptr)
            , src0(other.src0 ? other.src0->clone() : nullptr)
            , src1(other.src1 ? other.src1->clone() : nullptr)
        {
        }
    };

    struct DSStoreInstruction : public LocalWriteInstruction
    {
        std::optional<DSModifiers> ds;

        DSStoreInstruction(InstType                          instType,
                           const std::shared_ptr<Container>& dstAddr,
                           const std::shared_ptr<Container>& src0,
                           const std::shared_ptr<Container>& src1,
                           std::optional<DSModifiers>        ds      = std::nullopt,
                           const std::string&                comment = "")
            : LocalWriteInstruction(instType, dstAddr, src0, src1, comment)
            , ds(ds)
        {
        }

        DSStoreInstruction(const DSStoreInstruction& other)
            : LocalWriteInstruction(other)
            , ds(other.ds)
        {
        }

        // TODO: Not returning ds
        std::vector<InstructionInput> getParams() const override
        {
            return {dstAddr, src0, src1};
        }

        std::string preStr() const override
        {
            if(kernel().isaVersion[0] < 11)
            {
                std::string copy = instStr;
                auto        pos  = copy.find("store");
                if(pos != std::string::npos)
                {
                    copy.replace(pos, 5, "write");
                }
                return copy;
            }
            return instStr;
        }

        std::string getArgStr() const
        {
            std::string kStr = dstAddr->toString() + ", " + src0->toString();
            if(src1)
            {
                kStr += ", " + src1->toString();
            }
            return kStr;
        }

        std::string toString() const override
        {
            auto        newInstStr = preStr();
            std::string kStr       = newInstStr + " " + getArgStr();
            if(ds)
            {
                kStr += ds->toString();
            }
            return formatWithComment(kStr);
        }
    };

    struct BufferLoadU8 : public MUBUFReadInstruction
    {
        BufferLoadU8(const std::shared_ptr<RegisterContainer>& dst,
                     const std::shared_ptr<RegisterContainer>& vaddr,
                     const std::shared_ptr<RegisterContainer>& saddr,
                     const InstructionInput&                   soffset,
                     std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                     const std::string&                        comment = "")
            : MUBUFReadInstruction(InstType::INST_U8, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadU8(const BufferLoadU8& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadU8>(*this);
        }
    };

    struct BufferLoadD16HIU8 : public MUBUFReadInstruction
    {
        BufferLoadD16HIU8(const std::shared_ptr<RegisterContainer>& dst,
                          const std::shared_ptr<RegisterContainer>& vaddr,
                          const std::shared_ptr<RegisterContainer>& saddr,
                          const InstructionInput&                   soffset,
                          std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                          const std::string&                        comment = "")
            : MUBUFReadInstruction(
                InstType::INST_D16_HI_U8, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadD16HIU8(const BufferLoadD16HIU8& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadD16HIU8>(*this);
        }
    };

    struct BufferLoadD16U8 : public MUBUFReadInstruction
    {
        BufferLoadD16U8(const std::shared_ptr<RegisterContainer>& dst,
                        const std::shared_ptr<RegisterContainer>& vaddr,
                        const std::shared_ptr<RegisterContainer>& saddr,
                        const InstructionInput&                   soffset,
                        std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                        const std::string&                        comment = "")
            : MUBUFReadInstruction(
                InstType::INST_D16_U8, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadD16U8(const BufferLoadD16U8& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadD16U8>(*this);
        }
    };

    struct BufferLoadD16HIB16 : public MUBUFReadInstruction
    {
        BufferLoadD16HIB16(const std::shared_ptr<RegisterContainer>& dst,
                           const std::shared_ptr<RegisterContainer>& vaddr,
                           const std::shared_ptr<RegisterContainer>& saddr,
                           const InstructionInput&                   soffset,
                           std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                           const std::string&                        comment = "")
            : MUBUFReadInstruction(
                InstType::INST_D16_HI_B16, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadD16HIB16(const BufferLoadD16HIB16& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadD16HIB16>(*this);
        }
    };

    struct BufferLoadD16B16 : public MUBUFReadInstruction
    {
        BufferLoadD16B16(const std::shared_ptr<RegisterContainer>& dst,
                         const std::shared_ptr<RegisterContainer>& vaddr,
                         const std::shared_ptr<RegisterContainer>& saddr,
                         const InstructionInput&                   soffset,
                         std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                         const std::string&                        comment = "")
            : MUBUFReadInstruction(
                InstType::INST_D16_B16, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadD16B16(const BufferLoadD16B16& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadD16B16>(*this);
        }
    };

    struct BufferLoadB32 : public MUBUFReadInstruction
    {
        BufferLoadB32(const std::shared_ptr<RegisterContainer>& dst,
                      const std::shared_ptr<RegisterContainer>& vaddr,
                      const std::shared_ptr<RegisterContainer>& saddr,
                      const InstructionInput&                   soffset,
                      std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                      const std::string&                        comment = "")
            : MUBUFReadInstruction(InstType::INST_B32, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadB32(const BufferLoadB32& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadB32>(*this);
        }
    };

    struct BufferLoadB64 : public MUBUFReadInstruction
    {
        BufferLoadB64(const std::shared_ptr<RegisterContainer>& dst,
                      const std::shared_ptr<RegisterContainer>& vaddr,
                      const std::shared_ptr<RegisterContainer>& saddr,
                      const InstructionInput&                   soffset,
                      std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                      const std::string&                        comment = "")
            : MUBUFReadInstruction(InstType::INST_B64, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadB64(const BufferLoadB64& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadB64>(*this);
        }
    };

    struct BufferLoadB128 : public MUBUFReadInstruction
    {
        BufferLoadB128(const std::shared_ptr<RegisterContainer>& dst,
                       const std::shared_ptr<RegisterContainer>& vaddr,
                       const std::shared_ptr<RegisterContainer>& saddr,
                       const InstructionInput&                   soffset,
                       std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                       const std::string&                        comment = "")
            : MUBUFReadInstruction(InstType::INST_B128, dst, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferLoadB128(const BufferLoadB128& other)
            : MUBUFReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferLoadB128>(*this);
        }
    };

    struct FlatLoadD16HIU8 : public FLATReadInstruction
    {
        FlatLoadD16HIU8(const std::shared_ptr<RegisterContainer>& dst,
                        const std::shared_ptr<RegisterContainer>& vaddr,
                        std::optional<FLATModifiers>              flat    = std::nullopt,
                        const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_D16_HI_U8, dst, vaddr, flat, comment)
        {
        }

        FlatLoadD16HIU8(const FlatLoadD16HIU8& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadD16HIU8>(*this);
        }
    };

    struct FlatLoadD16U8 : public FLATReadInstruction
    {
        FlatLoadD16U8(const std::shared_ptr<RegisterContainer>& dst,
                      const std::shared_ptr<RegisterContainer>& vaddr,
                      std::optional<FLATModifiers>              flat    = std::nullopt,
                      const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_D16_U8, dst, vaddr, flat, comment)
        {
        }

        FlatLoadD16U8(const FlatLoadD16U8& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadD16U8>(*this);
        }
    };

    struct FlatLoadD16HIB16 : public FLATReadInstruction
    {
        FlatLoadD16HIB16(const std::shared_ptr<RegisterContainer>& dst,
                         const std::shared_ptr<RegisterContainer>& vaddr,
                         std::optional<FLATModifiers>              flat    = std::nullopt,
                         const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_D16_HI_B16, dst, vaddr, flat, comment)
        {
        }

        FlatLoadD16HIB16(const FlatLoadD16HIB16& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadD16HIB16>(*this);
        }
    };

    struct FlatLoadD16B16 : public FLATReadInstruction
    {
        FlatLoadD16B16(const std::shared_ptr<RegisterContainer>& dst,
                       const std::shared_ptr<RegisterContainer>& vaddr,
                       std::optional<FLATModifiers>              flat    = std::nullopt,
                       const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_D16_B16, dst, vaddr, flat, comment)
        {
        }

        FlatLoadD16B16(const FlatLoadD16B16& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadD16B16>(*this);
        }
    };

    struct FlatLoadB32 : public FLATReadInstruction
    {
        FlatLoadB32(const std::shared_ptr<RegisterContainer>& dst,
                    const std::shared_ptr<RegisterContainer>& vaddr,
                    std::optional<FLATModifiers>              flat    = std::nullopt,
                    const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_B32, dst, vaddr, flat, comment)
        {
        }

        FlatLoadB32(const FlatLoadB32& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadB32>(*this);
        }
    };

    struct FlatLoadB64 : public FLATReadInstruction
    {
        FlatLoadB64(const std::shared_ptr<RegisterContainer>& dst,
                    const std::shared_ptr<RegisterContainer>& vaddr,
                    std::optional<FLATModifiers>              flat    = std::nullopt,
                    const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_B64, dst, vaddr, flat, comment)
        {
        }

        FlatLoadB64(const FlatLoadB64& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadB64>(*this);
        }
    };

    struct FlatLoadB128 : public FLATReadInstruction
    {
        FlatLoadB128(const std::shared_ptr<RegisterContainer>& dst,
                     const std::shared_ptr<RegisterContainer>& vaddr,
                     std::optional<FLATModifiers>              flat    = std::nullopt,
                     const std::string&                        comment = "")
            : FLATReadInstruction(InstType::INST_B128, dst, vaddr, flat, comment)
        {
        }

        FlatLoadB128(const FlatLoadB128& other)
            : FLATReadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatLoadB128>(*this);
        }
    };

    struct GlobalLoadTR8B64 : public GLOBALLoadInstruction
    {
        GlobalLoadTR8B64(const std::shared_ptr<RegisterContainer>& dst,
                              const std::shared_ptr<RegisterContainer>& vaddr,
                              const std::shared_ptr<RegisterContainer>& saddr,
                              std::optional<GLOBALModifiers>            modifier  = std::nullopt,
                              const std::string&                        comment = "")
            : GLOBALLoadInstruction(InstType::INST_TR8_B64, dst, vaddr, saddr, modifier, comment)
        {
        }

        GlobalLoadTR8B64(const GlobalLoadTR8B64& other)
            : GLOBALLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<GlobalLoadTR8B64>(*this);
        }
    };

    struct GlobalLoadTR16B128 : public GLOBALLoadInstruction
    {
        GlobalLoadTR16B128(const std::shared_ptr<RegisterContainer>& dst,
                           const std::shared_ptr<RegisterContainer>& vaddr,
                           const std::shared_ptr<RegisterContainer>& saddr,
                           std::optional<GLOBALModifiers>            modifier = std::nullopt,
                           const std::string&                        comment  = "")
            : GLOBALLoadInstruction(InstType::INST_TR16_B128, dst, vaddr, saddr, modifier, comment)
        {
        }

        GlobalLoadTR16B128(const GlobalLoadTR16B128& other)
            : GLOBALLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<GlobalLoadTR16B128>(*this);
        }
    };

    struct BufferStoreB8 : public MUBUFStoreInstruction
    {
        BufferStoreB8(const std::shared_ptr<RegisterContainer>& src,
                      const std::shared_ptr<RegisterContainer>& vaddr,
                      const std::shared_ptr<RegisterContainer>& saddr,
                      const InstructionInput&                   soffset,
                      std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                      const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B8, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreB8(const BufferStoreB8& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreB8>(*this);
        }
    };

    struct BufferStoreD16HIU8 : public MUBUFStoreInstruction
    {
        BufferStoreD16HIU8(const std::shared_ptr<RegisterContainer>& src,
                           const std::shared_ptr<RegisterContainer>& vaddr,
                           const std::shared_ptr<RegisterContainer>& saddr,
                           const InstructionInput&                   soffset,
                           std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                           const std::string&                        comment = "")
            : MUBUFStoreInstruction(
                InstType::INST_D16_HI_U8, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreD16HIU8(const BufferStoreD16HIU8& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreD16HIU8>(*this);
        }
    };

    struct BufferStoreD16U8 : public MUBUFStoreInstruction
    {
        BufferStoreD16U8(const std::shared_ptr<RegisterContainer>& src,
                         const std::shared_ptr<RegisterContainer>& vaddr,
                         const std::shared_ptr<RegisterContainer>& saddr,
                         const InstructionInput&                   soffset,
                         std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                         const std::string&                        comment = "")
            : MUBUFStoreInstruction(
                InstType::INST_D16_U8, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreD16U8(const BufferStoreD16U8& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreD16U8>(*this);
        }
    };

    struct BufferStoreD16HIB16 : public MUBUFStoreInstruction
    {
        BufferStoreD16HIB16(const std::shared_ptr<RegisterContainer>& src,
                            const std::shared_ptr<RegisterContainer>& vaddr,
                            const std::shared_ptr<RegisterContainer>& saddr,
                            const InstructionInput&                   soffset,
                            std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                            const std::string&                        comment = "")
            : MUBUFStoreInstruction(
                InstType::INST_D16_HI_B16, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreD16HIB16(const BufferStoreD16HIB16& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreD16HIB16>(*this);
        }
    };

    struct BufferStoreD16B16 : public MUBUFStoreInstruction
    {
        BufferStoreD16B16(const std::shared_ptr<RegisterContainer>& src,
                          const std::shared_ptr<RegisterContainer>& vaddr,
                          const std::shared_ptr<RegisterContainer>& saddr,
                          const InstructionInput&                   soffset,
                          std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                          const std::string&                        comment = "")
            : MUBUFStoreInstruction(
                InstType::INST_D16_B16, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreD16B16(const BufferStoreD16B16& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreD16B16>(*this);
        }
    };

    struct BufferStoreB16 : public MUBUFStoreInstruction
    {
        BufferStoreB16(const std::shared_ptr<RegisterContainer>& src,
                       const std::shared_ptr<RegisterContainer>& vaddr,
                       const std::shared_ptr<RegisterContainer>& saddr,
                       const InstructionInput&                   soffset,
                       std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                       const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B16, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreB16(const BufferStoreB16& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreB16>(*this);
        }
    };

    struct BufferStoreB32 : public MUBUFStoreInstruction
    {
        BufferStoreB32(const std::shared_ptr<RegisterContainer>& src,
                       const std::shared_ptr<RegisterContainer>& vaddr,
                       const std::shared_ptr<RegisterContainer>& saddr,
                       const InstructionInput&                   soffset,
                       std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                       const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B32, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreB32(const BufferStoreB32& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreB32>(*this);
        }
    };

    struct BufferStoreB64 : public MUBUFStoreInstruction
    {
        BufferStoreB64(const std::shared_ptr<RegisterContainer>& src,
                       const std::shared_ptr<RegisterContainer>& vaddr,
                       const std::shared_ptr<RegisterContainer>& saddr,
                       const InstructionInput&                   soffset,
                       std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                       const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B64, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreB64(const BufferStoreB64& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreB64>(*this);
        }
    };

    struct BufferStoreB128 : public MUBUFStoreInstruction
    {
        BufferStoreB128(const std::shared_ptr<RegisterContainer>& src,
                        const std::shared_ptr<RegisterContainer>& vaddr,
                        const std::shared_ptr<RegisterContainer>& saddr,
                        const InstructionInput&                   soffset,
                        std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                        const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B128, src, vaddr, saddr, soffset, mubuf, comment)
        {
        }

        BufferStoreB128(const BufferStoreB128& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferStoreB128>(*this);
        }
    };

    struct BufferAtomicAddF32 : public MUBUFStoreInstruction
    {
        BufferAtomicAddF32(const std::shared_ptr<RegisterContainer>& src,
                           const std::shared_ptr<RegisterContainer>& vaddr,
                           const std::shared_ptr<RegisterContainer>& saddr,
                           const InstructionInput&                   soffset,
                           std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                           const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_F32, src, vaddr, saddr, soffset, mubuf, comment)
        {
            setInst("buffer_atomic_add_f32");
        }

        std::string typeConvert() const override
        {
            return "";
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferAtomicAddF32>(*this);
        }

        std::string toString() const override
        {
            std::string kStr = instStr + " " + getArgStr();
            if(mubuf)
                kStr += mubuf->toString();
            return formatWithComment(kStr);
        }
    };

    struct BufferAtomicCmpswapB32 : public MUBUFStoreInstruction
    {
        BufferAtomicCmpswapB32(const std::shared_ptr<RegisterContainer>& src,
                               const std::shared_ptr<RegisterContainer>& vaddr,
                               const std::shared_ptr<RegisterContainer>& saddr,
                               const InstructionInput&                   soffset,
                               std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                               const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B32, src, vaddr, saddr, soffset, mubuf, comment)
        {
            setInst("buffer_atomic_cmpswap_b32");
        }

        BufferAtomicCmpswapB32(const BufferAtomicCmpswapB32& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferAtomicCmpswapB32>(*this);
        }

        std::string typeConvert() const override
        {
            return "";
        }
    };

    struct BufferAtomicCmpswapB64 : public MUBUFStoreInstruction
    {
        BufferAtomicCmpswapB64(const std::shared_ptr<RegisterContainer>& src,
                               const std::shared_ptr<RegisterContainer>& vaddr,
                               const std::shared_ptr<RegisterContainer>& saddr,
                               const InstructionInput&                   soffset,
                               std::optional<MUBUFModifiers>             mubuf   = std::nullopt,
                               const std::string&                        comment = "")
            : MUBUFStoreInstruction(InstType::INST_B32, src, vaddr, saddr, soffset, mubuf, comment)
        {
            setInst("buffer_atomic_cmpswap_b64");
        }

        BufferAtomicCmpswapB64(const BufferAtomicCmpswapB64& other)
            : MUBUFStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<BufferAtomicCmpswapB64>(*this);
        }

        std::string typeConvert() const override
        {
            return "";
        }
    };

    struct FlatStoreD16HIB16 : public FLATStoreInstruction
    {
        FlatStoreD16HIB16(const std::shared_ptr<RegisterContainer>& vaddr,
                          const std::shared_ptr<RegisterContainer>& src,
                          std::optional<FLATModifiers>              flat    = std::nullopt,
                          const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_D16_HI_B16, vaddr, src, flat, comment)
        {
        }

        FlatStoreD16HIB16(const FlatStoreD16HIB16& other)
            : FLATStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatStoreD16HIB16>(*this);
        }
    };

    struct FlatStoreD16B16 : public FLATStoreInstruction
    {
        FlatStoreD16B16(const std::shared_ptr<RegisterContainer>& vaddr,
                        const std::shared_ptr<RegisterContainer>& src,
                        std::optional<FLATModifiers>              flat    = std::nullopt,
                        const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_D16_B16, vaddr, src, flat, comment)
        {
        }

        FlatStoreD16B16(const FlatStoreD16B16& other)
            : FLATStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatStoreD16B16>(*this);
        }
    };

    struct FlatStoreB32 : public FLATStoreInstruction
    {
        FlatStoreB32(const std::shared_ptr<RegisterContainer>& vaddr,
                     const std::shared_ptr<RegisterContainer>& src,
                     std::optional<FLATModifiers>              flat    = std::nullopt,
                     const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_B32, vaddr, src, flat, comment)
        {
        }

        FlatStoreB32(const FlatStoreB32& other)
            : FLATStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatStoreB32>(*this);
        }
    };

    struct FlatStoreB64 : public FLATStoreInstruction
    {
        FlatStoreB64(const std::shared_ptr<RegisterContainer>& vaddr,
                     const std::shared_ptr<RegisterContainer>& src,
                     std::optional<FLATModifiers>              flat    = std::nullopt,
                     const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_B64, vaddr, src, flat, comment)
        {
        }

        FlatStoreB64(const FlatStoreB64& other)
            : FLATStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatStoreB64>(*this);
        }
    };

    struct FlatStoreB128 : public FLATStoreInstruction
    {
        FlatStoreB128(const std::shared_ptr<RegisterContainer>& vaddr,
                      const std::shared_ptr<RegisterContainer>& src,
                      std::optional<FLATModifiers>              flat    = std::nullopt,
                      const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_B128, vaddr, src, flat, comment)
        {
        }

        FlatStoreB128(const FlatStoreB128& other)
            : FLATStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatStoreB128>(*this);
        }
    };

    struct FlatAtomicCmpswapB32 : public FLATStoreInstruction
    {
        FlatAtomicCmpswapB32(const std::shared_ptr<RegisterContainer>& vaddr,
                             const std::shared_ptr<RegisterContainer>& tmp,
                             const std::shared_ptr<RegisterContainer>& src,
                             std::optional<FLATModifiers>              flat    = std::nullopt,
                             const std::string&                        comment = "")
            : FLATStoreInstruction(InstType::INST_B32, vaddr, src, flat, comment)
            , tmp(tmp)
        {
            setInst("flat_atomic_cmpswap_b32");
        }

        FlatAtomicCmpswapB32(const FlatAtomicCmpswapB32& other)
            : FLATStoreInstruction(other)
            , tmp(other.tmp ? other.tmp->clone2() : nullptr)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<FlatAtomicCmpswapB32>(*this);
        }

        std::string getArgStr() const override
        {
            return vaddr->toString() + ", " + tmp->toString() + ", " + srcData->toString();
        }

        std::string typeConvert() const override
        {
            return "";
        }

        std::string toString() const override
        {
            std::string kStr = instStr + " " + getArgStr();
            if(flat)
                kStr += flat->toString();
            return formatWithComment(kStr);
        }

    private:
        std::shared_ptr<RegisterContainer> tmp;
    };

    struct DSLoadU8 : public DSLoadInstruction
    {
        DSLoadU8(const std::shared_ptr<RegisterContainer>& dst,
                 const std::shared_ptr<RegisterContainer>& src,
                 std::optional<DSModifiers>                ds      = std::nullopt,
                 const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_U8, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_u8");
        }

        DSLoadU8(const DSLoadU8& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadU8>(*this);
        }
    };

    struct DSLoadD16HIU8 : public DSLoadInstruction
    {
        DSLoadD16HIU8(const std::shared_ptr<RegisterContainer>& dst,
                      const std::shared_ptr<RegisterContainer>& src,
                      std::optional<DSModifiers>                ds      = std::nullopt,
                      const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_D16_HI_U8, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_u8_d16_hi");
        }

        DSLoadD16HIU8(const DSLoadD16HIU8& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadD16HIU8>(*this);
        }
    };

    struct DSLoadU16 : public DSLoadInstruction
    {
        DSLoadU16(const std::shared_ptr<RegisterContainer>& dst,
                  const std::shared_ptr<RegisterContainer>& src,
                  std::optional<DSModifiers>                ds      = std::nullopt,
                  const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_U16, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_u16");
        }

        DSLoadU16(const DSLoadU16& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadU16>(*this);
        }
    };

    struct DSLoadD16HIU16 : public DSLoadInstruction
    {
        DSLoadD16HIU16(const std::shared_ptr<RegisterContainer>& dst,
                       const std::shared_ptr<RegisterContainer>& src,
                       std::optional<DSModifiers>                ds      = std::nullopt,
                       const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_D16_HI_U16, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_u16_d16_hi");
        }

        DSLoadD16HIU16(const DSLoadD16HIU16& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadD16HIU16>(*this);
        }
    };

    struct DSLoadB16 : public DSLoadInstruction
    {
        DSLoadB16(const std::shared_ptr<RegisterContainer>& dst,
                  const std::shared_ptr<RegisterContainer>& src,
                  std::optional<DSModifiers>                ds      = std::nullopt,
                  const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B16, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_b16");
        }

        DSLoadB16(const DSLoadB16& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadB16>(*this);
        }
    };

    struct DSLoadB32 : public DSLoadInstruction
    {
        DSLoadB32(const std::shared_ptr<RegisterContainer>& dst,
                  const std::shared_ptr<RegisterContainer>& src,
                  std::optional<DSModifiers>                ds      = std::nullopt,
                  const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B32, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_b32");
        }

        DSLoadB32(const DSLoadB32& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadB32>(*this);
        }
    };

    struct DSLoadB64 : public DSLoadInstruction
    {
        DSLoadB64(const std::shared_ptr<RegisterContainer>& dst,
                  const std::shared_ptr<RegisterContainer>& src,
                  std::optional<DSModifiers>                ds      = std::nullopt,
                  const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B64, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_b64");
        }

        DSLoadB64(const DSLoadB64& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadB64>(*this);
        }
    };

    struct DSLoadB64TrB16 : public DSLoadInstruction
    {
        DSLoadB64TrB16(const std::shared_ptr<RegisterContainer>& dst,
                       const std::shared_ptr<RegisterContainer>& src,
                       std::optional<DSModifiers>                ds      = std::nullopt,
                       const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B64, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_b64_tr_b16");
        }

        DSLoadB64TrB16(const DSLoadB64TrB16& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadB64TrB16>(*this);
        }
    };

    struct DSLoadB128 : public DSLoadInstruction
    {
        DSLoadB128(const std::shared_ptr<RegisterContainer>& dst,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B128, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_load_b128");
        }

        DSLoadB128(const DSLoadB128& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoadB128>(*this);
        }

        static int issueLatency()
        {
            return 2;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSLoad2B32 : public DSLoadInstruction
    {
        DSLoad2B32(const std::shared_ptr<RegisterContainer>& dst,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B32, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 2;
            setInst("ds_load2_b32");
        }

        DSLoad2B32(const DSLoad2B32& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoad2B32>(*this);
        }
    };

    struct DSLoad2B64 : public DSLoadInstruction
    {
        DSLoad2B64(const std::shared_ptr<RegisterContainer>& dst,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSLoadInstruction(InstType::INST_B64, dst, src, ds, comment)
        {
            if(ds)
                ds->na = 2;
            setInst("ds_load2_b64");
        }

        DSLoad2B64(const DSLoad2B64& other)
            : DSLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSLoad2B64>(*this);
        }
    };

    struct DSStoreU16 : public DSStoreInstruction
    {
        DSStoreU16(const std::shared_ptr<RegisterContainer>& dstAddr,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_U16, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_u16");
        }

        DSStoreU16(const DSStoreU16& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreU16>(*this);
        }

        static int issueLatency()
        {
            return 2;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSStoreB8 : public DSStoreInstruction
    {
        DSStoreB8(const std::shared_ptr<RegisterContainer>& dstAddr,
                  const std::shared_ptr<RegisterContainer>& src,
                  std::optional<DSModifiers>                ds      = std::nullopt,
                  const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B8, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b8");
        }

        DSStoreB8(const DSStoreB8& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB8>(*this);
        }
    };

    struct DSStoreB16 : public DSStoreInstruction
    {
        DSStoreB16(const std::shared_ptr<RegisterContainer>& dstAddr,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B16, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b16");
        }

        DSStoreB16(const DSStoreB16& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB16>(*this);
        }
    };

    struct DSStoreB8HID16 : public DSStoreInstruction
    {
        DSStoreB8HID16(const std::shared_ptr<RegisterContainer>& dstAddr,
                       const std::shared_ptr<RegisterContainer>& src,
                       std::optional<DSModifiers>                ds      = std::nullopt,
                       const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B8_HI_D16, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b8_d16_hi");
        }

        DSStoreB8HID16(const DSStoreB8HID16& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB8HID16>(*this);
        }
    };

    struct DSStoreD16HIB16 : public DSStoreInstruction
    {
        DSStoreD16HIB16(const std::shared_ptr<RegisterContainer>& dstAddr,
                        const std::shared_ptr<RegisterContainer>& src,
                        std::optional<DSModifiers>                ds      = std::nullopt,
                        const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_D16_HI_B16, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b16_d16_hi");
        }

        DSStoreD16HIB16(const DSStoreD16HIB16& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreD16HIB16>(*this);
        }
    };

    struct DSStoreB32 : public DSStoreInstruction
    {
        DSStoreB32(const std::shared_ptr<RegisterContainer>& dstAddr,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B32, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b32");
        }

        DSStoreB32(const DSStoreB32& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB32>(*this);
        }

        static int issueLatency()
        {
            return 2;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSStoreB64 : public DSStoreInstruction
    {
        DSStoreB64(const std::shared_ptr<RegisterContainer>& dstAddr,
                   const std::shared_ptr<RegisterContainer>& src,
                   std::optional<DSModifiers>                ds      = std::nullopt,
                   const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B64, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b64");
        }

        DSStoreB64(const DSStoreB64& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB64>(*this);
        }

        static int issueLatency()
        {
            return 3;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSStoreB128 : public DSStoreInstruction
    {
        DSStoreB128(const std::shared_ptr<RegisterContainer>& dstAddr,
                    const std::shared_ptr<RegisterContainer>& src,
                    std::optional<DSModifiers>                ds      = std::nullopt,
                    const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B128, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b128");
        }

        DSStoreB128(const DSStoreB128& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB128>(*this);
        }

        static int issueLatency()
        {
            return 5;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSStoreB256 : public DSStoreInstruction
    {
        DSStoreB256(const std::shared_ptr<RegisterContainer>& dstAddr,
                    const std::shared_ptr<RegisterContainer>& src,
                    std::optional<DSModifiers>                ds      = std::nullopt,
                    const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B256, dstAddr, src, nullptr, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_store_b256");
        }

        DSStoreB256(const DSStoreB256& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStoreB256>(*this);
        }

        static int issueLatency()
        {
            return 10;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }

        std::string getArgStr2(bool upper = false) const
        {
            auto srcCopy = RegisterContainer(*dynamic_cast<RegisterContainer*>(src0.get()));
            int  regNum  = srcCopy.regNum / 2;
            if(upper)
            {
                int idx                       = srcCopy.regName->offsets.size() - 1;
                srcCopy.regName->offsets[idx] = srcCopy.regName->offsets[idx] + regNum;
            }
            srcCopy.regNum = regNum;
            return dstAddr->toString() + ", " + srcCopy.toString();
        }

        std::string toString() const override
        {
            std::string instStr = "ds_store_b128";
            if(kernel().isaVersion[0] < 11)
            {
                instStr = "ds_write_b128";
            }
            std::string kStr = instStr + " " + getArgStr2();
            if(ds)
                kStr += ds->toString();
            std::string kStr2 = instStr + " " + getArgStr2(true);
            auto dsCopy = ds ? std::make_shared<DSModifiers>(*ds) : std::make_shared<DSModifiers>();
            dsCopy->offset += 16;
            kStr2 += dsCopy->toString();
            return formatWithComment(kStr) + formatWithComment(kStr2);
        }
    };

    struct DSStore2B32 : public DSStoreInstruction
    {
        DSStore2B32(const std::shared_ptr<RegisterContainer>& dstAddr,
                    const std::shared_ptr<RegisterContainer>& src0,
                    const std::shared_ptr<RegisterContainer>& src1,
                    std::optional<DSModifiers>                ds      = std::nullopt,
                    const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B32, dstAddr, src0, src1, ds, comment)
        {
            if(ds)
                ds->na = 2;
            setInst("ds_store2_b32");
        }

        DSStore2B32(const DSStore2B32& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStore2B32>(*this);
        }

        static int issueLatency()
        {
            return 3;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSStore2B64 : public DSStoreInstruction
    {
        DSStore2B64(const std::shared_ptr<RegisterContainer>& dstAddr,
                    const std::shared_ptr<RegisterContainer>& src0,
                    const std::shared_ptr<RegisterContainer>& src1,
                    std::optional<DSModifiers>                ds      = std::nullopt,
                    const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B64, dstAddr, src0, src1, ds, comment)
        {
            if(ds)
                ds->na = 2;
            setInst("ds_store2_b64");
        }

        DSStore2B64(const DSStore2B64& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSStore2B64>(*this);
        }

        static int issueLatency()
        {
            return 3;
        }

        int getIssueLatency() const override
        {
            return issueLatency();
        }
    };

    struct DSBPermuteB32 : public DSStoreInstruction
    {
        DSBPermuteB32(const std::shared_ptr<RegisterContainer>& dst,
                      const std::shared_ptr<RegisterContainer>& src0,
                      const std::shared_ptr<RegisterContainer>& src1,
                      std::optional<DSModifiers>                ds      = std::nullopt,
                      const std::string&                        comment = "")
            : DSStoreInstruction(InstType::INST_B32, dst, src0, src1, ds, comment)
        {
            if(ds)
                ds->na = 1;
            setInst("ds_bpermute_b32");
        }

        DSBPermuteB32(const DSBPermuteB32& other)
            : DSStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<DSBPermuteB32>(*this);
        }
    };

    struct SAtomicDec : public SMemAtomicDecInstruction
    {
        SAtomicDec(const std::shared_ptr<Container>& dst,
                   const std::shared_ptr<Container>& base,
                   std::optional<SMEMModifiers>      smem    = std::nullopt,
                   const std::string&                comment = "")
            : SMemAtomicDecInstruction(InstType::INST_B32, dst, base, smem, comment)
        {
        }

        SAtomicDec(const SAtomicDec& other)
            : SMemAtomicDecInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAtomicDec>(*this);
        }
    };

    struct SLoadB32 : public SMemLoadInstruction
    {
        SLoadB32(const std::shared_ptr<Container>& dst,
                 const std::shared_ptr<Container>& base,
                 const InstructionInput&           soffset,
                 std::optional<SMEMModifiers>      smem    = std::nullopt,
                 const std::string&                comment = "")
            : SMemLoadInstruction(InstType::INST_B32, dst, base, soffset, smem, comment)
        {
        }

        SLoadB32(const SLoadB32& other)
            : SMemLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLoadB32>(*this);
        }
    };

    struct SLoadB64 : public SMemLoadInstruction
    {
        SLoadB64(const std::shared_ptr<Container>& dst,
                 const std::shared_ptr<Container>& base,
                 const InstructionInput&           soffset,
                 std::optional<SMEMModifiers>      smem    = std::nullopt,
                 const std::string&                comment = "")
            : SMemLoadInstruction(InstType::INST_B64, dst, base, soffset, smem, comment)
        {
        }

        SLoadB64(const SLoadB64& other)
            : SMemLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLoadB64>(*this);
        }
    };

    struct SLoadB128 : public SMemLoadInstruction
    {
        SLoadB128(const std::shared_ptr<Container>& dst,
                  const std::shared_ptr<Container>& base,
                  const InstructionInput&           soffset,
                  std::optional<SMEMModifiers>      smem    = std::nullopt,
                  const std::string&                comment = "")
            : SMemLoadInstruction(InstType::INST_B128, dst, base, soffset, smem, comment)
        {
        }

        SLoadB128(const SLoadB128& other)
            : SMemLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLoadB128>(*this);
        }
    };

    struct SLoadB256 : public SMemLoadInstruction
    {
        SLoadB256(const std::shared_ptr<Container>& dst,
                  const std::shared_ptr<Container>& base,
                  const InstructionInput&           soffset,
                  std::optional<SMEMModifiers>      smem    = std::nullopt,
                  const std::string&                comment = "")
            : SMemLoadInstruction(InstType::INST_B256, dst, base, soffset, smem, comment)
        {
        }

        SLoadB256(const SLoadB256& other)
            : SMemLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLoadB256>(*this);
        }
    };

    struct SLoadB512 : public SMemLoadInstruction
    {
        SLoadB512(const std::shared_ptr<Container>& dst,
                  const std::shared_ptr<Container>& base,
                  const InstructionInput&           soffset,
                  std::optional<SMEMModifiers>      smem    = std::nullopt,
                  const std::string&                comment = "")
            : SMemLoadInstruction(InstType::INST_B512, dst, base, soffset, smem, comment)
        {
        }

        SLoadB512(const SLoadB512& other)
            : SMemLoadInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SLoadB512>(*this);
        }
    };

    struct SStoreB32 : public SMemStoreInstruction
    {
        SStoreB32(const std::shared_ptr<Container>& src,
                  const std::shared_ptr<Container>& base,
                  const InstructionInput&           soffset,
                  std::optional<SMEMModifiers>      smem    = std::nullopt,
                  const std::string&                comment = "")
            : SMemStoreInstruction(InstType::INST_B32, src, base, soffset, smem, comment)
        {
        }

        SStoreB32(const SStoreB32& other)
            : SMemStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SStoreB32>(*this);
        }
    };

    struct SStoreB64 : public SMemStoreInstruction
    {
        SStoreB64(const std::shared_ptr<Container>& src,
                  const std::shared_ptr<Container>& base,
                  const InstructionInput&           soffset,
                  std::optional<SMEMModifiers>      smem    = std::nullopt,
                  const std::string&                comment = "")
            : SMemStoreInstruction(InstType::INST_B64, src, base, soffset, smem, comment)
        {
        }

        SStoreB64(const SStoreB64& other)
            : SMemStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SStoreB64>(*this);
        }
    };

    struct SStoreB128 : public SMemStoreInstruction
    {
        SStoreB128(const std::shared_ptr<Container>& src,
                   const std::shared_ptr<Container>& base,
                   const InstructionInput&           soffset,
                   std::optional<SMEMModifiers>      smem    = std::nullopt,
                   const std::string&                comment = "")
            : SMemStoreInstruction(InstType::INST_B128, src, base, soffset, smem, comment)
        {
        }

        SStoreB128(const SStoreB128& other)
            : SMemStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SStoreB128>(*this);
        }
    };

    struct SStoreB256 : public SMemStoreInstruction
    {
        SStoreB256(const std::shared_ptr<Container>& src,
                   const std::shared_ptr<Container>& base,
                   const InstructionInput&           soffset,
                   std::optional<SMEMModifiers>      smem    = std::nullopt,
                   const std::string&                comment = "")
            : SMemStoreInstruction(InstType::INST_B256, src, base, soffset, smem, comment)
        {
        }

        SStoreB256(const SStoreB256& other)
            : SMemStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SStoreB256>(*this);
        }
    };

    struct SStoreB512 : public SMemStoreInstruction
    {
        SStoreB512(const std::shared_ptr<Container>& src,
                   const std::shared_ptr<Container>& base,
                   const InstructionInput&           soffset,
                   std::optional<SMEMModifiers>      smem    = std::nullopt,
                   const std::string&                comment = "")
            : SMemStoreInstruction(InstType::INST_B512, src, base, soffset, smem, comment)
        {
        }

        SStoreB512(const SStoreB512& other)
            : SMemStoreInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SStoreB512>(*this);
        }
    };
} // namespace rocisa
