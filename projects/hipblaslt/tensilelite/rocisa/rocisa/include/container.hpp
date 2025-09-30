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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "base.hpp"
#include "enum.hpp"

namespace rocisa
{
    struct Container
    {
        Container() {}

        virtual ~Container() = default;

        virtual std::shared_ptr<Container> clone() const = 0;

        virtual std::string toString() const = 0;
    };

    struct DSModifiers : public Container
    {
        DSModifiers(int na = 1, int offset = 0, int offset0 = 0, int offset1 = 0, bool gds = false)
            : Container()
            , na(na)
            , offset(offset)
            , offset0(offset0)
            , offset1(offset1)
            , gds(gds)
        {
        }

        DSModifiers(const DSModifiers& other)
            : Container()
            , na(other.na)
            , offset(other.offset)
            , offset0(other.offset0)
            , offset1(other.offset1)
            , gds(other.gds)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<DSModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(na == 1)
            {
                kStr += " offset:" + std::to_string(offset);
            }
            else if(na == 2)
            {
                kStr += " offset0:" + std::to_string(offset0)
                        + " offset1:" + std::to_string(offset1);
            }
            if(gds)
            {
                kStr += " gds";
            }
            return kStr;
        }

        int  na;
        int  offset;
        int  offset0;
        int  offset1;
        bool gds;
    };

    struct FLATModifiers : public Container
    {
        FLATModifiers(int  offset12 = 0,
                      bool glc      = false,
                      bool slc      = false,
                      bool lds      = false,
                      bool isStore  = false)
            : Container()
            , offset12(offset12)
            , glc(glc)
            , slc(slc)
            , lds(lds)
            , isStore(isStore)
        {
        }

        FLATModifiers(const FLATModifiers& other)
            : Container()
            , offset12(other.offset12)
            , glc(other.glc)
            , slc(other.slc)
            , lds(other.lds)
            , isStore(other.isStore)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<FLATModifiers>(*this);
        }

        std::string toString() const override
        {
            auto        hasGLCModifier = rocIsa::getInstance().getAsmCaps()["HasGLCModifier"];
            std::string kStr;
            if(offset12 != 0)
            {
                kStr += " offset:" + std::to_string(offset12);
            }
            if(glc)
            {
                kStr += " " + getGlcBitName(hasGLCModifier);
            }
            if(slc)
            {
                kStr += " " + getSlcBitName(hasGLCModifier);
            }
            if(lds)
            {
                kStr += " lds";
            }
            return kStr;
        }

        int  offset12;
        bool glc;
        bool slc;
        bool lds;
        bool isStore;
    };

    struct GLOBALModifiers : public Container
    {
        GLOBALModifiers(int offset = 0)
            : Container()
            , offset(offset)
        {
        }

        GLOBALModifiers(const GLOBALModifiers& other)
            : Container()
            , offset(other.offset)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<GLOBALModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(offset != 0)
            {
                kStr += " offset:" + std::to_string(offset);
            }
            return kStr;
        }

        int  offset;
    };

    struct MUBUFModifiers : public Container
    {
        MUBUFModifiers(bool offen    = false,
                       int  offset12 = 0,
                       bool glc      = false,
                       bool slc      = false,
                       bool nt       = false,
                       bool lds      = false,
                       bool isStore  = false)
            : Container()
            , offen(offen)
            , offset12(offset12)
            , glc(glc)
            , slc(slc)
            , nt(nt)
            , lds(lds)
            , isStore(isStore)
        {
        }

        MUBUFModifiers(const MUBUFModifiers& other)
            : Container()
            , offen(other.offen)
            , offset12(other.offset12)
            , glc(other.glc)
            , slc(other.slc)
            , nt(other.nt)
            , lds(other.lds)
            , isStore(other.isStore)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<MUBUFModifiers>(*this);
        }

        std::string toString() const override
        {
            auto        hasGLCModifier = rocIsa::getInstance().getAsmCaps()["HasGLCModifier"];
            auto        hasSLCModifier = rocIsa::getInstance().getAsmCaps()["HasSLCModifier"];
            auto        hasNTModifier  = rocIsa::getInstance().getAsmCaps()["HasNTModifier"];
            std::string kStr;
            if(offen)
            {
                kStr += " offen offset:" + std::to_string(offset12);
            }
            if(glc || slc || lds)
            {
                kStr += ",";
            }
            if(glc)
            {
                kStr += " " + getGlcBitName(hasGLCModifier);
            }
            if(slc)
            {
                kStr += " " + getSlcBitName(hasGLCModifier);
            }
            if(hasNTModifier && nt)
            {
                kStr += " nt";
            }
            if(lds)
            {
                kStr += " lds";
            }
            return kStr;
        }

        bool offen;
        int  offset12;
        bool glc;
        bool slc;
        bool nt;
        bool lds;
        bool isStore;
    };

    struct SMEMModifiers : public Container
    {
        SMEMModifiers(bool glc = false, bool nv = false, int offset = 0)
            : Container()
            , glc(glc)
            , nv(nv)
            , offset(offset) // 20u 21s shaes the same
        {
        }

        SMEMModifiers(const SMEMModifiers& other)
            : Container()
            , glc(other.glc)
            , nv(other.nv)
            , offset(other.offset)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<SMEMModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(offset != 0)
            {
                kStr += " offset:" + std::to_string(offset);
            }
            if(glc)
            {
                kStr += " glc";
            }
            if(nv)
            {
                kStr += " nv";
            }
            return kStr;
        }

        bool glc;
        bool nv;
        int  offset;
    };

    struct SDWAModifiers : public Container
    {
        SDWAModifiers(SelectBit dst_sel    = SelectBit::SEL_NONE,
                      UnusedBit dst_unused = UnusedBit::UNUSED_NONE,
                      SelectBit src0_sel   = SelectBit::SEL_NONE,
                      SelectBit src1_sel   = SelectBit::SEL_NONE)
            : Container()
            , dst_sel(dst_sel)
            , dst_unused(dst_unused)
            , src0_sel(src0_sel)
            , src1_sel(src1_sel)
        {
        }

        SDWAModifiers(const SDWAModifiers& other)
            : Container()
            , dst_sel(other.dst_sel)
            , dst_unused(other.dst_unused)
            , src0_sel(other.src0_sel)
            , src1_sel(other.src1_sel)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<SDWAModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(dst_sel != SelectBit::SEL_NONE)
                kStr += " dst_sel:" + ::rocisa::toString(dst_sel);
            if(dst_unused != UnusedBit::UNUSED_NONE)
                kStr += " dst_unused:" + ::rocisa::toString(dst_unused);
            if(src0_sel != SelectBit::SEL_NONE)
                kStr += " src0_sel:" + ::rocisa::toString(src0_sel);
            if(src1_sel != SelectBit::SEL_NONE)
                kStr += " src1_sel:" + ::rocisa::toString(src1_sel);
            return kStr;
        }

        SelectBit dst_sel;
        UnusedBit dst_unused;
        SelectBit src0_sel;
        SelectBit src1_sel;
    };

    // dot2: for WaveSplitK reduction. Only a subset of DPP modifiers are used here
    struct DPPModifiers : public Container
    {
        int row_shr;
        int row_bcast;
        int bound_ctrl;

        DPPModifiers(int row_shr = -1, int row_bcast = -1, int bound_ctrl = -1)
            : row_shr(row_shr)
            , row_bcast(row_bcast)
            , bound_ctrl(bound_ctrl)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<DPPModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(row_shr != -1)
                kStr += " row_shr:" + std::to_string(row_shr);
            if(row_bcast != -1)
                kStr += " row_bcast:" + std::to_string(row_bcast);
            if(bound_ctrl != -1)
                kStr += " bound_ctrl:" + std::to_string(bound_ctrl);
            return kStr;
        }
    };

    struct VOP3PModifiers : public Container
    {
        VOP3PModifiers(const std::vector<int>& op_sel    = {},
                       const std::vector<int>& op_sel_hi = {},
                       const std::vector<int>& byte_sel  = {})
            : Container()
            , op_sel(op_sel)
            , op_sel_hi(op_sel_hi)
            , byte_sel(byte_sel)
        {
        }

        VOP3PModifiers(const VOP3PModifiers& other)
            : Container()
            , op_sel(other.op_sel)
            , op_sel_hi(other.op_sel_hi)
            , byte_sel(other.byte_sel)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<VOP3PModifiers>(*this);
        }

        std::string toString() const override
        {
            std::string kStr;
            if(!op_sel.empty())
            {
                kStr += " op_sel:" + vectorToString(op_sel);
            }
            if(!op_sel_hi.empty())
            {
                kStr += " op_sel_hi:" + vectorToString(op_sel_hi);
            }
            if(!byte_sel.empty())
            {
                kStr += " byte_sel:" + vectorToString(byte_sel);
            }
            return kStr;
        }

        std::vector<int> op_sel;
        std::vector<int> op_sel_hi;
        std::vector<int> byte_sel;

        std::string vectorToString(const std::vector<int>& vec) const
        {
            std::string result = "[";
            for(size_t i = 0; i < vec.size(); ++i)
            {
                result += std::to_string(vec[i]);
                if(i < vec.size() - 1)
                {
                    result += ",";
                }
            }
            result += "]";
            return result;
        }
    };

    struct EXEC : public Container
    {
        EXEC(bool setHi = false)
            : Container()
            , setHi(setHi)
        {
        }

        EXEC(const EXEC& other)
            : Container()
            , setHi(other.setHi)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<EXEC>(*this);
        }

        std::string toString() const override
        {
            auto wavefront = rocIsa::getInstance().getKernel().wavefront;
            return wavefront == 64 ? "exec" : "exec_lo";
        }

        bool setHi;
    };

    struct VCC : public Container
    {
        VCC(bool setHi = false)
            : Container()
            , setHi(setHi)
        {
        }

        VCC(const VCC& other)
            : Container()
            , setHi(other.setHi)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<VCC>(*this);
        }

        std::string toString() const override
        {
            auto wavefront = rocIsa::getInstance().getKernel().wavefront;
            return wavefront == 64 ? "vcc" : (setHi ? "vcc_hi" : "vcc_lo");
        }

        bool setHi;
    };

    struct HWRegContainer : public Container
    {
        HWRegContainer(const std::string& reg, const std::vector<int>& value)
            : Container()
            , reg(reg)
            , value(value)
        {
        }

        HWRegContainer(const HWRegContainer& other)
            : Container()
            , reg(other.reg)
            , value(other.value)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<HWRegContainer>(*this);
        }

        std::string toString() const override
        {
            std::string s = "hwreg(" + reg;
            for(const auto& v : value)
            {
                s += "," + std::to_string(v);
            }
            s += ")";
            return s;
        }

        std::string      reg;
        std::vector<int> value;
    };

    struct RegName
    {
        std::string      name;
        std::vector<int> offsets;

        RegName(const std::string& name = "", const std::vector<int>& offsets = {})
            : name(name)
            , offsets(offsets)
        {
        }

        RegName(const RegName& other)
            : name(other.name)
            , offsets(other.offsets)
        {
        }

        RegName(RegName&& other) noexcept
            : name(std::move(other.name))
            , offsets(std::move(other.offsets))
        {
        }

        RegName& operator=(const RegName& other)
        {
            if(this != &other)
            {
                name    = other.name;
                offsets = other.offsets;
            }
            return *this;
        }

        RegName& operator=(RegName&& other) noexcept
        {
            if(this != &other)
            {
                name    = std::move(other.name);
                offsets = std::move(other.offsets);
            }
            return *this;
        }

        int getTotalOffsets() const
        {
            int total = 0;
            for(int offset : offsets)
            {
                total += offset;
            }
            return total;
        }

        std::size_t hash() const
        {
            std::size_t seed = 0;
            seed ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            for(const auto& offset : offsets)
            {
                seed ^= std::hash<int>{}(offset) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }

        bool operator==(const RegName& other) const
        {
            return name == other.name && offsets == other.offsets;
        }

        bool operator!=(const RegName& other) const
        {
            return !(*this == other);
        }

        std::string toString() const
        {
            std::string ss = name;
            for(int offset : offsets)
            {
                ss += "+" + std::to_string(offset);
            }
            return ss;
        }

    private:
        RegName() {}
    };

    struct RegisterContainer : public Container
    {
        std::string            regType;
        std::optional<RegName> regName;
        int                    regIdx;
        int                    regNum;
        bool                   isInlineAsm;
        bool                   isMinus;
        bool                   isAbs;
        bool                   isMacro;

        RegisterContainer(const std::string&            regType,
                          const std::optional<RegName>& regName,
                          int                           regIdx = 0,
                          float                         regNum = 1)
            : Container()
            , regType(regType)
            , regName(std::move(regName))
            , regIdx(regIdx)
            , regNum(int(ceil(regNum)))
            , isInlineAsm(false)
            , isMinus(false)
            , isAbs(false)
            , isMacro(false)
        {
        }

        RegisterContainer(const std::string&            regType,
                          const std::optional<RegName>& regName,
                          bool                          isAbs,
                          bool                          isMacro,
                          int                           regIdx = 0,
                          float                         regNum = 1)
            : Container()
            , regType(regType)
            , regName(std::move(regName))
            , regIdx(regIdx)
            , regNum(int(ceil(regNum)))
            , isInlineAsm(false)
            , isMinus(false)
            , isAbs(isAbs)
            , isMacro(isMacro)
        {
        }

        RegisterContainer(const RegisterContainer& other)
            : Container()
            , regType(other.regType)
            , regName(other.regName)
            , regIdx(other.regIdx)
            , regNum(other.regNum)
            , isInlineAsm(other.isInlineAsm)
            , isMinus(other.isMinus)
            , isAbs(other.isAbs)
            , isMacro(other.isMacro)
        {
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<RegisterContainer>(*this);
        }

        std::shared_ptr<RegisterContainer> clone2() const
        {
            return std::make_shared<RegisterContainer>(*this);
        }

        RegisterContainer(RegisterContainer&& other) noexcept
            : Container()
            , regType(std::move(other.regType))
            , regName(std::move(other.regName))
            , regIdx(other.regIdx)
            , regNum(other.regNum)
            , isInlineAsm(other.isInlineAsm)
            , isMinus(other.isMinus)
            , isAbs(other.isAbs)
            , isMacro(other.isMacro)
        {
        }

        RegisterContainer& operator=(const RegisterContainer& other)
        {
            if(this != &other)
            {
                regType     = other.regType;
                regName     = other.regName;
                regIdx      = other.regIdx;
                regNum      = other.regNum;
                isInlineAsm = other.isInlineAsm;
                isMinus     = other.isMinus;
                isAbs       = other.isAbs;
                isMacro     = other.isMacro;
            }
            return *this;
        }

        RegisterContainer& operator=(RegisterContainer&& other) noexcept
        {
            if(this != &other)
            {
                regType     = std::move(other.regType);
                regName     = std::move(other.regName);
                regIdx      = other.regIdx;
                regNum      = other.regNum;
                isInlineAsm = other.isInlineAsm;
                isMinus     = other.isMinus;
                isAbs       = other.isAbs;
                isMacro     = other.isMacro;
            }
            return *this;
        }

        void setInlineAsm(bool setting)
        {
            isInlineAsm = setting;
        }

        void setMinus(bool isMinus)
        {
            this->isMinus = isMinus;
        }

        void setAbs(bool isAbs)
        {
            this->isAbs = isAbs;
        }

        RegisterContainer getMinus() const
        {
            RegisterContainer c = *this;
            c.setMinus(true);
            return c;
        }

        void replaceRegName(const std::string& srcName, int dst)
        {
            if(regName)
            {
                if(regName->name == srcName) // Exact match
                {
                    regIdx  = dst + regName->getTotalOffsets();
                    regName = std::nullopt;
                }
                else
                {
                    size_t pos = regName->name.find(srcName);
                    if(pos != std::string::npos)
                    {
                        regName->name.replace(pos, srcName.length(), std::to_string(dst));
                    }
                }
            }
        }

        void replaceRegName(const std::string& srcName, const std::string& dst)
        {
            if(regName)
            {
                size_t pos = regName->name.find(srcName);
                if(pos != std::string::npos)
                {
                    regName->name.replace(pos, srcName.length(), dst);
                }
            }
        }

        std::string getRegNameWithType() const
        {
            return regType + "gpr" + regName->name;
        }

        std::string getCompleteRegNameWithType() const
        {
            return regType + "gpr" + regName->toString();
        }

        std::pair<std::shared_ptr<RegisterContainer>, std::shared_ptr<RegisterContainer>>
            splitRegContainer() const
        {
            RegisterContainer r1        = *this;
            RegisterContainer r2        = *this;
            int               newRegNum = ceil(regNum / 2);
            if(regName)
            {
                r2.regName->offsets.push_back(1);
            }
            else
            {
                r2.regIdx += 1;
            }
            r1.regNum = newRegNum;
            r2.regNum = regNum - newRegNum;
            return {std::make_shared<RegisterContainer>(r1),
                    std::make_shared<RegisterContainer>(r2)};
        }
        std::size_t hash() const
        {
            std::size_t seed = 0;
            seed ^= std::hash<std::string>{}(regType) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(regIdx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(regNum) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            if(regName)
                seed ^= regName->hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }

        bool operator==(const RegisterContainer& other) const
        {
            return regType == other.regType && regIdx == other.regIdx && regNum == other.regNum
                   && regName == other.regName;
        }

        bool operator!=(const RegisterContainer& other) const
        {
            return !(*this == other);
        }

        std::string toString() const override
        {
            std::string minusStr = isMinus ? "-" : "";
            minusStr             = isAbs ? "abs(" + minusStr : minusStr;
            auto absStr          = isAbs ? ")" : "";
            if(isInlineAsm)
            {
                return minusStr + "%" + std::to_string(regIdx) + absStr;
            }

            if(regName)
            {
                std::string macroSlash = isMacro ? "\\" : "";
                if(regNum == 1)
                {
                    return minusStr + regType + "[" + macroSlash + regType + "gpr"
                           + regName->toString() + "]" + absStr;
                }
                else
                {
                    return minusStr + regType + "[" + macroSlash + regType + "gpr"
                           + regName->toString() + ":" + regType + "gpr" + regName->toString() + "+"
                           + std::to_string(regNum - 1) + "]" + absStr;
                }
            }
            else
            {
                if(regNum == 1)
                {
                    return minusStr + regType + std::to_string(regIdx) + absStr;
                }
                else
                {
                    return minusStr + regType + "[" + std::to_string(regIdx) + ":"
                           + std::to_string(regIdx + regNum - 1) + "]" + absStr;
                }
            }
        }

        bool sameRegBaseAddr(const RegisterContainer& b) const
        {
            if(regName && b.regName)
            {
                return regName->name == b.regName->name;
            }
            else if(!regName && !b.regName)
            {
                return regIdx == b.regIdx;
            }
            return false;
        }

        bool operator&&(const RegisterContainer& b) const
        {
            if(sameRegBaseAddr(b))
            {
                int lenA = regNum;
                int offsetA
                    = regName ? std::accumulate(regName->offsets.begin(), regName->offsets.end(), 0)
                              : 0;
                int lenB = b.regNum;
                int offsetB
                    = b.regName
                          ? std::accumulate(b.regName->offsets.begin(), b.regName->offsets.end(), 0)
                          : 0;
                std::pair<int, int> rangeA = {offsetA, offsetA + lenA};
                std::pair<int, int> rangeB = {offsetB, offsetB + lenB};
                if(rangeA.first > rangeB.first)
                {
                    std::swap(rangeA, rangeB);
                }
                return rangeA.second > rangeB.first;
            }
            return false;
        }
    };

    struct HolderContainer : public RegisterContainer
    {
        std::string holderName;
        int         holderIdx;
        int         holderType;

        HolderContainer(const std::string& regType, const std::string& holderName, float regNum)
            : RegisterContainer(regType, RegName(holderName), 0, regNum)
            , holderName(holderName)
            , holderIdx(0)
            , holderType(1)
        {
        }

        HolderContainer(const std::string& regType, const RegName& regName, float regNum)
            : RegisterContainer(regType, regName, 0, regNum)
            , holderName(regName.name)
            , holderIdx(0)
            , holderType(1)
        {
        }

        HolderContainer(const std::string& regType, int holderIdx, float regNum)
            : RegisterContainer(regType, std::nullopt, holderIdx, regNum)
            , holderName("")
            , holderIdx(holderIdx)
            , holderType(0)
        {
        }

        HolderContainer(const HolderContainer& other)
            : RegisterContainer(other)
            , holderName(other.holderName)
            , holderIdx(other.holderIdx)
            , holderType(other.holderType)
        {
        }

        HolderContainer(HolderContainer&& other) noexcept
            : RegisterContainer(std::move(other))
            , holderName(std::move(other.holderName))
            , holderIdx(other.holderIdx)
            , holderType(other.holderType)
        {
        }

        HolderContainer& operator=(const HolderContainer& other)
        {
            if(this != &other)
            {
                RegisterContainer::operator=(other);
                holderName = other.holderName;
                holderIdx  = other.holderIdx;
                holderType = other.holderType;
            }
            return *this;
        }

        HolderContainer& operator=(HolderContainer&& other) noexcept
        {
            if(this != &other)
            {
                RegisterContainer::operator=(std::move(other));
                holderName = std::move(other.holderName);
                holderIdx  = other.holderIdx;
                holderType = other.holderType;
            }
            return *this;
        }

        std::shared_ptr<Container> clone() const override
        {
            return std::make_shared<HolderContainer>(*this);
        }

        void setRegNum(int num)
        {
            if(holderType == 0)
            {
                regIdx = holderIdx + num;
            }
            else if(holderType == 1)
            {
                regName = std::move(RegName(holderName));
                regName->offsets.insert(regName->offsets.begin(), num);
            }
        }

        RegisterContainer getCopiedRC() const
        {
            if(holderType == 0)
            {
                return RegisterContainer{regType, std::nullopt, regIdx, (float)regNum};
            }
            return RegisterContainer{regType, regName, regIdx, (float)regNum};
        }

        std::pair<std::shared_ptr<HolderContainer>, std::shared_ptr<HolderContainer>>
            splitRegContainer() const
        {
            HolderContainer r1        = *this;
            HolderContainer r2        = *this;
            int             newRegNum = ceil(regNum / 2);
            if(!holderName.empty())
            {
                r2.regName->offsets.push_back(1);
            }
            else
            {
                r2.holderIdx += 1;
            }
            r1.regNum = newRegNum;
            r2.regNum = regNum - newRegNum;
            return {std::make_shared<HolderContainer>(r1), std::make_shared<HolderContainer>(r2)};
        }
    };

    // Helper function to generate register name
    inline RegName generateRegName(const std::string& rawText)
    {
        std::vector<int> offsets;
        size_t           pos      = rawText.find('+');
        std::string      baseName = rawText.substr(0, pos);
        while(pos != std::string::npos)
        {
            size_t nextPos = rawText.find('+', pos + 1);
            offsets.push_back(std::stoi(rawText.substr(pos + 1, nextPos - pos - 1)));
            pos = nextPos;
        }
        return std::move(RegName(baseName, offsets));
    }

    struct Holder
    {
        Holder(int idx)
            : idx(idx)
            , name(std::nullopt)
        {
        }
        Holder(const std::string& name)
            : idx(-1)
            , name(std::move(generateRegName(name)))
        {
        }

        Holder(const Holder& other)
            : idx(other.idx)
            , name(other.name)
        {
        }

        Holder(Holder&& other) noexcept
            : idx(other.idx)
            , name(std::move(other.name))
        {
        }

        Holder& operator=(const Holder& other)
        {
            if(this != &other)
            {
                idx  = other.idx;
                name = other.name;
            }
            return *this;
        }

        Holder& operator=(Holder&& other) noexcept
        {
            if(this != &other)
            {
                idx  = other.idx;
                name = std::move(other.name);
            }
            return *this;
        }

        int                    idx;
        std::optional<RegName> name;
    };

    std::shared_ptr<Item> replaceHolder(std::shared_ptr<Item> inst, int dst);

    // Overloaded functions to create specific GPR containers with default regNum = 1.f
    std::shared_ptr<RegisterContainer> vgpr(const Holder& holder, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> vgpr(int idx, float regNum = 1.f);
    std::shared_ptr<RegisterContainer>
        vgpr(const std::string& name, float regNum = 1.f, bool isMacro = false, bool isAbs = false);
    std::shared_ptr<RegisterContainer> sgpr(const Holder& holder, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> sgpr(int idx, float regNum = 1.f);
    std::shared_ptr<RegisterContainer>
        sgpr(const std::string& name, float regNum = 1.f, bool isMacro = false);
    std::shared_ptr<RegisterContainer> accvgpr(const Holder& holder, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> accvgpr(int idx, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> accvgpr(const std::string& name, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> mgpr(const Holder& holder, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> mgpr(int idx, float regNum = 1.f);
    std::shared_ptr<RegisterContainer> mgpr(const std::string& name, float regNum = 1.f);

    struct ContinuousRegister
    {
        uint32_t idx;
        uint32_t size;
    };
} // namespace rocisa
