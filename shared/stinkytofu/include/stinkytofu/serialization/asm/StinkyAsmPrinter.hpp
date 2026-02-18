/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include <iosfwd>
#include <sstream>
#include <string>

#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyMacro.hpp"

namespace stinkytofu
{
    // AsmPrinter configuration options
    struct AsmPrinterOptions
    {
        // Indentation for nested structures
        int indent = 2;
    };

    // Base class for printing different IR elements
    class AsmPrinterBase
    {
    public:
        AsmPrinterBase(std::ostream& os, const AsmPrinterOptions& options)
            : os(os)
            , options(options)
        {
        }

        virtual ~AsmPrinterBase() = default;

        std::ostream& getStream()
        {
            return os;
        }

        const AsmPrinterOptions& getOptions() const
        {
            return options;
        }

    protected:
        std::ostream&     os;
        AsmPrinterOptions options;
    };

    // Printer for StinkyRegister
    class RegisterPrinter : public AsmPrinterBase
    {
    public:
        RegisterPrinter(std::ostream& os, const AsmPrinterOptions& options = AsmPrinterOptions())
            : AsmPrinterBase(os, options)
        {
        }

        void print(const StinkyRegister& reg);
    };

    // Printer for AsmDirective (low-level IR)
    class DirectivePrinter : public AsmPrinterBase
    {
    public:
        DirectivePrinter(std::ostream& os, const AsmPrinterOptions& options = AsmPrinterOptions())
            : AsmPrinterBase(os, options)
        {
        }

        void print(const AsmDirective& directive);
    };

    // Printer for MacroInstruction (low-level IR)
    class MacroPrinter : public AsmPrinterBase
    {
    public:
        MacroPrinter(std::ostream& os, const AsmPrinterOptions& options = AsmPrinterOptions())
            : AsmPrinterBase(os, options)
        {
        }

        void print(const MacroInstruction& macro);
    };

    // Main AsmPrinter for StinkyInstruction
    class AsmPrinter : public AsmPrinterBase
    {
    public:
        AsmPrinter(std::ostream& os, const AsmPrinterOptions& options = AsmPrinterOptions())
            : AsmPrinterBase(os, options)
            , regPrinter(os, options)
        {
        }

        // Print a single StinkyInstruction
        void print(const StinkyInstruction& inst);

        // Print an entire Function
        void print(const Function& function);

    private:
        RegisterPrinter regPrinter;
    };

    // Utility functions for quick printing
    inline std::string toString(const StinkyRegister& reg)
    {
        std::ostringstream oss;
        RegisterPrinter    printer(oss, AsmPrinterOptions());
        printer.print(reg);
        return oss.str();
    }

    inline std::string toString(const StinkyInstruction& inst,
                                const AsmPrinterOptions& options = AsmPrinterOptions())
    {
        std::ostringstream oss;
        AsmPrinter         printer(oss, options);
        printer.print(inst);
        return oss.str();
    }

    inline std::string toString(const Function&          function,
                                const AsmPrinterOptions& options = AsmPrinterOptions())
    {
        std::ostringstream oss;
        AsmPrinter         printer(oss, options);
        printer.print(function);
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Function& function)
    {
        AsmPrinter printer(os, AsmPrinterOptions());
        printer.print(function);
        return os;
    }

    // Stream operator overloads for convenient printing
    inline std::ostream& operator<<(std::ostream& os, const StinkyRegister& reg)
    {
        RegisterPrinter printer(os, AsmPrinterOptions());
        printer.print(reg);
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const AsmDirective& directive)
    {
        DirectivePrinter printer(os, AsmPrinterOptions());
        printer.print(directive);
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const MacroInstruction& macro)
    {
        MacroPrinter printer(os, AsmPrinterOptions());
        printer.print(macro);
        return os;
    }

    // Stream operators for instruction modifiers
    inline std::ostream& operator<<(std::ostream& os, const SWaitCntData& waitCntData)
    {
        os << "vlcnt=" << (int)waitCntData.vlcnt << ", vscnt=" << (int)waitCntData.vscnt
           << ", dlcnt=" << (int)waitCntData.dlcnt << ", dscnt=" << (int)waitCntData.dscnt
           << ", kmcnt=" << (int)waitCntData.kmcnt;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SWaitTensorCntData& waitTensorCntData)
    {
        os << "tlcnt=" << (int)waitTensorCntData.tlcnt;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SDelayAluData& delayAluData)
    {
        auto typeToString = [](SDelayAluData::InstType type) -> const char* {
            switch(type)
            {
            case SDelayAluData::InstType::VALU:
                return "VALU";
            case SDelayAluData::InstType::SALU:
                return "SALU";
            case SDelayAluData::InstType::TRANS:
                return "TRANS";
            case SDelayAluData::InstType::NO_DEP:
                return "NO_DEP";
            default:
                return "UNKNOWN";
            }
        };

        auto skipToString = [](int8_t skip) -> const char* {
            switch(skip)
            {
            case 0:
                return "SAME";
            case 1:
                return "NEXT";
            case 2:
                return "SKIP_1";
            case 3:
                return "SKIP_2";
            case 4:
                return "SKIP_3";
            case 5:
                return "SKIP_4";
            default:
                return "UNKNOWN";
            }
        };

        // Helper to format instid with type and distance
        auto formatInstId = [&](SDelayAluData::InstType type, int8_t distance) -> std::string {
            // If distance is 0, print NO_DEP regardless of type (matches rocisa behavior)
            if(distance == 0)
            {
                return "NO_DEP";
            }

            std::string result = typeToString(type);

            // SALU uses CYCLE_N, others use DEP_N
            if(type == SDelayAluData::InstType::SALU)
            {
                result += "_CYCLE_" + std::to_string((int)distance);
            }
            else if(type != SDelayAluData::InstType::NO_DEP)
            {
                result += "_DEP_" + std::to_string((int)distance);
            }

            return result;
        };

        // Print InstID0
        os << "instid0(" << formatInstId(delayAluData.instid0Type, delayAluData.instid0Distance)
           << ")";

        // Print InstID1 if present
        if(delayAluData.hasInstId1)
        {
            os << " | instskip(" << skipToString(delayAluData.instSkip) << ")";
            os << " | instid1("
               << formatInstId(delayAluData.instid1Type, delayAluData.instid1Distance) << ")";
        }

        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SWaitAluData& waitAluData)
    {
        auto emitField = [&](SWaitAluData::Field field, const char* name) {
            if(waitAluData.hasField(field))
                os << " depctr_" << name << "(" << waitAluData.getField(field) << ")";
        };

        emitField(SWaitAluData::VA_VDST, "va_vdst");
        emitField(SWaitAluData::VA_SDST, "va_sdst");
        emitField(SWaitAluData::VA_SSRC, "va_ssrc");
        emitField(SWaitAluData::HOLD_CNT, "hold_cnt");
        emitField(SWaitAluData::VM_VSRC, "vm_vsrc");
        emitField(SWaitAluData::VA_VCC, "va_vcc");
        emitField(SWaitAluData::SA_SDST, "sa_sdst");

        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const DSModifiers& dsMod)
    {
        if(dsMod.na == 1)
        {
            os << " offset:" << dsMod.offset;
        }
        else if(dsMod.na == 2)
        {
            os << " offset0:" << dsMod.offset0 << " offset1:" << dsMod.offset1;
        }
        if(dsMod.gds)
        {
            os << " gds";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const FLATModifiers& flatMod)
    {
        if(flatMod.offset12 != 0)
        {
            os << " offset:" << flatMod.offset12;
        }
        if(flatMod.glc)
        {
            os << " glc";
        }
        if(flatMod.slc)
        {
            os << " slc";
        }
        if(flatMod.lds)
        {
            os << " lds";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const MUBUFModifiers& mubufMod)
    {
        if(mubufMod.offen)
        {
            os << " offen offset:" << mubufMod.offset12;
        }
        if(mubufMod.glc || mubufMod.slc || mubufMod.lds)
        {
            os << ",";
        }
        if(mubufMod.glc)
        {
            os << " glc";
        }
        if(mubufMod.slc)
        {
            os << " slc";
        }
        if(mubufMod.nt)
        {
            os << " nt";
        }
        if(mubufMod.lds)
        {
            os << " lds";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SMEMModifiers& smemMod)
    {
        if(smemMod.offset != 0)
        {
            os << " offset:" << smemMod.offset;
        }
        if(smemMod.glc)
        {
            os << " glc";
        }
        if(smemMod.nv)
        {
            os << " nv";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const MFMAModifiers& mfmaMod)
    {
        if(!mfmaMod.inputPermute.empty())
        {
            os << " " << mfmaMod.inputPermute;
        }
        if(mfmaMod.reuseA)
        {
            os << " matrix_a_reuse";
        }
        if(mfmaMod.reuseB)
        {
            os << " matrix_b_reuse";
        }
        if(!mfmaMod.negStr.empty())
        {
            os << " " << mfmaMod.negStr;
        }
        return os;
    }

    // Helper function to format vector as [a,b,c]
    inline std::string vectorToString(const std::vector<int>& vec)
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

    inline std::ostream& operator<<(std::ostream& os, const VOP3PModifiers& vop3pMod)
    {
        if(!vop3pMod.op_sel.empty())
        {
            os << " op_sel:" << vectorToString(vop3pMod.op_sel);
        }
        if(!vop3pMod.op_sel_hi.empty())
        {
            os << " op_sel_hi:" << vectorToString(vop3pMod.op_sel_hi);
        }
        if(!vop3pMod.byte_sel.empty())
        {
            os << " byte_sel:" << vectorToString(vop3pMod.byte_sel);
        }
        return os;
    }

} // namespace stinkytofu
