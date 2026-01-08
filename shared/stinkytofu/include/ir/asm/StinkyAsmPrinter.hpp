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

#include <ostream>
#include <sstream>
#include <string>

#include "ir/asm/StinkyAsmDirectives.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyMacro.hpp"
#include "stinkytofu.hpp"

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

        // Print an entire IRList
        void print(const IRList& irlist);

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

    inline std::string toString(const IRList&            irlist,
                                const AsmPrinterOptions& options = AsmPrinterOptions())
    {
        std::ostringstream oss;
        AsmPrinter         printer(oss, options);
        printer.print(irlist);
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const IRList& irlist)
    {
        AsmPrinter printer(os, AsmPrinterOptions());
        printer.print(irlist);
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

        os << "instid0(" << typeToString(delayAluData.type);

        // SALU always uses CYCLE_1, others use DEP_N
        if(delayAluData.type == SDelayAluData::InstType::SALU)
        {
            os << "_CYCLE_1";
        }
        else if(delayAluData.type != SDelayAluData::InstType::NO_DEP)
        {
            os << "_DEP_" << (int)delayAluData.distance;
        }

        os << ")";
        return os;
    }

} // namespace stinkytofu
