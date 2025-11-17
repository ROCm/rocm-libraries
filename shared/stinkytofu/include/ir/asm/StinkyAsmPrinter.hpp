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

#include <ostream>
#include <sstream>
#include <string>

#include "ir/asm/StinkyAsmIR.hpp"
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

    // Stream operator overloads for convenient printing
    inline std::ostream& operator<<(std::ostream& os, const StinkyRegister& reg)
    {
        RegisterPrinter printer(os, AsmPrinterOptions());
        printer.print(reg);
        return os;
    }

} // namespace stinkytofu
