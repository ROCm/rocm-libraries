/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>

#include <Tensile/Debug.hpp>

#include <Tensile/Macros.hpp>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{
/**
 * @brief Helper class for tabulated predicate debug output.
 *
 * In simplified mode (TENSILE_DB=0x10), only failing predicates are shown.
 * In verbose mode (TENSILE_DB=0x40010), all predicates are shown.
 */
class PredicateDebugger
{
public:
    static constexpr int COL_PASS      = 6;
    static constexpr int COL_PREDICATE = 30;
    static constexpr int INDENT_SIZE   = 2;

    static int& indent() { thread_local int level = 0; return level; }
    static void pushIndent() { indent()++; }
    static void popIndent() { if(indent() > 0) indent()--; }
    static void resetIndent() { indent() = 0; }

    static std::string& pendingTitle() { thread_local std::string t; return t; }
    static bool& headerPrinted() { thread_local bool p = false; return p; }
    static bool verbose() { return Debug::Instance().printPredicateEvaluationVerbose(); }

    static void printSeparator(std::ostream& stream)
    {
        stream << std::string(80, '-') << std::endl;
    }

    static void printLibraryFileBanner(std::ostream& stream, const std::string& filename)
    {
        stream << std::endl;
        stream << std::string(80, '=') << std::endl;
        stream << "LIBRARY: " << filename << std::endl;
        stream << std::string(80, '=') << std::endl;
    }

    static void printHeader(std::ostream& stream, const std::string& title)
    {
        resetIndent();
        pendingTitle()  = title;
        headerPrinted() = false;
        if(verbose())
            flushHeader(stream);
    }

    static void printFooter(std::ostream& stream, bool result)
    {
        if(!verbose() && !headerPrinted() && result)
        {
            pendingTitle().clear();
            return;
        }
        if(!headerPrinted() && !result)
            flushHeader(stream);
        if(headerPrinted())
        {
            printSeparator(stream);
            stream << "Result: " << (result ? "MATCH" : "NO MATCH") << std::endl;
            printSeparator(stream);
            stream << std::endl;
        }
        pendingTitle().clear();
        headerPrinted() = false;
    }

    static void printRow(std::ostream&      stream,
                         bool               pass,
                         const std::string& predicate,
                         const std::string& details = "")
    {
        if(pass && !verbose())
            return;
        flushHeader(stream);
        std::string passStr(pass ? "[OK]" : "[!!]");
        std::string indentStr(indent() * INDENT_SIZE, ' ');
        stream << std::left << std::setw(COL_PASS) << passStr << indentStr
               << std::setw(std::max(0, COL_PREDICATE - static_cast<int>(indentStr.size())))
               << predicate << details << std::endl;
    }

private:
    static void flushHeader(std::ostream& stream)
    {
        if(!headerPrinted() && !pendingTitle().empty())
        {
            stream << std::endl;
            printSeparator(stream);
            stream << "PREDICATE: " << pendingTitle() << std::endl;
            printSeparator(stream);
            headerPrinted() = true;
        }
    }
};
}  // namespace TensileLite

TENSILE_HIDDEN_END
