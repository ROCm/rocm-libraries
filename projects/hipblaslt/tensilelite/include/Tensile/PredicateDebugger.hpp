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

#include <iomanip>
#include <ostream>
#include <string>

namespace TensileLite
{
/**
 * @brief Helper class for tabulated predicate debug output
 *
 * Provides consistent formatting for predicate evaluation results
 * in a clean, readable format.
 */
class PredicateDebugger
{
 public:
    static constexpr int COL_PASS      = 6;
    static constexpr int COL_PREDICATE = 30;
    static constexpr int INDENT_SIZE   = 2;

    // Simple indent tracking
    static int& indent()
    {
        thread_local int level = 0;
        return level;
    }
    static void pushIndent() { indent()++; }
    static void popIndent() { if(indent() > 0) indent()--; }
    static void resetIndent() { indent() = 0; }

    /**
     * @brief Print a separator line
     */
    static void printSeparator(std::ostream& stream)
    {
        stream << std::string(80, '-') << std::endl;
    }

    /**
     * @brief Print the table header
     */
    static void printHeader(std::ostream& stream, const std::string& title)
    {
        resetIndent();
        stream << std::endl;
        printSeparator(stream);
        stream << "PREDICATE: " << title << std::endl;
        printSeparator(stream);
    }

    /**
     * @brief Print the table footer with overall result
     */
    static void printFooter(std::ostream& stream, bool result)
    {
        printSeparator(stream);
        stream << "Result: " << (result ? "MATCH" : "NO MATCH") << std::endl;
        printSeparator(stream);
        stream << std::endl;
    }

    /**
     * @brief Print a single predicate row
     */
    static void printRow(std::ostream&      stream,
                         bool               pass,
                         const std::string& predicate,
                         const std::string& details)
    {
        std::string passStr   = pass ? "[OK]" : "[!!]";
        std::string indentStr = std::string(indent() * INDENT_SIZE, ' ');

        stream << std::left << std::setw(COL_PASS) << passStr
               << indentStr << std::setw(std::max(0, COL_PREDICATE - static_cast<int>(indentStr.size())))
               << predicate
               << details << std::endl;
    }
};
}  // namespace TensileLite
