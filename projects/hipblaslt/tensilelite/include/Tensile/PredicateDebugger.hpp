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

#include <Tensile/Debug.hpp>

namespace TensileLite
{
/**
 * @brief Helper class for tabulated predicate debug output
 *
 * Provides consistent formatting for predicate evaluation results
 * in a clean, readable format.
 *
 * In simplified mode (TENSILE_DB=0x10), only failing predicates are shown
 * and empty tables are suppressed.
 * In verbose mode (TENSILE_DB=0x40010), all predicates are shown.
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

    // Pending header state for lazy printing
    static std::string& pendingTitle()
    {
        thread_local std::string title;
        return title;
    }
    static bool& headerPrinted()
    {
        thread_local bool printed = false;
        return printed;
    }

    /**
     * @brief Print a separator line
     */
    static void printSeparator(std::ostream& stream)
    {
        stream << std::string(80, '-') << std::endl;
    }

    /**
     * @brief Print library file banner
     */
    static void printLibraryFileBanner(std::ostream& stream, const std::string& filename)
    {
        stream << std::endl;
        stream << std::string(80, '=') << std::endl;
        stream << "LIBRARY: " << filename << std::endl;
        stream << std::string(80, '=') << std::endl;
    }

    /**
     * @brief Queue a table header for lazy printing
     *
     * In simplified mode, the header is only printed when a failing predicate
     * is encountered. In verbose mode, the header is printed immediately.
     */
    static void printHeader(std::ostream& stream, const std::string& title)
    {
        resetIndent();
        pendingTitle() = title;
        headerPrinted() = false;

        // In verbose mode, print header immediately
        if(Debug::Instance().printPredicateEvaluationVerbose())
        {
            flushHeader(stream);
        }
    }

    /**
     * @brief Actually print the header (called internally)
     */
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

    /**
     * @brief Print the table footer with overall result
     *
     * In simplified mode, only prints if the header was printed (i.e., there
     * were failing predicates). In verbose mode, always prints.
     */
    static void printFooter(std::ostream& stream, bool result)
    {
        bool verbose = Debug::Instance().printPredicateEvaluationVerbose();

        // In simplified mode, only print footer if header was printed
        // (meaning there were failures to show) OR if the overall result failed
        if(!verbose && !headerPrinted() && result)
        {
            // All passed, nothing was shown, skip footer too
            pendingTitle().clear();
            return;
        }

        // If we have a pending header (result failed but no individual failures shown),
        // print a summary line
        if(!headerPrinted() && !result)
        {
            flushHeader(stream);
        }

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

    /**
     * @brief Print a single predicate row
     *
     * In simplified mode (default when TENSILE_DB=0x10), only failing predicates are shown.
     * In verbose mode (TENSILE_DB=0x40010), all predicates are shown.
     */
    static void printRow(std::ostream&      stream,
                         bool               pass,
                         const std::string& predicate,
                         const std::string& details)
    {
        bool verbose = Debug::Instance().printPredicateEvaluationVerbose();

        // In simplified mode, skip printing passing predicates
        if(pass && !verbose)
            return;

        // Ensure header is printed before first row
        flushHeader(stream);

        std::string passStr   = pass ? "[OK]" : "[!!]";
        std::string indentStr = std::string(indent() * INDENT_SIZE, ' ');

        stream << std::left << std::setw(COL_PASS) << passStr
               << indentStr << std::setw(std::max(0, COL_PREDICATE - static_cast<int>(indentStr.size())))
               << predicate
               << details << std::endl;
    }
};
}  // namespace TensileLite
