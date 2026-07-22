/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <string>

namespace TensileLite
{
    // A tuning-file entry: the solution index found during offline tuning,
    // plus (when available) the Tensile solution name it referred to at
    // tuning time. `index` is only a fast lookup hint; `name` authorizes it.
    // If the current solution at that index has a different name, the entry
    // is rejected instead of silently selecting the wrong kernel. `name` is
    // empty for legacy (pre-solution_name) override files.
    struct TunedSolution
    {
        int         index = -1;
        std::string name;
    };

    // Validate only the persisted identity. Runtime replay separately checks
    // whether the resolved solution supports the full current problem.
    inline bool isTunedSolutionIdentityValid(const TunedSolution& tuned,
                                             bool                 indexResolved,
                                             const std::string&   resolvedName,
                                             bool                 buildVersionCurrent)
    {
        if(!indexResolved)
            return false;
        if(!tuned.name.empty())
            return tuned.name == resolvedName;
        return buildVersionCurrent;
    }

    inline bool isTuningFileVersionCurrent(const std::string& firstLine,
                                           const std::string& currentVersion)
    {
        static const std::string header = "Git Version: ";
        const std::size_t        pos    = firstLine.find(header);
        return pos != std::string::npos
               && firstLine.substr(pos + header.length()) == currentVersion;
    }
} // namespace TensileLite
