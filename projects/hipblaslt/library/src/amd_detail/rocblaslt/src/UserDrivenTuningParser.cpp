/* ************************************************************************
 *
 * MIT License
 *
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
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "UserDrivenTuningParser.hpp"

#include <fstream>
#include <mutex>

#ifndef TO_STR2
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#endif

#ifndef HIPBLASLT_VERSION_TWEAK
#error "HIPBLASLT_VERSION_TWEAK (hipblaslt-version.h) decides which tuning file rows are trusted"
#endif

namespace TensileLite
{
    const std::string& currentBuildStamp()
    {
        static const std::string stamp = TO_STR(HIPBLASLT_VERSION_TWEAK);
        return stamp;
    }

    void getContractionProblemsFromFile(const std::string& path)
    {
        OverrideMap& map = OverrideMap::getMap();

        // Runs on every heuristic query. Once the file is loaded, the read lock
        // isLoaded takes is all a query needs.
        if(map.isLoaded(path))
            return;

        std::lock_guard<std::mutex> lock(map.getLock());
        if(map.isLoaded(path))
            return;

        // A file that does not open is looked for again on the next query.
        std::ifstream file(path);
        if(!file.is_open())
            return;

        const auto loaded = loadTuningRows(file, map, currentBuildStamp());

        if(loaded.skippedUnnamed > 0)
            log_error(__func__,
                      "Ignored " + std::to_string(loaded.skippedUnnamed)
                          + " entries without a kernel name in " + path
                          + ": its Git Version line does not match this build. Re-run the "
                            "tuning with this build to use them.");
        else if(currentBuildStamp().empty() || loaded.fileBuildStamp != currentBuildStamp())
            log_info(__func__,
                     path
                         + " has no Git Version line matching this build; each entry is used "
                           "only while its kernel_name still matches.");

        // Only a clean read counts as loaded. A read that stopped on an I/O
        // error partway through would otherwise leave a partial map that is
        // never completed.
        if(!loaded.readError)
            map.markLoaded(path);
    }
} // namespace TensileLite
