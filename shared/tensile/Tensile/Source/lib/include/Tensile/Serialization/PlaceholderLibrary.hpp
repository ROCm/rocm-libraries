/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
//Replace std::regex, as it crashes when matching long lines(GCC Bug #86164).
#ifdef WIN32
#include "shlwapi.h"
#else
#include <fnmatch.h>
#endif

namespace Tensile
{
    namespace Serialization
    {
        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<PlaceholderLibrary<MyProblem, MySolution>, IO>
        {
            using Library = PlaceholderLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                iot::mapRequired(io, "value", lib.filePrefix);

                if(!iot::outputting(io))
                {
                    auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                    lib.masterSolutions = ctx->solutions;
                    lib.solutionsGuard  = ctx->solutionsGuard;

                    //Extract directory where TensileLibrary.dat/yaml file is located
                    lib.libraryDirectory = ctx->filename;
                    size_t directoryPos  = ctx->filename.rfind('/');
                    if(directoryPos != std::string::npos)
                        lib.libraryDirectory.resize(directoryPos + 1);
                    else
                        lib.libraryDirectory = '.';

                    //Extract file extension
                    size_t periodPos = ctx->filename.rfind('.');
                    lib.suffix       = ctx->filename.substr(periodPos);

                    for(auto condition : ctx->preloaded)
                    {
                        std::string pattern = RegexPattern(condition);
#ifdef WIN32
                        if(PathMatchSpecA(lib.filePrefix.c_str(), pattern.c_str()))
#else
                        if(fnmatch(pattern.c_str(), lib.filePrefix.c_str(), 0) == 0)
#endif
                        {
                            lib.loadPlaceholderLibrary();
                            break;
                        }
                    }
                }
            }

            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace Tensile
