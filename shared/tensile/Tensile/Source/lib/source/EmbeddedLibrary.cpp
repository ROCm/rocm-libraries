/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef TENSILE_DEFAULT_SERIALIZATION

#include <Tensile/EmbeddedLibrary.hpp>

#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedData.hpp>

namespace Tensile
{
    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        EmbeddedLibrary<MyProblem, MySolution>::NewLibrary(std::string const& key)
    {
        auto const& data = EmbeddedData<SolutionLibrary<MyProblem, MySolution>>::Get(key);
        if(data.size() == 0)
            return nullptr;
        if(data.size() > 1)
            throw std::runtime_error(concatenate("Expected one data item, found ", data.size()));

        return LoadLibraryData<MyProblem, MySolution>(data[0]);
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        EmbeddedLibrary<ContractionProblem, ContractionSolution>::NewLibrary(std::string const&);
} // namespace Tensile

#endif
