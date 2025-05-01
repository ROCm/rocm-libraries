/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#ifdef TENSILE_DEFAULT_SERIALIZATION
#ifdef TENSILE_YAML
#include <Tensile/llvm/Loading.hpp>
#endif

#ifdef TENSILE_MSGPACK
#include <Tensile/msgpack/Loading.hpp>
#endif
#endif

namespace TensileLite
{

    TENSILE_API Problem::~Problem()                 = default;
    TENSILE_API Hardware::Hardware()                = default;
    TENSILE_API Hardware::~Hardware()               = default;
    TENSILE_API Solution::~Solution()               = default;
    TENSILE_API SolutionAdapter::~SolutionAdapter() = default;

#ifdef TENSILE_DEFAULT_SERIALIZATION
    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LoadLibraryFile(std::string const& filename)
    {
        return LoadLibraryFilePreload<MyProblem, MySolution>(filename, {});
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LoadLibraryFilePreload(std::string const&                  filename,
                               const std::vector<LazyLoadingInit>& preloaded)
    {
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> rv;

#ifdef TENSILE_MSGPACK
        rv = MessagePackLoadLibraryFile<MyProblem, MySolution>(filename, preloaded);
        if(rv)
            return rv;
#endif

#ifdef TENSILE_YAML
        rv = LLVMLoadLibraryFile<MyProblem, MySolution>(filename, preloaded);
        if(rv)
            return rv;
#endif

        // Failed to load library, return nullptr.
        return nullptr;
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LoadLibraryData(std::vector<uint8_t> const& data)
    {
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> rv;

#ifdef TENSILE_MSGPACK
        rv = MessagePackLoadLibraryData<MyProblem, MySolution>(data);
        if(rv)
            return rv;
#endif

#ifdef TENSILE_YAML
        rv = LLVMLoadLibraryData<MyProblem, MySolution>(data);
        if(rv)
            return rv;
#endif

        // Failed to load library, return nullptr.
        return nullptr;
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(std::string const& filename);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryFilePreload<ContractionProblemGemm, ContractionSolution>(
            std::string const& filename, const std::vector<LazyLoadingInit>& preloaded);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryData<ContractionProblemGemm, ContractionSolution>(
            std::vector<uint8_t> const& data);
#endif
} // namespace TensileLite
