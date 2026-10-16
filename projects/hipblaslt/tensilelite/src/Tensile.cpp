/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "LibraryLoadError.hpp"

#ifdef TENSILE_YAML
#include <Tensile/llvm/Loading.hpp>
#endif

#ifdef TENSILE_MSGPACK
#include <Tensile/msgpack/Loading.hpp>
#endif

namespace TensileLite
{
    namespace
    {
        thread_local std::string g_lastLibraryLoadError;

        bool hasSuffix(std::string const& path, char const* suffix)
        {
            size_t const length = std::char_traits<char>::length(suffix);
            return path.size() >= length
                   && path.compare(path.size() - length, length, suffix) == 0;
        }

        // YAML and msgpack builds publish one SONAME. A file written for the
        // other parser reaches this function when that parser is absent.
        std::string explainLibraryLoadFailure(std::string const& filename)
        {
#if defined(TENSILE_YAML)
            char const* const parser = "YAML";
#elif defined(TENSILE_MSGPACK)
            char const* const parser = "msgpack";
#else
            char const* const parser = "no library parser";
#endif
#if !defined(TENSILE_YAML)
            if(hasSuffix(filename, ".yaml") || hasSuffix(filename, ".yml"))
            {
                return "file '" + filename + "' is YAML. This libtensilelite-host was built with the "
                       + parser + " parser only.";
            }
#endif
#if !defined(TENSILE_MSGPACK)
            if(hasSuffix(filename, ".dat") || hasSuffix(filename, ".msgpack"))
            {
                return "file '" + filename
                       + "' is msgpack. This libtensilelite-host was built with the " + parser
                       + " parser only.";
            }
#endif
            return "file '" + filename + "' was rejected by the " + parser
                   + " parser in this libtensilelite-host. The file's format_version may be newer "
                     "than this build, or its contents may not match that parser.";
        }
    }

    void NoteLibraryLoadError(std::string message)
    {
        g_lastLibraryLoadError = std::move(message);
    }

    void ClearLibraryLoadError()
    {
        g_lastLibraryLoadError.clear();
    }

    std::string LastLibraryLoadError()
    {
        return g_lastLibraryLoadError;
    }

    TENSILELITEHOST_EXPORT Problem::~Problem()                 = default;
    TENSILELITEHOST_EXPORT Hardware::Hardware()                = default;
    TENSILELITEHOST_EXPORT Hardware::~Hardware()               = default;
    TENSILELITEHOST_EXPORT Solution::~Solution()               = default;
    TENSILELITEHOST_EXPORT SolutionAdapter::~SolutionAdapter() = default;

#ifdef TENSILE_MSGPACK
    std::map<int, std::string> LoadLibraryMapping(std::string const& filename)
    {
        return MessagePackLoadLibraryMapping(filename);
    }
#endif

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
        ClearLibraryLoadError();

        // Name the parser mismatch before a parser turns the foreign file into
        // a generic read error such as "Invalid argument".
#if !defined(TENSILE_YAML)
        if(hasSuffix(filename, ".yaml") || hasSuffix(filename, ".yml"))
        {
            NoteLibraryLoadError(explainLibraryLoadFailure(filename));
            return nullptr;
        }
#endif
#if !defined(TENSILE_MSGPACK)
        if(hasSuffix(filename, ".dat") || hasSuffix(filename, ".msgpack"))
        {
            NoteLibraryLoadError(explainLibraryLoadFailure(filename));
            return nullptr;
        }
#endif

#ifdef TENSILE_MSGPACK
        rv = MessagePackLoadLibraryFile<MyProblem, MySolution>(filename, preloaded);
        if(rv)
        {
            ClearLibraryLoadError();
            return rv;
        }
#endif

#ifdef TENSILE_YAML
        rv = LLVMLoadLibraryFile<MyProblem, MySolution>(filename, preloaded);
        if(rv)
        {
            ClearLibraryLoadError();
            return rv;
        }
#endif

        if(LastLibraryLoadError().empty())
            NoteLibraryLoadError(explainLibraryLoadFailure(filename));
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

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(default)
#endif
    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(std::string const& filename);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryFilePreload<ContractionProblemGemm, ContractionSolution>(
            std::string const& filename, const std::vector<LazyLoadingInit>& preloaded);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        LoadLibraryData<ContractionProblemGemm, ContractionSolution>(
            std::vector<uint8_t> const& data);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
} // namespace TensileLite
