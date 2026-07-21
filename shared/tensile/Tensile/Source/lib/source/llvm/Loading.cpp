/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/llvm/Loading.hpp>

#include <fstream>

#include <Tensile/Debug.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/llvm/YAML.hpp>

namespace Tensile
{

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LLVMLoadLibraryFile(std::string const&                  filename,
                            const std::vector<LazyLoadingInit>& preloaded)
    {
        std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

        try
        {
            auto inputFile = llvm::MemoryBuffer::getFileAsStream(filename);

            LibraryIOContext<MySolution> context{filename, preloaded, nullptr};
            llvm::yaml::Input            yin((*inputFile)->getMemBufferRef(), &context);

            yin >> rv;

            if(yin.error())
            {
                return nullptr;
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading " << filename << " (YAML):" << std::endl
                          << exc.what() << std::endl;

            return nullptr;
        }

        return rv;
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        LLVMLoadLibraryData(std::vector<uint8_t> const& data, std::string filename)
    {
        std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

        try
        {
            LibraryIOContext<MySolution> context{filename, {}, nullptr};
            llvm::StringRef              dataRef((const char*)data.data(), data.size());
            llvm::yaml::Input            yin(dataRef, &context);

            yin >> rv;

            if(yin.error())
            {
                throw std::runtime_error(yin.error().message());
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading YAML data:" << std::endl << exc.what() << std::endl;

            return nullptr;
        }

        return rv;
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        LLVMLoadLibraryFile<ContractionProblem, ContractionSolution>(
            std::string const& filename, const std::vector<LazyLoadingInit>& preloaded);

    template std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>
        LLVMLoadLibraryData<ContractionProblem, ContractionSolution>(
            std::vector<uint8_t> const& data, std::string filename = "");
} // namespace Tensile
