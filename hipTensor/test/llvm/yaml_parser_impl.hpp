/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_TEST_YAML_PARSER_IMPL_HPP
#define HIPTENSOR_TEST_YAML_PARSER_IMPL_HPP

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/raw_ostream.h>

#include "yaml_parser.hpp"

namespace hiptensor
{
    template <typename ConfigT>
    /* static */
    std::optional<ConfigT> YamlConfigLoader<ConfigT>::loadFromFile(std::string const& filePath)
    {
        std::ifstream inputStream(filePath);
        if(!inputStream)
        {
            llvm::errs() << "Cannot open file for reading: " << filePath << "\n";
            return {};
        }
        else
        {
            llvm::outs() << "Opened file for reading: " << filePath << "\n";
        }

        std::string yamlString((std::istreambuf_iterator<char>(inputStream)),
                               std::istreambuf_iterator<char>());

        return loadFromString(yamlString);
    }

    template <typename ConfigT>
    /* static */
    std::optional<ConfigT>
        YamlConfigLoader<ConfigT>::loadFromString(std::string const& yaml /*= ""*/)
    {
        auto result = ConfigT{};

        auto in = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(yaml.c_str()));
        if(in->getBufferSize() == 0)
        {
            llvm::errs() << "Cannot use empty string for MemoryBuffer\n";
            return {};
        }

        llvm::yaml::Input reader(*in);

        reader >> result;

        if(reader.error())
        {
            llvm::errs() << "Error reading input config: " << yaml << "\n";
            return {};
        }

        return result;
    }

    template <typename ConfigT>
    /* static */
    void YamlConfigLoader<ConfigT>::storeToFile(std::string const& filePath, ConfigT const& config)
    {
        std::error_code      ec;
        llvm::raw_fd_ostream out(filePath, ec);

        if(ec)
        {
            llvm::errs() << "Cannot open file for writing: " << filePath << "\n";
            llvm::errs() << "Error: " << ec.message() << "\n";
            out.close();
        }
        else
        {
            llvm::outs() << "Opened file for writing: " << filePath << "\n";
        }

        llvm::yaml::Output writer(out);
        writer << const_cast<ConfigT&>(config);
        out.close();
    }
}

#endif // HIPTENSOR_TEST_YAML_PARSER_IMPL_HPP
