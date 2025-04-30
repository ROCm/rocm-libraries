// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <string_view>

#include <cstddef>

#include <stdint.h>

template<typename T>
void write_array(const std::string_view type_name,
                 const std::byte*       bytes,
                 const size_t           size_bytes,
                 const std::string_view symbol,
                 std::ostream&          os)
{
    const size_t size_elements = size_bytes / sizeof(T);

    rocrand_tools::write_preamble(os, "bin2typed")
        << "#include \"rocrand/rocrandapi.h\"\n"
        << "\n"
        << "// clang-format off\n"
        << "extern \"C\" ROCRANDAPI const " << type_name << " " << symbol << '[' << size_elements
        << "] =\n"
        << '{';

    const T* values = reinterpret_cast<const T*>(bytes);

    for(size_t k = 0; k < size_elements; k++)
    {
        constexpr size_t bits = 8 * sizeof(T);
        if(k % bits == 0)
        {
            os << "\n    ";
        }
        os << "0x" << std::hex << std::setw(sizeof(T) * 2) << std::setfill('0') << values[k]
           << ", ";
    }
    os << '\n' << "};\n";
}

bool write_as_array_of(const std::string_view type,
                       const std::byte*       bytes,
                       const size_t           size_bytes,
                       const std::string_view symbol,
                       std::ostream&          os)
{
    if(type == "unsigned int")
    {
        write_array<unsigned int>(type, bytes, size_bytes, symbol, os);
        return true;
    }
    if(type == "unsigned long long")
    {
        write_array<uint64_t>(type, bytes, size_bytes, symbol, os);
        return true;
    }
    std::cerr << "Unexpected type:" << type
              << " expected \"unsigned int\" or \"unsigned long long\"";
    return false;
}

int main(int argc, char const* argv[])
{
    using namespace std::string_view_literals;
    if(argc != 5 || argv[1] == "--help"sv)
    {
        std::cout
            << "Convert <binary file> to a C source file treating it as an array with the name "
               "<symbol> of unsigned ints or long longs"
            << " and write it to <output>\n"
            << "Usage:\n"
            << "  " << argv[0]
            << " <binary file> <symbol> \"unsigned int\"|\"unsigned long long\" <output>\n";
        return -1;
    }

    const std::filesystem::path binary_path(argv[1]);
    const std::string_view      symbol(argv[2]);
    const std::string_view      type(argv[3]);
    const std::filesystem::path output_path(argv[4]);

    std::ifstream input(binary_path, std::ios_base::in | std::ios_base::binary);
    std::ofstream output(output_path, std::ios_base::out | std::ios_base::trunc);

    const std::uintmax_t input_size  = std::filesystem::file_size(binary_path);
    auto                 input_bytes = std::make_unique<std::byte[]>(input_size);

    input.read(reinterpret_cast<char*>(input_bytes.get()), input_size);

    if(!write_as_array_of(type, input_bytes.get(), input_size, symbol, output))
    {
        return -1;
    }
    return 0;
}
