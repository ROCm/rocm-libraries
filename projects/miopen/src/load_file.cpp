// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/load_file.hpp>

#include <fstream>
#include <ios>
#include <iterator>
#include <vector>
#include <sstream>

namespace miopen {

std::vector<char> LoadFile(const fs::path& path)
{
    std::error_code error_code;
    const auto size = fs::file_size(path, error_code);
    if(error_code)
        MIOPEN_THROW(path.string() + ": " + error_code.message());
    std::ifstream in(path, std::ios::binary);
    if(!in.is_open())
        MIOPEN_THROW(path.string() + ": file opening error");
    std::vector<char> v(size);
    if(in.read(v.data(), v.size()).fail())
        MIOPEN_THROW(path.string() + ": file reading error");
    return v;
}

} // namespace miopen
