/* ************************************************************************
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
 * ************************************************************************ */
#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using IsaVersion = std::array<int, 3>;

std::pair<int, std::string>
            run(const std::vector<char*>& cmd, const std::string& input, bool debug = false);
std::string demangle(const char* name);

inline std::string getGfxNameTuple(const IsaVersion& isaVersion)
{
    /*Converts an ISA version to a gfx architecture name.

    Args:
        arch: An object representing the major, minor, and step version of the ISA.

    Returns:
        The name of the GPU architecture (e.g., 'gfx906').
    */
    const char int_to_hex[] = "0123456789abcdef";
    return std::string("gfx") + std::to_string(isaVersion[0]) + std::to_string(isaVersion[1])
           + int_to_hex[isaVersion[2]];
}

template <typename T>
bool checkInList(const T& a, const std::vector<T> b)
{
    return std::find(b.begin(), b.end(), a) != b.end();
}

template <typename T>
bool checkNotInList(const T& a, const std::vector<T> b)
{
    return std::find(b.begin(), b.end(), a) == b.end();
}
