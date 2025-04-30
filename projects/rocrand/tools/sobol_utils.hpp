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

#ifndef ROCRAND_TOOLS_SOBOL_UTILS_HPP_
#define ROCRAND_TOOLS_SOBOL_UTILS_HPP_

#if __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
namespace std
{
namespace filesystem = experimental::filesystem;
} // namespace std
#endif

#include <fstream>
#include <iostream>

#include <stdint.h>

namespace rocrand_tools
{

inline constexpr uint32_t SOBOL_DIM = 20000;
inline constexpr uint32_t SOBOL32_N = SOBOL_DIM * 32;
inline constexpr uint32_t SOBOL64_N = SOBOL_DIM * 64;

struct sobol_set
{
    uint32_t d;
    uint32_t s;
    uint32_t a;
    uint32_t m[18];
};

inline bool read_sobol_set(struct sobol_set* inputs, int N, const std::filesystem::path name)
{
    std::ifstream infile(name, std::ios::in);
    if(!infile)
    {
        std::cout << "Input file containing direction numbers cannot be found!\n";
        return false;
    }
    char buffer[1000];
    infile.getline(buffer, 1000, '\n');

    for(int32_t i = 0; i < N; i++)
    {
        infile >> inputs[i].d >> inputs[i].s >> inputs[i].a;
        for(uint32_t j = 0; j < inputs[i].s; j++)
            infile >> inputs[i].m[j];
    }

    return true;
}

template<typename DirectionVectorType>
void init_direction_vectors(struct rocrand_tools::sobol_set* inputs,
                            DirectionVectorType*             directions,
                            int                              n_directions,
                            int                              n)
{
    constexpr uint32_t shift_adjust = (sizeof(DirectionVectorType) * 8) - 1;
    for(int j = 0; j < n_directions; j++)
    {
        directions[j] = 1lu << (shift_adjust - j);
    }
    directions += n_directions;
    for(int i = 1; i < n; i++)
    {
        int ix = i - 1;
        int s  = inputs[ix].s;
        for(int j = 0; j < s; j++)
        {
            directions[j] = (DirectionVectorType)inputs[ix].m[j] << (shift_adjust - j);
        }
        for(int j = s; j < n_directions; j++)
        {
            directions[j] = directions[j - s] ^ (directions[j - s] >> s);
            for(int k = 1; k < s; k++)
            {
                directions[j] ^= ((((DirectionVectorType)inputs[ix].a >> (s - 1 - k)) & 1)
                                  * directions[j - k]);
            }
        }
        directions += n_directions;
    }
}

template<typename DirectionVectorType>
void write_matrix_with_offset(std::ofstream&             binary_out,
                              const DirectionVectorType* directions,
                              int32_t                    n,
                              uint32_t                   offset)
{
    for(int k = 0; k < n; k++)
    {
        const DirectionVectorType direction = directions[k] + offset;
        binary_out.write(reinterpret_cast<const char*>(&direction), sizeof(direction));
    }
}

} // namespace rocrand_tools

#endif
