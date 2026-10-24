// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <random>
#include <stdint.h>
#include <type_traits>

namespace rocrand_tools
{

inline constexpr uint32_t SOBOL_DIM        = 20000;
inline constexpr uint32_t SOBOL32_N        = SOBOL_DIM * 32;
inline constexpr uint32_t SOBOL64_N        = SOBOL_DIM * 64;
inline constexpr uint32_t SOBOL32_LMS_SEED = std::mt19937::default_seed + 32u;
inline constexpr uint32_t SOBOL64_LMS_SEED = std::mt19937::default_seed + 64u;

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

/// Generates a uniform random number reproducible across standard libraries
template<class DirectionVectorType, class Engine>
DirectionVectorType next_uint(Engine& gen)
{
    static_assert(std::is_same_v<DirectionVectorType, uint32_t>
                  || std::is_same_v<DirectionVectorType, uint64_t>);
    if constexpr(std::is_same_v<DirectionVectorType, uint32_t>)
    {
        return gen();
    }
    else
    {
        return (static_cast<uint64_t>(gen()) << 32) | static_cast<uint64_t>(gen());
    }
}

/// Draws a uniformly random unit lower-triangular matrix over GF(2), in column form.
/// Column k has zeros in rows < k, a one in row k, and uniform random bits in rows > k.
template<typename DirectionVectorType, typename Engine>
void random_unit_lower_triangular(DirectionVectorType* cols, Engine& gen)
{
    constexpr int bits = sizeof(DirectionVectorType) * 8;
    for(int k = 0; k < bits; k++)
    {
        const DirectionVectorType diagonal = DirectionVectorType{1} << (bits - 1 - k);
        cols[k] = diagonal | (next_uint<DirectionVectorType>(gen) & (diagonal - 1));
    }
}

/// Multiplies a direction vector by a matrix given in column form, over GF(2).
template<typename DirectionVectorType>
DirectionVectorType apply_matrix(const DirectionVectorType* cols, DirectionVectorType v)
{
    constexpr int       bits = sizeof(DirectionVectorType) * 8;
    DirectionVectorType out  = 0;
    for(int k = 0; k < bits; k++)
    {
        if((v >> (bits - 1 - k)) & 1)
            out ^= cols[k];
    }
    return out;
}

/// Matousek linear matrix scramble: applies an independent random unit
/// lower-triangular matrix to each dimension's direction vectors, in place.
template<typename DirectionVectorType, typename Engine>
void linear_matrix_scramble(DirectionVectorType* directions, uint32_t dim, Engine& gen)
{
    constexpr int       bits = sizeof(DirectionVectorType) * 8;
    DirectionVectorType cols[bits];
    for(uint32_t d = 0; d < dim; d++)
    {
        random_unit_lower_triangular(cols, gen);
        DirectionVectorType* v = directions + static_cast<size_t>(d) * bits;
        for(int j = 0; j < bits; j++)
            v[j] = apply_matrix(cols, v[j]);
    }
}

} // namespace rocrand_tools

#endif
