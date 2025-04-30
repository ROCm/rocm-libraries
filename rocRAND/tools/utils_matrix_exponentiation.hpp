// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_TOOLS_UTILS_MATRIX_EXPONENTIATION_HPP_
#define ROCRAND_TOOLS_UTILS_MATRIX_EXPONENTIATION_HPP_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

/// @brief Copies array \p src of size \p SIZE into \p src.
template<int SIZE>
void copy_arr(unsigned int* dst, const unsigned int* src)
{
    for(int i = 0; i < SIZE; i++)
    {
        dst[i] = src[i];
    }
}

/// @brief Given an NxMxN matrix \p m and a vector \p v of size N, it performs an exclusive OR
/// among the N-sized vectors m[i][j] for which it holds that v[i] & (1U << j) != 0
/// and stores the result into \p v.
template<int N, int M>
void mul_mat_vec_inplace(const unsigned int* m, unsigned int* v)
{
    unsigned int r[N] = {0};
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            if(v[i] & (1U << j))
            {
                for(int k = 0; k < N; k++)
                {
                    r[k] ^= m[N * (i * M + j) + k];
                }
            }
        }
    }
    copy_arr<N>(v, r);
}

/// @brief Multiplies NxMxN matrices \p b and \p a and stores the result in \p a.
template<int N, int M>
void mul_mat_mat_inplace(unsigned int* a, const unsigned int* b)
{
    for(int i = 0; i < N * M; i++)
    {
        mul_mat_vec_inplace<N, M>(b, a + i * N);
    }
}

/// @brief Computes exponentiation of matrix \p b to \p power by squaring and stores result into \p a.
template<int SIZE, int N, int M>
void mat_pow(unsigned int* a, const unsigned int* b, const unsigned long long power)
{
    // Identity matrix
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            for(int k = 0; k < N; k++)
            {
                a[(i * M + j) * N + k] = ((i == k) ? (1 << j) : 0);
            }
        }
    }

    // Exponentiation by squaring
    unsigned int y[SIZE];
    copy_arr<SIZE>(y, b);
    for(unsigned long long p = power; p > 0; p >>= 1)
    {
        if(p & 1)
        {
            mul_mat_mat_inplace<N, M>(a, y);
        }

        // Square the matrix
        unsigned int t[SIZE];
        copy_arr<SIZE>(t, y);
        mul_mat_mat_inplace<N, M>(y, t);
    }
}

/// @brief Writes the C++ code of the declaration (with variable name \p name) and initialization
/// of a precomputed state matrix \p a for a generator named \p generator to a given output
/// stream \p fout.
template<int JUMP_MATRICES, int SIZE, int N, int M>
void write_matrices(std::ofstream&    fout,
                    const std::string name,
                    std::string       generator,
                    unsigned int*     a,
                    bool              is_device)
{
    std::transform(generator.begin(), generator.end(), generator.begin(), ::toupper);
    fout << "// clang-format off" << std::endl;
    fout << "static const " << (is_device ? "__device__ " : "") << "unsigned int " << name << "["
         << generator << "_JUMP_MATRICES][" << generator << "_SIZE] = {" << std::endl;
    for(int k = 0; k < JUMP_MATRICES; k++)
    {
        fout << "    {" << std::endl;
        for(int i = 0; i < M; i++)
        {
            fout << "        ";
            for(int j = 0; j < N * N; j++)
            {
                fout << a[k * SIZE + i * N * N + j] << ", ";
            }
            fout << std::endl;
        }
        fout << "    }," << std::endl;
    }
    fout << "};" << std::endl;
    fout << std::endl;
    fout << "// clang-format on" << std::endl;
}

#endif // ROCRAND_TOOLS_UTILS_MATRIX_EXPONENTIATION_HPP_
