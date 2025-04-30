// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "utils_matrix_exponentiation.hpp"

#include <fstream>
#include <iostream>
#include <string>

const int XORWOW_N = 5;  // 5 values
const int XORWOW_M = 32; // 32-bit each

const int XORWOW_SIZE = XORWOW_M * XORWOW_N * XORWOW_N;

const int XORWOW_JUMP_MATRICES = 32;
const int XORWOW_JUMP_LOG2 = 2;

const int XORWOW_SEQUENCE_JUMP_LOG2 = 67;

static unsigned int jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE];
static unsigned int sequence_jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE];

struct rocrand_xorwow_state
{
    // Xorshift values (160 bits)
    unsigned int x[5];
    // Weyl sequence value
    unsigned int d;

    void discard()
    {
        const unsigned int t = x[0] ^ (x[0] >> 2);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = x[4];
        x[4] = (x[4] ^ (x[4] << 4)) ^ (t ^ (t << 1));

        d += 362437;
    }
};

void generate_matrices()
{
    unsigned int one_step[XORWOW_SIZE];

    for (int i = 0; i < XORWOW_N; i++)
    {
        for (int j = 0; j < XORWOW_M; j++)
        {
            rocrand_xorwow_state state;
            const unsigned int b = 1U << j;
            for (int k = 0; k < XORWOW_N; k++)
            {
                state.x[k] = (i == k ? b : 0);
            }
            state.d = 0;

            state.discard();

            for (int k = 0; k < XORWOW_N; k++)
            {
                one_step[(i * XORWOW_M + j) * XORWOW_N + k] = state.x[k];
            }
        }
    }

    {
        unsigned int a[XORWOW_SIZE];
        unsigned int b[XORWOW_SIZE];
        copy_arr<XORWOW_SIZE>(a, one_step);

        copy_arr<XORWOW_SIZE>(jump_matrices[0], a);
        for (int k = 1; k < XORWOW_JUMP_MATRICES; k++)
        {
            copy_arr<XORWOW_SIZE>(b, a);
            mat_pow<XORWOW_SIZE, XORWOW_N, XORWOW_M>(a, b, (1 << XORWOW_JUMP_LOG2));
            copy_arr<XORWOW_SIZE>(jump_matrices[k], a);
        }
    }

    {
        unsigned int a[XORWOW_SIZE];
        unsigned int b[XORWOW_SIZE];
        copy_arr<XORWOW_SIZE>(a, one_step);

        // For 67: A^(2^33)
        mat_pow<XORWOW_SIZE, XORWOW_N, XORWOW_M>(b, a, 1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 / 2));
        // For 67: (A^(2^33))^(2^34) = A^(2^67)
        mat_pow<XORWOW_SIZE, XORWOW_N, XORWOW_M>(
            a,
            b,
            1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 - XORWOW_SEQUENCE_JUMP_LOG2 / 2));

        copy_arr<XORWOW_SIZE>(sequence_jump_matrices[0], a);
        for (int k = 1; k < XORWOW_JUMP_MATRICES; k++)
        {
            copy_arr<XORWOW_SIZE>(b, a);
            mat_pow<XORWOW_SIZE, XORWOW_N, XORWOW_M>(a, b, (1 << XORWOW_JUMP_LOG2));
            copy_arr<XORWOW_SIZE>(sequence_jump_matrices[k], a);
        }
    }
}

int main(int argc, char const *argv[]) {
    if (argc != 2 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./xorwow_precomputed_generator ../../library/include/rocrand/rocrand_xorwow_precomputed.h" << std::endl;
        return -1;
    }

    generate_matrices();

    const std::string file_path(argv[1]);
    std::ofstream fout(file_path, std::ios_base::out | std::ios_base::trunc);
    fout << R"(// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_XORWOW_PRECOMPUTED_H_
#define ROCRAND_XORWOW_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/xorwow_precomputed_generator

)";

    fout << "#define XORWOW_N " << XORWOW_N << std::endl;
    fout << "#define XORWOW_M " << XORWOW_M << std::endl;
    fout << "#define XORWOW_SIZE (XORWOW_M * XORWOW_N * XORWOW_N)" << std::endl;
    fout << "#define XORWOW_JUMP_MATRICES " << XORWOW_JUMP_MATRICES << std::endl;
    fout << "#define XORWOW_JUMP_LOG2 " << XORWOW_JUMP_LOG2 << std::endl;
    fout << std::endl;

    write_matrices<XORWOW_JUMP_MATRICES, XORWOW_SIZE, XORWOW_N, XORWOW_M>(
        fout,
        "d_xorwow_jump_matrices",
        "xorwow",
        static_cast<unsigned int*>(&jump_matrices[0][0]),
        true);
    write_matrices<XORWOW_JUMP_MATRICES, XORWOW_SIZE, XORWOW_N, XORWOW_M>(
        fout,
        "h_xorwow_jump_matrices",
        "xorwow",
        static_cast<unsigned int*>(&jump_matrices[0][0]),
        false);

    write_matrices<XORWOW_JUMP_MATRICES, XORWOW_SIZE, XORWOW_N, XORWOW_M>(
        fout,
        "d_xorwow_sequence_jump_matrices",
        "xorwow",
        static_cast<unsigned int*>(&sequence_jump_matrices[0][0]),
        true);
    write_matrices<XORWOW_JUMP_MATRICES, XORWOW_SIZE, XORWOW_N, XORWOW_M>(
        fout,
        "h_xorwow_sequence_jump_matrices",
        "xorwow",
        static_cast<unsigned int*>(&sequence_jump_matrices[0][0]),
        false);

    fout << R"(
#endif // ROCRAND_XORWOW_PRECOMPUTED_H_
)";

    return 0;
}
