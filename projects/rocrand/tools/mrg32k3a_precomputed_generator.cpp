// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstring>

using namespace std;

#define ROCRAND_MRG32K3A_POW32 4294967296
#define ROCRAND_MRG32K3A_M1 4294967087
#define ROCRAND_MRG32K3A_M1C 209
#define ROCRAND_MRG32K3A_M2 4294944443
#define ROCRAND_MRG32K3A_M2C 22853
#define ROCRAND_MRG32K3A_A12 1403580
#define ROCRAND_MRG32K3A_A13 (4294967087 -  810728)
#define ROCRAND_MRG32K3A_A21 527612
#define ROCRAND_MRG32K3A_A23 (4294944443 - 1370589)

unsigned long long A1_[9] =
{
    0,                    1,   0,
    0,                    0,   1,
    ROCRAND_MRG32K3A_A13, ROCRAND_MRG32K3A_A12, 0
};

unsigned long long A2_[9] =
{
    0,                    1, 0,
    0,                    0, 1,
    ROCRAND_MRG32K3A_A23, 0, ROCRAND_MRG32K3A_A21
};

unsigned long long A1p76[9] = {82758667,
                               1871391091,
                               4127413238,
                               3672831523,
                               69195019,
                               1871391091,
                               3672091415,
                               3528743235,
                               69195019};

unsigned long long A2p76[9] = {1511326704,
                               3759209742,
                               1610795712,
                               4292754251,
                               1511326704,
                               3889917532,
                               3859662829,
                               4292754251,
                               3708466080};

unsigned long long A1p127[9] =
{
    2427906178, 3580155704, 949770784,
    226153695,  1230515664, 3580155704,
    1988835001, 986791581,  1230515664
};

unsigned long long A2p127[9] =
{
    1464411153, 277697599,  1610723613,
    32183930,   1464411153, 1022607788,
    2824425944, 32183930,   2093834863
};

void mod_mat_sq(unsigned long long * A,
                unsigned long long m)
{
    unsigned long long x[9];
    unsigned long long a;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            a = 0;
            for (size_t k = 0; k < 3; k++) {
                a += (A[i + 3 * k] * A[k + 3 * j]) % m;
            }
            x[i + 3 * j] = a % m;
        }
    }
    for (size_t i = 0; i < 3; i++) {
        A[i + 3 * 0] = x[i + 3 * 0];
        A[i + 3 * 1] = x[i + 3 * 1];
        A[i + 3 * 2] = x[i + 3 * 2];
    }
}


void init_matrices(unsigned long long * matrix, unsigned long long * A, int n, unsigned long long m)
{
    unsigned long long x[9];
    for (int i = 0; i < 9; i++)
        x[i] = A[i];

    for (int i = 0 ; i < n ; i++) {
        if (i > 0) {
            mod_mat_sq(x, m);
        }
        for (int j = 0; j < 9; j++)
            matrix[j + (i * 9)] = x[j];
    }
}

void write_matrices(std::ofstream& fout, const std::string name, unsigned long long * a, int n, int bits, bool is_device)
{
    fout << "static const ";
    fout << (is_device ? "__device__ " : "") << "unsigned long long " << name << "[MRG323A_N] = {"
         << std::endl;
    fout << "    // clang-format off" << std::endl;
    fout << "    ";
    for (int k = 0; k < n; k++)
    {
        fout << a[k] << ", ";
        if ((k + 1) % bits == 0 && k != 1)
            fout << std::endl << "    ";
    }
    fout << "// clang-format on" << std::endl;
    fout << "};" << std::endl;
    fout << std::endl;
}

int main(int argc, char const *argv[])
{
    if (argc != 2 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./mrg32k3a_precomputed_generator ../../library/include/rocrand/rocrand_mrg32k3a_precomputed.h" << std::endl;
        return -1;
    }

    unsigned int MRG323A_DIM = 64;
    unsigned int MRG323A_N = MRG323A_DIM * 9;
    unsigned long long * A1 = new unsigned long long[MRG323A_N];
    unsigned long long * A2 = new unsigned long long[MRG323A_N];
    unsigned long long*  A1P76       = new unsigned long long[MRG323A_N];
    unsigned long long*  A2P76       = new unsigned long long[MRG323A_N];
    unsigned long long * A1P127 = new unsigned long long[MRG323A_N];
    unsigned long long * A2P127 = new unsigned long long[MRG323A_N];

    init_matrices(A1, A1_, MRG323A_DIM, ROCRAND_MRG32K3A_M1);
    init_matrices(A2, A2_, MRG323A_DIM, ROCRAND_MRG32K3A_M2);
    init_matrices(A1P76, A1p76, MRG323A_DIM, ROCRAND_MRG32K3A_M1);
    init_matrices(A2P76, A2p76, MRG323A_DIM, ROCRAND_MRG32K3A_M2);
    init_matrices(A1P127, A1p127, MRG323A_DIM, ROCRAND_MRG32K3A_M1);
    init_matrices(A2P127, A2p127, MRG323A_DIM, ROCRAND_MRG32K3A_M2);
    const std::string file_path(argv[1]);
    std::ofstream fout(file_path, std::ios_base::out | std::ios_base::trunc);
    fout << R"(// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_MRG32K3A_PRECOMPUTED_H_
#define ROCRAND_MRG32K3A_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/mrg32k3a_precomputed_generator

)";

    fout << "#define MRG323A_DIM " << MRG323A_DIM << std::endl;
    fout << "#define MRG323A_N " << MRG323A_N << std::endl;
    fout << std::endl;

    write_matrices(fout, "d_A1", A1, MRG323A_N, 9, true);
    write_matrices(fout, "h_A1", A1, MRG323A_N, 9, false);
    write_matrices(fout, "d_A2", A2, MRG323A_N, 9, true);
    write_matrices(fout, "h_A2", A2, MRG323A_N, 9, false);
    write_matrices(fout, "d_A1P76", A1P76, MRG323A_N, 9, true);
    write_matrices(fout, "h_A1P76", A1P76, MRG323A_N, 9, false);
    write_matrices(fout, "d_A2P76", A2P76, MRG323A_N, 9, true);
    write_matrices(fout, "h_A2P76", A2P76, MRG323A_N, 9, false);
    write_matrices(fout, "d_A1P127", A1P127, MRG323A_N, 9, true);
    write_matrices(fout, "h_A1P127", A1P127, MRG323A_N, 9, false);
    write_matrices(fout, "d_A2P127", A2P127, MRG323A_N, 9, true);
    write_matrices(fout, "h_A2P127", A2P127, MRG323A_N, 9, false);

    fout << R"(
#endif // ROCRAND_MRG32K3A_PRECOMPUTED_H_
)";


    delete[] A1;
    delete[] A2;
    delete[] A1P76;
    delete[] A2P76;
    delete[] A1P127;
    delete[] A2P127;

    return 0;
}
