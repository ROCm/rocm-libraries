// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

#define ROCRAND_MRG31K3P_M1 2147483647U // 2 ^ 31 - 1
#define ROCRAND_MRG31K3P_M2 2147462579U // 2 ^ 31 - 21069
#define ROCRAND_MRG31K3P_A12 4194304U // 2 ^ 22
#define ROCRAND_MRG31K3P_A13 129U // 2 ^ 7 + 1
#define ROCRAND_MRG31K3P_A21 32768U // 2 ^ 15
#define ROCRAND_MRG31K3P_A23 32769U // 2 ^ 15 + 1

// Modulo multiplication with this matrix progresses g1 by one.
unsigned int A1[9] = {0, ROCRAND_MRG31K3P_A12, ROCRAND_MRG31K3P_A13, 1, 0, 0, 0, 1, 0};

// Modulo multiplication with this matrix progresses g2 by one.
unsigned int A2[9] = {ROCRAND_MRG31K3P_A21, 0, ROCRAND_MRG31K3P_A23, 1, 0, 0, 0, 1, 0};

// A1^(2^72)
unsigned int A1p72[9] = {1516919229,
                         758510237,
                         499121365,
                         1884998244,
                         1516919229,
                         335398200,
                         601897748,
                         1884998244,
                         358115744};

// A2^(2^72)
unsigned int A2p72[9] = {1228857673,
                         1496414766,
                         954677935,
                         1133297478,
                         1407477216,
                         1496414766,
                         2002613992,
                         1639496704,
                         1407477216};

// A1^(2^134)
unsigned int A1p134[9] = {1702500920,
                          1849582496,
                          1656874625,
                          828554832,
                          1702500920,
                          1512419905,
                          1143731069,
                          828554832,
                          102237247};

// A2^(2^134)
unsigned int A2p134[9] = {796789021,
                          1464208080,
                          607337906,
                          1241679051,
                          1431130166,
                          1464208080,
                          1401213391,
                          1178684362,
                          1431130166};

void mod_mat_sq(unsigned int* A, unsigned int m)
{
    unsigned int x[9];
    for(size_t i = 0; i < 3; i++)
    {
        for(size_t j = 0; j < 3; j++)
        {
            unsigned long long a = 0;
            for(size_t k = 0; k < 3; k++)
            {
                unsigned long long aik = A[i + 3 * k];
                unsigned long long akj = A[k + 3 * j];
                a += (aik * akj) % m;
            }
            x[i + 3 * j] = static_cast<unsigned int>(a % m);
        }
    }
    for(size_t i = 0; i < 3; i++)
    {
        A[i + 3 * 0] = x[i + 3 * 0];
        A[i + 3 * 1] = x[i + 3 * 1];
        A[i + 3 * 2] = x[i + 3 * 2];
    }
}

void init_matrices(unsigned int* matrix, unsigned int* A, int n, unsigned int m)
{
    unsigned int x[9];
    for(int i = 0; i < 9; i++)
        x[i] = A[i];

    for(int i = 0; i < n; i++)
    {
        if(i > 0)
        {
            mod_mat_sq(x, m);
        }
        for(int j = 0; j < 9; j++)
            matrix[j + (i * 9)] = x[j];
    }
}

void write_matrices(
    std::ofstream& fout, const std::string name, unsigned int* a, int n, int bits, bool is_device)
{
    fout << "static const ";
    fout << (is_device ? "__device__ " : "") << "unsigned int " << name << "[MRG31K3P_N] = {"
         << std::endl;
    fout << "    // clang-format off" << std::endl;
    fout << "    ";
    for(int k = 0; k < n; k++)
    {
        fout << a[k] << ", ";
        if((k + 1) % bits == 0 && k != 1)
            fout << std::endl << "    ";
    }
    fout << "// clang-format on" << std::endl;
    fout << "};" << std::endl;
    fout << std::endl;
}

int main(int argc, char const* argv[])
{
    if(argc != 2 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./mrg31k3p_precomputed_generator "
                     "../../library/include/rocrand/rocrand_mrg31k3p_precomputed.h"
                  << std::endl;
        return -1;
    }

    unsigned int  MRG31K3P_DIM = 64;
    unsigned int  MRG31K3P_N   = MRG31K3P_DIM * 9;
    unsigned int* A1_          = new unsigned int[MRG31K3P_N];
    unsigned int* A2_          = new unsigned int[MRG31K3P_N];
    unsigned int* A1p72_       = new unsigned int[MRG31K3P_N];
    unsigned int* A2p72_       = new unsigned int[MRG31K3P_N];
    unsigned int* A1p134_      = new unsigned int[MRG31K3P_N];
    unsigned int* A2p134_      = new unsigned int[MRG31K3P_N];

    init_matrices(A1_, A1, MRG31K3P_DIM, ROCRAND_MRG31K3P_M1);
    init_matrices(A2_, A2, MRG31K3P_DIM, ROCRAND_MRG31K3P_M2);
    init_matrices(A1p72_, A1p72, MRG31K3P_DIM, ROCRAND_MRG31K3P_M1);
    init_matrices(A2p72_, A2p72, MRG31K3P_DIM, ROCRAND_MRG31K3P_M2);
    init_matrices(A1p134_, A1p134, MRG31K3P_DIM, ROCRAND_MRG31K3P_M1);
    init_matrices(A2p134_, A2p134, MRG31K3P_DIM, ROCRAND_MRG31K3P_M2);
    const std::string file_path(argv[1]);
    std::ofstream     fout(file_path, std::ios_base::out | std::ios_base::trunc);
    fout << R"(// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_MRG31K3P_PRECOMPUTED_H_
#define ROCRAND_MRG31K3P_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/mrg31k3p_precomputed_generator

)";

    fout << "#define MRG31K3P_DIM " << MRG31K3P_DIM << std::endl;
    fout << "#define MRG31K3P_N " << MRG31K3P_N << std::endl;
    fout << std::endl;

    write_matrices(fout, "d_mrg31k3p_A1", A1_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A1", A1_, MRG31K3P_N, 9, false);
    write_matrices(fout, "d_mrg31k3p_A2", A2_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A2", A2_, MRG31K3P_N, 9, false);
    write_matrices(fout, "d_mrg31k3p_A1P72", A1p72_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A1P72", A1p72_, MRG31K3P_N, 9, false);
    write_matrices(fout, "d_mrg31k3p_A2P72", A2p72_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A2P72", A2p72_, MRG31K3P_N, 9, false);
    write_matrices(fout, "d_mrg31k3p_A1P134", A1p134_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A1P134", A1p134_, MRG31K3P_N, 9, false);
    write_matrices(fout, "d_mrg31k3p_A2P134", A2p134_, MRG31K3P_N, 9, true);
    write_matrices(fout, "h_mrg31k3p_A2P134", A2p134_, MRG31K3P_N, 9, false);

    fout << R"(
#endif // ROCRAND_MRG31K3P_PRECOMPUTED_H_
)";

    delete[] A1_;
    delete[] A2_;
    delete[] A1p72_;
    delete[] A2p72_;
    delete[] A1p134_;
    delete[] A2p134_;

    return 0;
}
