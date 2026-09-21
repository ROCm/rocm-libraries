// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ROCRAND_THREEFRY_COMMON_H_
#define ROCRAND_THREEFRY_COMMON_H_

#include <hip/hip_runtime.h>

// C240 constant for Skein Hash function Threefish
#define SKEIN_MK_64(hi32, lo32) ((lo32) + (((unsigned long long)(hi32)) << 32))
#define SKEIN_KS_PARITY64 SKEIN_MK_64(0x1BD11BDA, 0xA9FC1A22)
#define SKEIN_KS_PARITY32 0x1BD11BDA

namespace rocrand_device
{

template<typename value>
__forceinline__ __device__ __host__
value rotl(value x, int d);

template<>
__forceinline__ __device__ __host__
unsigned long long rotl<unsigned long long>(unsigned long long x, int d)
{
    return (x << (d & 63)) | (x >> ((64 - d) & 63));
};

template<>
__forceinline__ __device__ __host__
unsigned int rotl<unsigned int>(unsigned int x, int d)
{
    return (x << (d & 31)) | (x >> ((32 - d) & 31));
};

template<typename value>
__forceinline__ __device__ __host__
value skein_ks_parity();

template<>
__forceinline__ __device__ __host__
unsigned int skein_ks_parity<unsigned int>()
{
    return SKEIN_KS_PARITY32;
}

template<>
__forceinline__ __device__ __host__
unsigned long long skein_ks_parity<unsigned long long>()
{
    return SKEIN_KS_PARITY64;
}

template<class value>
__forceinline__ __device__ __host__
int threefry_rotation_array(int index)
    = delete;

template<>
__forceinline__ __device__ __host__
int threefry_rotation_array<unsigned int>(int index)
{
    // Output from skein_rot_search (srs32x2-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Tue Jul 12 11:11:33 2011
    // rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format
    static constexpr int THREEFRY_ROTATION_32_2[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    return THREEFRY_ROTATION_32_2[index];
}

template<>
__forceinline__ __device__ __host__
int threefry_rotation_array<unsigned long long>(int index)
{
    // Output from skein_rot_search: (srs64_B64-X1000)
    // Random seed = 1. BlockSize = 128 bits. sampleCnt =  1024. rounds =  8, minHW_or=57
    // Start: Tue Mar  1 10:07:48 2011
    // rMin = 0.136. #0325[*15] [CRC=455A682F. hw_OR=64. cnt=16384. blkSize= 128].format
    static constexpr int THREEFRY_ROTATION_64_2[8] = {16, 42, 12, 31, 16, 32, 24, 21};
    return THREEFRY_ROTATION_64_2[index];
}

template<typename state_value, typename value, unsigned int Nrounds>
__forceinline__ __device__ __host__
auto rounds_2(state_value X, value ks[3]) -> std::enable_if_t<Nrounds % 4 != 0, state_value>
{
    for(unsigned int round_idx = 0; round_idx < Nrounds; round_idx++)
    {
        X.x += X.y;
        X.y = rotl<value>(X.y, threefry_rotation_array<value>(round_idx & 7u));
        X.y ^= X.x;

        if((round_idx & 3u) == 3)
        {
            unsigned int inject_idx = round_idx / 4;
            // InjectKey(r = 1 + inject_idx)
            X.x += ks[(1 + inject_idx) % 3];
            X.y += ks[(2 + inject_idx) % 3];
            X.y += 1 + inject_idx;
        }
    }
    return X;
}

template<typename state_value, typename value, unsigned int Nrounds>
__forceinline__ __device__ __host__
auto rounds_2(state_value X, value ks[3]) -> std::enable_if_t<Nrounds % 4 == 0, state_value>
{
#pragma unroll
    for(unsigned int i = 0; i < Nrounds / 4; i++)
    {
        unsigned int round_idx = 4 * i;
        for(unsigned int j = 0; j < 4; j++)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>((round_idx + j) & 7u));
            X.y ^= X.x;
        }

        // (round_idx & 3u) == 3
        // InjectKey(r = 1 + i)
        X.x += ks[(1 + i) % 3];
        X.y += ks[(2 + i) % 3];
        X.y += 1 + i;
    }
    return X;
}

template<class value>
__forceinline__ __device__ __host__
int threefry_rotation_array(int indexX, int indexY)
    = delete;

template<>
__forceinline__ __device__ __host__
int threefry_rotation_array<unsigned int>(int indexX, int indexY)
{
    // Output from skein_rot_search: (srs-B128-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Mon Aug 24 22:41:36 2009
    // ...
    // rMin = 0.472. #0A4B[*33] [CRC=DD1ECE0F. hw_OR=31. cnt=16384. blkSize= 128].format
    static constexpr int THREEFRY_ROTATION_32_4[8][2] = {
        {10, 26},
        {11, 21},
        {13, 27},
        {23,  5},
        { 6, 20},
        {17, 11},
        {25, 10},
        {18, 20}
    };
    return THREEFRY_ROTATION_32_4[indexX][indexY];
}

template<>
__forceinline__ __device__ __host__
int threefry_rotation_array<unsigned long long>(int indexX, int indexY)
{
    // These are the R_256 constants from the Threefish reference sources
    // with names changed to R_64x4... */
    static constexpr int THREEFRY_ROTATION_64_4[8][2] = {
        {14, 16},
        {52, 57},
        {23, 40},
        { 5, 37},
        {25, 33},
        {46, 12},
        {58, 22},
        {32, 32}
    };
    return THREEFRY_ROTATION_64_4[indexX][indexY];
}

template<typename state_value, typename value, unsigned int Nrounds>
__forceinline__ __device__ __host__
auto rounds_4(state_value X, value ks[5]) -> std::enable_if_t<Nrounds % 4 != 0, state_value>
{
    for(unsigned int round_idx = 0; round_idx < Nrounds; round_idx++)
    {
        int rot_0 = threefry_rotation_array<value>(round_idx & 7u, 0);
        int rot_1 = threefry_rotation_array<value>(round_idx & 7u, 1);
        if((round_idx & 2u) == 0)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, rot_0);
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, rot_1);
            X.w ^= X.z;
        }
        else
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, rot_0);
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, rot_1);
            X.y ^= X.z;
        }

        if((round_idx & 3u) == 3)
        {
            unsigned int inject_idx = round_idx / 4;
            // InjectKey(r = 1 + inject_idx)
            X.x += ks[(1 + inject_idx) % 5];
            X.y += ks[(2 + inject_idx) % 5];
            X.z += ks[(3 + inject_idx) % 5];
            X.w += ks[(4 + inject_idx) % 5];
            X.w += 1 + inject_idx;
        }
    }
    return X;
}

template<typename state_value, typename value, unsigned int Nrounds>
__forceinline__ __device__ __host__
auto rounds_4(state_value X, value ks[5]) -> std::enable_if_t<Nrounds % 4 == 0, state_value>
{
#pragma unroll
    for(unsigned int i = 0; i < Nrounds / 4; i++)
    {
        unsigned int round_idx = 4 * i;

        // (round_idx & 2u) == 0
        unsigned int j = 0;
        for(; j < 2; j++)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>((round_idx + j) & 7u, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>((round_idx + j) & 7u, 1));
            X.w ^= X.z;
        }

        // (round_idx & 2u) != 0
        for(; j < 4; j++)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>((round_idx + j) & 7u, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>((round_idx + j) & 7u, 1));
            X.y ^= X.z;
        }

        // (round_idx & 3u) == 3
        // InjectKey(r = 1 + i)
        X.x += ks[(1 + i) % 5];
        X.y += ks[(2 + i) % 5];
        X.z += ks[(3 + i) % 5];
        X.w += ks[(4 + i) % 5];
        X.w += 1 + i;
    }
    return X;
}

} // end namespace rocrand_device

#endif // ROCRAND_THREEFRY_COMMON_H_
