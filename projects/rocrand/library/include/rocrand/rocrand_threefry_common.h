// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
__forceinline__ __device__ __host__ value rotl(value x, int d);

template<>
__forceinline__ __device__ __host__ unsigned long long
    rotl<unsigned long long>(unsigned long long x, int d)
{
    return (x << (d & 63)) | (x >> ((64 - d) & 63));
};

template<>
__forceinline__ __device__ __host__ unsigned int rotl<unsigned int>(unsigned int x, int d)
{
    return (x << (d & 31)) | (x >> ((32 - d) & 31));
};

template<typename value>
__forceinline__ __device__ __host__ value skein_ks_parity();

template<>
__forceinline__ __device__ __host__ unsigned int skein_ks_parity<unsigned int>()
{
    return SKEIN_KS_PARITY32;
}

template<>
__forceinline__ __device__ __host__ unsigned long long skein_ks_parity<unsigned long long>()
{
    return SKEIN_KS_PARITY64;
}

} // end namespace rocrand_device

#endif // ROCRAND_THREEFRY_COMMON_H_
