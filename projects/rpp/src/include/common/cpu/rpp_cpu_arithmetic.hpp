/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_CPU_ARITHMETIC_HPP
#define RPP_CPU_ARITHMETIC_HPP

inline void compute_sum_16_host(__m256i *p, __m256i *pSum)
{
    pSum[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSum[0]); //add 16 values to 8
}

inline void compute_sum_48_host(__m256i *p, __m256i *pSumR, __m256i *pSumG, __m256i *pSumB)
{
    pSumR[0] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), pSumR[0]); //add 16R values and bring it down to 8
    pSumG[0] = _mm256_add_epi32(_mm256_add_epi32(p[2], p[3]), pSumG[0]); //add 16G values and bring it down to 8
    pSumB[0] = _mm256_add_epi32(_mm256_add_epi32(p[4], p[5]), pSumB[0]); //add 16B values and bring it down to 8
}

inline void compute_sum_8_host(__m256d *p, __m256d *pSum)
{
    pSum[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSum[0]); //add 8 values and bring it down to 4
}

inline void compute_sum_24_host(__m256d *p, __m256d *pSumR, __m256d *pSumG, __m256d *pSumB)
{
    pSumR[0] = _mm256_add_pd(_mm256_add_pd(p[0], p[1]), pSumR[0]); //add 8R values and bring it down to 4
    pSumG[0] = _mm256_add_pd(_mm256_add_pd(p[2], p[3]), pSumG[0]); //add 8G values and bring it down to 4
    pSumB[0] = _mm256_add_pd(_mm256_add_pd(p[4], p[5]), pSumB[0]); //add 8B values and bring it down to 4
}

#endif // RPP_CPU_ARITHMETIC_HPP
