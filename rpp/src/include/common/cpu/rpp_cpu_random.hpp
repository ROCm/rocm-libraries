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

#ifndef RPP_CPU_RANDOM_HPP
#define RPP_CPU_RANDOM_HPP

#include "rpp_cpu_simd_load_store.hpp"
#include "rpp_cpu_simd_math.hpp"

#define XORWOW_COUNTER_INC              0x587C5     // Hex 0x587C5 = Dec 362437U - xorwow counter increment
#define XORWOW_EXPONENT_MASK            0x3F800000  // Hex 0x3F800000 = Bin 0b111111100000000000000000000000 - 23 bits of mantissa set to 0, 01111111 for the exponent, 0 for the sign bit
#define RPP_2POW32                      0x100000000         // (2^32)
#define RPP_2POW32_INV                  2.3283064e-10f      // (1 / 2^32)
#define RPP_2POW32_INV_MUL_2PI_DIV_2    7.3145906e-10f      // RPP_2POW32_INV_MUL_2PI / 2
#define RPP_2POW32_INV_MUL_2PI          1.46291812e-09f     // (1 / 2^32) * 2PI
#define RPP_2POW32_INV_DIV_2            1.164153218e-10f    // RPP_2POW32_INV / 2

alignas(64) const Rpp32u multiseedStreamOffset[8] = {0x15E975, 0x2359A3, 0x42CC61, 0x1925A7, 0x123AA3, 0x21F149, 0x2DDE23, 0x2A93BB};    // Prime numbers for multiseed stream initialization

const __m128 xmm_p2Pow32 = _mm_set1_ps(RPP_2POW32);
const __m128 xmm_p2Pow32Inv = _mm_set1_ps(RPP_2POW32_INV);
const __m128 xmm_p2Pow32InvDiv2 = _mm_set1_ps(RPP_2POW32_INV_DIV_2);
const __m128 xmm_p2Pow32InvMul2Pi = _mm_set1_ps(RPP_2POW32_INV_MUL_2PI);
const __m128 xmm_p2Pow32InvMul2PiDiv2 = _mm_set1_ps(RPP_2POW32_INV_MUL_2PI_DIV_2);

const __m256 avx_p2Pow32 = _mm256_set1_ps(RPP_2POW32);
const __m256 avx_p2Pow32Inv = _mm256_set1_ps(RPP_2POW32_INV);
const __m256 avx_p2Pow32InvMul2PiDiv2 = _mm256_set1_ps(RPP_2POW32_INV_MUL_2PI_DIV_2);
const __m256 avx_p2Pow32InvMul2Pi = _mm256_set1_ps(RPP_2POW32_INV_MUL_2PI);
const __m256 avx_p2Pow32InvDiv2 = _mm256_set1_ps(RPP_2POW32_INV_DIV_2);

// Helper func for randomization
template<Rpp32s STREAM_SIZE>
inline void rpp_host_rng_xorwow_f32_initialize_multiseed_stream(RpptXorwowState *xorwowInitialState, Rpp32u seed)
{
    Rpp32u xorwowSeedStream[STREAM_SIZE];

    // Loop to initialize seed stream of size STREAM_SIZE based on user seed and offset
    for (int i = 0; i < STREAM_SIZE; i++)
        xorwowSeedStream[i] = seed + multiseedStreamOffset[i];

    // Loop to initialize STREAM_SIZE xorwow initial states for multi-stream random number generation
    for (int i = 0; i < STREAM_SIZE; i++)
    {
        xorwowInitialState[i].x[0] = 0x75BCD15 + xorwowSeedStream[i];      // state param x[0] offset 123456789U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[1] = 0x159A55E5 + xorwowSeedStream[i];     // state param x[1] offset 362436069U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[2] = 0x1F123BB5 + xorwowSeedStream[i];     // state param x[2] offset 521288629U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[3] = 0x5491333 + xorwowSeedStream[i];      // state param x[3] offset 88675123U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[4] = 0x583F19 + xorwowSeedStream[i];       // state param x[4] offset 5783321U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].counter = 0x64F0C9 + xorwowSeedStream[i];    // state param counter offset 6615241U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
    }
}

template<Rpp32s STREAM_SIZE>
inline void rpp_host_rng_xorwow_f32_initialize_multiseed_stream_boxmuller(RpptXorwowStateBoxMuller *xorwowInitialState, Rpp32u seed)
{
    Rpp32u xorwowSeedStream[STREAM_SIZE];

    // Loop to initialize seed stream of size STREAM_SIZE based on user seed and offset
    for (int i = 0; i < STREAM_SIZE; i++)
        xorwowSeedStream[i] = seed + multiseedStreamOffset[i];

    // Loop to initialize STREAM_SIZE xorwow initial states for multi-stream random number generation
    for (int i = 0; i < STREAM_SIZE; i++)
    {
        xorwowInitialState[i].x[0] = 0x75BCD15 + xorwowSeedStream[i];      // state param x[0] offset 123456789U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[1] = 0x159A55E5 + xorwowSeedStream[i];     // state param x[1] offset 362436069U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[2] = 0x1F123BB5 + xorwowSeedStream[i];     // state param x[2] offset 521288629U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[3] = 0x5491333 + xorwowSeedStream[i];      // state param x[3] offset 88675123U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].x[4] = 0x583F19 + xorwowSeedStream[i];       // state param x[4] offset 5783321U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].counter = 0x64F0C9 + xorwowSeedStream[i];    // state param counter offset 6615241U from Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14), 1–6. https://doi.org/10.18637/jss.v008.i14
        xorwowInitialState[i].boxMullerFlag = 0;
        xorwowInitialState[i].boxMullerExtra = 0.0f;
    }
}

template<typename T>
inline void rpp_host_rng_xorwow_state_offsetted_avx(T *xorwowInitialStatePtr, T &xorwowState, Rpp32u offset, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter)
{
    xorwowState = xorwowInitialStatePtr[0];
    xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;

    __m256i pxOffset = _mm256_set1_epi32(offset);
    pxXorwowStateX[0] = _mm256_add_epi32(_mm256_setr_epi32(xorwowInitialStatePtr[0].x[0], xorwowInitialStatePtr[1].x[0], xorwowInitialStatePtr[2].x[0], xorwowInitialStatePtr[3].x[0], xorwowInitialStatePtr[4].x[0], xorwowInitialStatePtr[5].x[0], xorwowInitialStatePtr[6].x[0], xorwowInitialStatePtr[7].x[0]), pxOffset);
    pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1], xorwowInitialStatePtr[1].x[1], xorwowInitialStatePtr[2].x[1], xorwowInitialStatePtr[3].x[1], xorwowInitialStatePtr[4].x[1], xorwowInitialStatePtr[5].x[1], xorwowInitialStatePtr[6].x[1], xorwowInitialStatePtr[7].x[1]);
    pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2], xorwowInitialStatePtr[1].x[2], xorwowInitialStatePtr[2].x[2], xorwowInitialStatePtr[3].x[2], xorwowInitialStatePtr[4].x[2], xorwowInitialStatePtr[5].x[2], xorwowInitialStatePtr[6].x[2], xorwowInitialStatePtr[7].x[2]);
    pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3], xorwowInitialStatePtr[1].x[3], xorwowInitialStatePtr[2].x[3], xorwowInitialStatePtr[3].x[3], xorwowInitialStatePtr[4].x[3], xorwowInitialStatePtr[5].x[3], xorwowInitialStatePtr[6].x[3], xorwowInitialStatePtr[7].x[3]);
    pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4], xorwowInitialStatePtr[1].x[4], xorwowInitialStatePtr[2].x[4], xorwowInitialStatePtr[3].x[4], xorwowInitialStatePtr[4].x[4], xorwowInitialStatePtr[5].x[4], xorwowInitialStatePtr[6].x[4], xorwowInitialStatePtr[7].x[4]);
    *pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter, xorwowInitialStatePtr[1].counter, xorwowInitialStatePtr[2].counter, xorwowInitialStatePtr[3].counter, xorwowInitialStatePtr[4].counter, xorwowInitialStatePtr[5].counter, xorwowInitialStatePtr[6].counter, xorwowInitialStatePtr[7].counter);
}

template<typename T>
inline void rpp_host_rng_xorwow_state_offsetted_sse(T *xorwowInitialStatePtr, T &xorwowState, Rpp32u offset, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter)
{
    xorwowState = xorwowInitialStatePtr[0];
    xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;

    __m128i pxOffset = _mm_set1_epi32(offset);
    pxXorwowStateX[0] = _mm_add_epi32(_mm_setr_epi32(xorwowInitialStatePtr[0].x[0], xorwowInitialStatePtr[1].x[0], xorwowInitialStatePtr[2].x[0], xorwowInitialStatePtr[3].x[0]), pxOffset);
    pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1], xorwowInitialStatePtr[1].x[1], xorwowInitialStatePtr[2].x[1], xorwowInitialStatePtr[3].x[1]);
    pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2], xorwowInitialStatePtr[1].x[2], xorwowInitialStatePtr[2].x[2], xorwowInitialStatePtr[3].x[2]);
    pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3], xorwowInitialStatePtr[1].x[3], xorwowInitialStatePtr[2].x[3], xorwowInitialStatePtr[3].x[3]);
    pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4], xorwowInitialStatePtr[1].x[4], xorwowInitialStatePtr[2].x[4], xorwowInitialStatePtr[3].x[4]);
    *pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter, xorwowInitialStatePtr[1].counter, xorwowInitialStatePtr[2].counter, xorwowInitialStatePtr[3].counter);
}

inline void rpp_host_rng_xorwow_8_state_update_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Initialize avx-xorwow specific constants
    __m256i pxXorwowCounterInc = _mm256_set1_epi32(XORWOW_COUNTER_INC);

    // Save current first and last x-params of xorwow state and compute pxT
    __m256i pxT = pxXorwowStateXParam[0];                                                           // uint t  = xorwowState->x[0];
    __m256i pxS = pxXorwowStateXParam[4];                                                           // uint s  = xorwowState->x[4];
    pxT = _mm256_xor_si256(pxT, _mm256_srli_epi32(pxT, 2));                                         // t ^= t >> 2;
    pxT = _mm256_xor_si256(pxT, _mm256_slli_epi32(pxT, 1));                                         // t ^= t << 1;
    pxT = _mm256_xor_si256(pxT, _mm256_xor_si256(pxS, _mm256_slli_epi32(pxS, 4)));                  // t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    pxXorwowStateXParam[0] = pxXorwowStateXParam[1];                                                // xorwowState->x[0] = xorwowState->x[1];
    pxXorwowStateXParam[1] = pxXorwowStateXParam[2];                                                // xorwowState->x[1] = xorwowState->x[2];
    pxXorwowStateXParam[2] = pxXorwowStateXParam[3];                                                // xorwowState->x[2] = xorwowState->x[3];
    pxXorwowStateXParam[3] = pxXorwowStateXParam[4];                                                // xorwowState->x[3] = xorwowState->x[4];
    pxXorwowStateXParam[4] = pxT;                                                                   // xorwowState->x[4] = t;
    *pxXorwowStateCounterParam = _mm256_add_epi32(*pxXorwowStateCounterParam, pxXorwowCounterInc);  // xorwowState->counter += XORWOW_COUNTER_INC;
}

inline __m256i rpp_host_rng_xorwow_8_u32_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_8_state_update_avx(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Return u32 random number
    return  _mm256_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam);   // return x[4] + counter
}

inline __m256 rpp_host_rng_xorwow_8_f32_avx(__m256i *pxXorwowStateXParam, __m256i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_8_state_update_avx(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Initialize avx-xorwow specific constants
    __m256i px7FFFFF = _mm256_set1_epi32(0x7FFFFF);
    __m256i pxExponentFloat = _mm256_set1_epi32(XORWOW_EXPONENT_MASK);

    // Create float representation and return 0 <= pxS < 1
    __m256i pxS = _mm256_or_si256(pxExponentFloat, _mm256_and_si256(_mm256_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam), px7FFFFF));   // uint out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));
    return _mm256_sub_ps(*(__m256 *)&pxS, avx_p1);                                                                                                      // return  *(float *)&out - 1;
}

inline void rpp_host_rng_xorwow_4_state_update_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Initialize sse-xorwow specific constants
    __m128i pxXorwowCounterInc = _mm_set1_epi32(XORWOW_COUNTER_INC);

    // Save current first and last x-params of xorwow state and compute pxT
    __m128i pxT = pxXorwowStateXParam[0];                                                       // uint t  = xorwowState->x[0];
    __m128i pxS = pxXorwowStateXParam[4];                                                       // uint s  = xorwowState->x[4];
    pxT = _mm_xor_si128(pxT, _mm_srli_epi32(pxT, 2));                                           // t ^= t >> 2;
    pxT = _mm_xor_si128(pxT, _mm_slli_epi32(pxT, 1));                                           // t ^= t << 1;
    pxT = _mm_xor_si128(pxT, _mm_xor_si128(pxS, _mm_slli_epi32(pxS, 4)));                       // t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    pxXorwowStateXParam[0] = pxXorwowStateXParam[1];                                            // xorwowState->x[0] = xorwowState->x[1];
    pxXorwowStateXParam[1] = pxXorwowStateXParam[2];                                            // xorwowState->x[1] = xorwowState->x[2];
    pxXorwowStateXParam[2] = pxXorwowStateXParam[3];                                            // xorwowState->x[2] = xorwowState->x[3];
    pxXorwowStateXParam[3] = pxXorwowStateXParam[4];                                            // xorwowState->x[3] = xorwowState->x[4];
    pxXorwowStateXParam[4] = pxT;                                                               // xorwowState->x[4] = t;
    *pxXorwowStateCounterParam = _mm_add_epi32(*pxXorwowStateCounterParam, pxXorwowCounterInc); // xorwowState->counter += XORWOW_COUNTER_INC;
}

inline __m128i rpp_host_rng_xorwow_4_u32_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_4_state_update_sse(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Return u32 random number
    return  _mm_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam);   // return x[4] + counter
}

inline __m128 rpp_host_rng_xorwow_4_f32_sse(__m128i *pxXorwowStateXParam, __m128i *pxXorwowStateCounterParam)
{
    // Update xorwow state
    rpp_host_rng_xorwow_4_state_update_sse(pxXorwowStateXParam, pxXorwowStateCounterParam);

    // Initialize sse-xorwow specific constants
    __m128i px7FFFFF = _mm_set1_epi32(0x7FFFFF);
    __m128i pxExponentFloat = _mm_set1_epi32(XORWOW_EXPONENT_MASK);

    // Create float representation and return 0 <= pxS < 1
    __m128i pxS = _mm_or_si128(pxExponentFloat, _mm_and_si128(_mm_add_epi32(pxXorwowStateXParam[4], *pxXorwowStateCounterParam), px7FFFFF));    // uint out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));
    return _mm_sub_ps(*(__m128 *)&pxS, xmm_p1);                                                                                                 // return  *(float *)&out - 1;
}

template<typename T>
inline void rpp_host_rng_xorwow_state_update(T *xorwowState)
{
    // Save current first and last x-params of xorwow state and compute t
    Rpp32s t  = xorwowState->x[0];
    Rpp32s s  = xorwowState->x[4];
    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);

    // Update all 6 xorwow state params
    xorwowState->x[0] = xorwowState->x[1];                              // set new state param x[0]
    xorwowState->x[1] = xorwowState->x[2];                              // set new state param x[1]
    xorwowState->x[2] = xorwowState->x[3];                              // set new state param x[2]
    xorwowState->x[3] = xorwowState->x[4];                              // set new state param x[3]
    xorwowState->x[4] = t;                                              // set new state param x[4]
    xorwowState->counter = xorwowState->counter + XORWOW_COUNTER_INC;   // set new state param counter
}

template<typename T>
inline Rpp32u rpp_host_rng_xorwow_u32(T *xorwowState)
{
    // Update xorwow state
    rpp_host_rng_xorwow_state_update(xorwowState);

    // Return u32 random number
    return  xorwowState->x[4] + xorwowState->counter;   // return x[4] + counter
}

template<typename T>
inline Rpp32f rpp_host_rng_xorwow_f32(T *xorwowState)
{
    // Update xorwow state
    rpp_host_rng_xorwow_state_update(xorwowState);

    // Create float representation and return 0 <= outFloat < 1
    Rpp32u out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));  // bitmask 23 mantissa bits, OR with exponent
    Rpp32f outFloat = *(Rpp32f *)&out;                                                              // reinterpret out as float
    return  outFloat - 1;                                                                           // return 0 <= outFloat < 1
}

inline void rpp_host_rng_16_gaussian_f32_avx(__m256 *pRngVals, __m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter)
{
    __m256 pU, pV, pS;                                                                                  // Rpp32f u, v, s;
    pU = _mm256_cvtepi32_ps(rpp_host_rng_xorwow_8_u32_avx(pxXorwowStateX, pxXorwowStateCounter));       // u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pV = _mm256_cvtepi32_ps(rpp_host_rng_xorwow_8_u32_avx(pxXorwowStateX, pxXorwowStateCounter));       // v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pS = _mm256_cmp_ps(pU, avx_p0, _CMP_LT_OQ);                                                         // Adjust int32 out of bound values in float for u
    pU = _mm256_or_ps(_mm256_andnot_ps(pS, pU), _mm256_and_ps(pS, _mm256_add_ps(avx_p2Pow32, pU)));     // Adjust int32 out of bound values in float for u
    pS = _mm256_cmp_ps(pV, avx_p0, _CMP_LT_OQ);                                                         // Adjust int32 out of bound values in float for v
    pV = _mm256_or_ps(_mm256_andnot_ps(pS, pV), _mm256_and_ps(pS, _mm256_add_ps(avx_p2Pow32, pV)));     // Adjust int32 out of bound values in float for v
    pU = _mm256_fmadd_ps(pU, avx_p2Pow32Inv, avx_p2Pow32InvDiv2);                                       // u = u * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    pV = _mm256_fmadd_ps(pV, avx_p2Pow32InvMul2Pi, avx_p2Pow32InvMul2PiDiv2);                           // v = v * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
    pS = _mm256_sqrt_ps(_mm256_mul_ps(avx_pm2, log_ps(pU)));                                            // s = sqrt(-2.0f * std::log(u));
    sincos_ps(pV, &pU, &pV);                                                                            // std::sin(v) and std::cos(v) computation
    pRngVals[0] = _mm256_mul_ps(pU, pS);                                                                // u = std::sin(v) * s;
    pRngVals[1] = _mm256_mul_ps(pV, pS);                                                                // v = std::cos(v) * s;
}

inline void rpp_host_rng_8_gaussian_f32_sse(__m128 *pRngVals, __m128i *pxXorwowStateX, __m128i *pxXorwowStateCounter)
{
    __m128 pU, pV, pS;                                                                          // Rpp32f u, v, s;
    pU = _mm_cvtepi32_ps(rpp_host_rng_xorwow_4_u32_sse(pxXorwowStateX, pxXorwowStateCounter));  // u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pV = _mm_cvtepi32_ps(rpp_host_rng_xorwow_4_u32_sse(pxXorwowStateX, pxXorwowStateCounter));  // v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState);
    pS = _mm_cmplt_ps(pU, xmm_p0);                                                              // Adjust int32 out of bound values in float for u
    pU = _mm_or_ps(_mm_andnot_ps(pS, pU), _mm_and_ps(pS, _mm_add_ps(xmm_p2Pow32, pU)));         // Adjust int32 out of bound values in float for u
    pS = _mm_cmplt_ps(pV, xmm_p0);                                                              // Adjust int32 out of bound values in float for v
    pV = _mm_or_ps(_mm_andnot_ps(pS, pV), _mm_and_ps(pS, _mm_add_ps(xmm_p2Pow32, pV)));         // Adjust int32 out of bound values in float for v
    pU = _mm_fmadd_ps(pU, xmm_p2Pow32Inv, xmm_p2Pow32InvDiv2);                                  // u = u * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    pV = _mm_fmadd_ps(pV, xmm_p2Pow32InvMul2Pi, xmm_p2Pow32InvMul2PiDiv2);                      // v = v * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
    pS = _mm_sqrt_ps(_mm_mul_ps(xmm_pm2, log_ps(pU)));                                          // s = sqrt(-2.0f * std::log(u));
    sincos_ps(pV, &pU, &pV);                                                                    // std::sin(v) and std::cos(v) computation
    pRngVals[0] = _mm_mul_ps(pU, pS);                                                           // u = std::sin(v) * s;
    pRngVals[1] = _mm_mul_ps(pV, pS);                                                           // v = std::cos(v) * s;
}

inline float rpp_host_rng_1_gaussian_f32(RpptXorwowStateBoxMuller *xorwowState)
{
    if(!xorwowState->boxMullerFlag)
    {
        Rpp32f u, v, s;
        u = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
        v = (Rpp32f)rpp_host_rng_xorwow_u32(xorwowState) * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
        s = sqrt(-2.0f * std::log(u));
        u = std::sin(v) * s;
        v = std::cos(v) * s;
        xorwowState->boxMullerExtra = v;
        xorwowState->boxMullerFlag = 1;
        return u;
    }
    xorwowState->boxMullerFlag = 0;

    return xorwowState->boxMullerExtra;
}

#endif // RPP_CPU_RANDOM_HPP
