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

#ifndef RPP_CPU_SIMD_MATH_HPP
#define RPP_CPU_SIMD_MATH_HPP

#define NEWTON_METHOD_INITIAL_GUESS     0x5f3759df          // Initial guess for Newton Raphson Inverse Square Root

#define set1_ps_hex(x) _mm_castsi128_ps(_mm_set1_epi32(x))
#define set1_ps_hex_avx(x) _mm256_castsi256_ps(_mm256_set1_epi32(x))

#ifndef RPP_SIMD_COMMON_VARIABLES
#define RPP_SIMD_COMMON_VARIABLES
const __m128 xmm_p0 = _mm_setzero_ps();
const __m128 xmm_p3 = _mm_set1_ps(3.0f);

const __m256 avx_p0 = _mm256_set1_ps(0.0f);
const __m256 avx_p1 = _mm256_set1_ps(1.0f);
const __m256 avx_p3 = _mm256_set1_ps(3.0f);
#endif

const __m128 xmm_p1op255 = _mm_set1_ps(1.0f / 255.0f);
const __m128i xmm_newtonMethodInitialGuess = _mm_set1_epi32(NEWTON_METHOD_INITIAL_GUESS);

const __m256 avx_p1op255 = _mm256_set1_ps(1.0f / 255.0f);
const __m256i avx_newtonMethodInitialGuess = _mm256_set1_epi32(NEWTON_METHOD_INITIAL_GUESS);

const __m128 xmm_cephesSQRTHF = _mm_set1_ps(0.707106781186547524);
const __m128 xmm_cephesLogP0 = _mm_set1_ps(7.0376836292E-2);
const __m128 xmm_cephesLogP1 = _mm_set1_ps(-1.1514610310E-1);
const __m128 xmm_cephesLogP2 = _mm_set1_ps(1.1676998740E-1);
const __m128 xmm_cephesLogP3 = _mm_set1_ps(-1.2420140846E-1);
const __m128 xmm_cephesLogP4 = _mm_set1_ps(1.4249322787E-1);
const __m128 xmm_cephesLogP5 = _mm_set1_ps(-1.6668057665E-1);
const __m128 xmm_cephesLogP6 = _mm_set1_ps(2.0000714765E-1);
const __m128 xmm_cephesLogP7 = _mm_set1_ps(-2.4999993993E-1);
const __m128 xmm_cephesLogP8 = _mm_set1_ps(3.3333331174E-1);
const __m128 xmm_cephesLogQ1 = _mm_set1_ps(-2.12194440e-4);
const __m128 xmm_cephesLogQ2 = _mm_set1_ps(0.693359375);

const __m256 avx_cephesSQRTHF = _mm256_set1_ps(0.707106781186547524);
const __m256 avx_cephesLogP0 = _mm256_set1_ps(7.0376836292E-2);
const __m256 avx_cephesLogP1 = _mm256_set1_ps(-1.1514610310E-1);
const __m256 avx_cephesLogP2 = _mm256_set1_ps(1.1676998740E-1);
const __m256 avx_cephesLogP3 = _mm256_set1_ps(-1.2420140846E-1);
const __m256 avx_cephesLogP4 = _mm256_set1_ps(1.4249322787E-1);
const __m256 avx_cephesLogP5 = _mm256_set1_ps(-1.6668057665E-1);
const __m256 avx_cephesLogP6 = _mm256_set1_ps(2.0000714765E-1);
const __m256 avx_cephesLogP7 = _mm256_set1_ps(-2.4999993993E-1);
const __m256 avx_cephesLogP8 = _mm256_set1_ps(3.3333331174E-1);
const __m256 avx_cephesLogQ1 = _mm256_set1_ps(-2.12194440e-4);
const __m256 avx_cephesLogQ2 = _mm256_set1_ps(0.693359375);

static const __m128 _ps_0 = _mm_set1_ps(0.f);
static const __m128 _ps_1 = _mm_set1_ps(1.f);
static const __m128 _ps_0p5 = _mm_set1_ps(0.5f);
static const __m128 _ps_n0p5 = _mm_set1_ps(-0.5f);
static const __m128 _ps_1p5 = _mm_set1_ps(1.5f);
static const __m128 _ps_min_norm_pos = set1_ps_hex(0x00800000);
static const __m128 _ps_mant_mask = set1_ps_hex(0x7f800000);
static const __m128 _ps_inv_mant_mask = set1_ps_hex(~0x7f800000);
static const __m128 _ps_sign_mask = set1_ps_hex(0x80000000);
static const __m128 _ps_inv_sign_mask = set1_ps_hex(~0x80000000);

static const __m128i _pi32_1 = _mm_set1_epi32(1);
static const __m128i _pi32_inv1 = _mm_set1_epi32(~1);
static const __m128i _pi32_2 = _mm_set1_epi32(2);
static const __m128i _pi32_4 = _mm_set1_epi32(4);
static const __m128i _pi32_0x7f = _mm_set1_epi32(0x7f);

static const __m128 _ps_minus_cephes_DP1 = _mm_set1_ps(-0.78515625f);
static const __m128 _ps_minus_cephes_DP2 = _mm_set1_ps(-2.4187564849853515625e-4f);
static const __m128 _ps_minus_cephes_DP3 = _mm_set1_ps(-3.77489497744594108e-8f);
static const __m128 _ps_sincof_p0 = _mm_set1_ps(-1.9515295891E-4f);
static const __m128 _ps_sincof_p1 = _mm_set1_ps( 8.3321608736E-3f);
static const __m128 _ps_sincof_p2 = _mm_set1_ps(-1.6666654611E-1f);
static const __m128 _ps_coscof_p0 = _mm_set1_ps( 2.443315711809948E-005f);
static const __m128 _ps_coscof_p1 = _mm_set1_ps(-1.388731625493765E-003f);
static const __m128 _ps_coscof_p2 = _mm_set1_ps( 4.166664568298827E-002f);
static const __m128 _ps_cephes_FOPI = _mm_set1_ps(1.27323954473516f); // 4 / M_PI

static const __m256 _ps_0p5_avx = _mm256_set1_ps(0.5f);
static const __m256 _ps_n0p5_avx = _mm256_set1_ps(-0.5f);
static const __m256 _ps_1p5_avx = _mm256_set1_ps(1.5f);
static const __m256 _ps_min_norm_pos_avx = set1_ps_hex_avx(0x00800000);
static const __m256 _ps_inv_mant_mask_avx = set1_ps_hex_avx(~0x7f800000);
static const __m256 _ps_sign_mask_avx = set1_ps_hex_avx(0x80000000);
static const __m256 _ps_inv_sign_mask_avx = set1_ps_hex_avx(~0x80000000);

static const __m256i _pi32_1_avx = _mm256_set1_epi32(1);
static const __m256i _pi32_inv1_avx = _mm256_set1_epi32(~1);
static const __m256i _pi32_2_avx = _mm256_set1_epi32(2);
static const __m256i _pi32_4_avx = _mm256_set1_epi32(4);
static const __m256i _pi32_0x7f_avx = _mm256_set1_epi32(0x7f);

static const __m256 _ps_minus_cephes_DP1_avx = _mm256_set1_ps(-0.78515625f);
static const __m256 _ps_minus_cephes_DP2_avx = _mm256_set1_ps(-2.4187564849853515625e-4f);
static const __m256 _ps_minus_cephes_DP3_avx = _mm256_set1_ps(-3.77489497744594108e-8f);
static const __m256 _ps_sincof_p0_avx = _mm256_set1_ps(-1.9515295891E-4f);
static const __m256 _ps_sincof_p1_avx = _mm256_set1_ps( 8.3321608736E-3f);
static const __m256 _ps_sincof_p2_avx = _mm256_set1_ps(-1.6666654611E-1f);
static const __m256 _ps_coscof_p0_avx = _mm256_set1_ps( 2.443315711809948E-005f);
static const __m256 _ps_coscof_p1_avx = _mm256_set1_ps(-1.388731625493765E-003f);
static const __m256 _ps_coscof_p2_avx = _mm256_set1_ps( 4.166664568298827E-002f);
static const __m256 _ps_cephes_FOPI_avx = _mm256_set1_ps(1.27323954473516f); // 4 / M_PI

static const __m128 _ps_atanrange_hi = _mm_set1_ps(2.414213562373095);
static const __m128 _ps_atanrange_lo = _mm_set1_ps(0.4142135623730950);
static const __m128 _ps_cephes_PIF = _mm_set1_ps(3.141592653589793238);
static const __m128 _ps_cephes_PIO2F = _mm_set1_ps(1.5707963267948966192);
static const __m128 _ps_cephes_PIO4F = _mm_set1_ps(0.7853981633974483096);

static const __m128 _ps_atancof_p0 = _mm_set1_ps(8.05374449538e-2);
static const __m128 _ps_atancof_p1 = _mm_set1_ps(1.38776856032e-1);
static const __m128 _ps_atancof_p2 = _mm_set1_ps(1.99777106478e-1);
static const __m128 _ps_atancof_p3 = _mm_set1_ps(3.33329491539e-1);

static const __m256 _ps_atanrange_hi_avx = _mm256_set1_ps(2.414213562373095);
static const __m256 _ps_atanrange_lo_avx = _mm256_set1_ps(0.4142135623730950);
static const __m256 _ps_cephes_PIF_avx = _mm256_set1_ps(3.141592653589793238);
static const __m256 _ps_cephes_PIO2F_avx = _mm256_set1_ps(1.5707963267948966192);
static const __m256 _ps_cephes_PIO4F_avx = _mm256_set1_ps(0.7853981633974483096);

static const __m256 _ps_atancof_p0_avx = _mm256_set1_ps(8.05374449538e-2);
static const __m256 _ps_atancof_p1_avx = _mm256_set1_ps(1.38776856032e-1);
static const __m256 _ps_atancof_p2_avx = _mm256_set1_ps(1.99777106478e-1);
static const __m256 _ps_atancof_p3_avx = _mm256_set1_ps(3.33329491539e-1);

inline void rpp_normalize48(__m128 *p)
{
    p[0] = _mm_mul_ps(p[0], xmm_p1op255);
    p[1] = _mm_mul_ps(p[1], xmm_p1op255);
    p[2] = _mm_mul_ps(p[2], xmm_p1op255);
    p[3] = _mm_mul_ps(p[3], xmm_p1op255);
    p[4] = _mm_mul_ps(p[4], xmm_p1op255);
    p[5] = _mm_mul_ps(p[5], xmm_p1op255);
    p[6] = _mm_mul_ps(p[6], xmm_p1op255);
    p[7] = _mm_mul_ps(p[7], xmm_p1op255);
    p[8] = _mm_mul_ps(p[8], xmm_p1op255);
    p[9] = _mm_mul_ps(p[9], xmm_p1op255);
    p[10] = _mm_mul_ps(p[10], xmm_p1op255);
    p[11] = _mm_mul_ps(p[11], xmm_p1op255);
}

inline void rpp_normalize48_avx(__m256 *p)
{
    p[0] = _mm256_mul_ps(p[0], avx_p1op255);
    p[1] = _mm256_mul_ps(p[1], avx_p1op255);
    p[2] = _mm256_mul_ps(p[2], avx_p1op255);
    p[3] = _mm256_mul_ps(p[3], avx_p1op255);
    p[4] = _mm256_mul_ps(p[4], avx_p1op255);
    p[5] = _mm256_mul_ps(p[5], avx_p1op255);
}

inline void rpp_normalize24_avx(__m256 *p)
{
    p[0] = _mm256_mul_ps(p[0], avx_p1op255);
    p[1] = _mm256_mul_ps(p[1], avx_p1op255);
    p[2] = _mm256_mul_ps(p[2], avx_p1op255);
}

inline void rpp_multiply48_constant(__m256 *p, __m256 pMultiplier)
{
    p[0] = _mm256_mul_ps(p[0], pMultiplier);
    p[1] = _mm256_mul_ps(p[1], pMultiplier);
    p[2] = _mm256_mul_ps(p[2], pMultiplier);
    p[3] = _mm256_mul_ps(p[3], pMultiplier);
    p[4] = _mm256_mul_ps(p[4], pMultiplier);
    p[5] = _mm256_mul_ps(p[5], pMultiplier);
}

inline void rpp_multiply48_constant(__m128 *p, __m128 pMultiplier)
{
    p[0] = _mm_mul_ps(p[0], pMultiplier);
    p[1] = _mm_mul_ps(p[1], pMultiplier);
    p[2] = _mm_mul_ps(p[2], pMultiplier);
    p[3] = _mm_mul_ps(p[3], pMultiplier);
    p[4] = _mm_mul_ps(p[4], pMultiplier);
    p[5] = _mm_mul_ps(p[5], pMultiplier);
    p[6] = _mm_mul_ps(p[6], pMultiplier);
    p[7] = _mm_mul_ps(p[7], pMultiplier);
    p[8] = _mm_mul_ps(p[8], pMultiplier);
    p[9] = _mm_mul_ps(p[9], pMultiplier);
    p[10] = _mm_mul_ps(p[10], pMultiplier);
    p[11] = _mm_mul_ps(p[11], pMultiplier);
}

inline void rpp_multiply24_constant(__m256 *p, __m256 pMultiplier)
{
    p[0] = _mm256_mul_ps(p[0], pMultiplier);
    p[1] = _mm256_mul_ps(p[1], pMultiplier);
    p[2] = _mm256_mul_ps(p[2], pMultiplier);
}

inline void rpp_multiply16_constant(__m256 *p, __m256 pMultiplier)
{
    p[0] = _mm256_mul_ps(p[0], pMultiplier);
    p[1] = _mm256_mul_ps(p[1], pMultiplier);
}

inline void rpp_multiply16_constant(__m128 *p, __m128 pMultiplier)
{
    p[0] = _mm_mul_ps(p[0], pMultiplier);
    p[1] = _mm_mul_ps(p[1], pMultiplier);
    p[2] = _mm_mul_ps(p[2], pMultiplier);
    p[3] = _mm_mul_ps(p[3], pMultiplier);
}

inline __m128i _mm_mullo_epi8(__m128i a, __m128i b)
{
    __m128i zero = _mm_setzero_si128();
    __m128i Alo = _mm_cvtepu8_epi16(a);
    __m128i Ahi = _mm_unpackhi_epi8(a, zero);
    __m128i Blo = _mm_cvtepu8_epi16(b);
    __m128i Bhi = _mm_unpackhi_epi8(b, zero);
    __m128i Clo = _mm_mullo_epi16(Alo, Blo);
    __m128i Chi = _mm_mullo_epi16(Ahi, Bhi);
    __m128i maskLo = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i maskHi = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i C = _mm_or_si128(_mm_shuffle_epi8(Clo, maskLo), _mm_shuffle_epi8(Chi, maskHi));

     return C;
}

static inline Rpp32u HorMin(__m128i pmin)
{
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_shufflelo_epi16(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_srli_epi16(pmin, 8));
    return (_mm_cvtsi128_si32(pmin) & 0x000000FF);
}

static inline Rpp32u HorMax(__m128i pmax)
{
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_shufflelo_epi16(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_srli_epi16(pmax, 8));
    return (_mm_cvtsi128_si32(pmax) & 0x000000FF);
}

#if __AVX__
static inline Rpp32u HorMin256(__m256i pmin)
{
    __m128i pmin_128;
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = M256I(pmin).m256i_i128[0];
    pmin_128 = _mm_min_epu8(pmin_128, _mm_shufflelo_epi16(pmin_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = _mm_min_epu8(pmin_128, _mm_srli_epi16(pmin_128, 8));
    return (_mm_cvtsi128_si32(pmin_128) & 0x000000FF);
}

static inline Rpp32u HorMax256(__m256i pmax)
{
    __m128i pmax_128;
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = M256I(pmax).m256i_i128[0];
    pmax_128 = _mm_max_epi8(pmax_128, _mm_shufflelo_epi16(pmax_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = _mm_max_epi8(pmax_128, _mm_srli_epi16(pmax_128, 8));
    return (_mm_cvtsi128_si32(pmax_128) & 0x000000FF);
}
#endif

static  inline __m128 fast_exp_sse (__m128 x)
{
    __m128 t, f, e, p, r;
    __m128i i, j;
    __m128 l2e = _mm_set1_ps (1.442695041f);  /* log2(e) */
    __m128 c0  = _mm_set1_ps (0.3371894346f);
    __m128 c1  = _mm_set1_ps (0.657636276f);
    __m128 c2  = _mm_set1_ps (1.00172476f);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    t = _mm_mul_ps (x, l2e);             /* t = log2(e) * x */
#ifdef __SSE4_1__
    e = _mm_floor_ps (t);                /* floor(t) */
    i = _mm_cvtps_epi32 (e);             /* (int)floor(t) */
#else /* __SSE4_1__*/
    i = _mm_cvttps_epi32 (t);            /* i = (int)t */
    j = _mm_srli_epi32 (_mm_castps_si128 (x), 31); /* signbit(t) */
    i = _mm_sub_epi32 (i, j);            /* (int)t - signbit(t) */
    e = _mm_cvtepi32_ps (i);             /* floor(t) ~= (int)t - signbit(t) */
#endif /* __SSE4_1__*/
    f = _mm_sub_ps (t, e);               /* f = t - floor(t) */
    p = c0;                              /* c0 */
    p = _mm_mul_ps (p, f);               /* c0 * f */
    p = _mm_add_ps (p, c1);              /* c0 * f + c1 */
    p = _mm_mul_ps (p, f);               /* (c0 * f + c1) * f */
    p = _mm_add_ps (p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm_slli_epi32 (i, 23);          /* i << 23 */
    r = _mm_castsi128_ps (_mm_add_epi32 (j, _mm_castps_si128 (p))); /* r = p * 2^i*/
    return r;
}

#if __AVX2__
static inline __m256 fast_exp_avx(__m256 _x)
{
    __m256 c1 = _mm256_set1_ps(0.007972914726F);
    __m256 c2 = _mm256_set1_ps(0.1385283768F);
    __m256 c3 = _mm256_set1_ps(2.885390043F);
    __m256 c4 = _mm256_set1_ps(1.442695022F);
    __m256 x = _mm256_mul_ps(_x, c4); //convert to 2^(x)
    __m256 intPartf = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, intPartf);
    __m256 xx = _mm256_mul_ps(x, x);
    __m256 a = _mm256_add_ps(x, _mm256_mul_ps(c1, _mm256_mul_ps(xx, x))); //can be improved with FMA
    __m256 b = _mm256_add_ps(c3, _mm256_mul_ps(c2, xx));
    __m256 res = _mm256_div_ps(_mm256_add_ps(b, a), _mm256_sub_ps(b, a));
    __m256i intPart = _mm256_cvtps_epi32(intPartf); //res = 2^intPart. Can be improved with AVX2!
    __m128i ii0 = _mm_slli_epi32(_mm256_castsi256_si128(intPart), 23);
    __m128i ii1 = _mm_slli_epi32(_mm256_extractf128_si256(intPart, 1), 23);     
    __m128i res_0 = _mm_add_epi32(ii0, _mm256_castsi256_si128(_mm256_castps_si256(res)));
    __m128i res_1 = _mm_add_epi32(ii1, _mm256_extractf128_si256(_mm256_castps_si256(res), 1));
    return _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(res_0)), _mm_castsi128_ps(res_1), 1);
}
#endif

static inline void sincos_ps(__m256 x, __m256 *s, __m256 *c)
{
    // Extract the sign bit (upper one)
    __m256 sign_bit_sin = _mm256_and_ps(x, _ps_sign_mask_avx);
    // take the absolute value
    x = _mm256_xor_ps(x, sign_bit_sin);

    // Scale by 4/Pi
    __m256 y = _mm256_mul_ps(x, _ps_cephes_FOPI_avx);

    // Store the integer part of y in emm2
    __m256i emm2 = _mm256_cvttps_epi32(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm256_add_epi32(emm2, _pi32_1_avx);
    emm2 = _mm256_and_si256(emm2, _pi32_inv1_avx);
    y = _mm256_cvtepi32_ps(emm2);

    __m256i emm4 = emm2;

    // Get the swap sign flag for the sine
    __m256i emm0 = _mm256_and_si256(emm2, _pi32_4_avx);
    emm0 = _mm256_slli_epi32(emm0, 29);
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(emm0);

    // Get the polynom selection mask for the sine
    emm2 = _mm256_and_si256(emm2, _pi32_2_avx);
    emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());
    __m256 poly_mask = _mm256_castsi256_ps(emm2);
    // The magic pass: "Extended precision modular arithmetic - x = ((x - y * DP1) - y * DP2) - y * DP3;
    __m256 xmm1 = _mm256_mul_ps(y, _ps_minus_cephes_DP1_avx);
    __m256 xmm2 = _mm256_mul_ps(y, _ps_minus_cephes_DP2_avx);
    __m256 xmm3 = _mm256_mul_ps(y, _ps_minus_cephes_DP3_avx);
    x = _mm256_add_ps(_mm256_add_ps(x, xmm1), _mm256_add_ps(xmm2, xmm3));

    emm4 = _mm256_sub_epi32(emm4, _pi32_2_avx);
    emm4 = _mm256_andnot_si256(emm4, _pi32_4_avx);
    emm4 = _mm256_slli_epi32(emm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(emm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    __m256 z = _mm256_mul_ps(x,x);
    y = _ps_coscof_p0_avx;

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps_coscof_p1_avx);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps_coscof_p2_avx);
    y = _mm256_mul_ps(y, _mm256_mul_ps(z, z));
    __m256 tmp = _mm256_mul_ps(z, _ps_0p5_avx);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, avx_p1);

    // Evaluate the second polynom  (Pi/4 <= x <= 0)

    __m256 y2 = _ps_sincof_p0_avx;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps_sincof_p1_avx);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps_sincof_p2_avx);
    y2 = _mm256_mul_ps(y2, _mm256_mul_ps(z, x));
    y2 = _mm256_add_ps(y2, x);

    // Select the correct result from the two polynoms
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2,ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1,ysin2);
    xmm2 = _mm256_add_ps(y,y2);

    // Update the sign
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

static inline void sincos_ps(__m128 x, __m128 *s, __m128 *c)
{
    // Extract the sign bit (upper one)
    __m128 sign_bit_sin = _mm_and_ps(x, _ps_sign_mask);
    // take the absolute value
    x = _mm_xor_ps(x, sign_bit_sin);

    // Scale by 4/Pi
    __m128 y = _mm_mul_ps(x, _ps_cephes_FOPI);

    // Store the integer part of y in emm2
    __m128i emm2 = _mm_cvttps_epi32(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32(emm2, _pi32_1);
    emm2 = _mm_and_si128(emm2, _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    __m128i emm4 = emm2;

    // Get the swap sign flag for the sine
    __m128i emm0 = _mm_and_si128(emm2, _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

    // Get the polynom selection mask for the sine
    emm2 = _mm_and_si128(emm2, _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    // The magic pass: "Extended precision modular arithmetic - x = ((x - y * DP1) - y * DP2) - y * DP3;
    __m128 xmm1 = _mm_mul_ps(y, _ps_minus_cephes_DP1);
    __m128 xmm2 = _mm_mul_ps(y, _ps_minus_cephes_DP2);
    __m128 xmm3 = _mm_mul_ps(y, _ps_minus_cephes_DP3);
    x = _mm_add_ps(_mm_add_ps(x, xmm1), _mm_add_ps(xmm2, xmm3));

    emm4 = _mm_sub_epi32(emm4, _pi32_2);
    emm4 = _mm_andnot_si128(emm4, _pi32_4);
    emm4 = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    __m128 z = _mm_mul_ps(x,x);
    y = _ps_coscof_p0;

    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, _ps_coscof_p1);
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, _ps_coscof_p2);
    y = _mm_mul_ps(y, _mm_mul_ps(z, z));
    __m128 tmp = _mm_mul_ps(z, _ps_0p5);
    y = _mm_sub_ps(y, tmp);
    y = _mm_add_ps(y, _ps_1);

    // Evaluate the second polynom  (Pi/4 <= x <= 0)

    __m128 y2 = _ps_sincof_p0;
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, _ps_sincof_p1);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, _mm_mul_ps(z, x));
    y2 = _mm_add_ps(y2, x);

    // Select the correct result from the two polynoms
    xmm3 = poly_mask;
    __m128 ysin2 = _mm_and_ps(xmm3, y2);
    __m128 ysin1 = _mm_andnot_ps(xmm3, y);
    y2 = _mm_sub_ps(y2,ysin2);
    y = _mm_sub_ps(y, ysin1);

    xmm1 = _mm_add_ps(ysin1,ysin2);
    xmm2 = _mm_add_ps(y,y2);

    // Update the sign
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

static inline __m128 atan_ps( __m128 x )
{
    __m128 sign_bit, y;

    sign_bit = x;
    // Take the absolute value
    x = _mm_and_ps( x, _ps_inv_sign_mask );
    // Extract the sign bit (upper one)
    sign_bit = _mm_and_ps( sign_bit, _ps_sign_mask );

    // Range reduction, init x and y depending on range

    // x > 2.414213562373095
    __m128 cmp0 = _mm_cmpgt_ps( x, _ps_atanrange_hi );
    // x > 0.4142135623730950
    __m128 cmp1 = _mm_cmpgt_ps( x, _ps_atanrange_lo );

    // x > 0.4142135623730950 && !( x > 2.414213562373095 )
    __m128 cmp2 = _mm_andnot_ps( cmp0, cmp1 );

    // -( 1.0/x )
    __m128 y0 = _mm_and_ps( cmp0, _ps_cephes_PIO2F );
    __m128 x0 = _mm_div_ps( _ps_1, x );
    x0 = _mm_xor_ps( x0, _ps_sign_mask );

    __m128 y1 = _mm_and_ps( cmp2, _ps_cephes_PIO4F );
    // (x-1.0)/(x+1.0)
    __m128 x1_o = _mm_sub_ps( x, _ps_1 );
    __m128 x1_u = _mm_add_ps( x, _ps_1 );
    __m128 x1 = _mm_div_ps( x1_o, x1_u );

    __m128 x2 = _mm_and_ps( cmp2, x1 );
    x0 = _mm_and_ps( cmp0, x0 );
    x2 = _mm_or_ps( x2, x0 );
    cmp1 = _mm_or_ps( cmp0, cmp2 );
    x2 = _mm_and_ps( cmp1, x2 );
    x = _mm_andnot_ps( cmp1, x );
    x = _mm_or_ps( x2, x );

    y = _mm_or_ps( y0, y1 );

    __m128 zz = _mm_mul_ps( x, x );
    __m128 acc = _ps_atancof_p0;
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_sub_ps( acc, _ps_atancof_p1 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_add_ps( acc, _ps_atancof_p2 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_sub_ps( acc, _ps_atancof_p3 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_mul_ps( acc, x );
    acc = _mm_add_ps( acc, x );
    y = _mm_add_ps( y, acc );

    // Update the sign
    y = _mm_xor_ps( y, sign_bit );

    return y;
}

static inline __m128 atan2_ps( __m128 y, __m128 x )
{
    __m128 x_eq_0 = _mm_cmpeq_ps( x, _ps_0 );
    __m128 x_gt_0 = _mm_cmpgt_ps( x, _ps_0 );
    __m128 x_le_0 = _mm_cmple_ps( x, _ps_0 );
    __m128 y_eq_0 = _mm_cmpeq_ps( y, _ps_0 );
    __m128 x_lt_0 = _mm_cmplt_ps( x, _ps_0 );
    __m128 y_lt_0 = _mm_cmplt_ps( y, _ps_0 );

    __m128 zero_mask = _mm_and_ps( x_eq_0, y_eq_0 );
    __m128 zero_mask_other_case = _mm_and_ps( y_eq_0, x_gt_0 );
    zero_mask = _mm_or_ps( zero_mask, zero_mask_other_case );

    __m128 pio2_mask = _mm_andnot_ps( y_eq_0, x_eq_0 );
    __m128 pio2_mask_sign = _mm_and_ps( y_lt_0, _ps_sign_mask );
    __m128 pio2_result = _ps_cephes_PIO2F;
    pio2_result = _mm_xor_ps( pio2_result, pio2_mask_sign );
    pio2_result = _mm_and_ps( pio2_mask, pio2_result );

    __m128 pi_mask = _mm_and_ps( y_eq_0, x_le_0 );
    __m128 pi = _ps_cephes_PIF;
    __m128 pi_result = _mm_and_ps( pi_mask, pi );

    __m128 swap_sign_mask_offset = _mm_and_ps( x_lt_0, y_lt_0 );
    swap_sign_mask_offset = _mm_and_ps( swap_sign_mask_offset, _ps_sign_mask );

    __m128 offset0 = _mm_setzero_ps();
    __m128 offset1 = _ps_cephes_PIF;
    offset1 = _mm_xor_ps( offset1, swap_sign_mask_offset );

    __m128 offset = _mm_andnot_ps( x_lt_0, offset0 );
    offset = _mm_and_ps( x_lt_0, offset1 );

    __m128 arg = _mm_div_ps( y, x );
    __m128 atan_result = atan_ps( arg );
    atan_result = _mm_add_ps( atan_result, offset );

    // Select between zero_result, pio2_result and atan_result

    __m128 result = _mm_andnot_ps( zero_mask, pio2_result );
    atan_result = _mm_andnot_ps( pio2_mask, atan_result );
    atan_result = _mm_andnot_ps( pio2_mask, atan_result);
    result = _mm_or_ps( result, atan_result );
    result = _mm_or_ps( result, pi_result );

    return result;
}

// AVX2 version of the atan_ps() SSE version
static inline __m256 atan_ps(__m256 x)
{
    __m256 sign_bit, y;

    sign_bit = x;
    // Take the absolute value
    x = _mm256_and_ps(x, _ps_inv_sign_mask_avx);
    // Extract the sign bit (upper one)
    sign_bit = _mm256_and_ps(sign_bit, _ps_sign_mask_avx);

    // Range reduction, init x and y depending on range
    // x > 2.414213562373095
    __m256 cmp0 = _mm256_cmp_ps(x, _ps_atanrange_hi_avx, _CMP_GT_OS);
    // x > 0.4142135623730950
    __m256 cmp1 = _mm256_cmp_ps(x, _ps_atanrange_lo_avx, _CMP_GT_OS);

    // x > 0.4142135623730950 && !(x > 2.414213562373095)
    __m256 cmp2 = _mm256_andnot_ps(cmp0, cmp1);

    // -(1.0/x);
    __m256 y0 = _mm256_and_ps(cmp0, _ps_cephes_PIO2F_avx);
    __m256 x0 = _mm256_div_ps(avx_p1, x);
    x0 = _mm256_xor_ps(x0, _ps_sign_mask_avx);

    __m256 y1 = _mm256_and_ps(cmp2, _ps_cephes_PIO4F_avx);
    // (x-1.0)/(x+1.0)
    __m256 x1_o = _mm256_sub_ps(x, avx_p1);
    __m256 x1_u = _mm256_add_ps(x, avx_p1);
    __m256 x1 = _mm256_div_ps(x1_o, x1_u);

    __m256 x2 = _mm256_and_ps(cmp2, x1);
    x0 = _mm256_and_ps(cmp0, x0);
    x2 = _mm256_or_ps(x2, x0);
    cmp1 = _mm256_or_ps(cmp0, cmp2);
    x2 = _mm256_and_ps(cmp1, x2);
    x = _mm256_andnot_ps(cmp1, x);
    x = _mm256_or_ps(x2, x);

    y = _mm256_or_ps(y0, y1);

    __m256 zz = _mm256_mul_ps(x, x);
    __m256 acc = _ps_atancof_p0_avx;
    acc = _mm256_fmsub_ps(acc, zz, _ps_atancof_p1_avx);
    acc = _mm256_fmadd_ps(acc, zz, _ps_atancof_p2_avx);
    acc = _mm256_fmsub_ps(acc, zz, _ps_atancof_p3_avx);
    acc = _mm256_mul_ps(acc, zz);
    acc = _mm256_fmadd_ps(acc, x, x);
    y = _mm256_add_ps(y, acc);

    // Update the sign
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

// AVX2 version of the atan2_ps() SSE version
static inline __m256 atan2_ps(__m256 y, __m256 x)
{
    __m256 x_eq_0 = _mm256_cmp_ps(x, avx_p0, _CMP_EQ_OQ);
    __m256 x_gt_0 = _mm256_cmp_ps(x, avx_p0, _CMP_GT_OS);
    __m256 x_le_0 = _mm256_cmp_ps(x, avx_p0, _CMP_LE_OS);
    __m256 y_eq_0 = _mm256_cmp_ps(y, avx_p0, _CMP_EQ_OQ);
    __m256 x_lt_0 = _mm256_cmp_ps(x, avx_p0, _CMP_LT_OS);
    __m256 y_lt_0 = _mm256_cmp_ps(y, avx_p0, _CMP_LT_OS);

    // Computes a zero mask, set if either both x=y=0 or y=0&x>0
    __m256 zero_mask = _mm256_and_ps(x_eq_0, y_eq_0);
    __m256 zero_mask_other_case = _mm256_and_ps(y_eq_0, x_gt_0);
    zero_mask = _mm256_or_ps(zero_mask, zero_mask_other_case);

    // Computes pio2 intermediate result, set if (y!0 and x=0) & (pi/2 XOR (upper bit y<0))
    __m256 pio2_mask = _mm256_andnot_ps(y_eq_0, x_eq_0);
    __m256 pio2_mask_sign = _mm256_and_ps(y_lt_0, _ps_sign_mask_avx);
    __m256 pio2_result = _ps_cephes_PIO2F_avx;
    pio2_result = _mm256_xor_ps(pio2_result, pio2_mask_sign);
    pio2_result = _mm256_and_ps(pio2_mask, pio2_result);

    // Computes pi intermediate result, set if y=0&x<0 and pi
    __m256 pi_mask = _mm256_and_ps(y_eq_0, x_le_0);
    __m256 pi_result = _mm256_and_ps(pi_mask, _ps_cephes_PIF_avx);

    // Computes swap_sign_mask_offset, set if x<0 & y<0 of sign bit(uppermost bit)
    __m256 swap_sign_mask_offset = _mm256_and_ps(x_lt_0, y_lt_0);
    swap_sign_mask_offset = _mm256_and_ps(swap_sign_mask_offset, _ps_sign_mask_avx);

     // Computes offset, set based on pi, swap_sign_mask_offset and x<0
    __m256 offset0 = _mm256_xor_ps(_ps_cephes_PIF_avx, swap_sign_mask_offset);
    __m256 offset = _mm256_andnot_ps(x_lt_0, avx_p0);
    offset = _mm256_and_ps(x_lt_0, offset0);

    // Computes division of x,y
    __m256 arg = _mm256_div_ps(y, x);
    __m256 atan_result = atan_ps(arg);
    atan_result = _mm256_add_ps(atan_result, offset);

    // Select between zero_result, pio2_result and atan_result
    __m256 result = _mm256_andnot_ps(zero_mask, pio2_result);
    atan_result = _mm256_andnot_ps(pio2_mask, atan_result);
    atan_result = _mm256_andnot_ps(pio2_mask, atan_result);
    result = _mm256_or_ps(result, atan_result);
    result = _mm256_or_ps(result, pi_result);

    return result;
}

// Modified AVX2 version of the original SSE version at https://github.com/RJVB/sse_mathfun/blob/master/sse_mathfun.h
static inline __m256 log_ps(__m256 x)
{
    __m256 e;
    __m256i emm0;
    __m256 one = *(__m256 *)&avx_p1;
    __m256 invalid_mask = _mm256_cmp_ps(x, avx_p0, _CMP_LE_OQ);

    // cut off denormalized stuff
    x = _mm256_max_ps(x, *(__m256 *)&_ps_min_norm_pos_avx);

    // part 1: x = frexpf(x, &e);
    emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    // keep only the fractional part
    x = _mm256_and_ps(x, *(__m256 *)&_ps_inv_mant_mask_avx);
    x = _mm256_or_ps(x, *(__m256 *)&_ps_0p5_avx);

    emm0 = _mm256_sub_epi32(emm0, *(__m256i *)&_pi32_0x7f_avx);
    e = _mm256_cvtepi32_ps(emm0);

    e = _mm256_add_ps(e, one);

    // part 2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; }
    __m256 z, y;
    __m256 mask = _mm256_cmp_ps(x, avx_cephesSQRTHF, _CMP_LT_OQ);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);
    z = _mm256_mul_ps(x,x);
    y = avx_cephesLogP0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, avx_cephesLogP8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    tmp = _mm256_mul_ps(e, avx_cephesLogQ1);
    y = _mm256_add_ps(y, tmp);
    tmp = _mm256_mul_ps(z, *(__m256 *)&_ps_0p5_avx);
    y = _mm256_sub_ps(y, tmp);
    tmp = _mm256_mul_ps(e, avx_cephesLogQ2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN

    return x;
}

// Modified version of the original SSE version at https://github.com/RJVB/sse_mathfun/blob/master/sse_mathfun.h
static inline __m128 log_ps(__m128 x)
{
    __m128 e;
    __m128i emm0;
    __m128 one = *(__m128 *)&_ps_1;
    __m128 invalid_mask = _mm_cmple_ps(x, xmm_p0);

    // cut off denormalized stuff
    x = _mm_max_ps(x, *(__m128 *)&_ps_min_norm_pos);

    // part 1: x = frexpf(x, &e);
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    // keep only the fractional part
    x = _mm_and_ps(x, *(__m128 *)&_ps_inv_mant_mask);
    x = _mm_or_ps(x, *(__m128 *)&_ps_0p5);

    emm0 = _mm_sub_epi32(emm0, *(__m128i *)&_pi32_0x7f);
    e = _mm_cvtepi32_ps(emm0);

    e = _mm_add_ps(e, one);

    // part 2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; }
    __m128 z, y;
    __m128 mask = _mm_cmplt_ps(x, *(__m128 *)&xmm_cephesSQRTHF);
    __m128 tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);
    z = _mm_mul_ps(x,x);
    y = *(__m128 *)&xmm_cephesLogP0;
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP1);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP2);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP3);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP4);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP5);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP6);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP7);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP8);
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);
    tmp = _mm_mul_ps(e, *(__m128 *)&xmm_cephesLogQ1);
    y = _mm_add_ps(y, tmp);
    tmp = _mm_mul_ps(z, *(__m128 *)&_ps_0p5);
    y = _mm_sub_ps(y, tmp);
    tmp = _mm_mul_ps(e, *(__m128 *)&xmm_cephesLogQ2);
    x = _mm_add_ps(x, y);
    x = _mm_add_ps(x, tmp);
    x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN

    return x;
}

inline void compute_log_16_host(__m256 *p)
{
    p[0] = log_ps(p[0]);    // log compute
    p[1] = log_ps(p[1]);    // log compute
}

inline Rpp32f rpp_hsum_ps(__m128 x)
{
    __m128 shuf = _mm_movehdup_ps(x);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);        // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline Rpp32f rpp_hsum_ps(__m256 x)
{
    __m128 p0 = _mm256_extractf128_ps(x, 1); // Contains x7, x6, x5, x4
    __m128 p1 = _mm256_castps256_ps128(x);   // Contains x3, x2, x1, x0
    __m128 sum = _mm_add_ps(p0, p1);         // Contains x3 + x7, x2 + x6, x1 + x5, x0 + x4
    p0 = sum;                                // Contains -, -, x1 + x5, x0 + x4
    p1 = _mm_movehl_ps(sum, sum);            // Contains -, -, x3 + x7, x2 + x6
    sum = _mm_add_ps(p0, p1);                // Contains -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6
    p0 = sum;                                // Contains -, -, -, x0 + x2 + x4 + x6
    p1 = _mm_shuffle_ps(sum, sum, 0x1);      // Contains -, -, -, x1 + x3 + x5 + x7
    sum = _mm_add_ss(p0, p1);                // Contains -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    return _mm_cvtss_f32(sum);
}

/* Computes inverse square root */
inline Rpp32f rpp_rsqrt_ps(Rpp32f x)
{
    __m128 X = _mm_set_ss(x);
    __m128 tmp = _mm_rsqrt_ss(X);
    Rpp32f y = _mm_cvtss_f32(tmp);
    return y * (1.5f - x * 0.5f * y * y);
}

/* Compute inverse square root */
/* SSE matches to 6 decimal places with raw C version due to newton rhapson approximation*/
inline void rpp_rsqrt_sse(Rpp32f *input, Rpp64s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f mul)
{
    Rpp64s i = 0;
    __m128 rdivx4 = _mm_set1_ps(rdiv);
    __m128 mulx4 = _mm_set1_ps(mul * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m128 epsx4 = _mm_set1_ps(eps);
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            x = _mm_add_ps(x, epsx4);
            __m128 y = _mm_rsqrt_ps(x);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            __m128 mask = _mm_cmpneq_ps(x, xmm_p0);
            __m128 y = _mm_rsqrt_ps(x);
            y = _mm_and_ps(y, mask);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt_ps(input[i] * rdiv + eps) * mul;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt_ps(x) * mul : 0;
        }
    }
}

/* Compute inverse square root */
/* AVX2 matches to 6 decimal places with raw C version due to newton rhapson approximation*/
inline void rpp_rsqrt_avx(Rpp32f *input, Rpp32s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f scale)
{
    Rpp32s i = 0;
    __m256 rdivx8 = _mm256_set1_ps(rdiv);
    __m256 mulx8 = _mm256_set1_ps(scale * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m256 epsx8 = _mm256_set1_ps(eps);
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            x = _mm256_add_ps(x, epsx8);
            __m256 y = _mm256_rsqrt_ps(x);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            __m256 mask = _mm256_cmp_ps(x, avx_p0, _CMP_NEQ_OQ);
            __m256 y = _mm256_rsqrt_ps(x);
            y = _mm256_and_ps(y, mask);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt_ps(input[i] * rdiv + eps) * scale;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt_ps(x) * scale : 0;
        }
    }
}

static inline void fast_matmul4x4_sse(float *A, float *B, float *C)
{
    __m128 row1 = _mm_load_ps(&B[0]);                   // Row 0 of B
    __m128 row2 = _mm_load_ps(&B[4]);                   // Row 1 of B
    __m128 row3 = _mm_load_ps(&B[8]);                   // Row 2 of B
    __m128 row4 = _mm_load_ps(&B[12]);                  // Row 3 of B

    for(int i = 0; i < 4; i++)
    {
        __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);       // Example for row 0 computation -> A[0][0] is broadcasted
        __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);       // Example for row 0 computation -> A[0][1] is broadcasted
        __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);       // Example for row 0 computation -> A[0][2] is broadcasted
        __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);       // Example for row 0 computation -> A[0][3] is broadcasted

        __m128 row = _mm_add_ps(                        // Example for row 0 computation -> P + Q
                        _mm_add_ps(                     // Example for row 0 computation -> P = A[0][0] * B[0][0] + A[0][1] * B[1][0]
                            _mm_mul_ps(brod1, row1),    // Example for row 0 computation -> (A[0][0] * B[0][0], A[0][0] * B[0][1], A[0][0] * B[0][2], A[0][0] * B[0][3])
                            _mm_mul_ps(brod2, row2)),   // Example for row 0 computation -> (A[0][1] * B[1][0], A[0][1] * B[1][1], A[0][1] * B[1][2], A[0][1] * B[1][3])
                        _mm_add_ps(                     // Example for row 0 computation -> Q = A[0][2] * B[2][0] + A[0][3] * B[3][0]
                            _mm_mul_ps(brod3, row3),    // Example for row 0 computation -> (A[0][2] * B[2][0], A[0][2] * B[2][1], A[0][2] * B[2][2], A[0][2] * B[2][3])
                            _mm_mul_ps(brod4, row4)));  // Example for row 0 computation -> (A[0][3] * B[3][0], A[0][3] * B[3][1], A[0][3] * B[3][2], A[0][3] * B[3][3])

        _mm_store_ps(&C[4*i], row);                     // Example for row 0 computation -> Storing whole computed row 0
    }
}

// SSE implementation of fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline __m128 rpp_host_math_inverse_sqrt_4_sse(__m128 p)
{
    __m128 pHalfNeg;
    __m128i pxI;
    pHalfNeg = _mm_mul_ps(_ps_n0p5, p);                                         // float xHalfNeg = -0.5f * x;
    pxI = *(__m128i *)&p;                                                       // int i = *(int*)&x;
    pxI = _mm_sub_epi32(xmm_newtonMethodInitialGuess, _mm_srli_epi32(pxI, 1));  // i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);
    p = *(__m128 *)&pxI;                                                        // x = *(float*)&i;
    p = _mm_mul_ps(p, _mm_fmadd_ps(p, _mm_mul_ps(p, pHalfNeg), _ps_1p5));       // x = x * (1.5f - xHalf * x * x);

    return p;
}

// AVX2 implementation of fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline __m256 rpp_host_math_inverse_sqrt_8_avx(__m256 p)
{
    __m256 pHalfNeg;
    __m256i pxI;
    pHalfNeg = _mm256_mul_ps(_ps_n0p5_avx, p);                                          // float xHalfNeg = -0.5f * x;
    pxI = *(__m256i *)&p;                                                               // int i = *(int*)&x;
    pxI = _mm256_sub_epi32(avx_newtonMethodInitialGuess, _mm256_srli_epi32(pxI, 1));    // i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);
    p = *(__m256 *)&pxI;                                                                // x = *(float*)&i;
    p = _mm256_mul_ps(p, _mm256_fmadd_ps(p, _mm256_mul_ps(p, pHalfNeg), _ps_1p5_avx));  // x = x * (1.5f - xHalf * x * x);

    return p;
}

#endif // RPP_CPU_SIMD_MATH_HPP
