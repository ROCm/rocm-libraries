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

#ifndef RPP_HIP_RANDOM_HPP
#define RPP_HIP_RANDOM_HPP

// /******************** DEVICE RANDOMIZATION HELPER FUNCTIONS ********************/

#define XORWOW_COUNTER_INC              0x587C5             // Hex 0x587C5 = Dec 362437U - xorwow counter increment
#define XORWOW_EXPONENT_MASK            0x3F800000          // Hex 0x3F800000 = Bin 0b111111100000000000000000000000 - 23 bits of mantissa set to 0, 01111111 for the exponent, 0 for the sign bit
#define RPP_2POW32_INV                  2.3283064e-10f      // (1 / 2^32)
#define RPP_2POW32_INV_DIV_2            1.164153218e-10f    // RPP_2POW32_INV / 2
#define RPP_2POW32_INV_MUL_2PI          1.46291812e-09f     // (1 / 2^32) * 2PI
#define RPP_2POW32_INV_MUL_2PI_DIV_2    7.3145906e-10f      // RPP_2POW32_INV_MUL_2PI / 2

#ifndef RPP_HIP_MATH_DEPENDENCIES
#define RPP_HIP_MATH_DEPENDENCIES
__device__ __forceinline__ float rpp_hip_math_inverse_sqrt1(float x)
{
    float xHalf = 0.5f * x;
    int i = *(int*)&x;                              // float bits in int
    i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);     // initial guess for Newton's method
    x = *(float*)&i;                                // new bits to float
    x = x * (1.5f - xHalf * x * x);                 // One round of Newton's method

    return x;
}

__device__ __forceinline__ float rpp_hip_math_exp_lim256approx(float x)
{
  x = 1.0 + x * ONE_OVER_256;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;

  return x;
}
#endif

template<typename T>
__device__ __forceinline__ void rpp_hip_rng_xorwow_state_update(T *xorwowState)
{
    // Save current first and last x-params of xorwow state and compute t
    uint t  = xorwowState->x[0];
    uint s  = xorwowState->x[4];
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
__device__ __forceinline__ uint rpp_hip_rng_xorwow_u32(T *xorwowState)
{
    // Update xorwow state
    rpp_hip_rng_xorwow_state_update(xorwowState);

    // Return u32 random number
    return  xorwowState->x[4] + xorwowState->counter;   // return x[4] + counter
}

template<typename T>
__device__ __forceinline__ float rpp_hip_rng_xorwow_f32(T *xorwowState)
{
    // Update xorwow state
    rpp_hip_rng_xorwow_state_update(xorwowState);

    // Create float representation and return 0 <= outFloat < 1
    xorwowState->counter &= 0xFFFFFFFF;                                                             // set new state param counter
    uint out = (XORWOW_EXPONENT_MASK | ((xorwowState->x[4] + xorwowState->counter) & 0x7FFFFF));    // bitmask 23 mantissa bits, OR with exponent
    float outFloat = *(float *)&out;                                                                // reinterpret out as float
    return  outFloat - 1;                                                                           // return 0 <= outFloat < 1
}

template<typename T>
__device__ __forceinline__ void rpp_hip_rng_8_xorwow_f32(T *xorwowState, d_float8 *randomNumbersPtr_f8)
{
    randomNumbersPtr_f8->f1[0] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[1] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[2] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[3] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[4] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[5] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[6] = rpp_hip_rng_xorwow_f32(xorwowState);
    randomNumbersPtr_f8->f1[7] = rpp_hip_rng_xorwow_f32(xorwowState);
}

__device__ __forceinline__ float rpp_hip_rng_1_gaussian_f32(RpptXorwowStateBoxMuller *xorwowState)
{
    if(xorwowState->boxMullerFlag == 0)
    {
        float2 result_f2;
        uint x = rpp_hip_rng_xorwow_u32(xorwowState);
        uint y = rpp_hip_rng_xorwow_u32(xorwowState);
        float u = x * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
        float v = y * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
        float s = 1 / rpp_hip_math_inverse_sqrt1(-2.0f * __logf(u));
        __sincosf(v, &result_f2.x, &result_f2.y);
        result_f2 *= MAKE_FLOAT2(s);
        xorwowState->boxMullerExtra = result_f2.y;
        xorwowState->boxMullerFlag = 1;
        return result_f2.x;
    }
    xorwowState->boxMullerFlag = 0;
    return xorwowState->boxMullerExtra;
}

__device__ __forceinline__ float2 rpp_hip_rng_2_gaussian_f32(RpptXorwowStateBoxMuller *xorwowState)
{
    float2 result_f2;
    uint x = rpp_hip_rng_xorwow_u32(xorwowState);
    uint y = rpp_hip_rng_xorwow_u32(xorwowState);
    float u = x * RPP_2POW32_INV + RPP_2POW32_INV_DIV_2;
    float v = y * RPP_2POW32_INV_MUL_2PI + RPP_2POW32_INV_MUL_2PI_DIV_2;
    float s = 1 / rpp_hip_math_inverse_sqrt1(-2.0f * __logf(u));
    __sincosf(v, &result_f2.x, &result_f2.y);
    result_f2 *= MAKE_FLOAT2(s);

    return result_f2;
}

__device__ void rpp_hip_rng_8_gaussian_f32(d_float8 *rngVals_f8, RpptXorwowStateBoxMuller *xorwowState)
{
    rngVals_f8->f2[0] = rpp_hip_rng_2_gaussian_f32(xorwowState);
    rngVals_f8->f2[1] = rpp_hip_rng_2_gaussian_f32(xorwowState);
    rngVals_f8->f2[2] = rpp_hip_rng_2_gaussian_f32(xorwowState);
    rngVals_f8->f2[3] = rpp_hip_rng_2_gaussian_f32(xorwowState);
}

__device__ __forceinline__ float rpp_hip_rng_1_inverse_transform_sampling_f32(float lambdaValue, RpptXorwowStateBoxMuller *xorwowState)
{
    float shotNoiseValue = 0;
    float factValue = rpp_hip_math_exp_lim256approx(-lambdaValue);
    float sumValue = factValue;
    float randomNumber;
    do
    {
        randomNumber = rpp_hip_rng_xorwow_f32(xorwowState);
    } while(randomNumber > 0.95f);
    while (randomNumber > sumValue)
    {
        shotNoiseValue += 1;
        factValue *= (lambdaValue / shotNoiseValue);
        sumValue += factValue;
    }

    return shotNoiseValue;
}

#endif // RPP_HIP_RANDOM_HPP
