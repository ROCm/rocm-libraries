/*******************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef ROCFFT_BUTTERFLY_TEMPLATE_H
#define ROCFFT_BUTTERFLY_TEMPLATE_H

template <typename T, size_t Base, size_t Steps>
__device__ T TW_NSteps(const T* const twiddles, size_t u)
{
    size_t j      = u & ((1 << Base) - 1); // get the lowest Base bits
    T      result = twiddles[j];
    u >>= Base; // discard the lowest Base bits
    int i = 0;
    // static compiled, currently, steps can only be 2 or 3
    if(Steps >= 2)
    {
        i += 1;
        j      = u & ((1 << Base) - 1);
        result = T((result.x * twiddles[(1 << Base) * i + j].x
                    - result.y * twiddles[(1 << Base) * i + j].y),
                   (result.y * twiddles[(1 << Base) * i + j].x
                    + result.x * twiddles[(1 << Base) * i + j].y));
    }
    // static compiled
    if(Steps >= 3)
    {
        u >>= Base; // discard the lowest Base bits

        i += 1;
        j      = u & ((1 << Base) - 1);
        result = T((result.x * twiddles[(1 << Base) * i + j].x
                    - result.y * twiddles[(1 << Base) * i + j].y),
                   (result.y * twiddles[(1 << Base) * i + j].x
                    + result.x * twiddles[(1 << Base) * i + j].y));
    }
    // we probably don't have 4-steps for large-twiddle
    // if(Steps >= 4){...}

    return result;
}

template <typename T>
__device__ T TW3step(const T* const twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];

    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
               (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));

    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
               (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
__device__ void FwdRad2B1(T* R0, T* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void InvRad2B1(T* R0, T* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void FwdRad3B1(T* R0, T* R1, T* R2)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2;

    TR0 = (*R0).x + (*R1).x + (*R2).x;
    TR1 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) + C3QB * ((*R1).y - (*R2).y);
    TR2 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) - C3QB * ((*R1).y - (*R2).y);

    TI0 = (*R0).y + (*R1).y + (*R2).y;
    TI1 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) - C3QB * ((*R1).x - (*R2).x);
    TI2 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) + C3QB * ((*R1).x - (*R2).x);

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
}

template <typename T>
__device__ void InvRad3B1(T* R0, T* R1, T* R2)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2;

    TR0 = (*R0).x + (*R1).x + (*R2).x;
    TR1 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) - C3QB * ((*R1).y - (*R2).y);
    TR2 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) + C3QB * ((*R1).y - (*R2).y);

    TI0 = (*R0).y + (*R1).y + (*R2).y;
    TI1 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) + C3QB * ((*R1).x - (*R2).x);
    TI2 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) - C3QB * ((*R1).x - (*R2).x);

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
}

template <typename T>
__device__ void FwdRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);

    (*R3) = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void InvRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void FwdRad5B1(T* R0, T* R1, T* R2, T* R3, T* R4)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4;

    TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;
    TR1 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) + C5QB * ((*R1).y - (*R4).y)
          + C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) - C5QB * ((*R1).y - (*R4).y)
          - C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR2 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) - C5QB * ((*R2).y - (*R3).y)
          + C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));
    TR3 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) + C5QB * ((*R2).y - (*R3).y)
          - C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));

    TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;
    TI1 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) - C5QB * ((*R1).x - (*R4).x)
          - C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) + C5QB * ((*R1).x - (*R4).x)
          + C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI2 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) + C5QB * ((*R2).x - (*R3).x)
          - C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));
    TI3 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) - C5QB * ((*R2).x - (*R3).x)
          + C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
    ((*R3).x) = TR3;
    ((*R3).y) = TI3;
    ((*R4).x) = TR4;
    ((*R4).y) = TI4;
}

template <typename T>
__device__ void InvRad5B1(T* R0, T* R1, T* R2, T* R3, T* R4)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4;

    TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;
    TR1 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) - C5QB * ((*R1).y - (*R4).y)
          - C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) + C5QB * ((*R1).y - (*R4).y)
          + C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR2 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) + C5QB * ((*R2).y - (*R3).y)
          - C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));
    TR3 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) - C5QB * ((*R2).y - (*R3).y)
          + C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));

    TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;
    TI1 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) + C5QB * ((*R1).x - (*R4).x)
          + C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) - C5QB * ((*R1).x - (*R4).x)
          - C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI2 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) - C5QB * ((*R2).x - (*R3).x)
          + C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));
    TI3 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) + C5QB * ((*R2).x - (*R3).x)
          - C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
    ((*R3).x) = TR3;
    ((*R3).y) = TI3;
    ((*R4).x) = TR4;
    ((*R4).y) = TI4;
}

template <typename T>
__device__ void FwdRad6B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5;

    TR0 = (*R0).x + (*R2).x + (*R4).x;
    TR2 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) + C3QB * ((*R2).y - (*R4).y);
    TR4 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) - C3QB * ((*R2).y - (*R4).y);

    TI0 = (*R0).y + (*R2).y + (*R4).y;
    TI2 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) - C3QB * ((*R2).x - (*R4).x);
    TI4 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) + C3QB * ((*R2).x - (*R4).x);

    TR1 = (*R1).x + (*R3).x + (*R5).x;
    TR3 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) + C3QB * ((*R3).y - (*R5).y);
    TR5 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) - C3QB * ((*R3).y - (*R5).y);

    TI1 = (*R1).y + (*R3).y + (*R5).y;
    TI3 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) - C3QB * ((*R3).x - (*R5).x);
    TI5 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) + C3QB * ((*R3).x - (*R5).x);

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C3QA * TR3 + C3QB * TI3);
    (*R2).x = TR4 + (-C3QA * TR5 + C3QB * TI5);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (-C3QB * TR3 + C3QA * TI3);
    (*R2).y = TI4 + (-C3QB * TR5 - C3QA * TI5);

    (*R3).x = TR0 - TR1;
    (*R4).x = TR2 - (C3QA * TR3 + C3QB * TI3);
    (*R5).x = TR4 - (-C3QA * TR5 + C3QB * TI5);

    (*R3).y = TI0 - TI1;
    (*R4).y = TI2 - (-C3QB * TR3 + C3QA * TI3);
    (*R5).y = TI4 - (-C3QB * TR5 - C3QA * TI5);
}

template <typename T>
__device__ void InvRad6B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5;

    TR0 = (*R0).x + (*R2).x + (*R4).x;
    TR2 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) - C3QB * ((*R2).y - (*R4).y);
    TR4 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) + C3QB * ((*R2).y - (*R4).y);

    TI0 = (*R0).y + (*R2).y + (*R4).y;
    TI2 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) + C3QB * ((*R2).x - (*R4).x);
    TI4 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) - C3QB * ((*R2).x - (*R4).x);

    TR1 = (*R1).x + (*R3).x + (*R5).x;
    TR3 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) - C3QB * ((*R3).y - (*R5).y);
    TR5 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) + C3QB * ((*R3).y - (*R5).y);

    TI1 = (*R1).y + (*R3).y + (*R5).y;
    TI3 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) + C3QB * ((*R3).x - (*R5).x);
    TI5 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) - C3QB * ((*R3).x - (*R5).x);

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C3QA * TR3 - C3QB * TI3);
    (*R2).x = TR4 + (-C3QA * TR5 - C3QB * TI5);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (C3QB * TR3 + C3QA * TI3);
    (*R2).y = TI4 + (C3QB * TR5 - C3QA * TI5);

    (*R3).x = TR0 - TR1;
    (*R4).x = TR2 - (C3QA * TR3 - C3QB * TI3);
    (*R5).x = TR4 - (-C3QA * TR5 - C3QB * TI5);

    (*R3).y = TI0 - TI1;
    (*R4).y = TI2 - (C3QB * TR3 + C3QA * TI3);
    (*R5).y = TI4 - (C3QB * TR5 - C3QA * TI5);
}

template <typename T>
__device__ void FwdRad7B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6)
{

    T p0;
    T p1;
    T p2;
    T p3;
    T p4;
    T p5;
    T p6;
    T p7;
    T p8;
    T p9;
    T q0;
    T q1;
    T q2;
    T q3;
    T q4;
    T q5;
    T q6;
    T q7;
    T q8;
    /*FFT7 Forward Complex */

    p0 = *R1 + *R6;
    p1 = *R1 - *R6;
    p2 = *R2 + *R5;
    p3 = *R2 - *R5;
    p4 = *R4 + *R3;
    p5 = *R4 - *R3;

    p6 = p2 + p0;
    q4 = p2 - p0;
    q2 = p0 - p4;
    q3 = p4 - p2;
    p7 = p5 + p3;
    q7 = p5 - p3;
    q6 = p1 - p5;
    q8 = p3 - p1;
    q1 = p6 + p4;
    q5 = p7 + p1;
    q0 = *R0 + q1;

    q1 *= C7Q1;
    q2 *= C7Q2;
    q3 *= C7Q3;
    q4 *= C7Q4;

    q5 *= (C7Q5);
    q6 *= (C7Q6);
    q7 *= (C7Q7);
    q8 *= (C7Q8);

    p0 = q0 + q1;
    p1 = q2 + q3;
    p2 = q4 - q3;
    p3 = -q2 - q4;
    p4 = q6 + q7;
    p5 = q8 - q7;
    p6 = -q8 - q6;
    p7 = p0 + p1;
    p8 = p0 + p2;
    p9 = p0 + p3;
    q6 = p4 + q5;
    q7 = p5 + q5;
    q8 = p6 + q5;

    *R0     = q0;
    (*R1).x = p7.x + q6.y;
    (*R1).y = p7.y - q6.x;
    (*R2).x = p9.x + q8.y;
    (*R2).y = p9.y - q8.x;
    (*R3).x = p8.x - q7.y;
    (*R3).y = p8.y + q7.x;
    (*R4).x = p8.x + q7.y;
    (*R4).y = p8.y - q7.x;
    (*R5).x = p9.x - q8.y;
    (*R5).y = p9.y + q8.x;
    (*R6).x = p7.x - q6.y;
    (*R6).y = p7.y + q6.x;
}

template <typename T>
__device__ void InvRad7B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6)
{

    T p0;
    T p1;
    T p2;
    T p3;
    T p4;
    T p5;
    T p6;
    T p7;
    T p8;
    T p9;
    T q0;
    T q1;
    T q2;
    T q3;
    T q4;
    T q5;
    T q6;
    T q7;
    T q8;
    /*FFT7 Backward Complex */

    p0 = *R1 + *R6;
    p1 = *R1 - *R6;
    p2 = *R2 + *R5;
    p3 = *R2 - *R5;
    p4 = *R4 + *R3;
    p5 = *R4 - *R3;

    p6 = p2 + p0;
    q4 = p2 - p0;
    q2 = p0 - p4;
    q3 = p4 - p2;
    p7 = p5 + p3;
    q7 = p5 - p3;
    q6 = p1 - p5;
    q8 = p3 - p1;
    q1 = p6 + p4;
    q5 = p7 + p1;
    q0 = *R0 + q1;

    q1 *= C7Q1;
    q2 *= C7Q2;
    q3 *= C7Q3;
    q4 *= C7Q4;

    q5 *= -(C7Q5);
    q6 *= -(C7Q6);
    q7 *= -(C7Q7);
    q8 *= -(C7Q8);

    p0 = q0 + q1;
    p1 = q2 + q3;
    p2 = q4 - q3;
    p3 = -q2 - q4;
    p4 = q6 + q7;
    p5 = q8 - q7;
    p6 = -q8 - q6;
    p7 = p0 + p1;
    p8 = p0 + p2;
    p9 = p0 + p3;
    q6 = p4 + q5;
    q7 = p5 + q5;
    q8 = p6 + q5;

    *R0     = q0;
    (*R1).x = p7.x + q6.y;
    (*R1).y = p7.y - q6.x;
    (*R2).x = p9.x + q8.y;
    (*R2).y = p9.y - q8.x;
    (*R3).x = p8.x - q7.y;
    (*R3).y = p8.y + q7.x;
    (*R4).x = p8.x + q7.y;
    (*R4).y = p8.y - q7.x;
    (*R5).x = p9.x - q8.y;
    (*R5).y = p9.y + q8.x;
    (*R6).x = p7.x - q6.y;
    (*R6).y = p7.y + q6.x;
}

template <typename T>
__device__ void FwdRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + T(-(*R7).y, (*R7).x);

    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * T((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + T(-(*R6).y, (*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * T((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void InvRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + T((*R7).y, -(*R7).x);
    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) + C8Q * T((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + T((*R6).y, -(*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) + C8Q * T((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void FwdRad9B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8)
{
    // p2 is always multiplied by C9QF, so do it once in p2
    // update R0 and the end since the original R0 is used by others
    // we can also use v3 = R4 + R5 and p3 = R4 - R5
    // but it's ok to do without them and save regs
    T v0 = (*R1) + (*R8);
    T v1 = (*R2) + (*R7);
    T v2 = (*R3) + (*R6);

    T p0 = (*R1) - (*R8);
    T p1 = (*R2) - (*R7);
    T p2 = ((*R3) - (*R6)) * C9QF;

    // borrow R8 as temp
    (*R8) = (C9QB * p0) + (C9QD * p1) + (p2) + (C9QH * ((*R4) - (*R5)));
    (*R1) = ((*R0) + (C9QA * v0) + (C9QC * v1) - (C9QE * v2) - (C9QG * ((*R4) + (*R5))))
            + T((*R8).y, -(*R8).x);
    (*R8) = (*R1) + 2.0 * T(-(*R8).y, (*R8).x);
    // borrow R7 as temp
    (*R7) = -(C9QB * ((*R4) - (*R5))) + (C9QD * p0) - (p2) + (C9QH * p1);
    (*R2) = ((*R0) + (C9QA * ((*R4) + (*R5))) + (C9QC * v0) - (C9QE * v2) - (C9QG * v1))
            + T((*R7).y, -(*R7).x);
    (*R7) = (*R2) + 2.0 * T(-(*R7).y, (*R7).x);
    // borrow R6 temp
    (*R6) = C9QF * (p0 + ((*R4) - (*R5)) - p1);
    (*R3) = ((*R0) + v2 - C9QE * (v0 + v1 + ((*R4) + (*R5)))) + T((*R6).y, -(*R6).x);
    (*R6) = (*R3) + 2.0 * T(-(*R6).y, (*R6).x);
    // borrow p0 as temp
    p0 = -(C9QB * p1) - (C9QD * ((*R4) - (*R5))) + (p2) + (C9QH * p0);
    p1 = (*R0);
    (*R0) += (v0 + v1 + v2 + (*R4) + (*R5));
    (*R4) = (p1 + (C9QA * v1) + (C9QC * ((*R4) + (*R5))) - (C9QE * v2) - (C9QG * v0))
            + T(p0.y, -p0.x);
    (*R5) = (*R4) + 2.0 * T(-p0.y, p0.x);
}

template <typename T>
__device__ void InvRad9B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8)
{
    // p2 is always multiplied by C9QF, so do it once in p2
    // update R0 and the end since the original R0 is used by others
    T v0 = (*R1) + (*R8);
    T v1 = (*R2) + (*R7);
    T v2 = (*R3) + (*R6);

    T p0 = (*R1) - (*R8);
    T p1 = (*R2) - (*R7);
    T p2 = ((*R3) - (*R6)) * C9QF;

    // borrow R8 as temp
    (*R8) = (C9QB * p0) + (C9QD * p1) + (p2) + (C9QH * ((*R4) - (*R5)));
    (*R1) = ((*R0) + (C9QA * v0) + (C9QC * v1) - (C9QE * v2) - (C9QG * ((*R4) + (*R5))))
            + T(-(*R8).y, (*R8).x);
    (*R8) = (*R1) + 2.0 * T((*R8).y, -(*R8).x);
    // borrow R7 as temp
    (*R7) = -(C9QB * ((*R4) - (*R5))) + (C9QD * p0) - (p2) + (C9QH * p1);
    (*R2) = ((*R0) + (C9QA * ((*R4) + (*R5))) + (C9QC * v0) - (C9QE * v2) - (C9QG * v1))
            + T(-(*R7).y, (*R7).x);
    (*R7) = (*R2) + 2.0 * T((*R7).y, -(*R7).x);
    // borrow R6 temp
    (*R6) = C9QF * (p0 + ((*R4) - (*R5)) - p1);
    (*R3) = ((*R0) + v2 - C9QE * (v0 + v1 + ((*R4) + (*R5)))) + T(-(*R6).y, (*R6).x);
    (*R6) = (*R3) + 2.0 * T((*R6).y, -(*R6).x);
    // borrow p0 as temp
    p0 = -(C9QB * p1) - (C9QD * ((*R4) - (*R5))) + (p2) + (C9QH * p0);
    p1 = (*R0);
    (*R0) += (v0 + v1 + v2 + (*R4) + (*R5));
    (*R4) = (p1 + (C9QA * v1) + (C9QC * ((*R4) + (*R5))) - (C9QE * v2) - (C9QG * v0))
            + T(-p0.y, p0.x);
    (*R5) = (*R4) + 2.0 * T(p0.y, -p0.x);
}

template <typename T>
__device__ void FwdRad10B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5, TR6, TI6, TR7, TI7,
        TR8, TI8, TR9, TI9;

    TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;
    TR2 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) + C5QB * ((*R2).y - (*R8).y)
          + C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR8 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) - C5QB * ((*R2).y - (*R8).y)
          - C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) - C5QB * ((*R4).y - (*R6).y)
          + C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));
    TR6 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) + C5QB * ((*R4).y - (*R6).y)
          - C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));

    TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;
    TI2 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) - C5QB * ((*R2).x - (*R8).x)
          - C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI8 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) + C5QB * ((*R2).x - (*R8).x)
          + C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) + C5QB * ((*R4).x - (*R6).x)
          - C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));
    TI6 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) - C5QB * ((*R4).x - (*R6).x)
          + C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));

    TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;
    TR3 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) + C5QB * ((*R3).y - (*R9).y)
          + C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR9 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) - C5QB * ((*R3).y - (*R9).y)
          - C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR5 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) - C5QB * ((*R5).y - (*R7).y)
          + C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));
    TR7 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) + C5QB * ((*R5).y - (*R7).y)
          - C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));

    TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;
    TI3 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) - C5QB * ((*R3).x - (*R9).x)
          - C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI9 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) + C5QB * ((*R3).x - (*R9).x)
          + C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI5 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) + C5QB * ((*R5).x - (*R7).x)
          - C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));
    TI7 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) - C5QB * ((*R5).x - (*R7).x)
          + C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C5QE * TR3 + C5QD * TI3);
    (*R2).x = TR4 + (C5QA * TR5 + C5QB * TI5);
    (*R3).x = TR6 + (-C5QA * TR7 + C5QB * TI7);
    (*R4).x = TR8 + (-C5QE * TR9 + C5QD * TI9);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (-C5QD * TR3 + C5QE * TI3);
    (*R2).y = TI4 + (-C5QB * TR5 + C5QA * TI5);
    (*R3).y = TI6 + (-C5QB * TR7 - C5QA * TI7);
    (*R4).y = TI8 + (-C5QD * TR9 - C5QE * TI9);

    (*R5).x = TR0 - TR1;
    (*R6).x = TR2 - (C5QE * TR3 + C5QD * TI3);
    (*R7).x = TR4 - (C5QA * TR5 + C5QB * TI5);
    (*R8).x = TR6 - (-C5QA * TR7 + C5QB * TI7);
    (*R9).x = TR8 - (-C5QE * TR9 + C5QD * TI9);

    (*R5).y = TI0 - TI1;
    (*R6).y = TI2 - (-C5QD * TR3 + C5QE * TI3);
    (*R7).y = TI4 - (-C5QB * TR5 + C5QA * TI5);
    (*R8).y = TI6 - (-C5QB * TR7 - C5QA * TI7);
    (*R9).y = TI8 - (-C5QD * TR9 - C5QE * TI9);
}

template <typename T>
__device__ void InvRad10B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5, TR6, TI6, TR7, TI7,
        TR8, TI8, TR9, TI9;

    TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;
    TR2 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) - C5QB * ((*R2).y - (*R8).y)
          - C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR8 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) + C5QB * ((*R2).y - (*R8).y)
          + C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) + C5QB * ((*R4).y - (*R6).y)
          - C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));
    TR6 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) - C5QB * ((*R4).y - (*R6).y)
          + C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));

    TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;
    TI2 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) + C5QB * ((*R2).x - (*R8).x)
          + C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI8 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) - C5QB * ((*R2).x - (*R8).x)
          - C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) - C5QB * ((*R4).x - (*R6).x)
          + C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));
    TI6 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) + C5QB * ((*R4).x - (*R6).x)
          - C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));

    TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;
    TR3 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) - C5QB * ((*R3).y - (*R9).y)
          - C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR9 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) + C5QB * ((*R3).y - (*R9).y)
          + C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR5 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) + C5QB * ((*R5).y - (*R7).y)
          - C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));
    TR7 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) - C5QB * ((*R5).y - (*R7).y)
          + C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));

    TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;
    TI3 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) + C5QB * ((*R3).x - (*R9).x)
          + C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI9 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) - C5QB * ((*R3).x - (*R9).x)
          - C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI5 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) - C5QB * ((*R5).x - (*R7).x)
          + C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));
    TI7 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) + C5QB * ((*R5).x - (*R7).x)
          - C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C5QE * TR3 - C5QD * TI3);
    (*R2).x = TR4 + (C5QA * TR5 - C5QB * TI5);
    (*R3).x = TR6 + (-C5QA * TR7 - C5QB * TI7);
    (*R4).x = TR8 + (-C5QE * TR9 - C5QD * TI9);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (C5QD * TR3 + C5QE * TI3);
    (*R2).y = TI4 + (C5QB * TR5 + C5QA * TI5);
    (*R3).y = TI6 + (C5QB * TR7 - C5QA * TI7);
    (*R4).y = TI8 + (C5QD * TR9 - C5QE * TI9);

    (*R5).x = TR0 - TR1;
    (*R6).x = TR2 - (C5QE * TR3 - C5QD * TI3);
    (*R7).x = TR4 - (C5QA * TR5 - C5QB * TI5);
    (*R8).x = TR6 - (-C5QA * TR7 - C5QB * TI7);
    (*R9).x = TR8 - (-C5QE * TR9 - C5QD * TI9);

    (*R5).y = TI0 - TI1;
    (*R6).y = TI2 - (C5QD * TR3 + C5QE * TI3);
    (*R7).y = TI4 - (C5QB * TR5 + C5QA * TI5);
    (*R8).y = TI6 - (C5QB * TR7 - C5QA * TI7);
    (*R9).y = TI8 - (C5QD * TR9 - C5QE * TI9);
}

template <typename T>
__device__ void FwdRad16B1(T* R0,
                           T* R8,
                           T* R4,
                           T* R12,
                           T* R2,
                           T* R10,
                           T* R6,
                           T* R14,
                           T* R1,
                           T* R9,
                           T* R5,
                           T* R13,
                           T* R3,
                           T* R11,
                           T* R7,
                           T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + T(-(*R7).y, (*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + T(-(*R11).y, (*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + T(-(*R15).y, (*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) - C8Q * T((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + T(-(*R6).y, (*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) - C8Q * T((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) - C8Q * T((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + T(-(*R14).y, (*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) - C8Q * T((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8) = (*R0) - (*R8);
    (*R0) = 2.0 * (*R0) - (*R8);
    (*R9) = ((*R1) - C16A * (*R9)) - C16B * T((*R9).y, -(*R9).x);
    res   = (*R8);
    (*R1) = 2.0 * (*R1) - (*R9);

    (*R10) = ((*R2) - C8Q * (*R10)) - C8Q * T((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) - C16A * T((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);

    (*R12) = (*R4) + T(-(*R12).y, (*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) - C16A * T((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);

    (*R14) = ((*R6) + C8Q * (*R14)) - C8Q * T((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) - C16B * T((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

template <typename T>
__device__ void InvRad16B1(T* R0,
                           T* R8,
                           T* R4,
                           T* R12,
                           T* R2,
                           T* R10,
                           T* R6,
                           T* R14,
                           T* R1,
                           T* R9,
                           T* R5,
                           T* R13,
                           T* R3,
                           T* R11,
                           T* R7,
                           T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + T((*R3).y, -(*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + T((*R7).y, -(*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + T((*R11).y, -(*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + T((*R15).y, -(*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) + C8Q * T((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + T((*R6).y, -(*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) + C8Q * T((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) + C8Q * T((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + T((*R14).y, -(*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) + C8Q * T((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8)  = (*R0) - (*R8);
    (*R0)  = 2.0 * (*R0) - (*R8);
    (*R9)  = ((*R1) - C16A * (*R9)) + C16B * T((*R9).y, -(*R9).x);
    (*R1)  = 2.0 * (*R1) - (*R9);
    (*R10) = ((*R2) - C8Q * (*R10)) + C8Q * T((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) + C16A * T((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);
    (*R12) = (*R4) + T((*R12).y, -(*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) + C16A * T((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);
    (*R14) = ((*R6) + C8Q * (*R14)) + C8Q * T((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) + C16B * T((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

template <typename T>
__device__ void
    FwdRad11B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, dp, dm;

    x0  = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    dp  = (*R1) + (*R10);
    dm  = (*R1) - (*R10);
    x1.x += Q11i1j1R * dp.x - Q11i1j1I * dm.y;
    x1.y += Q11i1j1R * dp.y + Q11i1j1I * dm.x;
    x10.x += Q11i1j1R * dp.x + Q11i1j1I * dm.y;
    x10.y += Q11i1j1R * dp.y - Q11i1j1I * dm.x;
    x2.x += Q11i2j1R * dp.x - Q11i2j1I * dm.y;
    x2.y += Q11i2j1R * dp.y + Q11i2j1I * dm.x;
    x9.x += Q11i2j1R * dp.x + Q11i2j1I * dm.y;
    x9.y += Q11i2j1R * dp.y - Q11i2j1I * dm.x;
    x3.x += Q11i3j1R * dp.x - Q11i3j1I * dm.y;
    x3.y += Q11i3j1R * dp.y + Q11i3j1I * dm.x;
    x8.x += Q11i3j1R * dp.x + Q11i3j1I * dm.y;
    x8.y += Q11i3j1R * dp.y - Q11i3j1I * dm.x;
    x4.x += Q11i4j1R * dp.x - Q11i4j1I * dm.y;
    x4.y += Q11i4j1R * dp.y + Q11i4j1I * dm.x;
    x7.x += Q11i4j1R * dp.x + Q11i4j1I * dm.y;
    x7.y += Q11i4j1R * dp.y - Q11i4j1I * dm.x;
    x5.x += Q11i5j1R * dp.x - Q11i5j1I * dm.y;
    x5.y += Q11i5j1R * dp.y + Q11i5j1I * dm.x;
    x6.x += Q11i5j1R * dp.x + Q11i5j1I * dm.y;
    x6.y += Q11i5j1R * dp.y - Q11i5j1I * dm.x;
    dp = (*R2) + (*R9);
    dm = (*R2) - (*R9);
    x1.x += Q11i1j2R * dp.x - Q11i1j2I * dm.y;
    x1.y += Q11i1j2R * dp.y + Q11i1j2I * dm.x;
    x10.x += Q11i1j2R * dp.x + Q11i1j2I * dm.y;
    x10.y += Q11i1j2R * dp.y - Q11i1j2I * dm.x;
    x2.x += Q11i2j2R * dp.x - Q11i2j2I * dm.y;
    x2.y += Q11i2j2R * dp.y + Q11i2j2I * dm.x;
    x9.x += Q11i2j2R * dp.x + Q11i2j2I * dm.y;
    x9.y += Q11i2j2R * dp.y - Q11i2j2I * dm.x;
    x3.x += Q11i3j2R * dp.x - Q11i3j2I * dm.y;
    x3.y += Q11i3j2R * dp.y + Q11i3j2I * dm.x;
    x8.x += Q11i3j2R * dp.x + Q11i3j2I * dm.y;
    x8.y += Q11i3j2R * dp.y - Q11i3j2I * dm.x;
    x4.x += Q11i4j2R * dp.x - Q11i4j2I * dm.y;
    x4.y += Q11i4j2R * dp.y + Q11i4j2I * dm.x;
    x7.x += Q11i4j2R * dp.x + Q11i4j2I * dm.y;
    x7.y += Q11i4j2R * dp.y - Q11i4j2I * dm.x;
    x5.x += Q11i5j2R * dp.x - Q11i5j2I * dm.y;
    x5.y += Q11i5j2R * dp.y + Q11i5j2I * dm.x;
    x6.x += Q11i5j2R * dp.x + Q11i5j2I * dm.y;
    x6.y += Q11i5j2R * dp.y - Q11i5j2I * dm.x;
    dp = (*R3) + (*R8);
    dm = (*R3) - (*R8);
    x1.x += Q11i1j3R * dp.x - Q11i1j3I * dm.y;
    x1.y += Q11i1j3R * dp.y + Q11i1j3I * dm.x;
    x10.x += Q11i1j3R * dp.x + Q11i1j3I * dm.y;
    x10.y += Q11i1j3R * dp.y - Q11i1j3I * dm.x;
    x2.x += Q11i2j3R * dp.x - Q11i2j3I * dm.y;
    x2.y += Q11i2j3R * dp.y + Q11i2j3I * dm.x;
    x9.x += Q11i2j3R * dp.x + Q11i2j3I * dm.y;
    x9.y += Q11i2j3R * dp.y - Q11i2j3I * dm.x;
    x3.x += Q11i3j3R * dp.x - Q11i3j3I * dm.y;
    x3.y += Q11i3j3R * dp.y + Q11i3j3I * dm.x;
    x8.x += Q11i3j3R * dp.x + Q11i3j3I * dm.y;
    x8.y += Q11i3j3R * dp.y - Q11i3j3I * dm.x;
    x4.x += Q11i4j3R * dp.x - Q11i4j3I * dm.y;
    x4.y += Q11i4j3R * dp.y + Q11i4j3I * dm.x;
    x7.x += Q11i4j3R * dp.x + Q11i4j3I * dm.y;
    x7.y += Q11i4j3R * dp.y - Q11i4j3I * dm.x;
    x5.x += Q11i5j3R * dp.x - Q11i5j3I * dm.y;
    x5.y += Q11i5j3R * dp.y + Q11i5j3I * dm.x;
    x6.x += Q11i5j3R * dp.x + Q11i5j3I * dm.y;
    x6.y += Q11i5j3R * dp.y - Q11i5j3I * dm.x;
    dp = (*R4) + (*R7);
    dm = (*R4) - (*R7);
    x1.x += Q11i1j4R * dp.x - Q11i1j4I * dm.y;
    x1.y += Q11i1j4R * dp.y + Q11i1j4I * dm.x;
    x10.x += Q11i1j4R * dp.x + Q11i1j4I * dm.y;
    x10.y += Q11i1j4R * dp.y - Q11i1j4I * dm.x;
    x2.x += Q11i2j4R * dp.x - Q11i2j4I * dm.y;
    x2.y += Q11i2j4R * dp.y + Q11i2j4I * dm.x;
    x9.x += Q11i2j4R * dp.x + Q11i2j4I * dm.y;
    x9.y += Q11i2j4R * dp.y - Q11i2j4I * dm.x;
    x3.x += Q11i3j4R * dp.x - Q11i3j4I * dm.y;
    x3.y += Q11i3j4R * dp.y + Q11i3j4I * dm.x;
    x8.x += Q11i3j4R * dp.x + Q11i3j4I * dm.y;
    x8.y += Q11i3j4R * dp.y - Q11i3j4I * dm.x;
    x4.x += Q11i4j4R * dp.x - Q11i4j4I * dm.y;
    x4.y += Q11i4j4R * dp.y + Q11i4j4I * dm.x;
    x7.x += Q11i4j4R * dp.x + Q11i4j4I * dm.y;
    x7.y += Q11i4j4R * dp.y - Q11i4j4I * dm.x;
    x5.x += Q11i5j4R * dp.x - Q11i5j4I * dm.y;
    x5.y += Q11i5j4R * dp.y + Q11i5j4I * dm.x;
    x6.x += Q11i5j4R * dp.x + Q11i5j4I * dm.y;
    x6.y += Q11i5j4R * dp.y - Q11i5j4I * dm.x;
    dp = (*R5) + (*R6);
    dm = (*R5) - (*R6);
    x1.x += Q11i1j5R * dp.x - Q11i1j5I * dm.y;
    x1.y += Q11i1j5R * dp.y + Q11i1j5I * dm.x;
    x10.x += Q11i1j5R * dp.x + Q11i1j5I * dm.y;
    x10.y += Q11i1j5R * dp.y - Q11i1j5I * dm.x;
    x2.x += Q11i2j5R * dp.x - Q11i2j5I * dm.y;
    x2.y += Q11i2j5R * dp.y + Q11i2j5I * dm.x;
    x9.x += Q11i2j5R * dp.x + Q11i2j5I * dm.y;
    x9.y += Q11i2j5R * dp.y - Q11i2j5I * dm.x;
    x3.x += Q11i3j5R * dp.x - Q11i3j5I * dm.y;
    x3.y += Q11i3j5R * dp.y + Q11i3j5I * dm.x;
    x8.x += Q11i3j5R * dp.x + Q11i3j5I * dm.y;
    x8.y += Q11i3j5R * dp.y - Q11i3j5I * dm.x;
    x4.x += Q11i4j5R * dp.x - Q11i4j5I * dm.y;
    x4.y += Q11i4j5R * dp.y + Q11i4j5I * dm.x;
    x7.x += Q11i4j5R * dp.x + Q11i4j5I * dm.y;
    x7.y += Q11i4j5R * dp.y - Q11i4j5I * dm.x;
    x5.x += Q11i5j5R * dp.x - Q11i5j5I * dm.y;
    x5.y += Q11i5j5R * dp.y + Q11i5j5I * dm.x;
    x6.x += Q11i5j5R * dp.x + Q11i5j5I * dm.y;
    x6.y += Q11i5j5R * dp.y - Q11i5j5I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
}

template <typename T>
__device__ void
    InvRad11B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, dp, dm;

    x0  = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    dp  = (*R1) + (*R10);
    dm  = (*R1) - (*R10);
    x1.x += Q11i1j1R * dp.x + Q11i1j1I * dm.y;
    x1.y += Q11i1j1R * dp.y - Q11i1j1I * dm.x;
    x10.x += Q11i1j1R * dp.x - Q11i1j1I * dm.y;
    x10.y += Q11i1j1R * dp.y + Q11i1j1I * dm.x;
    x2.x += Q11i2j1R * dp.x + Q11i2j1I * dm.y;
    x2.y += Q11i2j1R * dp.y - Q11i2j1I * dm.x;
    x9.x += Q11i2j1R * dp.x - Q11i2j1I * dm.y;
    x9.y += Q11i2j1R * dp.y + Q11i2j1I * dm.x;
    x3.x += Q11i3j1R * dp.x + Q11i3j1I * dm.y;
    x3.y += Q11i3j1R * dp.y - Q11i3j1I * dm.x;
    x8.x += Q11i3j1R * dp.x - Q11i3j1I * dm.y;
    x8.y += Q11i3j1R * dp.y + Q11i3j1I * dm.x;
    x4.x += Q11i4j1R * dp.x + Q11i4j1I * dm.y;
    x4.y += Q11i4j1R * dp.y - Q11i4j1I * dm.x;
    x7.x += Q11i4j1R * dp.x - Q11i4j1I * dm.y;
    x7.y += Q11i4j1R * dp.y + Q11i4j1I * dm.x;
    x5.x += Q11i5j1R * dp.x + Q11i5j1I * dm.y;
    x5.y += Q11i5j1R * dp.y - Q11i5j1I * dm.x;
    x6.x += Q11i5j1R * dp.x - Q11i5j1I * dm.y;
    x6.y += Q11i5j1R * dp.y + Q11i5j1I * dm.x;
    dp = (*R2) + (*R9);
    dm = (*R2) - (*R9);
    x1.x += Q11i1j2R * dp.x + Q11i1j2I * dm.y;
    x1.y += Q11i1j2R * dp.y - Q11i1j2I * dm.x;
    x10.x += Q11i1j2R * dp.x - Q11i1j2I * dm.y;
    x10.y += Q11i1j2R * dp.y + Q11i1j2I * dm.x;
    x2.x += Q11i2j2R * dp.x + Q11i2j2I * dm.y;
    x2.y += Q11i2j2R * dp.y - Q11i2j2I * dm.x;
    x9.x += Q11i2j2R * dp.x - Q11i2j2I * dm.y;
    x9.y += Q11i2j2R * dp.y + Q11i2j2I * dm.x;
    x3.x += Q11i3j2R * dp.x + Q11i3j2I * dm.y;
    x3.y += Q11i3j2R * dp.y - Q11i3j2I * dm.x;
    x8.x += Q11i3j2R * dp.x - Q11i3j2I * dm.y;
    x8.y += Q11i3j2R * dp.y + Q11i3j2I * dm.x;
    x4.x += Q11i4j2R * dp.x + Q11i4j2I * dm.y;
    x4.y += Q11i4j2R * dp.y - Q11i4j2I * dm.x;
    x7.x += Q11i4j2R * dp.x - Q11i4j2I * dm.y;
    x7.y += Q11i4j2R * dp.y + Q11i4j2I * dm.x;
    x5.x += Q11i5j2R * dp.x + Q11i5j2I * dm.y;
    x5.y += Q11i5j2R * dp.y - Q11i5j2I * dm.x;
    x6.x += Q11i5j2R * dp.x - Q11i5j2I * dm.y;
    x6.y += Q11i5j2R * dp.y + Q11i5j2I * dm.x;
    dp = (*R3) + (*R8);
    dm = (*R3) - (*R8);
    x1.x += Q11i1j3R * dp.x + Q11i1j3I * dm.y;
    x1.y += Q11i1j3R * dp.y - Q11i1j3I * dm.x;
    x10.x += Q11i1j3R * dp.x - Q11i1j3I * dm.y;
    x10.y += Q11i1j3R * dp.y + Q11i1j3I * dm.x;
    x2.x += Q11i2j3R * dp.x + Q11i2j3I * dm.y;
    x2.y += Q11i2j3R * dp.y - Q11i2j3I * dm.x;
    x9.x += Q11i2j3R * dp.x - Q11i2j3I * dm.y;
    x9.y += Q11i2j3R * dp.y + Q11i2j3I * dm.x;
    x3.x += Q11i3j3R * dp.x + Q11i3j3I * dm.y;
    x3.y += Q11i3j3R * dp.y - Q11i3j3I * dm.x;
    x8.x += Q11i3j3R * dp.x - Q11i3j3I * dm.y;
    x8.y += Q11i3j3R * dp.y + Q11i3j3I * dm.x;
    x4.x += Q11i4j3R * dp.x + Q11i4j3I * dm.y;
    x4.y += Q11i4j3R * dp.y - Q11i4j3I * dm.x;
    x7.x += Q11i4j3R * dp.x - Q11i4j3I * dm.y;
    x7.y += Q11i4j3R * dp.y + Q11i4j3I * dm.x;
    x5.x += Q11i5j3R * dp.x + Q11i5j3I * dm.y;
    x5.y += Q11i5j3R * dp.y - Q11i5j3I * dm.x;
    x6.x += Q11i5j3R * dp.x - Q11i5j3I * dm.y;
    x6.y += Q11i5j3R * dp.y + Q11i5j3I * dm.x;
    dp = (*R4) + (*R7);
    dm = (*R4) - (*R7);
    x1.x += Q11i1j4R * dp.x + Q11i1j4I * dm.y;
    x1.y += Q11i1j4R * dp.y - Q11i1j4I * dm.x;
    x10.x += Q11i1j4R * dp.x - Q11i1j4I * dm.y;
    x10.y += Q11i1j4R * dp.y + Q11i1j4I * dm.x;
    x2.x += Q11i2j4R * dp.x + Q11i2j4I * dm.y;
    x2.y += Q11i2j4R * dp.y - Q11i2j4I * dm.x;
    x9.x += Q11i2j4R * dp.x - Q11i2j4I * dm.y;
    x9.y += Q11i2j4R * dp.y + Q11i2j4I * dm.x;
    x3.x += Q11i3j4R * dp.x + Q11i3j4I * dm.y;
    x3.y += Q11i3j4R * dp.y - Q11i3j4I * dm.x;
    x8.x += Q11i3j4R * dp.x - Q11i3j4I * dm.y;
    x8.y += Q11i3j4R * dp.y + Q11i3j4I * dm.x;
    x4.x += Q11i4j4R * dp.x + Q11i4j4I * dm.y;
    x4.y += Q11i4j4R * dp.y - Q11i4j4I * dm.x;
    x7.x += Q11i4j4R * dp.x - Q11i4j4I * dm.y;
    x7.y += Q11i4j4R * dp.y + Q11i4j4I * dm.x;
    x5.x += Q11i5j4R * dp.x + Q11i5j4I * dm.y;
    x5.y += Q11i5j4R * dp.y - Q11i5j4I * dm.x;
    x6.x += Q11i5j4R * dp.x - Q11i5j4I * dm.y;
    x6.y += Q11i5j4R * dp.y + Q11i5j4I * dm.x;
    dp = (*R5) + (*R6);
    dm = (*R5) - (*R6);
    x1.x += Q11i1j5R * dp.x + Q11i1j5I * dm.y;
    x1.y += Q11i1j5R * dp.y - Q11i1j5I * dm.x;
    x10.x += Q11i1j5R * dp.x - Q11i1j5I * dm.y;
    x10.y += Q11i1j5R * dp.y + Q11i1j5I * dm.x;
    x2.x += Q11i2j5R * dp.x + Q11i2j5I * dm.y;
    x2.y += Q11i2j5R * dp.y - Q11i2j5I * dm.x;
    x9.x += Q11i2j5R * dp.x - Q11i2j5I * dm.y;
    x9.y += Q11i2j5R * dp.y + Q11i2j5I * dm.x;
    x3.x += Q11i3j5R * dp.x + Q11i3j5I * dm.y;
    x3.y += Q11i3j5R * dp.y - Q11i3j5I * dm.x;
    x8.x += Q11i3j5R * dp.x - Q11i3j5I * dm.y;
    x8.y += Q11i3j5R * dp.y + Q11i3j5I * dm.x;
    x4.x += Q11i4j5R * dp.x + Q11i4j5I * dm.y;
    x4.y += Q11i4j5R * dp.y - Q11i4j5I * dm.x;
    x7.x += Q11i4j5R * dp.x - Q11i4j5I * dm.y;
    x7.y += Q11i4j5R * dp.y + Q11i4j5I * dm.x;
    x5.x += Q11i5j5R * dp.x + Q11i5j5I * dm.y;
    x5.y += Q11i5j5R * dp.y - Q11i5j5I * dm.x;
    x6.x += Q11i5j5R * dp.x - Q11i5j5I * dm.y;
    x6.y += Q11i5j5R * dp.y + Q11i5j5I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
}

template <typename T>
__device__ void FwdRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}

template <typename T>
__device__ void InvRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}

template <typename T>
__device__ void FwdRad17B1(T* R0,
                           T* R1,
                           T* R2,
                           T* R3,
                           T* R4,
                           T* R5,
                           T* R6,
                           T* R7,
                           T* R8,
                           T* R9,
                           T* R10,
                           T* R11,
                           T* R12,
                           T* R13,
                           T* R14,
                           T* R15,
                           T* R16)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12) + (*R13) + (*R14) + (*R15) + (*R16);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    x13 = (*R0);
    x14 = (*R0);
    x15 = (*R0);
    x16 = (*R0);
    dp  = (*R1) + (*R16);
    dm  = (*R1) - (*R16);
    x1.x += Q17i1j1R * dp.x - Q17i1j1I * dm.y;
    x1.y += Q17i1j1R * dp.y + Q17i1j1I * dm.x;
    x16.x += Q17i1j1R * dp.x + Q17i1j1I * dm.y;
    x16.y += Q17i1j1R * dp.y - Q17i1j1I * dm.x;
    x2.x += Q17i2j1R * dp.x - Q17i2j1I * dm.y;
    x2.y += Q17i2j1R * dp.y + Q17i2j1I * dm.x;
    x15.x += Q17i2j1R * dp.x + Q17i2j1I * dm.y;
    x15.y += Q17i2j1R * dp.y - Q17i2j1I * dm.x;
    x3.x += Q17i3j1R * dp.x - Q17i3j1I * dm.y;
    x3.y += Q17i3j1R * dp.y + Q17i3j1I * dm.x;
    x14.x += Q17i3j1R * dp.x + Q17i3j1I * dm.y;
    x14.y += Q17i3j1R * dp.y - Q17i3j1I * dm.x;
    x4.x += Q17i4j1R * dp.x - Q17i4j1I * dm.y;
    x4.y += Q17i4j1R * dp.y + Q17i4j1I * dm.x;
    x13.x += Q17i4j1R * dp.x + Q17i4j1I * dm.y;
    x13.y += Q17i4j1R * dp.y - Q17i4j1I * dm.x;
    x5.x += Q17i5j1R * dp.x - Q17i5j1I * dm.y;
    x5.y += Q17i5j1R * dp.y + Q17i5j1I * dm.x;
    x12.x += Q17i5j1R * dp.x + Q17i5j1I * dm.y;
    x12.y += Q17i5j1R * dp.y - Q17i5j1I * dm.x;
    x6.x += Q17i6j1R * dp.x - Q17i6j1I * dm.y;
    x6.y += Q17i6j1R * dp.y + Q17i6j1I * dm.x;
    x11.x += Q17i6j1R * dp.x + Q17i6j1I * dm.y;
    x11.y += Q17i6j1R * dp.y - Q17i6j1I * dm.x;
    x7.x += Q17i7j1R * dp.x - Q17i7j1I * dm.y;
    x7.y += Q17i7j1R * dp.y + Q17i7j1I * dm.x;
    x10.x += Q17i7j1R * dp.x + Q17i7j1I * dm.y;
    x10.y += Q17i7j1R * dp.y - Q17i7j1I * dm.x;
    x8.x += Q17i8j1R * dp.x - Q17i8j1I * dm.y;
    x8.y += Q17i8j1R * dp.y + Q17i8j1I * dm.x;
    x9.x += Q17i8j1R * dp.x + Q17i8j1I * dm.y;
    x9.y += Q17i8j1R * dp.y - Q17i8j1I * dm.x;
    dp = (*R2) + (*R15);
    dm = (*R2) - (*R15);
    x1.x += Q17i1j2R * dp.x - Q17i1j2I * dm.y;
    x1.y += Q17i1j2R * dp.y + Q17i1j2I * dm.x;
    x16.x += Q17i1j2R * dp.x + Q17i1j2I * dm.y;
    x16.y += Q17i1j2R * dp.y - Q17i1j2I * dm.x;
    x2.x += Q17i2j2R * dp.x - Q17i2j2I * dm.y;
    x2.y += Q17i2j2R * dp.y + Q17i2j2I * dm.x;
    x15.x += Q17i2j2R * dp.x + Q17i2j2I * dm.y;
    x15.y += Q17i2j2R * dp.y - Q17i2j2I * dm.x;
    x3.x += Q17i3j2R * dp.x - Q17i3j2I * dm.y;
    x3.y += Q17i3j2R * dp.y + Q17i3j2I * dm.x;
    x14.x += Q17i3j2R * dp.x + Q17i3j2I * dm.y;
    x14.y += Q17i3j2R * dp.y - Q17i3j2I * dm.x;
    x4.x += Q17i4j2R * dp.x - Q17i4j2I * dm.y;
    x4.y += Q17i4j2R * dp.y + Q17i4j2I * dm.x;
    x13.x += Q17i4j2R * dp.x + Q17i4j2I * dm.y;
    x13.y += Q17i4j2R * dp.y - Q17i4j2I * dm.x;
    x5.x += Q17i5j2R * dp.x - Q17i5j2I * dm.y;
    x5.y += Q17i5j2R * dp.y + Q17i5j2I * dm.x;
    x12.x += Q17i5j2R * dp.x + Q17i5j2I * dm.y;
    x12.y += Q17i5j2R * dp.y - Q17i5j2I * dm.x;
    x6.x += Q17i6j2R * dp.x - Q17i6j2I * dm.y;
    x6.y += Q17i6j2R * dp.y + Q17i6j2I * dm.x;
    x11.x += Q17i6j2R * dp.x + Q17i6j2I * dm.y;
    x11.y += Q17i6j2R * dp.y - Q17i6j2I * dm.x;
    x7.x += Q17i7j2R * dp.x - Q17i7j2I * dm.y;
    x7.y += Q17i7j2R * dp.y + Q17i7j2I * dm.x;
    x10.x += Q17i7j2R * dp.x + Q17i7j2I * dm.y;
    x10.y += Q17i7j2R * dp.y - Q17i7j2I * dm.x;
    x8.x += Q17i8j2R * dp.x - Q17i8j2I * dm.y;
    x8.y += Q17i8j2R * dp.y + Q17i8j2I * dm.x;
    x9.x += Q17i8j2R * dp.x + Q17i8j2I * dm.y;
    x9.y += Q17i8j2R * dp.y - Q17i8j2I * dm.x;
    dp = (*R3) + (*R14);
    dm = (*R3) - (*R14);
    x1.x += Q17i1j3R * dp.x - Q17i1j3I * dm.y;
    x1.y += Q17i1j3R * dp.y + Q17i1j3I * dm.x;
    x16.x += Q17i1j3R * dp.x + Q17i1j3I * dm.y;
    x16.y += Q17i1j3R * dp.y - Q17i1j3I * dm.x;
    x2.x += Q17i2j3R * dp.x - Q17i2j3I * dm.y;
    x2.y += Q17i2j3R * dp.y + Q17i2j3I * dm.x;
    x15.x += Q17i2j3R * dp.x + Q17i2j3I * dm.y;
    x15.y += Q17i2j3R * dp.y - Q17i2j3I * dm.x;
    x3.x += Q17i3j3R * dp.x - Q17i3j3I * dm.y;
    x3.y += Q17i3j3R * dp.y + Q17i3j3I * dm.x;
    x14.x += Q17i3j3R * dp.x + Q17i3j3I * dm.y;
    x14.y += Q17i3j3R * dp.y - Q17i3j3I * dm.x;
    x4.x += Q17i4j3R * dp.x - Q17i4j3I * dm.y;
    x4.y += Q17i4j3R * dp.y + Q17i4j3I * dm.x;
    x13.x += Q17i4j3R * dp.x + Q17i4j3I * dm.y;
    x13.y += Q17i4j3R * dp.y - Q17i4j3I * dm.x;
    x5.x += Q17i5j3R * dp.x - Q17i5j3I * dm.y;
    x5.y += Q17i5j3R * dp.y + Q17i5j3I * dm.x;
    x12.x += Q17i5j3R * dp.x + Q17i5j3I * dm.y;
    x12.y += Q17i5j3R * dp.y - Q17i5j3I * dm.x;
    x6.x += Q17i6j3R * dp.x - Q17i6j3I * dm.y;
    x6.y += Q17i6j3R * dp.y + Q17i6j3I * dm.x;
    x11.x += Q17i6j3R * dp.x + Q17i6j3I * dm.y;
    x11.y += Q17i6j3R * dp.y - Q17i6j3I * dm.x;
    x7.x += Q17i7j3R * dp.x - Q17i7j3I * dm.y;
    x7.y += Q17i7j3R * dp.y + Q17i7j3I * dm.x;
    x10.x += Q17i7j3R * dp.x + Q17i7j3I * dm.y;
    x10.y += Q17i7j3R * dp.y - Q17i7j3I * dm.x;
    x8.x += Q17i8j3R * dp.x - Q17i8j3I * dm.y;
    x8.y += Q17i8j3R * dp.y + Q17i8j3I * dm.x;
    x9.x += Q17i8j3R * dp.x + Q17i8j3I * dm.y;
    x9.y += Q17i8j3R * dp.y - Q17i8j3I * dm.x;
    dp = (*R4) + (*R13);
    dm = (*R4) - (*R13);
    x1.x += Q17i1j4R * dp.x - Q17i1j4I * dm.y;
    x1.y += Q17i1j4R * dp.y + Q17i1j4I * dm.x;
    x16.x += Q17i1j4R * dp.x + Q17i1j4I * dm.y;
    x16.y += Q17i1j4R * dp.y - Q17i1j4I * dm.x;
    x2.x += Q17i2j4R * dp.x - Q17i2j4I * dm.y;
    x2.y += Q17i2j4R * dp.y + Q17i2j4I * dm.x;
    x15.x += Q17i2j4R * dp.x + Q17i2j4I * dm.y;
    x15.y += Q17i2j4R * dp.y - Q17i2j4I * dm.x;
    x3.x += Q17i3j4R * dp.x - Q17i3j4I * dm.y;
    x3.y += Q17i3j4R * dp.y + Q17i3j4I * dm.x;
    x14.x += Q17i3j4R * dp.x + Q17i3j4I * dm.y;
    x14.y += Q17i3j4R * dp.y - Q17i3j4I * dm.x;
    x4.x += Q17i4j4R * dp.x - Q17i4j4I * dm.y;
    x4.y += Q17i4j4R * dp.y + Q17i4j4I * dm.x;
    x13.x += Q17i4j4R * dp.x + Q17i4j4I * dm.y;
    x13.y += Q17i4j4R * dp.y - Q17i4j4I * dm.x;
    x5.x += Q17i5j4R * dp.x - Q17i5j4I * dm.y;
    x5.y += Q17i5j4R * dp.y + Q17i5j4I * dm.x;
    x12.x += Q17i5j4R * dp.x + Q17i5j4I * dm.y;
    x12.y += Q17i5j4R * dp.y - Q17i5j4I * dm.x;
    x6.x += Q17i6j4R * dp.x - Q17i6j4I * dm.y;
    x6.y += Q17i6j4R * dp.y + Q17i6j4I * dm.x;
    x11.x += Q17i6j4R * dp.x + Q17i6j4I * dm.y;
    x11.y += Q17i6j4R * dp.y - Q17i6j4I * dm.x;
    x7.x += Q17i7j4R * dp.x - Q17i7j4I * dm.y;
    x7.y += Q17i7j4R * dp.y + Q17i7j4I * dm.x;
    x10.x += Q17i7j4R * dp.x + Q17i7j4I * dm.y;
    x10.y += Q17i7j4R * dp.y - Q17i7j4I * dm.x;
    x8.x += Q17i8j4R * dp.x - Q17i8j4I * dm.y;
    x8.y += Q17i8j4R * dp.y + Q17i8j4I * dm.x;
    x9.x += Q17i8j4R * dp.x + Q17i8j4I * dm.y;
    x9.y += Q17i8j4R * dp.y - Q17i8j4I * dm.x;
    dp = (*R5) + (*R12);
    dm = (*R5) - (*R12);
    x1.x += Q17i1j5R * dp.x - Q17i1j5I * dm.y;
    x1.y += Q17i1j5R * dp.y + Q17i1j5I * dm.x;
    x16.x += Q17i1j5R * dp.x + Q17i1j5I * dm.y;
    x16.y += Q17i1j5R * dp.y - Q17i1j5I * dm.x;
    x2.x += Q17i2j5R * dp.x - Q17i2j5I * dm.y;
    x2.y += Q17i2j5R * dp.y + Q17i2j5I * dm.x;
    x15.x += Q17i2j5R * dp.x + Q17i2j5I * dm.y;
    x15.y += Q17i2j5R * dp.y - Q17i2j5I * dm.x;
    x3.x += Q17i3j5R * dp.x - Q17i3j5I * dm.y;
    x3.y += Q17i3j5R * dp.y + Q17i3j5I * dm.x;
    x14.x += Q17i3j5R * dp.x + Q17i3j5I * dm.y;
    x14.y += Q17i3j5R * dp.y - Q17i3j5I * dm.x;
    x4.x += Q17i4j5R * dp.x - Q17i4j5I * dm.y;
    x4.y += Q17i4j5R * dp.y + Q17i4j5I * dm.x;
    x13.x += Q17i4j5R * dp.x + Q17i4j5I * dm.y;
    x13.y += Q17i4j5R * dp.y - Q17i4j5I * dm.x;
    x5.x += Q17i5j5R * dp.x - Q17i5j5I * dm.y;
    x5.y += Q17i5j5R * dp.y + Q17i5j5I * dm.x;
    x12.x += Q17i5j5R * dp.x + Q17i5j5I * dm.y;
    x12.y += Q17i5j5R * dp.y - Q17i5j5I * dm.x;
    x6.x += Q17i6j5R * dp.x - Q17i6j5I * dm.y;
    x6.y += Q17i6j5R * dp.y + Q17i6j5I * dm.x;
    x11.x += Q17i6j5R * dp.x + Q17i6j5I * dm.y;
    x11.y += Q17i6j5R * dp.y - Q17i6j5I * dm.x;
    x7.x += Q17i7j5R * dp.x - Q17i7j5I * dm.y;
    x7.y += Q17i7j5R * dp.y + Q17i7j5I * dm.x;
    x10.x += Q17i7j5R * dp.x + Q17i7j5I * dm.y;
    x10.y += Q17i7j5R * dp.y - Q17i7j5I * dm.x;
    x8.x += Q17i8j5R * dp.x - Q17i8j5I * dm.y;
    x8.y += Q17i8j5R * dp.y + Q17i8j5I * dm.x;
    x9.x += Q17i8j5R * dp.x + Q17i8j5I * dm.y;
    x9.y += Q17i8j5R * dp.y - Q17i8j5I * dm.x;
    dp = (*R6) + (*R11);
    dm = (*R6) - (*R11);
    x1.x += Q17i1j6R * dp.x - Q17i1j6I * dm.y;
    x1.y += Q17i1j6R * dp.y + Q17i1j6I * dm.x;
    x16.x += Q17i1j6R * dp.x + Q17i1j6I * dm.y;
    x16.y += Q17i1j6R * dp.y - Q17i1j6I * dm.x;
    x2.x += Q17i2j6R * dp.x - Q17i2j6I * dm.y;
    x2.y += Q17i2j6R * dp.y + Q17i2j6I * dm.x;
    x15.x += Q17i2j6R * dp.x + Q17i2j6I * dm.y;
    x15.y += Q17i2j6R * dp.y - Q17i2j6I * dm.x;
    x3.x += Q17i3j6R * dp.x - Q17i3j6I * dm.y;
    x3.y += Q17i3j6R * dp.y + Q17i3j6I * dm.x;
    x14.x += Q17i3j6R * dp.x + Q17i3j6I * dm.y;
    x14.y += Q17i3j6R * dp.y - Q17i3j6I * dm.x;
    x4.x += Q17i4j6R * dp.x - Q17i4j6I * dm.y;
    x4.y += Q17i4j6R * dp.y + Q17i4j6I * dm.x;
    x13.x += Q17i4j6R * dp.x + Q17i4j6I * dm.y;
    x13.y += Q17i4j6R * dp.y - Q17i4j6I * dm.x;
    x5.x += Q17i5j6R * dp.x - Q17i5j6I * dm.y;
    x5.y += Q17i5j6R * dp.y + Q17i5j6I * dm.x;
    x12.x += Q17i5j6R * dp.x + Q17i5j6I * dm.y;
    x12.y += Q17i5j6R * dp.y - Q17i5j6I * dm.x;
    x6.x += Q17i6j6R * dp.x - Q17i6j6I * dm.y;
    x6.y += Q17i6j6R * dp.y + Q17i6j6I * dm.x;
    x11.x += Q17i6j6R * dp.x + Q17i6j6I * dm.y;
    x11.y += Q17i6j6R * dp.y - Q17i6j6I * dm.x;
    x7.x += Q17i7j6R * dp.x - Q17i7j6I * dm.y;
    x7.y += Q17i7j6R * dp.y + Q17i7j6I * dm.x;
    x10.x += Q17i7j6R * dp.x + Q17i7j6I * dm.y;
    x10.y += Q17i7j6R * dp.y - Q17i7j6I * dm.x;
    x8.x += Q17i8j6R * dp.x - Q17i8j6I * dm.y;
    x8.y += Q17i8j6R * dp.y + Q17i8j6I * dm.x;
    x9.x += Q17i8j6R * dp.x + Q17i8j6I * dm.y;
    x9.y += Q17i8j6R * dp.y - Q17i8j6I * dm.x;
    dp = (*R7) + (*R10);
    dm = (*R7) - (*R10);
    x1.x += Q17i1j7R * dp.x - Q17i1j7I * dm.y;
    x1.y += Q17i1j7R * dp.y + Q17i1j7I * dm.x;
    x16.x += Q17i1j7R * dp.x + Q17i1j7I * dm.y;
    x16.y += Q17i1j7R * dp.y - Q17i1j7I * dm.x;
    x2.x += Q17i2j7R * dp.x - Q17i2j7I * dm.y;
    x2.y += Q17i2j7R * dp.y + Q17i2j7I * dm.x;
    x15.x += Q17i2j7R * dp.x + Q17i2j7I * dm.y;
    x15.y += Q17i2j7R * dp.y - Q17i2j7I * dm.x;
    x3.x += Q17i3j7R * dp.x - Q17i3j7I * dm.y;
    x3.y += Q17i3j7R * dp.y + Q17i3j7I * dm.x;
    x14.x += Q17i3j7R * dp.x + Q17i3j7I * dm.y;
    x14.y += Q17i3j7R * dp.y - Q17i3j7I * dm.x;
    x4.x += Q17i4j7R * dp.x - Q17i4j7I * dm.y;
    x4.y += Q17i4j7R * dp.y + Q17i4j7I * dm.x;
    x13.x += Q17i4j7R * dp.x + Q17i4j7I * dm.y;
    x13.y += Q17i4j7R * dp.y - Q17i4j7I * dm.x;
    x5.x += Q17i5j7R * dp.x - Q17i5j7I * dm.y;
    x5.y += Q17i5j7R * dp.y + Q17i5j7I * dm.x;
    x12.x += Q17i5j7R * dp.x + Q17i5j7I * dm.y;
    x12.y += Q17i5j7R * dp.y - Q17i5j7I * dm.x;
    x6.x += Q17i6j7R * dp.x - Q17i6j7I * dm.y;
    x6.y += Q17i6j7R * dp.y + Q17i6j7I * dm.x;
    x11.x += Q17i6j7R * dp.x + Q17i6j7I * dm.y;
    x11.y += Q17i6j7R * dp.y - Q17i6j7I * dm.x;
    x7.x += Q17i7j7R * dp.x - Q17i7j7I * dm.y;
    x7.y += Q17i7j7R * dp.y + Q17i7j7I * dm.x;
    x10.x += Q17i7j7R * dp.x + Q17i7j7I * dm.y;
    x10.y += Q17i7j7R * dp.y - Q17i7j7I * dm.x;
    x8.x += Q17i8j7R * dp.x - Q17i8j7I * dm.y;
    x8.y += Q17i8j7R * dp.y + Q17i8j7I * dm.x;
    x9.x += Q17i8j7R * dp.x + Q17i8j7I * dm.y;
    x9.y += Q17i8j7R * dp.y - Q17i8j7I * dm.x;
    dp = (*R8) + (*R9);
    dm = (*R8) - (*R9);
    x1.x += Q17i1j8R * dp.x - Q17i1j8I * dm.y;
    x1.y += Q17i1j8R * dp.y + Q17i1j8I * dm.x;
    x16.x += Q17i1j8R * dp.x + Q17i1j8I * dm.y;
    x16.y += Q17i1j8R * dp.y - Q17i1j8I * dm.x;
    x2.x += Q17i2j8R * dp.x - Q17i2j8I * dm.y;
    x2.y += Q17i2j8R * dp.y + Q17i2j8I * dm.x;
    x15.x += Q17i2j8R * dp.x + Q17i2j8I * dm.y;
    x15.y += Q17i2j8R * dp.y - Q17i2j8I * dm.x;
    x3.x += Q17i3j8R * dp.x - Q17i3j8I * dm.y;
    x3.y += Q17i3j8R * dp.y + Q17i3j8I * dm.x;
    x14.x += Q17i3j8R * dp.x + Q17i3j8I * dm.y;
    x14.y += Q17i3j8R * dp.y - Q17i3j8I * dm.x;
    x4.x += Q17i4j8R * dp.x - Q17i4j8I * dm.y;
    x4.y += Q17i4j8R * dp.y + Q17i4j8I * dm.x;
    x13.x += Q17i4j8R * dp.x + Q17i4j8I * dm.y;
    x13.y += Q17i4j8R * dp.y - Q17i4j8I * dm.x;
    x5.x += Q17i5j8R * dp.x - Q17i5j8I * dm.y;
    x5.y += Q17i5j8R * dp.y + Q17i5j8I * dm.x;
    x12.x += Q17i5j8R * dp.x + Q17i5j8I * dm.y;
    x12.y += Q17i5j8R * dp.y - Q17i5j8I * dm.x;
    x6.x += Q17i6j8R * dp.x - Q17i6j8I * dm.y;
    x6.y += Q17i6j8R * dp.y + Q17i6j8I * dm.x;
    x11.x += Q17i6j8R * dp.x + Q17i6j8I * dm.y;
    x11.y += Q17i6j8R * dp.y - Q17i6j8I * dm.x;
    x7.x += Q17i7j8R * dp.x - Q17i7j8I * dm.y;
    x7.y += Q17i7j8R * dp.y + Q17i7j8I * dm.x;
    x10.x += Q17i7j8R * dp.x + Q17i7j8I * dm.y;
    x10.y += Q17i7j8R * dp.y - Q17i7j8I * dm.x;
    x8.x += Q17i8j8R * dp.x - Q17i8j8I * dm.y;
    x8.y += Q17i8j8R * dp.y + Q17i8j8I * dm.x;
    x9.x += Q17i8j8R * dp.x + Q17i8j8I * dm.y;
    x9.y += Q17i8j8R * dp.y - Q17i8j8I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
    (*R13) = x13;
    (*R14) = x14;
    (*R15) = x15;
    (*R16) = x16;
}

template <typename T>
__device__ void InvRad17B1(T* R0,
                           T* R1,
                           T* R2,
                           T* R3,
                           T* R4,
                           T* R5,
                           T* R6,
                           T* R7,
                           T* R8,
                           T* R9,
                           T* R10,
                           T* R11,
                           T* R12,
                           T* R13,
                           T* R14,
                           T* R15,
                           T* R16)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12) + (*R13) + (*R14) + (*R15) + (*R16);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    x13 = (*R0);
    x14 = (*R0);
    x15 = (*R0);
    x16 = (*R0);
    dp  = (*R1) + (*R16);
    dm  = (*R1) - (*R16);
    x1.x += Q17i1j1R * dp.x + Q17i1j1I * dm.y;
    x1.y += Q17i1j1R * dp.y - Q17i1j1I * dm.x;
    x16.x += Q17i1j1R * dp.x - Q17i1j1I * dm.y;
    x16.y += Q17i1j1R * dp.y + Q17i1j1I * dm.x;
    x2.x += Q17i2j1R * dp.x + Q17i2j1I * dm.y;
    x2.y += Q17i2j1R * dp.y - Q17i2j1I * dm.x;
    x15.x += Q17i2j1R * dp.x - Q17i2j1I * dm.y;
    x15.y += Q17i2j1R * dp.y + Q17i2j1I * dm.x;
    x3.x += Q17i3j1R * dp.x + Q17i3j1I * dm.y;
    x3.y += Q17i3j1R * dp.y - Q17i3j1I * dm.x;
    x14.x += Q17i3j1R * dp.x - Q17i3j1I * dm.y;
    x14.y += Q17i3j1R * dp.y + Q17i3j1I * dm.x;
    x4.x += Q17i4j1R * dp.x + Q17i4j1I * dm.y;
    x4.y += Q17i4j1R * dp.y - Q17i4j1I * dm.x;
    x13.x += Q17i4j1R * dp.x - Q17i4j1I * dm.y;
    x13.y += Q17i4j1R * dp.y + Q17i4j1I * dm.x;
    x5.x += Q17i5j1R * dp.x + Q17i5j1I * dm.y;
    x5.y += Q17i5j1R * dp.y - Q17i5j1I * dm.x;
    x12.x += Q17i5j1R * dp.x - Q17i5j1I * dm.y;
    x12.y += Q17i5j1R * dp.y + Q17i5j1I * dm.x;
    x6.x += Q17i6j1R * dp.x + Q17i6j1I * dm.y;
    x6.y += Q17i6j1R * dp.y - Q17i6j1I * dm.x;
    x11.x += Q17i6j1R * dp.x - Q17i6j1I * dm.y;
    x11.y += Q17i6j1R * dp.y + Q17i6j1I * dm.x;
    x7.x += Q17i7j1R * dp.x + Q17i7j1I * dm.y;
    x7.y += Q17i7j1R * dp.y - Q17i7j1I * dm.x;
    x10.x += Q17i7j1R * dp.x - Q17i7j1I * dm.y;
    x10.y += Q17i7j1R * dp.y + Q17i7j1I * dm.x;
    x8.x += Q17i8j1R * dp.x + Q17i8j1I * dm.y;
    x8.y += Q17i8j1R * dp.y - Q17i8j1I * dm.x;
    x9.x += Q17i8j1R * dp.x - Q17i8j1I * dm.y;
    x9.y += Q17i8j1R * dp.y + Q17i8j1I * dm.x;
    dp = (*R2) + (*R15);
    dm = (*R2) - (*R15);
    x1.x += Q17i1j2R * dp.x + Q17i1j2I * dm.y;
    x1.y += Q17i1j2R * dp.y - Q17i1j2I * dm.x;
    x16.x += Q17i1j2R * dp.x - Q17i1j2I * dm.y;
    x16.y += Q17i1j2R * dp.y + Q17i1j2I * dm.x;
    x2.x += Q17i2j2R * dp.x + Q17i2j2I * dm.y;
    x2.y += Q17i2j2R * dp.y - Q17i2j2I * dm.x;
    x15.x += Q17i2j2R * dp.x - Q17i2j2I * dm.y;
    x15.y += Q17i2j2R * dp.y + Q17i2j2I * dm.x;
    x3.x += Q17i3j2R * dp.x + Q17i3j2I * dm.y;
    x3.y += Q17i3j2R * dp.y - Q17i3j2I * dm.x;
    x14.x += Q17i3j2R * dp.x - Q17i3j2I * dm.y;
    x14.y += Q17i3j2R * dp.y + Q17i3j2I * dm.x;
    x4.x += Q17i4j2R * dp.x + Q17i4j2I * dm.y;
    x4.y += Q17i4j2R * dp.y - Q17i4j2I * dm.x;
    x13.x += Q17i4j2R * dp.x - Q17i4j2I * dm.y;
    x13.y += Q17i4j2R * dp.y + Q17i4j2I * dm.x;
    x5.x += Q17i5j2R * dp.x + Q17i5j2I * dm.y;
    x5.y += Q17i5j2R * dp.y - Q17i5j2I * dm.x;
    x12.x += Q17i5j2R * dp.x - Q17i5j2I * dm.y;
    x12.y += Q17i5j2R * dp.y + Q17i5j2I * dm.x;
    x6.x += Q17i6j2R * dp.x + Q17i6j2I * dm.y;
    x6.y += Q17i6j2R * dp.y - Q17i6j2I * dm.x;
    x11.x += Q17i6j2R * dp.x - Q17i6j2I * dm.y;
    x11.y += Q17i6j2R * dp.y + Q17i6j2I * dm.x;
    x7.x += Q17i7j2R * dp.x + Q17i7j2I * dm.y;
    x7.y += Q17i7j2R * dp.y - Q17i7j2I * dm.x;
    x10.x += Q17i7j2R * dp.x - Q17i7j2I * dm.y;
    x10.y += Q17i7j2R * dp.y + Q17i7j2I * dm.x;
    x8.x += Q17i8j2R * dp.x + Q17i8j2I * dm.y;
    x8.y += Q17i8j2R * dp.y - Q17i8j2I * dm.x;
    x9.x += Q17i8j2R * dp.x - Q17i8j2I * dm.y;
    x9.y += Q17i8j2R * dp.y + Q17i8j2I * dm.x;
    dp = (*R3) + (*R14);
    dm = (*R3) - (*R14);
    x1.x += Q17i1j3R * dp.x + Q17i1j3I * dm.y;
    x1.y += Q17i1j3R * dp.y - Q17i1j3I * dm.x;
    x16.x += Q17i1j3R * dp.x - Q17i1j3I * dm.y;
    x16.y += Q17i1j3R * dp.y + Q17i1j3I * dm.x;
    x2.x += Q17i2j3R * dp.x + Q17i2j3I * dm.y;
    x2.y += Q17i2j3R * dp.y - Q17i2j3I * dm.x;
    x15.x += Q17i2j3R * dp.x - Q17i2j3I * dm.y;
    x15.y += Q17i2j3R * dp.y + Q17i2j3I * dm.x;
    x3.x += Q17i3j3R * dp.x + Q17i3j3I * dm.y;
    x3.y += Q17i3j3R * dp.y - Q17i3j3I * dm.x;
    x14.x += Q17i3j3R * dp.x - Q17i3j3I * dm.y;
    x14.y += Q17i3j3R * dp.y + Q17i3j3I * dm.x;
    x4.x += Q17i4j3R * dp.x + Q17i4j3I * dm.y;
    x4.y += Q17i4j3R * dp.y - Q17i4j3I * dm.x;
    x13.x += Q17i4j3R * dp.x - Q17i4j3I * dm.y;
    x13.y += Q17i4j3R * dp.y + Q17i4j3I * dm.x;
    x5.x += Q17i5j3R * dp.x + Q17i5j3I * dm.y;
    x5.y += Q17i5j3R * dp.y - Q17i5j3I * dm.x;
    x12.x += Q17i5j3R * dp.x - Q17i5j3I * dm.y;
    x12.y += Q17i5j3R * dp.y + Q17i5j3I * dm.x;
    x6.x += Q17i6j3R * dp.x + Q17i6j3I * dm.y;
    x6.y += Q17i6j3R * dp.y - Q17i6j3I * dm.x;
    x11.x += Q17i6j3R * dp.x - Q17i6j3I * dm.y;
    x11.y += Q17i6j3R * dp.y + Q17i6j3I * dm.x;
    x7.x += Q17i7j3R * dp.x + Q17i7j3I * dm.y;
    x7.y += Q17i7j3R * dp.y - Q17i7j3I * dm.x;
    x10.x += Q17i7j3R * dp.x - Q17i7j3I * dm.y;
    x10.y += Q17i7j3R * dp.y + Q17i7j3I * dm.x;
    x8.x += Q17i8j3R * dp.x + Q17i8j3I * dm.y;
    x8.y += Q17i8j3R * dp.y - Q17i8j3I * dm.x;
    x9.x += Q17i8j3R * dp.x - Q17i8j3I * dm.y;
    x9.y += Q17i8j3R * dp.y + Q17i8j3I * dm.x;
    dp = (*R4) + (*R13);
    dm = (*R4) - (*R13);
    x1.x += Q17i1j4R * dp.x + Q17i1j4I * dm.y;
    x1.y += Q17i1j4R * dp.y - Q17i1j4I * dm.x;
    x16.x += Q17i1j4R * dp.x - Q17i1j4I * dm.y;
    x16.y += Q17i1j4R * dp.y + Q17i1j4I * dm.x;
    x2.x += Q17i2j4R * dp.x + Q17i2j4I * dm.y;
    x2.y += Q17i2j4R * dp.y - Q17i2j4I * dm.x;
    x15.x += Q17i2j4R * dp.x - Q17i2j4I * dm.y;
    x15.y += Q17i2j4R * dp.y + Q17i2j4I * dm.x;
    x3.x += Q17i3j4R * dp.x + Q17i3j4I * dm.y;
    x3.y += Q17i3j4R * dp.y - Q17i3j4I * dm.x;
    x14.x += Q17i3j4R * dp.x - Q17i3j4I * dm.y;
    x14.y += Q17i3j4R * dp.y + Q17i3j4I * dm.x;
    x4.x += Q17i4j4R * dp.x + Q17i4j4I * dm.y;
    x4.y += Q17i4j4R * dp.y - Q17i4j4I * dm.x;
    x13.x += Q17i4j4R * dp.x - Q17i4j4I * dm.y;
    x13.y += Q17i4j4R * dp.y + Q17i4j4I * dm.x;
    x5.x += Q17i5j4R * dp.x + Q17i5j4I * dm.y;
    x5.y += Q17i5j4R * dp.y - Q17i5j4I * dm.x;
    x12.x += Q17i5j4R * dp.x - Q17i5j4I * dm.y;
    x12.y += Q17i5j4R * dp.y + Q17i5j4I * dm.x;
    x6.x += Q17i6j4R * dp.x + Q17i6j4I * dm.y;
    x6.y += Q17i6j4R * dp.y - Q17i6j4I * dm.x;
    x11.x += Q17i6j4R * dp.x - Q17i6j4I * dm.y;
    x11.y += Q17i6j4R * dp.y + Q17i6j4I * dm.x;
    x7.x += Q17i7j4R * dp.x + Q17i7j4I * dm.y;
    x7.y += Q17i7j4R * dp.y - Q17i7j4I * dm.x;
    x10.x += Q17i7j4R * dp.x - Q17i7j4I * dm.y;
    x10.y += Q17i7j4R * dp.y + Q17i7j4I * dm.x;
    x8.x += Q17i8j4R * dp.x + Q17i8j4I * dm.y;
    x8.y += Q17i8j4R * dp.y - Q17i8j4I * dm.x;
    x9.x += Q17i8j4R * dp.x - Q17i8j4I * dm.y;
    x9.y += Q17i8j4R * dp.y + Q17i8j4I * dm.x;
    dp = (*R5) + (*R12);
    dm = (*R5) - (*R12);
    x1.x += Q17i1j5R * dp.x + Q17i1j5I * dm.y;
    x1.y += Q17i1j5R * dp.y - Q17i1j5I * dm.x;
    x16.x += Q17i1j5R * dp.x - Q17i1j5I * dm.y;
    x16.y += Q17i1j5R * dp.y + Q17i1j5I * dm.x;
    x2.x += Q17i2j5R * dp.x + Q17i2j5I * dm.y;
    x2.y += Q17i2j5R * dp.y - Q17i2j5I * dm.x;
    x15.x += Q17i2j5R * dp.x - Q17i2j5I * dm.y;
    x15.y += Q17i2j5R * dp.y + Q17i2j5I * dm.x;
    x3.x += Q17i3j5R * dp.x + Q17i3j5I * dm.y;
    x3.y += Q17i3j5R * dp.y - Q17i3j5I * dm.x;
    x14.x += Q17i3j5R * dp.x - Q17i3j5I * dm.y;
    x14.y += Q17i3j5R * dp.y + Q17i3j5I * dm.x;
    x4.x += Q17i4j5R * dp.x + Q17i4j5I * dm.y;
    x4.y += Q17i4j5R * dp.y - Q17i4j5I * dm.x;
    x13.x += Q17i4j5R * dp.x - Q17i4j5I * dm.y;
    x13.y += Q17i4j5R * dp.y + Q17i4j5I * dm.x;
    x5.x += Q17i5j5R * dp.x + Q17i5j5I * dm.y;
    x5.y += Q17i5j5R * dp.y - Q17i5j5I * dm.x;
    x12.x += Q17i5j5R * dp.x - Q17i5j5I * dm.y;
    x12.y += Q17i5j5R * dp.y + Q17i5j5I * dm.x;
    x6.x += Q17i6j5R * dp.x + Q17i6j5I * dm.y;
    x6.y += Q17i6j5R * dp.y - Q17i6j5I * dm.x;
    x11.x += Q17i6j5R * dp.x - Q17i6j5I * dm.y;
    x11.y += Q17i6j5R * dp.y + Q17i6j5I * dm.x;
    x7.x += Q17i7j5R * dp.x + Q17i7j5I * dm.y;
    x7.y += Q17i7j5R * dp.y - Q17i7j5I * dm.x;
    x10.x += Q17i7j5R * dp.x - Q17i7j5I * dm.y;
    x10.y += Q17i7j5R * dp.y + Q17i7j5I * dm.x;
    x8.x += Q17i8j5R * dp.x + Q17i8j5I * dm.y;
    x8.y += Q17i8j5R * dp.y - Q17i8j5I * dm.x;
    x9.x += Q17i8j5R * dp.x - Q17i8j5I * dm.y;
    x9.y += Q17i8j5R * dp.y + Q17i8j5I * dm.x;
    dp = (*R6) + (*R11);
    dm = (*R6) - (*R11);
    x1.x += Q17i1j6R * dp.x + Q17i1j6I * dm.y;
    x1.y += Q17i1j6R * dp.y - Q17i1j6I * dm.x;
    x16.x += Q17i1j6R * dp.x - Q17i1j6I * dm.y;
    x16.y += Q17i1j6R * dp.y + Q17i1j6I * dm.x;
    x2.x += Q17i2j6R * dp.x + Q17i2j6I * dm.y;
    x2.y += Q17i2j6R * dp.y - Q17i2j6I * dm.x;
    x15.x += Q17i2j6R * dp.x - Q17i2j6I * dm.y;
    x15.y += Q17i2j6R * dp.y + Q17i2j6I * dm.x;
    x3.x += Q17i3j6R * dp.x + Q17i3j6I * dm.y;
    x3.y += Q17i3j6R * dp.y - Q17i3j6I * dm.x;
    x14.x += Q17i3j6R * dp.x - Q17i3j6I * dm.y;
    x14.y += Q17i3j6R * dp.y + Q17i3j6I * dm.x;
    x4.x += Q17i4j6R * dp.x + Q17i4j6I * dm.y;
    x4.y += Q17i4j6R * dp.y - Q17i4j6I * dm.x;
    x13.x += Q17i4j6R * dp.x - Q17i4j6I * dm.y;
    x13.y += Q17i4j6R * dp.y + Q17i4j6I * dm.x;
    x5.x += Q17i5j6R * dp.x + Q17i5j6I * dm.y;
    x5.y += Q17i5j6R * dp.y - Q17i5j6I * dm.x;
    x12.x += Q17i5j6R * dp.x - Q17i5j6I * dm.y;
    x12.y += Q17i5j6R * dp.y + Q17i5j6I * dm.x;
    x6.x += Q17i6j6R * dp.x + Q17i6j6I * dm.y;
    x6.y += Q17i6j6R * dp.y - Q17i6j6I * dm.x;
    x11.x += Q17i6j6R * dp.x - Q17i6j6I * dm.y;
    x11.y += Q17i6j6R * dp.y + Q17i6j6I * dm.x;
    x7.x += Q17i7j6R * dp.x + Q17i7j6I * dm.y;
    x7.y += Q17i7j6R * dp.y - Q17i7j6I * dm.x;
    x10.x += Q17i7j6R * dp.x - Q17i7j6I * dm.y;
    x10.y += Q17i7j6R * dp.y + Q17i7j6I * dm.x;
    x8.x += Q17i8j6R * dp.x + Q17i8j6I * dm.y;
    x8.y += Q17i8j6R * dp.y - Q17i8j6I * dm.x;
    x9.x += Q17i8j6R * dp.x - Q17i8j6I * dm.y;
    x9.y += Q17i8j6R * dp.y + Q17i8j6I * dm.x;
    dp = (*R7) + (*R10);
    dm = (*R7) - (*R10);
    x1.x += Q17i1j7R * dp.x + Q17i1j7I * dm.y;
    x1.y += Q17i1j7R * dp.y - Q17i1j7I * dm.x;
    x16.x += Q17i1j7R * dp.x - Q17i1j7I * dm.y;
    x16.y += Q17i1j7R * dp.y + Q17i1j7I * dm.x;
    x2.x += Q17i2j7R * dp.x + Q17i2j7I * dm.y;
    x2.y += Q17i2j7R * dp.y - Q17i2j7I * dm.x;
    x15.x += Q17i2j7R * dp.x - Q17i2j7I * dm.y;
    x15.y += Q17i2j7R * dp.y + Q17i2j7I * dm.x;
    x3.x += Q17i3j7R * dp.x + Q17i3j7I * dm.y;
    x3.y += Q17i3j7R * dp.y - Q17i3j7I * dm.x;
    x14.x += Q17i3j7R * dp.x - Q17i3j7I * dm.y;
    x14.y += Q17i3j7R * dp.y + Q17i3j7I * dm.x;
    x4.x += Q17i4j7R * dp.x + Q17i4j7I * dm.y;
    x4.y += Q17i4j7R * dp.y - Q17i4j7I * dm.x;
    x13.x += Q17i4j7R * dp.x - Q17i4j7I * dm.y;
    x13.y += Q17i4j7R * dp.y + Q17i4j7I * dm.x;
    x5.x += Q17i5j7R * dp.x + Q17i5j7I * dm.y;
    x5.y += Q17i5j7R * dp.y - Q17i5j7I * dm.x;
    x12.x += Q17i5j7R * dp.x - Q17i5j7I * dm.y;
    x12.y += Q17i5j7R * dp.y + Q17i5j7I * dm.x;
    x6.x += Q17i6j7R * dp.x + Q17i6j7I * dm.y;
    x6.y += Q17i6j7R * dp.y - Q17i6j7I * dm.x;
    x11.x += Q17i6j7R * dp.x - Q17i6j7I * dm.y;
    x11.y += Q17i6j7R * dp.y + Q17i6j7I * dm.x;
    x7.x += Q17i7j7R * dp.x + Q17i7j7I * dm.y;
    x7.y += Q17i7j7R * dp.y - Q17i7j7I * dm.x;
    x10.x += Q17i7j7R * dp.x - Q17i7j7I * dm.y;
    x10.y += Q17i7j7R * dp.y + Q17i7j7I * dm.x;
    x8.x += Q17i8j7R * dp.x + Q17i8j7I * dm.y;
    x8.y += Q17i8j7R * dp.y - Q17i8j7I * dm.x;
    x9.x += Q17i8j7R * dp.x - Q17i8j7I * dm.y;
    x9.y += Q17i8j7R * dp.y + Q17i8j7I * dm.x;
    dp = (*R8) + (*R9);
    dm = (*R8) - (*R9);
    x1.x += Q17i1j8R * dp.x + Q17i1j8I * dm.y;
    x1.y += Q17i1j8R * dp.y - Q17i1j8I * dm.x;
    x16.x += Q17i1j8R * dp.x - Q17i1j8I * dm.y;
    x16.y += Q17i1j8R * dp.y + Q17i1j8I * dm.x;
    x2.x += Q17i2j8R * dp.x + Q17i2j8I * dm.y;
    x2.y += Q17i2j8R * dp.y - Q17i2j8I * dm.x;
    x15.x += Q17i2j8R * dp.x - Q17i2j8I * dm.y;
    x15.y += Q17i2j8R * dp.y + Q17i2j8I * dm.x;
    x3.x += Q17i3j8R * dp.x + Q17i3j8I * dm.y;
    x3.y += Q17i3j8R * dp.y - Q17i3j8I * dm.x;
    x14.x += Q17i3j8R * dp.x - Q17i3j8I * dm.y;
    x14.y += Q17i3j8R * dp.y + Q17i3j8I * dm.x;
    x4.x += Q17i4j8R * dp.x + Q17i4j8I * dm.y;
    x4.y += Q17i4j8R * dp.y - Q17i4j8I * dm.x;
    x13.x += Q17i4j8R * dp.x - Q17i4j8I * dm.y;
    x13.y += Q17i4j8R * dp.y + Q17i4j8I * dm.x;
    x5.x += Q17i5j8R * dp.x + Q17i5j8I * dm.y;
    x5.y += Q17i5j8R * dp.y - Q17i5j8I * dm.x;
    x12.x += Q17i5j8R * dp.x - Q17i5j8I * dm.y;
    x12.y += Q17i5j8R * dp.y + Q17i5j8I * dm.x;
    x6.x += Q17i6j8R * dp.x + Q17i6j8I * dm.y;
    x6.y += Q17i6j8R * dp.y - Q17i6j8I * dm.x;
    x11.x += Q17i6j8R * dp.x - Q17i6j8I * dm.y;
    x11.y += Q17i6j8R * dp.y + Q17i6j8I * dm.x;
    x7.x += Q17i7j8R * dp.x + Q17i7j8I * dm.y;
    x7.y += Q17i7j8R * dp.y - Q17i7j8I * dm.x;
    x10.x += Q17i7j8R * dp.x - Q17i7j8I * dm.y;
    x10.y += Q17i7j8R * dp.y + Q17i7j8I * dm.x;
    x8.x += Q17i8j8R * dp.x + Q17i8j8I * dm.y;
    x8.y += Q17i8j8R * dp.y - Q17i8j8I * dm.x;
    x9.x += Q17i8j8R * dp.x - Q17i8j8I * dm.y;
    x9.y += Q17i8j8R * dp.y + Q17i8j8I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
    (*R13) = x13;
    (*R14) = x14;
    (*R15) = x15;
    (*R16) = x16;
}

#endif // ROCFFT_BUTTERFLY_TEMPLATE_H
