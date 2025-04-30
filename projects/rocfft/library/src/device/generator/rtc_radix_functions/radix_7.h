/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
