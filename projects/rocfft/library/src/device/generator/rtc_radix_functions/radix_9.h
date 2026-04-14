/*******************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
