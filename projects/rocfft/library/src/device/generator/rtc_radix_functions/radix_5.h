/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
