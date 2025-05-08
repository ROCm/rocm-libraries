/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
