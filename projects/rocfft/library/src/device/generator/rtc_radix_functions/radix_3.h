/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
