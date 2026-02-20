/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
