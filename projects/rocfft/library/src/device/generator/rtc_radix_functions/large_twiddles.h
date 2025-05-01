/*******************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
    static_assert(Steps < 4, "4-steps is not support");
    // if(Steps >= 4){...}

    return result;
}
