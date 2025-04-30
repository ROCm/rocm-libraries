/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include <chrono>

#include "hipsolver.h"
#include "utility.hpp"

hipsolver_rng_t hipsolver_rng(69069);
hipsolver_rng_t hipsolver_seed(hipsolver_rng);

template <>
char type2char<float>()
{
    return 's';
}

template <>
char type2char<double>()
{
    return 'd';
}

template <>
char type2char<hipsolverComplex>()
{
    return 'c';
}

template <>
char type2char<hipsolverDoubleComplex>()
{
    return 'z';
}

template <>
int type2int<float>(float val)
{
    return (int)val;
}

template <>
int type2int<double>(double val)
{
    return (int)val;
}

template <>
int type2int<hipsolverComplex>(hipsolverComplex val)
{
    return (int)val.real();
}

template <>
int type2int<hipsolverDoubleComplex>(hipsolverDoubleComplex val)
{
    return (int)val.real();
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing:*/

/* CPU Timer (in microseconds): no GPU synchronization
 */
double get_time_us_no_sync()
{
    namespace sc                         = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/* CPU Timer (in microseconds): synchronize with the default device and return wall time
 */
double get_time_us()
{
    hipDeviceSynchronize();
    return get_time_us_no_sync();
}

/* CPU Timer (in microseconds): synchronize with given queue/stream and return wall time
 */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    return get_time_us_no_sync();
}

#ifdef __cplusplus
}
#endif
