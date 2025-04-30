// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPRAND_KERNEL_H_
#define HIPRAND_KERNEL_H_

#ifndef QUALIFIERS
#define QUALIFIERS __forceinline__ __device__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

/** \defgroup hipranddevice hipRAND device functions
 *
 *  @{
 */

/**
 * \def HIPRAND_PHILOX4x32_DEFAULT_SEED
 * \brief Default seed for PHILOX4x32 PRNG.
 */
#define HIPRAND_PHILOX4x32_DEFAULT_SEED 0ULL
/**
 * \def HIPRAND_XORWOW_DEFAULT_SEED
 * \brief Default seed for XORWOW PRNG.
 */
#define HIPRAND_XORWOW_DEFAULT_SEED 0ULL
/**
 * \def HIPRAND_MRG32K3A_DEFAULT_SEED
 * \brief Default seed for MRG32K3A PRNG.
 */
#define HIPRAND_MRG32K3A_DEFAULT_SEED 12345ULL
/**
 * \def HIPRAND_MTGP32_DEFAULT_SEED
 * \brief Default seed for MTGP32 PRNG.
 */
#define HIPRAND_MTGP32_DEFAULT_SEED 0ULL
/**
 * \def HIPRAND_MT19937_DEFAULT_SEED
 * \brief Default seed for MT19937 PRNG.
 */
#define HIPRAND_MT19937_DEFAULT_SEED 0ULL

/**
 * \brief XORWOW PRNG state type
 */
typedef struct hiprandStateXORWOW hiprandStateXORWOW_t;

/**
 * \brief Deprecated alias of hiprandStateXORWOW_t
 * \deprecated Please use hiprandStateXORWOW_t directly.
 */
typedef hiprandStateXORWOW_t hipRandState_t;

/**
 * \brief PHILOX PRNG state type
 */
typedef struct hiprandStatePhilox4_32_10 hiprandStatePhilox4_32_10_t;

/**
 * \brief MRG32k3a PRNG state type
 */
typedef struct hiprandStateMRG32k3a hiprandStateMRG32k3a_t;

/**
 * \brief MTGP32 PRNG state type
 */
typedef struct hiprandStateMtgp32 hiprandStateMtgp32_t;

/**
 * \brief Scrambled SOBOL32 QRNG state type
 */
typedef struct hiprandStateScrambledSobol32 hiprandStateScrambledSobol32_t;

/**
 * \brief Scrambled SOBOL64 QRNG state type
 */
typedef struct hiprandStateScrambledSobol64 hiprandStateScrambledSobol64_t;

/**
 * \brief SOBOL32 QRNG state type
 */
typedef struct hiprandStateSobol32 hiprandStateSobol32_t;

/**
 * \brief SOBOL64 QRNG state type
 */
typedef struct hiprandStateSobol64 hiprandStateSobol64_t;
/** @} */ // end of group hipranddevice

#if defined(__HIP_PLATFORM_AMD__)
    #include "hiprand/hiprand_kernel_rocm.h"
#else
    #include "hiprand/hiprand_kernel_nvcc.h"
#endif

#endif // HIPRAND_KERNEL_H_
