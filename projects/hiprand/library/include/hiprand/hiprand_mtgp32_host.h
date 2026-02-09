// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPRAND_MTGP32_HOST_H_
#define HIPRAND_MTGP32_HOST_H_

/// \addtogroup hipranddevice
/// @{

#if defined(__HIP_PLATFORM_AMD__) || HIPRAND_DOXYGEN
    #include "hiprand/hiprand_kernel.h"

/// \cond
typedef mtgp32_params mtgp32_kernel_params_t;
typedef mtgp32_fast_params mtgp32_fast_param_t;
/// \endcond

/**
 * \brief Loads parameters for MTGP32
 *
 * Loads parameters for use by kernel functions on the host-side and copies the
 * results to the specified location in device memory.
 *
 * \param[in] params Pointer to an array of type mtgp32_params_fast_t allocated in host memory
 * \param[out] p Pointer to a mtgp32_kernel_params_t structure allocated in device memory
 *
 * \return
 * - HIPRAND_STATUS_ALLOCATION_FAILED if parameters could not be loaded
 * - HIPRAND_STATUS_SUCCESS if parameters are loaded
 */
inline hiprandStatus_t hiprandMakeMTGP32Constants(const mtgp32_params_fast_t params[],
                                                  mtgp32_kernel_params_t*    p)
{
    return to_hiprand_status(rocrand_make_constant(params, p));
}

/**
 * \brief Initializes MTGP32 states
 *
 * Initializes MTGP32 states on the host-side by allocating a state array in host
 * memory, initializes that array, and copies the result to device memory.
 *
 * \param[out] s Pointer to an array of states in device memory
 * \param[in] params Pointer to an array of type mtgp32_params_fast_t in host memory
 * \param[in] k Pointer to a mtgp32_kernel_params_t structure allocated in device memory
 * \param n Number of states to initialize
 * \param seed Seed value
 *
 * - On the cuRAND backend `k` must be initialized by calling hiprandMakeMTGP32Constants() with the
 *   the same `params`
 * - On the rocRAND backend `k` is ignored
 *
 * For maximum portability application should initialize `k` as required by cuRAND, but when only
 * targeting the rocRAND backend it may be left uninitialized.
 *
 * \return
 * - HIPRAND_STATUS_ALLOCATION_FAILED if states could not be initialized
 * - HIPRAND_STATUS_SUCCESS if states are initialized
 */
inline hiprandStatus_t hiprandMakeMTGP32KernelState(hiprandStateMtgp32_t* s,
                                                    mtgp32_params_fast_t  params[],
                                                    [[maybe_unused]] mtgp32_kernel_params_t* k,
                                                    int                                      n,
                                                    unsigned long long                       seed)
{
    return to_hiprand_status(rocrand_make_state_mtgp32(s, params, n, seed));
}

#else // for HIP NVCC platfrom

#include "hiprand/hiprand_kernel.h"
#include <curand_mtgp32_host.h>

__forceinline__ __host__
hiprandStatus_t hiprandMakeMTGP32Constants(const mtgp32_params_fast_t params[],
                                           mtgp32_kernel_params_t * p)
{
    return to_hiprand_status(
        curandMakeMTGP32Constants(params, p)
    );
}

__forceinline__ __host__
hiprandStatus_t hiprandMakeMTGP32KernelState(hiprandStateMtgp32_t *s,
                                             mtgp32_params_fast_t params[],
                                             mtgp32_kernel_params_t *k,
                                             int n,
                                             unsigned long long seed)
{
    return to_hiprand_status(
        curandMakeMTGP32KernelState(s, params, k, n, seed)
    );
}
#endif // __HIP_PLATFORM_AMD__

/// @} // end of group hipranddevice

#endif // HIPRAND_MTGP32_HOST_H_
