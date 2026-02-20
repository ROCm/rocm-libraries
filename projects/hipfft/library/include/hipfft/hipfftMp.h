/******************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/

#ifndef HIPFFT_MP_H
#define HIPFFT_MP_H

#include "hipfft/hipfft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Communicator type for multi-processing. */
typedef enum hipfftMpCommType_t
{
    HIPFFT_COMM_MPI  = 0x00, // MPI communicator
    HIPFFT_COMM_NONE = 0x01
} hipfftMpCommType;

/*! @brief Set a multi-processing communicator on a plan.
   *
   * @details Attach a multi-processing communication handle to a
   * hipFFT plan.  'comm_handle' points to the handle, whose type
   * depends on the multi-processing library being used.  With MPI
   * (Message Passing Interface) for example, 'comm_handle' points to
   * an MPI communicator.
   *
   * This function must only be called on a plan that has already
   * been allocated by ::hipfftCreate, but before the plan is
   * initialized with any of the 'hipfftMakePlan' functions.
   
   * @param[in] plan The FFT plan.
   * @param[in] comm_type Type of communication handle.
   * @param[in] comm_handle Pointer to the communication handle.
   *
   * @warning Experimental
   */
HIPFFT_EXPORT hipfftResult hipfftMpAttachComm(hipfftHandle     plan,
                                              hipfftMpCommType comm_type,
                                              void*            comm_handle);

/*! @brief Describe the partial input and output for a distributed transform in the current process.
   *
   * @details For the given plan, describe the 'brick' of input and
   * output that is distributed to the current process.  A brick is
   * defined with the lower and upper coordinates of the that brick
   * in the global index space of the transform.
   *
   * Strides for both the input and output bricks are also provided,
   * to describe how the brick is laid out in memory.
   *
   * All coordinates and strides are in row-major order, with the
   * slowest-moving dimension specified first.
   *
   * This function must only be called on a plan that has already
   * been allocated by ::hipfftCreate, but before the plan is
   * initialized with any of the 'hipfftMakePlan' functions.
   *
   * @param[in] plan The FFT plan.
   * @param[in] rank Dimension of the transform
   * @param[in] input_lower Array of length rank, specifying the lower index (inclusive) for the brick in the FFT input.
   * @param[in] input_upper Array of length rank, specifying the upper index (exclusive) for the brick in the FFT input.
   * @param[in] output_lower Array of length rank, specifying the lower index (inclusive) for the brick in the FFT output.
   * @param[in] output_upper Array of length rank, specifying the upper index (exclusive) for the brick in the FFT output.
   * @param[in] input_stride Array of length rank specifying the input brick's stride in memory
   * @param[in] output_stride Array of length rank specifying the output brick's stride in memory
   *
   * @warning Experimental
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetDistribution(hipfftHandle         plan,
                                                   int                  rank,
                                                   const long long int* input_lower,
                                                   const long long int* input_upper,
                                                   const long long int* output_lower,
                                                   const long long int* output_upper,
                                                   const long long int* input_stride,
                                                   const long long int* output_stride);

/*! @brief Set the input and output formats for a distributed transform.
 *
 * @details Specifies the distribution of data for each side of a
 * distributed transform.  Forward transforms use 'subformat_forward'
 * to describe input and 'subformat_inverse' to describe output.
 * Inverse transforms use 'subformat_inverse' to describe input and
 * 'subformat_forward' to describe output.
 *
 * 'subformat_forward' and 'subformat_inverse' must be set to
 * matching values.  One may be HIPFFT_XT_FORMAT_INPLACE and the
 * other HIPFFT_XT_FORMAT_INPLACE_SHUFFLED; Alternatively, one may be
 * HIPFFT_XT_FORMAT_DISTRIBUTED_INPUT and the other
 * HIPFFT_XT_FORMAT_DISTRIBUTED_OUTPUT.
 *
 * This function must only be called on a plan that has already
 * been allocated by ::hipfftCreate, but before the plan is
 * initialized with any of the 'hipfftMakePlan' functions.
 *
 * @param[in] plan The FFT plan.
 * @param[in] subformat_forward Format of the input for a forward transform.
 * @param[in] subformat_inverse Format of the input for an inverse transform.
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtSetSubformatDefault(hipfftHandle      plan,
                                                       hipfftXtSubFormat subformat_forward,
                                                       hipfftXtSubFormat subformat_inverse);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
