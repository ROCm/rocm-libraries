/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_INTERNAL_H_
#define GUARD_MIOPEN_INTERNAL_H_

/* Put experimental APIs here. */
/* Not part of the public C API: miopen.h does not include this header. */

#include <stddef.h>
#include <miopen/miopen.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Enum for specifying the alpha-beta case for convolution operations.
 *
 * Describes how alpha and beta scaling factors are applied in convolution operations.
 */
typedef enum
{
    /* DEFAULT       alpha = 1.0 and beta = 0.0 */
    /* SCALE         alpha = 4.2 and beta = 0.0 */
    /* BILINEAR      alpha = 3.2 and beta = 1.1 */
    /* ERROR_STATE   alpha = 0.0 and beta = 3.1 */

    DEFAULT     = 0, /* alpha = 1.0 and beta = 0.0.*/
    SCALE       = 1, /* alpha with some value and beta 0.0*/
    BILINEAR    = 2, /* both alpha and beta with some value*/
    ERROR_STATE = 3  /* alpha 0.0 and beta with some value, this should not occur.
                        But used to check for errors.*/
} miopenAlphaBetaCase_t;

/*! @brief Get extra workspace for backward weight kernel.
 *
 * @param  alpha_beta_case type of alpha beta case
 * @param  inputTensorDesc Input data tensor descriptor (output)
 * @param  outputTensorDesc Output data tensor descriptor (output)
 * @param  weightsTensorDesc Weights tensor descriptor
 * @param  buffer_size buffer size for CK Backward weights work space
 */
MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionABBackwardWeightsGetWorkSpaceSize(const miopenAlphaBetaCase_t alpha_beta_case,
                                                   const miopenTensorDescriptor_t inputTensorDesc,
                                                   const miopenTensorDescriptor_t outputTensorDesc,
                                                   const miopenTensorDescriptor_t weightsTensorDesc,
                                                   const miopenConvolutionDescriptor_t convDesc,
                                                   size_t* buffer_size);

#ifdef __cplusplus
}
#endif

#endif // GUARD_MIOPEN_INTERNAL_H_
