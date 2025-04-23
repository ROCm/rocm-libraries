/******************************************************************************
 * Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights
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

/*! @file hipfftXt.h
 *  hipfftXt.h defines extended interfaces and types for hipFFT
 *  */

#ifndef HIPFFTXT_H_
#define HIPFFTXT_H_
#include "hipfft/hipfft.h"
#include "hipfft/hiplibxt.h"

#ifdef __HIP_PLATFORM_NVIDIA__
#include <cufftXt.h>
#endif

#ifndef _WIN32
#include <cstddef>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipfftXtCallbackType_t
{
    HIPFFT_CB_LD_COMPLEX        = 0x0,
    HIPFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    HIPFFT_CB_LD_REAL           = 0x2,
    HIPFFT_CB_LD_REAL_DOUBLE    = 0x3,
    HIPFFT_CB_ST_COMPLEX        = 0x4,
    HIPFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    HIPFFT_CB_ST_REAL           = 0x6,
    HIPFFT_CB_ST_REAL_DOUBLE    = 0x7,
    HIPFFT_CB_UNDEFINED         = 0x8

} hipfftXtCallbackType;

typedef hipfftComplex (*hipfftCallbackLoadC)(void*  dataIn,
                                             size_t offset,
                                             void*  callerInfo,
                                             void*  sharedPointer);
typedef hipfftDoubleComplex (*hipfftCallbackLoadZ)(void*  dataIn,
                                                   size_t offset,
                                                   void*  callerInfo,
                                                   void*  sharedPointer);
typedef hipfftReal (*hipfftCallbackLoadR)(void*  dataIn,
                                          size_t offset,
                                          void*  callerInfo,
                                          void*  sharedPointer);
typedef hipfftDoubleReal (*hipfftCallbackLoadD)(void*  dataIn,
                                                size_t offset,
                                                void*  callerInfo,
                                                void*  sharedPointer);

typedef void (*hipfftCallbackStoreC)(
    void* dataOut, size_t offset, hipfftComplex element, void* callerInfo, void* sharedPointer);
typedef void (*hipfftCallbackStoreZ)(void*               dataOut,
                                     size_t              offset,
                                     hipfftDoubleComplex element,
                                     void*               callerInfo,
                                     void*               sharedPointer);
typedef void (*hipfftCallbackStoreR)(
    void* dataOut, size_t offset, hipfftReal element, void* callerInfo, void* sharedPointer);
typedef void (*hipfftCallbackStoreD)(
    void* dataOut, size_t offset, hipfftDoubleReal element, void* callerInfo, void* sharedPointer);

/*! @brief Set a callback on a plan
   *
   * @details Set either a load or store callback to run with a plan.
   * The type of callback is specified with the 'cbtype' parameter.
   * An array ofcallback and callback data pointers must be given -
   * one per device executing the plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] callbacks Array of callback function pointers.
   * @param[in] cbtype Type of callback being set.
   * @param[in] callbackData Array of callback function data pointers
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                               void**               callbacks,
                                               hipfftXtCallbackType cbtype,
                                               void**               callbackData);

/*! @brief Remove a callback from a plan
   *
   * @details Remove a previously-set callback from a plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] cbtype Type of callback being removed.
   */
HIPFFT_EXPORT hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype);

/*! @brief Set shared memory size for callback.
   *
   * @details Set shared memory required for a callback.  The
   * callback of the specified type must have already been set on the
   * plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] cbtype Type of callback being modified.
   * @param[in] sharedSize Amount of shared memory required, in bytes.
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetCallbackSharedSize(hipfftHandle         plan,
                                                         hipfftXtCallbackType cbtype,
                                                         size_t               sharedSize);

/*! @brief Initialize a batched rank-dimensional FFT plan with
    advanced data layout and specified input, output, execution data
    types.
   *
   *  @details Assumes that the plan has been created already, and
   *  modifies the plan associated with the plan handle. The number 
   *  of elements to transform in each direction of the input data 
   *  in the FFT plan is specified in n.
   * 
   *  The batch parameter tells hipFFT how many transforms to perform. 
   *  The distance between the first elements of two consecutive batches 
   *  of the input and output data are specified with the idist and odist 
   *  parameters.
   * 
   *  The inembed and onembed parameters define the input and output data
   *  layouts. The number of elements in the data is assumed to be larger 
   *  than the number of elements in the transform. Strided data layouts 
   *  are also supported. Strides along the fastest direction in the input
   *  and output data are specified via the istride and ostride parameters.  
   * 
   *  If both inembed and onembed parameters are set to NULL, all the 
   *  advanced data layout parameters are ignored and reverted to default 
   *  values, i.e., the batched transform is performed with non-strided data
   *  access and the number of data/transform elements are assumed to be  
   *  equivalent.
   *
   *  The inputType, outputType, executionType parameters specify the
   *  data types (precision, and whether the data is real-valued or
   *  complex-valued) of the transform input, output, and internal
   *  representation respectively.  Currently, the precision of all
   *  three parameters must match, and the execution type must always
   *  be complex-valued.  At least one of inputType and outputType
   *  must be complex.  A half-precision transform can be requested
   *  by using either the HIP_R_16F or HIP_C_16F types.
   *
   *  @param[out] plan Pointer to the FFT plan handle.
   *  @param[in] rank Dimension of transform (1, 2, or 3).
   *  @param[in] n Number of elements to transform in the x/y/z directions.
   *  @param[in] inembed Number of elements in the input data in the x/y/z directions.
   *  @param[in] istride Distance between two successive elements in the input data.
   *  @param[in] idist Distance between input batches.
   *  @param[in] inputType Format of FFT input.
   *  @param[in] onembed Number of elements in the output data in the x/y/z directions.
   *  @param[in] ostride Distance between two successive elements in the output data.
   *  @param[in] odist Distance between output batches.
   *  @param[in] outputType Format of FFT output.
   *  @param[in] batch Number of batched transforms to perform.
   *  @param[out] workSize Pointer to work area size (returned value).
   *  @param[in] executionType Internal data format used by the library during computation.
   */
HIPFFT_EXPORT hipfftResult hipfftXtMakePlanMany(hipfftHandle   plan,
                                                int            rank,
                                                long long int* n,
                                                long long int* inembed,
                                                long long int  istride,
                                                long long int  idist,
                                                hipDataType    inputType,
                                                long long int* onembed,
                                                long long int  ostride,
                                                long long int  odist,
                                                hipDataType    outputType,
                                                long long int  batch,
                                                size_t*        workSize,
                                                hipDataType    executionType);

/*! @brief Return size of the work area size required for a
    rank-dimensional plan, with specified input, output, execution
    data types.

 * @details See ::hipfftXtMakePlanMany for restrictions on inputType,
 * outputType, executionType parameters.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] rank Dimension of FFT transform (1, 2, or 3).
 *  @param[in] n Number of elements in the x/y/z directions.
 *  @param[in] inembed Number of elements in the input data in the x/y/z directions.
 *  @param[in] istride Distance between two successive elements in the input data.
 *  @param[in] idist Distance between input batches.
 *  @param[in] inputType Format of FFT input.
 *  @param[in] onembed Number of elements in the output data in the x/y/z directions.
 *  @param[in] ostride Distance between two successive elements in the output data.
 *  @param[in] odist Distance between output batches.
 *  @param[in] outputType Format of FFT output.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  @param[in] executionType Internal data format used by the library during computation.
 *  */
HIPFFT_EXPORT hipfftResult hipfftXtGetSizeMany(hipfftHandle   plan,
                                               int            rank,
                                               long long int* n,
                                               long long int* inembed,
                                               long long int  istride,
                                               long long int  idist,
                                               hipDataType    inputType,
                                               long long int* onembed,
                                               long long int  ostride,
                                               long long int  odist,
                                               hipDataType    outputType,
                                               long long int  batch,
                                               size_t*        workSize,
                                               hipDataType    executionType);

/*! @brief Execute an FFT plan for any precision and type.

 * @details An in-place transform is performed if the input and
 * output pointers have the same value.
 *
 * The direction parameter is ignored if for real-to-complex and
 * complex-to-real transforms, as the direction is already implied by
 * the data types.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] input Pointer to input data for the transform.
 *  @param[in] output Pointer to output data for the transform.
 *  @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 *  */
HIPFFT_EXPORT hipfftResult hipfftXtExec(hipfftHandle plan,
                                        void*        input,
                                        void*        output,
                                        int          direction);

/*! @brief Set multiple GPUs on a plan.
 *
 *  Instructs hipFFT to use multiple GPUs for a plan.
 * 
 *  This function must be called after the plan is allocated using
 *  ::hipfftCreate, but before the plan is initialized by any of the
 *  "MakePlan" functions.  Therefore, API functions that combine
 *  creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,
 *  ::hipfftPlan3d, and ::hipfftPlanMany) cannot use multiple GPUs.
 *
 * @param[in, out] plan
 * @param[in] count: length gpus array
 * @param[in] gpus: array of ints specifying deviceIDs
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus);

/*! @brief Layout details of a multi-device memory descriptor.
 */
typedef enum hipfftXtSubFormat_t
{
    HIPFFT_XT_FORMAT_INPUT             = 0x00,
    HIPFFT_XT_FORMAT_OUTPUT            = 0x01,
    HIPFFT_XT_FORMAT_INPLACE           = 0x02,
    HIPFFT_XT_FORMAT_INPLACE_SHUFFLED  = 0x03,
    HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,
    HIPFFT_FORMAT_UNDEFINED            = 0x05
} hipfftXtSubFormat;

/*! @brief Allocate memory on multiple devices.
 *
 *  Allocate memory on multiple devices for the specified plan.
 *  Returns a \ref hipLibXtDesc_t descriptor which includes pointers
 *  to the allocated memory, devices that memory resides on, and
 *  sizes allocated.
 *
 *  The subformat indicates whether the memory will be used for FFT
 *  input or output.
 *
 *  The memory must be freed by calling ::hipfftXtFree.
 * 
 * @param[in] plan FFT plan to allocate descriptor memory for.
 * @param[out] desc Pointer to descriptors for allocated memory, the devices used, and sizes.
 * @param[in] format Subformat determines whether memory is used for FFT input or output.
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtMalloc(hipfftHandle      plan,
                                          hipLibXtDesc**    desc,
                                          hipfftXtSubFormat format);

/*! @brief Copy data to/from \ref hipLibXtDesc_t descriptors.
 *
 *  Copy data according to the hipfftXtCopyType_t
 *
 *  - ::HIPFFT_COPY_HOST_TO_DEVICE: dest points to a \ref hipLibXtDesc_t structure that describes multi-device memory layout.  src points to a host memory buffer.
 *  - ::HIPFFT_COPY_DEVICE_TO_HOST: src points to a \ref hipLibXtDesc_t structure that describes multi-device memory layout.  dest points to a host memory buffer.
 *  - ::HIPFFT_COPY_DEVICE_TO_DEVICE: Both dest and src point to a \ref hipLibXtDesc_t structure that describes multi-device memory layout.  The two structures must describe memory with the same number of devices and memory sizes.
 * 
 * @param[in] plan Plan which has the descriptor.
 * @param[out] dest Buffer that will be populated.
 * @param[in] src Buffer that will be copied from.
 * @param[in] type Type of copy operation to perform.
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtMemcpy(hipfftHandle     plan,
                                          void*            dest,
                                          void*            src,
                                          hipfftXtCopyType type);

/*! @brief Free memory allocated by \ref hipfftXtMalloc.
 *
 * @param[in] desc Descriptor whose memory will be freed.
 * 
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtFree(hipLibXtDesc* desc);

/** @defgroup hipfftXtExecDescriptor Execute FFTs using \ref hipLibXtDesc_t descriptors.
 *
 *  Execute FFTs using \ref hipLibXtDesc_t descriptors.  Inputs and
 *  outputs are pointers to \ref hipLibXtDesc_t descriptors.
 *  In-place transforms are performed by passing the same pointer for
 *  input and output.
 * 
 * @warning Experimental
 */

/**
 * @addtogroup hipfftXtExecDescriptor
 * @{
*/

/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * 
 * @brief Execute single-precision complex-to-complex transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output,
                                                     int           direction);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * 
 * @brief Execute single-precision real forward transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * 
 * @brief Execute single-precision real backward transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * 
 * @brief Execute double-precision complex-to-complex transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output,
                                                     int           direction);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * 
 * @brief Execute double-precision real forward transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * 
 * @brief Execute double-precision real backward transform.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle  plan,
                                                     hipLibXtDesc* input,
                                                     hipLibXtDesc* output);
/**
 * @ingroup hipfftXtExecDescriptor
 * 
 * @param[in] plan The FFT plan.
 * @param[in] input Input data.
 * @param[out] output Output data.
 * @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * 
 * @brief Execute general transform - types of the descriptors must match the plan.
*/
HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptor(hipfftHandle  plan,
                                                  hipLibXtDesc* input,
                                                  hipLibXtDesc* output,
                                                  int           direction);

/**
 * @}
*/

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HIPFFT_H_
