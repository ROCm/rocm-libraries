/******************************************************************************
 * Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights
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

/*! @brief Set a function pointer callback on a plan (deprecated)
   *
   * @details Set either a load or store callback to run with a plan,
   * specified as a device function pointer.  The type of callback is
   * specified with the 'cbtype' parameter.  An array of callback and
   * callback data pointers must be given - one per device executing
   * the plan.
   *
   * The specified arrays must remain valid until either the plan is
   * destroyed or callbacks are cleared from the plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] callbacks Array of callback function pointers.
   * @param[in] cbtype Type of callback being set.
   * @param[in] callbackData Array of callback function data pointers
   */
HIPFFT_DEPRECATED_MSG(
    "Function pointer callbacks are deprecated; use hipfftXtSetJITCallback for JIT callbacks.")
HIPFFT_EXPORT hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                               void**               callbacks,
                                               hipfftXtCallbackType cbtype,
                                               void**               callbackData);

/*! @brief Remove a function pointer callback from a plan (deprecated)
   *
   * @details Remove a previously-set function pointer callback from a plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] cbtype Type of callback being removed.
   */
HIPFFT_DEPRECATED_MSG(
    "Function pointer callbacks are deprecated; use hipfftXtSetJITCallback for JIT callbacks.")
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

typedef hipfftComplex (*hipfftJITCallbackLoadC)(void*              data,
                                                unsigned long long offset,
                                                void*              cbdata,
                                                void*              sharedMem);
typedef hipfftDoubleComplex (*hipfftJITCallbackLoadZ)(void*              data,
                                                      unsigned long long offset,
                                                      void*              cbdata,
                                                      void*              sharedMem);
typedef hipfftReal (*hipfftJITCallbackLoadR)(void*              data,
                                             unsigned long long offset,
                                             void*              cbdata,
                                             void*              sharedMem);
typedef hipfftDoubleReal (*hipfftJITCallbackLoadD)(void*              data,
                                                   unsigned long long offset,
                                                   void*              cbdata,
                                                   void*              sharedMem);

typedef void (*hipfftJITCallbackStoreC)(
    void* data, unsigned long long offset, hipfftComplex element, void* cbdata, void* sharedMem);
typedef void (*hipfftJITCallbackStoreZ)(void*               data,
                                        unsigned long long  offset,
                                        hipfftDoubleComplex element,
                                        void*               cbdata,
                                        void*               sharedMem);
typedef void (*hipfftJITCallbackStoreR)(
    void* data, unsigned long long offset, hipfftReal element, void* cbdata, void* sharedMem);
typedef void (*hipfftJITCallbackStoreD)(
    void* data, unsigned long long offset, hipfftDoubleReal element, void* cbdata, void* sharedMem);

/*! @brief Set a JIT callback on a plan
   *
   * @details Set either a load or store callback to run with a plan.
   * The type of callback is specified with the `cbtype` parameter.
   * The callback is provided as SPIR-V on AMD platforms and as
   * LTO-IR fatbin on CUDA platforms.
   *
   *  Note: JIT callbacks cannot currently be used on multi-GPU transforms.
   *  This support will be added in a future release of hipFFT.
   *
   * JIT callbacks can be cleared from the plan by passing nullptr
   * for `symbol_name` and `bitcode_data`, and 0 for
   * `bitcode_len_bytes`.
   *
   * `cbdata` is an optional array of pointers of data that is passed
   * to the callback function.  As JIT callbacks are currently only
   * supported for single-device transforms, this array must have one
   * pointer element in it if it is specified.  The pointer will be
   * passed to the callback function's `cbdata` parameter when it is
   * executed on that device.
   *
   * If specified, the `cbdata` array must remain valid until either
   * the plan is destroyed or callbacks are cleared from the plan.
   *
   * This function must be called after the plan is allocated using
   * ::hipfftCreate, but before the plan is initialized by any of the
   * "MakePlan" functions.  Therefore, API functions that combine
   * creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,
   * ::hipfftPlan3d, and ::hipfftPlanMany) cannot set a JIT callback.
   *
   * @param[in] plan The FFT plan.
   * @param[in] symbol_name Name of the symbol in the compiled bitcode.
   * @param[in] bitcode_data Pointer to bitcode data.
   * @param[in] bitcode_len_bytes Number of bytes in the bitcode data.
   * @param[in] cbtype Type of callback function.
   * @param[in] cbdata Array of callback function data pointers.
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetJITCallback(hipfftHandle         plan,
                                                  const char*          symbol_name,
                                                  const void*          bitcode_data,
                                                  size_t               bitcode_len_bytes,
                                                  hipfftXtCallbackType cbtype,
                                                  void**               cbdata);

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

/*! @brief Instructs hipFFT to use multiple GPUs for a plan.
 *
 *  This function must be called after the plan is allocated using
 *  ::hipfftCreate, but before the plan is initialized by any of the
 *  "MakePlan" functions.  Therefore, API functions that combine
 *  creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,
 *  ::hipfftPlan3d, and ::hipfftPlanMany) cannot use multiple GPUs.
 *
 * @note The rocFFT backend currently requires all device IDs in
 *  the `gpus` array to be distinct.
 *
 * @param[in, out] plan The FFT plan.
 * @param[in] count Number of GPUs (length of the `gpus` array).
 * @param[in] gpus Array of device IDs.
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus);

/*! @brief Enumeration type identifying the kind of data distribution, observed by
 *  a multi-device data descriptor (i.e., an instance of ::hipLibXtDesc) for a given
 *  data set.
 */
typedef enum hipfftXtSubFormat_t
{
    /*!
     *  This subformat represents a division along the batch dimension for a
     *  compact (not padded) I/O data set of a plan initialized for a batched
     *  transform (i.e., `batch` > 1). This value identifies the input data
     *  of the plan for which it is created (by ::hipfftXtMalloc).
     *  Descriptors of this subformat can be used in out-of-place executions. */
    HIPFFT_XT_FORMAT_INPUT = 0x00,
    /*!
     *  Same distribution as ::HIPFFT_XT_FORMAT_INPUT, but identifies the
     *  output data of the plan for which it is created (by ::hipfftXtMalloc). */
    HIPFFT_XT_FORMAT_OUTPUT = 0x01,
    /*!
     *  This format represents a division along the batch (resp. slowest) dimension
     *  of a batched (resp. unbatched) data set. Descriptors of this subformat can
     *  be used for in-place executions of multi-device plans, i.e., they account
     *  for padding in the fastest dimensions for real data. */
    HIPFFT_XT_FORMAT_INPLACE = 0x02,
    /*!
     *  This subformat is restricted to unbatched, multi-dimensional data sets,
     *  and represents a division along the second-slowest dimension.
     *  Descriptors of this subformat can be used for in-place executions of
     *  multi-device plans for an unbatched transform. */
    HIPFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,
    /*! This subformat is specific to unbatched, one-dimensional transforms (not
     *  implemented with rocfft backend). */
    HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,
    /*! Undefined subformat that should never be used explicitly. */
    HIPFFT_FORMAT_UNDEFINED = 0x05
} hipfftXtSubFormat;

/*! @brief Allocates a multi-device descriptor (i.e., a ::hipLibXtDesc instance),
 *  tailored to a specific multi-device plan's needs. The allocated descriptor
 *  must be freed using ::hipfftXtFree when no longer needed.
 * 
 *  @details The data set of interest that is considered for the definition of
 *  the desired descriptor is the plan's expected input data set, unless
 *  `format` is ::HIPFFT_XT_FORMAT_OUTPUT. For values of `format` consistent
 *  with in-place operations (i.e., ::HIPFFT_XT_FORMAT_INPLACE or
 *  ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED), the created descriptor is usable for
 *  both the plan's input and output data sets.
 * 
 *  @note Multi-device descriptors resulting in a division of real data sets
 *    along their fastest dimension are not supported. As a result,
 *    ::hipfftXtMalloc does not produce usable descriptors for requests to
 *    create a multi-device descriptor of subformat value
 *    ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED (resp. ::HIPFFT_XT_FORMAT_INPLACE)
 *    given a plan initialized for an unbatched two-dimensional real forward
 *    (resp. inverse) transform.
 *
 * @param[in] plan Multi-device FFT plan to allocate descriptor memory for.
 * @param[out] desc Address of the pointer to the created descriptor.
 * @param[in] format Desired descriptor's subformat, identifying the plan's
 * I/O data set of interest (see ::hipfftXtSubFormat for more details).
 *
 * @warning Experimental
 */
HIPFFT_EXPORT hipfftResult hipfftXtMalloc(hipfftHandle      plan,
                                          hipLibXtDesc**    desc,
                                          hipfftXtSubFormat format);

/*! @brief Copy data to/from \ref hipLibXtDesc descriptors.
 *
 * @details If `type` is ::HIPFFT_COPY_HOST_TO_DEVICE (resp.
 * ::HIPFFT_COPY_DEVICE_TO_HOST), `src` (resp. `dest`) must point to a
 * host-residing memory buffer and `dest` (resp. `src`) must point to a
 * ::hipLibXtDesc descriptor. That descriptor must be fit for storing the plan's
 * input (resp. output) data set assuming it's distributed consistently with the
 * descriptor's own `subFormat` value (see ::hipfftXtSubFormat for more details).
 * The host buffer must be large enough to hold the plan's input (resp. output)
 * data (accounting for possible padding in real domain for plans and descriptors
 * specific to data for in-place real transforms).
 * ::HIPFFT_COPY_DEVICE_TO_DEVICE copies are supported only from descriptors of
 * format ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED to descriptors of format
 * ::HIPFFT_XT_FORMAT_INPLACE along with plans that support both formats.
 *
 * @param[in] plan Multi-device FFT plan of interest, defining the data set of
 * relevance for the desired copy operation.
 * @param[out] dest Opaque pointer to the copy destination.
 * @param[in] src Opaque pointer to the copy source.
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

/** @defgroup hipfftXtExecDescriptor Multi-device descriptor execution
 * 
 * Executes multi-device transforms using ::hipLibXtDesc descriptors as
 * `input` and `output`. In-place transforms require `input == output`.
 *
 * After a successful unbatched multi-dimensional in-place execution, the
 * given descriptor's `subFormat` is toggled: the descriptor's `subFormat`
 * changes from ::HIPFFT_XT_FORMAT_INPLACE to ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
 * and vice versa.  This reflects the data redistribution operated by the
 * transform.
 *
 * The `subFormat` value(s) of the given descriptor(s) must be compatible
 * with the desired execution's placement (see ::hipfftXtSubFormat). For
 * unbatched multi-dimensional transforms, only in-place execution is
 * supported (`input == output` required and the descriptor's subformat
 * must be ::HIPFFT_XT_FORMAT_INPLACE or ::HIPFFT_XT_FORMAT_INPLACE_SHUFFLED).
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
