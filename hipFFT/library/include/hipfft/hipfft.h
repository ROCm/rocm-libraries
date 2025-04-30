/******************************************************************************
 * Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights
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

/*! @file hipfft.h
 *  hipfft.h defines all the public interfaces and types
 *  */

#ifndef HIPFFT_H_
#define HIPFFT_H_

#if defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored #warningName)

// clang-format off
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS DISABLE_WARNING(-Wdeprecated-declarations)
#define DISABLE_WARNING_RETURN_TYPE DISABLE_WARNING(-Wreturn-type)
// clang-format on
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS
#define DISABLE_WARNING_RETURN_TYPE
#endif

#include "hipfft-export.h"
#include "hipfft-version.h"
#include <hip/hip_complex.h>
#include <hip/library_types.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

/*! @brief Result/status/error codes */
typedef enum hipfftResult_t
{
    /*! hipFFT operation was successful */
    HIPFFT_SUCCESS = 0,
    /*! hipFFT was passed an invalid plan handle */
    HIPFFT_INVALID_PLAN = 1,
    /*! hipFFT failed to allocate GPU or CPU memory */
    HIPFFT_ALLOC_FAILED = 2,
    /*! No longer used */
    HIPFFT_INVALID_TYPE = 3,
    /*! User specified an invalid pointer or parameter */
    HIPFFT_INVALID_VALUE = 4,
    /*! Driver or internal hipFFT library error */
    HIPFFT_INTERNAL_ERROR = 5,
    /*! Failed to execute an FFT on the GPU */
    HIPFFT_EXEC_FAILED = 6,
    /*! hipFFT failed to initialize */
    HIPFFT_SETUP_FAILED = 7,
    /*! User specified an invalid transform size */
    HIPFFT_INVALID_SIZE = 8,
    /*! No longer used */
    HIPFFT_UNALIGNED_DATA = 9,
    /*! Missing parameters in call */
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    /*! Execution of a plan was on different GPU than plan creation */
    HIPFFT_INVALID_DEVICE = 11,
    /*! Internal plan database error */
    HIPFFT_PARSE_ERROR = 12,
    /*! No workspace has been provided prior to plan execution */
    HIPFFT_NO_WORKSPACE = 13,
    /*! Function does not implement functionality for parameters given. */
    HIPFFT_NOT_IMPLEMENTED = 14,
    /*! Operation is not supported for parameters given. */
    HIPFFT_NOT_SUPPORTED = 16
} hipfftResult;

/*! @brief Transform type
 *  @details This type is used to declare the Fourier transform type that will be executed.
 *  */
typedef enum hipfftType_t
{
    /*! Real to complex (interleaved) */
    HIPFFT_R2C = 0x2a,
    /*! Complex (interleaved) to real */
    HIPFFT_C2R = 0x2c,
    /*! Complex to complex (interleaved) */
    HIPFFT_C2C = 0x29,
    /*! Double to double-complex (interleaved) */
    HIPFFT_D2Z = 0x6a,
    /*! Double-complex (interleaved) to double */
    HIPFFT_Z2D = 0x6c,
    /*! Double-complex to double-complex (interleaved) */
    HIPFFT_Z2Z = 0x69
} hipfftType;

typedef enum hipfftLibraryPropertyType_t
{
    HIPFFT_MAJOR_VERSION,
    HIPFFT_MINOR_VERSION,
    HIPFFT_PATCH_LEVEL
} hipfftLibraryPropertyType;

/*! @brief Perform a forward FFT.
 * */
#define HIPFFT_FORWARD -1
/*! @brief Perform a backward/inverse FFT.
 * */
#define HIPFFT_BACKWARD 1

#ifdef __HIP_PLATFORM_NVIDIA__
typedef int hipfftHandle;
#else
typedef struct hipfftHandle_t* hipfftHandle;
#endif

typedef hipComplex       hipfftComplex;
typedef hipDoubleComplex hipfftDoubleComplex;
typedef float            hipfftReal;
typedef double           hipfftDoubleReal;

/*! @brief Create a new one-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new one-dimensional FFT plan.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx FFT length.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to compute.
 *  */
HIPFFT_EXPORT hipfftResult hipfftPlan1d(hipfftHandle* plan,
                                        int           nx,
                                        hipfftType    type,
                                        int           batch /* deprecated - use hipfftPlanMany */);

/*! @brief Create a new two-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new two-dimensional FFT plan.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx Number of elements in the x-direction (slow index).
 *  @param[in] ny Number of elements in the y-direction (fast index).
 *  @param[in] type FFT type.
 *  */
HIPFFT_EXPORT hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type);

/*! @brief Create a new three-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new three-dimensional FFT plan.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx Number of elements in the x-direction (slowest index).
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction (fastest index).
 *  @param[in] type FFT type.
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type);

/*! @brief Create a new batched rank-dimensional FFT plan with advanced data layout.
 *
 * @details Allocate and initialize a new batched rank-dimensional
 *  FFT plan. The number of elements to transform in each direction of
 *  the input data is specified in n.
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
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] rank Dimension of transform (1, 2, or 3).
 *  @param[in] n Number of elements to transform in the x/y/z directions.
 *  @param[in] inembed Number of elements in the input data in the x/y/z directions.
 *  @param[in] istride Distance between two successive elements in the input data.
 *  @param[in] idist Distance between input batches.
 *  @param[in] onembed Number of elements in the output data in the x/y/z directions.
 *  @param[in] ostride Distance between two successive elements in the output data.
 *  @param[in] odist Distance between output batches.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  */
HIPFFT_EXPORT hipfftResult hipfftPlanMany(hipfftHandle* plan,
                                          int           rank,
                                          int*          n,
                                          int*          inembed,
                                          int           istride,
                                          int           idist,
                                          int*          onembed,
                                          int           ostride,
                                          int           odist,
                                          hipfftType    type,
                                          int           batch);
/*! @brief Allocate a new plan.
 *
 *  @param[out] plan Pointer to the FFT plan handle to be allocated.
 *  */
HIPFFT_EXPORT hipfftResult hipfftCreate(hipfftHandle* plan);

/*! @brief Set scaling factor.
 *
 *  @details hipFFT multiplies each element of the result by the given factor at the end of the transform.
 *
 *  The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.
 *
 *  This function must be called after the plan is allocated using
 *  ::hipfftCreate, but before the plan is initialized by any of the
 *  "MakePlan" functions.  Therefore, API functions that combine
 *  creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,
 *  ::hipfftPlan3d, and ::hipfftPlanMany) cannot set a scale factor.
 *
 *  Note that the scale factor applies to both forward and
 *  backward transforms executed with the specified plan handle.
 */
HIPFFT_EXPORT hipfftResult hipfftExtPlanScaleFactor(hipfftHandle plan, double scalefactor);

/*! @brief Initialize a new one-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already, and
 *  modifies the plan associated with the plan handle.
 *
 *  @param[in] plan Handle of the FFT plan.
 *  @param[in] nx FFT length.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to compute.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftMakePlan1d(hipfftHandle plan,
                                            int          nx,
                                            hipfftType   type,
                                            int     batch, /* deprecated - use hipfftPlanMany */
                                            size_t* workSize);

/*! @brief Initialize a new two-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already, and
 *  modifies the plan associated with the plan handle.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param[in] plan Handle of the FFT plan.
 *  @param[in] nx Number of elements in the x-direction (slow index).
 *  @param[in] ny Number of elements in the y-direction (fast index).
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize);

/*! @brief Initialize a new two-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already, and
 *  modifies the plan associated with the plan handle.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param[in] plan Handle of the FFT plan.
 *  @param[in] nx Number of elements in the x-direction (slowest index).
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction (fastest index).
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize);

/*! @brief Initialize a new batched rank-dimensional FFT plan with advanced data layout.
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
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] rank Dimension of transform (1, 2, or 3).
 *  @param[in] n Number of elements to transform in the x/y/z directions.
 *  @param[in] inembed Number of elements in the input data in the x/y/z directions.
 *  @param[in] istride Distance between two successive elements in the input data.
 *  @param[in] idist Distance between input batches.
 *  @param[in] onembed Number of elements in the output data in the x/y/z directions.
 *  @param[in] ostride Distance between two successive elements in the output data.
 *  @param[in] odist Distance between output batches.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                              int          rank,
                                              int*         n,
                                              int*         inembed,
                                              int          istride,
                                              int          idist,
                                              int*         onembed,
                                              int          ostride,
                                              int          odist,
                                              hipfftType   type,
                                              int          batch,
                                              size_t*      workSize);

HIPFFT_EXPORT hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
                                                int            rank,
                                                long long int* n,
                                                long long int* inembed,
                                                long long int  istride,
                                                long long int  idist,
                                                long long int* onembed,
                                                long long int  ostride,
                                                long long int  odist,
                                                hipfftType     type,
                                                long long int  batch,
                                                size_t*        workSize);

/*! @brief Return an estimate of the work area size required for a 1D plan.
 *
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftEstimate1d(int        nx,
                                            hipfftType type,
                                            int        batch, /* deprecated - use hipfftPlanMany */
                                            size_t*    workSize);

/*! @brief Return an estimate of the work area size required for a 2D plan.
 *
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize);

/*! @brief Return an estimate of the work area size required for a 3D plan.
 *
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize);

/*! @brief Return an estimate of the work area size required for a rank-dimensional plan.
 *
 *  @param[in] rank Dimension of FFT transform (1, 2, or 3).
 *  @param[in] n Number of elements in the x/y/z directions.
 *  @param[in] inembed
 *  @param[in] istride
 *  @param[in] idist Distance between input batches.
 *  @param[in] onembed
 *  @param[in] ostride
 *  @param[in] odist Distance between output batches.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftEstimateMany(int        rank,
                                              int*       n,
                                              int*       inembed,
                                              int        istride,
                                              int        idist,
                                              int*       onembed,
                                              int        ostride,
                                              int        odist,
                                              hipfftType type,
                                              int        batch,
                                              size_t*    workSize);

/*! @brief Return size of the work area size required for a 1D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetSize1d(hipfftHandle plan,
                                           int          nx,
                                           hipfftType   type,
                                           int     batch, /* deprecated - use hipfftGetSizeMany */
                                           size_t* workSize);

/*! @brief Return size of the work area size required for a 2D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize);

/*! @brief Return size of the work area size required for a 3D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize);

/*! @brief Return size of the work area size required for a rank-dimensional plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] rank Dimension of FFT transform (1, 2, or 3).
 *  @param[in] n Number of elements in the x/y/z directions.
 *  @param[in] inembed
 *  @param[in] istride
 *  @param[in] idist Distance between input batches.
 *  @param[in] onembed
 *  @param[in] ostride
 *  @param[in] odist Distance between output batches.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetSizeMany(hipfftHandle plan,
                                             int          rank,
                                             int*         n,
                                             int*         inembed,
                                             int          istride,
                                             int          idist,
                                             int*         onembed,
                                             int          ostride,
                                             int          odist,
                                             hipfftType   type,
                                             int          batch,
                                             size_t*      workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
                                               int            rank,
                                               long long int* n,
                                               long long int* inembed,
                                               long long int  istride,
                                               long long int  idist,
                                               long long int* onembed,
                                               long long int  ostride,
                                               long long int  odist,
                                               hipfftType     type,
                                               long long int  batch,
                                               size_t*        workSize);

/*! @brief Return size of the work area size required for a rank-dimensional plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[out] workSize Pointer to work area size (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize);

/*! @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
 *  */
HIPFFT_EXPORT hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate);

/*! @brief Set the plan's work area.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] workArea Pointer to the work area (on device).
 *  */
HIPFFT_EXPORT hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea);

/*! @brief Execute a (float) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 *  @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * */
HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata,
                                         int            direction);

/*! @brief Execute a (float) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle   plan,
                                         hipfftReal*    idata,
                                         hipfftComplex* odata);

/*! @brief Execute a (float) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftReal*    odata);

/*! @brief Execute a (double) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 *  @param[in] direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * */
HIPFFT_EXPORT hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleComplex* odata,
                                         int                  direction);

/*! @brief Execute a (double) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecD2Z(hipfftHandle         plan,
                                         hipfftDoubleReal*    idata,
                                         hipfftDoubleComplex* odata);

/*! @brief Execute a (double) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data (on device).
 *  @param[out] odata Output data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecZ2D(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleReal*    odata);

/*! @brief Set HIP stream to execute plan on.
 *
 * @details Associates a HIP stream with a hipFFT plan.  All kernels
 * launched by this plan are associated with the provided stream.
 *
 * @param[in] plan The FFT plan.
 * @param[in] stream The HIP stream.
 * */
HIPFFT_EXPORT hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);

/*
HIPFFT_EXPORT hipfftResult hipfftSetCompatibilityMode(hipfftHandle plan,
                                               hipfftCompatibility mode);
*/

/*! @brief Destroy and deallocate an existing plan.
 *
 *  @param[in] plan Handle of the FFT plan to be destroyed.
 *  */
HIPFFT_EXPORT hipfftResult hipfftDestroy(hipfftHandle plan);

/*! @brief Get rocFFT/cuFFT version.
 *
 *  @param[out] version cuFFT/rocFFT version (returned value).
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetVersion(int* version);

/*! @brief Get library property.
 *
 *  @param[in] type Property type.
 *  @param[out] value Returned value.
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HIPFFT_H_
