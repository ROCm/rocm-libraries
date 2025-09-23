// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef HIPFFT_MPI_ENABLE
#include <cufftmp/cufft.h>

#include <cufftmp/cudalibxt.h>
#include <cufftmp/cufftMp.h>
#include <cufftmp/cufftXt.h>
#else
#include <cudalibxt.h>
#include <cufft.h>
#include <cufftXt.h>
#endif

#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"
#include <cuda_runtime_api.h>
#include <iostream>

#ifdef HIPFFT_MPI_ENABLE
#include "hipfft/hipfftMp.h"
#endif

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

// ensure that hipfft's XtDesc structs look the same as cufft's
static_assert(sizeof(hipLibXtDesc) == sizeof(cudaLibXtDesc)
              && sizeof(hipXtDesc) == sizeof(cudaXtDesc));

static hipfftResult_t cufftResultToHipResult(cufftResult_t cufft_result)
{
    switch(cufft_result)
    {
    case CUFFT_SUCCESS:
        return HIPFFT_SUCCESS;

    case CUFFT_INVALID_PLAN:
        return HIPFFT_INVALID_PLAN;

    case CUFFT_ALLOC_FAILED:
        return HIPFFT_ALLOC_FAILED;

    case CUFFT_INVALID_TYPE:
        return HIPFFT_INVALID_TYPE;

    case CUFFT_INVALID_VALUE:
        return HIPFFT_INVALID_VALUE;

    case CUFFT_INTERNAL_ERROR:
        return HIPFFT_INTERNAL_ERROR;

    case CUFFT_EXEC_FAILED:
        return HIPFFT_EXEC_FAILED;

    case CUFFT_SETUP_FAILED:
        return HIPFFT_SETUP_FAILED;

    case CUFFT_INVALID_SIZE:
        return HIPFFT_INVALID_SIZE;

    case CUFFT_UNALIGNED_DATA:
        return HIPFFT_UNALIGNED_DATA;

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return HIPFFT_INCOMPLETE_PARAMETER_LIST;

    case CUFFT_INVALID_DEVICE:
        return HIPFFT_INVALID_DEVICE;

    case CUFFT_PARSE_ERROR:
        return HIPFFT_PARSE_ERROR;

    case CUFFT_NO_WORKSPACE:
        return HIPFFT_NO_WORKSPACE;

    case CUFFT_NOT_IMPLEMENTED:
        return HIPFFT_NOT_IMPLEMENTED;

    case CUFFT_NOT_SUPPORTED:
        return HIPFFT_NOT_SUPPORTED;

    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

static cufftType_t hipfftTypeToCufftType(hipfftType_t hipfft_type)
{
    switch(hipfft_type)
    {
    case HIPFFT_R2C:
        return CUFFT_R2C;

    case HIPFFT_C2R:
        return CUFFT_C2R;

    case HIPFFT_C2C:
        return CUFFT_C2C;

    case HIPFFT_D2Z:
        return CUFFT_D2Z;

    case HIPFFT_Z2D:
        return CUFFT_Z2D;

    case HIPFFT_Z2Z:
        return CUFFT_Z2Z;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

// static cudaDataType_t hipDataTypeToCudaDataType(hipDataType hip_data_type)
// {
//     switch(hipfft_type)
//     {
//     case HIP_R_16F:
//         return CUDA_R_16F;

//     case HIP_R_32F:
//         return CUDA_R_32F;

//     case HIP_R_64F:
//         return CUDA_R_64F;

//     case HIP_C_16F:
//         return CUDA_C_16F;

//     case HIP_C_32F:
//         return CUDA_C_32F;

//     case HIP_C_64F:
//         return CUDA_C_64F;

//     default:
//         throw HIPFFT_INVALID_VALUE;
//     }
// }

static libraryPropertyType hipfftLibraryPropertyTypeToCufftLibraryPropertyType(
    hipfftLibraryPropertyType_t hipfft_lib_prop_type)
{
    switch(hipfft_lib_prop_type)
    {
    case HIPFFT_MAJOR_VERSION:
        return MAJOR_VERSION;

    case HIPFFT_MINOR_VERSION:
        return MINOR_VERSION;

    case HIPFFT_PATCH_LEVEL:
        return PATCH_LEVEL;

    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

static cufftXtCallbackType_t hipfftCallbackTypeToCufftCallbackType(hipfftXtCallbackType_t type)
{
    switch(type)
    {
    case HIPFFT_CB_LD_COMPLEX:
        return CUFFT_CB_LD_COMPLEX;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        return CUFFT_CB_LD_COMPLEX_DOUBLE;
    case HIPFFT_CB_LD_REAL:
        return CUFFT_CB_LD_REAL;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        return CUFFT_CB_LD_REAL_DOUBLE;
    case HIPFFT_CB_ST_COMPLEX:
        return CUFFT_CB_ST_COMPLEX;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        return CUFFT_CB_ST_COMPLEX_DOUBLE;
    case HIPFFT_CB_ST_REAL:
        return CUFFT_CB_ST_REAL;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        return CUFFT_CB_ST_REAL_DOUBLE;
    case HIPFFT_CB_UNDEFINED:
        return CUFFT_CB_UNDEFINED;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

static cufftXtSubFormat hipfftXtSubFormatTocufftXtSubFormat(hipfftXtSubFormat f)
{
    switch(f)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        return CUFFT_XT_FORMAT_INPUT;
    case HIPFFT_XT_FORMAT_OUTPUT:
        return CUFFT_XT_FORMAT_OUTPUT;
    case HIPFFT_XT_FORMAT_INPLACE:
        return CUFFT_XT_FORMAT_INPLACE;
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
    case HIPFFT_FORMAT_UNDEFINED:
        return CUFFT_FORMAT_UNDEFINED;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

static cufftXtCopyType hipfftXtCopyTypeTocufftXtCopyType(hipfftXtCopyType t)
{
    switch(t)
    {
    case HIPFFT_COPY_HOST_TO_DEVICE:
        return CUFFT_COPY_HOST_TO_DEVICE;
    case HIPFFT_COPY_DEVICE_TO_HOST:
        return CUFFT_COPY_DEVICE_TO_HOST;
    case HIPFFT_COPY_DEVICE_TO_DEVICE:
        return CUFFT_COPY_DEVICE_TO_DEVICE;
    case HIPFFT_COPY_UNDEFINED:
        return CUFFT_COPY_UNDEFINED;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
}

static inline hipfftResult handle_exception() noexcept
try
{
    throw;
}
catch(hipfftResult e)
{
    return e;
}
catch(...)
{
    return HIPFFT_INTERNAL_ERROR;
}

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
try
{
    return cufftResultToHipResult(cufftPlan1d(plan, nx, hipfftTypeToCufftType(type), batch));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
try
{
    return cufftResultToHipResult(cufftPlan2d(plan, nx, ny, hipfftTypeToCufftType(type)));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
try
{
    return cufftResultToHipResult(cufftPlan3d(plan, nx, ny, nz, hipfftTypeToCufftType(type)));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch)
try
{
    if((inembed == nullptr) != (onembed == nullptr))
    {
        return HIPFFT_INVALID_VALUE;
    }
    else
    {
        auto cufftret = CUFFT_SUCCESS;
        cufftret      = cufftPlanMany(plan,
                                 rank,
                                 n,
                                 inembed,
                                 istride,
                                 idist,
                                 onembed,
                                 ostride,
                                 odist,
                                 hipfftTypeToCufftType(type),
                                 batch);
        return cufftResultToHipResult(cufftret);
    }
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult hipfftCreate(hipfftHandle* plan)
try
{
    return cufftResultToHipResult(cufftCreate(plan));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExtPlanScaleFactor(hipfftHandle plan, double scalefactor)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftMakePlan1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftMakePlan2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftMakePlan3d(plan, nx, ny, nz, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan,
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
                                size_t*      workSize)
try
{
    if((inembed == nullptr) != (onembed == nullptr))
        return HIPFFT_INVALID_VALUE;

    auto cufftret = CUFFT_SUCCESS;
    cufftret      = cufftMakePlanMany(plan,
                                 rank,
                                 n,
                                 inembed,
                                 istride,
                                 idist,
                                 onembed,
                                 ostride,
                                 odist,
                                 hipfftTypeToCufftType(type),
                                 batch,
                                 workSize);
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
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
                                  size_t*        workSize)
try
{
    return cufftResultToHipResult(cufftMakePlanMany64(plan,
                                                      rank,
                                                      n,
                                                      inembed,
                                                      istride,
                                                      idist,
                                                      onembed,
                                                      ostride,
                                                      odist,
                                                      hipfftTypeToCufftType(type),
                                                      batch,
                                                      workSize));
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftEstimate1d(nx, hipfftTypeToCufftType(type), batch, workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(cufftEstimate2d(nx, ny, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftEstimate3d(nx, ny, nz, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimateMany(int        rank,
                                int*       n,
                                int*       inembed,
                                int        istride,
                                int        idist,
                                int*       onembed,
                                int        ostride,
                                int        odist,
                                hipfftType type,
                                int        batch,
                                size_t*    workSize)
try
{
    return cufftResultToHipResult(cufftEstimateMany(rank,
                                                    n,
                                                    inembed,
                                                    istride,
                                                    idist,
                                                    onembed,
                                                    ostride,
                                                    odist,
                                                    hipfftTypeToCufftType(type),
                                                    batch,
                                                    workSize));
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftGetSize1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftGetSize2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    return cufftResultToHipResult(
        cufftGetSize3d(plan, nx, ny, nz, hipfftTypeToCufftType(type), workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan,
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
                               size_t*      workSize)
try
{
    return cufftResultToHipResult(cufftGetSizeMany(plan,
                                                   rank,
                                                   n,
                                                   inembed,
                                                   istride,
                                                   idist,
                                                   onembed,
                                                   ostride,
                                                   odist,
                                                   hipfftTypeToCufftType(type),
                                                   batch,
                                                   workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
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
                                 size_t*        workSize)
try
{
    return cufftResultToHipResult(cufftGetSizeMany64(plan,
                                                     rank,
                                                     n,
                                                     inembed,
                                                     istride,
                                                     idist,
                                                     onembed,
                                                     ostride,
                                                     odist,
                                                     hipfftTypeToCufftType(type),
                                                     batch,
                                                     workSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize)
try
{
    return cufftResultToHipResult(cufftGetSize(plan, workSize));
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
try
{
    return cufftResultToHipResult(cufftSetAutoAllocation(plan, autoAllocate));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
try
{
    return cufftResultToHipResult(cufftSetWorkArea(plan, workArea));
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
try
{
    return cufftResultToHipResult(cufftExecC2C(plan, idata, odata, direction));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
try
{
    return cufftResultToHipResult(cufftExecR2C(plan, idata, odata));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
try
{
    return cufftResultToHipResult(cufftExecC2R(plan, idata, odata));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
try
{
    return cufftResultToHipResult(cufftExecZ2Z(plan, idata, odata, direction));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
try
{
    return cufftResultToHipResult(cufftExecD2Z(plan, idata, odata));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
try
{
    return cufftResultToHipResult(cufftExecZ2D(plan, idata, odata));
}
catch(...)
{
    return handle_exception();
}

/*===========================================================================*/

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
try
{
    return cufftResultToHipResult(cufftSetStream(plan, stream));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftDestroy(hipfftHandle plan)
try
{
    return cufftResultToHipResult(cufftDestroy(plan));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetVersion(int* version)
try
{
    if(!version)
        return HIPFFT_INVALID_VALUE;
    return cufftResultToHipResult(cufftGetVersion(version));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value)
try
{
    return cufftResultToHipResult(
        cufftGetProperty(hipfftLibraryPropertyTypeToCufftLibraryPropertyType(type), value));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                 void**               callbacks,
                                 hipfftXtCallbackType cbtype,
                                 void**               callbackData)
try
{
    return cufftResultToHipResult(cufftXtSetCallback(
        plan, callbacks, hipfftCallbackTypeToCufftCallbackType(cbtype), callbackData));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype)
try
{
    return cufftResultToHipResult(
        cufftXtClearCallback(plan, hipfftCallbackTypeToCufftCallbackType(cbtype)));
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize)
try
{
    return cufftResultToHipResult(cufftXtSetCallbackSharedSize(
        plan, hipfftCallbackTypeToCufftCallbackType(cbtype), sharedSize));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtMakePlanMany(hipfftHandle   plan,
                                  int            rank,
                                  long long int* n,
                                  long long int* inembed,
                                  long long int  istride,
                                  long long int  idist,
                                  hipDataType    inputtype,
                                  long long int* onembed,
                                  long long int  ostride,
                                  long long int  odist,
                                  hipDataType    outputtype,
                                  long long int  batch,
                                  size_t*        workSize,
                                  hipDataType    executiontype)
try
{
    return cufftResultToHipResult(cufftXtMakePlanMany(plan,
                                                      rank,
                                                      n,
                                                      inembed,
                                                      istride,
                                                      idist,
                                                      inputtype,
                                                      onembed,
                                                      ostride,
                                                      odist,
                                                      outputtype,
                                                      batch,
                                                      workSize,
                                                      executiontype));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtGetSizeMany(hipfftHandle   plan,
                                 int            rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int  istride,
                                 long long int  idist,
                                 hipDataType    inputtype,
                                 long long int* onembed,
                                 long long int  ostride,
                                 long long int  odist,
                                 hipDataType    outputtype,
                                 long long int  batch,
                                 size_t*        workSize,
                                 hipDataType    executiontype)
try
{
    return cufftResultToHipResult(cufftXtGetSizeMany(plan,
                                                     rank,
                                                     n,
                                                     inembed,
                                                     istride,
                                                     idist,
                                                     inputtype,
                                                     onembed,
                                                     ostride,
                                                     odist,
                                                     outputtype,
                                                     batch,
                                                     workSize,
                                                     executiontype));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExec(hipfftHandle plan, void* input, void* output, int direction)
try
{
    return cufftResultToHipResult(cufftXtExec(plan, input, output, direction));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus)
try
{
    return cufftResultToHipResult(cufftXtSetGPUs(plan, count, gpus));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtMalloc(hipfftHandle plan, hipLibXtDesc** desc, hipfftXtSubFormat format)
try
{
    auto cufftret = cufftXtMalloc(
        plan, reinterpret_cast<cudaLibXtDesc**>(desc), hipfftXtSubFormatTocufftXtSubFormat(format));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtMemcpy(hipfftHandle plan, void* dest, void* src, hipfftXtCopyType type)
try
{
    auto cufftret = cufftXtMemcpy(plan, dest, src, hipfftXtCopyTypeTocufftXtCopyType(type));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtFree(hipLibXtDesc* desc)
try
{
    auto cufftret = cufftXtFree(reinterpret_cast<cudaLibXtDesc*>(desc));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
try
{
    auto cufftret = cufftXtExecDescriptorC2C(plan,
                                             reinterpret_cast<cudaLibXtDesc*>(input),
                                             reinterpret_cast<cudaLibXtDesc*>(output),
                                             direction);
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    auto cufftret = cufftXtExecDescriptorR2C(
        plan, reinterpret_cast<cudaLibXtDesc*>(input), reinterpret_cast<cudaLibXtDesc*>(output));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    auto cufftret = cufftXtExecDescriptorC2R(
        plan, reinterpret_cast<cudaLibXtDesc*>(input), reinterpret_cast<cudaLibXtDesc*>(output));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
try
{
    auto cufftret = cufftXtExecDescriptorZ2Z(plan,
                                             reinterpret_cast<cudaLibXtDesc*>(input),
                                             reinterpret_cast<cudaLibXtDesc*>(output),
                                             direction);
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    auto cufftret = cufftXtExecDescriptorD2Z(
        plan, reinterpret_cast<cudaLibXtDesc*>(input), reinterpret_cast<cudaLibXtDesc*>(output));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    auto cufftret = cufftXtExecDescriptorZ2D(
        plan, reinterpret_cast<cudaLibXtDesc*>(input), reinterpret_cast<cudaLibXtDesc*>(output));
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptor(hipfftHandle  plan,
                                    hipLibXtDesc* input,
                                    hipLibXtDesc* output,
                                    int           direction)
try
{
    auto cufftret = cufftXtExecDescriptor(plan,
                                          reinterpret_cast<cudaLibXtDesc*>(input),
                                          reinterpret_cast<cudaLibXtDesc*>(output),
                                          direction);
    return cufftResultToHipResult(cufftret);
}
catch(...)
{
    return handle_exception();
}

#ifdef HIPFFT_MPI_ENABLE
static cufftMpCommType_t hipfftMpCommTypeToCufftMpCommType(hipfftMpCommType_t hipfft_type)
{
    switch(hipfft_type)
    {
    case HIPFFT_COMM_MPI:
        return CUFFT_COMM_MPI;
    case HIPFFT_COMM_NONE:
        return CUFFT_COMM_NONE;
    }
    throw HIPFFT_INVALID_VALUE;
}

hipfftResult hipfftMpAttachComm(hipfftHandle plan, hipfftMpCommType comm_type, void* comm_handle)
try
{
    auto cuComm = hipfftMpCommTypeToCufftMpCommType(comm_type);
    return cufftResultToHipResult(cufftMpAttachComm(plan, cuComm, comm_handle));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetDistribution(hipfftHandle         plan,
                                     int                  rank,
                                     const long long int* input_lower,
                                     const long long int* input_upper,
                                     const long long int* output_lower,
                                     const long long int* output_upper,
                                     const long long int* stride_input,
                                     const long long int* stride_output)
try
{
    return cufftResultToHipResult(cufftXtSetDistribution(plan,
                                                         rank,
                                                         input_lower,
                                                         input_upper,
                                                         output_lower,
                                                         output_upper,
                                                         stride_input,
                                                         stride_output));
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetSubformatDefault(hipfftHandle      plan,
                                         hipfftXtSubFormat subformat_forward,
                                         hipfftXtSubFormat subformat_inverse)
try
{
    return cufftResultToHipResult(
        cufftXtSetSubformatDefault(plan,
                                   hipfftXtSubFormatTocufftXtSubFormat(subformat_forward),
                                   hipfftXtSubFormatTocufftXtSubFormat(subformat_inverse)));
}
catch(...)
{
    return handle_exception();
}

#endif
