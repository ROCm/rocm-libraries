/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_arithmetic_operations.h"
#include "host_tensor_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_tensor_executors.hpp"
#endif // HIP_COMPILE

/******************** fused_multiply_add_scalar ********************/

RppStatus rppt_fused_multiply_add_scalar_host(RppPtr_t srcPtr,
                                 RpptGenericDescPtr srcGenericDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32f *mulTensor,
                                 Rpp32f *addTensor,
                                 RpptROI3DPtr roiGenericPtrSrc,
                                 RpptRoi3DType roiType,
                                 rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        fused_multiply_add_scalar_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                         srcGenericDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                         dstGenericDescPtr,
                                         mulTensor,
                                         addTensor,
                                         roiGenericPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** add_scalar ********************/

RppStatus rppt_add_scalar_host(RppPtr_t srcPtr,
                               RpptGenericDescPtr srcGenericDescPtr,
                               RppPtr_t dstPtr,
                               RpptGenericDescPtr dstGenericDescPtr,
                               Rpp32f *addTensor,
                               RpptROI3DPtr roiGenericPtrSrc,
                               RpptRoi3DType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        add_scalar_f32_f32_host_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                       srcGenericDescPtr,
                                       reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                       dstGenericDescPtr,
                                       addTensor,
                                       roiGenericPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** subtract_scalar ********************/

RppStatus rppt_subtract_scalar_host(RppPtr_t srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32f *subtractTensor,
                                    RpptROI3DPtr roiGenericPtrSrc,
                                    RpptRoi3DType roiType,
                                    rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        subtract_scalar_f32_f32_host_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                            srcGenericDescPtr,
                                            reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                            dstGenericDescPtr,
                                            subtractTensor,
                                            roiGenericPtrSrc,
                                            roiType,
                                            layoutParams,
                                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** multiply_scalar ********************/

RppStatus rppt_multiply_scalar_host(RppPtr_t srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32f *mulTensor,
                                    RpptROI3DPtr roiGenericPtrSrc,
                                    RpptRoi3DType roiType,
                                    rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        multiply_scalar_f32_f32_host_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                            srcGenericDescPtr,
                                            reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                            dstGenericDescPtr,
                                            mulTensor,
                                            roiGenericPtrSrc,
                                            roiType,
                                            layoutParams,
                                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** magnitude ********************/

RppStatus rppt_magnitude_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        magnitude_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        magnitude_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                      reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        magnitude_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        magnitude_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                    static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** log ********************/

RppStatus rppt_log_host(RppPtr_t srcPtr,
                        RpptGenericDescPtr srcGenericDescPtr,
                        RppPtr_t dstPtr,
                        RpptGenericDescPtr dstGenericDescPtr,
                        Rpp32u *roiTensor,
                        rppHandle_t rppHandle)
{
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        log_generic_host_tensor(static_cast<Rpp8u *>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                srcGenericDescPtr,
                                reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                dstGenericDescPtr,
                                roiTensor,
                                rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        log_generic_host_tensor(reinterpret_cast<Rpp16f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                srcGenericDescPtr,
                                reinterpret_cast<Rpp16f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                dstGenericDescPtr,
                                roiTensor,
                                rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        log_generic_host_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                srcGenericDescPtr,
                                reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                dstGenericDescPtr,
                                roiTensor,
                                rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        log_generic_host_tensor(static_cast<Rpp8s *>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                srcGenericDescPtr,
                                reinterpret_cast<Rpp32f *>(static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                dstGenericDescPtr,
                                roiTensor,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus rppt_log1p_host(RppPtr_t srcPtr,
                          RpptGenericDescPtr srcGenericDescPtr,
                          RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr,
                          Rpp32u *roiTensor,
                          rppHandle_t rppHandle)
{
    if (srcGenericDescPtr->dataType != RpptDataType::I16) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->dataType == RpptDataType::I16) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        log1p_i16_f32_host_tensor(static_cast<Rpp16s *>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                  srcGenericDescPtr,
                                  reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                  dstGenericDescPtr,
                                  roiTensor,
                                  rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** fused_multiply_add_scalar ********************/

RppStatus rppt_fused_multiply_add_scalar_gpu(RppPtr_t srcPtr,
                                RpptGenericDescPtr srcGenericDescPtr,
                                RppPtr_t dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                Rpp32f *mulTensor,
                                Rpp32f *addTensor,
                                RpptROI3DPtr roiGenericPtrSrc,
                                RpptRoi3DType roiType,
                                rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    hip_exec_fused_multiply_add_scalar_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                 srcGenericDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                 dstGenericDescPtr,
                                 roiGenericPtrSrc,
                                 mulTensor,
                                 addTensor,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** add_scalar ********************/

RppStatus rppt_add_scalar_gpu(RppPtr_t srcPtr,
                              RpptGenericDescPtr srcGenericDescPtr,
                              RppPtr_t dstPtr,
                              RpptGenericDescPtr dstGenericDescPtr,
                              Rpp32f *addTensor,
                              RpptROI3DPtr roiGenericPtrSrc,
                              RpptRoi3DType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    hip_exec_add_scalar_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                               srcGenericDescPtr,
                               reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                               dstGenericDescPtr,
                               roiGenericPtrSrc,
                               addTensor,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** subtract_scalar ********************/

RppStatus rppt_subtract_scalar_gpu(RppPtr_t srcPtr,
                                   RpptGenericDescPtr srcGenericDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *subtractTensor,
                                   RpptROI3DPtr roiGenericPtrSrc,
                                   RpptRoi3DType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    hip_exec_subtract_scalar_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiGenericPtrSrc,
                                    subtractTensor,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** multiply_scalar ********************/

RppStatus rppt_multiply_scalar_gpu(RppPtr_t srcPtr,
                                   RpptGenericDescPtr srcGenericDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *mulTensor,
                                   RpptROI3DPtr roiGenericPtrSrc,
                                   RpptRoi3DType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    hip_exec_multiply_scalar_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiGenericPtrSrc,
                                    mulTensor,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** magnitude ********************/

RppStatus rppt_magnitude_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout == RpptLayout::NCDHW) || (srcDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout == RpptLayout::NCDHW) || (dstDescPtr->layout == RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_magnitude_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                  static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_magnitude_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_magnitude_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_magnitude_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                  static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** log ********************/

RppStatus rppt_log_gpu(RppPtr_t srcPtr,
                       RpptGenericDescPtr srcGenericDescPtr,
                       RppPtr_t dstPtr,
                       RpptGenericDescPtr dstGenericDescPtr,
                       Rpp32u *roiTensor,
                       rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::I8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_log_generic_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_log1p_gpu(RppPtr_t srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32u *roiTensor,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::I16) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->dataType == RpptDataType::I16) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log1p_i16_f32_tensor(static_cast<Rpp16s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                      srcGenericDescPtr,
                                      reinterpret_cast<Rpp32f *>(static_cast<Rpp16s*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                      dstGenericDescPtr,
                                      roiTensor,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
