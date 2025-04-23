/* 
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef __ago_haf_gpu_h__
#define __ago_haf_gpu_h__

#include "ago_internal.h"

#if ENABLE_OPENCL

// OpenCL string format
#define OPENCL_FORMAT(fmt) fmt

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code to load into local memory:
//   this code assumes following variables created by caller in "code"
//     gx      - global work item [0]
//     gy      - global work item [1]
//     gbuf    - global buffer pointer
//     gstride - global buffer stride
//     lx      - local work item [0]
//     ly      - local work item [1]
//     lbuf    - local buffer pointer
//
int HafGpu_Load_Local(int WGWidth, int WGHeight, int LMWidth, int LMHeight, int gxoffset, int gyoffset, std::string& code);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code to load into local memory based on image size:
//   this code assumes following variables created by caller in "code"
//     gx      - global work item [0]
//     gy      - global work item [1]
//     gbuf    - global buffer pointer
//     gstride - global buffer stride
//     lx      - local work item [0]
//     ly      - local work item [1]
//     lbuf    - local buffer pointer
//
int HafGpu_Load_Local_ImageSize(int WGWidth, int WGHeight, int LMWidth, int LMHeight, int gxoffset, int gyoffset, std::string& code, int srcImageBufferSize);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for linear filter
//
int HafGpu_LinearFilter_ANY_U8(AgoNode * node, vx_df_image dst_image_format, AgoData * iConv, bool roundingMode);
int HafGpu_LinearFilter_ANY_S16(AgoNode * node, vx_df_image dst_image_format, AgoData * iConv, bool roundingMode);
int HafGpu_LinearFilter_ANY_F32(AgoNode * node, vx_df_image dst_image_format, AgoData * iConv, bool roundingMode);
int HafGpu_LinearFilter_ANYx2_U8(AgoNode * node, vx_df_image dst_image_format, AgoData * iConv, AgoData * iConv2, bool roundingMode);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following half scale gaussian filters:
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_3x3
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5
//
int HafGpu_ScaleGaussianHalf(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following gaussian scale filters:
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_ORB_U8_U8_5x5 (interpolation = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR)
//
int HafGpu_ScaleGaussianOrb(AgoNode * node, vx_interpolation_type_e interpolation);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following special case Sobel filter kernels:
//   VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX
//   VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY
//   VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY
//   VX_KERNEL_AMD_SOBEL_MAGNITUDE_PHASE_S16U8_U8_3x3
//   VX_KERNEL_AMD_SOBEL_MAGNITUDE_S16_U8_3x3
//   VX_KERNEL_AMD_SOBEL_PHASE_U8_U8_3x3
//
int HafGpu_SobelSpecialCases(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following canny sobel filter kernels:
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L2NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L2NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM
//
int HafGpu_CannySobelFilters(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following canny non-max supression filter kernels:
//   VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8_U16_3x3
//   VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3
//
int HafGpu_CannySuppThreshold(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following harris sobel filter kernels:
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_3x3
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_5x5
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_7x7
//
int HafGpu_HarrisSobelFilters(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following harris score filter kernels:
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_3x3
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_5x5
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_7x7
//
int HafGpu_HarrisScoreFilters(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following non-max supression filter kernels:
//   VX_KERNEL_AMD_NON_MAX_SUPP_XY_ANY_3x3
//
int HafGpu_NonMaxSupp_XY_ANY_3x3(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for following non-linear filter kernels:
//   VX_KERNEL_AMD_DILATE_U8_U8_3x3, VX_KERNEL_AMD_DILATE_U1_U8_3x3,
//   VX_KERNEL_AMD_ERODE_U8_U8_3x3, VX_KERNEL_AMD_ERODE_U1_U8_3x3, 
//   VX_KERNEL_AMD_MEDIAN_U8_U8_3x3
//
int HafGpu_NonLinearFilter_3x3_ANY_U8(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for following non-linear filter kernels:
//   VX_KERNEL_AMD_DILATE_U8_U1_3x3, VX_KERNEL_AMD_DILATE_U1_U1_3x3,
//   VX_KERNEL_AMD_ERODE_U8_U1_3x3, VX_KERNEL_AMD_ERODE_U1_U1_3x3, 
//
int HafGpu_NonLinearFilter_3x3_ANY_U1(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for following fast corner detector kernels:
//   VX_KERNEL_AMD_FAST_CORNERS_XY_U8_NOSUPRESSION, VX_KERNEL_AMD_FAST_CORNERS_XY_U8_SUPRESSION,
//
int HafGpu_FastCorners_XY_U8(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following channel combine kernels
//   VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_UYVY
//   VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_YUYV
//
int HafGpu_ChannelCombine_U32_422(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following channel extractions:
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3
//
int HafGpu_ChannelExtract_U8_U32(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following format conversions:
//   VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY
//   VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV
//   VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY
//   VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV
//
int HafGpu_FormatConvert_420_422(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following format conversions:
//   VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12
//   VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12
//   VX_KERNEL_AMD_FORMAT_CONVERT_UV12_IUV
//   VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8
//
int HafGpu_FormatConvert_Chroma(AgoNode * node);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following color conversions:
//   VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV
//
int HafGpu_ColorConvert(AgoNode * node);

#endif

#endif
