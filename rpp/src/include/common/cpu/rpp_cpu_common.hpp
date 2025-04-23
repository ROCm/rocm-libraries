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

#ifndef RPP_CPU_COMMON_HPP
#define RPP_CPU_COMMON_HPP

#include <math.h>
#include <algorithm>
#include <typeinfo>
#include <cstring>
#include <omp.h>
#include "rppdefs.h"
#include "handle.hpp"
#include "rpp_cpu_simd_load_store.hpp"

#define PI_OVER_180                     0.0174532925
#define ONE_OVER_255                    0.00392156862745f
#define ONE_OVER_256                    0.00390625f
#define RPP_128_OVER_255                0.50196078431f
#define RPPMIN3(a,b,c)                  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX3(a,b,c)                  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPINRANGE(a, x, y)             ((a >= x) && (a <= y) ? 1 : 0)
#define RPPPRANGECHECK(value, a, b)     (value < (Rpp32f) a) ? ((Rpp32f) a) : ((value < (Rpp32f) b) ? value : ((Rpp32f) b))
#define RPPPIXELCHECK(pixel)            (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))
#define RPPPIXELCHECKF32(pixel)         (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1))
#define RPPPIXELCHECKI8(pixel)          (pixel < (Rpp32f) -128) ? ((Rpp32f) -128) : ((pixel < (Rpp32f) 127) ? pixel : ((Rpp32f) 127))
#define RPPISLESSER(pixel, value)       ((pixel < value) ? 1 : 0)
#define RGB_TO_GREY_WEIGHT_RED          0.299f
#define RGB_TO_GREY_WEIGHT_GREEN        0.587f
#define RGB_TO_GREY_WEIGHT_BLUE         0.114f
#define INTERP_BILINEAR_KERNEL_SIZE     2           // Kernel size needed for Bilinear Interpolation
#define INTERP_BILINEAR_KERNEL_RADIUS   1.0f        // Kernel radius needed for Bilinear Interpolation
#define INTERP_BILINEAR_NUM_COEFFS      4           // Number of coefficents needed for Bilinear Interpolation
#define NEWTON_METHOD_INITIAL_GUESS     0x5f3759df          // Initial guess for Newton Raphson Inverse Square Root
#define RPP_255_OVER_1PT57              162.3380757272f     // (255 / 1.570796) - multiplier used in phase computation
#define ONE_OVER_1PT57                  0.6366199048f       // (1 / 1.570796) i.e. 2/pi - multiplier used in phase computation

#if __AVX2__
#define SIMD_FLOAT_VECTOR_LENGTH        8
#else
#define SIMD_FLOAT_VECTOR_LENGTH        4
#endif

/*Constants used for Gaussian interpolation*/
// Here sigma is considered as 0.5f

// Computes strides for ND Tensor
inline void compute_strides(Rpp32u *strides, Rpp32u *shape, Rpp32u tensorDim)
{
    if (tensorDim > 0)
    {
        Rpp32u v = 1;
        for (Rpp32u i = tensorDim - 1; i > 0; i--)
        {
            strides[i] = v;
            v *= shape[i];
        }
        strides[0] = v;
    }
}

// Uses fast inverse square root algorithm from Lomont, C., 2003. FAST INVERSE SQUARE ROOT. [online] lomont.org. Available at: <http://www.lomont.org/papers/2003/InvSqrt.pdf>
inline float rpp_host_math_inverse_sqrt_1(float x)
{
    float xHalf = 0.5f * x;
    int i = *(int*)&x;                              // float bits in int
    i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);     // initial guess for Newton's method
    x = *(float*)&i;                                // new bits to float
    x = x * (1.5f - xHalf * x * x);                 // One round of Newton's method

    return x;
}

// copy ROI of voxel data from input to output
template<typename T>
void copy_3d_host_tensor(T *srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         T *dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         RpptROI3D *roi,
                         RppLayoutParams layoutParams)
{
    if((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
    {
        T *srcPtrDepth = srcPtr + (roi->xyzwhdROI.xyz.z * srcGenericDescPtr->strides[1]) + (roi->xyzwhdROI.xyz.y * srcGenericDescPtr->strides[2]) + (roi->xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
        T *dstPtrDepth = dstPtr;
        Rpp32u width = roi->xyzwhdROI.roiWidth * srcGenericDescPtr->dims[4];
        for(int i = 0; i < roi->xyzwhdROI.roiDepth; i++)
        {
            T *srcPtrRow = srcPtrDepth;
            T *dstPtrRow = dstPtrDepth;
            for(int j = 0; j < roi->xyzwhdROI.roiHeight; j++)
            {
                memcpy(dstPtrRow, srcPtrRow, width * sizeof(T));
                srcPtrRow += srcGenericDescPtr->strides[2];
                dstPtrRow += dstGenericDescPtr->strides[2];
            }
            srcPtrDepth += srcGenericDescPtr->strides[1];
            dstPtrDepth += dstGenericDescPtr->strides[1];
        }
    }
    else if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
    {
        T *srcPtrChannel = srcPtr + (roi->xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi->xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi->xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
        T *dstPtrChannel = dstPtr;
        int channels = srcGenericDescPtr->dims[1];
        for(int c = 0; c < channels; c++)
        {
            T *srcPtrDepth = srcPtrChannel;
            T *dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi->xyzwhdROI.roiDepth; i++)
            {
                T *srcPtrRow = srcPtrDepth;
                T *dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi->xyzwhdROI.roiHeight; j++)
                {
                    memcpy(dstPtrRow, srcPtrRow, roi->xyzwhdROI.roiWidth * sizeof(T));
                    srcPtrRow += srcGenericDescPtr->strides[3];
                    dstPtrRow += dstGenericDescPtr->strides[3];
                }
                srcPtrDepth += srcGenericDescPtr->strides[2];
                dstPtrDepth += dstGenericDescPtr->strides[2];
            }
            srcPtrChannel += srcGenericDescPtr->strides[1];
            dstPtrChannel += dstGenericDescPtr->strides[1];
        }
    }
}

inline void compute_xywh_from_ltrb_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->xywhROI.xy.x = roiPtrInput->ltrbROI.lt.x;
    roiPtrImage->xywhROI.xy.y = roiPtrInput->ltrbROI.lt.y;
    roiPtrImage->xywhROI.roiWidth = roiPtrInput->ltrbROI.rb.x - roiPtrInput->ltrbROI.lt.x + 1;
    roiPtrImage->xywhROI.roiHeight = roiPtrInput->ltrbROI.rb.y - roiPtrInput->ltrbROI.lt.y + 1;
}

inline void compute_xyzwhd_from_ltfrbb_host(RpptROI3DPtr roiPtrInput, RpptROI3DPtr roiPtrImage)
{
    roiPtrImage->xyzwhdROI.xyz.x = roiPtrInput->ltfrbbROI.ltf.x;
    roiPtrImage->xyzwhdROI.xyz.y = roiPtrInput->ltfrbbROI.ltf.y;
    roiPtrImage->xyzwhdROI.xyz.z = roiPtrInput->ltfrbbROI.ltf.z;
    roiPtrImage->xyzwhdROI.roiWidth = roiPtrInput->ltfrbbROI.rbb.x - roiPtrInput->ltfrbbROI.ltf.x + 1;
    roiPtrImage->xyzwhdROI.roiHeight = roiPtrInput->ltfrbbROI.rbb.y - roiPtrInput->ltfrbbROI.ltf.y + 1;
    roiPtrImage->xyzwhdROI.roiDepth = roiPtrInput->ltfrbbROI.rbb.z - roiPtrInput->ltfrbbROI.ltf.z + 1;
}

inline void compute_ltrb_from_xywh_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtrImage)
{
    roiPtrImage->ltrbROI.lt.x = roiPtrInput->xywhROI.xy.x;
    roiPtrImage->ltrbROI.lt.y = roiPtrInput->xywhROI.xy.y;
    roiPtrImage->ltrbROI.rb.x = roiPtrInput->xywhROI.xy.x + roiPtrInput->xywhROI.roiWidth - 1;
    roiPtrImage->ltrbROI.rb.y = roiPtrInput->xywhROI.xy.y + roiPtrInput->xywhROI.roiHeight - 1;
}

inline void compute_roi_boundary_check_host(RpptROIPtr roiPtrImage, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault)
{
    roiPtr->xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
    roiPtr->xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
    roiPtr->xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
    roiPtr->xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);
}

inline void compute_roi3D_boundary_check_host(RpptROI3DPtr roiPtrImage, RpptROI3DPtr roiPtr, RpptROI3DPtr roiPtrDefault)
{
    roiPtr->xyzwhdROI.xyz.x = std::max(roiPtrDefault->xyzwhdROI.xyz.x, roiPtrImage->xyzwhdROI.xyz.x);
    roiPtr->xyzwhdROI.xyz.y = std::max(roiPtrDefault->xyzwhdROI.xyz.y, roiPtrImage->xyzwhdROI.xyz.y);
    roiPtr->xyzwhdROI.xyz.z = std::max(roiPtrDefault->xyzwhdROI.xyz.z, roiPtrImage->xyzwhdROI.xyz.z);
    roiPtr->xyzwhdROI.roiWidth = std::min(roiPtrDefault->xyzwhdROI.roiWidth - roiPtrImage->xyzwhdROI.xyz.x, roiPtrImage->xyzwhdROI.roiWidth);
    roiPtr->xyzwhdROI.roiHeight = std::min(roiPtrDefault->xyzwhdROI.roiHeight - roiPtrImage->xyzwhdROI.xyz.y, roiPtrImage->xyzwhdROI.roiHeight);
    roiPtr->xyzwhdROI.roiDepth = std::min(roiPtrDefault->xyzwhdROI.roiDepth - roiPtrImage->xyzwhdROI.xyz.z, roiPtrImage->xyzwhdROI.roiDepth);
}

inline void compute_roi_validation_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault, RpptRoiType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI roiImage;
        RpptROIPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoiType::LTRB)
            compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoiType::XYWH)
            roiPtrImage = roiPtrInput;
        compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }
}

inline void compute_roi3D_validation_host(RpptROI3DPtr roiPtrInput, RpptROI3DPtr roiPtr, RpptROI3DPtr roiPtrDefault, RpptRoi3DType roiType)
{
    if (roiPtrInput == NULL)
    {
        roiPtr = roiPtrDefault;
    }
    else
    {
        RpptROI3D roiImage;
        RpptROI3DPtr roiPtrImage = &roiImage;
        if (roiType == RpptRoi3DType::LTFRBB)
            compute_xyzwhd_from_ltfrbb_host(roiPtrInput, roiPtrImage);
        else if (roiType == RpptRoi3DType::XYZWHD)
            roiPtrImage = roiPtrInput;
        compute_roi3D_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
    }
}

inline void saturate_pixel(Rpp32f &pixel, Rpp8u* dst)
{
    *dst = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel)));
}

inline void saturate_pixel(Rpp32f &pixel, Rpp8s* dst)
{
    *dst = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(pixel) - 128));
}

inline void saturate_pixel(Rpp32f &pixel, Rpp32f* dst)
{
    *dst = RPPPIXELCHECKF32(pixel);
}

inline void saturate_pixel(Rpp32f &pixel, Rpp16f* dst)
{
    *dst = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel));
}

#endif // RPP_CPU_COMMON_HPP
