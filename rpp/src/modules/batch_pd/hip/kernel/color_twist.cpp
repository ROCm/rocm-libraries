#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define RPPMIN3(a,b,c) ((a < b) && (a < c) ? a : ((b < c) ? b : c))
#define RPPMAX3(a,b,c) ((a > b) && (a > c) ? a : ((b > c) ? b : c))

__device__ uchar4 convert_one_pixel_to_rgb(float4 pixel)
{
    float r, g, b;
    float h, s, v;

    h = pixel.x;
    s = pixel.y;
    v = pixel.z;

    float f = h / 60.0f;
    float hi = floor(f);
    f = f - hi;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t = v * (1 - s * (1 - f));

    if (hi == 0.0f || hi == 6.0f)
    {
        r = v;
        g = t;
        b = p;
    }
    else if (hi == 1.0f)
    {
        r = q;
        g = v;
        b = p;
    }
    else if (hi == 2.0f)
    {
        r = p;
        g = v;
        b = t;
    }
    else if (hi == 3.0f)
    {
        r = p;
        g = q;
        b = v;
    }
    else if (hi == 4.0f)
    {
        r = t;
        g = p;
        b = v;
    }
    else
    {
        r = v;
        g = p;
        b = q;
    }

    unsigned char red = (unsigned char)(255.0f * r);
    unsigned char green = (unsigned char)(255.0f * g);
    unsigned char blue = (unsigned char)(255.0f * b);
    unsigned char alpha = 0.0;

    return (uchar4){red, green, blue, alpha};
}

__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel)
{
    float r, g, b, a;
    float h, s, v;

    r = pixel.x / 255.0f;
    g = pixel.y / 255.0f;
    b = pixel.z / 255.0f;
    a = pixel.w;

    float max = RPPMAX3(r, g, b);
    float min = RPPMIN3(r, g, b);
    float diff = max - min;

    v = max;

    if (v == 0.0f)
    { // black
        h = s = 0.0f;
    }
    else
    {
        s = diff / v;
        if (diff < 0.001f)
        { // grey
            h = 0.0f;
        }
        else
        { // color
            if ((max > r - 0.001) && (max < r + 0.001))
            {
                h = 60.0f * (g - b) / diff;
                if (h < 0.0f)
                {
                    h += 360.0f;
                }
            }
            else if (max == g)
            {
                h = 60.0f * (2 + (b - r) / diff);
            }
            else
            {
                h = 60.0f * (4 + (r - g) / diff);
            }
        }
    }

    return (float4){h, s, v, a};
}

extern "C" __global__ void color_twist_batch(unsigned char *input,
                                             unsigned char *output,
                                             float *alpha,
                                             float *beta,
                                             float *hue,
                                             float *sat,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned int *height,
                                             unsigned int *width,
                                             unsigned int *max_width,
                                             unsigned long long *batch_index,
                                             unsigned int *inc,
                                             unsigned int *dst_inc,
                                             const int in_plnpkdind,
                                             const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel;
    float4 hsv;

    unsigned int l_inc = inc[id_z];
    unsigned int d_inc  = dst_inc[id_z];

    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * in_plnpkdind;
    int out_pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * out_plnpkdind;

    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + l_inc];
    pixel.z = input[pixIdx + 2 * l_inc];
    pixel.w = 0.0;

    float alpha1 = alpha[id_z];
    float beta1 = beta[id_z];

    if ((id_y >= yroi_begin[id_z]) &&
        (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) &&
        (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        hsv.x += hue[id_z];
        if (hsv.x > 360.0)
        {
            hsv.x = hsv.x - 360.0;
        }
        else if (hsv.x < 0)
        {
            hsv.x = hsv.x + 360.0;
        }
        hsv.y *= sat[id_z];
        if (hsv.y > 1.0)
        {
            hsv.y = 1.0;
        }
        else if (hsv.y < 0.0)
        {
            hsv.y = 0.0;
        }
        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with hue modification
        output[out_pixIdx] = saturate_8u(alpha1 * pixel.x + beta1);
        output[out_pixIdx + d_inc] = saturate_8u(alpha1 * pixel.y + beta1);
        output[out_pixIdx + 2 * d_inc] = saturate_8u(alpha1 * pixel.z + beta1);
    }
    else
    {
        output[out_pixIdx] = pixel.x;
        output[out_pixIdx + d_inc] = pixel.y;
        output[out_pixIdx + 2 * d_inc] = pixel.z;
    }
}

extern "C" __global__ void color_twist_batch_int8(signed char *input,
                                                  signed char *output,
                                                  float *alpha,
                                                  float *beta,
                                                  float *hue,
                                                  float *sat,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned int *height,
                                                  unsigned int *width,
                                                  unsigned int *max_width,
                                                  unsigned long long *batch_index,
                                                  unsigned int *inc,
                                                  unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                                  const int in_plnpkdind,
                                                  const int out_plnpkdind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel;
    float4 hsv;

    unsigned int l_inc = inc[id_z];
    unsigned int d_inc  = dst_inc[id_z];

    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * in_plnpkdind;
    int out_pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * out_plnpkdind;

    pixel.x = (uchar)(input[pixIdx] + 128);
    pixel.y = (uchar)(input[pixIdx + l_inc] + 128);
    pixel.z = (uchar)(input[pixIdx + 2 * l_inc] + 128);
    pixel.w = 0.0;

    float alpha1 = alpha[id_z];
    float beta1 = beta[id_z];

    if ((id_y >= yroi_begin[id_z]) &&
        (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) &&
        (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        hsv.x += hue[id_z];
        if (hsv.x > 360.0)
        {
            hsv.x = hsv.x - 360.0;
        }
        else if (hsv.x < 0)
        {
            hsv.x = hsv.x + 360.0;
        }
        hsv.y *= sat[id_z];
        if (hsv.y > 1.0)
        {
            hsv.y = 1.0;
        }
        else if (hsv.y < 0.0)
        {
            hsv.y = 0.0;
        }
        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with hue modification
        output[out_pixIdx] = (signed char)(saturate_8u(alpha1 * pixel.x + beta1) - 128);
        output[out_pixIdx + d_inc] = (signed char)(saturate_8u(alpha1 * pixel.y + beta1) - 128);
        output[out_pixIdx + 2 * d_inc] = (signed char)(saturate_8u(alpha1 * pixel.z + beta1) - 128);
    }
    else
    {
        output[out_pixIdx] = pixel.x - 128;
        output[out_pixIdx + d_inc] = pixel.y - 128;
        output[out_pixIdx + 2 * d_inc] = pixel.z - 128;
    }
}

extern "C" __global__ void color_twist_batch_fp32(float *input,
                                                  float *output,
                                                  float *alpha,
                                                  float *beta,
                                                  float *hue,
                                                  float *sat,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned int *height,
                                                  unsigned int *width,
                                                  unsigned int *max_width,
                                                  unsigned long long *batch_index,
                                                  unsigned int *inc,
                                                  unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                                  const int in_plnpkdind,
                                                  const int out_plnpkdind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel;
    float4 hsv;

    unsigned int l_inc = inc[id_z];
    unsigned int d_inc  = dst_inc[id_z];

    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * in_plnpkdind;
    int out_pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * out_plnpkdind;

    pixel.x = (uchar)(input[pixIdx] * 255);
    pixel.y = (uchar)(input[pixIdx + l_inc] * 255);
    pixel.z = (uchar)(input[pixIdx + 2 * l_inc] * 255);
    pixel.w = 0.0;

    float alpha1 = alpha[id_z];
    float beta1 = beta[id_z];

    if ((id_y >= yroi_begin[id_z]) &&
        (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) &&
        (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        hsv.x += hue[id_z];
        if (hsv.x > 360.0)
        {
            hsv.x = hsv.x - 360.0;
        }
        else if (hsv.x < 0)
        {
            hsv.x = hsv.x + 360.0;
        }
        hsv.y *= sat[id_z];
        if (hsv.y > 1.0)
        {
            hsv.y = 1.0;
        }
        else if (hsv.y < 0.0)
        {
            hsv.y = 0.0;
        }
        pixel = convert_one_pixel_to_rgb(hsv);
        output[out_pixIdx] = (alpha1 * pixel.x + beta1) / 255.0;
        output[out_pixIdx + d_inc] = (alpha1 * pixel.y + beta1) / 255.0;
        output[out_pixIdx + 2 * d_inc] = (alpha1 * pixel.z + beta1) / 255.0;
    }
    else
    {
        output[out_pixIdx] = input[pixIdx];
        output[out_pixIdx + d_inc] = input[pixIdx + l_inc];
        output[out_pixIdx + 2 * d_inc] = input[pixIdx + 2 * l_inc];
    }
}

// extern "C" __global__ void color_twist_batch_fp16(
//     half *input, half *output, float *alpha,
//     float *beta, float *hue, float *sat,
//     int *xroi_begin, int *xroi_end, int *yroi_begin,
//     int *yroi_end, unsigned int *height,
//     unsigned int *width, unsigned int *max_width,
//     unsigned long *batch_index,
//     unsigned int *inc, unsigned int *dst_inc, // use width * height for pln and 1 for pkd
//     const int in_plnpkdind , const int out_plnpkdind      // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   if (id_x >= width[id_z] || id_y >= height[id_z])
//     return;
//   uchar4 pixel;
//   float4 hsv;

//   unsigned int l_inc = inc[id_z];
//   unsigned int d_inc  = dst_inc[id_z];
//   int pixIdx =
//       batch_index[id_z] + (id_y * max_width[id_z] + id_x) * in_plnpkdind;
//   int out_pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * out_plnpkdind;
//   pixel.x = (uchar)(input[pixIdx] * 255);
//   pixel.y = (uchar)(input[pixIdx + l_inc] * 255);
//   pixel.z = (uchar)(input[pixIdx + 2 * l_inc] * 255);
//   pixel.w = 0.0;
//   float alpha1 = alpha[id_z], beta1 = beta[id_z];

//   if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
//       (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
//     hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
//     hsv.x += hue[id_z];
//     if (hsv.x > 360.0) {
//       hsv.x = hsv.x - 360.0;
//     } else if (hsv.x < 0) {
//       hsv.x = hsv.x + 360.0;
//     }
//     hsv.y *= sat[id_z];
//     if (hsv.y > 1.0) {
//       hsv.y = 1.0;
//     } else if (hsv.y < 0.0) {
//       hsv.y = 0.0;
//     }
//     pixel = convert_one_pixel_to_rgb(hsv);
//     output[out_pixIdx] = (half)((alpha1 * pixel.x + beta1) / 255.0);
//     output[out_pixIdx + d_inc] = (half)((alpha1 * pixel.y + beta1) / 255.0);
//     output[out_pixIdx + 2 * d_inc] = (half)((alpha1 * pixel.z + beta1) / 255.0);
//   } else {
//     output[out_pixIdx] = input[pixIdx];
//     output[out_pixIdx + d_inc] = input[pixIdx + l_inc];
//     output[out_pixIdx + 2 * d_inc] = input[pixIdx + 2 * l_inc];
//   }
// }

RppStatus hip_exec_color_twist_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(color_twist_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_color_twist_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // hipLaunchKernelGGL(color_twist_batch_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
    //                    handle.GetInitHandle()->mem.mgpu.inc,
    //                    handle.GetInitHandle()->mem.mgpu.dstInc,
    //                    in_plnpkdind,
    //                    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_color_twist_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(color_twist_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_color_twist_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(color_twist_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}