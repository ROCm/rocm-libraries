#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define amd_max3_hip(a,b,c) ((a > b) && (a > c) ? a : ((b > c) ? b : c))
#define amd_min3_hip(a,b,c) ((a < b) && (a < c) ? a : ((b < c) ? b : c))

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

    unsigned char red = (unsigned char) (255.0f * r);
    unsigned char green = (unsigned char) (255.0f * g);
    unsigned char blue = (unsigned char) (255.0f * b);
    unsigned char alpha = 0.0; //(unsigned char)(pixel.w);
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

    float max = amd_max3_hip(r, g, b);
    float min = amd_min3_hip(r, g, b);
    float diff = max - min;

    v = max;

    if (v == 0.0f)
    {
        // black
        h = s = 0.0f;
    }
    else
    {
        s = diff / v;
        if (diff < 0.001f)
        {
            // grey
            h = 0.0f;
        }
        else
        {
            // color
            if (max == r)
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

extern "C" __global__ void huergb_pkd(unsigned char *input,
                                      unsigned char *output,
                                      const float hue,
                                      const float sat,
                                      const unsigned int height,
                                      const unsigned int width)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    float r, g, b, min, max, delta, h, s, v;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = (id_y * width + id_x) * 3;
    uchar4 pixel;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + 1];
    pixel.z = input[pixIdx + 2];
    pixel.w = 0;// Transpency factor yet to come
    float4 hsv;
    hsv = convert_one_pixel_to_hsv(pixel);
    hsv.x += hue;
    if(hsv.x > 360.0)
    {
        hsv.x = hsv.x - 360.0;
    }
    else if(hsv.x < 0)
    {
        hsv.x = hsv.x + 360.0;
    }

    hsv.y += sat;
    if(hsv.y > 1.0)
    {
        hsv.y = 1.0;
    }
    else if(hsv.y < 0.0)
    {
        hsv.y = 0.0;
    }

    pixel = convert_one_pixel_to_rgb(hsv);

    output[pixIdx] = pixel.x;
    output[pixIdx + 1] = pixel.y;
    output[pixIdx + 2] = pixel.z;
}

extern "C" __global__ void huergb_pln(unsigned char *input,
                                      unsigned char *output,
                                      const float hue,
                                      const float sat,
                                      const unsigned int height,
                                      const unsigned int width)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    float r, g, b, min, max, delta, h, s, v;
    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = (id_y * width + id_x);
    uchar4 pixel;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + width * height];
    pixel.z = input[pixIdx + 2 * width * height];
    pixel.w = 0;// Transpency factor yet to come
    float4 hsv;
    hsv = convert_one_pixel_to_hsv(pixel);
    hsv.x += hue;

    if(hsv.x > 360.0)
    {
        hsv.x = hsv.x - 360.0;
    }
    else if(hsv.x < 0)
    {
        hsv.x = hsv.x + 360.0;
    }

    hsv.y += sat;

    if(hsv.y > 1.0)
    {
        hsv.y = 1.0;
    }
    else if(hsv.y < 0.0)
    {
        hsv.y = 0.0;
    }

    pixel = convert_one_pixel_to_rgb(hsv);

    output[pixIdx] = pixel.x;
    output[pixIdx + width * height] = pixel.y;
    output[pixIdx + 2 * width * height] = pixel.z;
}


extern "C" __global__ void hue_batch(unsigned char *input,
                                     unsigned char *output,
                                     float *hue,
                                     unsigned int *xroi_begin,
                                     unsigned int *xroi_end,
                                     unsigned int *yroi_begin,
                                     unsigned int *yroi_end,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long long *batch_index,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                     const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel; float4 hsv;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        hsv.x += hue[id_z];

        if(hsv.x > 360.0)
        {
            hsv.x = hsv.x - 360.0;
        }
        else if(hsv.x < 0)
        {
            hsv.x = hsv.x + 360.0;
        }

        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with hue modification
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
    else
    {
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] =  pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }

}

extern "C" __global__ void saturation_batch(unsigned char *input,
                                            unsigned char *output,
                                            float *sat,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel; float4 hsv;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel);
        hsv.y *= sat[id_z];

        hsv.y = fmaxf(fminf(hsv.y, 1.0), 0.0);

        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with saturation modification
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
    else
    {
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
}

extern "C" __global__ void convert_batch_rgb_hsv(unsigned char *input,
                                                 float *output,
                                                 unsigned int *xroi_begin,
                                                 unsigned int *xroi_end,
                                                 unsigned int *yroi_begin,
                                                 unsigned int *yroi_end,
                                                 unsigned int *height,
                                                 unsigned int *width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 unsigned int *inc, // use width * height for pln and 1 for pkd
                                                 const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel;
    float4 hsv;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        output[pixIdx] = hsv.x;
        output[pixIdx + inc[id_z]] = hsv.y;
        output[pixIdx + 2 * inc[id_z]] = hsv.z;
    }
    else
    {
        output[pixIdx] = 0.0;
        output[pixIdx + inc[id_z]] = 0.0;
        output[pixIdx + 2 * inc[id_z]] = 0.0;
    }
}

extern "C" __global__ void convert_batch_hsv_rgb(float *input,
                                                 unsigned char *output,
                                                 unsigned int *xroi_begin,
                                                 unsigned int *xroi_end,
                                                 unsigned int *yroi_begin,
                                                 unsigned int *yroi_end,
                                                 unsigned int *height,
                                                 unsigned int *width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 unsigned int *inc, // use width * height for pln and 1 for pkd
                                                 const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 rgb;
    float4 pixel;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        rgb = convert_one_pixel_to_rgb(pixel); // Converting to HSV
        output[pixIdx] = rgb.x;
        output[pixIdx + inc[id_z]] = rgb.y;
        output[pixIdx + 2 * inc[id_z]] = rgb.z;
    }
    else
    {
        output[pixIdx] = 0;
        output[pixIdx + inc[id_z]] = 0;
        output[pixIdx + 2 * inc[id_z]] = 0;
    }
}

RppStatus hip_exec_hueRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(hue_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_saturationRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(saturation_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_convert_batch_rgb_hsv(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_width, Rpp32u max_height)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(convert_batch_rgb_hsv,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_convert_batch_hsv_rgb(Rpp32f *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_width, Rpp32u max_height)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(convert_batch_hsv_rgb,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}