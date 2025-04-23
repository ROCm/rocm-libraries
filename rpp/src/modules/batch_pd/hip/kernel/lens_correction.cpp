#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void lenscorrection_pkd(const  unsigned char* input,
                                              unsigned char* output,
                                              const float strength,
                                              const float zoom,
                                              const float halfWidth,
                                              const float halfHeight,
                                              const float correctionRadius,
                                              const unsigned int height,
                                              const unsigned int width,
                                              const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    float theta;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    if (r == 0)
    {
        theta = 1.0;
    }
    else
    {
        theta = atan(r) / r;
    }

    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;

    if ((x >= 0) && (y >= 0) && (x < width - 2) && (y < height - 2))
    {
        pix = input[x * channel + y * width * channel + id_z];
        pix_right = input[(x + 1) * channel + y * width * channel + id_z];
        pix_right_down = input[x * channel + (y + 1) * width * channel + id_z];
        pix_down = input[(x + 1) * channel + (y + 1) * width * channel + id_z];

        pixVal = (int)(pix * (1 - x_diff) * (1 - y_diff) + pix_right * (x_diff) * (1 - y_diff) + pix_right_down*(y_diff)*(1-x_diff)   +  pix_down * (x_diff * y_diff));
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
    else
    {
        output[dstpixIdx] = 0;
    }
}

extern "C" __global__ void lenscorrection_pln(const  unsigned char *input,
                                              unsigned char *output,
                                              const float strength,
                                              const float zoom,
                                              const float halfWidth,
                                              const float halfHeight,
                                              const float correctionRadius,
                                              const unsigned int height,
                                              const unsigned int width,
                                              const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pix, pix_right, pix_right_down, pix_down, pixVal;

    int dstpixIdx = id_x + id_y * width + id_z  * channel;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    float theta;
    if (r == 0)
    {
        theta = 1.0;
    }
    else
    {
        theta = atan(r) / r;
    }

    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;

    if ((x >= 0) && (y >= 0) && (x < width - 2) && (y < height - 2))
    {
        pix = input[x + y * width + id_z * channel];
        pix_right = input[(x +1) + y * width + id_z * channel];
        pix_right_down = input[x + (y + 1) * width + id_z * channel];
        pix_down = input[(x + 1) + (y + 1) * width + id_z * channel];

        pixVal = (int)(pix * (1 - x_diff) * (1 - y_diff) + pix_right * (x_diff) * (1 - y_diff) + pix_right_down * (y_diff) * (1 - x_diff) + pix_down * (x_diff * y_diff));
        output[dstpixIdx] = saturate_8u(pixVal);
    }
    else
    {
        output[dstpixIdx] = 0;
    }
}

extern "C" __global__ void lens_correction_batch(unsigned char *input,
                                                 unsigned char *output,
                                                 float *strength,
                                                 float *zoom,
                                                 unsigned int *xroi_begin,
                                                 unsigned int *xroi_end,
                                                 unsigned int *yroi_begin,
                                                 unsigned int *yroi_end,
                                                 unsigned int *height,
                                                 unsigned int *width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 const unsigned int channel,
                                                 unsigned int *inc, // use width * height for pln and 1 for pkd
                                                 const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int pix;
    int indextmp = 0;
    int dstpixIdx = 0;
    dstpixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z] ) * plnpkdindex;
    float strength_img = strength[id_z];

    if (strength_img == 0)
    {
        strength_img = 0.000001;
    }

    float temp_val = width[id_z] * width[id_z] + height[id_z] * height[id_z];
    float correctionRadius = (float)sqrt(temp_val) / (float)strength_img;
    float halfWidth = width[id_z] / 2.0;
    float halfHeight = height[id_z] / 2.0;

    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / correctionRadius;

    float theta;
    if (r == 0)
    {
        theta = 1.0;
    }
    else
    {
        theta = atan(r) / r;
    }

    float new_idx = (halfWidth + theta * newX * zoom[id_z]);
    float new_idy = (halfHeight + theta * newY * zoom[id_z]);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;

    if ((x >= xroi_begin[id_z]) && (y >= yroi_begin[id_z]) && (x < xroi_end[id_z]) && (y < yroi_end[id_z]))
    {
        for(indextmp=0; indextmp < channel; indextmp ++)
        {
            int pix = input[batch_index[id_z] + (x  + y * max_width[id_z]) * plnpkdindex + indextmp * inc[id_z]];
            // pixVal = (int)(pix * (1 - x_diff) * (1 - y_diff) + pix_right * (x_diff) * (1 - y_diff) + pix_right_down * (y_diff) * (1 - x_diff) +  pix_down * (x_diff * y_diff));
            output[dstpixIdx + indextmp*inc[id_z]] =  saturate_8u(pix);
        }
    }
    else
    {
        dstpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z]) * plnpkdindex;
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dstpixIdx] = 0;
            dstpixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_lens_correction_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(lens_correction_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}