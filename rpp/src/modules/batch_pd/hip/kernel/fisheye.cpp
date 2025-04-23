#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

extern "C" __global__ void fisheye_planar(const unsigned char* input,
                                          unsigned char* output,
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

    int dstpixIdx = id_x + id_y * width + id_z  * channel;
    float normY = ((float)(2 * id_y) / (float) (height)) - 1;
    float normX = ((float)(2 * id_x) / (float) (width)) - 1;
    float dist = sqrt((normX * normX) + (normY * normY));

    if ((0.0 <= dist) && (dist <= 1.0))
    {
        float newDist = sqrt(1.0 - dist * dist);
        newDist = (dist + (1.0 - newDist)) / 2.0;
        if (newDist <= 1.0)
        {
            float theta = atan2(normY, normX);
            float newX = newDist * cos(theta);
            float newY = newDist * sin(theta);
            int srcX = (int)(((newX + 1) * width) / 2.0);
            int srcY = (int)(((newY + 1) * height) / 2.0);
            int srcpixIdx = srcY * width + srcX + id_z * channel;

            if (srcpixIdx >= 0 && srcpixIdx < width * height * channel)
            {
                output[dstpixIdx] = input[srcpixIdx];
            }
        }
        else
        {
            output[dstpixIdx] = 0;
        }
    }
}

extern "C" __global__ void fisheye_packed(const unsigned char *input,
                                          unsigned char *output,
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

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    output[dstpixIdx] = 0;
    float normY = ((float)(2 * id_y) / (float) (height)) - 1;
    float normX = ((float)(2 * id_x) / (float) (width)) - 1;
    float dist = sqrt((normX * normX) + (normY * normY));

    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0 - dist * dist);
        newDist = (dist + (1.0 - newDist)) / 2.0;
        if (newDist <= 1.0)
        {
            float theta = atan2(normY, normX);
            float newX = newDist * cos(theta);
            float newY = newDist * sin(theta);
            int srcX = (int)(((newX + 1) * width) / 2.0);
            int srcY = (int)(((newY + 1) * height) / 2.0);
            int srcpixIdx = srcY * width  * channel + srcX  * channel + id_z;
            if (srcpixIdx >= 0 && srcpixIdx < width * height * channel)
            {
                output[dstpixIdx] = input[srcpixIdx];
            }
        }
    }
}

extern "C" __global__ void fisheye_batch(unsigned char *input,
                                         unsigned char *output,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,
                                         unsigned int *xroi_begin,
                                         unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned long long *batch_index,
                                         const unsigned int channel,
                                         unsigned int *inc, // use width * height for pln and 1 for pkd
                                         const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int dstpixIdx = 0;
    dstpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z]) * plnpkdindex;

    float normY = ((float)(2 * id_y) / (float) (height[id_z])) - 1;
    float normX = ((float)(2 * id_x) / (float) (width[id_z])) - 1;
    float dist = sqrt((normX * normX) + (normY * normY));

    if ((0.0 <= dist) && (dist <= 1.0))
    {
        float newDist = sqrt(1.0 - dist * dist);
        newDist = (dist + (1.0 - newDist)) / 2.0;
        if (newDist <= 1.0)
        {
            float theta = atan2(normY, normX);
            float newX = newDist * cos(theta);
            float newY = newDist * sin(theta);
            int srcX = (int)(((newX + 1) * width[id_z]) / 2.0);
            int srcY = (int)(((newY + 1) * height[id_z]) / 2.0);
            int srcpixIdx = batch_index[id_z] + (srcX + srcY * max_width[id_z]) * plnpkdindex;

            if(srcY < yroi_end[id_z] && (srcY >=yroi_begin[id_z]) && srcX < xroi_end[id_z] && (srcX >=xroi_begin[id_z]))
            {
                if(srcpixIdx >= batch_index[id_z] && srcpixIdx <= batch_index[id_z+1])
                {
                    for(int indextmp = 0; indextmp < channel; indextmp++)
                    {
                        output[dstpixIdx] = input[srcpixIdx];
                        dstpixIdx += inc[id_z];
                        srcpixIdx += inc[id_z];
                    }
                }
            }
        }
    }
    else
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dstpixIdx] = (unsigned char) 0;
            dstpixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_fisheye_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(fisheye_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}