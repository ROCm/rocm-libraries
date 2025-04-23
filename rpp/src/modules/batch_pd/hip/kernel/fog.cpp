#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void fog_planar(unsigned char *input,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel,
                                      const float fogValue)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixId= width * id_y + id_x;
    int c = width * height;
    float check = input[pixId];
    if(channel > 1)
    {
        check += input[pixId+c] + input[pixId + c * 2];
        check = check / 3;
    }
    if(check >= (240) && fogValue!=0)
    {}
    else if(check>=(170))
    {
        float pixel = ((float) input[pixId]) * (1.5 + fogValue) - (fogValue*4) + (7 * fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel > 1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + fogValue) + (7 * fogValue);
            input[pixId+c] = saturate_8u(pixel);
            pixel = ((float) input[pixId + c * 2]) * (1.5 + fogValue) + (fogValue * 4) + (7 * fogValue);
            input[pixId+c*2] = saturate_8u(pixel);
        }
    }
    else if(check <= (85))
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue*fogValue)) - (fogValue * 4) + (130 * fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel > 1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + (fogValue * fogValue)) + (130 * fogValue);
            input[pixId + c] = saturate_8u(pixel);
            pixel = ((float) input[pixId + c * 2]) * (1.5 + (fogValue * fogValue)) + (fogValue * 4) + 130 * fogValue;
            input[pixId + c * 2] = saturate_8u(pixel);
        }
    }
    else
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue * ( fogValue * 1.414))) - (fogValue * 4) + 20 + (100 * fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel>1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + (fogValue * (fogValue * 1.414))) + 20 + (100 * fogValue);
            input[pixId + c] = saturate_8u(pixel);
            pixel = ((float) input[pixId + c * 2]) * (1.5 + (fogValue * (fogValue * 1.414))) + (fogValue * 4) + (100 * fogValue);
            input[pixId + c * 2] = saturate_8u(pixel);
        }
    }
}

extern "C" __global__ void fog_pkd(unsigned char *input,
                                   const unsigned int height,
                                   const unsigned int width,
                                   const unsigned int channel,
                                   const float fogValue)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int i = width * id_y * channel + id_x * channel;
    float check = input[i] + input[i + 1] + input[i + 2];
    if(check >= (240 * 3) && fogValue != 0)
    {}
    else if(check >= (170 * 3) && fogValue != 0)
    {
        float pixel = ((float) input[i]) * (1.5 + fogValue) - (fogValue * 4) + (7 * fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + fogValue) + (7 * fogValue);
        input[i + 1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + fogValue) + (fogValue * 4) + (7 * fogValue);
        input[i + 2] = saturate_8u(pixel);
    }
    else if(check <= (85 * 3) && fogValue != 0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue*fogValue)) - (fogValue*4) + (130*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue*fogValue)) + (130*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue*fogValue)) + (fogValue*4) + 130*fogValue;
        input[i+2] = saturate_8u(pixel);
    }
    else if(fogValue != 0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue * (fogValue * 1.414))) - (fogValue * 4) + 20 + (100 * fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue * ( fogValue * 1.414))) + 20 + (100 * fogValue);
        input[i + 1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue * ( fogValue * 1.414))) + (fogValue * 4) + (100 * fogValue);
        input[i + 2] = saturate_8u(pixel);
    }
}

extern "C" __global__ void fog_batch(unsigned char *input,
                                     unsigned char *output,
                                     float *fogValue,
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

    float tempFogValue = fogValue[id_z];
    int indextmp = 0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {
        float check = input[pixIdx];
        if(channel == 3)
        {
            check = (check + input[pixIdx + inc[id_z]] + input[pixIdx + inc[id_z] * 2]) / 3;
        }
        if(check >= (240) && tempFogValue!=0)
        {
            output[pixIdx] = input[pixIdx];
            if(channel > 1)
            {
                output[pixIdx + inc[id_z]] = input[pixIdx + inc[id_z]];
                output[pixIdx + inc[id_z] * 2] = input[pixIdx + inc[id_z] * 2];
            }
        }
        else if(check >= (170) && tempFogValue != 0)
        {
            float pixel = ((float)input[pixIdx]) * (1.5 + tempFogValue) - (tempFogValue * 4) + (7 * tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel > 1)
            {
                pixel = ((float)input[pixIdx + inc[id_z]]) * (1.5 + tempFogValue) + (7 * tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float)input[pixIdx + inc[id_z] * 2]) * (1.5 + tempFogValue) + (tempFogValue * 4) + (7 * tempFogValue);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }
        else if(check <= (85) && tempFogValue != 0)
        {
            float pixel = ((float)input[pixIdx]) * (1.5 + (tempFogValue * tempFogValue)) - (tempFogValue * 4) + (130 * tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel > 1)
            {
                pixel = ((float)input[pixIdx + inc[id_z]]) * (1.5 + (tempFogValue * tempFogValue)) + (130 * tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float)input[pixIdx + inc[id_z] * 2]) * (1.5 + (tempFogValue * tempFogValue)) + (tempFogValue * 4) + 130 * tempFogValue;
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }
        else if(tempFogValue != 0)
        {
            float pixel = ((float)input[pixIdx]) * (1.5 + (tempFogValue * (tempFogValue * 1.414))) - (tempFogValue * 4) + 20 + (100 * tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel > 1)
            {
                pixel = ((float)input[pixIdx + inc[id_z]]) * (1.5 + (tempFogValue * (tempFogValue * 1.414))) + 20 + (100 * tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float)input[pixIdx + inc[id_z] * 2]) * (1.5 + (tempFogValue * (tempFogValue * 1.414))) + (tempFogValue * 4) + (100 * tempFogValue);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }
    }
}

RppStatus hip_exec_fog_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(fog_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}