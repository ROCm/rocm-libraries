#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void pixelate_pkd(unsigned char *input,
                                        unsigned char *output,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int channel)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int c = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (x * 7 >= width || y * 7 >= height || c >= channel)
    {
        return;
    }

    y = y * 7;
    x = x * 7;
    int sum = 0;
    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {
                sum += input[((y + i) * width * channel + (x + j) * channel + c)];
            }
        }
    }
    sum /= 49;

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(y + i < height && x + j < width)
            {
                output[((y + i) * width * channel + (x + j) * channel + c)] = saturate_8u(sum);
            }
        }
    }
}

extern "C" __global__ void pixelate_pln(unsigned char *input,
                                        unsigned char *output,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int channel)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int c = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (x * 7 >= width || y * 7 >= height || c >= channel)
    {
        return;
    }

    y = y * 7;
    x = x * 7;
    int sum = 0;

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {
                sum += input[(y + i) * width + (x + j) + c * height * width];
            }
        }
    }
    sum /= 49;

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(y + i < height && x + j < width)
            {
                output[(y + i) * width + (x + j) + c * height * width] = saturate_8u(sum);
            }
        }
    }
}

extern "C" __global__ void pixelate_batch(unsigned char *input,
                                    unsigned char *output,
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

    long pixIdx = 0;
    int x,y;

    if(id_x * 7 < width[id_z] && id_y * 7 < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        if((id_y * 7 >= yroi_begin[id_z]) && (id_y * 7 <= yroi_end[id_z]) && (id_x * 7 >= xroi_begin[id_z]) && (id_x * 7 <= xroi_end[id_z]))
        {
            y = id_y * 7;
            x = id_x * 7;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for(int i = 0; i < 7; i++)
            {
                for(int j = 0; j < 7; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z] && y + i >= 0 && x + j >= 0)
                    {
                        sum1 += input[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex];
                        if(channel == 3)
                        {
                            sum2 += input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]];
                            sum3 += input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2];
                        }
                    }
                }
            }
            sum1 /= 49;
            sum3 /= 49;
            sum2 /= 49;
            for(int i = 0; i < 7; i++)
            {
                for(int j = 0; j < 7; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z])
                    {
                        output[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex] = saturate_8u(sum1);
                        if(channel == 3)
                        {
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]] = saturate_8u(sum2);
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2] = saturate_8u(sum3);
                        }
                    }
                }
            }
        }
        else if((id_x * 7 < width[id_z]) && (id_y * 7 < height[id_z]))
        {
            y = id_y * 7;
            x = id_x * 7;
            for(int i = 0; i < 7; i++)
            {
                for(int j = 0; j < 7; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z])
                    {
                        output[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex] = input[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex];
                        if(channel == 3)
                        {
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]] = input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]];
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2] =  input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2];
                        }
                    }
                }
            }
        }
    }
}

RppStatus hip_exec_pixelate_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(pixelate_batch,
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
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}