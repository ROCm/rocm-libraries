#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned int xorshift(int pixid)
{
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
    return res;
}

extern "C" __global__ void jitter_pkd(unsigned char *input,
                                      unsigned char *output,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel,
                                      const unsigned int kernelSize)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;

    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
        for(int i = 0; i < channel; i++)
        {
            output[pixIdx + i] = input[index + i];
        }
    }
    else
    {
        for(int i = 0; i < channel; i++)
        {
            output[pixIdx + i] = input[pixIdx + i];
        }
    }
}

extern "C" __global__ void jitter_pln(unsigned char *input,
                                      unsigned char *output,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel,
                                      const unsigned int kernelSize)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x;
    int channelPixel = height * width;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;

    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
        for(int i = 0; i < channel; i++)
        {
            output[pixIdx + (height * width * i)] = input[index + (height * width * i)];
        }
    }
    else
    {
        for(int i = 0; i < channel; i++)
        {
            output[pixIdx + (height * width * i)] = input[pixIdx + (height * width * i)];
        }
    }
}

extern "C" __global__ void jitter_batch(unsigned char *input,
                                        unsigned char *output,
                                        unsigned int *kernelSize,
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

    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int bound = (kernelSizeTemp - 1) / 2;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            int nhx = xorshift(pixIdx) % (kernelSizeTemp);
            int nhy = xorshift(pixIdx) % (kernelSizeTemp);
            if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height[id_z] - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width[id_z] - 1)
            {
                int index = batch_index[id_z] + ((((id_y - bound) * max_width[id_z]) + (id_x - bound)) * plnpkdindex) + (((nhy * max_width[id_z]) + (nhx)) * plnpkdindex);
                for(int i = 0; i < channel; i++)
                {
                    output[pixIdx] = input[index];
                    pixIdx += inc[id_z];
                    index += inc[id_z];
                }
            }
        }
        else if((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}

RppStatus hip_exec_jitter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(jitter_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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