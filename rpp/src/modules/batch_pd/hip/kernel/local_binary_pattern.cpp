#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ int power_function(int a, int b)
{
    int product = 1;
    for(int i = 0; i < b; i++)
    {
        product *= product * a;
    }
    return product;
}

extern "C" __global__ void local_binary_pattern_pkd(unsigned char *input,
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

    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    unsigned int pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input[pixIdx - width * channel - channel];
    else
        neighborhood[0] = 0;

    if((id_y - 1) >= 0)
        neighborhood[1] = input[pixIdx - width * channel];
    else
        neighborhood[1] = 0;

    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input[pixIdx - width * channel + channel];
    else
        neighborhood[2] = 0;

    if((id_x + 1) <= width)
        neighborhood[3] = input[pixIdx + channel];
    else
        neighborhood[3] = 0;

    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input[pixIdx + width * channel + channel];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input[pixIdx + width * channel];
    else
        neighborhood[5] = 0;

    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input[pixIdx + width * channel -channel];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input[pixIdx - channel];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0 ; i < 8 ; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += power_function(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);
}

extern "C" __global__ void local_binary_pattern_pln(unsigned char *input,
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

    int pixIdx = id_y * width + id_x + id_z * width * height;

    unsigned char pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input[pixIdx - width - 1];
    else
        neighborhood[0] = 0;

    if((id_y - 1) >= 0)
        neighborhood[1] = input[pixIdx - width];
    else
        neighborhood[1] = 0;

    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input[pixIdx - width + 1];
    else
        neighborhood[2] = 0;

    if((id_x + 1) <= width)
        neighborhood[3] = input[pixIdx + 1];
    else
        neighborhood[3] = 0;

    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input[pixIdx + width + 1];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input[pixIdx + width];
    else
        neighborhood[5] = 0;

    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input[pixIdx + width -1];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input[pixIdx - 1];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0; i < 8; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += power_function(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);
}

extern "C" __global__ void local_binary_pattern_batch(unsigned char *input,
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

    if(id_x < width[id_z] - 1 && id_y < height[id_z] - 1 && id_x > 0 && id_y > 0)
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z] ) * plnpkdindex;
        unsigned char neighborhoodR[3][9];
        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            for(int i = 0; i < channel; i++)
            {
                if((id_x - 1) >= 0 && (id_y - 1) >= 0)
                    neighborhoodR[i][0] = input [pixIdx - (max_width[id_z] + 1) * plnpkdindex];
                else
                    neighborhoodR[i][0] = 0;

                if((id_y - 1) >= 0)
                    neighborhoodR[i][1] = input [pixIdx - (max_width[id_z]) * plnpkdindex];
                else
                    neighborhoodR[i][1] = 0;

                if((id_x + 1) < width[id_z] && (id_y - 1) >= 0)
                    neighborhoodR[i][2] = input [pixIdx - (max_width[id_z] - 1) * plnpkdindex];
                else
                    neighborhoodR[i][2] = 0;

                if((id_x + 1) < width[id_z])
                    neighborhoodR[i][3] = input [pixIdx + (1) * plnpkdindex];
                else
                    neighborhoodR[i][3] = 0;

                if((id_x + 1) < width[id_z] && (id_y + 1) < height[id_z])
                    neighborhoodR[i][4] = input [pixIdx + (max_width[id_z] + 1) * plnpkdindex];
                else
                    neighborhoodR[i][4] = 0;

                if((id_y + 1) < height[id_z])
                    neighborhoodR[i][5] = input [pixIdx + (max_width[id_z]) * plnpkdindex];
                else
                    neighborhoodR[i][5] = 0;

                if((id_x - 1) >= 0 && (id_y + 1) < height[id_z])
                    neighborhoodR[i][6] = input [pixIdx + (max_width[id_z] - 1) * plnpkdindex];
                else
                    neighborhoodR[i][6] = 0;

                if((id_x - 1) >= 0)
                    neighborhoodR[i][7] = input [pixIdx - (1) * plnpkdindex];
                else
                    neighborhoodR[i][7] = 0;

                neighborhoodR[i][8] = input[pixIdx];

                pixIdx += inc[id_z];
            }

            pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

            for(int j = 0; j < channel; j++)
            {
                int pixel = 0;
                for(int i = 0; i < 8; i++)
                {
                    if(neighborhoodR[j][i] - input[pixIdx] >= 0)
                    {
                        pixel += powf((float)2, (float)i);
                    }
                }
                output[pixIdx] = saturate_8u(pixel);
                pixIdx += inc[id_z];
            }
        }
        else if((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}

RppStatus hip_exec_local_binary_pattern_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(local_binary_pattern_batch,
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