#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

extern "C" __global__ void channel_extract_pln(unsigned char *input,
                                               unsigned char *output,
                                               const unsigned int height,
                                               const unsigned int width,
                                               const unsigned int channel,
                                               const unsigned int extractChannelNumber)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = OPpixIdx + extractChannelNumber * width * height;
    output[OPpixIdx] = input[IPpixIdx];
}

extern "C" __global__ void channel_extract_pkd(unsigned char *input,
                                               unsigned char *output,
                                               const unsigned int height,
                                               const unsigned int width,
                                               const unsigned int channel,
                                               const unsigned int extractChannelNumber)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int OPpixIdx = id_y * width + id_x ;
    int IPpixIdx = id_y * width * channel + id_x * channel + extractChannelNumber;
    output[OPpixIdx] = input[IPpixIdx];
}

extern "C" __global__ void channel_extract_batch(unsigned char *input,
                                                 unsigned char *output,
                                                 unsigned int *channelNumber,
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

    int tempchannelNumber = channelNumber[id_z];
    int indextmp = 0;
    unsigned long pixIdx = 0, outPixIdx = 0;

    if((id_y >= 0) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex + tempchannelNumber;
        outPixIdx = (batch_index[id_z] / 3) + (id_x + id_y * max_width[id_z]);
        output[outPixIdx] = input[pixIdx];
    }
}

RppStatus hip_exec_channel_extract_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(channel_extract_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}