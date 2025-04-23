#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

extern "C" __global__ void channel_combine_pln(unsigned char *input1,
                                               unsigned char *input2,
                                               unsigned char *input3,
                                               unsigned char *output,
                                               const unsigned int height,
                                               const unsigned int width,
                                               const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx1 = IPpixIdx;
    int OPpixIdx2 = IPpixIdx + width * height;
    int OPpixIdx3 = IPpixIdx + 2 * width * height;

    output[OPpixIdx1] = input1[IPpixIdx];
    output[OPpixIdx2] = input2[IPpixIdx];
    output[OPpixIdx3] = input3[IPpixIdx];
}

extern "C" __global__ void channel_combine_pkd(unsigned char *input1,
                                               unsigned char *input2,
                                               unsigned char *input3,
                                               unsigned char *output,
                                               const unsigned int height,
                                               const unsigned int width,
                                               const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = IPpixIdx * channel;
    output[OPpixIdx] = input1[IPpixIdx];
    output[OPpixIdx + 1] = input2[IPpixIdx];
    output[OPpixIdx + 2] = input3[IPpixIdx];
}

extern "C" __global__ void channel_combine_batch(unsigned char *input1,
                                                 unsigned char *input2,
                                                 unsigned char *input3,
                                                 unsigned char *output,
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

    unsigned long pixIdx = 0, InPixIdx = 0;

    if((id_y >= 0) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        InPixIdx = (batch_index[id_z] / 3) + (id_x + id_y * max_width[id_z]);
        output[pixIdx] = input1[InPixIdx];
        output[pixIdx + inc[id_z]] = input2[InPixIdx];
        output[pixIdx + inc[id_z] * 2] = input3[InPixIdx];
    }
}

RppStatus hip_exec_channel_combine_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *srcPtr3, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(channel_combine_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       srcPtr3,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}
