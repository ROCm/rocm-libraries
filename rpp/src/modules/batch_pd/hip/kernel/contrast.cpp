#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void contrast_stretch(unsigned char *input,
                                            unsigned char *output,
                                            const unsigned int min,
                                            const unsigned int max,
                                            const unsigned int new_min,
                                            const unsigned int new_max,
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

    int pixIdx = id_x + id_y * width + id_z * width * height;
    int res;
    res = (input[pixIdx] - min) * (new_max - new_min) / ((max - min) * 1.0) + new_min;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void contrast_batch(unsigned char *input,
                                          unsigned char *output,
                                          unsigned int min,
                                          unsigned int max,
                                          unsigned int *new_min,
                                          unsigned int *new_max,
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

    float tmpmin = min;
    float tmpmax = max;
    float tmpnew_max = new_max[id_z];
    float tmpnew_min = new_min[id_z];
    int indextmp = 0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            unsigned char valuergb = input[pixIdx];
            output[pixIdx] = saturate_8u((input[pixIdx] - tmpmin) * (tmpnew_max - tmpnew_min) / ((tmpmax - tmpmin) * 1.0) + tmpnew_min);
            pixIdx += inc[id_z];
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

RppStatus hip_exec_contrast_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, Rpp32u min, Rpp32u max)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(contrast_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       min,
                       max,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
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