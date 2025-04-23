#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

extern "C" __global__ void flip_horizontal_planar(const unsigned char *input,
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

    int oPixIdx = id_x + id_y * width + id_z * width * height;
    int nPixIdx = id_x + (height - 1 - id_y) * width + id_z * width * height;

    output[nPixIdx] = input[oPixIdx];
}

extern "C" __global__ void flip_vertical_planar(const unsigned char *input,
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

    int oPixIdx = id_x + id_y * width + id_z * width * height;
    int nPixIdx = (width-1 - id_x) + id_y * width + id_z * width * height;

    output[nPixIdx] = input[oPixIdx];
}

extern "C" __global__ void flip_bothaxis_planar(const unsigned char *input,
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

    int oPixIdx = id_x + id_y * width + id_z * width * height;
    int nPixIdx = (width-1 - id_x) + (height-1 - id_y) * width + id_z * width * height;

    output[nPixIdx] = input[oPixIdx];
}

extern "C" __global__ void flip_horizontal_packed(const unsigned char *input,
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

    int oPixIdx = id_x * channel + id_y * width * channel + id_z;
    int nPixIdx = id_x * channel + (height - 1 - id_y) * width * channel + id_z ;

    output[nPixIdx] = input[oPixIdx];
}

extern "C" __global__ void flip_vertical_packed(const unsigned char *input,
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

    int oPixIdx = id_x * channel + id_y * width * channel + id_z;
    int nPixIdx = (width-1 - id_x) * channel + id_y * width * channel + id_z;

    output[nPixIdx] = input[oPixIdx];
}

extern "C" __global__ void flip_bothaxis_packed(const unsigned char *input,
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

    int oPixIdx = id_x * channel + id_y * width * channel + id_z;
    int nPixIdx = (width - 1 - id_x) * channel + (height - 1 - id_y) * width * channel + id_z;

    output[nPixIdx] = input[oPixIdx];
}
extern "C" __global__ void flip_batch(unsigned char *srcPtr,
                                      unsigned char *dstPtr,
                                      unsigned int *flipAxis,
                                      unsigned int *height,
                                      unsigned int *width,
                                      unsigned int *max_width,
                                      unsigned long long *batch_index,
                                      unsigned int *xroi_begin,
                                      unsigned int *xroi_end,
                                      unsigned int *yroi_begin,
                                      unsigned int *yroi_end,
                                      const unsigned int channel,
                                      unsigned int *inc, // use width * height for pln and 1 for pkd
                                      const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    unsigned long dst_pixIdx;

    if(id_y < yroi_end[id_z] && (id_y >=yroi_begin[id_z]) && id_x < xroi_end[id_z] && (id_x >=xroi_begin[id_z]))
    {
        unsigned long src_pixIdx;

        if(flipAxis[id_z] == 0)
        {
            src_pixIdx = batch_index[id_z] + (id_x + (height[id_z] -1 -id_y) * max_width[id_z]) * plnpkdindex;
        }

        if(flipAxis[id_z] == 1)
        {
            src_pixIdx = batch_index[id_z] + ((width[id_z] -1 -id_x) + (id_y) * max_width[id_z]) * plnpkdindex;
        }

        if(flipAxis[id_z] == 2)
        {
            src_pixIdx = batch_index[id_z] + ((width[id_z] -1 -id_x) + (height[id_z] -1 -id_y) * max_width[id_z]) * plnpkdindex;
        }

        dst_pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
            src_pixIdx += inc[id_z];
            dst_pixIdx += inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            dst_pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
            dstPtr[dst_pixIdx] = srcPtr[dst_pixIdx];
            dst_pixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_flip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(flip_batch,
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
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}