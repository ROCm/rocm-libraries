#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define MAX2(a,b) ((a > b) ? a : b)
#define MIN2(a,b) ((a < b) ? a : b)

extern "C" __global__ void scale_pln(unsigned char *srcPtr,
                                     unsigned char *dstPtr,
                                     const unsigned int source_height,
                                     const unsigned int source_width,
                                     const unsigned int dest_height,
                                     const unsigned int dest_width,
                                     const unsigned int channel,
                                     const unsigned int exp_dest_height,
                                     const unsigned int exp_dest_width)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height)
    {
        return;
    }

    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(source_width - 1)) / exp_dest_width;
    float y_ratio = ((float)(source_height - 1)) / exp_dest_height;
    float x_diff, y_diff, ya, yb;

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void scale_pkd(unsigned char *srcPtr,
                                     unsigned char *dstPtr,
                                     const unsigned int source_height,
                                     const unsigned int source_width,
                                     const unsigned int dest_height,
                                     const unsigned int dest_width,
                                     const unsigned int channel,
                                     const unsigned int exp_dest_height,
                                     const unsigned int exp_dest_width)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height)
    {
        return;
    }

    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(source_width - 1)) / exp_dest_width;
    float y_ratio = ((float)(source_height - 1)) / exp_dest_height;
    float x_diff, y_diff, ya, yb;

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x + 1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y + 1) * source_width * channel + id_z];
    D = srcPtr[(x + 1) * channel + (y + 1) * source_width * channel + id_z];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void scale_batch(unsigned char *srcPtr,
                                       unsigned char *dstPtr,
                                       float *percentage,
                                       unsigned int *source_height,
                                       unsigned int *source_width,
                                       unsigned int *dest_height,
                                       unsigned int *dest_width,
                                       unsigned int *max_source_width,
                                       unsigned int *max_dest_width,
                                       unsigned int *xroi_begin,
                                       unsigned int *xroi_end,
                                       unsigned int *yroi_begin,
                                       unsigned int *yroi_end,
                                       unsigned long long *source_batch_index,
                                       unsigned long long *dest_batch_index,
                                       const unsigned int channel,
                                       unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                       unsigned int *dest_inc,
                                       const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) * 100 / (percentage[id_z] * dest_width[id_z]);
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) * 100 / (percentage[id_z] * dest_height[id_z]);
    float x_diff, y_diff, ya, yb;

    int indextmp = 0;
    unsigned long dst_pixIdx = 0;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];

            int pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                               B * (x_diff) * (1 - y_diff) +
                               C * (y_diff) * (1 - x_diff) +
                               D * (x_diff * y_diff));

            dstPtr[dst_pixIdx] = saturate_8u(pixVal);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

RppStatus hip_exec_scale_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(scale_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       plnpkdind);

    return RPP_SUCCESS;
}